# 保留全部 KV 的二维 SAVE 提交工具

本工具在不排序、不丢页、不缩短保存前缀的条件下，保留原长连续 run，并把单页区域内的正等差页段合成二维 DeviceToHost 拷贝。Python 在首提交前校验整批真实 Tensor/storage/descriptor；SDK 编译的 C++ 桥用一次 FFI 调用提交线性和二维拷贝。它同时包含原生批量提交与二维合并，不能把两者相对 Torch 逐次调用的差异全部归给二维 API。

本 PR 仅新增：

- `vllm_fl/patches/biv150_kv_pitched_copy.py`：已测 818e 原字节。
- `vllm_fl/patches/biv150_kv_pitched_bridge.cpp`：已测 0a4d 原字节。
- 本中文说明。

不包含 `.so` 二进制、完整 worker/事务/恢复协议、默认启用 hook、Event 实验或 serving 参数变更。`setup.py` 与包数据配置保持原样；不承诺 wheel 自动包含 C++ 源或编译桥，按下面的 Git 源码流程使用。

## 与连续页工具的关系

此模块自行定义 `Segment`/`SavePlan`，只导入 `ctypes`、`dataclasses`、`hashlib`、`pathlib`、`sys`；C++ 只依赖实际 SDK 头及标准库。**没有对 `biv150_kv_page_copy.py` 的代码依赖**，可以直接基于同一个 `main` 独立合入。字段 `original_c95f_runs/calls` 只是本模块计算的连续段对照统计，不会导入或调用连续页工具。实验使用 c95f 作独立 oracle，不表示本工具依赖该 PR。

## 固定支持范围

当前只支持 SAVE：`cuda:0` raw uint8 → pinned CPU，单层门或 42 个独立 normal 层；不提供 H2D RESTORE、packed 跨层页或任意 Tensor 布局。

每个 source shape 为 `(source_pages, 16384)`，destination 为 `(logical_pages, 16384)`，stride 必须 `(16384, 1)`，类型为实际 `torch.Tensor`。允许合法 storage offset，但必须通过真实 data_ptr/storage 边界验证；不接受跨层 source/host storage alias。目的页按原逻辑顺序紧密排列为 `0..logical_pages-1`。整批 payload 不超过 1 GiB，descriptor 不超过 65536；这只是此工具的防护范围，不替代调用方主机预算。

已测环境是天数 BI-V150 16 SM/32 GiB 实例，Corex 4.5.0、Torch 2.10、g++ 13.3、64 位 pointer/size_t 和 72 字节 descriptor ABI。SDK 将 CUDA runtime API 映射到 Corex IX；实际库 SHA256 为 `d39d579627d19ddb903855ebc4859661579633a7ecc4d1ad51866d79296ba527`。这不是对 NVIDIA/其他 Corex 版本或沐曦－曦云 C500 的可用性声明。

## 从 Git 源码编译

在匹配 SDK 的 Linux checkout 中手工编译，勿把旧环境 `.so` 当通用二进制：

```sh
mkdir -p build/kv-pitched
g++ -O2 -std=c++17 -fPIC -shared \
  -I/usr/local/corex/include \
  vllm_fl/patches/biv150_kv_pitched_bridge.cpp \
  -L/usr/local/corex/lib64 -Wl,-rpath,/usr/local/corex/lib64 \
  -lcudart -o build/kv-pitched/biv150_kv_pitched_bridge.so
ldd build/kv-pitched/biv150_kv_pitched_bridge.so
sha256sum build/kv-pitched/biv150_kv_pitched_bridge.so
```

保存编译命令、源码/SDK 头/实际运行库身份及产物 SHA。`Bridge` 校验调用方提供的 SO SHA 和 ABI；它本身不替调用方核实厂商、SDK 来源或 `/proc/self/maps`。调用方还须确认实际链接/进程加载的是获准运行库，不能把现场临时自算哈希当代码来源证明。

## API 与完成责任

1. `make_plan(source_ids, source_pages)` 保原顺序及全部逻辑页；只在原单页区域左到右合并正恒差，降序/非正步长不会强行改序。该策略不宣称全局最优分段。
2. 显式构造 `Bridge(path, expected_sha256=approved_so_sha)` 才载库；模块导入本身不载 SO。正常包导入仍遵循现有 `vllm_fl.__init__` 行为。
3. 用实际 `bridge.max_pitch(device=0)` 作 `prepare_save(..., max_pitch=...)` 输入。超过设备 pitch 的段在任何提交前拆成单页线性描述符，显式记录 fallback；不是 SDK 报错之后重试。`max_pitch=16384` 是原实验的受控 C-linear 对照，不应拿它冒充正常设备能力。
4. `prepare_save` 不读 Tensor 值、不分配 GPU buffer；要求调用方已经加载 Torch。`PreparedSave.keepalive` 强持 Tensor，但调用方还须持有桥、descriptor、流、完成事件与源页所有权。
5. `bridge.submit(prepared, stream_handle=...)` 固定 D2H；C++ 在首提交前复核全批边界/overflow/当前 device/maxpitch。成功返回 `completion_ack=False`，只证明提交，不证明数据已经可读或页可以释放。

以下只展示已有资源的顺序，不创建模型或启用服务：

```python
from pathlib import Path
import torch
from vllm_fl.patches.biv150_kv_pitched_copy import Bridge, make_plan, prepare_save

bridge = Bridge(Path("build/kv-pitched/biv150_kv_pitched_bridge.so"),
                expected_sha256=approved_so_sha)
plan = make_plan(source_ids, raw_layers[0].shape[0])
prepared = prepare_save(raw_layers, pinned_host_layers, plan,
                        max_pitch=bridge.max_pitch(device=0))
with torch.cuda.stream(copy_stream):
    copy_stream.wait_event(all_producers_done)
    submitted = bridge.submit(prepared, stream_handle=int(copy_stream.cuda_stream))
    done = torch.cuda.Event()
    done.record(copy_stream)
# 全部对象及源页所有权继续存活；pinned host不能先读/改/复用。
done.synchronize()
# 全批真实完成后，调用方才能产生SAVE ACK或free源页。
```

不同 producer 不能靠一枚未涵盖全部写入的事件冒充完成。取消、epoch、refcount/hash、源页 free、host slot 与完整恢复发布属于独立协议，均未接入本 PR。

`CopySubmissionError` 带错误码及已成功提交的调用数；中途失败可能已有异步工作，甚至成功计数为零也不能擅自假定无 pending 工作。调用方必须先 settle；未 settle 时保留全部强引用，禁止静默重试旧复制路径或发送成功 ACK。`copy_calls` 是成功原生 API 提交数，不是 DMA 描述符数或时间占比。

## 已有证据和收益边界

完整原件在团队 [实验与分析仓库](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput)，本 PR 只保存代码与说明：

| run | 已证范围 |
| --- | --- |
| `20261022T032653Z-kv-pitched-copy-gate` | 两组真实 SAVE 页数组，单层完整字节/源/双 guard/非默认流/RNG通过；实际 maxpitch 2147483647，大于所需33570816。wall波动大，不报稳定收益 |
| `20261022T035754Z-kv-pitched-threeway-gate` | 同两页序，C-linear保留37/26次调用，C2D为11/6，完整字节通过。首case64call镜像窗口无新增节流；同桥线性→2D的热 wall 均值0.655054→0.631340ms，约3.62%，仅单层热提交/完成信号 |
| `20261022T042225Z-kv-pitched-save-context16k-long` | 独立实验 worker 中实际42层SAVE：R16688/1043页/717717504B，原连续966→新210次API；源free前42层c95f独立对照及原FULL恢复后42层回读通过，原consumer处理16个scheduled tokens |

035754 准备成本在热窗口外；不能把单层差乘42或当整次 SAVE/E2E 收益。042225 有双回读，是正确性诊断，不能计算吞吐提升；其 worker/协议/SO/default hook 不在本 PR。未完成全模型准确率、全局随机输出逐token等价或多请求取消/反复抢占的完整验收，因此不默认启用。

固定源码 SHA256：Python `818e788b282ba51017239fd9c505f9709c219a16024d4d56d1df750d3a61b4cd`；C++ `0a4d44d8bd71488f57d5adaa7bdc80f8b16193b11884c167b8767b9e7d9c6d44`。保留本仓库许可证，不复制厂商头文件或二进制。此 PR 可单独回退三个新增文件；与连续页工具 PR 无合入顺序约束。
