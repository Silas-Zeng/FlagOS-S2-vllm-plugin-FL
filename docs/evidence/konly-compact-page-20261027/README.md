# K8/BF16 compact page：CPU 提案

基于团队 main `dac1b8fb48e7983e48283032efeee9b592c5fbe5`。当前 K-only 实验把 K code、FP32 scale 和原 BF16 V 放在原 16KiB backing 中，尚未回收物理容量。本提案单独为后续容量优化建立字节和分配计费契约，不修改生产源码、不注册新 backend/spec、不启动 GPU。

| 字段 | 原实验页 | 本提案页 |
| --- | ---: | ---: |
| K INT8 code | `[0,4096)` | `[0,4096)` |
| FP32 K scale | `[4096,4224)` | `[4096,4224)` |
| BF16 V | `[8192,16384)` | `[4352,12544)` |
| 物理页字节 | 16384 | 12544 |
| scale 全页 F32 stride | 4096 | 3136 |
| V 全页 BF16 stride | 8192 | 6272 |
| 已偏移 V 的 BF16 offset | 4096 | 2176 |

每页有效 payload 12,416B，选择 256B 对齐并保留 128B padding。该对齐是显式提案，设备效率尚未验证。理论页字节减少 **23.4375%**；相同每层字节预算下，忽略离散块和其他开销时可容纳的页数比例为 **1.30612245**。当前没有真实 allocator 容量、HBM 流量、服务吞吐或 TTFT 测量。

## 已验证范围

`target-kv-cache-interface.py` 来自用户个人实例当前安装的 `vllm/v1/kv_cache_interface.py`，SHA256 `7e3c220998267190a622e216b2b125306807cf8be3ec1e9a8e84fe6e7a293003`。归档保留原许可证和源码；无需导入 vLLM 或升级比赛框架。

CPU 检查只抽取该精确源码的 `KVCacheSpec`、`AttentionSpec`、`FullAttentionSpec` 三个类，在受限的 stdlib mock scope 中执行其属性、merge 和内存计费方法。提案保留语义 head size 128；使用 CPU 子类表示自己的页字节数。检查发现原 merge 按基类字段比较，私有格式必须额外拒绝和 stock spec 混合。`page_size_padded` 的原下界检查保留。

三页地址枚举的 37,248 个 payload 字节不重叠；32 页 CPU repack 保留全部 code/scale/V 字节，V 覆盖所有 65,536 个 BF16 位模式两次。10 项拒绝覆盖混合 spec、尺寸、下界、地址类型和非整页输入。Python 3.12.15 与 3.14 输出逐字一致，结果见 `cpu.json`。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_compact_page_cpu.py
```

## 接入前的必要工作

首版 default-off encoded 服务继续沿用 16KiB 格式。这里的提案需要独立格式 ID、明确 spec 注册和 uniform/group 类型匹配、backend reshape 与实际 allocator 字节核验，以及新的 typed-view broker。现有 reader/writer 的固定 `4096` F32 页 stride 和 `4096` BF16 V offset 不能直接复用；其数值、Graph、旁路 stream 及模型精度需要在新格式重验。

zeroer、原页拷贝/交换长度、prefix sharing、重分配后 view freshness、Graph 地址生命周期也必须覆盖。当前 `cpu.json` 对 registry/backend/allocator/zeroer/lifecycle/device/model/performance/service 全部明确为 false。容量回收与吞吐验收须使用比赛固定服务参数和原 benchmark。

该方向依赖 K-only 的读写与生命周期工作（PR79/80/82/83/85/86/87）；CPU repack 用于契约验证，尚无 GPU repacker 或服务运行入口。
