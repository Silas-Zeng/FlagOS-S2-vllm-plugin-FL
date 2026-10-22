# KV 连续页合并复制工具

本工具保留原逻辑页次序，把源和目的同时连续的相邻页合成一次 `narrow(...).copy_(..., non_blocking=True)`。它只处理原始字节，不改变 KV 数值，不分配 GPU packing buffer，也不选择请求、驱逐页或保存长度。

新增模块为 `vllm_fl/patches/biv150_kv_page_copy.py`，源码与已测 c95f 版本逐字相同。没有修改插件默认入口、README、模型参数、serving 参数或 setup；合入本 PR 不会自动启用 KV offload。后续服务协议须单独实现、审阅和显式启用。

## API 与范围

| API | 行为 |
| --- | --- |
| `make_plan(source_ids, destination_ids, source_pages, destination_pages, page_bytes)` | 校验非空、等长、唯一、界内页映射；保原序，仅合并两侧同时 `+1` 的邻接，不排序 |
| `submit_page_copy(source, destination, plan)` | 所有 shape/dtype/device/pinned 条件通过后，按 run 在当前 Torch stream 提交复制；返回提交字节与调用数 |

支持的 raw Tensor 是二维、contiguous、`torch.uint8`，shape 分别为 `(source_pages, page_bytes)` 和 `(destination_pages, page_bytes)`；一侧是 pinned CPU，另一侧为 `cuda:0`。`page_bytes` 只接受 16384 或 `42*16384`：前者对应本实验 normal 单层 raw 页，后者对应已测试的 packed 跨层 raw 页。它不负责把原 KV tensor 重排成这种布局，不能仅凭 shape 相同忽略真实 storage/stride。

`completion_ack=False` 始终表示**只提交、未确认完成**。`copy_calls` 是本工具调用 `copy_` 的次数，不等于 DMA 描述符数量，也不等于执行时间。

调用方必须负责：

- 源 KV 已由所有 producer 写完；若不在同一流，先建立真实跨流依赖。
- 持有 source、destination、plan、相关 stream/事件和页所有权，直到真实完成；pinned host 在此之前不能读取、改写或复用，源页也不能先 free/evict。
- 在复制所用的同一 stream 上记录完成 event，观察完成后才发送 SAVE ACK 或发布恢复页。全部层都完成才是整个保存/恢复完成。
- 预算、取消、请求 epoch、页 refcount/hash、私有目的分配和事务回滚属于调用方；本模块不提供 scheduler/allocator/RPC 协议。
- 提交阶段异常可能已留下部分异步工作；先 settle 并保引用，不能直接静默换旧路重试或发送成功 ACK。

以下仅展示已有资源的调用顺序，不是独立服务接入：

```python
import torch
from vllm_fl.patches.biv150_kv_page_copy import make_plan, submit_page_copy

plan = make_plan(src_ids, dst_ids, source.shape[0], destination.shape[0], 16384)
with torch.cuda.stream(copy_stream):
    copy_stream.wait_event(producer_done)
    submitted = submit_page_copy(source, destination, plan)
    done = torch.cuda.Event()
    done.record(copy_stream)
# source/destination/plan/copy_stream/done 和页所有权须继续存活。
done.synchronize()
# 到此才能由调用方确认完成；submitted 本身不是 ACK。
```

工具文件本身只导入 `dataclasses`；正常包导入仍遵循本仓库原有 `vllm_fl.__init__` 行为。本 PR 没有新增启动 hook。

## 已有验证与限制

完整原件保存在团队 [实验与分析仓库](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput)，按 run ID 和 SHA 定位；本代码 PR 不复制大日志。

- `20261021T191545Z-kv-page-copy-gate`：normal42/packed42，各连续与 8-run 构造布局，共四 case；每方向 704643072 字节。真实 factory 42 层 view、完整字节、guard、非目标区和 RNG 检查通过。连续布局调用数分别为 42/1，8-run 为 336/8；packed 是 helper 支持的构造，不能当作真实服务布局已采用。
- 独立结果 `analysis/2026-10-22-kv-page-copy-result.json` 明确保留单样本 wall/event 不一致：不能由 normal 与 packed 的差异推导稳定带宽或服务收益。
- 后续真实 normal42 SAVE/恢复事务曾调用相同 c95f 工具；这些 worker/事务/服务 hook 不在本 PR 中。工具的字节门不等于完整准确率或默认部署验收。

已测范围仅天数 BI-V150、Corex 环境；沐曦－曦云 C500 及其他平台没有由这些证据覆盖。未修改量化、采样、speculation 或官方 benchmark 参数。

代码 SHA256：`c95f59a55f5409e651085ff52d788193ea1d5901dcbc99d38fa504842b12f44c`。保留本仓库许可证。本 PR 可独立合入或仅删除这两个新增文件回退；二维 SAVE 工具没有对本模块的代码依赖。
