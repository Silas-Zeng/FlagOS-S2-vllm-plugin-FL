# PR128 完整 gate 计时负结果

该版本数值通过，但固定 `M2048 / K2048 / N12288` 的完整 gate 路径明显慢于已接受参考，建议关闭此实现且不合并。此结果不否定融合 GEMM、BF16 投影舍入和 SiLU×up 的优化方向。

计时源 `bee62b9e9a6daf849862dd303968dc0ae0a95725`；GPU 编译源仍为 `fbc6858bdccf7688936f9d0887b65e855762ab83`。候选实际 SO 为 `8eefb1b8012b62e6849eeaf5fc43f9600746a39f19354532255ba899d1c9e592`，参考 SO 为 `5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe`。生产代码未接入该候选。

| 口径 | 参考 event ms/step | 候选 event ms/step | 参考/候选 | 候选延迟倍数 |
| --- | ---: | ---: | ---: | ---: |
| Graph | 0.708319 | 7.300757 | 0.097020 | 10.3072× |
| Eager | 0.715389 | 7.309234 | 0.097875 | 10.2171× |

Host wall 同向：Graph 为 `0.712057 → 7.303428 ms/step`，eager 为 `0.725597 → 7.323612 ms/step`。全部八条 ABBA 记录与精确均值保存在 [summary.json](summary.json)。每种口径、每个角色预热 `128×4=512` 步；每条计时样本为 `16×4=64` 步，每个角色两条样本。没有把冷启动成本当收益。

两臂都包含激活量化和输出分配：参考直接调用原 `gate_silu_cached_probe`，候选调用同库 `quantize_rows_probe` 后分配输出并调用 `gate_out`。参考没有数值验证额外 `copy_`。权重、列 scale 和 marker 的值与地址固定；不声称观测到了厂商权重 packing 的命中计数。Graph 捕获中的 Python 调用和分配不等于 replay 时的 host 成本；主线 `FULL_DECODE_ONLY` 下 M2048 prefill 的对应口径是 eager。

此前五阶段数值独审 `57bc147def00db9c3d7b1c0421835f9e9d23e3ae2b91129b172cf71040775958` 已核十份完整输出零差异及 guard、只读、Graph B-before-fresh B、side C。本次计时的十份原件共 92,440,624 B 全 SHA 核对，十六次完整输出合计 201,326,592 个 BF16 值全部逐字节等于原数值 A/B 输出；新组合 Graph B 与随后 fresh B 相同。Root 全文审读复核器并独立执行，得到相同报告 SHA `6ce6c0363af2042191349e02eac869c60c3a6612bffb3031fe3ae7cf2edd7838`。

记录边界：

- 原计时仅保存每步 event/host 值，没有原始 event 端点或每 replay 时钟。源码固定除以 `64=16×4`；独审重算八条有效记录的 ABBA 均值和比值。
- 每次检查保存只读及 descriptor 稳定的布尔结果，没有逐次完整 metadata。实际 A/B 输入字节与已审原件核对，冻结源码负责每次比较。
- 计时输出为普通分配，没有新 guard backing；完整字节关联旧 guarded 数值结果。没有重跑旧 side C 或新增 guard 测试。
- 仅 synthetic 权重组件，不覆盖真实 34 层权重、模型精度或服务吞吐；单轮组件 ABBA 不授予统计稳定性结论。

本目录只包含精选 README 与 JSON。完整数组、日志、二进制和完整审查报告保留在私有归档；原失败与负性能记录未删除。
