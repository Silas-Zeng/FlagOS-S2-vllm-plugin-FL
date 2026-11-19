# C500 Triton 算子实现

> 当前默认服务按 A5 组合使用原框架 Attention 封装；新增 Triton Decode、Prefill gather、双请求分派和直接输出路径不启用。下文对应内核实现保留用于独立算子测试和原组合对照，不计为当前终版的活跃优化。当前两次性能均值和 103/105 官方参数准确率结果见仓库 README。


2026-11-18 起，本分支的 C500 新增计算算子统一使用 Triton。
此前通过 MACA/cu-bridge 编译加载的 Prefill GEMM 和 GQA Decode 已移除。
基线自带的厂商依赖及不支持形状的原有回退机制保持其原有用途。

| 模块 | 当前实现位置（FlagGems MetaX ops） | 说明 |
| --- | --- | --- |
| Gate/Up + SwiGLU | `packed_mlp.py` | Triton packed GEMM，保留投影 BF16 舍入后进行 FP32 激活与乘法 |
| Down | `packed_mlp.py` | 小 M Split-K；大 Prefill 三列块复用或 Triton mm |
| QKV / O | `prefill_linear.py` | Triton packed 投影与小 M 分片 |
| 分页 GQA Decode | `triton_attention.py` | 组内 8 个查询复用 KV，两阶段 FP32 分片与稳定合并 |

Decode 支持 BF16、Q16/KV2、D128、page16、batch 1–64，K/V stride
`(8192,128,4096,1)`。实际长度和页表在 GPU 上读取；不按请求读取 CPU 标量。
空长度输出为零，支持非连续查询行、随机物理页和 CUDA Graph 重放。
运行时长度不作为编译常量，输出直接写入模型 runner 的缓冲区。

移除的交付内容包括 `bsm_gate.py`、`bf16_attention.py`、
`bsm_gate_native/`、`bf16_attention_native/` 及相应 package-data。
不再使用 `FLAGGEMS_METAX_BSM_*` 或 `VLLM_FL_METAX_NATIVE_DECODE` 开关。

历史文档、报告及证据中包含旧实现的性能数据，不能作为此分支的新成绩。
本轮测试记录保存在 `/data/current_c500_triton_migration.txt` 指向的目录。
没有对测试进程增加内存限制。

## 启动环境（2026-11-19 验证）

使用 MACA 3.7.0.38 时，保持 `MACA_HOST_MEMORY_POLICY=0`（SDK 默认值）。
本环境中 `-1` 会导致 `full_kernel_scale` 在自动调优时出现
`resource limits is null` 和 `Triton Error [MACA]: out of memory`，
256 元素也能独立复现；更换编译缓存不能解决。恢复 `0` 后，
原缓存和新缓存均通过，完整服务也通过健康检查。
该结论针对当前 SDK 与 Triton 组合，不表示所有 MACA 版本都有相同问题。

容器 CPU 内存上限 64 GiB 时，本轮仍使用
`TORCHINDUCTOR_COMPILE_THREADS=1` 与 `TRITON_USE_PROFILER_DO_BENCH=1`
控制编译并行和 SDK 内部自动调优开销。后者影响内核配置选择与启动开销，
不修改官方端到端脚本、模型长度、并发或请求数。这两个环境设置不改变算子的数学定义。

独立复现脚本和结果：`/data/c500_full_diagnosis_20261119_151228/`。

## CU 回退位置的 Triton 调优（2026-11-19）

已分别测量 Gate/Up + SwiGLU、Down、O 投影及分页 GQA Decode。
Gate/Up、Down/O 的分块、流水线和 Split-K 候选未稳定胜过原有 Triton 实现，
因此这些正式计算路径没有新增改动。实验中的 CU 对照仅位于 `/data` 独立目录。

本轮新增改动集中在 `triton_attention.py`：

- QK 与 PV 的 head dimension 拆成 64 或 32 个值的片段，降低操作数布局转换开销。
- 相邻程序先处理同一请求的两个 KV head。
- batch 56–64 使用 D32、序列 tile 32、1 warp；其他 batch 使用 D64、序列 tile 64、4 warps。
- 默认分片数按 batch 选择：1→64、2→32、3–4→24、5–8→12、9–64→6。
- `FLAGGEMS_METAX_DECODE_SPLITS=0` 和 `FLAGGEMS_METAX_DECODE_BLOCK_N=0` 表示自动选择；正值可显式覆盖。
- 序列长度、页表仍是 GPU 运行时输入；没有 CPU 长度读取或固定上下文长度捷径。

13 项独立检查通过，包含随机物理页、非连续查询、空长度、16K 边界、
分支边界和 batch 1/33/64 的 CUDA Graph 输入更新重放。
官方模型准确率测试未在本轮重新执行。

实验、拒绝的候选和端到端结果见 `/data/current_c500_triton_tuning.txt` 指向目录。
调优前同卡 4K/1K 单轮为 9873.69 token/s；CU 对照为 12921.04 token/s。
第一轮调优测得 11446.09 token/s。最终阈值版本单轮为 **11436.72 token/s**，
较调优前提高 **15.83%**，达到同卡 CU 对照的 **88.51%**；TPOT 为 **25.23 ms**。
256 个请求及输入/输出 token 数全部校验通过。两个调优版本结果不混为两次均值。
16K 只进行了本轮算子验证，没有恢复用户已停止的端到端测试。

## 后续 Triton 调度优化（2026-11-19）

在上一版 11436.72 token/s 的基础上，进一步保留以下后端配置：

- D32 窄分块：`scenario="mla"`，1 warp、1 stage。
- D64 宽分块：`scenario="mla;singleshm"`，4 warps、2 stages、`pipeline_load_num=2`。
- 合并阶段：分片数不超过 16 时用 1 warp，否则保持 4 warps。

`mla` 在这里是 MetaX Triton 编译器的矩阵调度选项；数学运算仍是分页 GQA，
没有改变模型结构、数值精度或引入外部计算实现。

13 项正确性与 CUDA Graph 检查全部通过。同卡、相同官方 4K/1K 配置
（并发 64、256 请求）单轮测得 **11760.28 token/s**，TPOT **24.41 ms**，
较上一版提高 **2.83%**。所有请求成功，输入/输出 token 数与源码快照校验通过。
原始 Triton 迁移版为 9873.69 token/s，CU 对照为 12921.04 token/s。
官方模型准确率未在本轮重测，16K 端到端测试未恢复。

不采用有数值误差、编译器断言或没有稳定收益的候选；实验与最终结果保存在
`/data/current_c500_triton_tuning.txt` 指向目录。

## Trace 驱动的布局优化（2026-11-19）

对 11760.28 token/s 版本采集了 64 个 batch 64、query 长度 1 的 Decode 步，
KV 长度覆盖 4350–4541，GPU 内核映射覆盖率 100%。每步 GPU 内核累计时间
约 16.16 ms，其中 Attention 为 10.37 ms，占约 64.2%；GPU 内核间隙约
0.81 ms。该窗口用于定位稳态 Decode 瓶颈，不能直接代替完整请求的 TPOT。
采集插桩位于 `/data` 的插件副本，工作区原有 worker 修改保持不变。

使用相同页表行跨度、查询行跨度及记录的 KV 长度进行 Triton 读取对照：
Attention 主内核约 0.247 ms，纯分页 K/V 读取约 0.187 ms。
这里比较的是不同工作量内核的耗时，不能据此直接计算硬件访存停顿比例。
主内核编译结果没有寄存器溢出。32 个候选中，手工展开触发编译器断言，
大 tile 出现寄存器溢出，其余没有收益的分块、调度及工作块重排均未采用。

最终修改 `triton_attention.py` 中 batch 56–64 的窄分块路径：QK 使用
token 在前的分数布局，PV 使用维度在前的累加器，概率矩阵在两次乘法之间
保持相同方向，最后转置写回分片输出。输入输出仍为 BF16，累加、softmax
和分片合并使用 FP32；长度和页表继续从 GPU 运行时输入读取。

13 项 FP32 参考与 CUDA Graph 检查通过。交替测量当前版本与候选版本的
完整 Decode 算子，结果如下；这些微基准不作为官方端到端成绩。

| Batch | KV 长度 | 修改前 ms | 修改后 ms | 耗时降低 |
| --- | --- | --- | --- | --- |
| 56 | 4097 | 0.22346 | 0.21134 | 5.42% |
| 64 | 4097 | 0.23363 | 0.22342 | 4.37% |
| 64 | 4608 | 0.25490 | 0.24383 | 4.34% |
| 64 | 5120 | 0.28363 | 0.27086 | 4.50% |
| 64 | 16385 | 0.86015 | 0.81673 | 5.05% |

官方 4K/1K 配置（并发 64、256 请求）单轮测得 **11957.35 token/s**，
较上一版 11760.28 提高 **1.68%**；TPOT 从 24.41 降至 **24.06 ms**。
较最初 Triton 迁移版 9873.69 提高 **21.10%**，达到同卡 CU 对照
12921.04 的 **92.54%**。256 个请求全部成功，输入 1048576、输出 262144
token 均校验通过，计时期间源码哈希一致。CPU 内存峰值 47.66 GiB，
没有新增 OOM。不同版本各一轮的结果不合并为重复测量均值。

本轮没有恢复 16K 端到端测试，也没有重新执行官方模型准确率测试。
Trace、全部候选、正确性、官方压测与交付审计保存在
`/data/c500_profile_opt_20261119_171258/`，汇总为 `final_results.json`。

## Decode 投影的验证与调优（2026-11-19）

针对 batch 33–64 验证 Gate/Up、Down、QKV、O 和 LM head 的分块、
乘法方向、Split-K 与流水线配置，最终采用两项已有 Triton 内核的配置调整：

- `qkv_rope_cache.py`：QKV 分片投影的 K tile 从 128 改为 64，
  `num_stages` 从 2 改为 4。分片数保持 2，编译结果共享内存从 64 KiB
  降为 32 KiB。
- `prefill_linear.py` 与 `linear_residual_rms.py`：小 batch O 投影使用
  3 stages，覆盖独立投影和融合 residual/RMSNorm 两条调用路径。

BF16 舍入位置、FP32 分片累加和输出布局保持原定义。62 项已有检查通过；
另对 batch 33/48/64 的六组完整链路进行新旧对照，初始输出逐元素一致，
并验证 QKV 路径的 Graph 输入/权重刷新、负页槽及三个 padding 行。
完整 O 投影加 RMSNorm 耗时降低约 8%，QKV 加 RoPE/KV 写入降低约 3%–4%。

轮转 32 组不同权重（O 为 256 MiB，QKV 为 320 MiB），避免只比较单层
权重反复使用时的表现，测得 QKV 投影内核从 0.01756 降至 0.01654 ms，
降低 5.77%；O 投影内核从 0.01525 降至 0.01350 ms，降低 11.47%。

官方 4K/1K 单轮（并发 64、256 请求）结果为 **11997.84 token/s**，
上一版为 11957.35，观测差异 **+0.34%**；TPOT 从 24.06 降为 **24.00 ms**。
这是较小的单轮端到端差异，不能据此认定稳定提升幅度。
256 个请求、1048576 输入 token、262144 输出 token 及计时期间源码哈希
全部校验通过。KV 缓存仍为 **1114304 tokens**，没有新增常驻 GPU 缓冲区；
CPU 内存峰值 52.82 GiB，没有新增 OOM。

Gate/Up、Down 及不复制权重的 LM head 候选没有一致收益。
packed LM head 的算子虽快约 11%–13%，但额外 510 MiB 权重会压缩约
1.24 万个 KV token 的空间，故未纳入正式路径；候选及其七项检查存于实验目录。
三步 Top-p 搜索通过 cutoff、采样 ID、RNG 和 Graph 更新对照，真实 logits
下耗时反而增加，也未采用。

本轮未恢复 16K 端到端测试，未重测官方模型准确率。
配置、完整链路对照、实验候选、官方结果及 wheel 审计见
`/data/c500_decode_gemm_20261119_180735/`，汇总为 `final_results.json`。
