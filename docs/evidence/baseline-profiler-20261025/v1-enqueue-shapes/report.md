# v1 4K：通过 enqueue correlation 归属的 MM / attention 形状分析

本报告基于已完成 v1 trace 与 worker metadata，仅作只读 CPU 分析。没有 GPU 操作，也没有修改已共享的 v1 分析脚本。所有结果为 diagnostic_only，不能充当正式分数或优化收益。

## 主要结论

- 非 Graph MM 的 **99.2395% 调用、99.2418% GPU 时间来自 M=2048**：522/526 个非 Graph 步骤，69,948/70,484 次 MM，33.730174/33.987885 秒。四个尾部步骤分别为 M=2030、1848、1862、1862。
- 固定 M=2048 时，134 个 MM/步合计 **64.6172 ms**，标准差 **0.0658 ms**，min/max **64.3415/64.8002 ms**。其耗时基本不随 prefill/decode 组成变化。
- 固定 M=2048、且确实运行 GQA 的 512 个步骤中，prefill attention GPU 时间与 prefill causal query-key pair 数的 Pearson r=**0.993896**；GQA GPU 时间与 decode request 数 r=**0.976099**，与 decode query-key pair 数 r=**0.984110**。它们是观测相关性，不是因果优化收益。
- **8 个 mixed 步骤没有 GQA launch**，虽然调度了 60/61 个 decode requests。这些步骤 normalized attention 平均 **2.412269 ms/kernel**；其它 516 个 mixed 步骤为 **1.225976 ms/kernel**。没有记录具体 eligibility 分支，不能据此断言缺失原因。

## 关联方法和完整性

目标 kernel 为 mm_kernel、kernel_biv150_prefill_normalized、_gqa_decode_attention，共 **114,248 条**。每条都通过同一 processId 的 correlationId 唯一匹配 cuLaunchKernel/cudaLaunchKernel；CPU enqueue 的完整区间位于同线程、唯一 worker.execute_model NVTX 范围内，再关联该步骤的 scheduler metadata。匹配缺失 **0**、歧义 **0**、范围/线程不符 **0**。

**63,000 条 kernel 在所属 worker CPU 范围结束后才开始执行，63,378 条在范围之外结束。** 因而本分析保留完整 GPU dur，通过 enqueue 归属，不用 CPU 时间边界裁 GPU 区间。总耗时与已共享 v1 kernel 汇总逐项一致。

所有 21,672 个 GQA launch（516 步）都验证了 runtime grid.x 等于该步全部 scheduled decode-request count，且其余 grid 维度为 KV heads=2 和实际 segments。冻结源码的 launch grid 明确是 `(num_seqs, 2, segments)`。因此这 516 步可以确认 GQA query 数覆盖了全部 decode 请求；8 个无 GQA 步骤的具体 split 失败原因仍未知。

## M 与 MM shape

M 不是从 grid 反推：它来自 runtime scheduled-token metadata，加上已核对的 eager 路径。当前 TP=DP=1、enable_sp=false；全部目标步骤 M>1847，大于实际 max-capture-size512，SDK dispatcher 直接返回 `NONE, BatchDescriptor(num_tokens)`；model input slice 使用该长度。因此没有额外 token padding。相关源码/hash 在 padding_source_evidence.json。

MM 参数寄存器本身没有录入 trace，这是证据边界。实际 grid/block/shared-memory 是直接记录；N/K 和投影角色按冻结模型调用顺序推断，并对每步 134 次 MM、42 层/8 个 BF16 edge gate、完整 grid/block/shared-memory 序列逐项校验。**O/down 共用硬件几何，不能只靠 grid 区分。** 下表的角色拆分应保留这一标记。

| M | 步数 | MM 次数 | MM GPU 秒 | MM 平均 µs/launch |
|---:|---:|---:|---:|---:|
| 1848 | 1 | 134 | 0.064425 | 480.780 |
| 1862 | 2 | 268 | 0.128773 | 480.495 |
| 2030 | 1 | 134 | 0.064514 | 481.448 |
| 2048 | 522 | 69948 | 33.730174 | 482.218 |

M=2048 的模型顺序归属（N/K/角色为上述源码顺序推断）：

| 投影 | M×N×K | 次数 | GPU 秒 | 平均 µs/launch |
|---|---|---:|---:|---:|
| o | 2048×2048×2048 | 21924 | 5.355769 | 244.288 |
| down | 2048×2048×6144 | 21924 | 15.886079 | 724.598 |
| qkv | 2048×2560×2048 | 21924 | 6.637276 | 302.740 |
| bf16_edge_gate_up | 2048×12288×2048 | 4176 | 5.851050 | 1401.114 |

实际主 tile 的 BLOCK_M=256，所以 1848/1862/2030/2048 都有 8 个 M tiles；它们的 grid 可以相同。仅看到 grid 恒定，不能声称所有 M 都等于 2048。对应冻结 IR / 实际 tuner 选择的独立证据：`/private/tmp/biv150-mm-2048-config-evidence.json`（当前 tuner DB 是可变缓存，冻结 v1 source 与实际 trace 几何才提供运行佐证）。

## Prefill / mixed 平均 kernel 时间

| 阶段 | 步数 | MM 总秒 / 平均µs | prefill attention 总秒 / 平均µs | GQA 总秒 / 平均µs |
|---|---:|---:|---:|---:|
| prefill | 2 | 0.129363 / 482.698 | 0.104138 / 1239.733 | 0.000000 / — |
| mixed | 524 | 33.858522 / 482.205 | 27.379874 / 1244.087 | 12.174832 / 561.777 |

纯 prefill 只有最初两个步骤，不能把其均值推广为一般 prefill 分布。mixed 包括 524 步，其中 516 有 GQA，8 没有；GQA 的每 kernel 均值只对实际发生的 launch 求均值，按步汇总则保留未 launch 的零值。

## 上下文和请求数组成

Prefill causal query-key pairs 按每个 prefill request 的 prior computed tokens `c` 和本步新增 prefill tokens `q` 计算 `q*c + q*(q+1)/2`，再求和。它避免把“最大 context”错误地应用到整批 token：251 步同时处理两个 prefill requests，最大 context 很高时，主要 token 可能属于另一个低 context 请求。8 个无 GQA 的 mixed 步骤可能由 normalized kernel 兼做 decode；其额外 decode 工作不包含在这个 prefill-only pair 指标中。

| 每步 prefill pairs | 步数 | prefill attention GPU 秒 | 平均 µs/kernel |
|---|---:|---:|---:|
| 0-2500000 | 68 | 2.044832 | 715.978 |
| 2500001-3500000 | 125 | 4.860775 | 925.862 |
| 3500001-4500000 | 121 | 6.432740 | 1265.789 |
| 4500001-5500000 | 122 | 7.669301 | 1496.741 |
| 5500001-6500000 | 90 | 6.476363 | 1713.324 |

按实际 scheduled decode request 数分组（已对存在 GQA 的步骤逐个验证 grid.x 一致）：

| decode requests | 步数 / 有GQA步数 | GQA GPU 秒 | 平均 µs/kernel | 平均 ms/步（含无GQA） |
|---|---:|---:|---:|---:|
| 0-0 | 2 / 0 | 0.000000 | — | 0.000 |
| 1-4 | 9 / 9 | 0.023725 | 62.764 | 2.636 |
| 5-16 | 24 / 24 | 0.127181 | 126.171 | 5.299 |
| 17-32 | 32 / 32 | 0.359804 | 267.711 | 11.244 |
| 33-48 | 33 / 33 | 0.668755 | 482.507 | 20.265 |
| 49-63 | 426 / 418 | 10.995369 | 626.303 | 25.811 |

进一步固定 **M=2048、61 个 decode requests、1987 个 prefill tokens**，只保留有 GQA 的 212 步。下表 context 是实际 decode requests 的平均 `computed_tokens + 本步query_tokens`，而非请求上限：

| 平均 decode context tokens | 步数 | GQA GPU 秒 | 平均 µs/kernel |
|---|---:|---:|---:|
| 4097–4352 | 38 | 0.947376 | 593.594 |
| (4352,4608] | 50 | 1.334796 | 635.617 |
| (4608,4864] | 64 | 1.756941 | 653.624 |
| (4864,5120] | 60 | 1.659800 | 658.651 |

完整精确 scheduled M、prefill token 数、decode request 数、context 范围、实际 GQA grid 分布均保存在 grouped_dimensions.json；逐步原始特征与 GPU 总时在 per_step_kernel_times.csv。关联示例在 association_samples.json。

## 已有 PR31 覆盖：不要把六个 batch 重复作为新方向

已核对本地保存的 [PR31 当前说明](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/31)：head `d79b7b1a012943dce867d1adcc29a45e0d7b79bf`，实际验收 runtime `e972997`。该 PR 已探索 **B29/30 → 4 segments、B60..63 → 2 segments**。PR 记录完整 Level3 102/105、4K 5766.410667 tok/s、16K 4048.277609 tok/s；相对 PR28 的独立轮次仅 +0.0104% / +0.0035%，没有证明宏观增益。PR44 或后续方案不应把这六种选择重新包装为新方向。

为给这条已探索方向提供实际权重，新增独立脚本将 `_reduce_partials` 同样按 enqueue correlation 关联，验证全部 21,672 对 producer/reducer 在各自步骤内严格交替。以下为 **当前 v1 基线成本**，不是 PR31 的候选耗时：

| decode B | 步数 | 基线 segments | PR31 已试 segments | producer + reducer GPU 秒 | 占活跃窗口 |
|---|---:|---:|---:|---:|---:|
| 29 | 2 | 8 | 4 | 0.024540384 | 0.0106% |
| 30 | 2 | 8 | 4 | 0.024712542 | 0.0107% |
| 60 | 167 | 4 | 2 | 4.478870824 | 1.9356% |
| 61 | 212 | 4 | 2 | 5.792202426 | 2.5031% |
| 62 | 9 | 4 | 2 | 0.216997705 | 0.0938% |
| 63 | 8 | 4 | 2 | 0.189358401 | 0.0818% |

合计 **400 步、16,800 次 producer 和同数 reducer**：producer 10.551540722s，reducer 0.175141560s，两者共 **10.726682282s，占活跃窗口 4.6356%**，占已可见非 Graph GQA 两 kernel 成本 86.5703%。16,800 次 producer 也与 PR31 引述的 PR32 v5 4K 运行覆盖数一致。

该占比不含 wrapper、metadata、launch 或其他下游成本，不是完整服务成本，也不是候选加速比或收益上限。v1 opaque Graph 内部不能按这些 batch 分配；不能从这里推断全模型 attention 占比。PR31 已有完整服务结果应保留，后续增量判断需要结合 v2 Graph 证据及完整成本，不能只因这里有局部权重便重复同一分段实验。

原始 PR 快照 SHA、说明正文、逐 batch 数据和全部关联验证在 `pr31_existing_batch_exposure.json`；独立复现脚本 `/private/tmp/analyze_biv150_v1_pr31_exposure.py`。

## 复现与限制

脚本 `/private/tmp/analyze_biv150_v1_enqueue_shapes.py` 只读取原始 SQLite/worker JSON，写新的结果目录。旧共享脚本完全未改。

```sh
python3 /private/tmp/analyze_biv150_v1_enqueue_shapes.py   --root /private/tmp/biv150-baseline-profile-v1-evidence   --out /private/tmp/biv150-v1-enqueue-shape-analysis-rerun
```

原始 trace SHA256：`3c52ddb6b3aa82266bd26b5181510648d946a72b23cc38ebb2cd2376b5ddd66a`。本报告只覆盖 v1 可见的非 Graph kernels，不能推断 3718 个 opaque decode Graph 内部的 MM/attention 占比。静态缓存配置、源码推断、实际 GPU enqueue 观测已经分开标注。
