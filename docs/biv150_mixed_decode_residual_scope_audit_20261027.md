# Mixed decode residual-scope audit

2026-10-27，独立 CPU/源码审计。当前团队基线为
`241220723a09061c182764f8f738e5ec01190367`。本文不包含新 kernel、生产改动、
远程访问或整轮 profiling；不提议从旧结果直接恢复部署 PR20。

## C3 日志不是剩余未优化 decode 的计数

`biv150_compute_order.make_plan` 先在原 `scheduled` 顺序上计算
`mask = scheduled == 1`、连续 `prefix`、全部单 query 数 `dall`，再构造
`P = [flatnonzero(mask), flatnonzero(~mask)]` 和 `compute_q=scheduled[P]`。
`Plan` 保存的 `prefix/outside=dall-prefix` 仍是原顺序统计。
`forward_returned` 仅在 `selected==1` 时打印它们。

所以 `Dprefix=1/Dall=62/outside=61` 表示原先有 61 个单 query 请求不在连续
前缀内；该次 C3 选择后，它们已在本地计算顺序的 62 个单 query 前缀中。
它不表示仍有 61 个请求交给 2D attention，也不是每步分发追踪。
`scheduled==1` 本身还是 query 长度分类，不等同于基于 prompt/computed 状态
定义的全部调度阶段分类。

该逻辑只重排本次计算的 staging/metadata，并用 inverse logits 索引返回原请求
行顺序；持久请求表与 sampler 请求序不原地排序。不能把它描述为没有任何计算序
变化，也不能把它误解为改变用户可见采样请求对应关系。

源码证据：

- [make_plan](../vllm_fl/patches/biv150_compute_order.py)：原统计在 371–374 行，
  P/inverse/compute_q 在 399–405 行，Plan 参数在 422–424 行。
- 同文件 `Plan.forward_returned` 307–330 行：首次成功返回日志。
- [runner](../vllm_fl/worker/model_runner.py)：4237 行调用 make_plan，
  1981–1997 行用其 block-table/req/token 计算序，2254 行附近恢复 logits 对应。

## 当前调用链及真正可能剩下的范围

当前安装顺序为 decode launch policy → natural-prefix mixed wrappers → GQA →
normalization → Q-fragment/default prefill launch policy。后两者优化 prefill kernel
绑定，不是另一个接管散布 decode 的调度器。

| 路径 | 具体行为 |
| --- | --- |
| C3 适用的 mixed step | 全部 qlen1 先进入本地计算前缀；natural-prefix builder 切出 decode metadata |
| prefix attention | max_query_len=1，normalization 不选，原 GQA wrapper 在其严格守卫通过时选专用 producer/reducer |
| 多 query suffix | 走 normalization 的 generic-2D 结构克隆，并保留当前 stride/Q-fragment/w16 改善 |
| C3 或 prefix 分拆不适用 | 保留原路；若仍有多 query 与 qlen1 混合，GQA 的 max_query_len=1 守卫拒绝它们，2D 路径可能兼做 decode |

这不是对所有调用都选 GQA 的证明。C3 例如拒绝首请求不是 qlen1、N>64、
padding/capture 无法证明、resume/speculation/structured-output 或绑定改变等情形。
natural-prefix 和 GQA 自身还有 metadata/feature/stride 守卫。必须观察真正 operator
分发或 launch grid，不能仅凭 C3 首次日志覆盖整个服务。

源码：[mixed prefix split](../vllm_fl/patches/biv150_mixed_decode.py) 的 `_prefix`、
build/forward，及 [GQA gate](../vllm_fl/patches/biv150_gqa_decode.py) 的 `_eligible`。
当前 normalization 代码是 [biv150_prefill_normalized.py](../vllm_fl/patches/biv150_prefill_normalized.py)。

## cu_q 索引和散写方案已存在于 PR20

[PR20](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/20) 基于旧团队
main `3283672`，实现 `3789b2a`，短尾分发修订 `e444c9c`；只读 GitHub 核实其为
closed、未合并。该方案没有重排请求、修改 sampler 或 scheduler：

- GQA 从 `cu_query_ptr[seq:seq+2]` 得到 query_start，非 qlen1 直接 return；
  Q 使用 query_start 索引，scratch 仍按请求行索引。
- reducer 同样判断 qlen1，将结果散写到 `out[query_start]`。
- normalization prefill clone 对 qlen1 提前 return；其余 query 保持 prefill 路径。

因此“直接 cu_q 读非连续 decode + 散写 + prefill 跳过 qlen1”不是新算法。
[冻结旧源码](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/e444c9c/vllm_fl/patches/biv150_mixed_attention.py)
的 producer/reducer/make_prefill 正是这个实现。

旧正式 operator 入口 10/10 通过，6 组选新路、4 组回退，包括 packed KV、空行、
散布 prefills、prefill-first 与 Graph 0/15/31 decode 行变化。报告的 operator 比值：

| 旧 case | 旧 operator 比值 |
| --- | ---: |
| 63 decode + prefill，4K | 1.3872x |
| 31 decode + prefill，16K | 1.1973x |
| 31 decode + 17-query tail，16K | 1.8962x |
| 散布请求与多个 prefill | 1.0734x |

参照是通用归一化 operator，**不包含团队已存在的 natural-prefix/C3 整体路径**。
不能把这些比值叠加到当前 main，也不能把 scheduler mixed 次数解释成此 operator
的选择次数或服务耗时占比。

旧独立服务实际源码 e444c9c，完整 Level3 102/105、零预测错误；4K 256/256，
5017.400661 tok/s、TTFT 4756.173660ms。16K scored run 被 SIGTERM 打断，
114/128 完成、14 失败，不能验收。旧 gate+mixed 组合另有完整 16K 回退，
它不能单独证明 cu_q 映射是回退原因。旧报告/恢复证据保存在本人旧 mixed worktree
`/private/tmp/flagos-mixed-attention-20261023/docs/`；不应覆盖或重写这些旧结果。

## 已有 profile：516/524 mixed 步骤并没有遗漏 decode

[逐 enqueue 关联报告](evidence/baseline-profiler-20261025/v1-enqueue-shapes/report.md)
已验证全部 21,672 个 GQA launch、516 个 mixed step，其 `grid.x` 等于该步全部
scheduled decode-request 数。匹配缺失/歧义/线程归属错误均为零。

只有 8 个 mixed step 没有 GQA。只读复算同目录 `per_step_kernel_times.csv`，
再关联已保存 worker metadata，得到：

| step | decode 数 | prefill query lengths | 该步全部 normalized attention GPU ms |
| ---: | ---: | --- | ---: |
| 1154 | 60 | 1988 | 91.475659 |
| 1155 | 60 | 572,1416 | 109.272971 |
| 2179 | 60 | 921,1067 | 103.191569 |
| 2180 | 61 | 1987 | 100.244713 |
| 2181 | 60 | 800,1188 | 107.597151 |
| 3205 | 60 | 825,1163 | 98.559364 |
| 3206 | 61 | 1987 | 105.358013 |
| 3207 | 60 | 946,1042 | 94.823005 |

总和 **0.810522445s**，除以旧活跃窗口 **231.396551s** 为 **0.3502742%**。
这只是该组原有 attention kernel 可移除成本的极宽上界：包含真实 prefill，
替代实现仍需 prefill 加 GQA/归约，不能把全部 0.35% 当可回收收益。它也不包含
所有 wrapper/launch/downstream 成本，不能直接作为完整服务收益上界。
这些是旧 v1 基线的 GPU 时间，不是当前 w16 服务的预测。

8 步具体 eligibility 拒绝原因没有记录。scheduler 字典的迭代序没有被证明等于
InputBatch 请求序，不能从它猜“首请求一定是 prefill”。仅观测 8/524 的频率也
不能替代上述按实际 kernel duration 计算的成本。

## 旧实现不能直接安装到当前 main

C3 `_main_stack` 要求 normalization-over-GQA 的精确闭包与源身份；外加旧 mixed
wrapper 会拒绝原 compute-order 路径。旧 `make_prefill` 还只将函数名
`kernel_biv150_prefill_normalized` 替换为 mixed 名，而 main 已使用
`kernel_biv150_prefill_stride`；直接复制旧模块会在 namespace 查找时失败。
任何重新研究都必须保留当前 normalization、Q-fragment/w16 和编译缓存身份，
不能先损坏原路再把恢复它称为新增收益。

旧 scope 只有 8..64 请求，偏向单 dominant prefill 或短尾；并非所有 mixed 调用。
它按总请求数选择 segments，当前 GQA 按 decode-prefix 数选择，所以部分边界的
微测还混有分段差异。PR31 已试部分相近分段，不应重复包装。
builder scratch 不兼容时，旧模块每 stream 分配 8.125MiB 备用 scratch；
producer/reducer 为 prefill 行启动后 return，prefill grid 也有 qlen1 空 CTA。
无排序并不意味着没有额外 launch、indirection、scratch 或空 CTA 成本。

最小后续证伪条件是先找到**当前 main 上实际拒绝 C3/前缀拆分、耗时有意义的残留
operator 调用**，然后只复现那组 metadata/布局，并与当前完整调用链配对比较。
复用旧的边界/Graph 测试，保留 sampler 顺序；不要重跑整轮 profile，不要从
`outside` 计数创建又一份 PR20。不满足该条件时，此方向归为历史重复。

## V8 精度恢复仅保留条件性假设

目前只有冻结 b8 的 owner-reported 单 pointwise 违例，max_abs 约 0.01033；
实际失败坐标/完整 actual 输出未保存在该旧 probe 中，不能归因给尾页、短序列、
BF16(P*scale) 或半边界 codes。max_abs 也未必是违反相对容差的那个坐标。
先等带证据保存的数值诊断与 4K/B64、16K/B32 完整组件吞吐，不动原精度 gate。

若未来坐标证据确认主因是权重舍入，可在 CPU 分别估算：令 A=P*scale、
H=bf16(A)、R=A-H、C 为 V codes，则精确数学上 `H*C + R*C = A*C`。
额外 FP32 residual/尾部补偿只针对 H 的舍入，不能恢复 V 量化损失、修正
canonical/actual half-tie code 差，也不能自动修正 FP32 QK/exp/FMA 或最终 BF16
输出舍入。未知失败行时不能预设“只补 partial page/短 shape”就足够。

128 通道的 Vector residual 可能吃掉全部 V8 读收益；第二次 BF16 residual dot
也增加 PV 工作。当前双 cache probe 可比较的 BF16 原值，不代表未来压缩 cache
有免费的 BF16 回退。必须先看到明确吞吐余量，再用实际失败行评估这些成本和
恢复范围；本文不实现、不排队、不宣称这种补偿有效。
