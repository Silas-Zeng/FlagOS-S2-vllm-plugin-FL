# BI-V150 / MiniCPM5-2B 全栈优化扩展调研

调研日期：2026-10-25。主代理与三个智能体并行研究：其他模型适配、推理系统、比赛与论文；另核对队友实验原件。本文提供可验证的实验切口，不承诺收益；不同候选收益不可相加。旧 80 项继续保留，本轮的“新增”是相对该清单新增细节，不表示团队或学界从未探索。

本轮去重后保留 **71 张实验卡：66 项性能候选与 5 项配套使能/评估方法**。包含新切口、旧路线深化和远期条件项，不与旧 80 项简单相加；AD01/AD07/AD08/P26/P30 是配套方法，不单列吞吐收益。

## 1. 结论先行

后续值得同时保留四条线：**真实投影/attention 的新数据流，编译图中多余复制和数值转换，降低临时显存峰值并减少 KV 重算，以及按误差结构设计量化**。不能依据早期 profile 宣布“只剩稀疏 attention”或“已经到顶”；也不能因为框架方向多，就假设 CPU 是当前主要瓶颈。

这轮特别值得关注其他模型适配中的具体工程机制：非连续 residual 的直接消费、functionalization 后多余 clone 的消除、view/reshape 归一化后再融合、量化打包布局与 epilogue 协同、GPU 常驻工作计划、按资源约束进行任务调度。MiniCPM 是 dense GQA 模型，Qwen 的 GDN、DeepSeek 的 MLA/DSA、MoE 路由本身不能直接移植。

本轮仅本地源码、实验记录和公开资料研究，没有连接比赛实例、启动 GPU 作业或改变当前实验队列。

## 2. 最新版本与队友记录带来的修正

成功 fresh fetch 后，框架团队 `main` 为 **0bf9d2efeaa10144dd7f0ff4597298df029f9328**；实验库 `main` 为 **ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d**。实验库本地分支有分叉或未提交内容，本轮仅读取远端提交，没有合并覆盖。官方比赛分支与 FlagGems v5.3.5 不升级。

| 已读证据 | 对本轮清单的影响 |
| --- | --- |
| [Q fragment 已合入 PR23](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/docs/07-prefill-q-fragment-biv150.md)：官方两档四轮结果、shared 64→32 KiB、循环内 Q shared load 消除 | “把 Q 移出循环”已经完成。新研究须针对其他不变量、布局/寄存器分配或不同编译阶段，不能重复归功。 |
| [PR35 编译缓存语义](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/35)已合入 | 缓存身份完整性已是基线，不再当待做吞吐创新；其余编译改动仍须核对真实生成图。 |
| [队友 GPU 热点](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/team/methods/gpu-hotspots.md)：旧 mixed 窗口 2D attention 约 60.23%、GEMM 31.72% | 说明纯 decode 的旧热点比例不能外推整个官方负载；比例只属于该窗口，不能当当前 main 的整场占比。 |
| [输入准备诊断](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/team/methods/input-prep.md)：等待仅占预处理累计 0.0408%/0.0970% | 输入同步不列高优先级；CPU 其他开销也要测关键路径，不能用源码行数代替成本。 |
| [C3 视图清理](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/team/methods/c3-views.md)：短形状组件约省 5.97%，服务表观仅 +0.0177% 且容量不同 | 简单重复 view 缓存已探索，不能重新作为大收益建议。 |
| [可见 prefix mask 省略](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-22-visible-prefix-service.md)：组件小正，首次短服务约 −2.37%，新形状预热不完整 | 保留负结果和冷编译边界；重试需新结构或明确冷成本证据，不能重复原方案扫描。 |
| [CTA 分组](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-22-mm-cta-service-comparison.md)：gate 短服务约 +1.75%，KV 差 96 tokens；down 组件曾负 | CTA 排列是深化项，按投影分开评估；该短筛不是当前主线或正式收益。 |
| [固定 2048 预算调度审计](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-22-fixed-budget-scheduling-review.md)：prefill 向下对齐可能增加 M 形状/launch；全输出预留曾无正收益 | 不把 chunk 对齐、保留全部未来输出容量当免费优化。新调度必须计算未用预算、forward 次数、尾块及 TTFT。 |
| [含框架 PR24 组合的恢复窗口](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/pr24-kv-diagnostic.json)：抢占后目标 attention 168 次、累计约 1.018 秒 | KV 恢复有真实可研究的重复计算。该窗口混有其他请求、KV 容量变化及冷调优，1.018 秒不能全部算作可省延迟；已有后继 VERIFY1 工作应衔接。 |
| [历史 KV offload 官方结果](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/q-gqa-kv-official-final.json)：完整四轮 4957.95/3669.74，独立精度记录 103/105，`gain=null` | 不能说 KV 恢复从未实现或从未跑正式轮；也不能跨 DB/KV/代码版本宣称已证明当前主线净收益。 |
| [PR26 最新结论](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/b0ca19b/docs/biv150_residual_cache_probe_20261024.md)：99/105 后已补测 4K/16K，5751.16/4042.31，相对其受测对照 −0.2644%/−0.1474%，现关闭 | 用户要求的“近门槛失败先测吞吐”已执行。该版本未见可用于恢复精度的额外收益，不能继续写成待补吞吐；此结论不否定全部残差量化。 |
| [PR36](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/36)、[PR33](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/33)、[PR37](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/37)分别已有 BF16 投影、norm 量化、真正 INT8 GEMM→SiLU 融合探针 | 这三项属于正在探索的实现，不另起同构原型；编译失败也不能算数学或性能已证伪。 |

旧记录中“int8 KV 一律禁止”“int8 GEMM 无法落地”“无抢占”“prefill 占比只有 1%”“所有优化已到顶”等结论均有特定时间、实现和负载边界。自主开发量化在规则中列为允许，后续 native gate 已集成，后续也有真实抢占和 mixed profile；本轮采用较新证据。另一方面，稀疏丢页即使通过 105 题，也不自动满足“模型核心能力不可明显下降、评测须保持可比性”的限制。

## 3. 为什么显存容量值得与算子速度一起优化

按模型配置计算，BF16 KV 每 token 为 `42层 × 2(K/V) × 2KV头 × 128维 × 2字节 = 42 KiB`。单请求 16K 输入加 1K 输出为 **714 MiB**；64 个全部驻留需 **44.625 GiB**，尚未计模型和工作区。实际历史 KV 容量约 50–55 万 tokens，只相当于约 28–31 个这样的完整上下文。并发 64 不等于 GPU 上同时驻留 64 个完整请求；请求错峰、前缀命中和回收会改变实际峰值。

相比之下，4K+1K 的 64 个上下文合计约 13.125 GiB。因此某个内核微测更快，却多占数百 MiB 工作区，可能 4K 正、16K 负。节省 1 GiB 在纯容量算术上约对应 2.50 万 KV tokens，但只有分配器确实把它交给 KV，且负载受容量限制时，才有机会兑现吞吐。

大词表 head 是一个明确例子：单份 BF16 权重 **510 MiB**；M2048 的 INT32 中间矩阵约 **1020 MiB**。历史 head 量化已经释放原 BF16 权重，仍出现 KV 容量下降，所以应核查真实 profile/工作区峰值，不能简单归因于双份权重，也不能削减真实内存安全余量来制造容量。

## 4. 候选阅读方法

- **优先**：有当前形状或证据支持，先做最小原型/测量；不代表已证实收益。
- **条件**：只有确认热点、硬件能力、真实覆盖或新结构后才投入。
- **远期**：工程量较大，保留设计空间；不占当前验收队列。
- **深化**：与旧 80 项或队友记录同属一个大方向，但这里给出不同切口/复测条件；不能计成独立可叠加收益。
- 每项均应记录实际执行路径、完整成本、容量变化和数值边界。其他平台的加速比一律不换算为 BI-V150 预期。

## 5. 下一批优先证伪的 12 个切口

这是研究排序，已有设备队列保持原安排。P0 表示先做便宜的证据核查；P1 表示有条件进入组件原型；P2/P3 是条件研究或远期储备。所有候选的预期方向均为推断。

| 顺序 | 切口 / 卡片 | 首个决定去留的证据 |
| --- | --- | --- |
| 1 | **生成图多余复制/转换**：AD02、AD03、S11 | 当前 native/fake schema 对应的 FX/AOT 图确有 clone、往返 cast 或物化 reshape；不存在就不写 pass。 |
| 2 | **native/head 的临时峰值**：S19、S12、AD18 | 找到真正参与 KV sizing 的同时 live 大对象。当前原生 gate 与 head 量化候选分开，不能假设生产 BF16 head 正在生成 INT32。 |
| 3 | **保持舍入边界的真正 epilogue 融合**：AD09、旧 F09 / PR37 | 衔接现有探针，消除 INT32 写回同时保持原 BF16 舍入；不是重复 PR30 的尾处理线程扫参。 |
| 4 | **norm 归约拓扑与有界融合**：AD10、AD11、旧 B06 / PR33 | 同一真实输入行跨 M 的差异能定位到 reduction/cast/FMA；固定拓扑的完整组件仍有净收益。 |
| 5 | **非连续输入直接消费**：AD05、S18 | 真实路径有可移除 contiguous；不能只在合成 strided 数据上快。 |
| 6 | **已发生抢占的恢复成本决策**：S29、S24、S25 | 衔接队友现有 SAVE/FULL→VERIFY1；目标重复计算成本超过完整保存/恢复/等待成本，随后做无 profiler 对照。 |
| 7 | **真实 BF16 投影与大词表布局**：旧 A/G、AD19、AD22 | 接续 PR36；真实 M、权重和完整入口成本，含 packing/scratch/KV容量。编译失败不是性能否决。 |
| 8 | **固定图内有效行处理**：S16 | 有真实 padding 浪费且新内核可跳过；不得破坏 M≤16 残差补偿占用的填充行，闭源矩阵调用不自动省计算。 |
| 9 | **量化格式与 INT8 矩阵单元协同**：P01、P10 | QoQ 解包完整成本、SmoothAttention 的合法 Q/K 缩放及真实误差；旧 INT8 KV 负结果不直接覆盖新格式。 |
| 10 | **精度预算恢复**：P08、P09、AD08 | 用独立校准材料解释敏感层；仅在新候选实测有吞吐余量时选少量 BF16 回退，不重救已关闭的 PR26 原版本。 |
| 11 | **剩余 CPU 提交成本**：S04、S05、S06 | 当前关键路径确有索引构造、事件分配或重复 guard；输入等待旧占比很小，不作为主攻依据。 |
| 12 | **生产者直接输出下游量化输入**：AD12、AD13 | O/down 自研量化先有价值，再做 attention→quant 或 SwiGLU→quant；不先给当前 BF16 消费者多加量化。 |

旧 C01/C03/D08 的 GPU 工作计划、split scratch/reducer 联合布局仍保留，详见旧清单；本轮没有把它们换名重复计数。巨核 P17/P18、资源互补 S33、VMM S28 留作长期结构研究，先过 BI-V150 能力与完整成本门。

## 6. 其他模型适配与算子衔接：18 张卡

本节 A 表示在当前规则边界内自主实现；Q 表示存在额外规则问题。A 仍需数值、服务及代码审核。AD01/AD07/AD08 为配套使能，其他 15 张是性能候选或数值恢复切口。


| 参考适配 | 可以迁移的机制 | 不能直接搬到 MiniCPM 的内容 |
| --- | --- | --- |
| Llama / dense Qwen | packed linear、RMSNorm/SwiGLU 的生产者消费者融合、BF16 舍入契约、shape/stride dispatch | Qwen 的 Q/K norm 不存在于当前实际 Llama 计算图，不能给 MiniCPM 添加它 |
| Qwen3-Next / Qwen3.5，团队 Metax patch | OOT 注册时序、构造期静态布尔标志、packed 权重名字与量化例外映射同步 | GatedDeltaNet、卷积状态、chunk_delta_h、Mamba cache 不是 MiniCPM 算子 |
| Gemma2 / Gemma4，AITER | 分模型几何设置资源上限、分阶段评估融合；多层 norm 的舍入顺序说明“代数等价”并不自动数值等价 | Gemma 的四个 norm、sliding-window、logit softcap、head 256/512 参数不可照搬 |
| DeepSeek / GLM，CUDA / ROCm | opaque op 的融合边界、无效 cast/fill 删除、静态 scale 预处理、非连续 residual | MLA/DSA、稀疏 indexer、MoE 路由与 all-to-all 不存在；不能把其巨大加速比套给 2B dense 模型 |
| Ascend | 编译 pass 与可用内核一致的能力检查；融合时保留 dtype/cast/scale 顺序 | AscendC Cube/Vector、NPU graph、CANN 算子不能直接在 BI-V150 执行 |
| Iluvatar 当前实现 | native C++/COREX 自研核和 Triton 图拼接、真实 eager/Graph 分别验收 | CUDA 兼容不等于支持 Hopper TMA/WGMMA、AMD MFMA/CK、Blackwell FP4 |

主要原始证据：[Qwen 融合 pass](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)、[Gemma2 模型](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma2.py)、[Gemma4/AITER PR4044](https://github.com/ROCm/aiter/pull/4044)、[Ascend 算子注册不一致案例](https://github.com/vllm-project/vllm-ascend/issues/6737)。

### 当前源码落点


以下链接固定到团队仓库 `0bf9d2e`，便于后续与实际导入源码核对。

- L1：[native Python 注册与 fake schema](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/patches/biv150_native_w8a8.py#L23)：`load_library`、`after`、`apply`、`LlamaMLP.forward`；量化权重已在加载后生成，当前只选 gate_up、跳过头尾各 4 层。
- L2：[native C++ 执行](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/csrc/iluvatar/native_w8a8.cpp#L24)：`linear_impl` 有 `x.contiguous()`，aq/sa/output 分配和 `_int_mm` 的 INT32 结果；`gate_silu_auto` 按 M 分支。
- L3：[native CUDA quant/epilogue](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/csrc/iluvatar/native_w8a8.cu#L11)：普通、cached、residual quant 与 BF16 epilogue。
- L4：[RoPE FL](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/ops/rotary_embedding.py#L11)：OOT 使用 view、split、可选 cat；但看板证明实际 42 层共享对象绑定的是 `forward_native`，审计不能只看此 OOT 函数。
- L5：[RMSNorm FL](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/ops/layernorm.py#L11)、[FlagGems normalization 包装](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/dispatch/backends/flaggems/impl/normalization.py#L14)。
- L6：[CachedOp](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/dispatch/__init__.py#L151)、[OpManager](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/dispatch/manager.py#L86)：已有 epoch 缓存；失败记录按 op/impl，而非完整 shape 契约。
- L7：[GraphWrapper](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/compilation/graph.py#L80)、[break_graph 适配](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/compilation/break_graph.py#L1)。
- L8：[runner](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/worker/model_runner.py#L3919)：`_determine_batch_execution_and_padding`；同文件已有采样行 gather，不能把“只算采样行 head”当新功能。
- L9：[专用 GQA decode](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/patches/biv150_gqa_decode.py#L89)：`_gqa_decode_attention`、`_reduce_partials`、`_eligible`；当前 `_eligible` 排除 output_scale 路径。
- L10：[Metax GDN](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/dispatch/backends/vendor/metax/patches/gdn_linear_attn.py#L1)：注释记录 OOT 注册时序与 hasattr guard；构造期 bool 替代动态属性检查。其 zeros 不能盲目改 empty，文中明确存在未初始化累加问题。
- L11：[Qwen3.5 权重/量化映射](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/models/qwen3_5.py#L69)：`hf_to_vllm_mapper` 同步重写量化 ignored_layers。
- L12：[GLM metadata 与 reshape 兼容](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/patches/glm_moe_dsa.py#L55)：metadata producer/kernel 的能力条件曾不一致；RoPE 输出显式 reshape 避免 squeeze 假设。

### AD01：按实际图的语义节点建立融合入口，而非只拦 Python 模块

- **机制/代码**：Qwen/DeepSeek 的融合在 FX 图上匹配，再降到具体后端；我们 L1 的 `torch.ops.biv150_native` 是 opaque 边界，L4 实际又是 native。自研小范围 pass/专用 composite op，把真正相邻的 norm→quant→gate 识别出来，避免仅 patch 一个未执行的 OOT 方法。无需搬完整 vLLM IR。
- **硬件/规则/优先级**：编译框架机制可移植，kernel 仍须 BI-V150 自研；A，P0 使能。旧项：深化 B06/B09/H06，不是新 norm 融合算法，也不是 PR35 缓存键工作。
- **最小实验**：导出当前代表 M 的 FX 与生成代码，逐项列“源节点→命中 pass→生成 launch”；仅一个已通过数值的形状接入，比较整个 norm+gate 区域。未证明多余 launch 存在则不进入开发。
- **原始来源**：[vLLM IR 设计](https://github.com/vllm-project/vllm/blob/main/docs/design/vllm_ir.md)、[Qwen3/Helion 作者实验](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)。

### AD02：view/reshape/split 图规范化，提升同一融合模式的真实覆盖

- **机制/代码**：Qwen QK-norm-RoPE pass 专门统一 view 与 reshape。针对 L1 的 reshape、L4 的 view/split、L12 的显式 reshape，自研规范化只折叠等价形状变换，令同一个融合匹配覆盖 native、OOT 与 packed view，避免同一数学模式被图形态分裂。
- **硬件/规则/优先级**：无特殊指令；A，P1。旧项：B08/B09/H06 的具体编译覆盖深化，不能把 QK-norm 算子搬入 MiniCPM。
- **最小实验**：收集相同张量的 view/reshape 与非连续 view 三类图；证明 alias/stride 等价才替换；对比命中数、拷贝数和组合耗时，不以 match 数当性能。
- **来源**：[QKNormRoPEFusionPass](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)。

### AD03：消除 opaque custom op 前后无效 dtype 往返

- **机制/代码**：DeepSeek 融合曾引入 BF16→FP32→BF16 的无意义 copy，opaque 边界让编译器不能删。审查 L1 fake schema 的 dtype、L2 输入/输出与 B06 replacement；移除的是值不变转换，不能去掉原模型规定的 BF16 舍入。
- **硬件/规则/优先级**：无新硬件要求；A，P0 筛查。旧项：H06/F09 深化，和“提高到 FP32 精度”不同。
- **最小实验**：从生成代码找 `_to_copy/clone/contiguous`，检查输入输出位模式是否恒等；只改一处 ABI 或 replacement 的 dtype，确认对应 launch 真消失且输出 bitwise 一致。
- **来源**：[DeepSeek 融合退化 issue37832，含生成代码](https://github.com/vllm-project/vllm/issues/37832)。

### AD05：让 norm/activation-quant 直接消费非连续行，去掉中间 contiguous

- **机制/代码**：GLM5.2 上游专门支持 strided residual。L2 当前 `x.contiguous()`，可在自研 quant 中支持 row stride、连续 K 维，避免 gather/packed view 后的额外完整拷贝；L5 residual 同理。**当前真实输入是否非连续尚未证实**。
- **硬件/规则/优先级**：BI-V150 向量加载需维持末维对齐；A，P1 条件项。旧项：H02/H07 深化，不等同一般向量化。
- **最小实验**：真实 trace 记 stride/copy 字节；制作 row_stride=H、2H、带偏移三组，比较“contiguous+旧核”与“strided 新核”；真实始终连续则关闭该项。
- **来源**：[GLM5.2 RMSNorm PR49750](https://github.com/vllm-project/vllm/pull/49750)。

### AD07：显式 supports_args，避免一种不支持形状把整个快速实现永久降级

- **机制/代码**：L6 `_failed_impls` 按 op/impl 记失败；一个尾部 shape 异常可让后续全部形状 fallback。对新自研核添加 dtype/stride/shape/设备能力预检，用合法“不支持”分支回退，真正执行错误仍 fail fast，不能吞掉写入一半后的错误重试。
- **硬件/规则/优先级**：无特殊指令；A，P1 使能。旧项：H10 深化。保留主要选择逻辑；不能只切 vendor 实现获益。
- **最小实验**：支持形状→不支持形状→支持形状序列，记录第三步仍命中新核；同时注入真实执行错误验证没有隐藏故障。无历史降级证据则按可靠性收益记录。
- **来源**：[vLLM IR supports_args](https://github.com/vllm-project/vllm/blob/main/docs/design/vllm_ir.md)、[Ascend pass/内核注册条件不一致](https://github.com/vllm-project/vllm-ascend/issues/6737)。

### AD08：混合精度例外按“模型语义层”映射到 packed 权重

- **机制/代码**：L11 展示改权重名而不改 ignored_layers 会悄悄丢失高精度例外。L1 使用 prefix 与 packed gate_up；后续按层/投影回退须维护逻辑 gate/up/Q/K/V 到实际 packed 参数的映射，保留正确量化区间，才能用少量 BF16 回退换回近门槛精度。
- **硬件/规则/优先级**：沿用已开发量化；A，P1 使能/精度恢复。旧项：F05 的落实，不额外算一种量化算法。
- **最小实验**：不读题目结果作 dispatch；生成 42 层×投影的生效清单，逐项核对 tensor hash、被回退层实际 kernel；测试整层回退与仅 gate/up 分量回退的时间/精度代价。
- **来源**：[团队 Qwen3.5 hf_to_vllm_mapper](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/models/qwen3_5.py)。

### AD09：融合内核保留 BF16 舍入点，但在寄存器完成往返

- **机制/代码**：L3 gate/up epilogue、B06 norm→quant 必须复现当前分离实现的 cast 顺序。可以在寄存器中做 FP32→BF16→FP32，再继续 SiLU/quant，省显存中间张量而不擅自改为全 FP32 链。Gemma/Beam 多 norm 适配提供了这种边界必要性的对照。
- **硬件/规则/优先级**：BI-V150 BF16 round/cast 行为需验证；A，P0/P1。旧项：F09/B01/B06 数值深化，衔接当前 PR37/PR33 探针，不重报旧融合实验。
- **最小实验**：逐阶段保存分离路径中间值，比较每个舍入点；针对 PR33/B06 已过与未过 M 分开定位 cast、reduction、FMA 哪个改变结果；保留逐算子相同数学的最小融合核。
- **来源**：[Gemma2 实际计算次序](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma2.py)、[Ascend Beam 适配对 FP32/cast 顺序的要求](https://github.com/vllm-project/vllm-ascend/issues/19143)。

### AD10：固定 RMSNorm reduction 拓扑，用较小性能代价恢复融合精度

- **机制/代码**：不同 M 对应不同 block/warp 会改变 reduction 顺序。对 B06/L5 的问题形状，固定 H=2048 的归约树，把 M 只用于每 CTA 行数与 grid，不让归约树随批次变化；这不是开启完整 batch-invariant 模式。
- **硬件/规则/优先级**：需适配 BI-V150 warp/编译 lowering；A，P1。旧项：B05/F05/H06 深化，目标是让快速融合可过门槛，未承诺固定树本身更快。
- **最小实验**：同一行放进 M1/16/17/40/64/2048，比较 norm bit pattern 与后续 quant scale；固定树版本单独测性能；只对必要形状启用。
- **来源**：[vLLM batch invariance 文档](https://docs.vllm.ai/en/v0.18.1/features/batch_invariance/)、[固定 RMSNorm block 的问题说明51187](https://github.com/vllm-project/vllm/issues/51187)。Ascend 示例要求改服务参数，不能照用：[其限制文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/batch_invariance.md)。

### AD11：量化策略与融合 pass 成对覆盖，避免某个 scale 格式使融合静默失效

- **机制/代码**：Qwen/Helion 实验区分 per-token 与 UE8M0 per-group 后可融合的链不同。当前 L1/L3 是 INT8+FP32 scale；扩展 F02/F07 时必须同步定义 producer output schema 和 consumer scale layout，而非先换低精度格式再期待 B06 自动工作。
- **硬件/规则/优先级**：BI-V150 只选择自身支持的算术；A，P1。旧项：F02/B06/H06 交叉深化；不迁入 FP8/FP4 上游开关。
- **最小实验**：为每种候选量化格式列 norm→quant→GEMM 的实际 kernel 数、scale 字节、cast；比较整个组件，不以 GEMM 独立时延决定。
- **来源**：[Qwen3/Helion 的 per-token 与 per-group 融合差异](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)。

### AD12：attention 最终归约直接输出量化 activation，接自研 O projection

- **机制/代码**：L9 `_reduce_partials` 已得到归一化后的输出；若后续 O_proj W8A8 被独立证明有价值，可在最终阶段做 per-row/group amax+quant，直接输出 INT8/scale，避免 BF16 输出再完整扫描。当前 O_proj 未量化，L9 eligibility 排除 output_scale，所以这不是已有能力。
- **硬件/规则/优先级**：依赖先做可用的 BI-V150 O_proj 自研量化；A，P2。旧项：与 C10“attention+O GEMM 融合”、F10“QK/PV 低精度”不同；相关 F05/E04。
- **最小实验**：先只做 attention→quant 两阶段与融合阶段的数值/总时延对照；量化必须在所有 split 合并后，不能分别量化 partial。当前 reducer 每个 program 仅处理一个 query head 的128维，而 per-token scale 需跨16头/2048维：须选匹配下游GEMM的per-head/group量化，或重构跨头归约/增加amax阶段并计入成本；没有跨CTA同步证明，不能声称现有reducer一次launch就能完成整行动态量化。再接 O_proj 比完整 attention+O。
- **来源**：[vLLM attention-quant 融合设计及后端限制](https://docs.vllm.ai/en/latest/design/fusions/)。注意其 MLA 名义融合仍可能有中间 buffer，不能凭融合名称判收益。

### AD13：SwiGLU epilogue 直接产出 down-proj 量化输入

- **机制/代码**：L3 当前 INT32 dequant→BF16 舍入→SiLU×up→BF16；若 down projection 自研 W8A8 过关，则把激活最终 quant 融入 producer，省 BF16 FFN 激活落地与下一次扫描。与 B02 把 SiLU 放进 down GEMM 的每个 N tile 相比，此方案避免重复激活计算，但要支付跨 6144 维 amax。
- **硬件/规则/优先级**：需要 down 自研核、按 BI-V150 资源设计 amax；A，P2。旧项：B02/F09 的不同融合切分；当前 gate-only 不具备此前提。
- **最小实验**：分别验证静态 scale、per-row、per-group 的真实代价；BF16 舍入遵循 AD09；比较 gate+activation+down 全组件，不能只报 tail epilogue。
- **来源**：[上游 activation-quant 融合 pass](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)、[Qwen3/Helion](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)。

### AD14：输入 embedding gather 与第一层 RMSNorm 合一，保留原 residual 输出

- **机制/代码**：Llama 第一层 residual 来自原 embedding；可一次读取词表行，计算 RMSNorm，分别写原 residual 与归一化输入，省 norm 再读 embedding 与一个 launch。不是 G09 embedding 量化，权重可保持 BF16。
- **硬件/规则/优先级**：普通 gather/reduction，H2048；A，P2（仅每 forward 一次，收益上限较低）。旧项：新增边界融合；相关 B05/G09。
- **最小实验**：先确认比赛安装的 Llama 首层路径与源码一致；含重复 token、特殊 token、随机 vocab 和输入 embeddings fallback；比较 embedding+firstnorm 整段，并确认 residual 精确保留。
- **来源**：[Llama 模型的 embed 与首层 residual 结构](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)。这是从其数据流推导的候选，不是该源码已有融合。

### AD16：把目标 GEMM 所需 padding 的初始化并入 producer store

- **机制/代码**：GPT-OSS/AITER 把 norm 与 hidden padding 融合；MiniCPM H2048 本身不需要该 hidden padding，但 L2 `padded_rows=max(32,rows)` 的**行 padding**可由 B06 的 norm+quant producer 直接生成，取消独立补齐/清零步骤。L3 原 quant 已写 padding，因此只有新融合路径产生冗余补齐时成立。
- **硬件/规则/优先级**：BI-V150 native INT8 行约束；A，P2 条件项。旧项：B06/F04/H07 深化，不把已有 quant padding 算新优化。
- **最小实验**：M17..31 验证 aq/scale padding 已完整定义；M1..16 residual 双份行有专门语义必须独立保留；记录新增融合中是否真的少一次 fill。
- **来源**：[vLLM ROCm AITER norm-padding pass](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)。

### AD17：以完整写覆盖证明删除重复初始化，而非泛化 zeros→empty

- **机制/代码**：DeepSeek sparse-MLA 删除了消费者根本不读的 fill；对 L9 partial/reduce、L2/L3 padding scratch、L8 metadata tail 建立 producer 写集合与 consumer 读集合，只移除能证明无用的初始化。L10 反例说明某些累加输出必须 zeros。
- **硬件/规则/优先级**：通用数据流；A，P1 条件项。旧项：H07 深化，寄存器 `tl.zeros` 是累加初值不能删除。
- **最小实验**：scratch 预填 NaN/随机哨兵，覆盖空有效 split、尾页、重复 replay；观察所有被读取元素均有本轮写入；若无多余 fill 就关闭。
- **来源**：[DeepSeek PR44527 含具体读写证明](https://github.com/vllm-project/vllm/pull/44527)、[Metax GDN 的 zero 初始化要求](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/dispatch/backends/vendor/metax/patches/gdn_linear_attn.py)。

### AD18：native composite 增加 caller-owned output/workspace 接口

- **机制/代码**：L2 每次分配 output/aq/sa，`_int_mm` 返回新 INT32 acc；自研 out 变体允许上层按 shape 分配稳定 scratch，减少 eager allocator/框架封装开销，并给编译器展示生命周期。Graph 内 allocator 往往已池化，所以不得用 eager 分配次数推断 Graph 收益。
- **硬件/规则/优先级**：需要确认 COREX ATen `_int_mm.out` 可用或自研 native GEMM；A，P1。旧项：H07 的明确 C++ ABI 深化；不等于 F09 消除 INT32 materialization。
- **最小实验**：M16/40/64/2048 分别测 eager/Graph；完整管线同时计 workspace 峰值、malloc CPU 时间与 GPU 时间；至少双 workspace 验证重入/异步不会覆盖。
- **来源**：[vLLM custom op/IR 以 alias 管理内存](https://github.com/vllm-project/vllm/blob/main/docs/design/vllm_ir.md)、[vLLM 官方 release 中的 DeepSeek workspace 重用条目49236](https://github.com/vllm-project/vllm/releases)。后者只核实了官方发布摘要，PR 正文抓取超时，未据其推断具体实现。

### AD19：静态 scale/布局转换只在加载期做，按消费核格式保存唯一版本

- **机制/代码**：DeepSeek/AITER 把静态 e8m0→FP32 每步转换移到加载；L1 已经在加载期量化并保存 FP32 scale，**这部分已完成**。新增针对 A09/F07 后续 packed 权重：把 tile-shuffle、scale permutation、反索引一次完成，避免 runtime 小 kernel；不保留无用双份全权重。
- **硬件/规则/优先级**：具体布局必须 BI-V150 重设计，不能拷贝 AMD preshuffle；A，P2，与未来 packed GEMM 同一任务。旧项：A09/F07 实质细化，不单列已存在 weight quantize 的收益。
- **最小实验**：装载后核对逻辑通道的 scale 与权重一致，记录 runtime transpose/cast 是否归零、常驻显存和 KV 容量变化；完整 4K/16K 检查新显存峰值。
- **来源**：[DeepSeek scale 预处理 PR47773](https://github.com/vllm-project/vllm/pull/47773)。

### AD21：同一 forward 的 RoPE 位置查表结果跨层复用

- **机制/代码**：当前 42 层共享同一 RoPE 对象；如果生成图确实在每层重复 gather 同一 positions 的 cos/sin，可在 runner/forward context 生成一次本步 compact cos/sin，供每层旋转消费。缓存的是位置角度，不是各层不同的 Q 值，也不是队友已做 Q fragment hoist。
- **硬件/规则/优先级**：多写一次 compact 表可能反而变慢；A，P2 条件项。旧项：B09/D08 深化；不得向共享 RoPE 对象写层特定状态。
- **最小实验**：先核对 native 生成核，查表是否已被编译器共享或便宜地嵌入；比较“42 次直接 gather”与“一次 gather+42 次读小表”的完整成本。不同 positions 的 Graph replay 必须更新稳定缓冲。
- **来源**：[ROCm ATOM 的 cached RotaryEmbedding 与 fused/standalone 路径](https://github.com/ROCm/ATOM/blob/main/docs/model_ops_guide.md)、[团队 RoPE FL](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/ops/rotary_embedding.py)。跨层 hoist 是本次数据流推断，不是原文已证明收益。

### AD22：按资源模型决定融合边界，独立比较 launch 省量与 occupancy 损失

- **机制/代码**：AITER 的 RoPE+cache 在部分 prefill 曾比非融合更慢，Gemma4 需不同 stages/tile 才避免 LDS 过大。B01 现有两个实现多数慢，下一步应先提取寄存器、spill、shared memory、occupancy，再决定 paired-column、双 accumulator、分阶段 epilogue 哪个边界保留；不是再机械地增大融合。
- **硬件/规则/优先级**：资源阈值必须来自 BI-V150 编译产物；A，P1。旧项：B01/B09/H04/H09/H10 深化。队友 CTA grouping 试验已做，不能作为新结果。
- **最小实验**：只挑两个真实热点 M，与现有基线比“相同输入输出、总 launch 时间、spill、带宽”；用通用 M/stride/shape 条件保留赢家，不按 4K/16K 测试名分支。上游仅调配置的方案不能直接作为参赛开发量。
- **来源**：[AITER RoPE-cache 退化 issue4457](https://github.com/ROCm/aiter/issues/4457)、[Gemma4 resource/phase 调优 PR4044](https://github.com/ROCm/aiter/pull/4044)。

## 7. 系统、运行时、显存和调度：36 张卡


- 本机已有 PagedAttention、APC、chunked prefill、mixed batching、async scheduling、FULL_DECODE_ONLY、跨图共享 pool、由大到小 capture、attention metadata builder 分组复用；不能把“启用这些能力”当创新。
- 插件 main 已含 CPU metadata 复用两次 H2D、sampled-token integer gather/scatter、局部 compute order、native gate 量化缓存、Q fragment 改进。PR35 的模型/native/SO 语义编译缓存已合入，属于正确性/可复现修复，不能再次记待做吞吐项。
- `model_runner.py:3852` 仍有 `prepare_inputs_event.synchronize()`，但队友已测等待只占 preprocess 的 **0.0408% / 0.0970%**。因此“看到同步就删”不成立；下面的快照环/事件优化低优先级，须有新 profile 显示关键路径扩大才启动。
- `PlatformFL.is_cuda()` 只对 NVIDIA 返回真；`worker.py:determine_available_memory` 的 graph estimate 调用受此 guard 限制。**本 BI-V150 不能假设已经扣 graph estimate，也不能假设只节省 capture 后内存就自动扩大此前已确定的 KV 池。** 修改估计器本身可能减少 KV，绝不是天然增容。
- `_C_ops_registry.py` 确有 `weak_ref_tensor=lambda t:t` fallback；但旧账本的 35 图最终 hidden 合计约 16.53 MiB，attention scratch 是一 builder 约 8.125 MiB、42 层共享。没有 >1 GiB 可回收证据，不能按层错误相乘。
- `docs/kv-contiguous-copy.md` / `kv-pitched-copy.md` 的 PR3/4 只是工具，未默认启用服务 offload；但实验仓库已有完整 SAVE/FULL、单层 pitched H2D、服务比较和更早官方轮。**“主线工具未默认启用”不等于“队友没有做过恢复服务/官方测试”。**
- 队友旧“零抢占”日志只证明对应旧运行；后来确实出现抢占。当前 PR24 组合 `1ad9a708` 观察为 KV547600、32 请求全部完成、1 次抢占，重新准入 APC11536，重叠旧 CPU 已提交前缀5446。另一个带 profiler 窗口 KV546528，目标 Attention 168 次累计1018.292405ms；这是局部计算预算，不能当整段可省时延。
- 旧 full-output-growth 已消掉抢占但无吞吐正信号、TTFT变差；finite-growth common-DB 对比描述性吞吐 +1.0005%、TTFT +2.2820%，但 KV差96token，`pair_comparable=false`。不能继续以“避免一次抢占必然更快”为前提。
- 旧双 stream mixed attention 单层正确门通过，wall约+2.50%、event约−0.77%矛盾，没有选为服务优化。新重叠方向必须改变资源互补性或大幅降低提交成本，不能照搬旧两 attention 并发。



标记：**A** 是保持原请求/预算/数值合同的实现优化；**B** 是自研调度或跨设备内存策略，规则文字允许框架开发，但必须保持官方参数、请求/输出总量、正常公平性并在报告解释策略；**C** 是硬件能力/评测覆盖前提尚未证明的远期研究。优先级 0/1/2 表示近期小门、条件研究、储备。所有“未见”仅指本轮读到的 main/记录，实施前还要查团队最新任务。

### CPU、输入准备和输出路径（10项）

| ID | 具体实现增量及机制 | 当前/旧项关系 | 最小证伪实验、适用指标 | 边界/参考 |
|---|---|---|---|---|
| S01 / P1 | 请求生命周期使用稳定 slot，计算顺序仅传 index vector；GPU gather 出本步 block table/采样状态，减少加入/结束时整行搬移及双份 request state | V1已有 persistent batch；MRV2 的“状态与计算顺序解耦”是增量，非再做旧compute-order；旧H08深化 | CPU回放真实join/finish/preempt轨迹，统计移动字节/准备wall；只在churn明显时上单步GPU；稳定decode无churn则否决 | A；保留 request→RNG/采样序映射；[MRV2][SR01] |
| S02 / P1 | 对**有变化**的block table打包(row,start,len,value)差量并单H2D/单scatter；GPU常驻表 | 队友zero-dirty skip已做，不能再报“未变表不传”；本项只省changed rows的全行上传 | 16K跨页增长和request churn，比较真实变更元素/传输字节/额外scatter成本；若页表已按dirty范围传则停止 | A；epoch与撤销更新必须对应；[MRV2][SR01] |
| S03 / P2 | 将可变CPU状态与在途pinned快照分离，用有事件所有权的2–3槽ring，减少整个prepare阶段等待上一copy | 等待已有0.0408/0.0970%负收益上限；不立即重做旧同步实验 | 仅新组合出现明显wait关键路径后，先延迟H2D压力验证不可覆盖，再比较整step；预期上限按新wait计 | A；绝不能直接删barrier；[MRV2][SR01]、[异步race报告][SR03] |
| S04 / P1 | sampled-token重排的两份int64索引合成一个预分配pinned slab和device slab，一次传输、保持已融合gather/scatter | main已有融合kernel，仍在非common-index路径每步构造两tensor并两次`.to`；属于剩余分配/提交成本 | 回放真实重排次数；M1–64和mixed，比较含CPU构造/H2D的完整路径；若命中很少则降级 | A；保留duplicate-destination回退和storage合同；[本地runner][SL01] |
| S05 / P1 | AsyncGPUModelRunnerOutput 的token D2H采用可回收host buffer/event对象池；实际完成后才回收，避免每步新event和新CPU tensor | 当前已用独立copy stream，不是“开启异步输出” | 1024-token decode，统计Python分配/driver事件成本、输出队列深度；完整event+tolist wall无下降即停止 | A；取消和延迟消费者下不复用在途buffer；[本地runner][SL01]、[PyTorch异步复制][SR04] |
| S06 / P1 | 将shape无关的模型/vendor/类绑定/源身份校验冷移；建立不可变能力对象，热路仅查真实shape/stride/alias | 旧H08/K2深化；main normalized/gqa/mixed wrappers仍有Python适配；保留核心dispatch，不删fallback | 已暖eager mixed/prefill的Python采样，数42层调用次数与CPU空洞；改前后route/fallback逐项相同 | A；不能拿“绕过所有选择逻辑”当优化；[vLLM V1][SR05]、[本地patch][SL02] |
| S07 / P2 | scheduler→runner变更使用紧凑SoA/增量record，避免每step重复构造长字典、request副本和大整数list | V1已有delta通信；只针对profile确认仍重复的字段，不能重做整套引擎 | 相同SchedulerOutput语义CPU回放，量allocated objects/序列化字节/CPU；6核环境占比不足即不做 | A；请求id/epoch/取消顺序一致；[vLLM V1][SR05] |
| S08 / P2 | 新请求tokenization/preprocessing移出阻塞API event loop，采用受cgroup约束的有界任务队列，cache不可变tokenizer数据 | 纯文本MiniCPM未见此路径瓶颈，不能套多模态processor大收益 | 请求burst时间戳→入engine时间，CPU quota6核总消耗与TTFT；正式客户端若已传token则无效 | A；相同tokenizer/模板/截断，不能改变benchmark或多开资源；[官方AsyncLLM][SR06] |
| S09 / P2 | detokenizer/SSE输出复用请求状态及增量字符串片段，减少重复token列表/完整字符串拼接 | V1已有增量输出；只查残余复制，非重新发明streaming | 记录每token输出CPU成本和排队；UTF-8跨token、stop string跨片段、取消、logprobs全测；不延迟首token聚包 | A；保持可见输出和停止语义；[官方输出处理代码][SR07] |
| S10 / P2 | 热统计与诊断计数改成预分配累加器，批量快照；减少每层对象/字符串/flush，保留默认可观测字段 | 旧K4/CPU方向深化；不能只关日志开关提速；已一次性marker不用再优化 | profile确认日志/GC对关键路径的比例；相同外部统计含义，比较CPU节流和p99 gap | A；不删除错误/指标，也不把cgroup throttled_time当wall损失；[vLLM V1][SR05] |

### 编译、Graph与生命周期（8项）

| ID | 具体实现增量及机制 | 当前/旧项关系 | 最小证伪实验、适用指标 | 边界/参考 |
|---|---|---|---|---|
| S11 / P0 | 审计自研custom-op mutation/alias/FakeTensor合同，消除functionalization引入的多余clone/copy，显式out返回复用合法storage | H07深化；PR33 norm+quant和native op最相关，不等于换compile开关 | 比较真实FX/AOT图中clone/copy、storage liveness与峰值；CPU schema门+变输入Graph/alias压力门 | A；不得谎报mutates_args或跨live值覆写；[vLLM IR][SR08] |
| S12 / P1 | 按算子活跃区间规划统一临时arena，串行层/不并发graph共享切片，按最大真实需求分配 | 旧H07；已有WorkspaceManager和一builder scratch，不可再乘42；只处理未被其管理的native/采样scratch | 输出unique storage及同时live字节曲线；先找到可复用重叠，然后poison/交替shape/replay | A；并发stream/异步输出不能错误共用；[PyTorch Graph][SR09]、[MLC arena pass](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/lift_global_buffer_alloc.py) |
| S13 / P2 | 实现CoreX上真正的弱引用tensor句柄，或受liveness证明的graph-output持有策略 | 队友已有identity fallback诊断；不是新发现，现已知finalhidden规模很小 | 先证明额外strong ref确实阻碍pool复用且字节值得；否则不写ABI；多图交替及延迟sampler读验证 | A/C；Python弱引用替换不等价；[PyTorch Graph][SR09]、[旧内存账本][SL03] |
| S14 / P2 | 分离graph永久池、首次lazy runtime分配与真实activation峰值，校正被证实的重复/错误预算 | main有估计器但Iluvatar guard通常跳过；上游GLM bug仅线索，不是本机bug | 一次stage账本+实际is_cuda/estimator选中；没有预算误计证据即停止，不能关估计或少留安全空间刷容量 | A；固定0.85，所有合法大shape仍需安全；[上游报告][SR10]、[本地worker][SL04] |
| S15 / P1 | 将同一runtime真实必需的lazy kernel工作区显式准备并稳定复用，使profile与正式路径的内存所有权一致 | 不能只把分配移出profile藏峰值；要真正降低重复副本/同时live峰值 | 对load→profile→KV→capture→sampler逐阶段storage归因；若只是计账时刻变化不记收益 | A；旧“清cache”与PR35已完成不重报；[MRV2 dummy设计][SR01]、[本地profile][SL01] |
| S16 / P1 | 保持固定Graph尺寸，但在自研kernel内使用device实际active row数屏蔽整块padding计算/访存 | H10深化，区别于改capture_sizes或Graph模式；已mask的attention不重复 | M29→32、M60–63→64等真实padding几何，记录省掉的tile与额外load；整图wall无下降即否决；M≤16残差补偿行不能跳过，原生_int_mm的固定行计算不会因外层mask自动减少 | A；kernel输出padding/alias合同保持；不得按4K/16K标签分支；[Graph设计][SR11] |
| S17 / P1 | 编译签名只specialize真正影响生成代码的维度；长度/索引等设runtime scalar，stride/layout分别守卫，减少cold及cache碎片；另一个备选是将小M分发下沉至有界已编译入口，分别评价资源合并与分发成本 | H09/H10深化；不能把cold-only收益当暖态吞吐；PR35语义身份保留 | 从现cache提取重复等价二进制/编译次数；真实shape trace重放，验证算子选择未变且暖态无回退 | A；不合并上游MRV2；[Triton JIT接口][SR12]、[MLC small-batch specialization](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/low_batch_specialization.py)；单入口特化须同时检查reg/spill |
| S18 / P1 | 处理视图/contiguous转换的producer-consumer约定：producer直接生成消费者需要的stride，并保持view直通，减少编译图转置副本 | H07、E10与融合方向深化，需全链路而非微核 | 真实QKV/量化输入/最终hidden图中的copy与transpose字节；每条边分别测收益，检查相邻消费者 | A；KV页布局更改须同时覆盖writer、copy、restore；[vLLM IR][SR08] |

### 容量、KV与数据迁移（10项）

| ID | 具体实现增量及机制 | 当前/旧项关系 | 最小证伪实验、适用指标 | 边界/参考 |
|---|---|---|---|---|
| S19 / P0 | 对LM head/native INT32做有界tile输出与即时epilogue，避免大M完整INT32峰值；实际profile也使用同一实现 | 旧G01/G08/F09深化，历史head量化候选已释放原BF16副本，不能再归因双份权重；当前生产BF16 head另行区分；这是容量目标而非只快GEMM | 实测峰值、合法profile全部路径、KV tokens；先比较短16Kwaiting轨迹，再两档吞吐/TTFT | A；保持BF16舍入/采样语义；[本地profile][SL01]、[旧内存账本][SL03] |
| S20 / P1 | 对量化权重、原权重view、module closure、compiled constant做unique-storage所有权审计，删掉确无fallback消费者的孤儿副本 | 历史head量化候选已查释放原BF16副本，不重复；新混合精度/多布局候选可能另有副本 | 加载后→capture后列storage ref归属；若fallback仍需要则不能删；至少省下可解释字节才上服务 | A；不是以评测固定shape断言永不用BF16；[PyTorch Graph][SR09] |
| S21 / P1 | 对sampling penalty masks/history按真实请求需求延迟建立；索引或稀疏状态代替无条件max_rows×vocab布尔/FP32展开 | 旧G06深化；默认top-p可能根本不走penalty，先审计 | 找到实际常驻/同时live对象再实现；测试原本启用的penalty/logprobs模式，记录profile峰值和复杂度 | A；不能省略用户要求的logprobs/penalty；[MRV2 sampler设计][SR01] |
| S22 / P1 | 针对普通运行的新页分配实现extent亲和：保留逻辑页语义，在同等可驱逐集合中提高连续物理段比例 | restore私有页排序已做；本项为持续增长/普通页分配，不是重复PR3/4 copy | 先CPU回放真实freequeue/hash/refcount轨迹，量extent长度与额外eviction；GPU看attention locality和copy总成本 | B；共享/APC命中不破坏，不能以固定benchmark请求名挑页；[PagedAttention][SR13] |
| S23 / P1 | 在已有SAVE/FULL协议上做多次抢占的增量快照：已保存且未修改的完整KV页不再D2H，只传新页/尾页 | 本轮未见完整增量快照验收；已有单victim/full保存不能再列“从零做offload” | 先查同request重复preemption是否足够多；无复用即无收益；页版本/epoch、取消、修改后再抢占全byte门 | B；不能保存未完成GPU写入或误把同pageID视作同内容；[FastServe][SR14]、[HiCache][SR15] |
| S24 / P1 | 原APC存活前缀 + host缺失后缀联合恢复，减少H2D字节，完整R逻辑保持 | 队友已有条件方案surviving-prefix，尚需当前组合真实H证明；不是新概念 | 只读记录真实hash/refcount/provenance，H=0即停止；H>0后分别验证本地段与恢复段的完成证据 | B；ref0命中并不减少总free页需求，不能吹成免容量等待；[APC设计][SR16]、[旧方案][SL05] |
| S25 / P1 | SAVE/RESTORE API提交批处理进一步复用descriptor、C ABI及event池；只处理真实热点，保留全部字节 | PR3/4和实验pitched H2D已有；候选增量是调用方准备/生命周期成本，非再造2Dcopy | 真实42层事务准备+submit+completion成本分解；已有copycalls≠DMA次数≠时延；无host开销则停止 | A/B；提交非ACK，异常先settle；[团队copy说明][SL06]、[PyTorch异步复制][SR04] |
| S26 / P2 | 分层/分组恢复流水：提前恢复后继层KV，当前层计算与后一组H2D重叠，避免整42层恢复完成才前进 | 旧FULL有完整事务，不是直接改成部分ACK；需新per-layer fence设计，16SM/PCIe竞争未知 | 一个真实恢复窗口含完整依赖，比较copy+compute关键路径，先确认H2D硬件重叠能力；host≤20GiB | B/C；Graph捕获与静态地址、取消、layer readiness、共享页均需证明；[NanoFlow][SR17]、[HiCache][SR15] |
| S27 / P2 | UVA供GPU读取大prefill token-ID/极小只读metadata，减少复制和GPU常驻缓冲 | MRV2已有思路但BI-V150未证支持；当前 `_C::get_cuda_view_from_cpu_tensor` identity fallback不是UVA实现 | 先SDK能力与pinned指针访问门；顺序4K/16K token读取对照H2D，不支持或PCIe散读更慢即止 | C；不把NVIDIA CUDA功能假设为CoreX；[MRV2][SR01] |
| S28 / P2 | VMM固定虚拟地址下的弹性activation/KV物理页借用，按实际phase归还reserve；不改gpu_memory_utilization | 旧memory-lifecycle和Elastic KV已有研究，尚无CoreX VMM/可借空间证据；高成本储备 | 先API能力门与真实可借liveness；无数百MiB以上可信空间不实现；最坏shape不OOM后才谈容量 | B/C；禁止偷用预算外0.15或按benchmark裁合法shape；[vAttention][SR18]、[Elastic KV][SR19] |

### 调度与执行重叠（8项）

| ID | 具体实现增量及机制 | 当前/旧项关系 | 最小证伪实验、适用指标 | 边界/参考 |
|---|---|---|---|---|
| S29 / P1 | **重算/恢复成本决策器**：实际R/APC幸存页/copy成本/队列等待预测决定是否用已有SAVE/FULL；先shadow决策 | 旧offload正式轮存在，无新因果收益；当前PR24 VERIFY1队友正在推进，不能另起重复队列 | 当前新组合最短shadow记录：可省target kernel时间 vs全事务成本；无法区分/可省极小则回原路径 | B；只能用运行时信息，不能用105题答案/评测标签；[INFERMAX][SR20]、[FastServe][SR14] |
| S30 / P2 | 有限滚动窗口准入+TTFT约束：只考虑到最近可证明释放点的增长，不为所有1024输出满额预留；需与已有finite-growth比较 | 已有full-growth负例和finite-growth混杂，不算新启动；应先修正模型而非反复扫水位 | CPU回放实际token/page轨迹，报告空耗预算、预占、最老waiting；能解释旧TTFT退化才试一个新策略 | B；2048/maxseq/FCFS参数不改，所有请求有进展；[Sarathi][SR21]、[INFERMAX][SR20] |
| S31 / P2 | 成本感知victim：同等调度权限下比较真实丢失计算、可保APC、恢复页数/传输段数，选最少净损失 | 队友progress-victim/调度研究已存在，不能称未探索；新成本函数需真实GPU窗口支撑 | shadow记录原victim与候选代价差；收益不足或破坏最老请求服务界限即不进入实现 | B；不能按请求名称或答案长度oracle选择；优先保持已有fairness约束；[INFERMAX][SR20]、[FastServe][SR14] |
| S32 / P2 | 固定2048总cap的成本模型chunk分配：只在多prefill竞争时优化真实GEMM M及attention tile浪费，并受TTFT限制 | 旧BQ16中间chunk向下对齐已有反证：launch可增、shape更多、有效省量很小；不重扫此方案 | 离线实际query tile覆盖/新增forward/总M tile，不能把chunk数当KV重复读取倍数；模型无优势即止 | B；不是修改budget/default，不能强行制造固定评测形状；[Sarathi][SR21]、[固定预算旧报告][SL07] |
| S33 / P2 | 在设备kernel资源互补成立时做attention与另一nano-batch的GEMM重叠，整步统一join，区别于旧两attention抢HBM | 旧E4/NanoFlow储备深化；旧decode/prefill attention双流无净证明 | 两个真实kernel单独/并发的完整wall与资源占用；16SM吞吐损失大于重叠即止，不迁移NVIDIA倍数 | B/C；不改官方ubatch开关，以自研算子/依赖图实现；[NanoFlow][SR17] |
| S34 / P2 | 既有mixed双attention重叠若保留，先把每层Event构造/strong-ref积累改为整step复用的严格生命周期实现 | 原型93c已正确但wall/event冲突，不能写“新方案”；先算42层retention峰值与提交成本 | 仅在旧成本分解说明host开销遮住潜力时再做单一ABBA；否则归档不占卡 | A；readiness与record_stream不同；capture继续旧路径；[PyTorch streams][SR09]、[旧重叠报告][SL08] |
| S35 / P2 | 调度准备与output处理分阶段，先发布GPU下一步所需最小整数状态，非关键JSON/统计延后；保持现有单步async深度 | async已开启，不是加深queue开关；新价值只能来自当前CPU关键依赖缩短 | profile output→schedule→launch的真正依赖，最晚状态消费点；没有GPU bubble即不做 | A/B；EOS/stop/max_tokens/cancel不能延迟生效，多余token不能算分；[SGLang v0.4][SR22]、[官方固定源码][SR23] |
| S36 / P2 | 对频繁请求结束建立GPU完成事件驱动的精确页回收/元数据清理，避免无关整step CPU处理延长页占用 | 旧terminal-drain无可释放终态已经否定；本项仅新证据显示“已真实完成却等待CPU”的情况 | 时间戳证明finish GPU event→真实free间存在容量关键延迟；无此证据即不实现 | A/B；不能提前free在途输出/共享页，不改变token终态判定；[ORCA][SR24]、[PagedAttention][SR13] |

## 8. 论文与跨框架机制：17 张卡

本节 R=保持语义的实现，Q=自主量化，S=固定参数下的系统开发，C=额外规则或能力前提未明。L/C/D/P 分别标本地证据、公开源码、设计文档、论文；它们不是 BI-V150 实测。P26/P30 是评估/测量支撑方法，其余 15 张是性能候选。学习旋转或仿射参数的路线保留边界，不将训练原模型权重默认归为已获准量化。


### 量化：从“更低bit”转向真实硬件成本与精度恢复

| ID / 旧ID关系 | 机制及本项目落点 | BI依赖、证据/规则 | 最小实验与优先级 |
|---|---|---|---|
| P01 / 深化F07，区别普通W4A16 | **QoQ渐进W4→INT8**：权重先形成有保护范围的INT8，再组内压成INT4；内核解包回INT8进入已有INT8矩阵指令，避免K循环里对INT32 partial不断浮点反量化。先保留现有W8A8激活路径，只替换非敏感FFN权重的存储/解包。 | 依赖BI的INT8吞吐、packed整数指令和寄存器成本；[QServe论文](https://arxiv.org/html/2405.04532v1) P/Q。 | 单个真实gate/up或down形状，测读权重+解包+GEMM+epilogue完整时间及常驻内存；M16/32/64和prefill各一个，不先全模型。P1。 |
| P02 / 深化F08、F02 | **DuQuant式通道置换与局部旋转**：对难量化通道先局部旋转，再跨组置换平衡组间离群分布；比只做一次全局Hadamard多一个可控自由度。对FFN中间通道，gate/up同步置换、down逆置换可离线吸收；旋转不能随意越过SiLU。 | 置换可无额外在线算子；局部旋转仍有成本；[DuQuant](https://arxiv.org/abs/2406.01721) P/Q。 | 固定bit和内核，只比较原分组与合法置换分组的独立校准误差；先数学验证置换等价，再测包装布局。P1。 |
| P03 / 深化F08，不是重跑随机种子挑成绩 | **学习可离线吸收的旋转**：SpinQuant指出随机旋转质量差异大；用独立校准集优化允许的旋转参数，尽量选能吸收到相邻权重的部分，减少实时变换。 | BI不要求特殊低bit硬件，但变换若在线就可能抵消收益；[SpinQuant](https://arxiv.org/abs/2405.16406) P/Q+C。 | 先1个block的BF16等价和W8/4误差对比；记录校准集与验证集分离。主办方对学习变换口径若不明，只留离线研究。P2。 |
| P04 / 新变换族；关联F08 | **FlatQuant可分解仿射变换**：用每层适配的Kronecker分解仿射变换压平激活和权重分布，运行时与quant融合；可用于量化困难层的精度恢复，而非盲目把所有层改W4。 | 需要BI小型变换核与quant共同计时；[FlatQuant](https://arxiv.org/abs/2410.09426)、[作者仓库](https://github.com/ruikangliu/FlatQuant) P+D/Q+C。 | 在99/105路线确有吞吐余量后，对定位到的1–2个敏感投影比较“BF16回退”和“变换+量化”的误差/时间。PR26已补全两档且无新余量并关闭；本项仅适用于未来有性能余量的新版本，暂P2。 |
| P05 / 新权重码本；关联F07/F08 | **HIGGS/FLUTE非均匀权重量化**：Hadamard之后用MSE优化的非均匀格点或小向量码本，而非均匀整数间隔；代码本的解码必须与矩阵核联合。 | FLUTE现有CUDA形状和GPU特化不能直接用于BI；需要自研查表/解包；[HIGGS论文](https://arxiv.org/abs/2411.17525)、[FLUTE作者仓库](https://github.com/HanGuo97/flute) P+D/Q。 | 先CPU比较相同总bit(含scale/codebook)的误差，再仅做一个M32 W4A16核；若解码+Hadamard成本不合算停止。P2。 |
| P06 / 新权重向量量化；旧E07只针对KV | **QuIP#格点码本**：以E8等结构码本联合表示多个权重，可作为2–4bit权重的长期备选；不能把论文微调阶段默认带入比赛。 | 不依赖原生INT4，但依赖高效码本解码和BF16/GEMM；[QuIP#](https://arxiv.org/abs/2402.04396) P/Q+C。 | 只做不微调原模型的校准版本，先验证存储节省和一层重建；若必须靠权重微调才能过门槛，则暂不落地。P3。 |
| P07 / 新编码族；区别P06 | **QTIP trellis编码权重**：用可bitshift解码的trellis而非指数膨胀的高维码本；理论上能同时保持高维量化质量和小解码状态。 | 随机tile解码、warp64映射与寄存器压力未知；[QTIP](https://arxiv.org/abs/2406.11235) P/Q。 | 单个权重tile的独立解码测试，再测M1/32/64完整矩阵核；解码带宽若低于普通W4立即降级。P3。 |
| P08 / 深化F05/F06，新增证据方法 | **重要性加权校准而非仅weight MSE**：参考llama.cpp imatrix，用独立输入激活统计为权重误差加权；对当前易失分的层/通道分配精度预算。 | 校准可离线做；不引入GGUF运行时；[llama.cpp imatrix官方说明](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) C+D/Q。 | 同一层比较weight-MSE排序与activation-weighted排序能否预测实际block/logit误差；不使用math_500测试题来拟合。P1。 |
| P09 / 深化F05/E09，新增搜索方法 | **把逐层精度选择做成有预算的优化**：HIGGS给出按层量化级别的动态规划思路；本项目可把目标改成“实测时间+KV容量代价”，选少量BF16回退层，而非凭经验整段回退。 | 候选每层成本需要本平台实测；误差可加模型只是近似，不能替代整体验收；[HIGGS](https://arxiv.org/abs/2411.17525) P/Q。 | 采3个候选精度×已定位敏感层，建立Pareto表；保留99/105候选诊断资格，但无性能余量不强救。P1。 |
| P10 / 深化E01/F01；注意RoPE约束 | **SmoothAttention把K量化难度转给Q**：成对缩放Q/K保持点积，减弱K的通道离群；对RoPE维度须满足旋转对的缩放约束，不能任意每维缩放后照旧RoPE。 | Q BF16路径、KV量化与RoPE形式共同决定；[QServe SmoothAttention](https://arxiv.org/html/2405.04532v1) P/Q。 | 单层真实K/Q，先验证BF16代数等价，再INT8/INT4 K误差及QK+PV整核；与旧“KV8直接反量化”是不同结构。P1。 |

### 编译与融合：跨算子看内存生命周期和任务依赖

| ID / 旧ID关系 | 机制及本项目落点 | BI依赖、证据/规则 | 最小实验与优先级 |
|---|---|---|---|
| P16 / 深化B01–B03/H06；新增搜索空间 | **Mirage式联合算法/布局/融合边界搜索**：B01两个已试结构失败后，搜索partial fusion、不同中间布局和reduction分解，而非同结构再扫warps。 | Mirage CUDA后端不能直接跑BI；借其多层图方法，以BI成本剪枝；[Mirage OSDI官方论文页](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) P/R。 | 从gate/up→BF16 round→SiLU→down局部图列3–5种真正不同结构；必须把原BF16舍入点显式建模，实数等价不等于浮点等价。P1。 |
| P17 / 新框架内子图执行方式；关联A07/B03 | **单解码步局部megakernel**：借MPK worker/task队列把一段层或FFN子图放在持久kernel，不跨请求改变采样；减少launch和global同步，先局部而非整模型。 | BI跨block acquire/release、活跃block驻留保证、动态共享内存/warp64；[MPK论文](https://arxiv.org/abs/2512.22219)、[运行时源码](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/persistent_kernel.cuh) P+C/R。 | 先两个核的persistent任务壳对比现有FULL graph，含调度占SM成本；若没有launch gap则停止，不宣称GPU graph未解决的问题都存在。P2。 |
| P18 / 新细粒度依赖；深化B08/C10 | **QKV→attention按tile流水**：特定请求/head的QKV tile齐备即可开始该attention任务，无需等待整个QKV kernel全局结束；与把两个完整核简单顺序融合不同。 | BI内存可见性、队列粒度、GQA8共享与KVstore先后关系；[MPK](https://arxiv.org/abs/2512.22219) P；具体MiniCPM切分为推演/R。 | 合成单层，比较QKV+attention总时间、最长尾部和额外flag流量；保留FP32累计/掩码语义。P2。 |

### 内存与调度：固定参数下，减少浪费和实际抢占

| ID / 旧ID关系 | 机制及本项目落点 | BI依赖、证据/规则 | 最小实验与优先级 |
|---|---|---|---|
| P22 / 新更细KV分配；区别仅调page size | **仅尾页采用更细粒度子分配**：借TokenAttention思路减少每个活跃请求尾页碎片，完整页仍保留现有快路径；比全量token表更小改动。 | 每token本模型BF16 KV约42KiB，必须先算尾碎片上界；[LightLLM](https://lightllm-en.readthedocs.io/en/latest/framework/token_attention.html) D；混合尾页方案为推演/S。 | 先统计峰值活跃请求×尾空槽。block16时，按64个驻留请求、每请求最多15个空槽，理论尾碎片上界仅39.375MiB；实际约30驻留时更小。只在页级门槛确有影响时研究；核需验证tail地址和graph更新。P3。 |
| P26 / 新全局资源预算方法；关联H07/E09/F05 | **显存–时间联合规划**：借FlexGen的约束优化，不移植CPU/disk offload；在固定.85显存预算下比较常驻权重格式、不同scratch、KV容量、抢占概率，避免“单核快但KV池缩水”重演。 | 需要BI真实字节与调度数据；[FlexGen](https://arxiv.org/abs/2303.06865) P；本项目目标函数为推演/R+Q+S。 | CPU表格/枚举即可先做：候选真实常驻bytes、额外scratch peak、可容纳KV token、完整组件时间；筛掉明显劣解。P0。 |
| P28 / 深化G05，新增精确selection方法 | **Guess–Verify–Refine的exact Top-K**：当前行抽样估阈值，一次遍历多阈值计数，再在边界桶精确选择，坏猜测回退；不能拿DeepSeek稀疏indexer直接给本模型加稀疏attention。可迁移到确实启用top-k的采样分支。 | 官方吞吐若top_k不生效则不加速；ties/NaN/inf/RNG需匹配；[TensorRT GVR V2](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog29_GVR_V2_Self_Sampling_Exact_TopK_for_Sparse_Attention.html) D/R。 | 先核实评分和精度两路采样参数；只测真实触发形状，完整worst-case fallback计时；top-p需另证明质量累计逻辑。P2/P3。 |
| P30 / 新测量驱动开发手段；深化H04/H05/H06 | **用编译器内探针区分核内瓶颈**：KPerfIR提供在编译IR层观测load、layout转换、矩阵指令等待/重叠的思路；用来决定B01失败究竟是spill、layout还是pipeline问题，而非再次盲扫。 | 需BI Triton/FlagTree lowering可插入低开销计时，不能直接使用CUDA结论；[KPerfIR OSDI议程原始摘要](https://www.usenix.org/conference/osdi25/technical-sessions) P；这是优化支撑，不是直接吞吐收益/R。 | 先选1个已失败核与1个基线，dump编译IR/寄存器/spill；仅在离线profile版本插探针，正式性能仍测无探针版本。P0/P1。 |


## 9. 比赛、社区与进一步阅读

### 竞赛与社区材料怎样使用

| 入口 | 本项目可借鉴的内容 |
| --- | --- |
| [MLSys 2026 FlashInfer](https://mlsys26.flashinfer.ai/)与[获奖 writeups](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups) | 按真实 shape 分族、联合考虑重排/融合/资源成本。其 MoE、DSA、GDN 题目不是 MiniCPM 算子；上一轮已读 Dogacel、Team Wombat、UW SyFI 报告，继续作为结构搜索参考。 |
| [AMD GPU MODE E2E Speedrun](https://www.amd.com/en/developer/resources/technical-articles/2026/new-gpumode-virtual-hackathon--e2e-model-speedrun.html) | 强调组合后的整模型表现；单核排名与服务吞吐分开。AMD 特定矩阵指令、量化格式与硬件吞吐不能外推。 |
| [昇腾 S7 官方入口](https://www.hiascend.com/developer/contests/details/769391f54a104a349b3d1b1e72035388) | 追踪融合、分块、流水结构及公开实现；只借数据流，不借 Cube/Vector 指令或该赛题的精度容差。 |
| [KernelSwift 官方入口](https://deeplink.org.cn/kernelswift_competition) | 国内异构算子优化资料检索入口；本轮正文抓取为空，不据此声称新的榜单/获奖方案。 |
| [FlagOS KernelGenBench](https://github.com/flagos-ai/KernelGenBench) | 有 vLLM/ATen/矩阵计算用例和多芯片评估接口，可借形状覆盖及独立验证方式；不替代本赛题整模型验收，也不安装新版本替换比赛栈。 |


1. [QServe / OmniServe作者论文](https://arxiv.org/html/2405.04532v1)：最值得借的是“量化格式与实际解码指令共同设计”，而非只记W4A8KV4标签。对应P01、P10；论文硬件倍率不得迁移。
2. [MLC LLM compiler_pass源码目录](https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_llm/compiler_pass)：本轮已下钻三个分配/特化文件，支持S12/S17/S18的深化；能具体检查现框架遗漏的pass，而不是建议换TVM。
3. [LightLLM TokenAttention](https://lightllm-en.readthedocs.io/en/latest/framework/token_attention.html)及[Efficient Router](https://lightllm-en.readthedocs.io/en/latest/framework/router.html)：连续优先分配、细粒度尾碎片、预测峰值准入，适合先离线模拟。
4. [TurboMind架构原文](https://github.com/InternLM/lmdeploy/blob/main/docs/en/inference/turbomind.md)：persistent batch和KV manager；注意文档包含历史实现，必须审查当前插件是否已有等价机制。
5. [Mirage OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)和[MPK OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/cheng)：一个侧重等价程序/布局搜索，一个侧重SM任务图/巨核；不要混为“换编译器即提速”。
6. [NanoFlow作者论文](https://arxiv.org/abs/2408.12757)：为不同算子资源互补提供系统方法；单卡只借本地子批与重叠，多卡通信优化无适用性。
7. [HIGGS作者论文](https://arxiv.org/abs/2411.17525)与[FLUTE源码](https://github.com/HanGuo97/flute)：小码本权重量化和按层预算规划；有新格式也有硬件特化限制。
8. [TensorRT冷页codec设计](https://nvidia.github.io/TensorRT-LLM/developer-guide/kv-cache-cold-page-codec.html)：最新设计把索引、布局、生命周期与批量迁移边界写得很清楚；可借契约，不应主动给2B模型加磁盘offload。
9. [AMD官方2026 GPU MODE E2E Speedrun](https://www.amd.com/en/developer/resources/technical-articles/2026/new-gpumode-virtual-hackathon--e2e-model-speedrun.html)：从kernel赛到端到端赛的对照材料；其MLA/MoE/MXFP4硬件路线不能直接移植dense GQA MiniCPM/BI。
10. [昇腾算子挑战赛S7官方规则](https://www.hiascend.com/developer/contests/details/769391f54a104a349b3d1b1e72035388)：作为国内算子赛题与开放实现的检索入口；可跟踪融合/流水结构，不能套AscendC/Cube指令到BI。

补充长期编码资料：[DuQuant](https://arxiv.org/abs/2406.01721)、[SpinQuant](https://arxiv.org/abs/2405.16406)、[FlatQuant](https://arxiv.org/abs/2410.09426)、[QuIP#](https://arxiv.org/abs/2402.04396)、[QTIP](https://arxiv.org/abs/2406.11235)、[llama.cpp imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)。它们互为备选，不建议同时全量移植。

### 外部同赛题负经验


补读外部同赛题仓库快照中的 `docs/baselines/iluvatar-biv150/a8-a9-negative-results.md`。主代理确认该repo remote是 `xy200303/minicpm-flagserve`，不是已确认队友仓库。本轮只读取，不背书其中规则口径和历史硬件假设。该快照对应本地 commit `045b064`，公开仓库入口为 [minicpm-flagserve](https://github.com/xy200303/minicpm-flagserve)；尚未独立核实其远端最新版。

1. 文档的A8a提醒：混合步host开销即便看起来51ms/step，仍可能被250–300ms GPU阶段掩盖，且混合步占比低。对S35必须做关键路径上界，不把host函数累计时间直接算可得收益。
2. A8b的lm_head配置扫参输给vendor只说明那个版本/形状的普通tile扫描收益弱；其M64 K2048 N130560 BF16权重流量约535MB，提示G01应该先算有效带宽。不能据此否定确定性splitK、精确head+selection融合等不同结构。
3. A9正文先用文本re-tokenize探针否定ngram，后来明确修正：真实token draft接受率100%，但实际吞吐仍亏在eager/graph/qlen执行成本。可迁移的教训是“接受率和理论token跳步不能代替整条路径测量”；不能引用旧结论说随机负载不适合投机。
4. 文档最后引用64GB评测机、权重带宽占步长3%的推断与本队单32GiB状态不同，不能沿用它排除KV/权重量化。本队历史G08已经显示显存和KV容量会改变16K/TTFT。
5. 这些记录的serve/spec参数和其自述规则不替代本队官方规则；本轮没有推荐直接启用投机开关。

### 新增论文的反例也要读

[Elastic KV Cache](https://arxiv.org/abs/2608.23658)实现了弹性借用，却报告在其测试条件下收益前提未成立。对本项目的启发是先量出可借 reserve、切换成本和实际容量压力，不能由“能动态分配”推出“应当更快”。不照搬其调整 chunk 参数的实验设置。

[KVPR](https://arxiv.org/abs/2411.17089)用部分 activation 重算 KV 并与传输重叠。但本模型 GQA 的每层 K+V 只有 512 个 BF16 元素，输入 activation 有 2048 个：若直接保存完整输入供重算，字节量反而是 KV 的 4 倍。这是依据模型几何的筛选推断；除非能复用本来就存在的 activation 或提出不同表示，不优先将该机制搬进本机。

[vLLM Triton Attention 深入分析](https://vllm-project.github.io/2026/03/04/vllm-triton-backend-deep-dive.html)具体讨论了固定图网格的空工作与 persistent GPU 工作分配，可继续支持旧 C01/D08；不能把该文 Llama/H100/MI300 结果作为 BI-V150 收益。

### 去重映射

为保留研究来源而不重复计数，本轮合并：AD04→S11，AD06→S06，AD15→S16，AD20→S12；P11/P12→S12–S14，P13→S16，P14→S17，P15→S18，P19→S33，P20→S35，P21→S22，P23→S01/S02，P24→S30，P25→S25/S26，P27→S03–S05，P29→S32。因此编号有空缺，属于有意去重。其他有关联的卡片按不同输入/消费者、实现层次或可证伪条件保留，收益仍不可相加。

## 10. 如何把候选变成有效实验

先维护一份轻量队列，而不是同时写几十个原型。每张实验卡至少包含：目标实际调用、生产基线 commit、候选 commit、对应旧条目/队友实验、改变的数据流、首次可证伪条件、完整成本、精度与容量风险、原始结果位置。已有 PR33/36/37 和队友 KV 后继继续各自推进，新的研究卡不改变它们的设备安排。

1. **无卡阶段**：核对 shape/stride/dtype、编译与 dispatch、内存和同步依赖；推导额外字节/工作量。优先找能直接排除的方案，例如依赖 BI-V150 没有的硬件指令，或收益上界小于额外复制。
2. **第一设备门**：允许先用合成输入做编译/索引冒烟；进入组件收益判断前，用真实形状和真实权重检验完整组件，不只计核心 kernel。不同输入多次图回放，观察实际选择及回退；新增 packing、建表、归约、事件、临时空间均计入。
3. **模型诊断**：固定代码与环境，记录 KV tokens、实际图/shape 分布、请求成功数、token 总量、TTFT；用相称的真实时间证据判断覆盖。频度不能代替时间，占用率不能代替效率。
4. **近门槛候选**：105 题至少 100 题正确才达 0.95。若 99/105，保留失败原件，按用户要求先测吞吐潜力，明确 `precision_gate=false`、不可合并；有可用性能余量才探索按层/形状高精度回退。PR26 已完整执行这一流程并因无新余量关闭，不为挑出一次通过而重复同版精度。
5. **正式接入**：确认两档吞吐/TTFT和精度条件；正式报告保持官方四轮、丢首轮的口径。研发充分预热单轮、短负载、profiler 窗口都单独标识。同步最新团队分支，独立 PR，满足相应验收后合入。

测量前尤其核对：受测源码/原生 SO/实际导入路径/编译图身份、tuning DB、可用 KV 容量、冷热条件以及是否有其他 GPU 负载。它们会改变对照结论；不通过改服务参数、占位伪造正式对照或反复重启挑容量来凑成绩。若为隔离容量影响做有界诊断控制，须另行标注，不能作为正式验收成绩。

### 不应进入本轮实现队列的捷径

| 类别 | 本项目的处理 |
| --- | --- |
| 改 `max_num_batched_tokens`、`max_num_seqs`、显存比例、graph mode、客户端并发、输出长度或 benchmark | 官方参数固定；在代码里硬改同一个默认值也不包装成算法创新。 |
| 直接 merge 最新 vLLM/插件/FlagGems，以新版本替换比赛基线 | 仅阅读机制，自主实现独立改动，保持比赛版本边界和主要算子选择逻辑。 |
| 使用其他模型、draft 模型、开启现成 speculative decoding 开关 | 不作为合规候选。自投机、无草稿模型投机等也不能仅凭“没换模型”推定允许；列规则待确认区，不据此开发参赛路径。 |
| 丢 KV 页、裁剪上下文、跳层/early exit、删词表、缩短输出、改采样分布 | 这些可能使模型核心能力明显下降，或使评测结果失去可比性；即使过一次精度也不自动合规。稀疏/近似 attention 与自主量化不能混为同一规则。 |
| 识别 benchmark 名称/题号/轮次、预存答案、优先缓存已知评测 prompt | 排除。通用前缀缓存只对真实相同输入复用计算，不能预装评测数据或按轮次定向保留。 |
| 异平台原生 TMA/WGMMA、warp 大小、barrier、VMM/Graph API 的照搬 | 先证明 CoreX/BI-V150 的对应能力；不可用时保留机制研究或选择其他实现，不宣称可直接移植。 |
| 切换框架、借多 GPU 的 PD 分离/通信收益 | 当前单 BI-V150 不适用。可以借鉴单卡任务规划、布局和缓冲区设计，最终实现仍在规定框架内。 |

规则依据为用户提供的《赛道二_官方规则原文.txt》及《赛道二_MiniCPM_赛题要求与优化目标.md》。[官方赛道入口](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)本轮网页工具未能读取正文，因此不声称已核实新增官方解释。明确允许的是有实际开发量的框架/算子/显存优化及自主量化；表中的未知规则边界不计为可直接实施项目。

### 两个防止错误排序的计算

若某段占当前总时间比例为 `p`，自身加速为 `s`，其余成本不变，理想整体加速上界为 `1 / (1 - p + p/s)`。若只占 1%，即使完全消除也仅约 1.01×；另一方面，节省显存减少重算会改变总工作量，不能只用固定工作量公式否定其价值。

对于两段尝试多流重叠，理想最多省 `min(T1,T2)`，还要扣同步、并发资源竞争和新增缓冲成本。在 16 SM 的 BI-V150 上，不同 stream 不代表实际并发，更不代表更快。

这些是筛选模型，不是已测收益预测。长期方向可以保留，但应由新证据改变优先级，避免只因名称新、论文倍率大或一次微测正就投入整套服务开发。

## 11. 系统部分来源说明


下面外部资料只支持机制；候选落地与优先级为本次根据本地代码/证据作的推断。上游issue是作者报告，不能推定BI-V150也有同一bug；主线文档可能变化，实际实施必须固定版本/commit。

[SR01]: https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md
[SR02]: https://github.com/vllm-project/vllm/issues/61621
[SR03]: https://github.com/vllm-project/vllm/issues/61239
[SR04]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
[SR05]: https://vllm.ai/blog/2025-01-27-v1-alpha-release
[SR06]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py
[SR07]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/output_processor.py
[SR08]: https://github.com/vllm-project/vllm/blob/main/docs/design/vllm_ir.md
[SR09]: https://docs.pytorch.org/docs/2.10/notes/cuda.html
[SR10]: https://github.com/vllm-project/vllm/issues/59318
[SR11]: https://docs.vllm.ai/en/latest/design/cuda_graphs.html
[SR12]: https://triton-lang.org/main/python-api/generated/triton.jit.html
[SR13]: https://arxiv.org/abs/2309.06180
[SR14]: https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang
[SR15]: https://github.com/sgl-project/sglang/blob/main/benchmark/hicache/README.md
[SR16]: https://docs.vllm.ai/en/v0.24.0/design/prefix_caching/
[SR17]: https://arxiv.org/abs/2408.12757
[SR18]: https://arxiv.org/abs/2405.04437
[SR19]: https://arxiv.org/html/2608.23658v1
[SR20]: https://arxiv.org/abs/2411.07447
[SR21]: https://arxiv.org/abs/2403.02310
[SR22]: https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
[SR23]: https://github.com/sgl-project/sglang/blob/85e1a6f3aa5a2288ca85fe3fe922c733b6533fa7/python/sglang/srt/managers/scheduler.py
[SR24]: https://www.usenix.org/conference/osdi22/presentation/yu
[SL01]: https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/worker/model_runner.py
[SL02]: https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/patches/biv150_prefill_normalized.py
[SL03]: https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-21-memory-lifecycle-feasibility.md
[SL04]: https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/vllm_fl/worker/worker.py
[SL05]: https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-22-kv-surviving-prefix-reuse-plan.md
[SL06]: https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/0bf9d2efeaa10144dd7f0ff4597298df029f9328/docs/kv-pitched-copy.md
[SL07]: https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-22-fixed-budget-scheduling-review.md
[SL08]: https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/ff58b7c07dbf1184c0d1bdd2ffd6c8c2f6c5cc2d/analysis/2026-10-21-mixed-attention-stream-overlap.md
