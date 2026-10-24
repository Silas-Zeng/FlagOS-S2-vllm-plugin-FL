# F09：不物化INT32输出的INT8 GEMM＋BF16边界SiLU

状态：Draft PR37，首轮24/24及M32真实权重34/34通过；已准备严格M32服务入口，模型联合验收待运行。

## 真实权重通过与服务候选

1f06b7e在34个压缩层的真实gate/up权重上，五模式、native全输出与独立CPU INT64、
changedGraph全部通过。34层完整三轮中位1.02869–1.03509×，102轮最低1.02350×。
激活仍为合成，不是完整模型精度。原始结果及权重SHA见`evidence/int8-gemm-silu-20261025/real-weights-final/`。

新服务候选将同一Triton核移到包内，工具直接复用，注册opaque custom op及fake。
只在压缩权重标记、M32/K2048、原native dtype/layout/device与min_rows17均匹配时运行，
其他行数、BF16真权重、非连续输入及不支持参数均调用原gate_silu_auto，M<=16原残差路由保持。
LlamaMLP现有入口只切换到有界分发，不更改attention、模型权重策略、调度或正式参数。
新元数据如实记录gate_m32_or_auto及范围/配置，验收要求这些字段和新增选择标记，
source manifest包含新kernel；PR35官方缓存身份已涵盖全部biv150补丁源。

新route设备检查覆盖26个行数/权重组合、非连续/参数回退、原路径逐位一致，
opaque fake/fullgraph及改变输入Graph。随后保持原完整105题、充分预热单轮4K/16K、
请求tokens/TTFT/compute-order/Qfragment门槛；只有模型联合验收后才能决定是否保留。
当前a19ff5c私有build、原9/40、新26行数/权重组合及非连续/参数回退、fullgraph/fake/changedGraph已全部通过，
原始设备分发前检见`evidence/int8-gemm-silu-20261025/service-prechecks/`。
完整验收已启动：队列178221/start320757084、driver178225、acceptance178862，服务178865已ready，
当前实际ps确认存活；尚无105题精度或新4K16K成绩，不能称服务验证通过或模型提速。

## 首轮完整结果

运行78f3f00e5a0f661f945466d6710e20c22f714a30私有main原SO，原9/40检查通过；
六行数×四配置均通过五模式、完整native及CPU INT64采样对照、changed-input Graph。
全部完整三轮时长和原检查/state见`evidence/int8-gemm-silu-20261025/initial-final/`。
只有M32、tile(32,64,128)、warps4为稳定正结果，三轮1.029099/1.029529/1.026805倍；
原调用约0.056ms，候选约0.0545ms，显式INT32输出省去1572864字节。
M17/40/64/256/2048当前配置均慢，不能为了“融合”覆盖这些形状，也不能仅报中间量省去大小而忽略完整GEMM退化。

保留M32候选，1f06b7e添加34个实际压缩层gate/up投影权重（逐项SHA）的验证，
完整五模式、CPU INT64、native全输出、changedGraph与三轮完整计时。激活仍为合成，
不是完整模型精度或路由覆盖证明；若实际权重数值/性能不一致，记录并继续限定或关闭该变体。

## 初始实现和验收范围
基线为已核实团队main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`（PR35编译缓存语义身份），
不合入官方上游框架最新分支；原native CPP/CUDA不改，候选仅增加上述包内有界路由。

此前PR30只调native GEMM后的SiLU尾处理，仍写INT32矩阵；本实验重新写两个INT8累加器的
Triton GEMM，沿用native单份 `[2048,12288]` INT8权重、原quantizer、原scale布局。
两个投影在寄存器内反量化，各自转换BF16再FP32激活，最终只写 `[M,6144]` BF16输出。
不存在显式INT32全局输出分配或store，不要求权重重排及第二常驻副本。
原生GEMM内部调度可能仍然更快，因此必须看整个调用，不能仅凭省去张量判赢。

初步四配置改变M/N/K tile结构，覆盖普通量化路径M17/32/40/64/256/2048，
每配置zero/tiny/normal/large/outlier五类输入。小M残差路径M<=16尚不支持，
未实施的残差融合不能算完成；后续须独立设计准确的双残差组合。

正确性比较完整native `gate_silu_auto`输出，另用CPU INT64点积、原scale乘法分组、
投影BF16舍入和FP32激活参考，保持原.005/.005。native也须通过CPU采样对照。
INT32 dot界 `2048×127×127=33032192` 安全；GPU编译/整数矩阵单元行为尚待实证。
Graph capture后更改输入，重放对原完整入口；三轮交替充分预热Graph时长包含
原quantizer、各自必要分配、完整GEMM与SiLU，记录原INT32张量省去的字节数。
没有把旧epilogue局部加速作为新kernel收益。

私有目录先构建main原生SO，重复既有9与40检查，再运行组件；GPU/9031空闲及
前驱PID/start_ticks核验，不并发性能验收、不改site-packages。
首个形状所有配置均失败即停止，保留完整异常链，避免无意义重复同一编译错误。

本轮使用合成冻结投影权重，未证明服务真实命中、Level3、TTFT或吞吐提升。
任何组件收益之后还须实际模型接入、真实层/形状覆盖、两档完整联合验收，才能合并。


## 新清单AD02/AD03/S11：当前生成图区块的便宜证据门

从当前a19ff5c服务的私有cache只读导出10个含实际call入口的生成模块，按导出SHA验证，
只用AST读取、不执行生成Python。原文件/source manifest与审计结果在`evidence/int8-gemm-silu-20261025/generated-graph-audit/`。
缓存存在不证明每个生成模块都在该次请求中命中，静态语句数不等于launch次数或耗时；不以此代替runtime profile。

10个wrapper中未发现显式ATen/prims clone/copy/_to_copy/contiguous/convert_element_type调用。
包含本候选opaque gate的动态区块（pc/cpc7…，SHA12c12152…）显示：
O GEMM→norm→BF16 buf2→gate custom op→down GEMM→后续norm/QKV/RoPE。
Gate输入由empty_strided_cuda((M,2048),(2048,1),BF16)分配，当前这条边是连续的，
因此不能把C++源码x.contiguous()直接计为一次真实复制。buf4=buf3为Python别名，reinterpret_tensor为view；
编译器还复用buf8作为RoPE Q输出，不能把已有合法复用重报为新内存优化。
这只排除本批生成wrapper的显式拷贝问题，不覆盖opaque内部、attention外部边界、其他shape/版本或真实峰值。

另有首层模块ik/cikyut…显示embedding gather与RMSNorm已经处在同一生成kernel，并分别输出
原embedding/residual与norm结果。不能仅据kernel名含_to_copy就认为多了一次copy launch，
也不能把AD14的“少一个launch”作为新实现收益。其核内仍可研究寄存器驻留/回读成本，但要单独计完整成本。

对PR33尤其重要：pre-gate norm kernel仅写归一化输出，未单独写出该步residual；
后续norm接收down输出、先前O输出和旧residual，在FP32中做down+(O+residual)，再存输出。
在此生成路径的两个加法之间没有BF16舍入。原PR33孤立probe要求独立BF16 residual输出，
其组件通过不能证明插入完整图后仍保持这条跨op数值契约；opaque融合若强制落地BF16 residual可能改变行为。
下一版norm接入应保留原透明residual表达式/活跃输入或匹配实际子图，不因孤立测试过了就改模型边界。
这不是已证明的多余cast可删，也不是关闭全部norm融合方向；它改变接入位置与验证范围。


## a19ff5c完整服务部分结果（16K计分中，不作合并结论）

Level3 .9714，105条预测/105唯一问题/0失败。充分预热单轮4K：5764.652468708367 total tok/s，
TTFT3685.0361691467697ms，256/256，输入1048576/输出262144 tokens。对PR28独立充分预热单轮
5765.810177399745约−0.0201%，不证明新增服务收益，也不将这点差异判为确定退化。
16K仍计分；只有请求/token完整性、TTFT与新增性能证据共同具备后再决策。
当前源版本与此前prechecks一致，完整服务SO摘要3e114fc0201dd8e4ef31b5aa7a3c75364469a6d69e54d3ae2d6ac929e254830d，
与组件私有build不同，不能移植SO身份。启动KV547632，不据这一单次容量差异宣称增容。
原始部分accuracy/scores/预热JSON及逐文件capture SHA在service-partial/；该目录不是16K终态。


## a19ff5c完整终态：验收门通过，增量服务收益尚未建立

Level3 .9714；充分预热单轮4K5764.652468708367、16K4049.9530857620834 tok/s，
TTFT3685.0361691467697/124203.12683230804ms。两档完整256/128请求、0失败、token总数正确，
scope/Qfragment/新route选择等工作验收门全通过。原始manifest/acceptance/state/warmups/scores见service-final/。
对PR28独立单轮5765.810177399745/4048.1358315864945分别约−.0201%/+0.0449%，
不能据此确认新增端到端收益，因此继续Draft而不合并；保留34层M32完整组件正收益，不把整个F09方向判失败。
这些是研发单轮，不替代官方四轮报告。


## P30/下一轮M64：从真实Corex资源出发，而不是假定spill

只读24个已编译fused_gate缓存：warp_size均64，旧配置汇编vgpr_spill_count均0；不能把旧M64慢直接归因为spill。
TTGIR当前MMA主要warpsPerCTA=[1,4]/[1,8]沿N分配，编译元数据保留8/16KiB shared与VGPR计数。
新增纯CPU审计器不导入Torch或访问GPU，逐case核对M/BM/BN/BK并保存原始IR/asm/cubin哈希。

新component复用完全相同的生产fused_gate kernel AST，仅改变launch tile：
保留旧冠军(32,64,128,4)，比较(64,32,128,4)、(64,64,128,4)、(64,64,128,8)。
BM64减少M64的M向CTA分块，BN32方案保持每CTA双acc元素数与旧方案相同，8warp方案探索不同M/N warp映射；
可能改善权重/CTA复用，也可能增加寄存器成本，均须实测，不能从CTA数推出DRAM/occupancy收益。
不改量化、BF16舍入/SiLU、权重布局或服务guard，不把M64新配置直接接入生产。

先用真实layer20 gate/up权重，M32/64/40/63/256/2048×4配置=24case，原五模式、完整native对照、
独立CPU INT64抽样、变输入Graph与三轮交替完整quant+GEMM+epilogue成本全部保留。
按正常worker默认FlagGems enable和inference_mode记录真实_int_mm dispatch前后表；资源采集错误单列，不冒充数值失败。
同时记录实际kernel API/汇编VGPR/spill/shared与IR，用于决定后继34层验证；当前没有新设备结果或收益结论。
新队列只接在BF16后继之后，GPU/9031空闲且私有native SO SHA一致才开始。


## 6e5dc2f的P30真实设备结果

24/24组合五模式/CPU INT64抽样/完整native/变输入Graph均通过。实际M64旧(32,64,128,4)仅约.7449x；
新(64,64,128,8)三轮最小1.1091x、中位1.1105x，VGPR61、spill0、shared16KiB。
(64,64,128,4)约1.0436x但VGPR92；(64,32,128,4)约.2534x，不能凭每CTA acc元素数相同推断性能。
M32仍旧冠军1.0308x；M40/M63/256/2048新tiles全部负收益，保留旧fallback，不全局扩展。
这只是layer20真实weights+合成激活的完整组件收益，非服务成绩。

下一冻结版本用相同(64,64,128,8)与worker默认策略/inference_mode验证全部34个gate/up weight；
旧M32工具默认与所有既有结果保留。通过前不把M64加入服务guard，也不依赖PR40的新allocation顺序。
