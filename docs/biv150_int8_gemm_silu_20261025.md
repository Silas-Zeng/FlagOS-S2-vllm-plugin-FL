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
当前代码已接入候选，不能称服务验证通过或模型提速。

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
