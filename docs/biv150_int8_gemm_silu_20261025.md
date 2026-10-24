# F09：不物化INT32输出的INT8 GEMM＋BF16边界SiLU

状态：独立组件Draft，CPU结构/范围及语法检查通过，GPU数值与完整性能待测，未安装服务hook。
基线为已核实团队main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`（PR35编译缓存语义身份），
不合入官方上游框架最新分支，不修改团队生产csrc/vllm_fl。

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
