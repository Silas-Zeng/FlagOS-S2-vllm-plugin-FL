# B06：有界 residual RMSNorm＋量化，真实 norm 权重组件验证

## v3真实权重完整矩阵：保留206个通过组合，定位32个失败

运行a8c02d71526dc8ea7427c288cd07f9c0beceee2d私有构建和既有9/40检查通过，完整238组合收齐。
M1/8/16/17/32/64/256通过数分别34/32/33/32/29/28/18（各34），合计206/238，退出1。
原始结果/state/原检查见`evidence/bounded-norm-quant-20261024/v3-final/`。
这说明只按行数限定仍不足以覆盖所有实际norm权重，不能把全部selected_rows直接装入服务。
通过组合三轮完整norm+quant+GEMM+SiLU中位约1.0257–1.0644，M1代表层约1.0480–1.0521；
没有reset拷贝，保留正结果，但不能按单次夹具通过的层名单直接合并或宣称模型精度通过。

新增d7d2c93独立重放32份原失败输入，比较native编译norm与原单行融合、串行分块后归约的
十种树结构，记录norm bit差与residual是否相同，并收集实际native编译配置。
该诊断不改现有融合数学/原容差，也不使用模型或gate性能分数。任何选出的结构均须新完整矩阵。
32份数值重放已完成：各固定树逐位匹配数7..12/32，最佳512分块也仅12/32，
不存在一条已证实通用准确的树，不能将其直接安装。原单行norm相对native仅少量BF16 bit差，
在norm层原容差内仍可能放大成gate输出超差，原gate失败保持。
末尾元数据采集因本机PyCodeCache无cache属性退出；原result含全部32份数值结果，
raw log/result保存在`evidence/bounded-norm-quant-20261024/reduction-audit/`。
现已只读核实安装版用modules列表，工具改为该接口，并在元数据采集前持久化数值summary；
没有为修复末尾汇总重复GPU实验，也没有声称已取得实际native最优配置。
分支已af4ec05非重写同步团队main0bf9d2e，原CPP函数/CUDA/Python保持最新main。

当前工具修复：cff7704私有build、原9和40检查通过，但probe在启动时因缺少
`audit_biv150_residual_reference`退出，尚未运行238组合，不是norm核数值失败。
原日志/state/检查文件保留在`evidence/bounded-norm-quant-20261024/rejected-v2/`。
现将CPU量化/输出两个参考函数独立提取成`biv150_residual_cpu_reference.py`，
CPU核对两函数AST与此前已验证audit逐字语义相同，并新增所有本地probe import存在性检查。
该修复不恢复失败PR26生产路由、不改Triton数学、不放宽容差；新冻结版本仍需设备验证。

当前独立基线为团队 main `31a0a7f`（含已验收 PR28、队友 Metax PR34）。
PR26 新路由模型精度99/105=.9429失败，原1cc4e22组件队列正确拒绝启动，没有设备结果。
新提交非重写同步main，再移除所有PR26/PR31生产改动及无关实验文件，
仅在main原CPP中增加本探针的两个独立算子。原CPP函数逐条保持，CUDA和vllm_fl逐字保持main。
PR26/31的历史成果和证据留在各自PR，没有将失败的auto路由带入本轮。
此 Draft 尚不安装模型 hook，无模型精度或吞吐成绩。

PR29 全范围融合15/18通过，失败的M40/2048不再进入这个候选。保留已有通过的行数1/8/16/17/32/64/256，但旧结果不能证明所有真实层权重均通过。因此本轮读取模型 safetensors 中34个实际压缩层（4..37）的post_attention_layernorm.weight，记录每个权重的SHA，覆盖238个形状/权重组合及1428个数值模式；投影仍是冻结的合成权重，不把组件测试冒充模型验收。未按题目、答案或4K/16K测试名选层/形状。

已核实实际模型config和服务解析为LlamaForCausalLM，pin安装llama.py以及native IR/层归一化源码。LlamaDecoderLayer在attention之后调用post_attention_layernorm(hidden_states,residual)，原编译语义是FP32残差加法及归一化，末尾BF16输出；源码级preweight cast是否消除由既有实际Inductor证据约束。新核仍用原libdevice.rsqrt/量化舍入/原INT8 GEMM/SiLU，不放宽.005/.005。

与旧探针的区别是：输入与residual均不修改，residual输出分配独立存储，normalized仅在数值诊断时单独写出；性能路径省掉normalized中间量。原基线也使用非原地native impl，因此两条完整norm+gate测量均不需要每次做输入恢复拷贝。每个真实norm权重的六模式检查输入不变、输出别名、residual逐位一致、norm/gate原容差、量化对自身原核逐位一致、CPU归一化与小M残差gate参考；每项有变更输入Graph重放。M1..16仍沿用残差量化语义，较大M沿用普通量化。

性能只测代表层4/20/37，每个形状三轮交替、充分预热Graph，计完整norm+量化+INT8 GEMM+SiLU及residual输出。所有238项记录均保留；任何失败保存输入fixture和SHA，按形状总结。失败时不进入模型、不能用少数成功记录声称全范围通过。

CPP仅新增接收已量化激活的组件入口和量化对照入口，不改既有生产函数；CUDA生产文件与vllm_fl保持依赖版本。尚未注册服务fake/hook，因此不宣称已接入实际Llama层。设备组件通过后仍需实际完整层与服务hook验证、105题精度、固定充分预热4K/16K及TTFT，才可考虑合入。

新队列只等待所拥有的前驱任务终止，并确认GPU和9031空闲。私有build后重复main原9和40分发检查，
再测本轮融合；不再要求失败PR26的formal completed，也不调用不属于本基线的residual-cache或PR31策略检查。
CPU证明本轮没有修改main既有生产CPP/CUDA/Python，设备检查仍保留main既有9/40要求。
当前仅语法、CPP原函数保持及Triton接口AST检查通过；上机结果待回填。
