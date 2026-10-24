# B06：有界 residual RMSNorm＋量化，真实 norm 权重组件验证

## 新联合探针：融合算子不物化carry，保留原透明表达式

4f12b62代表层六组完成：5/6原严格门过、6/6 Graph/eager逐位一致，M32三组min1.0478–1.0642、M64三组1.0216–1.0259。失败保留在layer37/M32的变输入Graph，gate12坐标超差；真实down权重CPU重放表明其传播到818个residual/819个next-norm BF16位差，并非下游没有计算。
生成wrapper确实无首个norm单独launch/残差buffer，下一norm仍直接接受down、O和旧residual。但参考编译时把特征维也泛化为符号、eps当动态scalar，与生产静态2048/6144/12288和常量epsilon不同。故先保留这批原始结果，不把2–6%认作当前生产组件增益；后继统一标记权重和输入feature维static，pin模型eps=1e-6后重测。即时CPU快照进一步固定比较边界；数学/容差不变。
新模式支持在三个代表层同一行数三轮最小比均>1且Graph一致时，自动扩展该行数到全部34个压缩层。只按统一行数作性能筛选，禁止按夹具层名单接入；任何原数值失败仍保留并允许诊断timing。当前未有该特征特化版本的GPU结果。
参考generic图先用2048行预编译，匹配实际服务profile；权重尺寸和输入feature维固定，行数保留动态。候选保留实际M32/M64支持条件，按捕获顺序先M64再M32。两边epsilon固定为已验证模型配置的1e-6，不以泛化hidden/动态scalar制造额外基线开销；正式服务和benchmark均未更改。

842bbb4首轮在目录前置检查停止：Torch导入会按TORCHINDUCTOR_CACHE_DIR提前创建目录，其原来位于结果目录内，与结果目录exclusive创建冲突。没有进入数学/性能测试，不能据此否定融合。后继将cache移至结果目录外的独立同级目录，使用新冻结source/out；首轮源码/日志/队列保持不动。

fresh共享main仍0bf9d2e，且为当前PR33分支祖先。新增benchmark完整比较post-attention norm→native gate→真实down projection→下一层input norm，而不只比较孤立norm/gate。候选opaque算子只返回gate；原norm的residual表达式继续留在外部图中，不能由融合核先转BF16写回再送给下一norm。
原_norm_quant只增加constexpr WRITE_CARRY，关闭时不分配/写入残差，默认开启保持旧probe；移除该条件后的AST与459432d逐项相同。CPU ABI/生产C++数学/CUDA/Python不变、原CPU参考AST检查通过，脚本编译过。
先用真实层4/20/37的post-norm、gate/up、down以及下一层input-norm权重，测试M32/M64六组、六模式及输入不变、变输入Graph/eager逐位一致与三轮交替完整链条成本。保持原.005/.005失败标记；有限且Graph一致的近误差案例仍测吞吐，不能将其重命名为数值通过。需要查看新生成代码，确认无新增残差buffer或未消掉的首个norm；源码意图不作为该证明。
复用a8c02d已验证私有probe库，要求CPP/CUDA源逐字节一致并pin库SHA，不重装依赖或替换活动服务库。GPU只在当前BF16匹配对照精确PID/start退出、GPU/9031空闲后串行运行。此时尚无GPU结果，不装入模型，不认定精度或服务收益；现有206/238正结果、32失败与所有反例保留。

## 新AD01/AD09/S11证据：接入前还需保持跨op residual表达式

PR37候选a19ff5c的当前私有编译cache新增只读证据（该PR的generated-graph-audit/）：
包含opaque gate的动态生成区块，pre-gate norm只写BF16 norm结果，未单独写出本步residual；
后续norm直接消费down输出、先前O输出与旧residual，生成FP32的down+(O+residual)，
两个加法之间没有BF16 cast/store。仅这是该生成区块的静态契约，不推断所有main版本或全部请求命中。

当前独立probe输出BF16 residual是一个孤立组件边界。即使某行/层数值通过，
把这个输出直接接回真实图也可能新增舍入/物化、改变下一norm的结果，不能认为只剩归约树问题。
下一版应先核对目标生产图，保留原透明residual表达式/原活跃输入或在实际子图上融合，
再做norm→quant→gate与后继norm的联合数值门；不能将此视为单独提高精度或改变模型数学的授权。
该证据改变接入与验证范围，不删除原206/238正结果或宣称整个norm方向失败。

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
