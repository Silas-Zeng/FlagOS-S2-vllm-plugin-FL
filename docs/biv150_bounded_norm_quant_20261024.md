# B06：有界 residual RMSNorm＋量化，真实 norm 权重组件验证

依赖 PR26 的服务候选 `8a1fc9b`，其中包含 PR31 与已合入 main6a013bd 的 PR28。此 Draft 尚不安装模型 hook，无模型精度或吞吐成绩。

PR29 全范围融合15/18通过，失败的M40/2048不再进入这个候选。保留已有通过的行数1/8/16/17/32/64/256，但旧结果不能证明所有真实层权重均通过。因此本轮读取模型 safetensors 中34个实际压缩层（4..37）的post_attention_layernorm.weight，记录每个权重的SHA，覆盖238个形状/权重组合及1428个数值模式；投影仍是冻结的合成权重，不把组件测试冒充模型验收。未按题目、答案或4K/16K测试名选层/形状。

已核实实际模型config和服务解析为LlamaForCausalLM，pin安装llama.py以及native IR/层归一化源码。LlamaDecoderLayer在attention之后调用post_attention_layernorm(hidden_states,residual)，原编译语义是FP32残差加法及归一化，末尾BF16输出；源码级preweight cast是否消除由既有实际Inductor证据约束。新核仍用原libdevice.rsqrt/量化舍入/原INT8 GEMM/SiLU，不放宽.005/.005。

与旧探针的区别是：输入与residual均不修改，residual输出分配独立存储，normalized仅在数值诊断时单独写出；性能路径省掉normalized中间量。原基线也使用非原地native impl，因此两条完整norm+gate测量均不需要每次做输入恢复拷贝。每个真实norm权重的六模式检查输入不变、输出别名、residual逐位一致、norm/gate原容差、量化对自身原核逐位一致、CPU归一化与小M残差gate参考；每项有变更输入Graph重放。M1..16仍沿用残差量化语义，较大M沿用普通量化。

性能只测代表层4/20/37，每个形状三轮交替、充分预热Graph，计完整norm+量化+INT8 GEMM+SiLU及residual输出。所有238项记录均保留；任何失败保存输入fixture和SHA，按形状总结。失败时不进入模型、不能用少数成功记录声称全范围通过。

CPP仅新增接收已量化激活的组件入口和量化对照入口，不改既有生产函数；CUDA生产文件与vllm_fl保持依赖版本。尚未注册服务fake/hook，因此不宣称已接入实际Llama层。设备组件通过后仍需实际完整层与服务hook验证、105题精度、固定充分预热4K/16K及TTFT，才可考虑合入。

排队在PR26服务之后，要求其指定commit完整completed，同时确认GPU和9031空闲。私有build后先重复原9、40、大/小M实际分发检查和28784 policy CPU检查。当前仅本机语法、CPP原函数保持及Triton接口AST检查；上机结果待回填。
