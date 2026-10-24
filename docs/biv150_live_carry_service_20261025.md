# M64 norm/quant＋INT8 GEMM/SiLU 有界接入

从fresh团队main0bf9d2e单独建立接入分支。只提升PR33/PR37已验证的M64组合，不带旧CPP probe API或全局M32/M64 gate替换。四个数学函数AST与7dca973组件一致；压缩层全部34层完整链条三轮最小比1.049584–1.070327，Graph/eager34/34逐位一致，原严格29/34，其余5个失败保留，尚无模型质量/吞吐声明。

仅在已审核MiniCPM/BI-V150 native安装之后，核实原Llama模块SHA及forward身份，将原forward中相邻post-norm＋MLP两句替为helper，其余参数、attention、input-norm、返回保持原AST。helper在M64/隐藏2048/原INT8压缩gate/原SiluAndMul/连续BF16且模型epsilon1e-6时，opaque算子返回融合gate；原norm的透明residual继续独立留在图中，down projection仍原方法。M32/prefill/边缘BF16层等直接执行原两句。

不新增持久权重副本，不改原CPP/CUDA、量化权重/激活数学或调度/benchmark参数。自有开关进入编译语义指纹，新源由既有patches glob覆盖。CPU源/策略/最小AST替换/回退及carry对象身份/缓存门通过后，仍必须检查实际安装模型forward、fake/fullgraph/变输入Graph、生成图FP32残差契约，再做首次105题及固定充分预热4K/16K/TTFT验收。99/105可测吞吐但不能合并；当前未达到上述GPU与服务门。

新增顺序pipeline：源CPP/CUDA逐字节核实后复用已验证私有主线SO，副本置于本冻结source独立build目录，不装共享包。旧gate9/cached40之后，实际SDK原/改写forward在真实权重proxy模块上核对M64提升、M32原路径、FakeTensor/fullgraph和变输入Graph；attention/input-norm为identity proxy，不能称为完整模型。选中结果须对直接提升同数学组件逐位一致，回退须对原方法逐位一致，原近误差数值报告仍单独保留。

这些门通过才自动启动完整服务。首次105若99/105继续吞吐但precision_gate=false，绝不合并。4K预热改为128请求/128输出、并发64，以覆盖滚动请求mixed阶段；16K仍64请求/128输出，稳定后正式4K256请求和16K128请求、输入4096/16384及输出1024、seed0、并发64完全原样。后续匹配对照使用相同预热；不与不同预热旧单轮直接认定小增益。

首次pipeline ff2a869在CPU AST来源校验停止，未启动GPU；本地Python3.14 ast.dump省略的空字段在远端3.12输出中存在，产生不同摘要。后继校验改为显式包含所有AST字段的canonical JSON，并重新从原7d组件函数生成预期值；旧3.14摘要保留，内核/生产代码未改。
