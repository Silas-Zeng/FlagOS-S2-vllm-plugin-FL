# M64 norm/quant＋INT8 GEMM/SiLU 有界接入

从fresh团队main0bf9d2e单独建立接入分支。只提升PR33/PR37已验证的M64组合，不带旧CPP probe API或全局M32/M64 gate替换。四个数学函数AST与7dca973组件一致；压缩层全部34层完整链条三轮最小比1.049584–1.070327，Graph/eager34/34逐位一致，原严格29/34，其余5个失败保留，尚无模型质量/吞吐声明。

仅在已审核MiniCPM/BI-V150 native安装之后，核实原Llama模块SHA及forward身份，将原forward中相邻post-norm＋MLP两句替为helper，其余参数、attention、input-norm、返回保持原AST。helper在M64/隐藏2048/原INT8压缩gate/原SiluAndMul/连续BF16且模型epsilon1e-6时，opaque算子返回融合gate；原norm的透明residual继续独立留在图中，down projection仍原方法。M32/prefill/边缘BF16层等直接执行原两句。

不新增持久权重副本，不改原CPP/CUDA、量化权重/激活数学或调度/benchmark参数。自有开关进入编译语义指纹，新源由既有patches glob覆盖。CPU源/策略/最小AST替换/回退及carry对象身份/缓存门通过后，仍必须检查实际安装模型forward、fake/fullgraph/变输入Graph、生成图FP32残差契约，再做首次105题及固定充分预热4K/16K/TTFT验收。99/105可测吞吐但不能合并；当前未达到上述GPU与服务门。
