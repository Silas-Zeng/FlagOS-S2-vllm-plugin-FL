# CPU metadata视图复用（实验草稿）

本方法减少MiniCPM5-2B输入准备中的重复元数据构造。当前证据仅来自天数BI-V150；沐曦－曦云C500尚未验证。代码仓库由团队共用，不代表两平台统一选定本方法。

原runner已经用NumPy计算token位置，并在CPU得到带零尾的optimistic序列长度。本方法将位置复制到一次分配、反复复用的pinned staging，再以两次异步H2D写入原 `positions` 和 `seq_lens` GPU缓冲。请求索引与scheduled数量的CPU镜像继续更新；该严格scope内不再提交三个仅供原GPU表达式使用的元数据副本。`num_computed_tokens` 的原H2D仍保留。

接入只位于 `vllm_fl/worker/model_runner.py` 的 `_prepare_inputs`。原 `synchronize_input_prep` 生命周期保护持久staging的跨步复用，GPU目的地址不替换，不新增stream、同步、RNG操作或请求重排。未命中配置与描述守卫时，执行完整原表达式；已经开始复制后的错误向上传播，不在部分修改后重试旧路径。

runtime要求MiniCPM5-2B的BF16、42层、hidden size2048、16个Q头/2个KV头，相关并行度均为1，原async scheduling与prepare事件存在，pinned memory开启，并排除speculation、LoRA、pooling、多模态、特殊位置编码、共享KV快速prefill等未知组合。tensor/NumPy dtype、维度、连续性、同一computed CPU视图指针、GPU地址不别名等也会检查。当前流处于capture时退原路。

首次命中还核对已加载依赖文件的SHA256：

| 文件 | SHA256 |
| --- | --- |
| `vllm/v1/worker/gpu_input_batch.py` | `de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65` |
| `vllm/v1/utils.py` | `4e4cda209e10d4477d251003779dec26ea5ef47541c3310995c007c7ee5fdcdb` |

依赖不存在或SHA改变会报错，不能把它描述成适配任意vLLM版本的补丁。本实现也未检测芯片型号；其他平台应由负责人独立评审，不能仅凭scope通过认定已支持。

已上卡原件身份：

| 文件 | SHA256 |
| --- | --- |
| `vllm_fl/patches/biv150_cpu_metadata.py` | `0e96152454a9197f65a34c2fe8805bd5073e1fa473070d59d4ec4f20a6d8846f` |
| `vllm_fl/worker/model_runner.py` | `50493fb68b194ed935650a3a743d2f56696d7379b702f4d7717eecac0325a9de` |

`20261021T174115Z-cpu-metadata-gate` 通过五种形状：decode62、mixed259、mixed952、mixed2048、prefill2048。检查位置、序列长度、computed计数、input IDs、query起点、discard mask、accepted计数、logits索引、slots、CPU镜像与固定GPU地址，并检查连续host改值/原event/图consumer。该门方法级wall `0.712748 → 0.656465 ms/call`，不代表完整服务提升。

`20261021T175049Z-cpu-metadata-decode4k` 是一次4K输入、128输出、64请求的短服务，对照为随后运行的 `175441`。两者64完成、0失败，输入262144/output8192 token，KV521936，同DB三SHA，CPU6/32GiB、preempt0→0及原服务参数。耗时 `39.185280642 → 39.073248478 s`，total tok/s约 `+0.286723%`；平均TTFT约 `+0.234431%`，变慢。保留这次小信号，不称稳定收益；相同DB不保证编译缓存和所有主机状态完全相同。

原始证据位于实验记录库 [FlagOS-S2-Minicpm-throughput](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput)：`experiments/framework/20261021T174115Z-cpu-metadata-gate/`、`experiments/framework/20261021T175049Z-cpu-metadata-decode4k{,-control}/`、`experiments/framework/20261021T175441Z-baseline-decode4k{,-control}/`。独立比较见该库 `analysis/2026-10-22-small-framework-comparison.json`；PR准备材料另存证据哈希，未将全部实验原件复制进代码库。

本方法没有新的整模型准确率或完整官方吞吐结果，原稳定SQL+C3的准确率不能自动覆盖它。PR保持Draft，后续必要补验由团队决定。回退本方法时恢复基点runner的原表达式和import，并移除本runtime；不要顺带回退已有prefill、top-p、SQL或C3。
