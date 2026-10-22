# 异步采样token整数更新（实验草稿）

本方法把异步输入准备中的一次整数gather和scatter合并到一个新实现的kernel。目前只在天数BI-V150验证；沐曦－曦云C500未验证，runtime的 `current_platform.vendor_name == "iluvatar"` 守卫会让其他vendor走原表达式。团队共用代码库不等于两平台统一采用本方法。

原 `_prepare_input_ids` 已在CPU形成上一批请求索引与本批目标token位置，并异步复制两份int64索引到GPU。原代码从上一批 `prev_sampled_token_ids` 取int32 token，再scatter进本批 `input_ids`。新kernel沿同一映射直接读取与写入，避免中间张量及分开的gather/scatter提交；未修改采样、随机数、请求排序、数值精度、服务参数或benchmark。

接入只在 `vllm_fl/worker/model_runner.py` 增加一个import，并将该scatter表达式放进helper失败时的原路径。identity slice、无previous/common、draft/spec、prompt embeds等原分支保持原代码。原两份索引H2D、原stream顺序及消费者保持；helper按实际current stream登记四个输入/输出的 `record_stream`，仅管理分配器生命周期，不建立跨stream就绪关系。

受测runtime要求iluvatar、MiniCPM5-2B BF16、42层/hidden size2048/Q头16/KV头2，相关并行度均为1、async scheduling，排除speculation、LoRA、prompt embeds、encoder-decoder与特殊位置编码等。token为int32，索引为int64，同device、正stride、四份storage不同；CPU映射须长度相同、索引合法、目标唯一。重复源读取允许，重复目标回退，避免并行写无法表达原last-writer语义。capture退原路。配置或tensor描述不符合时执行原表达式，不在已提交后重试旧路径。

kernel以运行时参数传长度、source/destination行数和stride，`do_not_specialize`排除这些值的专门化。固定BLOCK128、num_warps2、num_stages1，无autotuning或按请求数生成源码。

受测源码SHA256：

| 文件 | SHA256 |
| --- | --- |
| `vllm_fl/patches/biv150_async_token_update.py` | `1bb85e31c7b3fefa81414a54b04a1f0b6cf949c0876f87149956c821bb483adc` |
| `vllm_fl/worker/model_runner.py` | `f0d0bada4222038b69ed139d804622e27acd6aa4e19a37ff089b0ca35f393f41` |

`20261021T063353Z-async-token-gate` 的12类真实方法case通过，覆盖identity、无previous、empty、无common、删除/condense、reorder、新prefill空洞、正stride、int64回退、prompt-embed回退、spec/draft回退和cross-stream。该门使用真实方法与替身配置/缓冲区，不加载模型；cross-stream由caller显式wait保证ready，未穷尽所有allocator schedule。SQL+C3适配证明原/新方法体与该门逐字节一致，适配过程没有改kernel。

`20261021T174741Z-async-token-decode4k` 是一次4K输入、128输出、64请求的短服务；对照为 `174402`。两者64完成、0失败，输入262144/output8192 token，KV521840，同DB三SHA、CPU6/32GiB、preempt0→0和原服务参数。耗时 `39.410040799 → 39.134667002 s`，total tok/s约 `+0.703657%`，平均TTFT约 `−0.865638%`。保留单轮小正信号，不称稳定收益；FIRST_SELECTED只证真实命中，不能当全步计数。

证据在实验记录库 [FlagOS-S2-Minicpm-throughput](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput)：`experiments/framework/20261021T063353Z-async-token-gate/`、`experiments/framework/20261021T174741Z-async-token-decode4k{,-control}/`、`experiments/framework/20261021T174402Z-baseline-decode4k{,-control}/`；适配证明为 `analysis/2026-10-22-async-token-adaptation.json`，比较为 `analysis/2026-10-22-small-framework-comparison.json`。PR准备材料保留证据哈希，不把全部实验原件复制进代码库。

本方法没有新整模型准确率、完整官方吞吐或C500结果；旧稳定候选的准确率不能自动覆盖它。保持Draft，后续必要补验由团队判断。CPU metadata草稿是另一独立方法，与本方法共同修改runner；当前没有组合后的设备/准确率证据。回退本方法仅恢复对应import与原scatter表达式，并移除本runtime，不回退基点已有方法。
