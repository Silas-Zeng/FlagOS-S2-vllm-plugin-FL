# D02 / B09：真实短 query 与 RoPE/KV 入口审计

## 当前：v5两档实际覆盖已完成，非性能优化成绩

运行27202003a15c3dfd2bb9ca36406b0c020fc79abb（生产代码与已验收PR31 e972997相同），
原检查、两档充分预热诊断完成，实际1个LlamaForCausalLM、42个RoPE层入口，errors为空。
原始state/source manifest/scores/warmups/verdict/完整trace见`evidence/short-query-rope-audit-20261024/v5-final/`。

| 阶段 | runner步 | forward query token | qlen2..8步/行/token | token占比 | 新split eager kernel launch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 4244 | 1310464 | 2 / 2 / 7 | 0.000534% | 16800 |
| 16K | 4850 | 2281166 | 1 / 1 / 8 | 0.000351% | 31206 |

短query在两档均很少，D02暂降优先级；频度不是设备时间，未给出严格提速上界。
新split计数证明PR31 B29/30/60..63确实被服务调用，但不证明整机增量收益。
FULL Graph不在eager统计内，不能把这些数当所有attention launch总量。

42个RoPE路径共享一个对象，类为RotaryEmbeddingFL，但实际选中的_forward_method均为
`RotaryEmbedding.forward_native`，源码SHA c8180449…；不能把OOT候选当实际调用。
因此B09后续应对照native编译路径与KV写入的组合总成本。先前缓存证明存在Q/K combo，
本轮绑定审计仍不证明某个具体生成kernel被选中；尚未实现新的RoPE/KV融合核。

以下早期段落为历史快照，其“待上机”由上方v5实际证据取代。

基于已核实团队 main `5eda0b0da0ecbe1839105c49f58462329fd9ea22`；仅新增独立工具，不修改框架、调度、生产算子或 source guard。此 Draft PR 尚无新算子、无服务提速成绩。

当前同步已合入main6a013bd和PR31的0248416，生产csrc/vllm_fl与PR31冻结e972997完全一致，诊断明确依赖PR31正式通过。compute-order pin同步为精确80b88140…，并重复PR31 CPU chooser检查；不是放宽source guard。实际模型按config/server选择LlamaForCausalLM，不能由目录名称推断使用models.minicpm。

## V4预检拒绝及修正

冻结91084aa启动服务后，在第一轮warmup发请求之前被预检拒绝：42层父RoPE加上42个内部`apply_rotary_emb`被`endswith("rotary_emb")`误计为84，attention为42。源base.py确实在父RoPE构造中设置ApplyRotaryEmb子模块。该诊断没有产生有效qlen覆盖或吞吐成绩，原finally已清理服务；随后PR26正常接续。拒绝state/trace/log保留，不能把它描述成模型/算子失败或实际存在84个attention层。

现在只匹配完整`.self_attn.rotary_emb`路径，仍用remove_duplicate=False保留共享对象的42个层入口。CPU夹具明确含每个父模块下的`apply_rotary_emb`，验证42父路径、排除42子路径；不是简单放宽计数到84。

另新增GQA JIT原run后的CPU计数：仅当前forward context为NONE且非warmup时记录实际grid及与SEGMENTS的一致性；原run的结果与异常不改，FULL capture/回放不计入eager。PR31新增B29/30/60..63不是FULL capture batch，因此这项可验证对应eager调用是否命中新split；它不是通用Graph覆盖图，也不测设备时间。CPU检查含warmup/FULL排除和原返回/异常保持，待新冻结独立诊断。

## 已核实源码

- 当前实际 runner 是 `vllm_fl.worker.model_runner.ModelRunnerFL`，不是上游 GPUModelRunner。其 `execute_model` 在 `synchronize_input_prep()` 生命周期内，从原请求顺序生成 CPU int32 scheduled 数组，再调用本模块导入的 `_biv150_compute_order_plan`，然后 `_prepare_inputs`。该调用位于 `max_query_len` 判断之前，因此可观测纯 decode 与混合调用；观察 imported alias 才能命中真实入口。
- 当前实际 Graph wrapper 是项目 `vllm_fl.compilation.graph.GraphWrapper`。在原构造函数返回后读取其实际 runnable 的 42 个 rotary 对象与 42 个 attention impl；记录 `_forward_method` 的绑定、源码 SHA、native/OOT 候选与 custom_ops 配置。不会执行这些 forward，也不改 attention wrapper 的函数/闭包。
- 已核对模型config及实际server resolved architecture：权重目录名为MiniCPM5-2B，但实际架构是 **LlamaForCausalLM**，不是models.minicpm。观察器pin实际models.llama SHA `1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384`，并检查配置/记录实际runnable类。Llama在QKV后先做rotary，再调用attention；decoder post_attention_layernorm接收hidden_states与residual，适用residual RMSNorm路径。TritonAttentionImpl.do_kv_cache_update 使用 `triton_reshape_and_cache_flash`；现成 `do_rope_and_kv_cache_update` 依赖 ROCm AITER。该接口存在不表示天数当前使用融合，也不能仅凭 RotaryEmbeddingFL 类存在断定 OOT 路径实际生效。此前读取minicpm.py只能说明另一个实现，不能当作当前模型路径证据。

## 采集与边界

私有 PYTHONPATH bootstrap 在原 native.install 成功返回后装观察器，显式环境变量关闭时不生效。仅包装 CPU schedule alias 与 GraphWrapper 构造函数；原 plan 结果对象、原异常与原构造行为保留。CPU scheduled/computed 镜像生成精确 qlen 直方图、qlen2..8 的次数、query token 数与 CPU 乐观 KV 长度；不复制 GPU 张量、不插入设备事件、不读取请求内容。此频度不能代替算子耗时或提速上界。

使用原固定服务命令和原充分预热后的 4K/16K 请求，单独诊断启动；诊断时间不计入正式成绩。各阶段先收到引擎 phase ack，才发请求；缺实际模型对象、无 schedule 命中、出现 scope/fallback 错误即拒绝继续负载。独立私有 build 与原 9 项算子检查、40 项 dispatch 检查先行。

排队脚本先核对前驱 PID/start_ticks，等待其终止；还要求指定 commit 的正式验收 state.completed 为 true，并确认 GPU/9031 空闲，不中断在跑任务。若前驱失败则停止等待人工分析，不自动占卡重跑。

## 当前验证

本机 CPU 检查通过：执行真实 pinned runner 的连续 schedule/alias/prepare AST 语句，覆盖 qlen1..8 与 9/16/32/2048；核对完整直方图、短 query/KV 二元表、原数组/结果/异常不变、实际方法未替换，以及 bootstrap 开关与原 install 返回值。

真实模型绑定、实际 qlen 频度与设备成本仍待上机。获取频度和实际入口后，D02 再决定短 query 专核，B09 再比较原组合总成本并设计天数融合；不能把此审计描述成两个优化方向已经实现。

注意 `get_rope` 可缓存并跨层复用同一个 nn.Module，默认 named_modules 会去重。观察器明确使用 remove_duplicate=False，统计 42 个层路径并同时记录独立 rotary 对象数；CPU 检查包含 42 条路径共享一个对象的情况。不能把对象数少于 42 误判成缺层。

## 只读编译证据与排队

已核实实际安装的 rotary 工厂 SHA `f703d3f7ca6be6b41837d4a07a27b94d1442149554bb4f4837cd3722671a5bbd`，含 `_ROPE_DICT`。修正版 `efd80d3` 本机与个人实例 CPU 检查均通过。

从正在正式验收的 `84c1be5` 服务缓存只读取得 M2048 Q/K 两个单独源及一个双输出 `SequentialComboKernelGrid` 源，保存在 `docs/evidence/short-query-rope-audit-20261024/compiled-rope/`，原路径与 SHA 见 summary.json。三者均无 slot_mapping。**生成缓存不证明具体 kernel 运行时被选中**；尤其不能根据两份单独文件就认定实际需要两个 RoPE launch，简单 Q/K 拼核也不能直接算新优化。下一步比较必须包含已有 combo 候选与 cache 写入的完整成本。

第一版等待队列122662只在 waiting、无child时受控取消，未占GPU；新冻结快照 `code/short-query-rope-audit-v2-20261024` 为 efd80d3，队列123076/start319596243，等待 PR31 的119285/start319363061。还要求PR31 e972997的正式验收completed后才启动，输出short-query-rope-audit-v2-20261024。此处记录的是排队身份，不代表已完成GPU诊断。
