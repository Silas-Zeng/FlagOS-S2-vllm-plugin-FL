# 专用 GQA decode attention kernel

新增 `vllm_fl/patches/biv150_gqa_decode.py`，在 `vllm_fl/worker/model_runner.py` 末尾追加一次安装，替换 3D split-K decode 分支的 kernel 实现；不改 `kernel_unified_attention`、grid、builder 常量与 reducer，也不改 #9/#10 已合并的行为。守卫不通过即按原路径启动。

Kernel 以 (序列, KV 头, 段) 为 program，`ROWS = queries_per_kv = 8` 去掉通用 `BLOCK_M = 16` 在单 token decode 下空转的一半行；页索引按标量加载且随循环位置推进，主循环无 mask，只有尾页做 clamp；softmax 用 `exp2` 并把 `scale * log2(e)` 折进分数。段间 partial 沿用 vLLM scratch 布局，另写带 stride 的归约，因此实际段数可以小于分配段数 16——这是不改 builder 常量、从而让 #10 的拆分守卫继续看到 16 并保持生效的原因。归约用的最大值必须换算回自然对数（`running_max * ln2`），否则多段序列结果错误；这两点在开发记录里各对应一次可复现的失败。

依赖与 #4/#6 相同：注意力源码 `triton_unified_attention` `54c495dc…`、`triton_attention_helpers` `59d9d34b…`。SQL 不是算法依赖。安装顺序在 mixed 拆分之后，本 kernel 只接管它自己守卫通过的输入。

实测仅天数 BI-V150：

- kernel 级同种子 A/B：batch64/kv4096 由 0.948 ms 到 0.648 ms（1.46x），batch16/kv16384 由 0.981 ms 到 0.536 ms（1.83x）；KV 有效带宽 283 到 414 GB/s、274 到 501 GB/s，本 suite 的权重带宽锚点 `lm_head` 约 553 GB/s。`pages_per_step = 2`、`num_warps = 2`、`ROWS = 16` 均较差。
- 端到端同机同 prompt 暖探针，公共基 `87ed537` 对照与候选：4K 3787.49 到 5058.23 tok/s（+33.6%），16K 2589.32 到 3234.26 tok/s（+24.9%）；TPOT 68.69 到 48.16 ms、76.41 到 56.38 ms。较早快照上的官方成绩为 4K 4809.36、16K 3364.03。
- 精度：MATH-500 Level 3 官方协议 0.9714（102/105），门槛 0.95；队友在同协议下 0.9810，差 1 题，按实测记录不合并口径。与通用 kernel 的输出最大绝对差 4.9e-4（4K 形状）/ 1.2e-4（16K 形状），非位级一致，故精度跑是必要条件。

收益边界：只在 bf16、单 token、16 查询头 / 2 KV 头、head_dim 128、`seq_len <= seq_threshold_3D`（本模型为 64）且无 KV 量化、softcap、sink、alibi、mm_prefix、chunk、TD 的 decode 步生效；prefill、多 token chunk、更大 batch 的 2D 路径全部走原 kernel。段数按 `TARGET_PROGRAMS`（默认 256）自适应，超参只影响本 kernel 自身网格宽度。

16K 的本实例官方对照尚未跑，现有 +24.9% 是同一实例上关闭本 kernel 的短探针比值，不是官方比值；其余数字均有对应 run 目录与日志。

不改官方服务参数、模型、量化与 benchmark；属自研 kernel；未合并上游新分支。C500 未验证。
