# N48 K 成对读取：直接完整步骤 ABBA

本入口只比较实际 `2ad51a…`（原 8 KiB）与 `ef732e…`（K 成对读取，仍为 8 KiB）。它从 PR114 `91f2ff6` 正常合入已共享 PR113 `dd48e9c`，复用其中已经实际跑过的直接 Native 完整步骤。PR113 的 6 KiB 性能负结果不作为本实验分母，也不授予本候选性能资格。

新增三个文件均为原适配入口的有限身份、标签及对齐范围调整；原 writer、reducer、CUDA、量化、fixture 和容差未修改。CPU 检查将整个执行正文逆回 namespace/arm 标签并去掉新增对齐观测后，与原 PR113 正文逐字比较；`timed/capture/reducer/poison/fingerprints/verify_inputs/step` 七段仍逐字检查。

- 两臂各自运行相同 dc5 writer、实际 Native producer 和原 reducer；ABBA 为 `old8k, kpair, kpair, old8k`。
- 固定 44769 个 12544 字节逻辑页；两个 case 为 live41/base17408 和 live42/base16896，Q/K/V 源 pitch2560。原三 phase 页预算、负 slot、padding、完整 cache、Graph-first、side stream、FP64 门及 48 类 ideal diagnostic 均沿用。
- 原 128 warm replays、64 timing replays、unroll8，以及完整 writer+producer+reducer 成本不变；两臂直接同输入比较。
- 四个 JIT 角色是 initial encoder、两个 append writer 与 reducer；两个 Native producer 分别验证实际库/编译/审查/consumer 身份。
- 新 consumer 绑定 `91f2ff6`、summary `a09ed7e1…`、manifest `d9b75246…` 和 Root 独立复跑报告 `ac96a073…`。全部 175 个原件、59,894,428 字节按原门重验。它只给此前 29 页、maxctx193、live41/42、两种 pitch 的组件资格。
- 两臂在首次 encoder/producer 前记录并核验整个 cache 的 8 字节对齐。整 cache 4B 非 8B、scalar fallback 的设备资格仍未覆盖；不新增该测试范围。

默认运行只输出 CPU plan。`--preflight-cpu` 对实际准入原件做原有校验；只有显式 `--gpu --root-owned-exclusive` 和全部实际参数才能运行设备。新完整步骤数值、Graph/side、性能、42 层容量、模型精度及正式吞吐仍待 Root 串行验证。旧 `performance_acceptance` 是原 driver 的初始化字段；应根据本次实际 ABBA 时间判断收益，不将该字段单独解释成自动验收门。

CPU source/ABI 检查及 8 个拒绝用例已过；真实准入 CPU preflight 已过。精选摘要见 `docs/evidence/native114-direct-fullstep-cpu-20261029/selected-results.json`。完整原件、完整 Root 报告与完整 preflight 输出仅保存在本地；不向 GitHub 新增数组、cache、日志或二进制。Root 后续将原件复制到个人实例的独立 replay 目录，保留 `out/`、`exports/` 和 download manifest 的原结构。

入口：

```sh
python3 -B -W error tools/check_biv150_native_k8_compact_n48_kpair_fullstep_cpu.py
python3 -B -W error tools/probe_biv150_native_k8_compact_n48_kpair_fullstep.py
```

精确本地 preflight argv 与不可自动启动的远端模板由本机 `/private/tmp/cjj-agent-native114-direct-fullstep-launch-arguments-20261029.json` 保存；包含真实收据 SHA 和远端尚需选择的独占路径。该文件不自动启动任何任务。
