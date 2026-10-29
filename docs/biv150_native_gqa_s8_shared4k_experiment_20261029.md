# BI-V150 BF16 Native GQA S8 shared4k 实验

本实验在 PR105 基线 `b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd` 上加入 N30/S8 BF16 Native GQA 路径，并以 shared4k 变体复用 K/V 的 4 KiB shared-memory stage。构建与组件门通过只证明该组件在被测形状可运行，不等同于官方准确率资格。

官方 16K 四轮暖均为 4908.9133，相对 PR105 观测约 +1.577875%；该轮 KV 条件多 48，不能视为严格同条件因果结果。官方 4K 首暖为 6204.95 对 6441.35，约 -3.670038%，按首暖负向停止规则未继续；4K KV 条件也不同。当前没有本候选的官方准确率结果。

因此此 PR 仅用于保留可检索的实验代码和负向结论，创建后应关闭且不合并当前 main。当前 main 的源码 pins 与该实验源冲突，且 4K 结果为负。保留源码和原始证据，不以组件 10 项通过替代官方准确率门。

背景与证据索引：实验总表 [PR5](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/pull/5)；方法记录 [native-gqa-s8.md](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/team/methods/native-gqa-s8.md)。两档官方结果与组件原件见本地归档 `archive/native-gqa-s8-20261029/`。

构建边界：仓库中的 `tools/build_biv150_native_gqa_s8.py` 保留原S8构建入口，并不构建shared4k变体。受测shared4k库由组件入口单独构建；精确CPP/CU、编译器、编译参数、模块名和成品SHA见 [构建原始回执](../tools/native_gqa_s8_shared4k_receipt.json)。完整component builder随实验库102212原件保存。本PR保存受测实现，不承诺一键生产部署；不能只改成品SHA沿用资格。
