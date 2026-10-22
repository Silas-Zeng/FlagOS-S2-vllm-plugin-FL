# MiniCPM5-2B 框架优化协作

本仓库以比赛指定的 vllm-plugin-FL `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` 为起点。新增代码与维护文档通过 PR 提交，按依赖合并；官方上游及原公开 fork 保持只读。

## 当前已合并方法

2026-10-22 已按 **#5 → #6 → #7 → #8 → #9 → #10 → #1 → #2 → #3 → #4** 合并：

| PR | 内容 |
|---|---|
| [#5](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/5) | 实验配置与诊断开关 |
| [#6](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/6) | prefill Attention 启动策略 |
| [#7](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/7) | 直方图 zoom top-p 过滤 |
| [#8](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/8) | SQLite 目标表反射 |
| [#9](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/9) | 3D decode 的 w2s1 启动 |
| [#10](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/10) | Attention 内部天然单 token 前缀拆分 |
| [#1](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/1) | CPU metadata 复用 |
| [#2](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/2) | 异步 token 的整数 gather/scatter 更新 |
| [#3](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/3) | 显式调用的 KV 连续页复制工具 |
| [#4](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/4) | KV 正等差二维 SAVE 规划与 SDK 桥 |

#1/#2 的冲突采用已经组合验证过的 runner，保留两个 helper 和各自回退路径。代码合并提交为 `0f2b925588114d47c478707afa41e58c769c2ec4`；其全部 425 个文件与既有组件的预期组合一致，runner SHA256 为 `6f918ae6e2f32a1244bb5ac7411a5cffdd5071b1636e899e61ab6f0f20df2141`。本次复用已有验证，未重复运行测试。

## 验证与协作

当前实测来自天数 BI-V150。CPU 组合已有局部正确性与短服务验证，尚无该完整组合的独立整模型准确率结论；历史 SQL+C3 的 103/105 只对应当时源码。两个 KV 工具需要显式调用，合并不表示已启用 KV offload。沐曦－曦云 C500 的适用性由对应平台另行验证。

- [实验总表](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/main/team/experiments.md)：尝试、改动、测法、结果及是否保留；按硬件区分。
- [方法与原始证据](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/tree/main/team)：同一方法的短记集中查阅。
- [FlagGems](https://github.com/Silas-Zeng/FlagGems)：固定比赛版本，实际库内算子改动另开 PR。
- [上游原始说明](README.upstream.md)：保留原文与许可证。

后续实现按方法提 PR，中文说明依赖、适用输入、验证及收益边界。已验证且未变的部分复用结论；新冲突或新增行为针对性核验。小输入和大输入可分别评估适用策略，不改变官方评测参数。失败方法进入实验记录，不因每轮实验重复开 PR。旧快照及历史证据继续保留。
