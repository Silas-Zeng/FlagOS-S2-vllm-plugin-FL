# MiniCPM5-2B 框架优化协作

本仓库的主线以比赛指定的原始 vllm-plugin-FL 框架为基点：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。本项目新增实现按优化方法通过 PR 审阅，未经合并的优化不属于 main。

- [查看方法 PR](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pulls)：每项说明尝试、改动、测试、结果及是否保留。
- [实验总入口](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput)：BI-V150 与沐曦 C500 分别记录，原始证据通过链接查阅。
- [FlagGems](https://github.com/Silas-Zeng/FlagGems)：固定比赛版本，实际算子改动另开 PR。
- [上游原始说明](README.upstream.md)：保留原文与许可证。

本次整理保留此前导入优化快照的 Git 历史及归档分支；主线源码恢复为上述原始基点。历史组合测试结果只对应其精确源码，不能自动归给单个 PR 或原始框架。团队维护的 README、PR 和方法记录使用中文，英文专有名词和代码标识保持原样。
