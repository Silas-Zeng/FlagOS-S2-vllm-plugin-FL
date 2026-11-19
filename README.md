# MiniCPM5-2B 框架优化协作

沐曦 C500 当前迁移分支采用 Triton 计算算子，已移除新增的原生 Prefill GEMM
和 GQA Decode 扩展。实现范围与验证记录见 [C500 Triton 算子实现](docs/metax_triton_migration.md)。
下文 BI-V150 的历史记录与 C500 的实现、成绩分别维护。

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
| [#22](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/22) | computed H2D 与计算顺序的兼容组合 |
| [#23](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/23) | prefill Attention 的 Q fragment 跨 K 循环保持 |

#1/#2 的冲突采用已经组合验证过的 runner，保留两个 helper 和各自回退路径。代码合并提交为 `0f2b925588114d47c478707afa41e58c769c2ec4`；其全部 425 个文件与既有组件的预期组合一致，runner SHA256 为 `6f918ae6e2f32a1244bb5ac7411a5cffdd5071b1636e899e61ab6f0f20df2141`。本次复用已有验证，未重复运行测试。

## 最新 BI-V150 基线

`PR15+PR21+Q fragment` 已完成原口径官方 4K/16K 各四轮，跳过首轮并使用第 2–4 轮均值：

| 负载 | Total tok/s | Mean TTFT |
|---|---:|---:|
| 4K | 5313.59 | 4182.55 ms |
| 16K | 3764.67 | 137191.04 ms |

相对此前的 PR15+PR21，4K 为 +3.80%，16K 为 +9.56%。完整 Math500 Level 3 为 100/105（95.24%），105 请求成功、无空最终答案；2 题达到原输出上限。Q fragment 在实际服务中选中且 fallback 为 0，方法见 [prefill Q fragment](docs/07-prefill-q-fragment-biv150.md)。

两次官方吞吐使用相同起始 DB 种子，但候选 KV 为 `521840`，旧基线为 `521936`；因此不把差异写成严格等 KV 的单项因果拆分。仅验证天数 BI-V150，沐曦－曦云 C500 的适用性由对应平台另行验证。两个 KV 工具需要显式调用，合并不表示已启用 KV offload。

## 验证与协作

- [实验总表](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/main/team/experiments.md)：尝试、改动、测法、结果及是否保留；按硬件区分。
- [方法与原始证据](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/tree/main/team)：同一方法的短记集中查阅。
- [FlagGems](https://github.com/Silas-Zeng/FlagGems)：固定比赛版本，实际库内算子改动另开 PR。
- [上游原始说明](README.upstream.md)：保留原文与许可证。

后续实现按方法提 PR，中文说明依赖、适用输入、验证及收益边界。已验证且未变的部分复用结论；新冲突或新增行为针对性核验。小输入和大输入可分别评估适用策略，不改变官方评测参数。失败方法进入实验记录，不因每轮实验重复开 PR。旧快照及历史证据继续保留。

## 当前 C500 默认 Attention 路径

当前默认服务采用经 A5 两次测量确认的原框架 Attention 封装路径；新增 Triton Decode、Prefill gather、双请求拆分和直接写入优化不在默认服务路径启用。HND KV 布局、RoPE/KV 融合、GEMM/MLP、残差归一化、LM Head、采样及必要的 CPU 元数据优化保留。

C500 同卡两次内部测量均值：4K/1K 为 **12276.94 total tok/s**，16K/1K 为 **10757.125 total tok/s**；相对原完整组合单轮参照分别提高约 2.33% 和 3.98%。官方参数的 MATH-500 Level 3 完整评测为 **103/105（98.10%）**，无 API 错误、空响应或输出上限停止。性能均值为内部两次测量，不代替官网默认轮数的正式性能成绩。

优化版阈值 Top-P 与官方排序截断在并列 logits 边界上的已知语义差异仍然存在；任务准确率达标不证明两者完全等价。
