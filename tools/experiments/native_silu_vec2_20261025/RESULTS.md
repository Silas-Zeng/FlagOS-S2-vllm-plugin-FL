# SiLU vec2 实验结论：正确但无净收益

相邻两列用 `int2/float2` 读取并打包写回 BF16，尝试减少地址、访存指令与 CTA 数量。原量化、GEMM、两次 BF16 舍入和计算顺序保留。

BI-V150，随机 `M2048/N12288/K2048`，私有重编测试库，32 次调用的 Graph ABBA。15 个数值/Graph case 的逐位、finite、guard 通过，独立 CPU 参考、inputs/RNG 保持通过。

| 范围 | 原实现 → vec2 Event（ms） | 耗时变化 | wall 变化 |
|---|---:|---:|---:|
| 隔离 SiLU 尾处理 | 0.216360 → 0.231472 | +6.9849% | +6.5243% |
| 完整 gate | 0.712861 → 0.730258 | +2.4404% | +2.6352% |

结论：停止此候选，实验 PR 关闭留痕；减少访存指令的设计并未转为实测收益，具体瓶颈未进一步归因。未接模型或替换生产 SO，不代表整模准确率或官方吞吐结果。

[实验短记](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/team/runs/bi-v150/20261025T143942955934Z-native-silu-vec2-component.md) · [原始结果](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/experiments/framework/20261025T143942955934Z-native-silu-vec2-component/result.json)（随[实验 PR5](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/pull/5)本批同步）。

本目录原16文件来自受测包 `7d9ed065`，15项 `asset-pins.json` 与相对导入保持原字节；README 是冻结时的实验说明，本页补充终态。源码引用基点为 `0915f01d`。入口仍为 `python3 entry.py`，由唯一设备负责人安排；生产 `csrc/` 不改。
