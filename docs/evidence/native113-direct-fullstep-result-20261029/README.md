# PR113：6 KiB stage 未带来完整步骤收益

独立复核确认：同输入、同 writer/reducer 的 old8k/new6k 直接 ABBA 中，6 KiB 候选在两档
N48/S4 长上下文上均更慢。数值与身份检查通过，性能没有收益，是两个不同结论。

| case | live / base context | 旧 8 KiB 完整 step | 新 6 KiB 完整 step | old/new | 新增时延 |
|---|---:|---:|---:|---:|---:|
| valid41 | 41 / 17408 | 1.315694 ms | 1.352761 ms | 0.972600 | +2.82% |
| valid42 | 42 / 16896 | 1.282474 ms | 1.343832 ms | 0.954341 | +4.78% |

四条原 event 计时按每档 `old,new,new,old` 重新计算，host wall 结果也一致。
每档的选定计时值保存在 `selected-results.json`。原记录的 `performance_acceptance=false`
并非这里的自动性能拒绝门；负结论依据这次直接计时及两个小于 1 的比值。
不能使用旧 TL48 比值跨运行相减，也不能把这两个 case 当作官方 4K/16K 整模型成绩。

受测源码是 82db，实际库为旧 2ad / 新 5cf。两臂使用相同 44769 个逻辑页、12544 B
页布局、Q/K/V source pitch2560；预热 128、计时 64、unroll8；step 包含 append、
producer、reducer。初始 encode/source packing 成本依照原流程单列。

本地独立检查重 hash 全部 114 原件 / 94,678,536 B，复核 36 个完整快照 / 144 个原
数值门，重建完整初始 cold/packed 数据及所有 phase cache digests；Graph-first、side、
post-timing bitwise、四个实际 JIT 角色及两个 Native SO 身份均通过。没有改变原数学、
容差、时钟或理想 partial 的诊断定位。新运行实际 `GPU_executed=true`；旧 112 的两个
记录勘误仅保留为历史前置，没有移植成新运行事实。

此前 runtime API 的每 SM active-block capacity 从 16 增至 21，没有建立实际驻留或
加速结论。本次同输入完整步骤没有把该容量上限转化为收益，不应据此将 6 KiB 候选按
性能优化合入。没有测试该候选的 42 层容量、模型精度、共享 QKV 输入或正式服务吞吐。

独立报告 SHA：`3b81df7251deeb08a7efdc7cab75c70cea0296aad65794b555890ef2776d2fe1`。
复核脚本 SHA：`50cbbafd6a7c9f27e309bc3a4b472036d59be0a94771aaf8c67fdc83c56a574d`。
准备阶段的逆变换检查失败、实际身份 manifest 键名错误及其修正均保留本地记录；后者
发生在数组计算前，只有身份键名修正。完整数组、cache、日志、二进制及原始报告均保留
本地；本提交只有精选文本汇总，不包含或间接上传被拒绝共享的原始内容。
