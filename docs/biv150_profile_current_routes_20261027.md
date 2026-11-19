# 当前主线 profiling 准备：源码身份和窗口内真实选路

当前团队 main 为 `241220723a09061c182764f8f738e5ec01190367`。本轮实际尝试fetch超时；随后通过本人CJJ1008的原始GitHub API复核main完整SHA，使用本地已有同SHA对象新建此独立分支，不将失败fetch写成成功同步。生产 `vllm_fl/`、`csrc/` 与该版本逐字相同。这是 CJJ1008 本人独立分支 codex/cjj-profile-current-20261027 的诊断准备，承接共享PR62的019ce53/901b4a8工具改动；不更新原PR62。已在固定e767314运行，取得有效4K窗口；16K于预热阶段主动停止，见末节。旧 `0bf9` trace 不能作为当前热点证据。

main2412→K128 完整配对已结束并完成独立审计及owned cleanup：main4K5778.4583/16K4480.9883，candidate4K5698.2074(-1.3888%)/16K4515.4556(+0.7692%)，精度分别102/105及101/105，前后原始语义通过。当前没有可确认的双档端到端收益，保持PR47所在main2412。审计时原候选driver/server/Engine均消失，controller为terminal zombie，9031关闭、ixsmi确认空闲；启动前仍必须重复检查。不得因两臂之间空档或旧状态文件推断任务终止。

诊断维持官方服务/benchmark 的计算参数，唯一额外服务配置为 CUDA profiler，benchmark 额外 `--profile`。每档仍是三次短预热、一次完整相同负载预热、一次完整 profile；profile 数字均不能作为正式吞吐或 TTFT 成绩。私有 Triton/FlagGems/Inductor 缓存和逐 SHA 校验的 tuning snapshot 不变；清理函数 AST 与 main2412 相同，只清理由本任务进程身份确定的 owned 会话。

新增约束：

- driver 默认为 main2412，支持显式完整 `--production-baseline-commit`。启动前核对受测 HEAD、干净生产源码和该基线逐字一致。若团队主线前进，先 fetch、同步并重新检查生产基线，不能把本文件的旧 SHA 当成最新。
- 用真实 worker 报告的 Torch/C++ ABI，加生产 `build_fingerprint` 算法、实际默认环境、源码字节和 native SO 内容，核对所有 `BIV150_COMPILE_SEMANTICS`。driver 不导入 Torch。worker 和插桩文件的实际导入路径必须属于指定源码。
- main2412 要求八项安装成功，含 Q32 和 small-out lowering/large-linear；未来生产代码包含 K128 时再要求 K128。没有手工打开 Q32/K128 或关闭已验收 provider。
- 每个实际 profiler 窗口开始/结束只读取已有 Python 对象与计数器，记录 Q-fragment、Q32、可选 K128 的 manager 身份、active/binding/disabled、选择/回退/编译计数、已选 kernel hash，以及自然 C3 的 selected 计数。没有新增逐调用 hook、pre_run_hook、张量读取、设备查询或窗口边界以外的同步。
- 窗口内实际主路径（当前 Q32，若以后生产包含 K128 则 K128）selected 增量必须为正；自然 C3 selected 也须增加。manager 替换、策略失活、被永久禁用、证明记录错误、计数倒退或选路零增量均使 profile 无效。外层 K128 覆盖时允许 Q32 增量为零。
- 保留原 boundary/full Graph 克隆的真实输出、别名、API 和插桩开销校验；不能只凭 HTTP 200、安装成功或启动阶段一次 selected 报告热点。

本地 `python3 tools/check_biv150_profile_current_cpu.py` 已通过。它覆盖默认 provider 安装失败、源码/环境指纹错配、窗口内路径未使用、被禁用/替换/解绑、C3 未使用、K128 覆盖 Q32、bootstrap 原初始化/异常及 ABI 记录，还有私有缓存、完整预热顺序和清理 AST。全部是假 CPU fixture，不构成目标 GPU 可用性、真实 C3 闭包兼容、Graph 数值或当前热点结论。实际运行时若任一新约束失败，应保留失败原件并修复插桩，不得放宽生产 C3 的 source pin 或将失败 trace 当有效结果。

实际 GPU 诊断的下一步是在空闲且同步后的独立源码上运行 `tools/run_biv150_baseline_profile.py --help` 所示命令，显式传入受测完整 commit、最新生产基线、个人实例上的指定 native SO 和 tuning snapshot。默认先做 4K；取得有效窗口及原始 trace、编译 metadata、Graph 校验后，按真实耗时定位下一处优化，而不是按 shape 出现次数猜时间占比。


## 4K复核完成；停止重复16K，直接验证GQA

固定e767314/main2412的4K窗口通过：4246步、4个Graph样本逐元素一致；Q32窗口selected+11382、自然compute-order selected366→732，均active且无拒绝/回退。实际GQA hash仍641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff。

本地解包后3个原文件SHA全部重验，现有离线分析器通过，得到1988个插桩节点区间。新GQA每图采样区间合计中位22.696683ms，旧22.691785ms；MM新6.109044ms，旧6.109190ms。完整图boundary31.521263ms/full34.003229ms，插桩对比+7.874149%。这些不是原生算子延迟或全轮占比，不能将图内合计直接当整轮时间。旧完整4K/16K证据继续适用，不再重复整轮profiling作为算子实验前提。

21:57:50UTC对确核PID590512/start339826093/SID590512的本人driver发SIGTERM，取消剩余16K重复诊断；当时仅warm-16384-1，没有有效16K窗口。既有清理逻辑停止owned服务/Engine，collector继续正常落盘；待真实终态复查后再用GPU。4K诊断bench数字不作为正式成绩，不声称本轮完整两档通过。

证据见docs/evidence/current-main-profile-20261027/：4K原窗口archive1,887,075字节/SHA26c45fda4b8284fa27cf2b1feed22d324b3c23621e500eca2801487d1cbf27be，摘要/CSV及独立审计JSON。k128-final-independent.json(SHAa0063ce239d2aa1ce03b50aec1d1bacff33fadb8e306dbc32d9481385f93f20d)保存上一配对结果及初末语义，说明保持main的依据。

源码合同kv-sme-layout-source-contract.json(SHA75bdee28b8ebae1bf814802bfd9b1725c674b3c4986b69b30f974a6d70efe78e)包含实际SDK及SHA。优先可证伪实验为同一packedKV的3D指针view：把页内16×2两轴合并成32，让原frontend自然改变SME适用位，GPU地址/stride/容量保持不变；记录实际编译IR及独立CPU FP64数值。物理K/V分平面是后备假设，需要另审cache planner/cross-layer packing，当前未实施。适用位不能代替硬件指令、带宽或性能证明。
