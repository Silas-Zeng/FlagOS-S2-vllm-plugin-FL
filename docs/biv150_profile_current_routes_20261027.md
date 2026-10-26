# 当前主线 profiling 准备：源码身份和窗口内真实选路

当前团队 main 为 `241220723a09061c182764f8f738e5ec01190367`，已实际 fetch 并合入此独立分支。生产 `vllm_fl/`、`csrc/` 与该版本逐字相同。这是 CJJ1008 本人独立分支 codex/cjj-profile-current-20261027 的诊断准备，承接共享PR62的019ce53/901b4a8工具改动；不更新原PR62。当前尚未启动新GPU profiling。旧 `0bf9` trace 不能作为当前热点证据。

main2412→K128 完整配对已结束并完成独立审计及owned cleanup：main4K5778.4583/16K4480.9883，candidate4K5698.2074(-1.3888%)/16K4515.4556(+0.7692%)，精度分别102/105及101/105，前后原始语义通过。当前没有可确认的双档端到端收益，保持PR47所在main2412。审计时原候选driver/server/Engine均消失，controller为terminal zombie，9031关闭、ixsmi确认空闲；启动前仍必须重复检查。不得因两臂之间空档或旧状态文件推断任务终止。

诊断维持官方服务/benchmark 的计算参数，唯一额外服务配置为 CUDA profiler，benchmark 额外 `--profile`。每档仍是三次短预热、一次完整相同负载预热、一次完整 profile；profile 数字均不能作为正式吞吐或 TTFT 成绩。私有 Triton/FlagGems/Inductor 缓存和逐 SHA 校验的 tuning snapshot 不变；清理函数 AST 与 main2412 相同，只清理由本任务进程身份确定的 owned 会话。

新增约束：

- driver 默认为 main2412，支持显式完整 `--production-baseline-commit`。启动前核对受测 HEAD、干净生产源码和该基线逐字一致。若团队主线前进（包括 PR63 合入），先 fetch、同步并重新检查生产基线，不能把本文件的旧 SHA 当成最新。
- 用真实 worker 报告的 Torch/C++ ABI，加生产 `build_fingerprint` 算法、实际默认环境、源码字节和 native SO 内容，核对所有 `BIV150_COMPILE_SEMANTICS`。driver 不导入 Torch。worker 和插桩文件的实际导入路径必须属于指定源码。
- main2412 要求八项安装成功，含 Q32 和 small-out lowering/large-linear；未来生产代码包含 K128 时再要求 K128。没有手工打开 Q32/K128 或关闭已验收 provider。
- 每个实际 profiler 窗口开始/结束只读取已有 Python 对象与计数器，记录 Q-fragment、Q32、可选 K128 的 manager 身份、active/binding/disabled、选择/回退/编译计数、已选 kernel hash，以及自然 C3 的 selected 计数。没有新增逐调用 hook、pre_run_hook、张量读取、设备查询或窗口边界以外的同步。
- 窗口内实际主路径（当前 Q32，若以后生产包含 K128 则 K128）selected 增量必须为正；自然 C3 selected 也须增加。manager 替换、策略失活、被永久禁用、证明记录错误、计数倒退或选路零增量均使 profile 无效。外层 K128 覆盖时允许 Q32 增量为零。
- 保留原 boundary/full Graph 克隆的真实输出、别名、API 和插桩开销校验；不能只凭 HTTP 200、安装成功或启动阶段一次 selected 报告热点。

本地 `python3 tools/check_biv150_profile_current_cpu.py` 已通过。它覆盖默认 provider 安装失败、源码/环境指纹错配、窗口内路径未使用、被禁用/替换/解绑、C3 未使用、K128 覆盖 Q32、bootstrap 原初始化/异常及 ABI 记录，还有私有缓存、完整预热顺序和清理 AST。全部是假 CPU fixture，不构成目标 GPU 可用性、真实 C3 闭包兼容、Graph 数值或当前热点结论。实际运行时若任一新约束失败，应保留失败原件并修复插桩，不得放宽生产 C3 的 source pin 或将失败 trace 当有效结果。

实际 GPU 诊断的下一步是在空闲且同步后的独立源码上运行 `tools/run_biv150_baseline_profile.py --help` 所示命令，显式传入受测完整 commit、最新生产基线、个人实例上的指定 native SO 和 tuning snapshot。默认先做 4K；取得有效窗口及原始 trace、编译 metadata、Graph 校验后，按真实耗时定位下一处优化，而不是按 shape 出现次数猜时间占比。
