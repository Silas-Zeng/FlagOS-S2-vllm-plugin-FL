# 服务实际 SME1 gather：独立数值复核

受测源码 `73470bbcc28f63cba715a64b94766fbd550301b4` 的独立组件数值复核通过。完整 90 件原件（20,747,155 B）、归档与执行端文件索引均逐 SHA 核验；摘要和审计摘要见 [selected-result.json](selected-result.json)。本目录仅共享精选结果，不含数组、日志、二进制或完整审计。

候选的 5 次 warmup（含自然 `18/1` 首次绑定、该 seed 不 launch）随后执行 8 次原 helper 和 4 次独立 guard gather。两种几何 `65/9`、`2018/61` 各覆盖 ordinary／构造 Plan 的 A、B 两阶段；8 份 helper 输出共 573,440 个 uint16 与独立逐行 oracle、保存的 index_select 控制输出逐位相同。另 4 次 guard 协议共 286,720 个输出词通过全部 7 项原始完整 guard、poison、输入只读检查，B 阶段改变数据和索引且复用物理背衬。计数仅针对候选 gather，不表示其他框架／控制 CUDA 操作为零。

实际 C 函数地址经 `PyCFunction_GetFunction → dladdr` 绑定至私有 `H7/512414b6…` ELF，偏移 5216 精确对应本地 `launch` 符号。seed 加四组对象记录的地址和 DSO 身份一致；SDK driver/build 两份源 SHA、ABI 生成 C、现场 platform key 与 H7 缓存键一致。GPU 组为 `b0fa4a52…`／cubin `4a9f93b8…`，7 个缓存子文件保持原字节；实际资源为 6 regs、0 runtime spills、0 shared、4 warps、1 stage、SME1。初次 backend 编译及后续 candidate 缓存失配编译仍被原门禁止。现场 cold 预测为 SME5 的后续几何，实际复用的仍是同一 SME1 对象，没有强制 use_sme。

## 证据边界

- 原 helper 的 8 次 hidden-after 有冻结执行源中的完整同步读回相等断言，但没有单独保存 after 数组，不能称为离线重放。新增 4 次独立 guard 则保存并完整核验了 hidden、indices、output 的 after 字节；该协议不等于给原 helper 输出分配注入 canary。
- Plan 是组件构造的确切类型和原 staging 路径，不证明真实服务 make_plan/C3 或自然 Plan 命中。Graph 仅验证外部 prepare 在 capture 中提前拒绝；不授 Graph 数值、replay、side-stream、模型精度、服务或吞吐资格。
- empty CUDA Graph、libentry fallback 以及三条 TemporaryDirectory ResourceWarning 均在原日志及私有审计中保留，没有将告警删除或改成测试成功的理由。
- 旧 d74 路径失败与 `5f3` 报告均保留。本次新进程取得 C 地址，不能反推已经退出的旧服务实际 C 归属；旧 `4105` 只确定是当时保存的模块路径标签。

下一步仅建议修正 `tools/biv150_hidden_gather_snapshot.py::compiled_record` 的 launcher 归属观察：使用只读 C 地址／dladdr，将模块标签单列；在 observer、确切 SME1 pins 和新 source bridge 中绑定此次资格。先重新取得真实服务 smoke 的 C 地址和 ordinary 前后计数；host submit 仍不等于 GPU 完成，Plan 缺口继续保留。新服务若冷编译字节不同，先保存 ACK 再独立审查，不能扩大白名单、强制 SME 或沿用旧失败为成功。当前未实现该后继、未启动新服务。
