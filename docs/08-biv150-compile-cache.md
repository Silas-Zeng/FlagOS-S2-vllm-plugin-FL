# BI-V150 插件编译缓存身份

PR24 压缩 gate 权重后，本实例仍命中较早的 BF16 AOT 图。新 wrapper 尚未被 trace，旧图记录的源码检查不能发现它，导致请求成功也无法证明模型正确。

本改动在 `PlatformFL.check_and_update_config` 中，通过 vLLM 的 `additional_config` 扩展编译缓存键。目标仅为 iluvatar 上符合 MiniCPM5-2B 几何的模型；指纹包含实际计算实现、启用状态、native SO 内容和 ABI。库路径解析与真正加载共用，日志目录、绝对路径、mtime 不进入指纹；其他团队配置保留。没有更改模型运算、官方 serve 参数或 benchmark。

45 项 CPU 验证涵盖本实例 `VllmConfig.compute_hash` 到 AOT 键的调用链、路径稳定、源码/SO/ABI 变化失效和 SupportsHash 的序列化，并经独立复核。PR25 原数值实现不变，新构建 SO 在 BI-V150 上通过 9 项 gate 与 40 项 cached dispatch 检查。

2026-10-24 默认共享缓存验证：首次生成的新图含第4–37层共34个 native gate 调用；随后同源码重启实际加载同一 AOT 文件，4 个语义请求全部正确，最长输入4908 tokens。受测源码为 `7596cc0eb14fdbfc56e42eb1cd68d0f37b890d93`，SO SHA256 为 `14f4457a82388a634e9d381190251223dc925d278753a27011e29999e42dd8fd`。`BIV150_COMPILE_SEMANTICS` 日志只报告缓存身份，不代替真实图和回答检查。

首次诊断的模型答案为85，但检查器未识别带闭合 `<think>` 的返回格式而退出。原失败保留；修正检查器后复用已完成的49项组件检查和首启证据，只重启一次服务。成功记录 `20261024T141536Z-pr25-cache-restart-recheck` 是缓存语义诊断，不是官方准确率或吞吐成绩；其原始封包 SHA256 为 `f2e8093fae0d2228292b674744372303792c2cabb072e73dc0f76de5f20eaee3`。

旧实验资格修正和设备原件汇总见[实验 PR #5](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/pull/5)。PR24 隔离缓存的完整准确率为100/105，不能替代本 PR25 组合的完整准确率。本改动是正确性及复现修复，不声明独立吞吐收益。
