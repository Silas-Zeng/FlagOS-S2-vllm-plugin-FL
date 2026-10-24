# BI-V150 插件编译缓存身份

PR24 压缩 gate 权重后，本实例仍命中较早的 BF16 AOT 图。新 wrapper 尚未被 trace，旧图记录的源码检查不能发现它，导致请求成功也无法证明模型正确。

本改动在 `PlatformFL.check_and_update_config` 中，通过 vLLM 的 `additional_config` 扩展编译缓存键。目标仅为 iluvatar 上符合 MiniCPM5-2B 几何的模型；指纹包含实际计算实现、启用状态、native SO 内容和 ABI。库路径解析与真正加载共用，日志目录、绝对路径、mtime 不进入指纹；其他团队配置保留。没有更改模型运算、官方 serve 参数或 benchmark。

CPU 验证涵盖本实例 `VllmConfig.compute_hash` 到 AOT 键的调用链、路径稳定、源码/SO/ABI 变化失效和 SupportsHash 的序列化。设备验证需确认首次生成的新图确有第4–37层34个 native gate 调用，重启实际复用同一新缓存且回答正确。`BIV150_COMPILE_SEMANTICS` 日志只报告缓存身份，不代替这两项检查。

旧实验资格修正和原件见[实验 PR #5](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/pull/5)。本改动是正确性及复现修复，不声明独立吞吐收益。设备结果在验证完成后补充。
