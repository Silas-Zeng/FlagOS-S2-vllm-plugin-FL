# PR122：服务实际 SME1 编译组的独立数值入口

这是以已共享 `8b7ea7a876f01e059f35fa7e2f266ba248af03ef` 为父的 tools-only 准备；没有新设备结果，没有改变生产/GPU 源码、原 observer、PINS、receipt 或精度门。原服务 smoke 保持 failed：163 个请求完成，但实际编译组不匹配旧 SME5 资格；19 个 ordinary host submit、0 个 Plan，不能将构造的 Plan fixture 当自然服务覆盖。

实际目标为 `b0fa4a52f87904075643190bf02047cdd59bd288aff4a1c1cd62a6416ed0976f`，cubin `4a9f93b84bd32f725d101ffb15f081f6b7345af149ba64abcc4e117369ef928e`，host launcher `4105509e003b73dd782a52edfb86254c751201e42485c09b757cfb1fa7ad9c59`。准入绑定 Root 已逐字重放的 71 项证据报告 `5f3f7f73…`、原 failed state、两份真实 worker ACK、原 bridge、七个缓存 child 和真实 launcher。该报告只证明证据链，不提前授予新数值资格。

## 最小执行差异

复用原 `run_biv150_dynamic_hidden_gather_helper.py` 的完整 execute 函数；七处精确替换可逆回原 `bc3ff184…`。helper 原 caller、CPU/GPU 索引生产者、单次 receipt 消费、四 case（65/9、2018/61 × ordinary/构造 Plan）各 A/B 两阶段、完整位比较、hidden 只读与 guard、OFF/FULL/capture/spec 拒绝均保留。原 reference SME0/direct-bit 报告仍只作为旧 fixture 的出处。

1. 将原七个 child 和 4105 launcher 复制到新私有缓存，只重定位 group index 路径。原 child 字节不改。
2. 用正常 JIT binder 对自然 `(18, 1)` 张量调用一次 `warmup`。不传 `use_sme`，不调用 kernel；严格要求自然选择 1、禁止 backend 编译和 override。不能命中原 b0fa 时停止，不生成一个新 debug 产物再充当旧对象。
3. 保留原四次 case warmup，并要求每次与 seed 为同一个 CompiledKernel 对象，实际 ABI/options/resources/六产物/loaded launcher 均等于服务原件。后续两种形状若冷选会是 SME5，这仅作为诊断记录；实际执行必须命中已载入的 SME1 对象。
4. 原四 case × AB 共 8 次实际 helper `_launch`。另用同一已载入对象复用原 bits fixture/checks，执行 2 种 shape × AB 共 4 次直接 kernel 调用，保存完整 hidden/indices/output、全输出 poison 和两侧 guard。helper 自行分配的输出没有注入外围 canary，不能与这项额外协议混淆。

共 5 次 warmup、8 次 helper 数值调用、4 次额外直接 guard 调用。不开 timing，不重测旧 SME5 成本。构造 Plan 绕过的是既有外围 fixture 的 make_plan，不修改原 receipt/execute_model/数学；不能据此声称模型 C3 或自然 Plan 已覆盖。Graph 仅检查原拒绝边界，没有 Graph 数值资格。

旧 cache 的标准 FileCacheManager、force/override、backend 和候选编译拒绝函数原样复用，仅在其独立函数 namespace 中替换期望 HASH。历史拒绝文本仍带 `actual805a` 标签，不改变它们的拒绝逻辑；新入口每次实际对象及导出产物明确绑定 b0fa/4a9f/4105。

## CPU 与设备入口

默认执行 `tools/run_biv150_hidden_gather_service_object_numeric.py` 只输出计划，不导入 Torch。CPU 检查入口是 `tools/check_biv150_hidden_gather_service_object_cpu.py`，接收 `--service-out`、`--review` 和 `--out`；用原件验证复制/拒绝/逆变换和原 oracle。加 `--ready-prefix` 与旧 `--compile-out --actual-review --bits-review` 后，实际走正常 execute 前置检查并复制私有缓存，在第一个 Torch import 处主动阻断；不使用伪 GPU Tensor、不运行设备。

设备由 Root 单独协调，仍需 closed-owner、端口/GPU 独占和干净冻结提交。参数沿原 helper：

```text
python -B -u tools/run_biv150_hidden_gather_service_object_numeric.py
  --run-helper --root-owned-exclusive --expected-source-commit <本工具冻结提交>
  --compile-out /workspace/experiments/minicpm-s2/cjj-dynamic-hidden-gather-compile-v1-20261029
  --actual-review <旧 bdb0fb89 编译报告>
  --bits-review <旧 9fe89661 完整 bits 报告>
  --production-out /workspace/experiments/minicpm-s2/cjj-native122-service-smoke-v1-20261030
  --production-review <Root 已重放的 5f3f7f73 证据报告>
  --out <新绝对路径，不能位于源码或旧证据内>
```

设备 Python 必须使用既有正确环境及原 overlay/PYTHONPATH；上面的换行只是参数展示，不能用本地 stdlib-only CPU 环境代替设备环境。`--timing` 不存在。原 failed smoke、完整 raw、三版审查失败/修正记录均留本地，不随本提交上传。输出仍为 `helper-summary.json` 与全部数值/guard 原件，Root 回收后独立重算再决定后续服务资格。

CPU 冻结检查已实际完成：`3d59ec8` 上58项通过，包含真实 execute 前置检查到首次 Torch import 的主动停止；未加载 Torch、未 warmup、未运行 GPU。此后仅把设备命令示例去掉 CPU 使用的 `-S`，确保保留正常 SDK site-packages；数值工具字节未变。初次开发中的 seam 计数断言将原字段出现次数误写为3（实际2），已在冻结前修正；没有设备尝试。
