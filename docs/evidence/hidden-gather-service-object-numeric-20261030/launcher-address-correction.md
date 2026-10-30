# d74 失败与 launcher 身份更正

原 d74 短数值结果保持失败：owner `226425 / 365006699` 已结束，21 原件共 255586 B，summary SHA `8c66900e31b2bba42f57b3fed97646a12a6dbb25bbe690daf293bcc04d54aaae`。自然 18/1 选择 SME1、完成一次 warmup，GPU 六产物与服务 b0fa/4a9f 逐字相同；随后 `actual private original launcher path` 拒绝，0 helper case、0 helper 数值调用，没有性能或精度结果。真实 CUDA context 已初始化；0 helper 调用不能解释成0全部CUDA活动。

## 原判断缺口

服务快照通过 `kernel._run.launch.__self__.__file__` 记录了 4105 文件；该文件属于可变的 Python 模块元数据，没有以 C 函数地址证明归属。旧 5f3 报告、旧 snapshot/PINS、原失败状态及 raw 均不改写。5f3 中“actual loaded handler”应收窄为“快照记录的模块文件已归档”，不能据此断言 gather 实际执行了 4105。

可复核的新证据：

- 4105 的 C 参数格式为 `iiiKKppOOOOOOOdOiiiiiOO`，不对应 gather 的三指针、两个整数、两个 constexpr；5124 为 `iiiKKppOOOOOOOOOiiOO`。
- 已保存的真实 SDK driver SHA `5023c317…` 根据服务原 ABI 生成的 C 源 SHA 为 `16b5948ce5a3d75dd077147571f2c544cc6d99e6c6890c988652ef26bb2c6f0f`，参数格式正是 5124。
- Root 新取的 SDK build.py SHA `d6d0c78e…` 以 `sha256(src + platform_key)` 选缓存。用 `x86_64,Linux,64bit,` 计算精确得到 H7PO 缓存目录。该 platform 字符串目前是由真实缓存键匹配推断；新设备入口仍必须读取实际 `platform_key()` 并重新匹配，不将推断当现场记录。
- 原服务 raw 和 d74 原件都存在 H7/5124，字节均为 `512414b6fbfae2135b62153619ef1f5919bd2fd07018058cd45035de1d83e361`。
- helper 的计数在 `_launch` 返回之后，分配/launch 不在 prelaunch 的异常捕获内，runner 也不捕获后重试。因此19条是成功 host-return，不能解释成其吞掉同步参数解析异常；仍不是 GPU 完成或数值正确性证明。

历史服务进程的 C 函数地址未保存，无法事后闭合其实际映射。使用同 SDK loader 的本地 Python 3.12 两个最小 C 扩展实验没有复现自动模块路径串写；不能声称已经查明具体串写机制。

## 最小修正与不变边界

新 tools-only 子提交使用 CPython `PyCFunction_GetFunction` 获取 builtin `launch` 的 C 地址，再用 `dladdr` 得到实际 DSO。它不调用该函数，不修改函数、不创建替代 launcher。先保存地址、DSO路径/SHA、原模块路径/SHA及实际 ELF，再要求：

1. 原 METH_VARARGS `launch`，真实 C 地址归属新私有 H7/5124 文件；未知文件、不同SHA仍拒绝。
2. 实际 driver/build 源精确匹配已读SDK，根据实际 `kernel.src` 生成的 C 源及现场 platform key 必须仍匹配 H7。原 ABI/options/resources/六GPU产物门照常执行。
3. 自然18/1 seed、私有原缓存、禁backend编译、原四case×AB/8 helper调用、额外4次完整guard、原receipt/Graph拒绝/数值门均不改。原 helper输出未注入canary与额外直接guard仍分开。

原服务 ACK 中的4105仍照原字节验证，作为历史模块标签保留；新复制的执行目标改为同份服务原件中、由真实ABI生成器支持的5124。这不是同时接受任意launcher，也不是以文件名代替函数地址。生产源码、GPU源码、原 observer 与原 actual_pins 文件全部不变。

新增 CPU 审查器 `tools/check_biv150_hidden_gather_launcher_address_cpu.py` 核全21原件、SDK/ABI/C/cache键、两个原host文件，并用 CPU builtin 验证修改模块文件标签不改变C地址归属；此测试不证明历史串写原因。设备数值仍未通过，Root 未授权新的 GPU 续跑；本提交只交本地审查，不更新服务资格。

CPU 审稿记录：系统 Python 的39项初检通过；指定 Python 3.12 的 `_json` 为内建模块、原本没有 `__file__`，首次 fixture 假设该属性存在而停止。后继仅令测试完整恢复“属性缺失”状态；绑定实现本来就允许旧模块路径缺失，没有改变设备身份或数值门。
