# PR112 N48/S4 实际编译对象证据

冻结源码 `83b6c8c9282445219782ee1377680778a2aa8d3a` 编出的实际 SO
`2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`
通过本次独立 CPU 静态审查。它的实际 GPU 机器码与 PR110 实际 `7892a6aa…`
逐字相同；宿主入口限制为 N48/S4、Q pitch 2048 或 2560，使用独立
`cjj_native_k8_compact_n48_s4_probe` namespace。此次没有运行 GPU。

`actual-so-review.json` SHA256：
`30c4c7daf2ccf3839bfabc9936f0a845400d26242fe93cc95bec99d270e8f398`。
`review-pending-root.json` 的六个批准字段及 `manual_Root_approval` **全部保持 false**。
它只记录实际 S4 资源与独立报告，不能作为已批准的运行 receipt。
没有授予 N48 数值、Graph、side stream、writer、fullstep、42 层容量、模型精度、服务或吞吐资格。

Root 随后独立重放，得到逐字相同的 `30c4c7da…` 报告和 `46ea96cb…` pending receipt，
另存 `root-approved-consumer-review.json`，SHA256
`14c8f26fa0d7afb7d00d480aad9ec5cf84f490d11cca8da8cb5f6a959a114a50`。
该独立副本将六个审核字段和 `manual_Root_approval` 置 true，仅批准运行 N48/S4、两个 Q pitch
的有界 consumer。原模板与 pending receipt 原样保留；所有历史 `source_proof` 字段均未回填。
Root 的批准仍不等于上述任何设备数值、Graph 或服务资格。

本次重 hash 33 个原件、1,412,002 字节。编译 owner 为 PID 61412 / start ticks
355787063，回收时同 start、状态 Z。summary completed/compile_passed 为 true、errors 为空；
前端的 attention/encoder/codec/reducer/input-initialization 计数均为零，Torch CUDA
initialized 前后为 false。这不是“没有任何 CUDA API 调用”的证明。

审查绑定了实际 SO 的全部 2,764 条解码指令，SO 中完整 GPU ELF 与保存的编译对象相同。
两份 GPU `.text` 与实际 7892 相同；全部 kernel 元数据（包含 `.args` 参数 ABI）、实际
IR 函数体、attribute groups、编号 metadata 仅发生符号重命名。S4 共 1,339 条指令，codec
86 条。保留的 unused S2 也作了字节核对，但未授予 S2 admission 或任何运行资格。

| 实际 S4 ELF 字段 | 值 |
| --- | ---: |
| VGPR / SGPR | 100 / 66 |
| shared / private bytes | 8192 / 0 |
| VGPR spill / SGPR spill | 0 / 0 |
| warp size | 64 |
| kernel argument bytes / alignment | 144 / 8 |

这些是 ELF 元数据，不是运行后资源观测或 occupancy/性能测量。完整实际指令、ABI 和 IR
函数体相同后，复用 7892 已审的 S4 CFG：K load/store + barrier 在 V SME issue 之前；
QK 和 P softmax/packing 在 V wait 之前；wait 支配 V 尾页清零及 PV。部分 rho exp、running-L
FMA 仍在 wait 后面，不能声称所有 softmax 都被提前。

实际地址保持 K 页 12544B、V 页 6272 BF16，V 参数由宿主 base+4352B 加一次；scale 为完整
页 0 基址、页 stride 3136 F32，内核内部 +1024 F32 一次，SME 行宽 512B。零长度行的
producer 早退，O/M/L 保留 poison；最终零输出由另行执行的 reducer 处理。源码 CPU 核对
枚举了 393216 个 partial 与 3072 个 scalar 存储位置，未发现 N32 行上限或输出重叠。
两组 N32 参考保留全部 48 行原始全局身份，不重新生成第二组局部行数据。

原始 `ixobjdump --IR` 的长 producer 签名在固定宽度处换行/截断，符号名变短后截断位置
变化，甚至保留了断开的 `noundef` 显示。没有修改原输出或声称完整打印文本可重编译。
参数 ABI 检查使用完整实际 ELF 元数据；函数体、attributes 和 metadata 严格比较。
继承 CFG/IR 报告内的行号仍指向 7892 原件，新对象位置在 `functions` 数组中。
宿主限制是精确已编译源码、构建记录及二进制 guard 文本核对，没有做 x86 反汇编或实际
TORCH_CHECK 执行。独立生成的 `.ll/.s` 也完整保存并重 hash，主要结论来自实际 SO 导出。

`source-review.json` 和 `source_review_original_local.py` 是编译前的原始本地源审核；
报告当时的 candidate SO 字段为 null，不回填或改写旧报告。该脚本保留当时的本机绝对路径。
`existing-cpu-check.json` 保存六个 fixture/拒绝条件/物理页预算的 CPU 复跑结果，不是设备数值测试。
原源码“first draft S2/future R32”注释是为严格逆变换保留的非执行历史文字，不能用于判断当前 admission。

完整原包 `compile-original-records.tar.gz` 为 354868B，SHA256
`a7bc3cdf7025de4a3969be20d8d8f84c947c652c20f85c6aab658f24ab33f876`，
包含 33 个原件及完整 download manifest；34 个成员逐个记录于 `archive-index.json`。
同一原包持久保存在工作区 `repos/experiment-raw-archives/2026-10-29/`。
每个共享文件小于 2 MiB。参考 7892 原包、原始审查和 helper 使用已存在的 Git 对象，
精确 commit/path/SHA 见 `existing-dependencies.json`，没有重复打包。

从仓库根目录进行轻量核包（不重新审机器码）：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native112-n48-compile-20261029/replay.py
```

需要完整 CPU 重放时使用新的输出目录；入口读取本地已有依赖 Git 对象，不访问网络，
不 import Torch/NumPy/Triton，不编译、不加载 SO，不启动 GPU：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native112-n48-compile-20261029/replay.py --run --out /private/tmp/native112-new-cpu-replay
```

完整重放必须产出与本包相同的报告及全 false 待批准模板。若缺少依赖 commit，先按团队
正常 Git 流程取得 PR110 的证据提交 `b54eb6b7edd64f0704443fa8e0c64052126116eb`；入口不会
自动 fetch、换用其他库或放宽检查。重放入口不签发或修改 Root 批准，后续设备结果应另外保存。
