# PR113：直接 2ad old8k / 5cf new6k 完整步骤 ABBA（待新 consumer）

CPU 适配与实际准入检查已完成。113 consumer 原件 175 件 / 59716288 B 已完成
独立数组复核，Root 重放报告 SHA `eabf9148…19d25b9` 与作者逐字相同，六个实际 pin
已绑定；证据见 `docs/evidence/native113-direct-native-abba-admission-20261029/`。
这只准入下一独占完整步骤实验，完整步骤和模型结果尚未产生；不能借用 493e/c079 的候选资格。

两臂均为实际 Native N48/S4：旧库 `2ad51a98…b837b1` 使用 8192 B shared，新库
`5cf698e8…521bed` 使用 6144 B。它们使用同一批 44769 页、同一组输入/物理页映射、Q/K/V
source pitch2560、同 dc5 原始 encoder/append 和原 S4 reducer，按
`old8k,new6k,new6k,old8k` 计时。valid41×17408 / valid42×16896、三 phase 的尾/零/padding、
独立 cache、原 .005 数值门和完整原件留存全部保持。每次完整 step 是 append+producer+reducer；
初始 encoder/source packing 成本仍单列。warm128、timing64、unroll8 和 GPU event/host
wall 两种时钟不变。结果直接给本次 old8k/new6k 比值，不与旧 TL48 比例跨运行相减。

`biv150_native_k8_compact_n48_shared6k_fullstep_source.py` 在原 112 已生成 run 上仅做六处
有计数与逆变换的身份/记录改动及标签改名：加载两份实际 SO、替换 producer、移除没有
执行的 TL 模块、改为四角色 JIT 检查、修正成功执行标志。七段 `timed / capture /
reducer / poison / fingerprints / verify_inputs / step` 在标签逆变换后逐字相同。
实际待执行文本 SHA `1456450bb06dd52452160a9c5c89c81a7ccd0043044558b490db95997176daed`。

TL producer 不再执行，因此它及仅为 TL 服务的 production reader/quant helper import
检查不适用。dc5 writer 自带的 RN32/RNE quantizer 完整保留，由原冻结 writer SHA 和实际
writer JIT 身份绑定。仍执行的 JIT 角色是 initial_encoder、old8k_append、new6k_append、
reducer；原 typed writer ABI、函数对象身份、runtime n_regs>0 / n_spills==0 条件逐字保留。
两个 Native producer 通过各自真实 SO、compile、Root review 和 consumer 资格准入，
不伪造它们的 Triton handle。

旧 2ad 的准入仍调用原 consumer gate，绑定 493e/c079 和实际编译审查，并另核原完整
步骤 5a6a/2d8d。原记录的 GPU_executed plan 标记和 TL SGPR26 两项勘误一起保留，不改原件。
新 consumer gate 重用原全部数组/计数/Graph/side/epoch/阈值检查，仅参数化真实 owner
及原件计数，另外核 c91 保存的完整实际执行文本。未知 pin 的拒绝发生在框架 import 之前。

新运行的 default CPU plan 仍为 GPU_executed=false；设备运行但尚未完成时标记 unknown，
只有所有原门完成才设为 true，避免再把旧 plan 标记误带成完成结果。原件范围增加双方
准入输入 JSON、适配前端和实际执行文本，其余完整数组/cache 证据范围不缩减。

CPU 检查结果见 `docs/evidence/native113-direct-native-abba-cpu-20261029/`：七段源码保持、
四角色 mock 正例，以及未知新资格、额外 TL handle、writer spill、未知 registers、错误
int64 stride ABI、外来 writer 对象六个拒绝均通过。mock 不代表实际设备资源；没有运行
数值数组、GPU、SSH 或性能测试。

```sh
python3 -B tools/probe_biv150_native_k8_compact_n48_shared6k_fullstep.py
python3 -B tools/check_biv150_native_k8_compact_n48_shared6k_fullstep_cpu.py
```

实际资格齐备后使用同入口的 `--preflight-cpu`，再由 Root 独占执行 `--gpu --root-owned-exclusive`。
参数保留原 candidate consumer/compile/review/library 五类输入，并增加旧库的
`--old-library --old-consumer-raw --old-consumer-review --old-compile-summary --old-assembly-review
--old-fullstep-summary --old-fullstep-review`；所有路径必须绝对且输出目录必须不存在。
两份实际准入已通过 CPU preflight；设备命令仍须由 Root 填入真实绝对路径和新输出目录，并独占运行。

如果这里的 shared 改动或后续 K pair 在直接组件对照中有收益，而 compact 模型整组仍为负，
应再把可独立机制移植到已验收的 8b86/16KiB reader（N32/S4、N64/S2）单独验证；本提交没有
开始第二实现，也没有授予该移植任何资格。
