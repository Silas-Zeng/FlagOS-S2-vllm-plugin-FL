# 下一步：保留原数值门，直接比较 2ad / 5cf

以下是待实现/待实测的适配方案，不能作为设备通过记录。

1. 四 case consumer 复用 `probe_biv150_native_k8_compact_n48_consumer.py` 的完整执行代码，
   仅改变候选 factory/compile/admission、private operator 和新增前端留档。
   保留 41/42 live × pitch 2048/2560、29 页、maxctx193、两个 host 拒绝、6 phases、
   64 行 backing 的 padding/poison、完整 FP64/逐位/Graph-B-first/side/cache/epoch 门。
   两次 N32 7892 是原有数值控制，不是性能分母。
2. 候选编译/产物/SO 检查复用原 compile_binding。显式读取未修改的 Root `b481…` 完整
   审查 schema，严格绑定 `1dc / 5cf / compile-summary / inspection / source SHA / S4 6144`。
   不能把它伪装成旧 `S2+S4 / 8192` 模板。Root 只批准独占数值测试；不把它转换成
   `hardware_SME_alignment_verified=true` 或性能资格。
3. 新 consumer 实际原件及独立 review 尚未产生，fullstep 的新准入 pin 保持未知，必须阻止
   执行，不能借用 493e/c079。
4. fullstep 在原 112 两臂驱动上，把 TL producer 改成实际旧 `2ad`，Native 改成新 `5cf`。
   臂名为 `old8k / new6k`，顺序 `old8k,new6k,new6k,old8k`；同时加载核对两库 schema 和
   独立实际资格。旧 2ad 用 493e/c079 及 5a6a/2d8d 原资格，新 5cf 必须等上一步。
5. 保留 44769 页、valid41×17408 与 valid42×16896 两 shape、Q/K/V source pitch2560、
   同 dc5 初始 encoder/append、原 reducer、三阶段变化数据、完整 cache/golden/Graph/side、
   原 .005 容差及全部原件范围。两臂都使用原 I8-QK 黄金参考，臂名只映射至原 k8 helper。
6. JIT 角色明确变为 4 个：initial_encoder、两臂 append、reducer。原 TL producer 不执行，
   因而其 JIT 身份检查不适用；替代它的是两份实际 Native SO 准入，不伪报一个 TL handle。
   所有仍执行的 JIT typed ABI/runtime n_regs>0/runtime n_spills=0 门原样保留。
7. 原 timed/capture/reducer/poison/fingerprints/verify_inputs/step 七段，在标签逆变换后
   逐字核对；保留 warm128、timing64、unroll8、event+host clocks、timing 前后数组和全
   cache 检查。初始 encoder/source packing 成本仍单列，完整 step 包含 append+producer+reducer。
8. 报告直接给本次 `old8k/new6k`，不与旧 TL48 比值跨运行相减。不需要为第三控制臂重写驱动。
   无 runtime occupancy 结果时也不从寄存器/shared 理论数反算驻留 CTA。
