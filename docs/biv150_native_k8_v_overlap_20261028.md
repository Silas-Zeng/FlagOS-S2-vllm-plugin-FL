# NativeK8：延后 V 完成等待的独立实验

基于最新团队 main `1cbac531ca9afaa0e9a68624d5d10cb1d0412055`。生产路径不改，以下仅为工具目录中的独立候选；未编译、未运行 GPU、未获得精度或吞吐资格。

原 PR93 / PR98 的 NativeK8 CUDA 源码来自 `37713909f1f25b69d5ef6c155fad3b87351046fb`，SHA256 `bf84c82ff7e73684c3923ca8944c6564f37c21fb636134e106d0542f5ba47cd9`，对应原设备库 `22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd`。该库的已有通过结果不能直接授予新库资格。

## 修改与假设

原执行顺序为 K 普通装载和 shared store → barrier → V SME 发射 → V wait8 / barrier / 尾页清零 → INT8 QK → softmax → BF16 PV。本候选保持 K、V 发射顺序，只启用源码原有的晚等待位置：在 QK、softmax 和 P 打包后，V 尾页清零与 PV 读取前执行 wait8 和 barrier。与 PR105 在 BF16 路径采用的等待重叠机制相同，尚未在 K8 路径验证。

PR96 V-first 将 V 发射放在 K 装载之前，仍在 QK 前等待。本实验不组合 PR96，也不组合 Q reload、paired K load 或 compact page。K/Q 量化、INT8 QK、scale 乘法、softmax/FMA、BF16 P 舍入、PV、分段、输出、所有 host ABI 检查和原 16 KiB cache 格式均由全文件逆变换约束为原字节。

K shared 区域为 `[0,2048)`，V 为 `[4096,8192)`。原 K 屏障、PV 前晚等待后的屏障及每页结束屏障保留。17 种 valid 值的 V 尾清零共享字检查只验证地址与位操作；不能证明实际设备异步完成语义。编译器可能把等待提前，必须看最终 SO，不能仅凭源码顺序宣称隐藏延迟。

## 本地 CPU 检查

```bash
python -B tools/check_biv150_native_k8_v_overlap_cpu.py
python -B tools/compile_biv150_native_k8_v_overlap.py
```

默认入口不导入 Torch / Triton / NumPy。编译函数相对原工具仅改变 extension 名称；CUDA flags 只将 `COREX_NATIVE_GQA_V_OVERLAP` 从 0 改为 1。新命名空间为 `cjj_native_k8_v_overlap_probe`。

## Root 后续设备步骤

确认旧任务真正结束、服务端口及设备占用后，串行执行：

```bash
python tools/compile_biv150_native_k8_v_overlap.py \
  --compile-only --root-owned-exclusive \
  --compiler /usr/local/corex/bin/clang++ \
  --out /workspace/experiments/minicpm-s2/NEW_UNIQUE_V_OVERLAP_OUTPUT
```

产物保存原始 LLVM、ASM、SO、编译命令、失败信息与 SHA manifest。需要用 `ixobjdump --IR` / `--sass` 检查最终 SO 的两个 specialization、完整页与尾页：V 发射顺序没变、QK/softmax 确实先于 V 完成等待、PV 读取前完成等待和 tail clear、无新增 spill、原 codec 与算术未变化。

数值检查复用 PR93 `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46` 的 `probe_biv150_native_k8_consumer_v2.py` 和 fullstep，Root 仅适配新库身份与 operator namespace。不得把原固定 22e SHA 的入口直接对新库标为通过；新 SO 必须单独审查后运行原全行 O/M/L/final、独立量化 FP64、尾页、只读/guard、换值 Graph B 先于 fresh Eager B、side C 门。保留原 64/S2、32/S4 与 Q pitch 2048/2560 范围。设备数值通过后再测完整 writer+producer+reducer 成本，再考虑接入 PR98；原始105题、真实分支覆盖及两档正式服务比较仍需新验证。
