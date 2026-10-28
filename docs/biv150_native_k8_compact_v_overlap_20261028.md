# compact NativeK8：延后 V 等待的独立候选

从 PR108 证据提交 `46b44a1326973ccb6b874452841726298507414c` 建立，已包含团队 main `a455a3de364051fbea80226870c239de3a48e716`。仅新增工具目录候选和 CPU 检查，不修改生产、PR109 运行源码、现有 40f9 资格或原 compact/8b 文件。当前只有源码检查通过；没有新 SO、设备数值、42 层服务或性能资格。

PR108 的 compact40f9 完整步骤约为 0.3968/0.7690 ms；历史 16 KiB 8b 完整步骤约为 0.3414/0.6579 ms。不同 cache 格式和实现之间的差异只用于提出实验方向，不能据此归因于 reader，也不能据此预测本候选收益。

## 只移植现有等待机制

父 CUDA 是 PR100 `51da6fd73e7fff1513081fb330323ce72c00856c` 的 `biv150_native_k8_compact_qk.cu`，SHA256 `3bd67edd58f1ea822a908dcc9217aceeef93124a7e9e00e6d7815dcdffe7a81d`。它仍在 V SME 发射后立即 wait8，再进行 QK；原有晚等待分支被 `OverlapV=false` 关闭。

本候选复用 PR106 `a1eb7deee5e8ef99136128984256762096a3657f` 的同一组六项变换，移除前置 wait/clear，并启用已存在的晚等待 consumer。K 普通装载、shared store 和 barrier 后才发射 V，QK 与 P softmax/打包之后才等待 V，尾页清零及 PV 读取之前完成等待和 barrier。每页结尾的 shared-read barrier 保留。没有组合 V-first、PR103 Q32 hoist、新 codec、路由或输出规则。

新命名空间是 `cjj_native_k8_compact_v_overlap_probe`。CUDA/CPP/codec 各使用独立文件；公共 codec header 仍是原 `37d5c754...`。保留 12544B 物理页、K page stride 12544、scale page stride 3136、V page stride 6272、V 预偏移 4352B，以及原 CPP 的 64/S2、32/S4 与 query pitch 2048/2560 范围。40/48 没有新增 native 入口。

dc5 writer/views 只通过原冻结 Git 对象及 SHA 引用：writer `a0cacf3f...`、views `1ab738e4...`。这个组件分支不新增 encoder 或复制服务 writer；以后 fullstep 仍须复用原 dc5 writer 和 PR108 的既有流程。

## CPU 证据

```bash
python -B tools/check_biv150_native_k8_compact_v_overlap_cpu.py
python -B tools/compile_biv150_native_k8_compact_v_overlap.py
```

[CPU 报告](evidence/native-k8-compact-v-overlap-cpu-20261028/check.json) 验证：

- 新文件撤销命名空间与 PR106 六项变换后，逐字恢复原 compact 文件。原 compact、原 8b 与共同 header 不变。
- 在新文件上撤销 compact 地址变化并转回 8b 命名空间后，CUDA/CPP/codec 逐字等于已验证 a1eb 源码；地址变化与等待变化可以独立还原。
- 复用原 compact 全页地址证明；K shared `[0,2048)` 与 V shared `[4096,8192)` 不重叠。沿用 PR106 的 17 种尾长、1024 个 V word 清零检查及三处 barrier/等待顺序检查。
- 拒绝改变数学、地址、tail clear、等待屏障或 host 范围的 11 个源码变体。编译函数仅换 extension 名称后可逆恢复原 compile-only 函数；编译 flags 与 PR106 相同。
- 默认 CPU 入口不导入 Torch/Triton/NumPy，不访问设备。缺少 Root 独占协调参数的编译请求仍拒绝。

这些检查不证明异步装载完成语义、实际编译调度、寄存器数量或设备数值。

## Root 后续串行步骤

新分支可先建立独立 Draft，依赖 PR108 的 compact 布局与 fullstep；机制来源为已合入的 PR106。当前 40f9 库不能代替新库，现有通过收据不能直接填到新命名空间。

完成现有任务并检查 owner/设备状态后，用新输出目录编译：

```bash
python tools/compile_biv150_native_k8_compact_v_overlap.py \
  --compile-only --root-owned-exclusive \
  --compiler /usr/local/corex/bin/clang++ \
  --out /workspace/experiments/minicpm-s2/NEW_COMPACT_V_OVERLAP_BUILD
```

入口沿用 PR106 compile-only 主体，保存原始编译命令、LLVM、ASM、SO、错误与 artifact manifest；不运行 encoder、attention、reducer 或输入初始化。最终 SO SHA 只能从本次实际产物获得，当前为未知。

随后用 `ixobjdump --IR` 和 `--sass` 审核最终 SO 的 S2/S4、完整页与尾页，检查资源、codec、地址、算术和 CFG。准确的等待边界沿用 `actual_V_wait_after_QK_and_P_softmax_packing`：要求实际 QK 与 P 的 softmax/打包位于 wait 前，允许 rho exponent 和 running-L FMA 被编译器留在 wait 后。不能把源码顺序或机器码文本行号当作 CFG 支配关系。

实际 SO 通过独立检查后，再最小适配原四 case consumer 的库/命名空间身份并保留完整数组、FP64、guard、Graph、side 检查；之后复用 PR108 的 dc5 writer、34276 页和原 BF16 对照完整 ABBA。这里不新增数值框架，也不提前放开当前服务的 40f9 资格。模型105、真实容量、两档正式吞吐与 TTFT 仍需独立实测。

## PR110 后继：仅适配真实身份的验证前端

[手工阶段配置](evidence/native-k8-compact-v-overlap-frontends-cpu-20261029/compile-manual-spec.json) 不是 launcher 输入，`automatic_execution=false`、`root_exclusive_confirmed=false`。其中 owner44037/start355048326 只是任务下发时的 PR109 Math105 快照；真正执行前仍由 Root 重新确认当时的任务结束及设备独占。编译继续使用上面的既有入口，CUDA/CPP/codec/header/compile/factory 对 `940d099` 均不变，没有第二个编译工具。

已有 `probe_biv150_native_k8_compact.inspect_SO` 可以直接导出新库的最终机器码；它只读取本次 compile-summary 的真实库身份，不涉及 attention 启动。Root 编译完成后，在同一冻结 checkout 执行：

```bash
PYTHONPATH=tools /usr/local/bin/python - <<'PY'
from pathlib import Path
from probe_biv150_native_k8_compact import inspect_SO
out = Path('/workspace/experiments/minicpm-s2/cjj-native-k8-compact-v-overlap-compile-v1-20261029')
result = inspect_SO(out, Path('/usr/local/corex/bin/ixobjdump'))
assert result['completed']
PY
```

这只生成 `actual-so-inspection.json` 及 IR/SASS，不表示审核通过。40f9 的真实 S2/S4 基线资源都是 VGPR100、SGPR66、shared8192B、private/spill0；新库资源必须重新读取，不能把这些数值填入候选收据。实际审核还需确认：

- 最终 SO 字节、IR/SASS、候选源码及独立 namespace 一致；codec 保留原语义。
- K 装载/barrier 之后发射 V，QK/P softmax/打包在 wait 前，wait 支配尾清零及 PV shared 读，尾页与每页结束 barrier 完整。
- 12544/3136/6272 及 V+4352B、page16/head128、32/64 scopes 与全部量化/QK/PV 算术未变；不混入103。

新的 `tools/probe_biv150_native_k8_compact_v_overlap_consumer.py` 复用 PR100 `51da` 的数值 `run`，只改一处 operator 名称和一处前端源码身份 tuple。原 compact layout/LiveEpoch、四 case、29页/maxctx193、完整数组保存、FP64、Graph B 先于 Eager B、side C、原 BF16 proxy 和容差都不变。compile binding 只改编译 frontend/flags 指向；原审核门继续执行，并增加 `actual_K_then_V_issue_order_preserved`、`actual_V_wait_after_QK_and_P_softmax_packing` 两个字段。缺失、false、资源未知仍拒绝。

```bash
python tools/probe_biv150_native_k8_compact_v_overlap_consumer.py \
  --review-template --compile-summary "$BUILD/compile-summary.json"
```

模板需要真实 compile+final-SO inspection，审核项默认 false，资源默认 None。Root 根据实际 SO 完成独立审核后保存 `$REVIEW` 并计算真实 `$REVIEW_SHA256`；已有22e codec收据只表示 codec 参考，不授予候选 attention 资格。

```bash
python tools/probe_biv150_native_k8_compact_v_overlap_consumer.py \
  --gpu --root-owned-exclusive \
  --compile-summary "$BUILD/compile-summary.json" \
  --assembly-review "$REVIEW" --assembly-review-sha256 "$REVIEW_SHA256" \
  --compact-library "$BUILD/build/cjj_native_k8_compact_v_overlap_probe.so" \
  --original-library /workspace/experiments/minicpm-s2/cjj-native-k8-compile-v1-20261028/build/cjj_native_k8_qk_probe.so \
  --codec-summary /workspace/experiments/minicpm-s2/cjj-native-k8-codec-v1-20261028/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --out "$NEW_CONSUMER_OUT"
```

完整步骤入口为 `tools/probe_biv150_native_k8_compact_v_overlap_fullstep.py`。PR108 数值 run 只再替换 namespace；原 dc5 writer、两处写入、34276页、BF16 aace 组件对照、预热/ABBA、完整 cache/guard/padding/FP64/Graph/side 保存范围均保留。实际四 case 原件必须先归档为原流程的 `out/`、`exports/*owner.json`、`download-manifest.json` 结构。

新入口要求真实 consumer-summary SHA 和 archive-manifest SHA，校核真实 owner终态及其提交中的执行源码。旧 saved-array replay 仅参数化路径、owner、文件计数与 review身份，整个数组/数值重放主体逐字不变。所有保存数组与输入仍独立重算校验后才进入 fullstep，不能只凭 summary 的通过字段。

```bash
python tools/probe_biv150_native_k8_compact_v_overlap_fullstep.py \
  --gpu --root-owned-exclusive \
  --parent-source /workspace/experiments/minicpm-s2/code/cjj-native-k8-fullstep-v1-20261028 \
  --compact-consumer-raw "$ACTUAL_CONSUMER_ARCHIVE" \
  --variant-consumer-sha256 "$ACTUAL_CONSUMER_SUMMARY_SHA256" \
  --variant-archive-manifest-sha256 "$ACTUAL_DOWNLOAD_MANIFEST_SHA256" \
  --compile-summary "$BUILD/compile-summary.json" \
  --assembly-review "$REVIEW" --assembly-review-sha256 "$REVIEW_SHA256" \
  --codec-summary /workspace/experiments/minicpm-s2/cjj-native-k8-codec-v1-20261028/codec-summary.json \
  --original-library /workspace/experiments/minicpm-s2/cjj-native-k8-compile-v1-20261028/build/cjj_native_k8_qk_probe.so \
  --consumer-summary /workspace/experiments/minicpm-s2/cjj-native-k8-consumer-v2-20261028/consumer-summary.json \
  --consumer-proof-sha256 c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128 \
  --k8-library "$BUILD/build/cjj_native_k8_compact_v_overlap_probe.so" \
  --bf16-library /workspace/experiments/minicpm-s2/cjj-main-b3-native-gqa-build-v1-20261028/corex_native_gqa_s2_probe.so \
  --out "$NEW_FULLSTEP_OUT"
```

上述 `$...` 均需由 Root 用实际路径/文件 SHA 设置，当前没有假设它们已存在。候选 SO SHA 未产生，也不复用40f9或8b的 attention收据。四 case 的原22e数学对照与 fullstep 的 aace组件对照按原流程保留；当前生产控制仍为40f9，没有服务接入改动。

[后继 CPU 报告](evidence/native-k8-compact-v-overlap-frontends-cpu-20261029/check.json) 包含源码逆变换、JSON身份 roundtrip、6个审核字段负例、真实40f9编译收据被拒绝、完整数组重放主体字节保持、3eb→adapter 导入顺序及默认CPU/缺参拒绝。没有构造假候选 SO 或通过收据，没有执行保存数组重放或任何 GPU 阶段。

```bash
python -B tools/check_biv150_native_k8_compact_v_overlap_frontends_cpu.py \
  --parent-source /ABS/FROZEN_3eb_CHECKOUT \
  --old40f9-compile-summary /ABS/ACTUAL40f9/compile-summary.json
```
