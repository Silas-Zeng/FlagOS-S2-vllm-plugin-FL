# PR66 两序列 prefill 服务候选（尚未目标机验收）

主任务已授权最小真实接入。此独立候选分支默认启用
`VLLM_FL_BIV150_PREFILL_MULTI2`，显式 `0` 关闭；团队 main 未改动。
root 正在使用独立冻结 e936576 运行通用18例 GPU 边界，当前工作没有修改那个
worktree，也没有 SSH、GPU、服务重启或 site-packages 操作。边界通过前不得将本
服务候选接入目标机。

初始254b7a0四个组件由主任务报告完成：4K big/small-first
1.203091964x/1.218231016x，16K 1.077525666x/1.105188212x。它们不证明本服务
集成版本的 C3、完整105题、正式吞吐或 TTFT。当前 `precision_gate=false`，PR
保持 Draft，不得合并。

## 选择、父链和编译安全

新增 `vllm_fl/patches/biv150_prefill_multi2.py`。worker 在现有 Q32 installer
之后安装新 policy；原 Q32、Q-fragment、normalization 三个文件逐字不变。
只有 `num_seqs==2 && query_ptr.shape[0]>=1024` 且原 BM128 完整 `_eligible`
通过、无 async/capture/hooks/foreign override/device 冲突时，才复制参数选择
BM256/BQ32/w16/K64/stage1。无 GPU `.item()`/CPU读回，不按历史 qlens、模型层号
或 benchmark 分支。其他调用原样经过 Q32 父链；单序列仍由原 Q32 决定。

选中 grid 必须为 `(qrows//32+num_seqs,2)`，不复用单序列 `+1`。新的 verified
manager 使用独立 JIT/device cache，仍走原真实 binder、ABI/constants/attrs/options、
内容寻址 TTGIR override 和 LLVM/Q-load proof。只有自有 compile callback 的
`_UnavailableBeforeLaunch` 才允许在首次 launch 前关闭该候选并回原链。
设备 launch 异常直接传播，不重试原核。安装遇未知/失活 Q32 父链、源不符或
cache 别名会拒绝；不会跳过未知未来 K128 外层。

默认启用只发生在此未合并候选源码。`compile_semantics` 已纳入有效 multi2 开关：
缺省与显式1同键，显式0异键，Q32/normalization/Q-fragment 父功能关闭后子开关
不再改变键；新模块源内容也进入正式 source fingerprint。

严格身份：runner 新 SHA 为
`0161c595ff2f1b115039f03980fbb6de0447f1edf0d55fba1793b6a8055c230c`，
compute_order.EXPECTED_RUNNER 已更新；SOURCES 新增原 Q32 与 multi2 的实际 pin。
multi2 SHA 为
`fc3cdd066aacc6dfdc5029bb8aae9589f868b6800534511d2198f1f0d88da342`。
没有删除任何 SDK/local source 或 C3 审计。

## CPU 检查与待验证项

- `check_biv150_multi2_service_cpu.py` 实际加载三个模块的原源码，使用真实
  installer、Q32/multi2 policy、Manager/_Launch，只有 tensor/JIT/binder/compile
  是显式假 fixture。默认开/显式关、1023/1024/1025/1026/2048、单序列Q32、三序列
  原路、grid、别名/语义/capture/async/hooks/device、断链/失活/未知源/cache别名、
  prelaunch-only 回退和已launch异常不重试均通过。
- `check_biv150_multi2_cache_cpu.py` 通过有效开关和新源内容的官方 cache identity
  检查；不加载 Torch、SO 或设备。
- `check_biv150_compute_order_identity_cpu.py` 继续校验全部本地 source pin；移除
  本次 installer 后逐字恢复旧 runner，恢复已合入 w16/stride 的有界参数化后仍
  精确逆回原 pre-Q32 Q-fragment，移除新 feature 后精确恢复原 compile_semantics。
  修复的是过时模板，没有用删除保护来通过测试。
- `check_biv150_multi2_acceptance_cpu.py` 验证显式关闭穿过 driver 清理层，候选
  缺省不写开关，并用 AST 对照 main2412207 确认正式 serve 命令及
  `3short + 1fullwarm + 1score` 吞吐循环完全未变。
- 原 Q32 controller/cache 检查、26夹具 descriptor CPU 检查、语法与 diff 检查通过。

证据位于 `docs/evidence/prefill-multi2-20261027/service-*-cpu.json`。
这些不是实际 SDK lowering、live C3/Worker cold-plan、数值、资源或模型通过。
`check_biv150_multi2_stack.py --run-gpu --out NEW_FILE` 已准备真实导入路径、SDK/local
pins 与 multi2 off/default-on C3 closure 审计，当前仅 dry-run，须等 root 边界通过
并排定空闲窗口后执行。

## 安装方式和固定服务参数

不需要 pip 安装。root 将冻结源码放到独立新目录，显式设置 `PYTHONPATH` 并用
指定现有 Python 环境启动；worker 的 installer 自动生效。保持以下正式命令：

```bash
PYTHONPATH="$SOURCE" BIV150_NATIVE_W8A8_LIBRARY="$GATE_LIBRARY" \
  env -u VLLM_FL_BIV150_PREFILL_MULTI2 \
  /usr/local/bin/vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

显式关闭诊断使用同一冻结源码/同一命令，只将前缀改为
`PYTHONPATH="$SOURCE" BIV150_NATIVE_W8A8_LIBRARY="$GATE_LIBRARY" VLLM_FL_BIV150_PREFILL_MULTI2=0`。
`GATE_LIBRARY` 须预先核对为下述5cbb哈希文件。该变量是启动时配置；
不声称能热切换已安装 policy。正式验收应使用下面 owned driver，而不是手工服务。

## 既有完整验收 driver 的用法

`run_biv150_small_out_acceptance.py` 新增 `--prefill-multi2 default/0/1`。
`default` 不写 server env；0/1 在原清理层之后显式传入。driver 保留 native SO
SHA256 `5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe`、
同一 SQLite snapshot、实际 source fingerprint、固定105题、mixed前后语义、
3短暖场+1完整暖场+单轮4K/16K score 和 owned cleanup。新审计要求 Q32/multi2
真实安装、正确 MULTI2 SELECTED 配置/grid、无 MULTI2 FALLBACK，以及实际 C3
FIRST_SELECTED 且没有 SCOPE_REJECTED。SELECTED 是路径证据，模型精度仍看独立105题。

root 可将本提交两份 driver 冻结导出为新的文件，不覆盖历史 exports。以下路径
是**拟用新目录/新导出文件**；`FROZEN_HEAD` 必须由 root 设置为交付的完整 commit，
并在启动前核对目标源码 HEAD/干净状态、source/hash、GPU和9031、前序owner cleanup。

```bash
# SOURCE: root 上传的本提交独立冻结源码；FROZEN_HEAD: 本次完整提交 SHA。
RUN_ROOT=/workspace/experiments/minicpm-s2
cp "$SOURCE/tools/run_biv150_small_out_acceptance.py" \
  "$RUN_ROOT/exports/run_biv150_multi2_acceptance_v1_20261027.py"
cp "$SOURCE/tools/run_biv150_latest_main_pair.py" \
  "$RUN_ROOT/exports/run_biv150_multi2_pair_v1_20261027.py"

/usr/local/bin/python "$RUN_ROOT/exports/run_biv150_multi2_pair_v1_20261027.py" \
  --tool "$RUN_ROOT/exports/run_biv150_multi2_acceptance_v1_20261027.py" \
  --out "$RUN_ROOT/cjj-multi2-service-off-default-v1-20261027" \
  --baseline-source "$SOURCE" --candidate-source "$SOURCE" \
  --baseline-commit "$FROZEN_HEAD" --candidate-commit "$FROZEN_HEAD" \
  --baseline-out-lowering 1 --baseline-large-linear 1 --baseline-production-defaults \
  --baseline-prefill-multi2 0 \
  --library "$RUN_ROOT/code/baseline-profiler-v2-20261025/build-native-w8a8/biv150_native_w8a8.so" \
  --tuning-snapshot "$RUN_ROOT/exports/small-bf16-fullwarm-tuning-snapshot-v6-20261025" \
  --fixtures "$RUN_ROOT/exports/native-mixed-fixtures-20261025"
```

controller 两 arm 串行，候选没有 `--prefill-multi2` 子参数或 env，验证默认启用；
control 为同源码显式0。也可把 baseline-source/commit 改为 root 确认的最新团队
main，移除 `--baseline-prefill-multi2 0`，得到 fresh-main/default-candidate 对照。
不得混用不同 native SO、tuning 初始 snapshot 或已占用 GPU。现有 snapshot/fixture
路径在此仅来自历史已验证记录，启动前由 root 实查，不假定历史文件状态不变。

controller正常结束不等于精度门通过；最终仍独立审查105个唯一问题/无失败预测、
score>=.95、请求/token总数、吞吐/TTFT、两轮mixed语义、C3、实际多序列命中和
cleanup。若近门槛失败但吞吐有潜力，保留Draft与`precision_gate=false`按既定政策
评估回退，不能直接合并。没有任何自动merge步骤。
