# Compact NativeK8 service + compact Q32 hoist：独立组合候选

本候选把 PR109 的 compact12544 服务与 PR103 的 compact Q32 hoist 接到同一个 runtime，三个开关均默认关闭。当前完成的是 CPU 源码与宿主接线验收；没有运行本组合的 GPU、模型精度、吞吐或 TTFT 测试。

从 PR109 `45321a9410885ed0fc6aa9be6aeeac13a7f7f89b`（包含 main `a455a3de364051fbea80226870c239de3a48e716`）建立独立分支，正常合并 PR103 `d8690464b196ed251750ac4e75b6eaea32e95779`。合并提交 `c0bbdf0a9dcb44573eba8840b7f5804d467cc69d` 保留这两个父提交，没有叠加 PR110。原 PR109 和 PR103 的代码、报告、编译与数值证据均保留；本组合不继承它们的服务验收资格。

## 改动与冲突处理

| 文件 | 处理与理由 |
| --- | --- |
| `compile_semantics.py` | 以 PR109 为底，保留其 NativeK8 preflight、原 8b86 身份和 compact Native40f9 事实。只将 compact/hoist 的旧互斥改为两种 cache 格式互斥；记录显式 Q32 开关，并把两个原有 hoist 依赖纳入 compact digest。 |
| `worker/model_runner.py` | 保留 PR109 提前 native preflight、天数 vendor 限制和旧 16KiB native 的格式限制。允许 compact 自己的 hoist；两种 cache 格式仍互斥。保留周围旧换行格式，仅三条触及行使用 LF。 |
| `biv150_konly_compact_sources.json` | 保留 PR109 原 Native40f9 qualification、main a455 绑定，加入 PR103 的 proposal/adapter 和两个原依赖，重算实际宿主 SHA。没有将组合的设备、服务或精度状态改为通过。 |
| `compact_native_qualification.py` | 在 compact native 和 hoist 同时开启时，先以 stdlib standalone 方式核对完整 source manifest，再固定核对 6c 的 Q32、compact proposal、compact adapter 与两个原依赖。未知源码、旧 16KiB adapter、改源码后自行刷新 manifest 均拒绝。返回的 native qualification 和 ABI 未改。 |
| compact identity / service / runtime | 正常合并 PR103 的完整依赖检查与实际 adapter 安装。Q32 每次调用先做 compact role/epoch/source 检查，再沿原 prepare/after_launch；native decode 接线保持 PR109。两条路径在同一 runtime 中独立分流。 |

六个 compact GPU kernel 与 `6c08ec2561b1c3cee96838bffd4e296b757ad877`、`dc5fd2e5b86a3d57aceaad26e8556d169ec7462a`、PR109 逐字一致。新增到 PR109 的 compact proposal/adapter，及复用的 fresh binder/native/resource 审核器，都与 6c 逐字一致。Native CUDA/C++/codec 仍是 PR100 原字节，SO 只接纳 `40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453`；没有新 ABI、量化公式或 GPU 计算源码。

原有数学和吞吐脚本、正式 CLI 参数未改。相对 PR109，原先存在的 tools 文件仅修改了两份 CPU 检查器；新增的 PR103 probe 仍保持其原始冻结来源，不能拿旧 probe 的结果冒充本组合。

## CPU 结果与边界

[完整源绑定与汇总](evidence/compact-service-q32-combination-20261029/source-and-review.json)记录确切源码 SHA；[native/组合检查](evidence/compact-service-q32-combination-20261029/native-and-combined-cpu.json)、[Q32 分流检查](evidence/compact-service-q32-combination-20261029/q32-routing-cpu.json)、[C3 隔离检查](evidence/compact-service-q32-combination-20261029/standalone-C3-cpu.json)均通过。

- 复用实际 installer 和既有 42 层 CPU metadata fixture；两适配器同时安装，同一个 runtime 的两个 C3 身份绑定均可检测替换。Torch、Tensor、JIT pin realization 与 operator load 是明确 mock，没有实际加载 SO 或分配 GPU。
- 原 helper94db 在隔离进程下生成 native OFF/ON × hoist OFF/ON 四个不同指纹，compact implementation digest 一致；阻止并确认没有导入 Torch、Triton、vLLM、vllm_fl、NumPy。 该检查使用 CPU ABI，并关闭 W8A8/GQA 库读取；记录不是实机 C3 证书。
- 六类组合源码拒绝通过，包括禁用 adapter、未记录的 Q32 改动、连同 manifest 一起改写 Q32、替换成旧 adapter、改写原 native auditor、过期宿主 C3。
- 原 decode 四个范围（64/S2 与 32/S4，各 pitch2048/2560）、40/48 私有 fallback、postlaunch 不重试仍通过；加入会报错的另一分支 adapter，确认 decode 不调用 Q32 prepare，prefill 不调用 NativeK8 decode。
- 原 compact 12288 个地址、11 个 role/epoch/source 拒绝、完整 CodeType、单次 reader、capture 走 normalized2D128 均通过；默认关闭不导入或绑定。

这些 CPU 结果没有证明实际 JIT、fresh IR/resources、Graph、42 层设备容量、TP、Math105、吞吐或 TTFT。PR103 原六阶段结果仍只属于 6c：量化 FP64 硬门和位一致通过，但全部 12 个 BF16 proxy 失败，阈值没有降低；它的 Graph 仅覆盖 normalized2D128 bypass，不能授予 Q32 hoist Graph 资格。详见原[编译证据](evidence/compact-q32-hoist-compile-20261028/README.md)和[数值证据](evidence/compact-q32-hoist-numeric-20261028/README.md)。PR109 的模型/服务结果也不自动适用于这个组合。

## 复跑 CPU

在仓库根目录执行。`reviewed_so` 必须指向实际 40f9 SO；它只被读取，不被加载。其余均是既有 CPU 入口，helper94db 原字节已随证据保存。

```bash
set -e
reviewed_so=/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out/build/cjj_native_k8_compact_probe.so
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_native_qualified_service_cpu.py --library "$reviewed_so"
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_q32_hoist_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_q32_hoist_cpu.py --installer-on
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_konly_compact_install_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_hoist_standalone_cpu.py \
  --helper docs/evidence/compact-service-q32-combination-20261029/actual-owned-service-helper94db.py \
  --compact-native-library "$reviewed_so"
```

## 后续设备验收

Root 审核并建立独立 Draft PR 后，保持 PR109 原数学与吞吐协议、vLLM 参数、完整 warmup 与评分次数。组合仅额外显式打开 compact service、compact native decode、service Q32 hoist；旧 encoded service 和旧 16KiB native 均保持关闭，SO 使用实际 40f9 路径/SHA，TRITON_CACHE_DIR 必须是该候选独立的绝对路径。

每个实际 fresh Q32 编译对象仍要走原 binder、TTIR/TTGIR/LLVM/ASM/cubin、资源及 after_launch 检查，不能搬用旧对象 receipt。须记录真实导入、C3、实际 fresh 对象与 runtime 选择/失败；捕获或 selection 次数不能当作实际请求执行频率。未知 specialization 在 reader 前保持原 compact unhoisted 路径；reader 后失败仍传播，不能重试。组合模型精度达标且与最新 main 同协议吞吐/TTFT 对照满足要求后，才具备合并讨论条件。
