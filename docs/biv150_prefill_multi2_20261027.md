# 双序列预填充组件与通用 scope 边界

最新状态：主任务已授权在本人独立候选分支加入默认启用的最小真实服务集成，
显式0可关闭；团队 main 不变。实现、严格 pin/cache 与验收入口见
[服务候选记录](biv150_prefill_multi2_service_20261027.md)。下文“生产不变/尚未接入”
说明的是254b7a0/e936576组件冻结版本；root 的 e936576 边界 worktree 保持隔离。

基于实际 fetch 的团队 main `241220723a09061c182764f8f738e5ec01190367`，本人独立分支 `codex/cjj-prefill-multi2-20261027`、Draft PR66（CJJ1008）。生产 `vllm_fl/`、`csrc/` 不变。本次扩展只做本地 CPU 源码和方案工作；GPU 调度由主任务串行负责。

主任务已报告冻结 `254b7a09e81b6efe85033ed01d0819c52c62f1e6` 四个代表组件均完成、`errors=[]`，4K big/small-first 为 1.203091964x/1.218231016x，16K 为 1.077525666x/1.105188212x；实际 owner PID602039/start340447575 已为 Z。该结果来自主任务串行 GPU 执行，原件路径为 `/workspace/experiments/minicpm-s2/cjj-prefill-multi2-k64-v2-20261027`，本次没有 SSH 取回或独立归档这些原件。它支持继续通用边界验证，不能作为模型精度或正式吞吐通过；`precision_gate=false`。下文历史八例 CPU 记录对应 `254b7a0`，当前扩展检查为 26 例。

现有16K暖场日志出现 normalized `(total2033,maxQ1945,numSeq2)`、`(total2033,maxQ1994,numSeq2)`。当前 Q32 和 PR63 K128 生产策略都只选择 `num_seqs==1`，这些双序列输入仍由原 BM128/w8/K64 处理。这是覆盖缺口的源码依据，**出现次数不是耗时占比**，也不意味着扩大tile一定会赢。完整调度/每序列KV长度尚未捕获。

组件准备四组代表输入：Q `[1945,88]` 及其逆序、`[1994,39]` 及其逆序；前两组各KV4096，后两组各KV16384。只从日志总量推导Q长度组合，序列顺序和KV长度为测试选择，不能冒称原始捕获step。覆盖非对齐Q边界及长短序列顺序变化。

两组均为 2033 行。每 KV head 的非空逻辑 Q tile 数，BQ16 分别为 128/128，BQ32 为 64/65；这是结构计数，不是实际 launch grid 或提速预测。增加四个**显式选择**的边界例，默认 GPU 四例不变：`double4k-ragged-tail` 及 `-reverse` 使用 Q `[1945,88]` / KV `[1945,4095]`；`double16k-ragged-tail` 及 `-reverse` 使用 Q `[1994,39]` / KV `[16383,39]`。这补充不等 KV、零前缀、尾页有效长度 9/15/7 与序列逆序。均为设计边界，非捕获 step。

`tools/bench_biv150_prefill_multi2.py` 的 baseline 安装完整当前 Q-fragment/Q32，未来源码含K128时也要求安装实际K128，保留真实外层policy。仅在Q32的原 BM128 fallback 上放透传tap；单序列父路径不被关闭，双序列按父策略自然回退。必须观察实际BM128编译/选择和资源，父manager仍active、未disabled、绑定未变且两序列未误选父路径。K128存在却失活、依赖异常或链路变化会拒绝运行，不伪造“当前baseline”。

candidate使用私有normalized函数/JIT，BM256/w16，先K64/stage1，另可有界测试K128/maxnreg128。使用生产原严格Q-hoist/lowered proof，不使用PR64的early-Q变换。数值协议源自已共享PR64 `032d6d3`，复用原CPU FP64/reference_guard、changed-input非默认stream Graph、真实加载资源/driver capacity和三轮交错计时；保留IR和失败原件。Graph计时直接重放已验证核，组件数据不能冒充实际服务命中或模型吞吐。

`python3 tools/check_biv150_prefill_multi2_cpu.py` 通过：现有Q32/未来K128两种模拟完整policy链共八次双序列自然BM128回退，失活K128/断链/其他依赖丢失、父路径误选和四类scope错误均拒绝。只是显式假CPU fixture，没有导入Torch、没有运行目标编译器；不证明真实安装、C3闭包、数值、资源或速度。

2026-10-27 补充 CPU 核查：直接加载未改动的真实 `biv150_q_fragment_service._eligible` 和 `biv150_prefill_q32._Policy` 源码，在显式假 tensor metadata 上检查八例。BM128 与 BM256/w16 descriptor 均符合实际 eligibility；Q32 对八例自然回 BM128，另一个 2033 行单序列仍选原 Q32。未通过的 sinks/输出 alias/两序列起点表长度拒绝；port 9031 busy、GPU busy、未知 ixsmi 输出拒绝，idle 允许。预飞改成显式异常，Python `-O` 不能删除保护。CPU 输出保存在 `docs/evidence/prefill-multi2-20261027/cpu-routing.json`，仍为 `precision_gate=false`。

GPU 每例继续比较同一输入的真实 BM128 fallback 与私有 BM256/w16：uniform Q 用 CPU FP64 检查全部行，随机 Q 检查既有参考抽样行，同时全量 candidate/baseline 使用原 `.005/.005`；`reference_guard` 不允许新增或恶化基线逐元素误差。changed-input Graph 在非默认 stream 验证与 eager 逐位一致，三轮计时顺序为 AB/BA/AB。结果含所有输入 tensor SHA256、实际 `num_seqs=2` / tile / Q-hoist 选择及资源；数值失败保存完整 CPU Q/KV/页表/长度和输出。`completed=true` 也只代表组件完成，始终 `official_score=false, precision_gate=false, service_accuracy_validated=false`。

仅在当前服务/测试完成 cleanup、GPU 和 9031 实际空闲后，由主任务确认冻结 commit、使用指定 Python 环境执行。为避免加载 site-packages 的旧插件，必须在远端此实验源码根目录显式设置 `PYTHONPATH`，脚本也检查实际 `vllm_fl.__file__`。每个配置/复跑使用新输出和独立 Triton cache，无需安装或重装依赖。例（`python` 指主任务确认的指定解释器）：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
TRITON_CACHE_DIR=/workspace/experiments/minicpm-s2/multi2-k64-r1/triton-cache \
python tools/bench_biv150_prefill_multi2.py --run-gpu --kv-tile 64 \
  --out /workspace/experiments/minicpm-s2/multi2-k64-r1
```

若第一配置有机制和收益，再单独比较 `--kv-tile 128 --maxnreg 128`；资源升高或某档无益时保留负结果。原baseline必须仍是该输入的真实BM128 fallback；若未来团队改变双序列默认策略，应停止并更新本探针，不能对过时fallback宣称收益。只有真实组件可行后才设计生产二序列选择，并重证严格C3绑定和完整105题/两档吞吐/TTFT；此Draft不包含任何放宽生产scope的改动。

边界组件独立运行可追加四个 `--case double4k-ragged-tail --case double4k-ragged-tail-reverse --case double16k-ragged-tail --case double16k-ragged-tail-reverse`，使用另一个新输出/cache。当前 Mac 系统 Python 没有 Torch，因此未执行 CPU FP64 数值部分；这由目标环境组件负责，不能把上述 descriptor/source 测试写成数值通过。

服务 dispatch 结论：当前不会命中此候选。拟采用低层已有 `num_seqs==2 && query_ptr.shape[0]>=1024` 描述条件，并保留原 `_eligible` 对 dtype/stride/alias/attention 语义的完整保护。总 Q 行数与序列数均为 CPU 可见描述，不要求单序列最大 Q>=1024，也不要求两序列都 qlen>1，不引入 `.item()` 或 GPU→CPU 读回。为覆盖这个通用选择，下面增加 qlen1、均衡、空 query、阈值及 grid 边界；不能按 2033、历史 qlens 或 benchmark 名称硬编码。仅在这些 GPU 边界通过后考虑默认关闭的新 policy，确认 C3、实际 `SELECTED`、父路径及完整105题/两档吞吐/TTFT 后才能转 Ready/合并。

与旧调研的关系：本项属于 2026-10-24 D03/H04（真实 query tile 与寄存器资源边界）和 2026-10-25 AD22（按资源模型确定边界）深化，不重复宣称已经合入的 Q-hoist 本身是新优化。当前 4K graph profile 只采到 decode 图，缺失 eager prefill 时间且不是完整 4K/16K pair，无法据此给本项整机收益上界；继续利用现有暖场形状和组件证据，不重跑整套 profiling。

## 通用两序列边界（待 GPU）

默认四例保持不变。`--suite scope` 明确运行 18 例：

| 子组 | Q / KV 例 | 预期 |
| --- | --- | --- |
| 总 Q=1023 | 511+512 fresh；1+1022 / KV16383+1022 | 两例均保持 BM128 fallback，不调用宽 tile manager |
| 总 Q=1024 | 512+512 fresh；1+1023/逆序；2+1022/逆序 | BM256；含 fresh、KV16383、KV17/4095 尾页 |
| 总 Q=1025 | 512+513 / KV4095+16383；1+1024/逆序 | BM256；含一行 query 和部分尾页 |
| 总 Q=1026 grid | 513+513 fresh；1+1025/逆序 | BM256；实际 grid 必须是 `(34,2)` |
| 均衡2048 | 1024+1024 fresh；KV16384+16383 | BM256 |
| 空 query 描述 | 0+1024 / KV0+1024 及逆序 | BM256；无 query 序列不贡献输出或参考行 |
| 模型长 KV | Q1+1023 / KV131071+1023 | BM256；极不均衡前缀、最大长度附近部分尾页 |

原验证入口不再拒绝 qlen1 或最大 Q<1024，而是检查恰好两条非负 query、总行数正、每序列 KV>=Q。因为 descriptor 条件也不能看到 qlen0，加入可表示的空 query 边界；共享 CPU reference 只对 qlen0 跳过空输出，其余数学和容差不变。

私有 `ComponentScopePolicy` 仅用于诊断，从 `q.shape[0]` 和 `seqused_k.shape[0]` 选 route。低于1024时，候选一侧也调用完整当前 baseline，要求全输出逐位一致、manager selected-call 增量为0，并记录实际 BM128 配置。此时的1x附近计时仅是 fallback 对照，不记作宽 tile 性能。达到1024时运行原 BM256/w16/K64 私有核。每例保持双输入、uniform 全行 FP64、random 抽样 FP64、原 reference_guard、Graph 输出 NaN poisoning 和 AB/BA/AB 计时。

**Grid 必须沿用 SDK 的 `qrows//BLOCK_Q + num_seqs`，不能沿用 Q32 单序列的 `+1`。** 对 513+513、1025+1 及逆序，BQ32 有34个非空逻辑 tile，`+1` 只有33程序，会漏尾。CPU 测试拒绝该错误 grid，GPU 选择 proof 也对实际 tap.grid 校验完整公式；输出 NaN poisoning 会暴露漏写。总 Q=2033 的旧代表形状不足以替代这些边界。

当前 CPU 测试：真实 Q32 源码在26夹具均自然回 BM128，单序列2033行仍保留 Q32；私有 selector 为24次宽 route、2次 fallback，三个错误 `+1` grid 均拒绝。全部为显式假 tensor descriptor/launch fixture，没有导入 Torch、执行目标编译器或 GPU；`precision_gate=false`。以下命令由主任务在独占 GPU 窗口执行，新输出目录必须不存在：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
TRITON_CACHE_DIR=/workspace/experiments/minicpm-s2/cjj-multi2-scope-k64-v1-20261027/triton-cache \
python tools/bench_biv150_prefill_multi2.py --run-gpu --suite scope --kv-tile 64 \
  --out /workspace/experiments/minicpm-s2/cjj-multi2-scope-k64-v1-20261027
```

## 最小默认关闭服务方案（尚未实现）

新增独立 `biv150_prefill_multi2.py`，开关拟为 `VLLM_FL_BIV150_PREFILL_MULTI2=0`。不开关时不改绑定。打开时在已知 Q32 父链外安装一个带 `__getattr__` 透传的新 policy，使用独立 JIT/device cache 与 BM256/w16/K64/stage1 的 verified manager；不修改 Q32 的单序列条件或原 manager。未知父链应拒绝安装，不能绕过未来 K128 策略。

热路仅使用 `num_seqs==2`、总 Q>=1024、原 baseline128 的完整 `_eligible`，以及现有 async、capture、pre-run hooks、device、foreign ir_override 保护。选中后复制 launch 参数，改 BM256/BQ32/w16，grid 用 `(qrows//32+num_seqs,2)`。复用 `_VerifiedManager` 的严格编译/lowered proof；只有**首次 launch 前**失败才关闭候选并回原父链，device launch 异常必须直接传播，不得重试原路。共享原 metadata、地址、RoPE/KV 写入和 compute-order 逻辑均不变。

需要审查/更新的身份与 cache 项：

- 在 `model_runner.py` 的 Q32 安装之后增加独立 installer，实际文件 SHA 必变；当前为 `268804cb09d84baa8497d020adaf04329ead8bf29951609d8bf602bf92b04613`，必须把 `biv150_compute_order.EXPECTED_RUNNER` 更新为新实际 SHA，并按需在 `SOURCES` 固定新模块。不能直接删 source/C3 检查。
- Q32 当前 SHA 为 `3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b`；Q-fragment `38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82`、normalization `957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1`。最小方案不改这三个文件，原 pin 应保持；新模块和 runner 用实际新增 hash。
- `check_biv150_compute_order_identity_cpu.py` 的 runner 精确 inverse 要认识新增 installer，再逐字还原旧 runner；现工具还包含早期 Q-fragment 签名 inverse，须对当前已泛化 warps/stages 的 main 源码核对后修复模板，不能为通过测试删除 identity 保护。`check_biv150_q32_stack.py` 要补默认关闭/打开后实际 C3 链和设备 cold-plan 审计。
- 保留每个私有 JIT 的独立 `device_caches`、真实 binder、完整 ABI/constants/attrs/options（含num_seqs的真实特化语义）、内容寻址 override 和 `normal_hash != transformed_hash`。不按缓存目录名字或手工重用旧 compiled 对象选择路径；保存实际 ttgir/llir/资源及父链未变证据。

以上只是接入设计；本提交没有新增生产模块或修改任何 source pin。通用边界 GPU 通过后，由主任务决定接入和正式验收。
