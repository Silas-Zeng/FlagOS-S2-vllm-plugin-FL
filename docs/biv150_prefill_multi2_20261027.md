# 双序列预填充的大tile组件试验（仅CPU准备）

基于实际 fetch 的团队 main `241220723a09061c182764f8f738e5ec01190367`，本人独立分支 `codex/cjj-prefill-multi2-20261027`、Draft PR66（CJJ1008）。生产 `vllm_fl/`、`csrc/` 不变。本次只做本地 CPU 源码和方案工作；GPU 调度由主任务串行负责。

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

服务 dispatch 结论：当前不会命中此候选；若组件有净收益，需要新的独立 multi2 policy/manager，保留 Q32/K128 单序列父链及 BM128 fallback，复用严格 compile/lowered-proof 和捕获期/异步/设备/side-hook 保护，不改 compute-order。不能直接把原 `num_seqs==1` 改成 `in (1,2)`：现低层 descriptor 只有总 Q 行数，不能证明每序列 `qlen>1` 或最大 Q>=1024。接入前须补 `[2032,1]` 及逆序、均衡双序列等通用二序列组件证据，或从已有 CPU attention metadata 构造一次可审计的 eligibility，不增加 GPU→CPU 读回；不能按 2033 或 benchmark 名称硬编码。确认实际两序列 `SELECTED`、父路径未变、完整精度与两档吞吐/TTFT 后才能转 Ready/合并。

与旧调研的关系：本项属于 2026-10-24 D03/H04（真实 query tile 与寄存器资源边界）和 2026-10-25 AD22（按资源模型确定边界）深化，不重复宣称已经合入的 Q-hoist 本身是新优化。当前 4K graph profile 只采到 decode 图，缺失 eager prefill 时间且不是完整 4K/16K pair，无法据此给本项整机收益上界；继续利用现有暖场形状和组件证据，不重跑整套 profiling。
