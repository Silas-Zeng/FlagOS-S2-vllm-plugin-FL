# BI-V150 baseline profiling

基线为团队 main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`。这是诊断草稿，当前没有生产算子改动或吞吐收益。

## 已验证的可观测性

- 同一小型 BF16 GEMM + add，Torch profiler 的 eager 记录有 9 个 kernel，Graph 的 3 次回放没有内部 kernel 记录。
- Corex ixsys 独立复核：eager 窗口 9 个 kernel、合计 138.596 µs；Graph 窗口仅有 3 个整图记录、合计 108.172 µs，内部 kernel 记录为 0。这是可见性检查，不能据此比较 eager/Graph 性能。
- 原生 Corex Graph Event 节点可用。专用 256×256 小图在克隆图上插入每个 kernel 前后事件，保留全部原始依赖；两组固定输入、各三次回放均逐位一致。每次回放前将中间与输出缓冲填 NaN，排除了输出残留造成的假通过。
- 两个节点的回放计时约为 6.85 µs、4.61 µs；所有调用返回成功，资源释放无错误。该脚本逐调用记录日志，外部事件间隔包含 CPU 写日志空窗，不能用于性能或观测开销判断。

`tools/check_biv150_graph_event_timing.py` 默认不使用 GPU；显式 `--run --out NEW.json` 执行专用图能力检查。原图不被修改。源文件 SHA256 为 `717be1049456c5a4059f9e95f7a21fb0754c1e61acf63bf675819dc8a7f60c2c`。

## 下一步与判定边界

服务诊断接入已准备：`run_biv150_baseline_profile.py` 启动独立 ixsys/service session，复用原 `benchmark()`，每长度先固定三轮预热，仅诊断轮添加 `--profile`。`biv150_profile_service.py` 保留正常图，在 padded rows 32/64 的第 32、128、512、900 次回放抽样；原图、仅边界、全节点图分开验证。输出 storage 与当前输入、模型参数/buffer、全部绑定 KV cache 交叉检查后才可填 NaN，并用原图恢复正控及每个 clone 的有限值/逐位检查排除输出残留。CPU scope 使用实际 SDK 的 `_PROFILER_FUNC`，不依赖缺失的 Python nvtx 包；每步元数据只留 RAM，窗口结束统一落盘。

第二个实机小图验证了 driver function 核名映射：64×2048 @ 2048×2560 加逐元素运算，2 输入×5 轮均逐位一致。仅边界图中位 31.908 µs，全节点图 45.512 µs，约增加 42.6%；这只证明探针开销明显，不能套用于完整模型。完整结果见 `graph-probe-library-result.json`。实际 SDK 导入及构造 smoke 还确认此版本需同时传递 Python `__new__` 与 pybind `__init__` 的 `keep_graph=True`，第二次 smoke 才成功保留原图。完整模型的热点结果仍待实际运行。

在实际基线服务保留原图，用克隆图对预定的少量执行步采样；将节点对应到真实 kernel 名称、形状及源码来源。同时观察 prefill、decode、混合执行和 CPU 调度/同步空窗。4K、16K 分开报告。比赛计算、调度参数及 benchmark 不变，额外配置仅用于观测。

必须先量化事件插入开销，避免把探针开销误认为算子耗时。计时热路径禁止打印、写盘或每步同步。诊断运行的吞吐不能作为比赛成绩；后续优化仍需独立精度、吞吐和 TTFT 验收。

## 原始证据

`graph-profiler-capability-20261025.tar.gz` 为三次能力测试的完整脚本、日志、trace/SQLite 和结果，共 62869 字节，SHA256 `d77694708720fcd00457b7f36eec36e2963170cfba7ec43fa2be618fd99792b4`。抽取结果另见 `graph-event-result.json`、`ixsys-visibility.json`。仅在用户个人实例执行；队友实例未访问。

## 实际服务接入修正（v2）

第一轮实际服务运行于 `9ac118e09cf0984af3a358aa3edf9754c6650fb4`，生产代码等同 main `0bf9d2e`。4K 完成了三轮固定 warmup 和完整 diagnostic workload，保留了 4,246 步调度元数据及完整 ixsys SQLite 时间线；但错误地挂到 SDK CUDAGraphWrapper，实际服务使用 FL GraphWrapper，因此采样数为0，未执行16K。该结果只支持 CPU、非Graph kernel 和整Graph分析，不能当内部节点热点。

v2 改为 `vllm_fl.compilation.graph.GraphWrapper.concrete_graph_entries` / `entry.graph`。真实该 wrapper 的32次replay微测成功产生1次采样：原图 poison positive control、boundary/full clone 均 finite 且 torch.equal 数值完全相等（不声称区分 signed zero 的字节逐位一致），原生和Triton节点均可命名，Triton function handle 可对应 hash/cache源码。补测64行捕获、63个有效token同样通过；只验证模型实际使用的行，完整 allocation 仍参与别名保护。原始窗口与摘要见同目录 actual-wrapper-smoke-v3 文件。

启动 instrumentation 改为 WorkerFL.__init__ 前执行，避免 cpuinfo 等普通Python子进程被导入Torch或污染JSON stdout；8项CPU启动时序/隔离检查通过。新增离线节点分析器按诊断窗口关联4K/16K，不把不同 kernel 区间相加当GPU墙钟，也不把诊断吞吐当正式分数。

以上均为诊断工具验证，尚无新的可合并性能收益。

第一轮完整4K的时间线、非Graph热点及每请求阶段核对见 [v1报告](v1-timeline/report.md)。该报告是runtime9ac118e的已有证据；正在运行的v2固定为b37125e，不随文档提交变化。

## v2完整4K节点采样已通过

运行版本b37125e，4243步、4个选定纯decode样本（实际64行）。4次原图positive control及boundary/full clone全部finite、数值完全相等，API与scope errors为空，离线结构分析通过。16K仍在运行，这不是完整4K/16K成对结果。原始完整窗口、benchmark和验证记录保存在[v2-4k归档](v2-4k/archive.json)，可直接重做离线分析。

图边界计时中位31.515ms，全部节点插桩后33.991ms，配对增量中位7.857%。按每样本插桩区间和排序：GQA主计算22.692ms、BF16 MM6.109ms、INT8 GEMM1.798ms、GQA reducer0.440ms。这是带插桩的图内排序，不能当作未插桩核时间或服务吞吐；GQA主计算是明确的首要decode热点，现有证据尚未区分其带宽/延迟/指令瓶颈。

M2048原生BF16 GEMM探针已另开Draft PR44。尚未执行GPU验证，也没有新增可合并吞吐收益。

## v2成对4K/16K节点与调度核对通过

16K于15:49完成，4818步、4个实际32行样本；4K/16K合计8次采样、3976条kernel区间，数值/API/别名审计和离线完整性检查均通过。成对报告见[v2-pair/summary.md](v2-pair/summary.md)，16K原始完整窗口1413187B、SHA d61a4a8132e122126962d7ddc11d7a9019f602b0e5fc356700cbc91c4f542f2d。它与v2-4k归档共同构成可复现输入。

16K边界图median52.012573ms、full54.510248ms，配对插桩增量4.802052%。按每样本带插桩区间和排序GQA44.102310ms、BF16MM5.541262ms、INT8GEMM1.743987ms、GQA reducer0.350513ms。与4K一样主计算显著大于reducer；仍然不将这些区间和当native墙钟/正式分数。

调度分析器已修正跨window累计counter：第二窗9061−4243=4818，而非直接拿9061与4818比。prompt边界的8次重算被按区间分摊，已知数据不再标unknown。15项CPU检查（含真数据、缺前窗、重复/缺失sequence、跨PID/时间及逐请求prompt长度错误）通过；root重跑两个窗口CPU账目均通过。16K抢占8次、重复调度位置51038，其中prompt48880、生成历史2158；4K均为0。逻辑分页范围不可冒充物理KV占用，metadata重算量也不等同GPU成本。完整轨迹尚在collector flush时不读SQLite、不宣称其完整性。
