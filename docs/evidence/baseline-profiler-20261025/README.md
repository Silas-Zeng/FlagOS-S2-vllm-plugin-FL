# BI-V150 baseline profiling

基线为团队 main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`。这是诊断草稿，当前没有生产算子改动或吞吐收益。

## 已验证的可观测性

- 同一小型 BF16 GEMM + add，Torch profiler 的 eager 记录有 9 个 kernel，Graph 的 3 次回放没有内部 kernel 记录。
- Corex ixsys 独立复核：eager 窗口 9 个 kernel、合计 138.596 µs；Graph 窗口仅有 3 个整图记录、合计 108.172 µs，内部 kernel 记录为 0。这是可见性检查，不能据此比较 eager/Graph 性能。
- 原生 Corex Graph Event 节点可用。专用 256×256 小图在克隆图上插入每个 kernel 前后事件，保留全部原始依赖；两组固定输入、各三次回放均逐位一致。每次回放前将中间与输出缓冲填 NaN，排除了输出残留造成的假通过。
- 两个节点的回放计时约为 6.85 µs、4.61 µs；所有调用返回成功，资源释放无错误。该脚本逐调用记录日志，外部事件间隔包含 CPU 写日志空窗，不能用于性能或观测开销判断。

`tools/check_biv150_graph_event_timing.py` 默认不使用 GPU；显式 `--run --out NEW.json` 执行专用图能力检查。原图不被修改。源文件 SHA256 为 `717be1049456c5a4059f9e95f7a21fb0754c1e61acf63bf675819dc8a7f60c2c`。

## 下一步与判定边界

在实际基线服务保留原图，用克隆图对预定的少量执行步采样；将节点对应到真实 kernel 名称、形状及源码来源。同时观察 prefill、decode、混合执行和 CPU 调度/同步空窗。4K、16K 分开报告。比赛计算、调度参数及 benchmark 不变，额外配置仅用于观测。

必须先量化事件插入开销，避免把探针开销误认为算子耗时。计时热路径禁止打印、写盘或每步同步。诊断运行的吞吐不能作为比赛成绩；后续优化仍需独立精度、吞吐和 TTFT 验收。

## 原始证据

`graph-profiler-capability-20261025.tar.gz` 为三次能力测试的完整脚本、日志、trace/SQLite 和结果，共 62869 字节，SHA256 `d77694708720fcd00457b7f36eec36e2963170cfba7ec43fa2be618fd99792b4`。抽取结果另见 `graph-event-result.json`、`ixsys-visibility.json`。仅在用户个人实例执行；队友实例未访问。
