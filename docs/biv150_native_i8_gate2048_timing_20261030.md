# PR128 完整 gate 成本筛选

基于已过数值的 `5de17de` 独立 child，生产/CUDA/CPP/原数值入口均不改。
原数值精选结果见 `evidence/native128-numeric-20261030/`。默认仅输出 CPU plan；
实际运行须完整原数值归档、Root 重放报告、受限 timing receipt 与独占安排。

A 每步直接返回实际 5cbb `gate_silu_cached_probe(x,wq,cols,marker,256)`。
B 每步调用同库 `quantize_rows_probe(x,256)`，再分配 BF16 输出并调用实际 8eef
`gate_out`。两臂都包括激活量化和输出分配；不计入旧数值验证给 A 额外添加的
`copy_`。输入直接复用已审原件，固定权重/cols/marker 的值与地址不变。
本模块不声称能观察厂商权重 packing 命中；所有初次完整调用均在计时前完成。

独立 Graph 各捕获 4 个完整 step。组合后的量化+gate Graph 必须先消费换值 x(B)，
其完整输出对已审 B 原件逐位相同，然后才执行 fresh eager B。Graph 与 eager 两个
mode 各先预热 128×4=512 step，再 A-B-B-A，各臂 16×4=64 step。记录四条 CUDA event
及 host wall 原时长/64，计时后检查完整输出、全部输入只读及地址布局，不把归档复制
放进计时。没有数学参考重写、扫参或服务安装。

Graph event/wall 仅代表捕获图中的组件；Python 分配/调用不在 Graph replay 内发生。
Eager 的两次 candidate 调用和输出分配保留在实际计时循环内。主线 FULL_DECODE_ONLY
下 M2048 prefill 的适用口径是 eager；两种结果都不是模型吞吐收益。

CPU：`python3 -B -S -W error tools/check_biv150_native_i8_gate2048_timing_cpu.py`。
默认 plan：`python3 -B -S tools/probe_biv150_native_i8_gate2048_timing.py`。
设备入口沿原 numeric 参数，将模式改成 `--run-timing`，另加 `--numeric-raw`、
`--numeric-review`、`--timing-admission`。先以 `--preflight-only` 核全部实际文件。
模型、服务与性能资格仍为 false，结果由 Root 独立复核。
