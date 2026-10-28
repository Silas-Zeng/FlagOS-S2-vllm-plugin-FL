# PR111 069 原身份 pin 修复后的完整吞吐负结果

本次完成了吞吐流程及清理，原验收状态为 `completed=true / cleanup_complete=true / errors=[]`。两档各一次完整 workload warmup 后计分一次，256/128 请求全部成功。**这是吞吐证据审核通过，不是模型或合并验收通过：本次没有 Math105，precision_gate / full_acceptance 仍为 false。** 相对已接受的 PR107 记录，4K 没有明确收益，16K 下降约 7.80%；保留负结果，不追加本版精度测评。

| 输入 | 本次 total tok/s | PR107 total tok/s | 单轮变化 | 本次 mean TTFT ms | 本次耗时 s |
|---|---:|---:|---:|---:|---:|
| 4096 | 6349.934797850074 | 6374.4399021712225 | −0.3844275685% | 3916.6952344930905 | 206.41471790289506 |
| 16384 | 4762.888742465713 | 5165.670512577975 | −7.7972795425% | 82147.6295537359 | 467.8303694422357 |

PR107 是受测 b5d1a241 后合入 main a455 的已接受配置记录；本次没有重新测量 main。双方固定 benchmark 为 e236、每档 full warm 1 + score 1。`comparison-to-PR107.json` 由本目录短脚本从两份 SHA 固定的原审核报告计算。单轮不建立噪声分布或统计等价：4K 按持平记录，16K 是本次负对照。PR107 的 encoded K8/8b86 与本版 compact K8/7892+2ad 有多处差异，两者也分别启用了各自 Q32 路径，不能把全部差异归给 N48、Q32、布局或某个算子。

## 实际身份与原件

- 受测源码 `069838d6fbb7591de4384aac8caef49324c43aef`；owner `71022/start356895544` 同 start 结束。270 个生产源码文件及同 commit 内的符号链接逐项绑定，不把本 docs child 当成 GPU 受测源码。
- controller `ae99bfb55c69dc1bf9464ffe0a1b7e627882bd4bc9d185d4177f9a902c29ea56`；helper `94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976`；正式 benchmark `e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d`。正式服务 CLI、并发 64、输出 1024、seed 0 和 warmup 协议未改。
- 原 W8A8 5c / NativeGQA 0c66；compact base SO `7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4`，N48 SO `2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`。32/S4、64/S2 使用 base，48/S4 使用 N48；40/S16 等保持原 fallback。
- 完整 3520 原件、92,752,394 B 全部重 hash。完整 gzip 16,030,062 B，SHA `2f31c1e3875752e552ca7444d0ab1c545742e7ad22611f7459b785267d5fbdd6`。八包是同一个 gzip 的连续、不重叠字节切片，每包不超过 2 MiB；没有删去 cache、实际 Q32 产物或日志。
- 完整 gzip 同时持久保存于 workspace `repos/experiment-raw-archives/2026-10-29/cjj-native111-q32-n48-pinfix-throughput-v2-raw-20261029.tar-stream-v1.tar.gz`。完整路径、分包 SHA、原件清单 SHA 见 `archive-parts.json`。

原 a6d 的失败状态和旧成功版审核脚本完整保留在 [此前失败证据](../native111-a6d-failed-service-20261029/README.md)，没有改成成功。本次 069 仅修复精确 EXPECTED_RUNNER 与对应交叉 pin；源码逆向及 CPU 回归见 [pin 修复证据](../native111-compute-order-pin-fix-20261029/README.md)。原完整计算方法没有改写。本目录从共享 b99 正常新建 docs child，旧历史不重写。

## 运行证据与限制

完整日志重新调用原 controller 的 evidence 函数通过，没有过滤拒绝或 fallback 行。本次真实 FIRST_SELECTED 记录来自 PID71070，N62、M2048、Dprefix2、Dall61、outside59；原 forward 返回，sampler_rows62，inverse 为 original_request_rows。它证明这个已观察调用启用了 compute-order，并不证明所有请求、token 位一致或实际 replay 次数。与 a6d 的时延差异也不能全部归因于 pin 修复。

4K warm 为 5985.020049 tok/s；16K warm 为 4187.445995 tok/s，两者不计成绩。每档 warm 和 score 的原命令、固定输入/输出 token 总数、duration 算术均核对。4K score 观察窗口新增六个 index metadata 且 selected tuning rows 改变；16K 无新增、无变化。窗口包含 client preflight，不能把这些变化全部算成计时区间 JIT，也不能声称 full warm 已排除所有编译或波动。

Host SELECTED/CAPTURED 观察到 32/S4、48/S4、64/S2；没有 Graph observer。42 层 compact allocation 的 44769 页、12,544 B/page 是配置/分配元数据。它们不是每次请求走原生路径的频率，不能用 capture 数推算 replay 数。三条 smoke 和四组 32/64/40/48 路短请求共 187 条已核计数与返回，不是 Math105。

实际新 Q32 编译对象完整保留，重放原 proposal/native/ABI coherence 门、核每个 artifact SHA，解析实际 cubin 并与保存 ASM 元数据一致。runtime 为 135→122 regs、57344→33792 B dynamic shared、n_spills 两方 0；**ELF SGPR spill 为 99→96，VGPR spill/private/static shared 为 0**。两种 spill 字段不可混称全零；没有另做独立 cubin 反汇编，也不从编译重审授数值或模型精度资格。

本次后验 source/library/snapshot 原 controller 检查已执行，API 正常退出且 owner 同 start 清理完整。该说法限定于源码实际实现的检查；N48 的证据是既有精确库准入和实际加载/接线，controller 没有另存一份末尾 N48 文件重 hash。未发现 compact final stats 导出，也不补造。官方固定表的吞吐与 TTFT 界限均满足，但官方表不是最新 main，对照退化与未测精度不能因此消除。

## 本地 CPU 重放

作者和 Root 分别复跑同一原件，结果逐字相同，均为 `1a595917940d95cdac0c271f4ed0f0ac7c6acb0cfcec1c04aab0c683bf616aad`。这是两次 CPU 审核，不是第二轮设备计分。`review_throughput.py` 保留预先冻结的 6d411900 原脚本，未因本次负结果改门。

在仓库根目录执行；所有输出必须是尚不存在的独立路径：

```sh
python3 -B docs/evidence/native111-pinfix069-throughput-20261029/restore_raw.py
python3 -B docs/evidence/native111-pinfix069-throughput-20261029/restore_raw.py --extract-to /private/tmp/native111-069-replay-raw
python3 -B docs/evidence/native111-pinfix069-throughput-20261029/review_throughput.py --raw /private/tmp/native111-069-replay-raw --repo "$PWD" --launch docs/evidence/native111-pinfix069-throughput-20261029/original-launch.json --base-so /private/tmp/cjj-native110-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_v_overlap_probe.so --n48-so /private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so --out /private/tmp/native111-069-replayed-review.json
python3 -B docs/evidence/native111-pinfix069-throughput-20261029/compare_to_PR107.py --out /private/tmp/native111-069-replayed-comparison.json
```

`restore_raw.py` 默认只检查八包和完整 gzip；显式 extract 才展开，并逐项核 manifest。重放需要本仓库 069 Git 对象、相同生产源码、现有 ELF/msgpack helper 和已有 110/112 编译证据中的实际 7892/2ad SO；未在本目录重复打包旧库。相关组件原件已经随依赖进入仓库。全部是 local CPU 检查，不导入 Torch/Triton/vLLM，不联系服务器、不执行 GPU。
