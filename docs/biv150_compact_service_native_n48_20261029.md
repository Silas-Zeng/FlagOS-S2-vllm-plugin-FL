# PR111 晚 V 等待与 N48/S4 服务接入准备

本候选从 PR111 `acd6974733661fbde47c3ce76d87d514c0861fbe` 独立建分支，正常合并
PR110 `7893f28bc27a52ca93ef3882d7fd9b6079c04919` 与 PR112
`83b6c8c9282445219782ee1377680778a2aa8d3a`。团队 main 仍以 Root 02:17 实查的
`a455a3de364051fbea80226870c239de3a48e716` 为集成基础，没有升级官方框架分支。
随后正常合并已共享的 PR110 `e89d8188aed255a5c5713f3d99e096d6467eb659` 与 PR112
`09e502134fcf9fa7168c9f342c2a3d33754cee1c`，纳入完整组件证据和 consumer 工具；没有改动其内核。

compact adapter 的 N32/S4、N64/S2 接入实际 `7892a6aa…` 晚 V 等待库；新增 N48/S4
选择实际 `2ad51a98…` 的独立 namespace。两个路径都接受 Q pitch 2048/2560。N48 有自己的
默认关闭开关，必须先启用 base Native，再完成自己的组件资格和独立路径/SHA 检查。
N40/S16、其他形状或未启用 N48 时继续调用原 private compact Triton reader。

| 接入范围 | 实际 SO | 当前组件状态 |
| --- | --- | --- |
| N32/S4、N64/S2 | `7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4` | consumer 与 fullstep 已独立审核 |
| N48/S4 | `2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1` | 静态对象、consumer 与 fullstep 均已独立审核 |

N48 的原 pending 记录保留为真实 Git 历史拒绝 fixture。当前 source record 只在 Root
接受完整原件独立审核后更新为 component-reviewed，绑定 fullstep summary
`5a6aefeb7befb777e890402fa9742774ea23ae5e696be67c74f04418df37b131` 和独立 review
`2d8d5b873eb30084da7dd0e25332088462dd5cfef62c45fe45c75bccd07ede5d`。
consumer `493e210f…` / review `c07937c3…` 仍保留原小范围；新 fullstep 另行覆盖
44769 页、41/42 有效行、Q/K/V stride2560 的两个完整 step case。没有环境变量资格覆盖入口。
默认开关仍关闭；这次只允许组件准入，不代表模型精度、服务或共享 QKV 分配已通过。

7892 采用已共享的实际静态报告 `0b3d0098…`、consumer summary `6f719fe8…` 及独立 review
`896c7c99…`。新 fullstep summary 为 `f12332f0…`，独立 review 为 `0cd4cb16…`；
其真实范围是 **34276 页、新 KV 行距 256**。未将其写成服务行距 2560 或 42 层容量资格。
N48 fullstep 的实际几何为 44769 页、Q/K/V 行距 2560。
其独立 Q/K/V backing 不等于模型共享 QKV 分配资格；service_verified 仍为 false。
这两套组件证据都不等于服务 Math105、容量、吞吐或 TTFT 验收。

分段选择和 reducer 完全沿用现有 `biv150_gqa_decode.py`：N48 原本就是 S4，N40 特例仍是
S16。adapter 只按收到的 `(batch, SEGMENTS)` 选择算子，不修改段数、grid 或 scratch view。
因此 producer 和 reducer 继续消费同一份 split 宽度。新代码保留源码/库/namespace/schema
绑定、只在冷 worker 加载、一次 reader 调用、postlaunch 错误传播，以及 stock BF16 的格式隔离。
selection/capture 记录仅代表宿主事件，仍不能当作真实请求频率或 Graph replay 计数。

PR111 的 compact Q32 hoist 能力原样保留，默认关闭；每个 fresh JIT 对象仍走原 binder、IR、
资源和 after-launch 检查。六个 compact 计算核、Q32 proposal/adapter、复用的两个 hoist
依赖和 runtime 与 `acd697` 逐字相同。110/112 的 CUDA/C++/codec 与各自冻结源逐字相同。

新的 `tools/run_biv150_compact_native_n48_service.py` 来源于实际执行过的 PR109 controller
`d8a5bd43…`。旧版本虽然解析 `--q32-hoist`，随后会拒绝该参数，并在 `native_env` 中清零
hoist；不能把组合参数直接传给旧版本。新版本显式保留该开关，新增独立 N48 路径/SHA参数，
先做组件资格检查，再进入原生命周期。真实 helper 仍是 `94dbbb0f…`。比赛 SERVICE_COMMAND、
Math105/补充64并发、e236 benchmark、完整 warmup、分数计算和清理函数保持原字节。
最初改变的原 controller 函数为 native_env、native_events、evidence、run；本次仅增加从真实
source 读取 CPU-plan 资格状态，并把两处旧 compact40f9 Graph 拒绝文案更新为当前库，拒绝行为不变。

CPU-only 计划入口（不添加 `--run-service`，不会启动服务）：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/run_biv150_compact_native_n48_service.py --q32-hoist
```

实际运行还需在 Root 协调后使用冻结 source/commit、真实库路径、固定 helper 和 tuning snapshot；
新参数为 `--native-k8-n48-library` / `--native-k8-n48-sha256`。原 base 参数为
`--native-k8-library` / `--native-k8-sha256`。Q32 使用本文件的新 controller 的 `--q32-hoist`。
原吞吐协议保持 `--full-workload-warmup --warmup-plan full-only`，每档一轮完整 warm 和一轮 score。

当前 CPU 验证见 [正常准入证据](evidence/compact-service-native-n48-qualified-cpu-20261029/)。
历史 pending 检查与 Root 的 Python3.12 复跑原件保持不变。新检查复用原 metadata/storage
fixture 与42层 mock installer，读取真实 SO 文件，不实际加载 GPU 库：

- 真实 preflight、controller 前置门和 helper94db 的六种 C3 组合均检查真实 source record。
- N32/S4、N48/S4、N64/S2 在两个 Q pitch 都经过原 attach、_load、verify 和分流；
  CPU dispatcher 独立记录两个 namespace、schema 和 loaded set，没有覆写 Adapter 方法。
- 原 shape/split fallback、坏 stride、stale epoch、输出 alias、错 schema、dispatcher 变化、
  postlaunch 不重试和错库/SHA 拒绝均核对。真实26e pending 文件仍必须拒绝。
- Native7892 + Native2ad + Q32 使用原42层 CPU mock installer；Q32 编译门仍明确是 mock，
  不授予实际 fresh JIT 资格。Native48 生产默认开关仍关闭。

重放新 CPU 检查时提供已保存的真实库文件和新的输出路径：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_native_n48_service_cpu.py --base-library /absolute/path/to/cjj_native_k8_compact_v_overlap_probe.so --n48-library /absolute/path/to/cjj_native_k8_compact_n48_s4_probe.so --out /private/tmp/new-native111-cpu-review.json
```

本分支没有启动 GPU、服务、模型精度或吞吐测试，没有新的服务成绩。历史 PR109 成绩及
PR103 Graph/数值结论不转授本组合；正式合并仍需实际服务精度、吞吐和 TTFT 对照。

## 历史 Root 独立 CPU 复跑与依赖同步（26e3）

在共享 `8229685` 后，正常合入 PR112 `e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8`，
补齐冻结 fullstep 前端及其独立源码审查。该合并未改变生产代码、GPU 数学源码或服务 controller。

Root 使用 Python 3.12 独立重放 CPU 集成检查，原报告 SHA256 为
`e05a24edc45599a67e8ff0b74f27e6df301834527dadebc5d3284963e37c47eb`。
与本分支 Python 3.14 的 `17b02e00…` 报告逐字段比较，只有 helper 解析的 Python 可执行路径
和四项 Python 版本路径不同，其余字段完全相同；两份原报告均保留，详见
[完整比较记录](evidence/compact-service-native-n48-cpu-20261029/root-python312-comparison.json)。

26e3 当时只同步依赖和 CPU 证据，尚未收到 fullstep 独立审核；该历史提交保留 null/pending，
其旧隔离 mock 不构成准入或服务资格。本次更新使用上述真实完整审核，不修改历史报告。

## fullstep 记录勘误和资格限制

原 summary 的 `GPU_executed=false` 是沿用 plan 默认值后未更新的记录错误；原件保留不改。
owner 的 `--gpu` 命令、实际库加载、五类 fresh JIT、36 份完整输出和 Graph/side/ABBA
证据共同确认实际执行。144 个原硬门通过，未降低数值阈值。完整 cache 的独立复核依据是
运行时全字节比较记录和可重建分块哈希；完整 GPU cache/input payload 并未保存，不能声称重新
读取了 GPU 内存。

TL 的 runtime `n_spills=0`，而 ELF `SGPR spills=26`、`VGPR spills=0`、private=0；
JIT dynamic shared=6144B 与 ELF static shared=0 分列。原门只要求正寄存器数和 runtime
n_spills=0；额外离线“所有 ELF spill 必须为0”的错误假设及其失败已保留，不作为放宽原门。
不能写成“TL 全部零 spill”。Native2ad S4 的 static VGPR/SGPR spill=0、private=0、
shared=8192B 则来自独立实际 SO 审查14c8。

两个相同形状 TL48/native2ad 完整 step 的单次组件比值约1.1523×/1.1558×，不是模型吞吐。
Root 后续先执行现有服务短检查、两档完整 warm 后各单轮吞吐；若有收益再补官方105和最新 main
对照。这里没有新模型成绩，也不授予42层容量或稳定端到端性能资格。
