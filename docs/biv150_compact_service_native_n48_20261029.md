# PR111 晚 V 等待与 N48/S4 服务接入准备

本候选从 PR111 `acd6974733661fbde47c3ce76d87d514c0861fbe` 独立建分支，正常合并
PR110 `7893f28bc27a52ca93ef3882d7fd9b6079c04919` 与 PR112
`83b6c8c9282445219782ee1377680778a2aa8d3a`。团队 main 仍以 Root 02:17 实查的
`a455a3de364051fbea80226870c239de3a48e716` 为集成基础，没有升级官方框架分支。

compact adapter 的 N32/S4、N64/S2 接入实际 `7892a6aa…` 晚 V 等待库；新增 N48/S4
选择实际 `2ad51a98…` 的独立 namespace。两个路径都接受 Q pitch 2048/2560。N48 有自己的
默认关闭开关，必须先启用 base Native，再完成自己的组件资格和独立路径/SHA 检查。
N40/S16、其他形状或未启用 N48 时继续调用原 private compact Triton reader。

| 接入范围 | 实际 SO | 当前组件状态 |
| --- | --- | --- |
| N32/S4、N64/S2 | `7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4` | consumer 与 fullstep 已独立审核 |
| N48/S4 | `2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1` | 静态对象、consumer 已审；fullstep 资格仍 pending |

N48 pending 时，worker、真实 helper C3 和新 controller 在加载 Torch/库或启动服务前拒绝。
没有用编译报告、父 consumer、历史 `source_proof` 布尔值或 CPU mock 替代实际数值资格。
`compact_native_qualification.py` 中的 N48 fullstep 哈希为 `None`，state 为 pending。
N48 consumer summary `493e210f…` / 独立 review `c07937c3…` 已真实填入，仅覆盖
29 页、max context 193、41/42 条有效行、两种 Q pitch 的 N48/S4、Graph/side，未转授服务资格。
后续只能在正常审核提交中填入相应真实 summary/review 哈希及实际几何，更新 manifest 后再启用。
没有通过环境变量临时覆盖资格记录的入口。

7892 采用已共享的实际静态报告 `0b3d0098…`、consumer summary `6f719fe8…` 及独立 review
`896c7c99…`。新 fullstep summary 为 `f12332f0…`，独立 review 为 `0cd4cb16…`；
其真实范围是 **34276 页、新 KV 行距 256**。未将其写成服务行距 2560 或 42 层容量资格。
N48 的目标 fullstep 几何为 44769 页、新 KV 行距 256，资格未完成前该记录保持空。
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
改变的原 controller 函数仅 native_env、native_events、evidence、run，新增早期资格预检函数。

CPU-only 计划入口（不添加 `--run-service`，不会启动服务）：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/run_biv150_compact_native_n48_service.py --q32-hoist
```

实际运行还需在 Root 协调后使用冻结 source/commit、真实库路径、固定 helper 和 tuning snapshot；
新参数为 `--native-k8-n48-library` / `--native-k8-n48-sha256`。原 base 参数为
`--native-k8-library` / `--native-k8-sha256`。Q32 使用本文件的新 controller 的 `--q32-hoist`。
原吞吐协议保持 `--full-workload-warmup --warmup-plan full-only`，每档一轮完整 warm 和一轮 score。

CPU 验证与源绑定见 [证据目录](evidence/compact-service-native-n48-cpu-20261029/)。新检查复用
原 metadata/storage fixture 与 42 层 mock installer，读取真实 SO 文件但不加载：

- 实际 helper94db 确认 overlay origin；base Native ON/OFF × Q32 ON/OFF 的四个 C3 独立。
- N32/S4、N64/S2 两个 pitch 的实际 adapter 元数据路径、capture、坏 stride 拒绝、postlaunch 不重试。
- batch 1/8/16/40 和未批准 N48 的原 grid/SEGMENTS/Triton fallback；未改 producer/reducer 分段。
- base Native + Q32 的现成 42 层 CPU mock 安装；原 Q32 核心与 installer-on 检查通过。
- N48 生产 preflight 和 helper C3 明确拒绝。只有隔离的 CPU 路由单测显式跳过资格，核对
  N48 算子/日志选择；该 mock 结果不授予组件、安装或服务资格。

重放新 CPU 检查时提供已保存的真实库文件和新的输出路径：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_compact_native_n48_service_cpu.py --base-library /absolute/path/to/cjj_native_k8_compact_v_overlap_probe.so --n48-library /absolute/path/to/cjj_native_k8_compact_n48_s4_probe.so --out /private/tmp/new-native111-cpu-review.json
```

本分支没有启动 GPU、服务、模型精度或吞吐测试，没有新的服务成绩。历史 PR109 成绩及
PR103 Graph/数值结论不转授本组合；正式合并仍需实际服务精度、吞吐和 TTFT 对照。
