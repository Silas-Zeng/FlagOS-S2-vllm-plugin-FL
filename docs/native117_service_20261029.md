# PR117 lookahead：最小服务身份接入

本候选为冻结 `b4e96b16fc26161319affda2b91e8ca9821cb5da` 的独立 child；团队集成基线为 `a455a3de364051fbea80226870c239de3a48e716`。沿用实际 GPU 源 `2c5b558c2238915443eda010da2450360d5161b8` 的已审核库，不重新编译：

- SHA：`5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b`。
- operator：`cjj_native_k8_klookahead_probe::produce`，参数/schema 除 namespace 外与原8b逐字相同。
- 固定格式：16KiB K8/F32scale/BF16V；仅 N64/S2、N32/S4；Q pitch 2048/2560；width16 scratch、table pitch8192。

生产仅三处身份切换：`biv150_konly_encoded_native.py` 的库/namespace，`compile_semantics.py` 的同一 C3 身份，以及 `biv150_konly_encoded_sources.json` 的库/adapter 源码 SHA。默认关闭、原环境变量名、调度/CLI、writer、codec、reducer、Q32 hoist、mixed fallback 及所有几何/错误规则保持。无 PR118 扩展，无 compact 路径。三文件的严格逆变换逐字还原 a455/b4e；其余生产和 GPU 源码无改动。

原117组件已报告同输入完整步骤4K约1.03737×、16K约1.04258×；组件数值不授予本服务候选的模型、Graph请求或吞吐资格。保留 `new_service_qualified=false` 和运行时已有资格边界，不借 PR107 的105分数。

CPU 验证复用 b4e 的 `check_biv150_native_k8_service_cpu.py` 和 `check_biv150_native_k8_fingerprint_cpu.py`，只在本地临时适配精确身份/逆变换参考；所有原检查保留。已核真实5898 ELF/schema/两个launch符号、四个shape/pitch接线、格式/epoch/alias/scalar拒绝、原private fallback、postlaunch error不重试，以及未修改94db helper的实际 C3 fingerprint。CPU dispatcher 为记录器，没有加载SO或执行设备。临时适配器 `/private/tmp/cjj-agent-native117-service-cpu-adapter-20261029.py`；报告 `/private/tmp/cjj-agent-native117-service-cpu-20261029.json` SHA `07a350271591729b15abd5b6920eb23a56bb57741e1ed80b02d5cae029ee0446`。

## Root 后续入口（本提交未改）

- Controller：`/private/tmp/run_native_k8_v_overlap_service_smoke_v1_20261028.py`。独立副本只更新 K8_SHA/operator/schema、Graph manifest pin；固定 CLI、94db helper、owner/端口/清理规则保持。
- Launcher：`/private/tmp/flagos_launch_reviewed_service_v7_overlap_20261028.py`。独立副本需同步 a455 基线、新候选 commit/tag 及对应 controller 身份；旧入口仍写死1cb和旧tag，不可直接当新候选启动器。
- Graph observer：`/private/tmp/cjj-native-v-overlap-graph-coverage-bootstrap-v3-20261028/`。保留原 `sitecustomize.py`、bootstrap与probe结构；独立副本更新 probe 的库SHA/operator/adapter源pin，再重建 `bootstrap-sources.json`。保留原capture→同graph→实际请求replay逻辑，不把SELECTED日志当执行频率。
- Helper：`/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py`，SHA `94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976`，无需修改。

已有117短consumer覆盖N32/N64两种Q pitch及changed-B Graph先于fresh eager、side stream。完整4K step的Q pitch是2048，16K是2560；两档writer新K/V源pitch均256。实际服务smoke仍需记录packed Q/K/V pitch2560、新5898库与N32/N64的真实FULL graph请求回放；不把CPU capture记录或旧8b证书充作该证据。4K大上下文Q2560未由原完整step覆盖，可随原短输出64 burst smoke及后续105验证，不另造数值框架。新105及正式固定参数A/B、TTFT均待Root串行实测，正式计分关闭诊断。
