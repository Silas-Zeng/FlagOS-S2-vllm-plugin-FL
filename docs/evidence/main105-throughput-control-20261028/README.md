# main105 正式服务对照

受测源码 `1cbac531ca9afaa0e9a68624d5d10cb1d0412055`，5c W8A8、0c66 BF16 GQA、原 v6 tuning；NativeK8、Q32 hoist、Graph 诊断均关闭。固定比赛 CLI / e236 benchmark 不变，每档一个完整同负载 warm，再计一轮。

| 场景 | total tok/s | Mean TTFT ms | 成功/失败 |
|---|---:|---:|---:|
| 4K | 6371.243813391296 | 3307.1300876945315 | 256/0 |
| 16K | 4815.312927015575 | 104764.17440234218 | 128/0 |

Root 重放独立脚本，报告 SHA `92eb5d3b001763673e669295d7a52bed7cbc92aae469515afc6d257a63c02405`；30原件869640B、238源文件hash、CLI、token/时长算术、warm/score分离及服务清理核对通过。完整原件归档 `originals.tar.gz`（266634B，SHA256 `96c0c7a420af071b4c037104db24a7c324b1a1044e9bd10895bf4b4fba45f213`），含额外下载manifest。脚本使用其记录的本地冻结源码、helper和controller路径，原件可解压至 `/private/tmp` 重放。

4K score 窗口存在只读大归档传输（240s超时），未同时运行其他 GPU 任务；4K benchmark 全窗口还出现4项新 index metadata、selected tuning rows 改变。现有记录不能量化其计分段开销，不能断言影响为零，也不能直接解释为正常波动。16K score 前归档已停止，16K无新增metadata且tuning rows不变。对照仍应如实保留，候选若仅小幅获益需按疑点补测，不能挑选有利结果。

原 a975 PR98 的6512.18/5012.05是不同源码/旧aace库/预热方案的历史结果，不能代替当前候选对照。此记录没有重做main模型精度，也不授予PR107资格。
