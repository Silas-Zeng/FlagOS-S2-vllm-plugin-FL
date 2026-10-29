# PR119：既定吞吐调用边界的可选计数快照

这是 f8c 原型的 tools-only 后继，不改生产/数学、正式服务命令、benchmark函数、Math105、Graph诊断函数或owner清理。只读父为 `/private/tmp/cjj-native117-parent107-controller-20261029.py`，SHA `558579df4ebd3ac359167a429e9b3c6905952509de5910ebf843162ba0346600`。新增controller是可审的完整源文件；builder只接受该SHA，并能逆变换逐字恢复父文件。保留完整源而不在运行时拼接外部控制器，使controller、bootstrap、统计tool和生产包都能从同一冻结commit执行。

**没有实际HTTP/GPU worker/整机验证，没有新精度或性能结果。** 本提交不自动启动任务。旧f8c文档所述“未接controller”保留为原型阶段记录，本篇描述新增可选接缝。

## 执行边界

新入口 `tools/run_biv150_parent107_stats_service.py` 只增加一个可选参数 `--runtime-stats-bootstrap`，默认关闭。打开时要求原 `--throughput --full-workload-warmup --warmup-plan full-only --throughput-case both --q32-hoist`，以及原Native8b库路径/SHA；禁止baseline、Math105及Graph bootstrap组合。**当前主干A本身就是Native8b/Q32 ON，不能用 `--baseline` 模拟它。** 原5c、GQA、原tuning snapshot及所有其余参数仍由原controller准入，不另改赛题超参数。

controller必须从 `<source>/tools/run_biv150_parent107_stats_service.py` 执行。原HEAD/clean/source/package_resolution检查保留；新增tool/manifest/bootstrap/builder/controller均逐字对照该commit的Git对象，拒绝软链接和异源模块。builder只在准备阶段使用，运行时不构建/写回源码。

每档的原完整warm和score保留一次：

1. warm-before ACK → 原full-warm benchmark → 原validate_score。
2. warm-after ACK → 原warmups/cache-before记录 → 原score benchmark → 原validate_score。
3. score-after ACK → 原cache-after/score记录。

两档依原顺序4096、16384，共六份原始JSON快照和四份host counter增量。warm-after兼作score-before，窗口包含整个benchmark客户端及其前置请求，未改e236内部计时。snapshot在原调用之间有额外CPU/内存开销，不能声称零扰动；后续对照须两臂使用相同观察位置。callback不是模型逐步hook。

RPC确认包括原始响应SHA、run/phase、工具与SDK pins、实现digest、实际Native8b库、Q32开启、WorkerFL/ModelRunnerFL、未重启的PID/start_ticks，以及原Owned registry和当下/proc身份、同API会话。仅获得固定GPU worker方法的返回；不把API父进程对象当worker。第一次快照明确ACK后才发benchmark。任何超时/失败/身份变化/计数回退或窗口溢出都停止后续benchmark并进入原cleanup；不重试或伪造0。UniProc timeout不是取消承诺，因此未ACK不能继续计分。

## Token与输出

token由controller在内存生成；只在CPU预检结束后注入服务子进程env，以及私有loopback请求header。不会写argv、manifest、日志、快照或摘要。请求不使用环境HTTP代理且禁止重定向。benchmark/client移除三个CJJ键和该bootstrap路径；其余环境逐字保持。即使参数关闭，也清除继承的这三个键，保证默认关闭。现有其他sitecustomize仍由f8c启动门拒绝，不能自动覆盖或启用任意RPC/devmode/pickle。

输出在原out的 `runtime-stats/`：六个固定phase `.json` 保存服务返回的原始字节，四个 `*-delta.json` 由原counter_delta核对，`index.json`给出SHA和ACK状态。文件只新增不覆盖；即使某个完整JSON随后因增量校验失败，已经获得的原始快照和index仍保留，原state标失败。manifest仅增加公开的runtime_stats_snapshot字段，最终完成必须六个ACK齐全。原completed/model_precision等语义不变。

Graph capture/host次数仍不是replay频次，统计仍不等于GPU时间、实际模型步数或head形状；不能因此给新模型精度资格。

## CPU复核

```
python tools/build_biv150_runtime_stats_controller.py \
  --parent /private/tmp/cjj-native117-parent107-controller-20261029.py \
  --output tools/run_biv150_parent107_stats_service.py --check
python tools/check_biv150_runtime_stats_controller_cpu.py
```

CPU测试使用原558文件、真实冻结Git对象，以及opaque HTTP/owner/benchmark fixture。实际运行新full-only控制流、原validate_score、六份快照/四份增量逻辑；验证ACK严格先于benchmark，错SHA/来源/owner/超时均拒绝、无覆盖、token不出私有env/header。另逐字核所有未改顶层函数/class和run内Math105、supplemental64、cleanup片段，逆变换闭合父SHA，默认关闭的CPU计划仅增加disabled字段。其余真实FastAPI/Msgpack/ZMQ/worker启动及服务结果仍待Root串行验证。

准备期间可给测试 `--freeze-ref <已暂存的Git tree>` 检查源码对象；正式controller仍要求真实commit等于HEAD且工作区干净，不以tree替代服务准入。最终随提交报告使用同一代码对象；未上传任何GPU raw、私密token或SDK原文。
