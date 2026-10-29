# 现有 RAM 计数的显式窗口快照（默认关闭原型）

父基线 `a455a3de364051fbea80226870c239de3a48e716`。仅tools/docs：不改 `vllm_fl`、SDK、正式服务CLI、e236 benchmark、writer/reader/数学或每步计数。目的只是在下一次本来就要运行的 fullwarm1+score1 前后读取现有 GPU worker RAM 计数，补当前 mixed/Q32 形状分布证据。**未真实启动HTTP服务、未真实RPC到GPU worker、没有新性能或精度资格。** 与PR118独立，不替换其运行源码。

## 为什么需要固定小接缝

已取得个人实例14份现行SDK源码并核SHA，pins见 `tools/biv150_runtime_stats_sdk_pins.json`，未把SDK原文打入共享提交。`serve/__init__.py`仅在dev注册中引入generic RPC；正常instrumentator的 `/load`只返回API server_load_metrics，`/metrics`只导出Prometheus registry，其余是health/version/docs。profile路由只在profiler已配置时启停profiler。这些都不能返回私有Runtime/native/Q32的原shape记录。

实际链路：build_app返回app→init_app_state赋值state.engine_client→AsyncLLM.collective_rpc→core_client的UTILITY消息→EngineCore→UniProcExecutor→run_method→WorkerWrapperBase。RPC调用普通callable会受insecure serialization限制；这里**只发送固定字符串方法名及纯JSON参数**，不启devmode/insecure serialization，不增加可指定任意方法/代码/路径的接口。

工具只在冷导入接缝增加：

- WorkerWrapperBase上一个唯一固定只读方法，self是wrapper；通过其实际worker/model_runner与service._STATE验证对象。源路径、BI-V150已有target记录、已armed registry、runner绑定及未finalized状态均须匹配，缺对象不会返回0冒充成功。
- api_server.build_app原函数执行完后，在返回的app注册固定POST `/_cjj/runtime-stats-snapshot-v1`。引擎到请求阶段才查询。需loopback peer与私有header token；请求体仅phase/run_id。固定方法不把token送入引擎RPC，也不将token/摘要写入日志或快照。

两处安装均校验14份SDK源码，重装/名字冲突/晚于目标模块导入都拒绝。私有sitecustomize若发现另一个现存sitecustomize，拒绝启动并要求先审查组合，避免悄悄覆盖既有启动行为。没有per-forward wrapper，也不调用CUDA接口、同步、reset/finalize、清缓存或新profile。

## Root后续接入方式（本提交未接controller）

bootstrap与实际vllm_fl必须位于同一个冻结checkout。由controller在内存生成token（例如secrets.token_urlsafe(32)），只放进本次服务子进程env及私有请求header；不得放入命令参数、launch JSON、共享日志或保存完整env。新增服务环境键：

```
CJJ_RUNTIME_STATS_SNAPSHOT=1
CJJ_RUNTIME_STATS_RUN_ID=<本次固定tag，只含小写字母数字连字符>
CJJ_RUNTIME_STATS_TOKEN=<仅内存随机值，不在文档/日志展开>
```

服务PYTHONPATH在原source前加入 `<source>/tools/runtime_stats_bootstrap`，其余保持原值。**benchmark/client子进程须移除这三个键及该bootstrap路径**，保留原client环境。正式CLI不改。已有其他sitecustomize必须先审组合；原型不会自动链入未知代码。

controller以urllib等内存请求向本机9031发送固定POST：header `x-cjj-stats-token`，body只含 `{"phase":"warm-before-4096","run_id":"<tag>"}`。允许六个phase：warm-before/warm-after/score-after ×4096/16384。禁止把token拼成curl命令或记录request headers。

每档原fullwarm调用前取before、完成验证后取warm-after、score完成验证后取score-after；warm-after兼score-before，中间只允许已有CPU cache清单读取。所有模型请求已结束且无其他客户端，**先等明确ack，再继续benchmark**；UniProc的timeout参数没有实现硬中断，不能超时后留下未完成快照与计分并行。校验响应SHA、run/phase、PID/start_ticks、source/digest/实际库与既有owner/manifest一致，再写私有新文件；不覆盖旧快照。保存原JSON和SHA，不保存认证header。

snapshot在worker调用序列内一次JSON序列化，返回冻结字符串；读取 runtime.stats、完整shape_records/overflow、kernel_records以及Q32 geometry_records原key映射。native.stats和Q32.stats已在runtime.stats内。导出有真实CPU/内存开销且32MiB上限，超限明确失败不截断；保持两臂相同观察位置，不能声称绝对零扰动。

## 增量与资格边界

`counter_delta(before,after)`不改原数据；仅接受同档warm-before→warm-after或warm-after→score-after，要求worker PID/start_ticks、实现/工具/SDK身份一致，拒绝计数下降、记录消失/身份改变、shape/native溢出增量。它将runtime selected与shape、Native成功次数与records、Q32 admitted与geometry计数分别核对。全生命周期完备标志单列，不能把窗口完备当作从启动起无遗漏。Q32没有4096上限，可能在同key编译对象变化后重建，因此检测回退/身份变更，不强行做差。

Native成功在runtime.note之前返回，必须同时导native.records；fallback reason只有总数，具体N还需对应TL shape记录。原完整record含hash/stride/capture/SEGMENTS等，增量身份保留这些维度；不能只按N混合。Q32 admitted是prepare准入次数，需结合原runtime实际profile/hash，不单独声称成功执行频次。

窗口包括整次benchmark客户端调用及其测试请求/前置，不等于benchmark内部timer的纯计分段；不为划窗修改e236。FULL_DECODE_ONLY replay不会再跑Python计数，capture=true不是replay次数；返回明确graph_replays_counted=false。只能把非Graph自然mixed的host调用作为形状覆盖线索，不能无依据换算模型step、GPU时间或head M。现有统计不足以补head all_random/sample_hidden形状，本工具不偷偷增加head计数。

不得用finalize做重复窗口快照，它缓存首次结果；已有shutdown/atexit仍留作补充，API退出0也不保证worker最终导出。原型不会改变这些生命周期路径。

## CPU验证与待验证项

运行：`python tools/check_biv150_runtime_stats_snapshot_cpu.py`，依赖已保存的 `/private/tmp/cjj-current-api-rpc-source-20261029` 14份源码，无Torch/Triton/FastAPI SDK导入、无GPU。实际RPC方法AST串联使用普通JSON模拟传输；app/worker/Linux身份为明确的CPU fixture，**不是完整FastAPI/Msgpack/ZMQ或真机执行**。

Python3.12与3.14报告逐字一致，见 `docs/evidence/runtime-stats-snapshot-20261029/cpu.json`。覆盖默认OFF、冷loader原exec后安装、原app参数/返回保留、缺worker/runner/runtime、未armed/已finalized、错误身份、任意method请求、坏token/非本机、计数回退/溢出/对账、已有sitecustomize拒绝且不执行，以及token不出现在响应/模拟RPC/启动错误中。封版前静读a455纠正了两项最初fixture假设：actual_target实际是列表，attribute_bindings实际为三元组；现测试直接求值原赋值AST并核生产源逐字等a455，未转授早期mock结果。

Root审查并建本人Draft后，仍需在既定服务前置验证：SDK真实冷导入、FastAPI识别Request、实际worker方法可达、owner/源/库对应、响应完整及无遗留RPC。观察失败应保留缺口；不能造0、不能凭CPU通过宣称当前30/61等频次已获证实。
