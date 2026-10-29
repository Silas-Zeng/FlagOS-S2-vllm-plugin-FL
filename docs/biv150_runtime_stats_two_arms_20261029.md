# PR119：仅两组已审核身份的同边界观察准备

独立child父为 `d9c91e011915396754efe8023c817831c733bdae`，保留其worktree/报告。此提交没有SSH、GPU、推送或整机结果。生产代码仍逐字等main A；Root后续在独立合流分支接入已审117，B只保留其既定三个生产身份差异。不能直接在本child的A生产源码上宣称B服务可运行。

| 臂 | 只读父controller SHA | 生成controller | 生成SHA | 唯一Native库SHA |
| --- | --- | --- | --- | --- |
| A 主干/Q32 ON | `558579df4ebd3ac359167a429e9b3c6905952509de5910ebf843162ba0346600` | `tools/run_biv150_parent107_stats_service.py` | `6556a609261372769bf9e3e638e2771ceef3dffabd90df7a06093035f2df9f36` | `8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5` |
| B 117/Q32 ON | `2cfc74c98e366015fe7bf3d6227ad06f8c1bcf87a341b72bd68fbf42062dcb7c` | `tools/run_biv150_native117_stats_service.py` | `3208684e9c64b66855478e4dc3ed2548eac2bd342b1f98aabe55deee74d0e237` | `5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b` |

A的6556原件不改。B只把**同一14处接缝**应用于已审2cfc父，逆变换逐字回2cfc。B父原有的库/namespace/Graph manifest与说明差异原样保留；未修改其benchmark、Math105、Graph函数或owner清理。Worker snapshot源码与sitecustomize逐字保留f8c/d9；默认关闭、固定HTTP方法、token、六边界、计数拒绝规则均不变。

Session仅认上表两个确切入口及controller SHA，且必须在同一Git source。每个入口的原CLI仍只接受自己的单一库SHA；参数/C3的library SHA与operator必须匹配该入口，实际worker返回的库path/SHA必须匹配参数所绑定的库。不能把“两个SHA都在名单”当作可以跨臂混用，未知库或入口均拒绝。公开manifest新增controller相对路径以区分两臂。原始库文件SHA仍由各controller原准入检查；这里不以CPU fixture替代实际SO验证。

两臂都使用 `--runtime-stats-bootstrap --throughput --full-workload-warmup --warmup-plan full-only --throughput-case both --q32-hoist`；各自原库/源码/commit/helper/tuning等参数保持对应review，不使用 `--baseline`。bootstrap仍只供服务env，client清除三键和该路径。真实RPC/owner会话闭合及整机效果仍待Root串行验证。

CPU复跑：

```
python tools/build_biv150_runtime_stats_controller.py --parent /private/tmp/cjj-native117-parent107-controller-20261029.py --output tools/run_biv150_parent107_stats_service.py --check
python tools/build_biv150_runtime_stats_controller.py --parent /private/tmp/cjj-agent-native117-service-controller-v1-20261029.py --output tools/run_biv150_native117_stats_service.py --check
python tools/check_biv150_runtime_stats_controller_cpu.py --arm A
python tools/check_biv150_runtime_stats_controller_cpu.py --arm B
```

沿用原CPU测试，只按两组已知父选择入口，并增加少量身份拒绝：真正运行两个parser的CPU-plan各自接受本臂/拒绝另一臂，Session拒绝交叉参数/C3/operator/未知入口，worker回报另一已允许库也拒绝；两组各自跑同样六ACK/四原benchmark调用的opaque fixture。不是实际FastAPI、GPU或新吞吐验收。历史d9报告保留，新报告单独存放。

两臂各59项，Python3.12/3.14各自报告逐字一致；报告只证明CPU接缝。A报告SHA `a1a30bd07f90158f09e1d510d67a0aef705ab7cc31eb7e1f8e927eaf0b51dff6`，B报告SHA `0bee6db70384efff13e86869d5bbebdf517d3f52bf2914563cd3912aeb6e41f9`，保存在 `docs/evidence/runtime-stats-two-arms-20261029/cpu-{A,B}.json`。准备使用已暂存Git对象核对，冻结后以真实HEAD复跑并核报告一致。
