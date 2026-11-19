# Runtime.note：保留原JSON记录的重复序列化索引

基线为团队 main `a455a3de364051fbea80226870c239de3a48e716`，独立分支 `codex/cjj-runtime-note-key-20261029`。不包含120的storage观察改动、117 lookahead或119工具变更；未启动SSH/GPU/服务、未发布。本候选只改 Runtime 统计记账。

## 当前路径与证据边界

原 `biv150_konly_encoded_runtime.py:285–287,391` 在实际 Python writer/Triton reader返回后调用 `note`，不限于capture。`374–378` 的Q32路由要求非capture；117真实Math105 manifest有2048行、hoist已选择的Q32事件，因此不能把该开销解释成只发生在capture。Runtime生产字节在a455和3ca相同。已捕获Graph replay不重新进入这些Python函数；Native成功路径使用自己的统计，本次不改它。

最初读取的117 smoke和Math105原件没有 FINAL_STATS，SELECTED仅每profile/hash首次出现，不能据此恢复频率。随后取得119的真实A臂4K封存snapshot：source `f725598bd28ff9ba7800c11bb88c2a7832e6aee8`，生产逐字等a455；run `cjj-native117-stats-a-throughput-v2-20261029`。warm-after→score-after 窗口有65,646次note、202个活跃完整record、9个新增key，全部capture=false；最终343个记录，overflow=0。profile host计数分别为writer83 22,050、decode 21,546、multi2 10,290、q32 11,382、normalized2d128 378。这个窗口含client preflight，不是整次A验收，也不计Graph replay次数或GPU时间。旧0bf9/b371比例不外推本基线。

实际snapshot保留本地 `cjj-native117-a4k-live-frozen-snapshots-20261029/runtime-stats/`。`warm-after-4096.json` SHA `0d520b01461f2db577c9698e96816f8c1e81a75e10fdfb4506c11f018c3ec395`；`score-after-4096.json` SHA `aead05d1f79cfd371f451506679ed9eac2d717d4bfcfdda3f79d062d427ac8a6`。已有collector/delta审核由独立review agent完成；本任务只重建metadata并比较note，不冒充新服务验收。

## 最小改动

生产只改 `biv150_konly_encoded_runtime.py`（SHA `8dd35763e7cd43154d28bffc6760506bb256efd3b1ea674e2871949ebafb6964`）：新增一个最多4096项的内部索引，将严格的primitive metadata tuple映射到原JSON key。首次出现时仍调用原 `json.dumps`；之后相同完整metadata可复用该字符串。原shape_records及其字符串key、record字段、计数、shape cap、overflow和最终导出格式保留，119直接序列化shape_records仍可用。

metadata每次照旧从输入获取，索引不含Tensor或Tensor ID，不放行ownership或pointer/stride变化。内部tuple保留dtype/device、shape/stride/offset、grid、profile/hash、capture和全部原selected scalars；pointer名排序消除kwargs顺序影响。scalar类型标签与实际primitive类型都进入快键，避免bool/int碰撞。浮点（包括负零/NaN/Inf）、嵌套typed值、非整数几何等一律使用原JSON路径，保留原分组与错误。

只缓存已经进入原shape_records的key，索引和原记录各自不超过4096；shape溢出、已有记录在满cap之后命中、首次kernel证据/日志、异常与shutdown导出语义保留。原Registry、Native adapter、GPU数学、routing、owner/epoch/alias校验及launch后不重试都没有改。C3 implementation digest自然覆盖新Runtime源；manifest无单独Runtime SHA条目，Native adapter/SO pins不改。

## CPU对照与微测

`tools/check_biv150_runtime_note_keys_cpu.py` 加载Git中的原a455 Runtime作对照。49项功能检查通过，包含普通/重复/capture记录、kwargs顺序、IntEnum与bool/int、负零/NaN/Inf、list/tuple/dict、非整数/错误metadata、递归typed错误、首次日志异常、live metadata变化、原记录value外部变化、cap4096/overflow以及shutdown原字节导出/一次性/不覆盖错误。10次相同记录的key序列化由10次变为1次。精确源码逆变换能恢复a455整个文件；原Native接线/拒绝/C3矩阵和94db fingerprint也通过。

从真实snapshot重建全部343个record的CPU metadata，原/候选各执行686次note，原JSON key和record逐字一致，包括capture记录；没有加载tensor数据或GPU。输入没有逐调用顺序，所以这不是服务请求流重放。记录所含全部scalar均为int/bool，343种metadata都进入快键。实际writer的K pitch256、V pitch2560和decode热点N61/S4均按原记录保留，未用此前合成模板替代。

本机ARM、Python3.12.14、无Torch/GPU的实际 `Runtime.note` 微测：每场景3轮ABBA；重复场景每臂5000次，先预热300次；全miss场景每臂1024个不同记录。计入完整note及metadata模板取值；不只测json.dumps。下表从每个profile选该窗口最高host计数的真实metadata；最后一行在真实writer几何上合成不同cache_pages，专门保留冷路径代价。

| 真实metadata CPU场景 | 原A 中位process CPU时间 | 候选B | A/B |
|---|---:|---:|---:|
| decode重复 | 15.27 μs/call | 9.71 μs/call | 1.572 |
| multi2重复 | 22.00 μs/call | 13.92 μs/call | 1.581 |
| normalized2d128重复 | 21.96 μs/call | 13.87 μs/call | 1.583 |
| q32重复 | 22.09 μs/call | 14.05 μs/call | 1.572 |
| writer83重复 | 7.72 μs/call | 4.67 μs/call | 1.653 |
| writer合成全miss | 8.41 μs/call | 15.47 μs/call | 0.544 |

此前合成metadata模板结果也保留；其完整几何不是实际snapshot：

| CPU模板场景 | 原A 中位process CPU时间 | 候选B | A/B |
|---|---:|---:|---:|
| 重复writer83 | 7.50 μs/call | 4.58 μs/call | 1.638 |
| 重复decode | 15.69 μs/call | 10.05 μs/call | 1.561 |
| 重复q32 | 19.26 μs/call | 11.91 μs/call | 1.617 |
| writer全miss | 8.76 μs/call | 10.85 μs/call | 0.807 |

首次记录有额外索引成本：真实writer几何的合成全miss慢约84%，前两次不同模板微测慢约24%/47%。冷路径波动与分配成本保留，不能只报告重复命中的收益。增加有界metadata索引也会消耗额外CPU内存。是否值得服务采用仍需实际A/B；纯Graph replay没有这项收益。不能把本机微秒乘以服务host计数当作实机节省，更不能推导吞吐提升。

```bash
python3 -B -S tools/check_biv150_runtime_note_keys_cpu.py
python3 -B -S tools/check_biv150_runtime_note_keys_cpu.py --benchmark --iterations 5000 --rounds 3
python3 -B -S tools/check_biv150_runtime_note_keys_cpu.py \
  --snapshot-before /absolute/path/to/warm-after-4096.json \
  --snapshot-after /absolute/path/to/score-after-4096.json \
  --benchmark --iterations 5000 --rounds 3
python3 -B -S tools/check_biv150_native_k8_service_cpu.py --library /absolute/path/to/8b86.so
python3 -B -S tools/check_biv150_native_k8_fingerprint_cpu.py --helper /absolute/path/to/actual94db.py --library /absolute/path/to/8b86.so
```

完整报告仅保留本地 `/private/tmp/`，不共享原始服务日志/数组：

- 最终49项CPU：`cjj-agent-note-key-cpu-20261029.json`，SHA `7db9240618168f28a48e60888eb580523214296dc097785745aa5e0198c2d797`。
- ABBA微测：`cjj-agent-note-key-benchmark-v1-20261029.json`，SHA `fece33445c61659eff256364dc8d73dda6a97e18bedb7dba3884ee4b9cd5c2ca`；生产Runtime源同上，产生于最后一个非字符串key功能用例加入之前。
- 原接线：`cjj-agent-note-key-original-wiring-cpu-20261029.json`，SHA `f1ec9048656834f0ff2a4643c75d411b9d12a7f1d2f416a2b0446a32015c6f44`。
- 原94db fingerprint：`cjj-agent-note-key-fingerprint-cpu-20261029.json`，SHA `38160e85e03413c88896a8919808657e3cffd38f3c7ae336773ed11f06e70f94`。
- 真实343条metadata及ABBA：`cjj-agent-note-key-actual-metadata-cpu-20261029.json`，SHA `e7247abcee2edc02566a7cfd19c04a430d980adf596de47f725f959dcd8125d3`。
- 去掉计时的可逐字复跑报告：`cjj-agent-note-key-actual-metadata-correctness-20261029.json`，SHA `a343da5a725a36a134fcbd5d2c5e4b445c3772290a0363cb485a711d4eed3411`。

下一步由Root审查及独立服务A/B判断；没有新GPU验证框架，不从统计记录推断Graph replay次数或GPU时间。冻结117/119 launcher不会自动接受本新source，需要独立明确身份适配。
