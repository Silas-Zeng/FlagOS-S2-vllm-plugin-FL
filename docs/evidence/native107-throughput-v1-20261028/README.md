# PR107 固定参数双场景正式结果

受测源码 `b5d1a241f45efd21517d04e25a7dc5d44ac1ca06`，5c/0c66/8b86/v6；controller558579与原helper94db/e236。每档完整同负载warm后计一轮；Graph诊断关闭，计分未并行传输大归档。

| 场景 | 最新main1cb total tok/s | 候选 total tok/s | 相对变化 | 候选TTFT ms | TTFT相对main |
|---|---:|---:|---:|---:|---:|
| 4K | 6371.243813391296 | 6374.4399021712225 | +0.050164%，持平 | 3675.7097294175765 | +11.1449998% |
| 16K | 4815.312927015575 | 5165.670512577975 | +7.2759048% | 98573.61323895384 | −5.9090440% |

256/128均全部成功、0失败，token与时长算术一致。独立CPU复核报告SHA `57bb07087627709f2ccc27b73d4def3d78793c5a562b5fb0d7a89aa96bd0c218`；48原件3954804B、252不可变源码对象（含1个同commit内symlink）、实际SO、C3、CLI和清理均核对通过。完整小原件及额外下载manifest在 `originals.tar.gz`（536943B，SHA256 `3b74cad11bf0eeb719bc3dcb1770622fa2d3497111a2fd292279492ded6755c1`）。

4K候选计分全窗口新增6项index metadata，main为4项，两边selected tuning rows改变；16K双方无新增metadata且tuning rows不变。仅有单轮同机配对结果，不宣称稳定收益、因果归因或4K小差异是真实收益。main4K曾并行只读大归档的限制保留，不能反推零影响。正式日志的selection/capture不是计分期间实际请求频率。

同源[独立精度与请求覆盖](../native107-math105-v1-20261028/README.md)：官方100/105、独立并发64为102/105；补充105题都有新Native64正记录，32题另有Native32记录。精度截断2/1保留。无最终observer摘要，仍不声称完整频率、最终零overflow或所有token均Native。

两档均满足[官网固定性能门](../official-iluvatar-baseline-20261028/README.md)，且也低于历史内部TTFT工作线4770.3209/156938.85ms。4K延迟相对最新main升11.145%，**没有满足相对main TTFT+1%比较线**；该比较线不能冒充官网硬门。合并判断应明确：4K吞吐在1%区间内持平，16K观测提升7.276%，精度与官方TTFT均合格。

Root在21:41 fresh fetch/API确认main仍1cb；PR可干净合并。13项仓库CI均skipped，并非通过；设备验证依据是本目录原件及先前完整数值/Graph/side证据。最终合并前仍须再次核对远端头。
