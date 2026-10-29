# PR120：4K A/B/A 验证，当前版本不合并

当前 storage identity cache 候选未证明稳定的整机吞吐收益：B 对 A1 提升 **4.065%**，对随后的 A2 却低 **0.855%**；未改动的主干 A2 相对 A1 高 **4.962%**。记录结果后关闭当前 Draft，不继续本版 16K 或 Math105，不合并。本组有界验证也不足以证明差异全部来自随机噪声，或证明候选在所有工作负载下均无效。

## 正式计分与未计分预热

| 运行顺序 | 受测源码 | 4K 正式 total tok/s | 正式 TTFT ms | 完整预热 tok/s（不计分） | 正式请求 |
|---|---|---:|---:|---:|---:|
| A1 主干 | `899ce77470b0b539aebc108fd122f668c2d5894d` | 6537.093244 | 3567.526923 | 6899.106385 | 256/256，0 失败 |
| B 候选 | `2a7fab6e705f5bf044d236d12e544c5dc388125f` | 6802.828752 | 3546.024228 | 6754.656328 | 256/256，0 失败 |
| A2 主干复核 | 同 A1 | 6861.484031 | 3569.049482 | 6754.821940 | 256/256，0 失败 |

三次均先做 3 次短预热（每次 64 请求、128 输出 token，seed 920000–920002），再做 1 次完整预热和 1 次正式计分（均 256 请求、4096 输入、1024 输出，seed 0、最大并发 64）。正式吞吐按实际 1,310,720 个总 token 除以计分时长重算；完整预热不替代 score。B 的完整预热比 A1 低 2.094%，与 A2 相差 −0.00245%。

正式服务 CLI、benchmark `e2363389…`、94db helper、v6 初始 tuning snapshot 及四份 SO 相同；pure Native `5898…`、mixed Native `ebbc…` 与 Q32 均开启。A1/A2 生产字节及 C3 相同；B 的生产差异仅原 Native host 校验模块及相应来源 manifest，使用其独立严格 bridge。未叠加 gather 或新的 operand-hoist 实验。三轮都完成并成功清理 owned 进程，errors 为空。

## 缓存与观察边界

- A1 计分边界新增 5 份 `_index_linearized_jit_function` metadata，对应 input rows 1782、1922、1755、1755、1782。
- B 和 A2 均新增 4 份，对应 1774、1420、1861、469；两者新增键和 metadata SHA 相同，但 TTIR 字节并非据此保证相同。
- 三轮 selected tuning rows 均变化，没有已存在 metadata 被修改。边界可能包含 benchmark 客户端 preflight，未隔离这些操作耗时，也未记录 benchmark 表的完整历史。因此不能用新增文件数量解释吞吐差异，或断言 B/A2 编译与调度耗时相同。

只有按时间顺序的一组 A/B/A，没有随机重复试验或置信区间。原 host selected/captured 记录不能代替计分窗口实际调用频率或 Graph replay 次数；本轮无统计 RPC、无 FINAL_STATS。原 `native_quality_covered`、`native_model_precision_verified`、`mixed_same_worker_smoke_verified` 均保持 false，不因本摘要修改。这些统计缺口与已成功完成的请求是分开记录的。本次未运行候选 Math105 或 16K，不能继承旧版本模型资格。

## 保留的组件结果

此前真实 CUDA Tensor/descriptor 组件已验证完整 host `validate` 的 accept/reject、异常顺序与返回行为。8 个 shape/stream 组共 96 个完整样本、42 个 Registry owner 下，每次完整校验节省 **48.310–49.207 μs**；例如默认 stream N64 从 **310.638 μs 降至 262.180 μs**。独立复核报告 SHA 为 `bd03aa5f2dd7d805eb87929a69707329a7a261361c33af054671e7fb5bbea591`。

这是组件真实收益，仍可作为未来工作负载的假设。FULL CUDA Graph replay 绕过每步 Python 校验，当前关键路径占比没有被本组实验确定；不能将节省值乘以层数直接推成模型收益。组件准备协议见[原说明](../../biv150_native_storage_actual_validate_20261029.md)。

## 精选证据

[精选 JSON](selected-results.json) 保存精确成绩、三份独立报告及 download manifest SHA、协议、缓存形状和明确的 false 边界。ABA 复核重新核对三组完整原件 SHA、实际 commands、token/时长算术与缓存增量；实际原件、完整日志和审计脚本留在本地，本提交仅这两份精选文件。提交本身不执行 GitHub 关闭；由 Root 审阅并发布后关闭当前 Draft。
