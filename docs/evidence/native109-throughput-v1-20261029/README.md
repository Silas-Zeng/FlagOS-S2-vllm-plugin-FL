# PR109 完整吞吐：负结果与独立 CPU 审核

受测源码 **fc25c86a13fc273bb7368b83af4ca264524b4e12**，controller **d8a5bd435845bdb3dd1e4d3741f4ca38fe85b4cc152036517cbcf034f0eb4495**；5c / 0c66 / compact40f9 / v6。每档一次完整同负载 warmup，再计一轮 score；官方 e236 benchmark、vLLM 服务 CLI 和测试参数未改，Graph observer 关闭。

本次两档吞吐均低于已合并 PR107，建议保留证据后关闭 PR109，不合并这个实现。它是单轮观测负结果，没有重复分布可据以估计方差；也没有依据把差异抹为噪声。

| 评分场景 | PR107 total tok/s | PR109 total tok/s | 变化 | PR109 mean TTFT ms | TTFT 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 6374.4399021712225 | 6129.566376801242 | −3.84149085% | 3789.0170364207734 | +3.08259670% |
| 16K | 5165.670512577975 | 4041.348194844961 | −21.76527355% | 98628.0094105241 | +0.05518330% |

**这是多项实现差异的整组对照。** PR107 是 16384B 页、Native8b86、Q32 hoist ON；PR109 是 12544B 页、Native40f9、Q32 hoist OFF，cache 容量及调度形状也改变。不能把 −21.77% 单独归因于紧凑布局、Q32 或某个算子。

## 原始请求、算术与身份

| 轮次 | 完成 / 失败 | 时长 s | total tok/s | 用于评分 |
| --- | ---: | ---: | ---: | --- |
| 4K full warm | 256 / 0 | 232.41662858659402 | 5639.527636085861 | 否 |
| 4K score | 256 / 0 | 213.83568093180656 | 6129.566376801242 | 是 |
| 16K full warm | 128 / 0 | 615.1129854288884 | 3622.462950357596 | 否 |
| 16K score | 128 / 0 | 551.3565999688581 | 4041.348194844961 | 是 |

四轮共 768 个已计入 benchmark 结果的成功请求：输入 6291456、输出 786432 tokens；每个输入档位的总量及每请求 1024 输出均与 CLI 一致。total/output/request throughput 的 token 或请求数除以 duration 算术全部闭合。正式评分只计 384 个请求，warmup 不混入成绩；没有将 warmup 的 TTFT 或吞吐当验收门。CLI concurrency 固定 64，结果文件的峰值并发 69/65 原值保留。没有保存逐请求 benchmark 延迟数组，不能追加其分布或把这些汇总当成完整服务请求追踪。

另外保存的 3 个 smoke 和 32/64/40/48 burst 共 187 条回复，API ID 唯一、usage 加法闭合，每条输出 16 tokens；它们是短诊断，未计分，也不证明模型准确率或 Native 请求覆盖。client preflight 不由上述请求明细完整覆盖。

36 份原件 / **3077775B** 全部重算 SHA 与大小。268 个源文件与不可变 fc25 Git 对象逐一核对，MoE symlink 解析在同一 commit 内。实际包解析、2 条 worker C3、9 个原 installer、40f9 namespace/schema/load receipt 均一致；本地保存的 40f9 与 0c66 SO 重新计算 SHA，没有加载库。W8A8 和 tuning DB 二进制身份仍是原运行记录。

原 owner **53242/start355306781** 取回时同 start、状态 Z。controller completed/cleanup true/errors[]；身份核实后的 API 退出 0，没有剩余 owned process。没有追加远程 GPU-idle 观察。原 `official_score=false` 表示这是团队按官方协议运行的测试，不是主办方发放的分数。

4K score 完整命令窗口新增 **6 个 index metadata**，已有 metadata 无变化，selected tuning rows 改变；16K 三者均无变化。窗口包括 client preflight，不能据此量化计时段编译开销，也不能声称完全稳定。完整 warmup 已完成并不消除这一限制。

## 16K decode 调查线索，仍是条件推断

| 指标 | PR107 | PR109 | 变化 |
| --- | ---: | ---: | ---: |
| mean TPOT ms | 91.6359315565178 | 149.00374418670685 | +62.60405897% |
| median ITL ms | 38.00062369555235 | 72.71357858553529 | +91.34838198% |
| mean TTFT ms | 98573.61323895384 | 98628.0094105241 | +0.05518330% |

TTFT 几乎不变而生成阶段延迟显著升高，更支持优先调查 decode。它不是独立的阶段耗时分解或因果证明。

审核用 `result.date − duration` 至 `result.date` 建立近似评分窗，保留每条 server logger 的原始行、行号、时间、Running/Waiting、prompt/generation throughput 和 KV usage。结束时间只有秒精度，结果序列化延迟未测，窗口不能当成精确逐请求区间。

| 16K logger 观察 | PR107 | PR109 |
| --- | --- | --- |
| 近似 score 窗 | 21:26:03.647672–21:33:15，10月28日 | 01:01:22.643400–01:10:34，10月29日 |
| 约 10 秒一次的样本 | 43 条 | 55 条 |
| 显示 prompt=0.0、generation>0、Running>0 | 11 条 | 16 条 |
| 上述子集 Running 分布 | 32×7、31×2、25×1、1×1 | 42×9、41×5、3×1、2×1 |

“prompt=0.0”是格式化后的区间速率，不能把舍入显示当作逐步无 prefill 证明。Running 是活跃请求快照，不等于本步调度 token 数、padding 后行数或 Native 调用数。以上计数绝不能称为实际 Native 频率。

相关 10月28日 SDK snapshot 的两个实际函数、该次日志中的 capture sizes，以及 fc25 的 `_pick_segments` 做了纯 CPU 条件推导。若采用相同 dispatcher、统一 FULL decode、query length=1、scheduler cap≥64，且本步调度数等于这些 live counts，则：

| 假定本步行数 | padding 后 N | S（原默认 TARGET_PROGRAMS=256） | 当前 Native 范围 |
| --- | ---: | ---: | --- |
| 31 / 32 | 32 | 4 | 支持 |
| 40 | 40 | 16 | 不支持 |
| 41 / 42 / 43 | 48 | 4 | 不支持 |
| 64 | 64 | 2 | 支持 |

该 SDK 文件是相关存档，**不是本次 TP 原件中新采集的 SDK 快照**。实际每步是否选择 FULL、调度数和耗时仍未知。报告保留完整函数源码及 SHA，任何一项条件不能直接当成实测。

PR109 原日志记录了 42 层、44769 页/层、716304 tokens 的分配 metadata，PR107 为 548320 tokens。增加容量与 Running 区间变化、可能触及未覆盖的 48/S4 路径相互一致，适合据此设计独立组件测试；不能从分配 metadata 推导数据精度、实际 Native 频率或单一性能原因。

原 168 条 SELECTED/CAPTURED 与 71 条 fallback 是 host/capture 观察；没有 replay observer，也没有最终 compact stats。最终 lifetime counters、overflow、错误计数未知，不因清理成功而补成 0。

## 精度依赖与完整归档

复用父提交 **74ee48de6e8023586baa6eae35db082450ad2224** 已共享的[完整 Math105 证据](../native109-math105-v1-20261029/README.md)：官方 **103/105**、1 个 max-token 截断；补充并发64 **102/105**、0 截断，均无请求失败。独立审核 **daa1948b4df1eecbbfda6f73e6d90ede3c03023a631b4a41ea260f0691dfe9cc**。本次确认源 commit、全部 268 源 SHA、controller、C3、库与吞吐一致，并重核原件 SHA，没有重新批改答案。Native 请求覆盖和 full acceptance 仍未通过。精度已达标不能抵销本次吞吐负结果，也不转授给 PR110/111。

`review.json`：**cdd4137d4df17305d5e044b37233f9faf53120c9f177942577b4f9c9e4c08b5f**。`review.py` 为本次实际执行的原字节，基于既有 PR107 审核复用固定协议、源码、C3、缓存和清理检查。

`originals.tar.gz` 含全部 36 份原件及 download manifest，共 37 个文件，**309614B**，SHA256 **8eab1bea0f6d0c3ed6da3269ab2d7deb30e4fb9a1ad9c6d490b7dea2da969f8d**。没有遗漏已收集文件；原收集排除的生成 cache tree 不能被称为已完整归档。`archive-members.json` 逐项记录原件 SHA/大小。

相同完整包持久保存于 `/Users/chenjunjie/Documents/flagos/repos/experiment-raw-archives/2026-10-29/cjj-native109-throughput-v1-raw-20261029.tar.gz`。精度包、helper94db 和 PR107 对照原包均引用父提交已有文件，没有重新打包；依赖 SHA 见 `existing-dependencies.json`。单文件及证据包均不超过 2MiB。

## 轻量核包和可选 CPU 重放

在本证据分支 checkout 执行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native109-throughput-v1-20261029/replay.py
```

默认只核共享文件、既有依赖及完整包中 37 个原始 member，不重做审核。显式 `--run --out /private/tmp/native109-TP-root-NEW.json` 可重放有限 CPU 审核；需要原 40f9、0c66 SO，可分别以 `--compact-so`、`--gqa-so` 指定绝对路径。它从已有父提交包提取对照和精度原件，读取 Git 对象，不重新请求模型、执行 GPU 或重新评估答案。

本提交仅有 docs/evidence；没有 SSH、GPU、服务重测、源码修改、推送或关闭 PR 操作。关闭由 Root 在证据共享后执行。
