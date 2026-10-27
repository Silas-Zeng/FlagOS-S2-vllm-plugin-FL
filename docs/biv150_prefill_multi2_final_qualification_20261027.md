# PR66 二序列 prefill：最终固定pair验收

候选e08在保持原Q32链的前提下，为通用两序列且总query行>=1024启用经证明的
BM256/BQ32/w16/K64/s1。其余shape沿用原路径，grid完整保留qrows//32+num_seqs。
这是CPU可见descriptor条件，不按历史qlens、benchmark名或评测题分流，不GPU.item同步。
候选分支默认启用；显式关闭为 `VLLM_FL_BIV150_PREFILL_MULTI2=0`。

模型源冻结 `e08c79456f9b943dbc4d9d02d0a4e030b4eceac1`；对照团队main
`241220723a09061c182764f8f738e5ec01190367`。后续提交只补tools/docs/evidence。
原controller603821/start340686064于02:07:25UTC完成，root确认同start Z、两arm
completed/cleanup、9031closed/GPUidle；这是当时观察，后续GPU调度由root另行负责。
root02:08实际fresh fetch仍为main2412。固定服务和benchmark参数均未改。

## 原始精度与计时

每arm完整同一105题、105个唯一successful reviews/predictions，零失败预测：baseline
102/105=0.97142857，candidate101/105=0.96190476，均满足>=0.95（至少100正确）。
保留全部原题与原分数，不重抽、不重测。3题结果不同：1题baseline错/candidate对，
2题反向，净少1正确；不把这一对观察解释成零数值差异或稳定的模型能力变化。

| 官方负载 | main2412 total tok/s | candidate e08 total tok/s | 单对观察变化 | baseline/candidate mean TTFT ms |
| --- | ---: | ---: | ---: | --- |
| 4K | 5806.143119573 | 5827.850270757 | +0.373865% | 3353.948127 / 3224.458702 |
| 16K | 4479.567553448 | 4517.191826643 | +0.839909% | 111293.245951 / 110248.876448 |

两arm每档均3次短warmup+1次完整workload warmup+唯一1次score；4K为256请求、
input1048576/output262144，16K为128请求、input2097152/output131072；并发64，
failed0，TPS=(input+output)/duration重算一致。两档TTFT/所有working gates均过。
每arm完整105仅一次，正式score每档一次；这是固定baseline/candidate单pair，不是
统计显著、稳定效应估计或本改动20%收益。原working_gates.goal_20_percent使用旧
research参考值，不等于本PR相对最新main的收益。

baseline/candidate初始和后置raw SSE mixed语义正确，所有官方CLI/请求/token计数
及4warm+1score原件已审。native library固定
`5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe`，
相同initial tuning snapshot、production defaults及实际loaded-source fingerprint验证。
manifest引用的每arm4个generated out模块共11个调用点均逐SHA/AST核对；未装包或改源。

## 实际服务路径与审计

baseline有原Q32 parent且没有MULTI2；candidate真实服务launch后SELECTED为rows2047、
num_sequences2、grid65×2、B256/Q32/w16/K64/s1，hash
`43facedad89899d27893be51ca20fb578ea33c4664ac39e0063009bb0e7903b0`。
该事件与实际Q_FRAGMENT compiler proof联结，Q load位于循环外、LLVM循环内Q reads0/
总32。两arm均有真实C3 FIRST_SELECTED正常scope/返回原模型、逆logits回原请求顺序，
未出现Q32/MULTI2 fallback或scope reject。runtime/compiler记录中的prelaunch null
n_regs/n_spills仍不是实际loaded寄存器或occupancy证明，未借组件资源替代服务测量。
两armowned cleanup完整，原进程身份与root终态idle观察保留。

CEA原v1完整审计失败 `Manifest selection events changed`：合法日志print(dict)的
grid tuple与manifest JSON list直接比较误报，原失败JSON与旧CEA source均保留。
新增v2仅在runtime严格验证后将该tuple复制成list再比较整个dict/list，其他gate逐字
不变，无改成绩、raw或容差。原16fault、JSON/repr正例、7字段负例及真实manifest/log
重现均已留证，见 `docs/biv150_multi2_final_audit_v2_20261027.md`。

root02:15:03.675973UTC目标CPU完整v2审计的structural/precision/formal/eligible均true；
本地从133 raw再次完整运行同一冻结v2，与目标镜像相同。仅审计生成的文件引用路径作
local/target root映射，audit utc/root属主机元数据；原state/manifest完整严格相等。
冻结v2 source SHA256为 `2c79015c8d58ab410261d1eb3333900b1836f6de89c2a8452a7e89cda970f480`。
原v1失败SHA为 `8cf451c91e7dbb04f6f2d9465335d86b89280ca891c4af9c07c2b47d548d0c4f`；
原v2成功JSON137414B，SHA为
`136e96135092d3c725b69c9496ad503caf5a81f4cec51ac76b5933c00f2017a5`。
审计eligible只供review，本agent不ready/merge；root审查后自行操作。

## 完整原件归档

目录 `docs/evidence/prefill-multi2-20261027/final-pair-v2/`。
133个下载原件共11,744,840B，全部SHA/长度复验，含完整105、commands/benchmarks、
SSE、state/manifest/log及8个manifest引用生成模块，排除权重/普通cache。
原download-manifest SHA为
`bda4e1935243da8de4114d1c243e86c261d9958b4c1af988afa7cb90583c12c8`。
三包合计136成员：133raw+原download-manifest+v1失败/v2成功原JSON；member清单与
每SHA在archive-manifest.json。按路径排序，tar元数据/gzip时间固定，重复压缩与解包
逐byte一致，每包<2MiB：

| 包 | 字节 | SHA256 |
| --- | ---: | --- |
| baseline-raw.tar.gz | 944028 | `203fd2160b40dda5defd1a7ec2cec5df00461a8e71ed6440818af82f9e292e49` |
| candidate-raw.tar.gz | 907014 | `2d3a1e2b0fe68785bc2645b82a6d58202b7793c023305ca947dfc52b893d85d9` |
| common-raw.tar.gz | 40801 | `9a5dfc63ed5e2308ac922b8b782da8cfcf9d02b809c1b01b708b7255bac2cff0` |

`completed-pair-independent-v2.json`为目标审计原JSON逐字复制，summary.json为独立raw
重算与镜像摘要。重建时三包解到同一新目录，再运行纯CPU命令，out必须不存在：

```bash
python3 tools/archive_biv150_multi2_final_pair_cpu.py \
  --snapshot EXTRACTED_ROOT \
  --v2-audit docs/evidence/prefill-multi2-20261027/final-pair-v2/completed-pair-independent-v2.json \
  --out NEW_EVIDENCE_DIR
```

EXTRACTED_ROOT中的audits/两个文件是包额外保留件；重放入口仅额外接纳这两份，逐SHA
核验，不改原download-manifest或任何raw文件。
早先254b四例/e936十八scope组件包及baseline-only CEA export均不改，说明见
`docs/biv150_multi2_completed_evidence_20261027.md`。scope6个随机oracle继承例外、
component时延非服务时延及未保存ASM/cubin字节等边界继续保留，未被最终pair覆盖改写。
