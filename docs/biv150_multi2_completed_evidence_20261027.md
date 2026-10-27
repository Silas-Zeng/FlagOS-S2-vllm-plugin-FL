# PR66 已完成组件原件与 baseline 单臂证据

证据快照为2026-10-27 01:18:32 UTC。此时原controller603821/start340686064
仍为同一进程，baseline已完成，candidate处于accuracy；没有重启或补跑。
本次只处理root已下载的本地原件，不SSH、不访问GPU、不改生产e08或9317终态auditor。
模型source仍冻结`e08c79456f9b943dbc4d9d02d0a4e030b4eceac1`，新PR head只是证据提交。

## 原件及确定性归档

`docs/evidence/prefill-multi2-20261027/completed-components-v1/` 包含：

- `completed-components-raw.tar.gz`：全部18原件与原download-manifest，19个member。
  两个out各有result、baseline/candidate各3个保存IR，共各7文件，另加两份owner/log。
- `archive-manifest.json`：全部member的bytes/SHA、压缩包SHA及baseline export SHA。
- `baseline-independent-v1.json`：root在目标机用冻结9317/CEA纯CPU独立arm审计的原始JSON，
  逐字复制，明确baseline_only=true、pair_complete=false、candidate_not_audited=true。
- `summary.json`：本工具对原始结果、实际IR、计时与scope的CPU重算摘要。

18原件共4,687,534B，每个SHA/长度已再次复验。压缩包352,702B（<2MiB），SHA256
`329b52ce881456602b45fdebfb8848aa439eca2f5dd3ef30017ee08a4b3ab343`。
按路径排序、tar mtime/uid/gid=0、mode0644、空用户名组名、gzip mtime0；同输入重复生成
逐字相同，且逐member解压回验。baseline独立export SHA256为
`d6982a65b9209fda3e7bb0dff1799f177fa0870a201c12cbb985a2e46ea38fd3`。

可从归档解到新目录后纯CPU复验/重建（out必须不存在）：

```bash
python3 tools/summarize_biv150_multi2_evidence_cpu.py \
  --snapshot NEW_EXTRACTED_SNAPSHOT \
  --baseline-audit docs/evidence/prefill-multi2-20261027/completed-components-v1/baseline-independent-v1.json \
  --sdk-source PINNED_SDK54c495.py \
  --out NEW_EVIDENCE_DIR
```

Git中保留254b/e936/e08原对象；工具用其实际受测script SHA核对记录，不执行GPU脚本。
SDK54c495需匹配SHA；仅抽取已有纯Python stride/helper及Q-hoist LLVM CFG分析，不导入Torch。

## 组件结论及数值范围

受测254b四个代表case、e936十八个scope case均completed=true、errors=[]。
原结果始终official_score/precision_gate/service_accuracy_validated/service_installed=false。
表中速度均为原AB/BA/AB三轮baseline_ms/candidate_ms比值的中位数，在CPU重新计算；
计时只覆盖resident-input的captured 2D kernel，不含host dispatch、输入copy或模型。
它不是服务吞吐、置信区间或稳定效应估计。所有baseline均为当前真实两序列路径：
BM128/BQ16/w8/K64/s1，原Q32 parent仍active且selected_calls=0；Q32的num_seqs==1
门不命中两seq，故这里128对照未降级main。不能将此结论搬到单seq V8/PR76基线。
候选wide为BM256/BQ32/w16/K64/s1；小于1024总行的两例严格走原128。

| 源/shape | qlens | KV lengths | wide | 三轮中位speedup |
| --- | --- | --- | --- | ---: |
| 254b / double4k-big-first | [1945, 88] | [4096, 4096] | True | 1.203091964 |
| 254b / double4k-small-first | [88, 1945] | [4096, 4096] | True | 1.218231016 |
| 254b / double16k-big-first | [1994, 39] | [16384, 16384] | True | 1.077525666 |
| 254b / double16k-small-first | [39, 1994] | [16384, 16384] | True | 1.105188212 |
| e936 / scope-below1023-balanced-fresh | [511, 512] | [511, 512] | False | 0.996425561 |
| e936 / scope-below1023-q1-first | [1, 1022] | [16383, 1022] | False | 0.999950713 |
| e936 / scope-at1024-balanced-fresh | [512, 512] | [512, 512] | True | 1.262012374 |
| e936 / scope-at1024-q1-first | [1, 1023] | [16383, 1023] | True | 1.069627207 |
| e936 / scope-at1024-q1-last | [1023, 1] | [1023, 16383] | True | 1.071957066 |
| e936 / scope-at1024-q2-first | [2, 1022] | [17, 4095] | True | 1.177411413 |
| e936 / scope-at1024-q2-last | [1022, 2] | [4095, 17] | True | 1.181675321 |
| e936 / scope-above1025-balanced-long | [512, 513] | [4095, 16383] | True | 1.177990754 |
| e936 / scope-above1025-q1-first | [1, 1024] | [1, 16383] | True | 1.201332684 |
| e936 / scope-above1025-q1-last | [1024, 1] | [16383, 1] | True | 1.202407620 |
| e936 / scope-grid1026-balanced | [513, 513] | [513, 513] | True | 1.092022018 |
| e936 / scope-grid1026-q1-first | [1, 1025] | [16383, 1025] | True | 1.090483895 |
| e936 / scope-grid1026-q1-last | [1025, 1] | [1025, 16383] | True | 1.093277490 |
| e936 / scope-balanced2048-fresh | [1024, 1024] | [1024, 1024] | True | 1.142675575 |
| e936 / scope-balanced2048-long | [1024, 1024] | [16384, 16383] | True | 1.249389434 |
| e936 / scope-empty-first | [0, 1024] | [0, 1024] | True | 1.132305920 |
| e936 / scope-empty-last | [1024, 0] | [1024, 0] | True | 1.134502370 |
| e936 / scope-maxkv131071-q1 | [1, 1023] | [131071, 1023] | True | 1.065343876 |

16个eligible scope的中位speedup范围1.065343876–1.262012374；2个fallback分别
0.996425561/0.999950713，未进入wide manager。完整grid是qrows//BQ+num_seqs，
包括513+513、1025+1/反向，1023/1024/1025总行、qlen0/1、fresh/长KV/partial tail。

44个输入pattern的全输出direct都与实际baseline逐元素数值相等（numeric_equal=true、
max_abs_diff=0）；这是torch.equal结果，未将其解释为原始字节memcmp。原非默认侧流做
eager、输入copy、capture/replay，输出先poison；两个pattern的Q与packed input SHA
不同，capture只做一次，88个route×pattern Graph对eager都是exact。此处复核的是
受测源码及原结果守卫，不是重跑原GPU或重建未保存的输入/输出tensor。

CPU reference覆盖须保留区别：uniform pattern0对所有query行；random pattern1
只检查所选query行。scope的以下6个pattern1保留原baseline_cpu/candidate_cpu
allclose=false，各1个超阈（max_abs0.015625）；原reference_guard继承不超过一BF16
ULP的baseline误差，baseline_exceptions_over_one_bf16_ulp=0、new_or_worsened=0。
不能将继承守卫通过描述为全元素独立oracle exact/全allclose通过：

- `scope-below1023-balanced-fresh`，pattern1。
- `scope-at1024-balanced-fresh`，pattern1。
- `scope-grid1026-q1-first`，pattern1。
- `scope-empty-first`，pattern1。
- `scope-empty-last`，pattern1。
- `scope-maxkv131071-q1`，pattern1。

两侧actualManager一直active/private JIT cache，无fallback/3D/record error/omission。
保存的actual TTGIR/LLIR在CPU重算Q-hoist机制，与原Manager记录逐字段相同：Q load
位于循环外，LLVM cyclic CFG中Q memory reads为0，总32。实际加载metadata为baseline
114regs/0spill/32768B shared、wide113regs/0spill/65536B shared；不据此推断实测占用。
这里只归档actual.ttgir/actual.llir及q-fragment override字节；ASM/cubin、TTIR/source
仅在原metadata中有hash，**没有它们的字节原件**。

## 完成的 baseline 单臂，不是pair验收

root独立arm审计已实际读原始105 reviews/predictions、raw SSE、官方commands/scores、
cleanup/fingerprint，并核controller内嵌镜像。该导出保留原件hash及分题scores，
本地再次核对105分题合计102与working_gates；此包没有另行复制全部模型原始文件。

| baseline main2412 | 总token/s | mean TTFT ms | 完成请求 | input/output tokens |
| --- | ---: | ---: | ---: | --- |
| 4K | 5806.143119573 | 3353.948127216 | 256/256、failed0 | 1048576/262144 |
| 16K | 4479.567553448 | 111293.245950801 | 128/128、failed0 | 2097152/131072 |

baseline102/105、初始/后置mixed、官方计数、TTFT、working gates、实际源指纹与
owned cleanup通过；baseline_only审计使用9317工具SHA `cea62aeaea0915ec9a4b04c05879888014517ae70308130b07d995aee945b2dc`。
独立audit的baseline_all_gates_passed=true不能覆盖原子state acceptance_passed=false，
也不能推出candidate精度或pair通过。快照中candidate仅在accuracy、未完成cleanup；
不是精度失败结论，也没有candidate服务成绩可比。后续完整pair终态仍由root读取并用
冻结auditor重审，未报告新的服务gain，不合并、不重新抽题。
