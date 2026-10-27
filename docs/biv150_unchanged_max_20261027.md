# Unchanged-max rescale skip experiment

## 9b71 终态：组件守卫通过，未观察到收益

本次CPU留证不改受测9b71的factory、benchmark、IR validator、生成kernel225c或
vllm_fl/csrc。原运行owner623736/start342729110同start终态Z；不启动新设备任务。
最新正常fetch/API核实团队main56f1838192abc940b3fb9accef490022ac125000；其GQA
控制源码与原受测main2412逐字相同（SHA2596cc09…）。本实验分支生产树未同步66的
其他生产改动，不能据此声称全生产树等同main56f；本提交仅归档tools/docs。

原result completed=true，B64/4096与B32/16384均passed=true，五模式
original/ragged/zero_qk/one_head_update/tiny_v的scratch/maxima/sums/out全张量
bitwise一致且有限；独立CPU FP64每模式只抽取row0、1、末行，不是全元素独立exact
oracle。changed-input Graph和side-stream的exact/finite守卫均通过。两种配置的
baseline/candidate实际post-launch均93regs/0spills/8192shared，w1/s1/use_sme14；
实际TTGIR/LLVM/ASM使用未修改v2 validator再次CPU审查，与原mechanism整字典一致。
identity primitive有实际exp/mul及signed-zero/subnormal/normal有限测试集合通过；
不推广为全部浮点输入的设备等价证明。

| 实际组件 | AB / BA / AB speedup | 三轮中位 |
|---|---|---|
| B64 / 4096 | .9975662644143614 / .9976054861688967 / .9975631304901982 | .9975662644143614 |
| B32 / 16384 | .9989044274096636 / .9988296004804774 / .9995290636650316 | .9989044274096636 |

比值由原baseline_ms/candidate_ms重算，完整producer+原reducer计时；三轮顺序由
原timings键顺序及冻结benchmark循环核对。均略低于1，未观察到微测收益；单组微小
差异没有稳定性/统计显著性结论，不追加整机测试。未做模型105、服务吞吐或TTFT，
official_score/model_evaluated/precision_gate仍false，不合并。本exact-no-new-max
版本等待root审阅后关闭；不删除分支，不据此否定整个C09或其他条件rescale方向。

38原件1815467B及原download-manifest完整保存于
`docs/evidence/gqa-unchanged-max-20261027/v3-completed/`。确定性gzip254601B，
SHA e049e97c7a0c8502eddc34c6634f480f49d536e76214788efa2a6612451348e3；
包括actual source/TTIR/TTGIR/LLVM/ASM/cubin、owner、空log及两个opaque pyc。
不导入原件Python/pyc、不加载cubin。原manifest SHA45315390…，result SHA9b9625df…。
原v1/v2失败包、provenance、失败result均保留并逐文件SHA重验，原因分别为旧parser
拒绝、fixture赋值dtype错误；不改写历史失败为通过。source proof内历史pending标记
原样保留；后续component运行状态另列，不能把source factory报告当设备验收。

标准库重放命令（解包上述自有证据包到新目录，SNAPSHOT须含原39成员）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/archive_biv150_unchanged_max_cpu.py \
  --snapshot SNAPSHOT --out NEW_OUTPUT_DIRECTORY
```

归档/manifest/summary解包重放逐字一致；六项bitwise、reference、Graph、资源、median、
completion故障注入均拒绝。Python3.14默认省略AST空字段，与目标3.12的PV AST打印
不同；checker显式恢复目标包含空字段的打印并严格核对原固定hash，不改变原数据或
数学门。CPU重放不重新执行设备数值比较，不能冒称新设备测试。

## 以下保留开发过程与早期失败记录

基于2026-10-27重新fetch并fast-forward检查的团队main2412207，独立本人分支。
只tools/docs，不改vllm_fl/csrc，不SSH/JIT/GPU、不进入队列。

现有真实BF16 4K/16K原件每full PAGE16仍执行2条alpha exp、32条accumulator mul；
running_sum更新已融合成2条FMA。8行逻辑acc在native M16上携带32个FP32值/lane，
不能把这32条乘法当成编译器已消掉。原件审查见
`/private/tmp/cjj-gqa-rescale-equivalence-review-20261027/review.md`。

PR71的factory虽支持threshold0，但只实测threshold8，现已closed；本项不会重开它。
threshold8保留旧anchor，改变P的BF16舍入。本项始终保留原new_max与P，只当8个
实际Q heads都满足old==new且new有限才跳过校正，任一更新或非有限anchor走校正。
它是未验证control的更严格实现，不是已证明的新算法或已有性能收益。

旧PR71真实compiler把校正路径running_sum FMA拆成mul/phi/add。本factory改用
显式tl.fma；PV dot仍接收原顺序的accumulator，不能变成zero-C dot后普通加法。
source inverse只证明两段full/tail变换范围，不能证明编译器保持这些目标。
后续必须审实际TTGIR/LLIR/ASM的CFG、校正FMA与PV C链，失败即停。

有限相等max时alpha目标为exp2(±0)=+1；IEEE乘1保留正常有限值和signed0。
Inf==Inf时差为NaN，因此有限guard不可删。firstpage的-inf anchor必须走校正，
emptysegment保留原早退，tail保留原mask。finite BF16输入也可能在QK中overflow。
未知设备FTZ、subnormal、native exp2(0)和MMA舍入不由CPU模拟或推断；不增加全acc
动态归约。未来tiny BF16/signedzero/subnormal和逐元素bitwise硬证伪失败就不计时，
不能用allclose掩盖差异。标题中experiment不代表已经证明完全等价。

CPU检查及显式--gpu组件入口已准备。默认命令只stdlib，不导入Torch/Triton、不JIT：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_unchanged_max_cpu.py --out NEW_CPU_JSON
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_unchanged_max.py
```

CPU合同覆盖46个八head状态（首page、单head更新、旧/新NaN、Inf、signed0/subnormal），
另有4090个host IEEE乘1位模式。该host例不是GPU模拟器。精确逆回原kernel字节及
参数表，full/tail/empty早退原文保留；默认CLI子进程证明未导入GPU SDK。
初版8e6855f的IR parser只用结构性合成fixture，并对原baseline/PR71实际TTGIR做负例。
下述validator v2新增已下载的真实candidate原件静态正例；旧run结果仍是parser拒绝，
不能把事后CPU审查写成原运行已经通过数值或性能门。

未来只有root明确排期后才使用`--gpu --source FROZEN_TREE --commit FROZEN_HEAD
--out FRESH_DIRECTORY`。入口首先验证空闲状态、冻结源码/JIT SHA及vllm_fl/csrc字节。
独立小primitive必须保留真实exp/mul指令，并逐bit检查native exp2(±0)=1和mul1
对signed0、FP32subnormal、normal/max有限值的恒等性，失败立即停止。

随后仅准备4K/B64和16K/B32的packed/page16、table pitch8192与narrowed scratch，
沿用原input factory/seed、真实base._pick_segments、producer1warp/1stage与原
reducer4warps，核对真实baseline641478/e29a0c身份。保存所有实际kernel.asm原件，
TTGIR必须有两个full/tail条件区、原new_max/P/BF16P链、校正显式FMA和PV的条件C；
LLVM CFG必须证明4个alpha调用、64个acc mul和4个FMA均在校正分支，16个PV MMA
保留非zero C；ASM还要对应原ml_fma_f32与同型BF16 MMA。未知lowering即拒绝，
若仍unconditional rescale、FMA拆分或PV C链改变，不能认可机制或进入计时。

逐bit比较原/候选的producer FP32 scratch/max/sum及BF16最终输出，而不只比较
allclose。先原seed、ragged/empty/tail、zero QK强制无更新、单head递增强制校正、
tiny BF16 V及signed0，再动态Q/lengths Graph和侧流；输出及scratch poison覆盖
未写/陈旧值，Graph/侧流也须有限且exact。独立CPU FP64门保留原
`candidate_rmse<=max(.005,1.2*baseline_rmse)`，RMS0与maxdiff非0矛盾会直接失败。
全部当case硬门通过才运行三轮交替完整producer+reducer成本；失败case不计时。
post-launch资源逐字段记录known_equal/known_changed/unknown，None不作为相等证明。

只准备4K/B64与16K/B32，仍等root已有66终态及4a诊断后统一安排，不自动启动。
full105、正式吞吐和TTFT均未验收，不能合并。

## PV debug位点的CPU修正

当前团队main已到56f1838；其GQA本体仍与2412逐字相同，SHA仍2596cc。本分支保持
同一实际GQA算子baseline；未来服务集成再同步新的生产main。此修正只改工厂proof、
审计和CPU检查，生成kernel字节及SHA225c9cc1完全不变。

真实baseline的PV赋值位于179/211行，但LLVM/ASM debug site落在参数行180/212。
旧审计只认赋值行，会保守地把PV计数成0。现在由原
`tl.dot(probability.to(V.dtype), V)`的精确AST生成full/tail call span，接受该调用内
的debug行；不接受其他dot、QK位点或调用范围外的相邻行。校正CFG、P/BF16P链、
PV非zero条件C、原FMA、32mul及实际ASM diamond门均不变。

CPU新增真实baseline LLIR+ASM位点回归：各识别16条PV MMA（full/tail各8），且
明确排除16条QK MMA。另有参数行合成parser正例、错误QK和范围外下一行的LLVM/ASM
拒绝，共16个拒绝检查。此定位回归不把原baseline认作条件rescale候选，也不是新
candidate设备正例；原baseline/PR71完整机制负例仍保持拒绝。


## Validator v2：原始产物的 CPU replay

本次只修复审计和CPU检查；factory与benchmark逐字不变，生成kernel仍为
`225c9cc1be483f7cee32eb8c61b87f97f238ea740ee1348c4f5a4023adf35313`。
提交前本次fetch origin main成功，CJJ1008 CLI亦核对团队main为56f1838，GQA仍为2596cc。
源码工厂继续固定原2412同一GQA；本次没有同步上游SDK或改生产代码。

首次真实运行的source是8e6855f。identity primitive保留真实exp/mul且signed-zero、
subnormal、normal的逐位恒等门已过；B64 baseline/candidate均首次launch并记录
post-launch 93regs/0spill/8192shared。随后TTGIR解析器抛出
`Missing conditional alpha/accumulator scale/explicit FMA/skip state`，尚未进行完整GQA
数值门或计时，B32未运行。**原passed=false和timing_attempted=false保持原字节**；
这不是数值或吞吐负例，也不是完整组件通过。

原26文件925773B（含2份仅作不透明字节保存、不会执行的pyc）和下载manifest保存在
`docs/evidence/gqa-unchanged-max-20261027/v1-raw.tar.gz`，压缩134547B，SHA
`18aa36f0973c3a678a0617b185df6fbc2de837373a1937b8cbe5eaaae30519b5`。
CPU replay仅在内存读tar成员，逐个核对文件大小/SHA；不会解包、导入Python或加载cubin。
历史result SHA为`cae41baa9391b04eb616ee1f88ca6a25b2b885e72274bb0f94314c8aba50a2ef`，
log SHA为`d57b608d60357f509a0574bb9354911f71e70a6d06ca59d7b509d68985366b50`。
附带原PR71 threshold8 TTGIR负例（SHA见v1-provenance.json），原无条件baseline与它
都必须拒绝；不会重开71或把它当作本实现的设备正例。

v2是针对这些实际lowering格式的保守识别器，并非通用MLIR/LLVM/BI ISA验证器：

- TTGIR按SCF/reduce词法region解析SSA，保留多结果#idx。full/tail中的同名
  `%rescale`、`%probability_sum`、`%new_max`不再相互覆盖。原full区域227–238与
  tail318–329各保留校正FMA/acc乘alpha、skip加Psum及不改C；PV242/333分别消费
  `%12#1`和`%running_sum#1`。P/new_max/BF16P链和相同Psum门继续检查。
- LLVM仍要求两个动态gate及4alpha/64acc-mul/4FMA/16PV计数和校正支配关系。
  另按实际SSA逐个追踪full join613和tail join1321：每处32个C phi的skip输入是
  原C、correct输入是同一C乘该行alpha，完整装入8个原生PV C向量；没有删去任何
  native M16分量。两个denominator phi连接同一L/Psum的FMA与skip-add，PV输出
  不能再post-add到C。这里不推断opaque MMA内部的C→D跨lane ABI。
- ASM按真实BB label和taken/fallthrough边恢复控制流，将`.loc 0`的branch绑定到
  `.loc 97/136`的原saved-mask设置和配对restore。full gate838经BB0_7显式跳回
  较早的BB0_3；tail gate1624经BB0_72自然fallthrough到BB0_73。各校正区仍须
  2exp/32原C乘法/2原`ml_fma_f32`，skip仅两条原L+Psum，join包含8组C/D相同的PV。
  缺mask、错误join、nested CFG、投机/hoist或未知形式仍拒绝；不会无条件放行loc0。

CPU保留46个predicate、4090个host IEEE目标及既有16个拒绝检查；synthetic fixture
只用于源码/位点回归，不再把其缺失实际mask/C-phi的数据当机制正例。实际原件新增
35个单点mutant拒绝（9 TTGIR、11 LLVM、15 ASM），覆盖漏门、#idx/old-anchor混淆、
wrong join、tail错误fallthrough、alpha出分支、FMA缺失、skip改C、重复/错alpha缩放、
zero-C、PV post-add、mask restore/complement以及QK/相邻错误debug站点。
新静态报告在`validator-v2-cpu.json`，与旧cpu.json/result/log分开。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_unchanged_max_cpu.py \
  --out NEW_VALIDATOR_V2_JSON
```

该命令stdlib-only、无SSH/JIT/GPU。设备仍须从新validator tag重跑完整组件的bitwise、
CPU FP64 reference、poison、changed-input Graph/side与producer+reducer成本门；
原数值门、输入/seed、资源未知标记和正式full105/吞吐/TTFT要求均不变。

## b633 第二次运行的 fixture dtype 修正

第二次实际运行的全部TTGIR/LLVM/ASM v2门已过；B64的original/ragged/zero_qk
完成后，one_head_update fixture的advanced-index赋值因BF16目标/FP32 RHS不一致而报错。
本修复仅给原`progression[:, None]`增加`.to(c['k'].dtype)`，不改progression公式、
数据分布、kernel225c、IR validator或数值阈值。邻近tiny_v tensor已显式BF16，
lengths tensor显式int32；其他cache copy使用原同dtype张量，标量赋值沿用原实现。

原26文件932619B逐SHA核验，另包`v2-raw.tar.gz`（135939B）保留原result/log与
全部实际IR；来源及SHA见`v2-fixture-failure-provenance.json`。原passed=false、
timing_attempted=false及dtype错误文字不改；这次是fixture构造失败，完整组件
数值/计时仍需新tag重跑。只复跑既有CPU检查及默认CLI，没有新增镜像测试或设备操作。
