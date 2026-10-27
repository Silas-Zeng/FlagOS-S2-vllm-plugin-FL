# Unchanged-max rescale skip experiment：CPU Draft

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
IR parser只用结构性合成fixture验证其拒绝规则，并对本地原baseline/PR71实际
TTGIR做负例；**没有真实新candidate正例**。不能因parser单测通过声称设备机制通过。

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
