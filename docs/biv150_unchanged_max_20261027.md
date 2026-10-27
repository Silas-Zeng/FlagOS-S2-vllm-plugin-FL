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

接下来补CPU合同/拒绝检查及显式--gpu组件入口，使用真实base._pick_segments、
producer1warp/1stage和原reducer4warps；先exact/实际IR机制门，再完整producer+
reducer配对成本。独立FP64参考原门不放宽，unknown资源不视为相等。
只准备4K/B64与16K/B32，仍等root已有66终态及4a诊断后统一安排，不自动启动。
full105、正式吞吐和TTFT均未验收，不能合并。
