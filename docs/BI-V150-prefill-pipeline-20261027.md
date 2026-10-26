# BI-V150 prefill流水深化（2026-10-27）

基于刚fetch的团队main `241220723a09061c182764f8f738e5ec01190367`（PR47已合）。生产vllm_fl/csrc无改动；本PR共享隔离组件实验，不是正式服务成绩。

当前真实默认大prefill为BM256/Q32/K64/w16/stage1。旧工具的BM128/w8对照已经过时，不能用它重复报告相对当前主线的收益。本工具默认经实际`biv150_prefill_q32.install()`安装该策略，在策略内部private manager取tap，断言真实launch常量、hash、Q-hoist证明及loaded资源。直接替换外层holder会绕过Q32，本PR明确避免。保留`--baseline bm128`用于历史复现。

先测两种有结构依据的变化，均保留当前生产的严格IR transform/逆变换/lowered证明，不在实验中放宽生产编译证明：

- BM256/w16/K64/stage2：检查K/V预取流水与Q共享内存生命周期；额外共享内存、寄存器及收益必须实测，不据参数推断。
- BM256/w16/K128/stage1：减少循环迭代，观察更宽score tile带来的寄存器/共享内存代价。若当前证明不支持，保留actual TTGIR/LLVM作为下一步依据，不把fallback计为候选。

隔离个人实例Python环境与私有编译缓存；先确认GPU空闲、9031无服务，不改site-packages。先用4K/16K各2048行排除明显无效路径，随后才覆盖fresh1024/2048、1024×16K及真实1985/2017尾行。随机/全行uniform causal CPU FP64参考、与当前基线的数值比较、非默认stream、changed-input Graph门槛沿用，不放松精度。Graph直接测已选中私有内核，是组件诊断，不能当正式服务吞吐。

CPU CLI无GPUdry run、差异空白检查通过；生产目录与main逐字相同。若组件有明确价值再实现实际服务选择、更新缓存语义、完整精度和充分预热官方4K/16K单轮验收；只有相对最新主线满足精度/TTFT且真实更优才合并。方向无价值则保留原件与关闭原因后关闭Draft。

## 第一批实机结果

个人实例2026-10-27 02:47–02:49，严格串行、当前Q32对照、2048行组件，均数值/CPU参考/changed-input Graph通过，三组交错预热计时：

| 候选 | 4K baseline/candidate ms | 16K baseline/candidate ms | 速度比4K/16K | 实际regs/spill/shared |
| --- | --- | --- | --- | --- |
| 当前Q32 stage1 | 约1.449 | 约6.566 | 1/1 | 113/0/65536 |
| K64 stage2 | 约1.713 | 约8.082 | 0.84587/0.81236 | 111/0/81920 |
| K128 stage1 | 约1.756 | 约8.251 | 0.82563/0.79524 | 131/0/65536 |

stage2随机/全uniform与baseline数值全等；K128随机有舍入差异，但原数值及CPU oracle门槛均通过，未改容差。两项已正常结束、GPU空闲，不进入长服务验收；不是1%波动。

驱动cuDeviceGetAttribute返回：warp64、SM16、每SM寄存器262144、共享内存131072、threads8192；每block4096 threads/262144 regs。106号max-blocks返回0，不能把它当有效驻留能力。按1024线程CTA，K128的131regs使两CTA所需268288>262144；stage2两CTA共享163840>131072。它们都跨过当前Q32可能容纳两CTA的资源上限，但这些上界计算尚不证明实际驻留数或性能因果。

因此再做唯一有针对性的K128/maxnreg128试验，观察是否能换回并行度；不放宽IR/数学门槛。实际结果为**127regs、0spill、65536B shared**；驱动对实际已加载function报告baseline/candidate均可驻留2CTA。这是驱动容量查询，不是测得的活跃驻留轨迹。

| maxnreg128候选形状 | 相对当前Q32组件速度比 |
| --- | --- |
| 2048×4K | 1.04990 |
| 2048×16K | 1.02972 |
| fresh1024 | 1.08077 |
| fresh2048 | 1.05111 |
| 1024×16K | 1.04117 |
| suffix1985×4K | 1.04995 |
| suffix2017×16K | 1.03101 |

7形状×2输入图案均通过原数值/CPU oracle/changed-input Graph，随机图案含允许范围内舍入差异。全部三轮交错计时原件与两项负结果在`docs/evidence/prefill-pipeline2-20261027/`；archive443625B/50members/SHA7caecdc17997fe529469bb5c7364f978ac6e417efccaa4764027e4cf48e7ef33。本地逐一重验49原文件SHA及11个case计时比例/门槛；未冒称模型吞吐。

已实现默认实际服务候选：仅Q32适用的单序列>=1024行，K128/w16/maxnreg128；沿用严格Q-hoist，cold时先load函数并检查regs<=128/0spill/shared<=65536，再允许launch。未知编译/资源条件在launch前回当前Q32 K64；设备执行错误传播且不重试。独立child flag缓存语义在Q32/parent关闭时折叠，runner只增4行installer并更新严格source pin，compute-order主体不变。CPU策略故障、cache/原路保持、runner逆变换及严格compute-order函数相等检查通过。

## 实际默认策略预验证和模型配对

冻结候选`b0d34e6e9213e8026ea65989f3917995cf8fd640`在个人实例实际策略16形状×2图案全部通过（13适用形状26selected、3回退形状6patterns，0compile fallback）；实际2047行与4097/16385部分KV页均覆盖，输入字节不变。真实C3闭包检查通过，feature env未手工启用。archive12159B/6members/SHAe30a46ecd8b7225ac02af33672a9d4d9b7d9c2a61cec3457fe861603d07c2c15。本地重读完整结果及真实GPU log解码、错误register记录拒绝验证通过；这仍不是105题/吞吐资格。

03:12:45北京时间正式配对已启动：`R/prefill-k128-main2412-pair-v1-20261027`（R=/workspace/experiments/minicpm-s2）。此前再次Git fetch与GitHub实时main核对均为`2412207`；baseline为`R/code/main-w16-qualified-20261027@2412207`，candidate为`R/code/prefill-k128-service-v2-20261027@b0d34e6`。双方生产defaults、固定CLI、同snapshot/nativeSO5c、完整精度/3短+1完整warm/每档1次正式score，baseline out/large显式传1并使用`--baseline-production-defaults`，不能套旧controller默认0/0削弱主线。

Controller568657/start338951699/SID568657已经实际存活，baseline driver568661、server568665，原件launch相邻JSON可审查。baseline期望package fingerprint74b1413...，candidate4b7cab4bd0a11c827d2dd7a780fb2d85075d5128415e7f8a97ab1a6054ccaae8（Native=True）。harness10d11ba...与controller99eda2ee...按SHA固定，未改全局包。配对仍在运行，无完整精度/吞吐结论、不合并。CPU独立审计`tools/audit_biv150_k128_pair.py`专门验证新main真实Q32对照和实际K128/maxnreg/资源/IR/自然C3、105题/预热/TTFT/固定token/初始post语义等；旧w16审计要求baseline没有Q32，不能原样套此新pair。

03:40基线完成精度：独立重读全部105唯一prediction/review/report，无错误或漏题，**102/105=0.9714285714**、通过。对应独立JSON已共享。本轮baseline实际KV为548416tokens（上轮548320不能直接套），实际加载fp74b验证通过，真实Q32 w16/K64选中。随后进入吞吐：4K三短和完整warm已完成，完整warm5871.7426tok/s为**未计分暖场**；正式得分及candidate仍须等原件，不能把暖场当历史最好或配对收益。

03:50基线正式4K完成：**5778.458274320222 total tok/s**、mean TTFT **3285.4543949870276ms**、256/256成功/0失败，input1048576/output262144严格匹配。真实C3首选N62/M2048/Dprefix2/Dall61/outside59、无SCOPE_REJECTED，Q32选中。原计分命令区间selected-config changed=True、8条新kernel metadata，保留此范围含client preflight的局限，不将其称为纯计时区间编译成本或完全无编译扰动。独立partial JSON已共享；16K及candidate尚未完成，不能给出新优化端到端收益。

04:18:38基线完整结束并完成owned cleanup，原driver568661/server568665/Engine568837均已实际不在`/proc`。16K正式原件为`score-16384-infqps-concurrency64-minicpm-20261027-041828.json`：**4480.988345123086 total tok/s**、mean TTFT **111265.90971279802ms**、duration497.2617262941785s、128/128/0失败，input2097152/output131072。工作TTFT限值156938.85ms内；4K也在4770.3209ms内。16K计分命令区间selected-config未变、0条新metadata；完整warm3940.9087仍是未计分暖场。保留单轮和调度/缓存差异限制，不把基线分数当新优化收益。

基线独立审计和原件archive已共享：941710B、62members、SHA`74dfc21fffc38ff3441b9a5678b25b9077106fae3feea8a213ba5ba7f6c1e7ea`。本机重验61个原文件SHA，独立重算105题、两档score/预热/TTFT、实际Q32/C3，逐项核对固定服务和正式benchmark完整命令。`tools/audit_biv150_peer_semantics.py`重新从原始SSE、decode/prefill response和固定SHA请求fixture验证前后两次语义/HTTP时间重叠，与state一致；它不发HTTP或使用GPU。修改decode内容、fixture payload、结果标志和重叠时间四个负例均拒绝。两条请求的语义诊断不能替代105题精度或证明模型步重叠。基线state中`acceptance_passed=false`为此harness保留待配对判断的状态，不能改写成已通过整个候选验收。

04:19已观测同一controller568657自然转入candidate，driver579266、server579270均实际存活；候选仍为冻结b0d，当前本地新增的审计/证据文件不改变其生产代码。当前没有候选完整结果，无新增端到端收益或合并结论；后续仍须完整独立配对审计、进程清理及fresh团队main核对。
