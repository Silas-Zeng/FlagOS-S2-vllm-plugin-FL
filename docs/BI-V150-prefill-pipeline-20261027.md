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

因此再做唯一有针对性的K128/maxnreg128试验，观察是否用很少的寄存器溢出换回并行度；不会放宽IR/数学门槛。工具额外记录驱动对实际已加载function与真实dynamic-shared的occupancy容量查询，查询失败原样保留，不能冒称实测occupancy。若没有实际组件收益则停止该变体，不继续参数遍历。新试验尚未运行。
