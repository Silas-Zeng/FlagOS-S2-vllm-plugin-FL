# BI-V150 prefill流水深化（2026-10-27）

基于刚fetch的团队main `241220723a09061c182764f8f738e5ec01190367`（PR47已合）。生产vllm_fl/csrc无改动；本PR先共享隔离组件实验，尚无新性能结论。

当前真实默认大prefill为BM256/Q32/K64/w16/stage1。旧工具的BM128/w8对照已经过时，不能用它重复报告相对当前主线的收益。本工具默认经实际`biv150_prefill_q32.install()`安装该策略，在策略内部private manager取tap，断言真实launch常量、hash、Q-hoist证明及loaded资源。直接替换外层holder会绕过Q32，本PR明确避免。保留`--baseline bm128`用于历史复现。

先测两种有结构依据的变化，均保留当前生产的严格IR transform/逆变换/lowered证明，不在实验中放宽生产编译证明：

- BM256/w16/K64/stage2：检查K/V预取流水与Q共享内存生命周期；额外共享内存、寄存器及收益必须实测，不据参数推断。
- BM256/w16/K128/stage1：减少循环迭代，观察更宽score tile带来的寄存器/共享内存代价。若当前证明不支持，保留actual TTGIR/LLVM作为下一步依据，不把fallback计为候选。

隔离个人实例Python环境与私有编译缓存；先确认GPU空闲、9031无服务，不改site-packages。先用4K/16K各2048行排除明显无效路径，随后才覆盖fresh1024/2048、1024×16K及真实1985/2017尾行。随机/全行uniform causal CPU FP64参考、与当前基线的数值比较、非默认stream、changed-input Graph门槛沿用，不放松精度。Graph直接测已选中私有内核，是组件诊断，不能当正式服务吞吐。

当前CPU检查：CLI无GPUdry run、差异空白检查通过；生产目录与main逐字相同。GPU尚未启动。若组件有明确价值再实现实际服务选择、更新缓存语义、完整精度和充分预热官方4K/16K单轮验收；只有相对最新主线满足精度/TTFT且真实更优才合并。方向无价值则保留原件与关闭原因后关闭Draft。
