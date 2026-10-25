# A01/A02：M32/M64 BF16 BK128、无 Split-K 组件探针

团队基线0bf9d2e，比赛框架/FlagGems不升级。当前只准备算子组件，没有生产安装、GPU作业或服务收益。PR41首次模型精度与后续吞吐保持个人GPU唯一负载，本探针不并发启动。

旧PR36的dominant真实权重矩阵仅Split2/4三配置，全部负收益；早期synthetic表的无split仅BK64。本轮不是重复该表：选4个未测BK128/Split1结构：(BM,BN,BK,split,warps)为(32,64,128,1,4)、(32,64,128,1,8)、(32,128,128,1,8)、(64,64,128,1,8)。前3个减少M32填充/每CTA资源，最后一个比较M64与SDK同tile宽但更少warps。是否有益待测，不能凭warp数推断。

数学完全沿用PR36稳定BF16矩阵核的6个函数（逐函数canonical AST证明），只换有限结构表；原[N,K] BF16权重、FP32累加、最终BF16输出，无量化、权重副本或打包。Split1不产生FP32 partial/reduction工作区，无atomic。CPU分片/索引边界42组与几何配置160组通过，不代表GPU数值正确。

`tools/bench_biv150_bf16_bulk_unsplit.py`在worker相同FlagGems策略下比较实际`torch.mm(out=caller_buffer)`；真实4/20/37层QKV/O/down、M32/64、4配置共72记录。5模式完整输出用CPU FP64点积转BF16独立核对，原生/候选strict结果各自保留；finite的近误差可继续计时但不重标精度通过。变输入Graph须对eager逐位一致，保留完整失败输出和权重身份。3轮交替Graph计时覆盖完整设备序列及分配缓冲，记录寄存器/spill/shared及最小轮加速；Python分发wall与真实服务覆盖须后续另测，不能只报最优轮。

不按某层成功建立白名单。某结构若在代表层相应几何与行数有一致正收益，才扩展全部42层；实际服务接入需另外验证真实覆盖、首次105≥0.95、充分预热固定单轮4K/16K与TTFT、完整请求/tokens和匹配对照。99/105仍可测吞吐但不得合并。若无统一增益则归档并关闭Draft。

未启动命令示例：`python tools/bench_biv150_bf16_bulk_unsplit.py --out /workspace/experiments/minicpm-s2/bf16-bulk-unsplit-repr-20261025`。必须等待前序精度/吞吐进程及GPU/端口真实空闲，不重启或打断既有作业。
