# A01/A02：M32/M64 BF16 BK128、无 Split-K 组件探针

最终结论：已在个人BI-V150完成136记录（含8层BF16 gate/up），全部都有3轮计时、无编译/执行错误、Graph/eager逐位一致；但136条最差轮加速比全部<1，最佳单条仅0.983。当前BK128无split四种结构没有收益，关闭且不合并，不继续扫配置或跑完整服务。准备中的冷权重ring因为没有正候选而停止，未运行。
候选和原生均未通过全部五模式CPU FP64 strict检查（候选0/136、原生0/136，候选Graph CPU37/136），相关一ULP等数值误差完整保留；没有把近误差当通过，也没有因近误差而跳过有限值计时。此次排除的决定性证据是全部性能负。
完整result、actual native源、46份编译metadata、driver/状态日志共享为66612字节tar，SHA6e75de03e528d3d366130f378b8e4e0dcd1f03f018199f031dc54a0efc25b8b8，见repr-v1/。136份失败fixture共309087164字节保留在原路径，逐文件SHA索引共享；未覆盖不同配置原件。只作为组件结论，未开展模型精度或整服务计分。以下保留前期方案与契约。

团队基线0bf9d2e，比赛框架/FlagGems不升级。当前只准备算子组件，没有生产安装、GPU作业或服务收益。PR41首次模型精度与后续吞吐保持个人GPU唯一负载，本探针不并发启动。

旧PR36的dominant真实权重矩阵仅Split2/4三配置，全部负收益；早期synthetic表的无split仅BK64。本轮不是重复该表：选4个未测BK128/Split1结构：(BM,BN,BK,split,warps)为(32,64,128,1,4)、(32,64,128,1,8)、(32,128,128,1,8)、(64,64,128,1,8)。前3个减少M32填充/每CTA资源，最后一个比较M64与SDK同tile宽但更少warps。是否有益待测，不能凭warp数推断。

数学完全沿用PR36稳定BF16矩阵核的6个函数（逐函数canonical AST证明），只换有限结构表；原[N,K] BF16权重、FP32累加、最终BF16输出，无量化、权重副本或打包。Split1不产生FP32 partial/reduction工作区，无atomic。CPU分片/索引边界42组与几何配置160组通过，不代表GPU数值正确。

`tools/bench_biv150_bf16_bulk_unsplit.py`在worker相同FlagGems策略下比较实际`torch.mm(out=caller_buffer)`；真实4/20/37层QKV/O/down、M32/64、4配置共72记录。5模式完整输出用CPU FP64点积转BF16独立核对，原生/候选strict结果各自保留；finite的近误差可继续计时但不重标精度通过。变输入Graph须对eager逐位一致，保留完整失败输出和权重身份。3轮交替Graph计时覆盖完整设备序列及分配缓冲，记录寄存器/spill/shared及最小轮加速；Python分发wall与真实服务覆盖须后续另测，不能只报最优轮。

不按某层成功建立白名单。某结构若在代表层相应几何与行数有一致正收益，才扩展全部42层；实际服务接入需另外验证真实覆盖、首次105≥0.95、充分预热固定单轮4K/16K与TTFT、完整请求/tokens和匹配对照。99/105仍可测吞吐但不得合并。若无统一增益则归档并关闭Draft。

未启动命令示例：`python tools/bench_biv150_bf16_bulk_unsplit.py --out /workspace/experiments/minicpm-s2/bf16-bulk-unsplit-repr-20261025 --edge-gates`。必须等待前序精度/吞吐进程及GPU/端口真实空闲，不重启或打断既有作业。


补充原主线实际算子覆盖：`--edge-gates` 从现有安装器 `selected=(gate_up_proj,)`、`skip_edges=4` 推导保留BF16的0–3、38–41全部8层，不依据本次精度/吞吐挑层。将这8层原BF16 gate/up按原顺序拼为[N,K]=(12288,2048)（与主线加载方式相同），仍M32/64×4配置、完整CPU FP64/5模式/Graph/交替计时，额外64记录；代表QKV/O/down加它们为136，`--all-layers --edge-gates` 为1072。该选项不是修改量化例外或生产层白名单，只扩展独立组件覆盖；目前仍未上GPU。
同时修正失败fixture命名，加入完整BM/BN/BK/split/warps，避免4种Split1配置覆盖同一文件、使早先记录SHA失配。每种配置首次失败原件和所有模式误差均保留。

实际服务静态证据已补：冻结4068f08首次完整服务缓存中，两份模型Runner在norm后调用原 `mm(out)`，权重view为(2048,12288)、stride(1,2048)，输出BF16(s72,12288)，随后SiLU/乘法核再接down。保存完整两份源码原件12683字节SHAecdf7318…及索引到docs/evidence/bf16-bulk-unsplit-20261025/。后续若探索BF16 gate/SiLU融合，必须保留GEMM后BF16边界；原激活是FP32 `gate * tl.sigmoid(gate) * up` 最后写BF16，无独立的中间BF16 SiLU边界。不能直接套用INT8路径的exp/div实现当逐位等价。这仍是静态源码覆盖，不是运行时次数或耗时归因，也没有新增性能结论。

计时审查补正：实际服务为mm(out)，因此基线和候选均在3轮交替Graph计时前各分配一次输出缓冲，所有unroll/replay复用相同地址。候选通过原run_into写入调用方缓冲，并独立核对CPU FP64 oracle及与分配型matmul逐位相同。原分配型接口的数值/变输入Graph检查仍保留，但计时不再让其展开4份输出工作集，与基线单个out不对称。Python分配/分发成本仍没有被Graph事件计时覆盖，后续真实服务另验。此时仍无GPU结果；尚未改变已冻结原验收任务。
