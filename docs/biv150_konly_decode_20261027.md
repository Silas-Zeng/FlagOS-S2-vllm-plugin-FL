# Private K-only I8 decode source frontier

本项接在PR79已通过的961 given-code primitive之后；961全部primitive源码/参考/检查器
及原件保持不变。原CPU checker的pending字段是961历史snapshot，真实设备状态由独立evidence checker记录。ae8d3b8先归档了compile-v2和numeric-v1完整41个原文件，逐SHA重放。
提交前再次fetch origin main成功，CJJ1008 CLI核对main仍为56f1838。
这一步新增完整private GQA producer、初始encoder、append writer与preQ源码及CPU合同；
只有显式compile-only入口，**完整设备numeric/Graph/timing runner尚未准备**。
因此本冻结不能用来计时，也不把cost plan写成已经测得的完整step成本。

producer以当前main56f1838的GQA2596为唯一来源，private source factory支持两版：
inline每个active split/head CTA在Q-hoist位置量化8行Q；preQ独立kernel每token/KVhead
量化一次，producer只读I8 Qcode与FP32 Qscale。两者保持逻辑ROWS8，不把softmax/PV
扩到16行；每PAGE16只将QK改为signed-I8×I8→I32，再严格按
`RN32(RN32(INT32*sQ)*sK)`后乘原softmax scale。full/tail Kscale分别无mask读取或
以column_valid/other1读取，原tail score mask仍在后面。
原while、分段、new_max/P/BF16P、running_sum、PV C顺序、FP32 running states、
输出与原reducer均不改；exact source inverse覆盖全部非QK代码和原ABI部分。

## Codec v2 与保留的 v1

新增格式标识`cjj-konly-rn32-rne-v2`。scale仍为zero→1，否则
`max(RN32(maxabs/127),2^-126)`；code为`clamp(RNE(RN32(value/scale)),-127,127)`。
CPU使用整数有理数直接得到IEEE RN32（含subnormal/underflow与signed-zero），
不依赖设备scale，也不将CPU FP64除法冒充GPU行为。
GPU源码显式`tl.div_rn`，用floor/fraction/奇偶实现整数ties-to-even，未加入昂贵的
归一化/FMA纠半或隐含row-reciprocal替换。

v1 exact-ratio RNE参考原样保存；BF16 peak240、value120的scale相同，v1 code63，
v2 code64（负值相应-63/-64）。该差异有独立CPU记录，不回写961证据。
reader始终解释code*scale，故961 given-code I8 MMA证明仍适用输入操作数语义，
但不证明新GPU quantizer或完整decoder精度。

## 实际 SDK frontend 证据

root只读取得的actual spec math.py SHA474139…包含FP32 `div_rn`，206–210行直接
调用`create_precise_divf`；actual top language/__init__.py SHAef5fdb…导出该名字。
原文件、manifest和缺失记录保存在`docs/evidence/konly-i8-qk-20261027/div-rn-frontend/`。
spec language/__init__.py确实缺失；安装范围未找到相应C++ lowering源码。
CPU只核frontend，不声称native精确除法或目标数值已经验证。

actual math.py与公开FlagTree pinned6b143f的
[同一路径](https://github.com/flagos-ai/FlagTree/blob/6b143f17b1be27565a39da685bf9e79dc83282ed/third_party/iluvatar/spec/triton/language/math.py#L206-L210)
逐字同SHA。该ref的
[ElementwiseOpToLLVM.cpp:1399](https://github.com/flagos-ai/FlagTree/blob/6b143f17b1be27565a39da685bf9e79dc83282ed/third_party/iluvatar/backend/lib/TritonILUVATARGPUToLLVM/ElementwiseOpToLLVM.cpp#L1399)
将PreciseDivFOp映射LLVM FDivOp；安装backend二进制是否同版本仍未知。
后续必须显式compile并以v2 codec逐bit目标证伪；不能把公开源码当硬件舍入证明。

## Packed 地址及 writer

仍分配16384B/page：Kcode0..4095、FP32 Kscale4096..4223、保留空区4224..8191、
原BF16 V8192..16383。Kcode stride是byte `(16384,256,128,1)`；V stride是BF16
元素`(8192,256,128,1)`；scale用整page FP32 view，page stride4096、base offset1024、
slot2/head1。page与slot索引在乘法前转I64，已覆盖大页号导致>2^43的偏移。
不得把两种stride单位混用。BF16 dtype包装本身不能判定payload协议。

initial encoder由原BF16 K/V读取，另分配destination，写Kcode/Kscale并逐bit复制V。
append独立writer只更新slot map的有效token，两KV heads各写128个code、1个scale及
128个原BF16 V；无效slot先return，保留未写区/其他slot。CPU合同拒绝重叠有效slot。
禁止对重叠原K缓存原地压缩；physical16KiB没有回收，也没有启用内置quant mode。
将来服务接入必须覆盖generic/2D/allwriters及元数据/Graph/容量协议，本项无服务hook。

## CPU 检查与 compile frontier

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_decode_cpu.py --out NEW_JSON
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_konly_decode.py --mode inline
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_konly_decode.py --mode preq
```

默认全stdlib、不import Torch/Triton，不JIT、不SSH、不碰GPU。CPU覆盖92 codec向量
11776元素、v1/v2半点差异、FP32 underflow与signed-zero、12288个V元素地址、保留区、
首尾/无效slot、分离源/目的、23项拒绝，以及0/1/15/16/17/31/33长度的独立closed FP64
原BF16与quantized-operand attention参考。Python整数dot作为INT64范围内精确目标，
量化operand FP64参考重建给定code*scale后独立求和；二者的CPU浮点求和误差只做
oracle自检，不改变任何设备/组件误差阈值。

仅为root未来串行排期提供显式`--compile-only --commit FROZEN --out FRESH`，并选择
`--mode inline|preq --case 4k|16k`。本轮未调用。入口校验clean冻结、main生产字节、
actual frontend/JIT SHA、无现有GPU负载/9031，使用empty metadata buffers warmup，
不跑随机初始化、encoder或attention。它编译Qprep/encoder/双writer/producer/原reducer，
逐个保存全部实际IR/ASM/cubin；register/spill仅pre-launch未知，不能声称等资源。
实际decoder guard要求full/tail逻辑I8 shape、8个native I8 calls的完整两条K32 C链、
16个原BF16 PV、64个scaled C分量、无zero-C/post-add以及正确cast位点；未知即拒绝。
这些compile门仍未由真实新decoder正例验证，不把其source单测当实机通过。

## 后续完整组件的硬合同

只先做4K/B64和16K/B32，沿用当前main原input factory/seed、真实
`base._pick_segments`、table pitch8192、narrowed scratch、producer1warp/1stage、原reducer4。
完整baseline step必须含BF16 K/V append+producer+reducer；inline candidate含Kcode/
Kscale/BF16 V append+含Q量化的producer+reducer；preQ再加一次Qprep。初始encoder
单独记录，不藏进预处理，也不把producer-only数字写成完整step收益。

设备前先验证v2 codec Qprep/encoder/writer实际code/scale比特、INT64/given-code QK与
量化operand FP64参考；再验证原BF16输入的独立量化误差和最终输出。沿用原误差门，
不得用实际writer输出倒推oracle，不使用GPU FP64 RMS，RMS0与maxdiff非0矛盾直接拒绝。
poison/canary、invalid slot、reserved/unmapped bytes、原V、changed-input metadata
Graph和side stream都是硬门；内存、finite、codec、baseline或Graph硬失败不能diagnostic
计时。只有候选近精度失败可显式diagnostic计时，precision_gate仍false，不能正式验收。
完整设备runner、实测成本、full105/正式吞吐/TTFT和全部prefill消费者均未通过。
