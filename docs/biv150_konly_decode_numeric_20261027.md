# Complete private K-only decode runner (CPU-prepared)

本提交只新增runner/oracle/probe及证据，不改ae00f586的factory、support kernel、compile入口、
codec v2或生产代码；961 given-code primitive与全部原件也保持原字节。源码边界与runner
提交分开记录。默认仅stdlib CPU plan，无Torch/Triton/NumPy导入、无JIT/GPU/SSH/排队。

## 已有实机证据和当前边界

root完成ae00 inline/4K和preQ/4K compile-only，各105原件，分别3428557B/3127036B。
两包逐文件大小/SHA核验并独立重放各自generated source的actual IR；7个角色分别保存
每份actual source/TTIR/TTGIR/LLVM/ASM/cubin，不因helper源码相同推断文件相同。
inline candidate hash de788113…，preQ 8bc115b7…，BF16 baseline仍641478e6…；
各保留8 I8 QK、16 BF16 PV及完整C链。所有register/spill仍pre-launch None，非相等证明。
两份原result/errors、source/out/debug及root replay分别归档，原报告不改为numeric通过。

压缩包490046B和454466B，小于2MiB，路径与SHA见同目录compile-v1-provenance.json。
CPU检查会在内存读取tar并逐SHA核验210文件，不解包执行binary/pyc；实际codec RN32、
完整decoder数值和成本仍未测试。当前本地没有Torch，未假称Torch桥接或CUDA路径已跑。

## 单一设备入口与先决硬门

新增`tools/bench_biv150_konly_decode_numeric.py`。显式`--run-gpu`才触发冻结/clean、
main生产字节、actual frontend/JIT SHA、GPU空闲/9031检查。`--codec-only`复用同一前缀，
只做下面codec与writer门，不另建重复实现，不运行GQA或计时。

首先用同一frozen helper的M1/M8两种实例，验证141个zero/signedzero、半点、指数极值、
BF16 subnormal/max向量，code与scale对独立RN32→RNE参考逐bit一致，检查输入不变与
输出canary。随后3页encoder/writer probe验证v2 peak240边界，原V的signedzero/
subnormal/max位模式，0/15/16/47及-1/48/INT64极值slot，invalid token输入被NaN毒化。
源与目的分配完全分离；全部buffer与保留/未写区逐byte比较，不接受容差。

完整case初始encoder还要对整个4K/B64或16K/B32缓存的所有code/scale/V/保留区与
原CPU BF16输入oracle逐byte一致，非抽样证明。Qprep对全batch Qcode/Qscale逐bit；
inline不读/写被毒化的外部Qtemp。给每个cache、Qtemp和scratch/output留guard。

CPU oracle直接从BF16位域构造FP64，不依赖FP32 FTZ/DAZ；每row scale用整数有理数
直接求IEEE RN32。quotient先以CPU binary64算并转binary32；所有距整数half边界不超过
两FP32 ULP的cell重新使用整数有理数RN32，保证最终RNE code不依赖double rounding。
未命中边界的至多一FP32 ULP差异不能改变整数RNE。实际NumPy array路径已在本机CPU
覆盖全部65280种有限BF16位模式、141 probe向量，与有理数目标逐code/scale一致；
3页实际array pack也与独立stdlib writer byte oracle一致。NumPy非默认导入，不安装依赖。
该CPU证明不代表设备`tl.div_rn`舍入已过，GPU prefix仍需逐bit证伪。

## 完整数值、Graph/side 和成本

只首先支持4K/B64、16K/B32，原input factory与20261023+batch+kv seed、真实
base._pick_segments、table pitch8192、narrowed scratch、producer1warp/1stage、原reducer4。
初始原BF16 cache只读保留并核hash；BF16 append是ae00同合同ENCODE=False的private
control，不冒称已复现FlagGems服务writer调度。正式服务必须另绑定真实writer与全部reader。

原始、ragged/changed-input、zero-Q三轮，独立CPU INT64 dot及重建quantized operands的
FP64闭式attention，与原BF16 FP64闭式reference并行保存。原始选0/1/末row，ragged选
前8及末row，覆盖0/1/15/16/17/63/100/长尾。reference来自原CPU BF16和新token输入，
不读取实际encoder/writer数据作为oracle。baseline与quantized-operand pointwise仍为
atol=.005/rtol=.005；candidate对原BF16的relative RMS仍<=max(.005,1.2*baseline RMS)。
RMS在CPU用稳定hypot，不使用GPU double；RMS0/maxdiff非0直接拒绝。inline/preQ各自
按原门评判，互相位级差异只记录，不新增二者必须同F32位模式的约束。

capture三个完整closure并重播1/2/0/2/1 changed Q/newK/newV/slot/table/lengths，重播前
毒化scratch、当前append目标cache和Qtemp，与该pipeline同输入eager逐位比较。side
stream重复同一硬门。每次验证从冷源重置cache，避免上一fixture覆盖影响下一组reference。
整cache byte oracle、输入不变、cache边界、Qtemp guard、未激活/未用scratch poison均是硬门。
每轮保存NPZ输入控制/输出位模式与FP64参考、source seed/hash及compiler artifacts，便于复核。

只有finite、baseline reference、codec、cache bytes、输入不变、scratch/canary、Graph/side、
实际IR全过才允许计时。candidate数值失败默认不计时；显式`--numeric-diagnostic`才可
记录1轮diagnostic成本，numeric_passed/precision_gate仍false。硬失败不能diagnostic计时。

直接用GPU Graph事件计完整closure：

- BF16 control：append + 当前main producer + 原reducer。
- inline：Kcode/Kscale/原BF16 V append + 含Q8的producer + 原reducer。
- preQ：一次Qprep + 相同append + producer + 原reducer。

三轮交替顺序（diagnostic一轮）保存完整step中位数；另外分阶段记录append、producer、
reducer、preQ和独立initial encoder。阶段和不替代完整step，initial encoder不当服务免费，
prefill真实写入成本以后必须入账。计时后再毒化并验证同输入eager一致。未知资源记unknown，
已知变化记known_changed，不把额外Q临时量或寄存器变化归因成纯I8 MMA收益。

## CPU 重放

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_decode_numeric_cpu.py --out NEW_JSON
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_decode_numeric_cpu.py --arrays --out NEW_ARRAY_JSON
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_konly_decode_numeric.py --mode preq
```

第二条仅用已有NumPy做CPU array测试，仍无Torch/Triton/JIT/GPU。默认检查65280个
BF16位域、515个硬失败/缺成本阶段拒绝、极小RMS非0、两版210原件actual-IR replay、
默认CLI隔离和ae00 runtime字节不变。设备入口、Torch桥接和性能尚未实跑，不能据此合并。
