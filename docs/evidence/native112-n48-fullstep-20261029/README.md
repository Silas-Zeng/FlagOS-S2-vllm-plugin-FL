# PR112 N48 compact 完整步骤实际复核

实际 `e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8` 运行的两档完整步骤通过原数值、
ABI/JIT 和资源身份检查。对照是同形状的生产 compact TL48，候选是实际库
`2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`。
两臂都包含 dc5 stride2560 append、producer、同一个 stride-aware reducer。
这不是旧 BF16 或两次 N32 的性能比较，也不是模型吞吐分数。

最终 `independent-review.json` SHA256：
`2d8d5b873eb30084da7dd0e25332088462dd5cfef62c45fe45c75bccd07ede5d`。
最终 `replay_fullstep.py` SHA256：
`5e7a4911f6535c67c32735adf11286e635c0c08f56c476d1e1cf6443db75b978`。

| case | 非空行 / query 行 | 基础 KV 长度 | TL 完整步均值 ms | Native 完整步均值 ms | TL / Native |
| --- | ---: | ---: | ---: | ---: | ---: |
| valid41 | 41 / 48 | 17408 | 1.5162745714 | 1.3158536553 | 1.1523124667× |
| valid42 | 42 / 48 | 16896 | 1.4828296900 | 1.2829557061 | 1.1557918040× |

上述是本次 GPU event ABBA 均值；host-wall 比值分别为 1.1523047084、1.1557921225。
保持每臂 warm128、unroll8、每次测量64 replays，顺序 TL/Native/Native/TL。初始化、源 packing、
完整 GPU 编码及 CPU oracle 成本分别记录，不计入 append+producer+reducer 的步计时。
没有据这一次组件运行宣称稳定收益、模型收益或正式分数。

## 原数值和数据范围

全部 118 个原件、91,645,512 字节重新核对 SHA。summary 为
`5a6aefeb7befb777e890402fa9742774ea23ae5e696be67c74f04418df37b131`，download manifest 为
`c5e3d5dedccf320652a8b6519ec4532138d314bf7186cdeb5946818077702324`。
owner PID 62378 / start ticks 356248262 在 collector 记录为同 start、状态 Z；owner 的 source
为 e1c，bundle 为 `0643e3e48b2dfebfe6d8201719e6887f42787859ad7f244149719e14350548e2`，
命令显式包含 `--gpu --root-owned-exclusive`。

复用原 `1c527358…` fullstep auditor，通过26处可逆身份/布局/JIT角色适配，
`metric_same`、`check_snapshot`、`load`、`array`、`equal_fields` 五个函数逐字不变。
使用 f569 的真实 fixture：两臂均调用原完整 D128 I64 量化 QK/FP64 reference，
没有把 TL 当作 BF16。36 份完整输出 backing、144 项硬门、Graph-before-eager、side、
pre/post timing 全数组和 padding/canary 检查通过。

设备门仍是逐点 `.005 + .005*abs(reference)` 与 relative RMSE `.005`；最大 RRMSE 分别约
0.00227620、0.00227038。最大绝对误差约0.00726，符合原绝对加相对联合门，不能写成绝对误差
一律小于0.005。两档各12项原 ideal-partial diagnostic 都失败，继续与 pagewise BF16-P
partial 硬参考分列，未计为通过。之前29页 consumer 的48项 ideal诊断也没有被改写。

完整44769页、每臂561,582,336字节 compact cache 的各阶段比较记录、初始 BF16 counter、
canonical 编码 oracle、全部线性 K/V/identity-slot 源均重建核对；两档两臂 phase0 的完整原
FP64 golden 重新生成，其余保存的各阶段 golden 按原算法检查完整实际输出。
CPU golden 重算容差仍是 `rtol=2e-12, atol=2e-12`，不改变 GPU 数值门。
GPU cache/输入本体没有全量存入 raw：独立复核的是可重建预期 hash 与运行时完整字节比较记录，
不能称作重新读取了 GPU 内存。

Q、K、V 均是独立的 stride2560 backing。valid41 预留44649页、剩120页；valid42 预留44394页、
剩375页。零长度 padding 的负 slot、整页范围、分片地址及 encoder/consumer 的 V full-base
与 +4352B 角色保持正确。A/B/C 是三个独立定义的前缀快照，不是连续生成三步的证明。
本次没有覆盖共享模型 QKV allocation、42层实际容量、模型精度、服务或最新 main 吞吐。

## 两个必须保留的记录说明

**原 summary 的 `GPU_executed=false` 没有改写。** 这是 `plan()` 的默认字段被复制后，
`run()` 未更新造成的记录缺陷。实际执行由 owner命令/source/bundle、真实加载2ad库与独立schema、
五角色实际 JIT 产物、完整输出数组、Graph/side变化和正值 ABBA event/host clocks 交叉佐证。
报告中的 `GPU_execution_record_erratum` 单独说明这些证据，不能把原字段悄悄改成 true。

**TL 的 ELF SGPR spill 为26，不能笼统称“所有资源零 spill”。** 原 frontend 门是每个实际
loaded handle 的 `n_regs>0`、`n_spills==0`；该门通过。早期离线脚本额外假设 ELF 的所有
spill计数也必须为零，该假设超出了原门，实际 TL 参考否定了它。Root 核对原门后批准纠正
记录口径；数值、阈值、运行源码和原件全部不变。

| 角色 | handle n_regs / n_spills | ELF VGPR / SGPR | ELF VGPR spill / SGPR spill | ELF private B | ELF静态 shared B | JIT动态 shared B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 初始encoder、两臂append | 9 / 0 | 9 / 44 | 0 / 0 | 0 | 0 | 0 |
| TL48 producer | 144 / 0 | 144 / 104 | 0 / **26** | 0 | 0 | 6144 |
| reducer | 11 / 0 | 11 / 36 | 0 / 0 | 0 | 0 | 1024 |

上述实际 cubin ELF、ASM header/metadata、handle及 JIT metadata 逐项核对，ABI 和实际函数源
均绑定。TL IR 包含8个 signed-I8 QK、16个 BF16 PV 调用和24个未带 fast/arcp 的 F32 divide；
完整 IR/ASM/cubin 保留在小原件包。不把 runtime `n_spills` 与 ELF scalar-register spill混为一谈，
也不据一个字段推断 local-memory流量、occupancy或性能因果。候选2ad的100VGPR/66SGPR、8192B
shared、0 private及0 VGPR/SGPR spill来自之前已审核的固定实际 SO，继续通过14c8批准和2ad身份绑定。

`audit-history-and-errata.json` 保留两条说明、诊断失败数量和SHA链。两次失败脚本/log完整保留：

- v1仅因JSON反序列化的6组label pairs为list，本地为tuple而失败。规范化JSON后仍要求整体严格相等。
- v2完成全部原数学/缓存/ABBA检查后，在新增的“ELF SGPR也为0”假设处失败。
  原数学报告SHA `db985c037f33727c238ba4c7b885692f6ac548c702083f88f2ec3c941006699b` 原样保存。
  最终只完成资源层核对和说明，没有重跑这份已完成的数学结果或GPU。

两个最小diff分别在 `v1-to-v2.patch`、`v2-to-final.patch`，不是隐藏失败后替换原记录。

## 文件与重放

`original-metadata-only.tar.gz` 含40个非blob原件和原download manifest，共41成员，
1,867,498字节，SHA256 `9e4970c9414bc1c2263917a11a35c7dc5fd7bfd54253b4863be63483006b8528`。
包含原summary、owner、log、实际frontend/frozen sources和全部JIT source/IR/ASM/cubin/metadata。
`metadata-package.json` 逐项列出成员及省略的78个blob文件，每项都有原路径/大小/SHA。
pyc和二进制launcher仅原样保留作为传输证据，不由校验入口导入。

完整gzip 42,714,563字节，SHA256
`3955dedbe0c7b21ea6aa48063da3597096262e2d322ca04aa9fcae5942fd583e`，持久位置见
`full-archive-location.json`。已重hash Root的现有副本，没有重复复制大raw。每个共享文件均≤2MiB；
小包省略数组，不能代替完整数值重放所需的完整archive。旧consumer/compile大原件继续作为外部
依赖，不重复打包。入口不会联网fetch、启动服务、导入Torch/Triton或执行GPU。

轻量验证共享字节和包内原件，不重新执行数学：

```sh
python3 -B docs/evidence/native112-n48-fullstep-20261029/verify_shared_evidence.py
```

默认 `replay_fullstep.py` 只输出CPU plan。完整独立重放需要 NumPy、所有完整原件、相关本地Git
对象（e1c、f569、fc25、dc5、3eb）和新的输出路径。下面路径对应本次本机原件，其他机器需显式
传 `--source`、`--consumer-raw`、`--consumer-review`、`--compile-raw` 等真实位置：

```sh
python3 -B docs/evidence/native112-n48-fullstep-20261029/replay_fullstep.py \
  --run --source /private/tmp/flagos-cjj-compact-native-n48-fullstep-20261029 \
  --raw /private/tmp/cjj-native112-n48-fullstep-v1-raw-20261029 \
  --summary-sha256 5a6aefeb7befb777e890402fa9742774ea23ae5e696be67c74f04418df37b131 \
  --manifest-sha256 c5e3d5dedccf320652a8b6519ec4532138d314bf7186cdeb5946818077702324 \
  --out /private/tmp/native112-new-fullstep-review.json
```

本次最终化使用同一入口，额外传
`--resume-original-audit docs/evidence/native112-n48-fullstep-20261029/original-numeric-audit.json`，
只接受固定 db985 原报告，避免重复已通过的数学。这个模式继续核真实身份和资源层，不能用任意
“passed” JSON代替。完整重放会重新计时CPU审查，因此报告中的秒数及中间报告SHA可能变化；原始
2d8/db985仍保持不变。最终脚本仅在原数学门和额外身份资源核对完成后写最终报告。
