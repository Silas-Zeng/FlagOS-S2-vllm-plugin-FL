# PR106 a1eb 实际 SO 的 CPU 审查

对象是实际扩展 `cjj_native_k8_v_overlap_probe.so`，SHA256 `8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5`，源提交 `a1eb7deee5e8ef99136128984256762096a3657f`。本次只解析磁盘文件，没有加载扩展或调用设备。编译记录中 CUDA initialized 前后均 false，所有设备调用计数为 0。

结论：**未发现资源退化，主要晚等待机制在实际机器码中保留，可以由 Root 决定进入原四 case 数值验证。** 不能把此报告当作 GPU 数值、模型精度、吞吐或 occupancy 通过。

实际 SO 的 `ixobjdump --IR` 和 `--sass` 都成功。审查重新核对下载清单及输出 SHA，并逐条将 2768 条反汇编指令的两个 32 位字与 SO 内嵌 GPU ELF `.text` 绑定。下列行号均属于这两份实际 SO 导出文件，非独立编译的 `.s`。

| 路径 | IR：K barrier / V SME / QK / V wait / V barrier / 首条 PV | SASS：K barrier / V SME / V wait / V barrier / 首条 PV |
|---|---|---|
| S2 完整页 | 594 / 602–605 / 611–629 / 743 / 746 / 765 | 722 / 741、743、744、748 / 611 / 617 / 643 |
| S2 尾页 | 1132 / 1140–1143 / 1156–1185 / 1318 / 1363 / 1384 | 944 / 952、955、958、961 / 1092 / 1150 / 1177 |
| S4 完整页 | 2217 / 2225–2228 / 2234–2252 / 2366 / 2369 / 2388 | 2065 / 2084、2086、2087、2091 / 1954 / 1960 / 1986 |
| S4 尾页 | 2755 / 2763–2766 / 2779–2808 / 2941 / 2986 / 3007 | 2287 / 2295、2298、2301、2304 / 2435 / 2493 / 2520 |

四条路径的 IR 和机器 CFG 都已检查：K 就绪 barrier 支配 V 发射；QK、P 的两条 exponent、八条求和加法和两条 BF16 转换先于 V wait；V wait 支配 PV 读取和 MMA，尾页还支配两条 V 清零/保留低半字写入。S2 尾页 SASS 写入在 1127、1139；S4 在 2470、2482。完整页 SASS 的 wait/PV 块被排到 QK 块之前，实际通过后向分支进入；已按指令地址构建 CFG，未使用文本行号大小当作执行顺序。S2 `0x139F8` / `0x13A10` 回到 `0x13300`，随后在 `0x13320` 等待。S4 结构相同。

两个需要准确表述的编译调度细节：完整页第一条 QK MMA 被排到第一条 V SME 前，后续 QK 与 V SME 交错；K 到 V 的顺序仍保留，并未变成 PR96 的 V-first。机器码在 wait 之后仍保留部分 rho exponent 和 running-L FMA，因此**不能声称全部 softmax 指令都在 wait 之前**。源码及 IR 中的晚等待已兑现为 QK/P softmax 主要计算与 V 搬运重叠，隐藏了多少延迟需要设备实测。

资源与原始 `22e4a5e2…` SO 比较：S2/S4 都为 VGPR 100、SGPR 68、shared 8192 B、warp 64，private segment 0、VGPR/SGPR spill 均 0，完全一致。实际 codec `.text` SHA256 `936ce115af95401dda3a37cf310b62baa606003a18910ff46de4707fbaba19cf` 与原始 SO 逐字相同。两个 specialization 各 291 条数值 IR 指令（含量化 round/clamp、fabs/max、I8 QK、缩放/除法、softmax、BF16 P 和 PV）在仅重命名 SSA 后，运算、常数、浮点 flags 和操作数复用关系与原始基线 IR 一致；这是一项数值指令片段比较，辅以前次源码逆变换结果，不声称整程序形式证明。

文件：

- [实际 IR](/private/tmp/cjj-native-k8-v-overlap-compile-v1-raw-20261028/out/actual-so-inspection/IR.stdout.txt)
- [实际 SASS](/private/tmp/cjj-native-k8-v-overlap-compile-v1-raw-20261028/out/actual-so-inspection/SASS.stdout.txt)
- [机器可读报告](/private/tmp/cjj-native-k8-v-overlap-actual-SO-review-20261028.json)，SHA256 `b65851e654cf810598c194343caca391572f21a539283632d46de0cc8ee44a1b`
- [可复跑 CPU 审查脚本](/private/tmp/review_cjj_native_k8_v_overlap_actual_SO_20261028.py)

未填写 approved receipt；`precision_gate=false`、`GPU_numeric_verified=false`、`performance_verified=false`。
