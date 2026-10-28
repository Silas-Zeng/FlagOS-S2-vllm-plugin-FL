# PR106 fullstep 独立 CPU 复核

受测源码 `569c2e929b51ab639a7b154d826516fa0b4fac33`；新 K8 SO `8b86cb…`，BF16 对照为旧 **aace** 组件，**不是 main105 的 0c66**。没有重跑 GPU、SSH、修改原件或 PR。

复核通过：192 原件 / 144,868,555B 逐 SHA；新 4474/8b86 与旧 c2e/22e consumer 分别独立重放完整四 case，没有转授旧资格。复用原 3eb 后验数值框架，仅作明确身份适配。

两档合计36个完整 buffer 快照、144个 O/M/L/output 数组硬门、首尾 guard/非活跃 poison、Graph B 和 side C 先于 fresh eager、同 arm 四数组逐位比较全部通过。真实 width16 scratch、非零 storage offset、Kcode/scale full-base 与 V-offset 角色、writer 6-pointer/6-I64 ABI、四个实际 JIT function 源码/产物及已记录资源相符。原硬门仍为逐元素 `absdiff <= .005 + .005*abs(ref)` 且 RRMSE <= .005；最终输出两 arm 分别对原 BF16、I64 量化QK解量化后的理想 FP64，partial 对原逐页 BF16-P 实现参考。两档原 ideal-partial 诊断各12/12 false，原值保留，不改 dtype/阈值或把诊断计成通过。

重建了完整冷 counter/编码 oracle、两档 phase0 两 arm 全行 FP64，以及所有 phase 的整 cache 预期 chunk digest；与保存参考/运行时完整字节比较账本一致。最大输出 RRMSE：4K .0022672，16K .0023030。全部144数组门亦已对各 phase 保存 FP64 重算指标。没有声称独立重新构造了 fullstep phase1/2 的 FP64 golden。

| 组件 | BF16平均事件 ms/step | K8平均事件 ms/step | BF16/K8事件比 | host wall比 |
|---|---:|---:|---:|---:|
| N64 / 4K / S2 / Qpitch2048 | 0.416383967 | 0.341390297 | 1.219671358 | 1.219629428 |
| N32 / 16K / S4 / Qpitch2560 | 0.805618048 | 0.657919794 | 1.224492795 | 1.224449785 |

ABBA 顺序、34276物理页、8192列 block table、unroll8、warm128/replay64 及原计时定义不变。计时包含 append+producer+reducer，初始编码成本独立记账。新 KV source pitch256，不授真实服务 pitch2560 写入路径计时资格。与历史22e是跨 run 比较，即使 BF16 接近，也不能据此宣称稳定、因果或模型吞吐收益。

证据边界：保存的完整输出 guard/poison 可独立重放；完整 GPU cache/input body 没有导出，只能核 CPU 重建的 cache chunk 与原运行时全字节/readonly 断言账本，不能说本次重新读取了 GPU 内存。SO不在此fullstep下载包；身份由原驱动内容 hash及另外真实编译归档核验。没有官方105、服务、容量或正式成绩验收。

完整报告：`/private/tmp/cjj-native-k8-v-overlap-fullstep-independent-review-v1-20261028.json`

SHA256：`058d5bd0f4d650e08527263d2cee986f905079d0983eadff860f3b4547ec90b5`

新/旧 consumer 重放报告：同路径 stem 后加 `-consumer-replay.json`，SHA `a13c67aaea594e504e713b76928e4b593daed5026469171c1b960619aaa76797`。

复跑使用全新输出路径（计时字段会变化）：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error \
 /private/tmp/check_cjj_native_k8_v_overlap_fullstep_actual_20261028.py \
 --out /private/tmp/cjj-native-k8-v-overlap-fullstep-independent-review-v2-20261028.json \
 --regenerate
```

使用 Python3.12；原件根目录 `/private/tmp/cjj-native-k8-v-overlap-fullstep-v1-raw-20261028` 保持只读。
