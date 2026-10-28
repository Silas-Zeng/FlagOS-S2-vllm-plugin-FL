# NativeK8 V-overlap 独立服务候选

本分支从本人 PR98 `41f0f4c45fb87a824bdefaa39f91847fff96543b` 创建，正常合并本人 PR106 `6a85b304b920fb1efbfb4710e74aefdc58245713`，merge commit 为 `cdb9be1`。两边均包含团队 main `1cbac531ca9afaa0e9a68624d5d10cb1d0412055`。原 PR98 和原 `22e4a5...` 服务候选保留作对照；此次没有更新官方框架版本。

生产代码只切换已测组件身份：`biv150_konly_encoded_native.py` 的固定 SO SHA、operator/schema namespace；`compile_semantics.py` 的对应 C3 指纹身份；`biv150_konly_encoded_sources.json` 中的 adapter 源码 SHA 与库 SHA。库固定为 `8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5`，namespace 为 `cjj_native_k8_v_overlap_probe`。源码逆变换检查恢复 PR98 两个 Python 文件的原字节和 manifest 原内容，PR106 的四个 GPU 源文件也逐字保持。

默认仍关闭。保留已有环境变量名称、固定服务 CLI、原 KV 格式和 16384 B page、prefill/writer/decoder fallback、shape/stride 选择、C3 运行时绑定和全部数值规则；没有新增路由、通用接口或自动编译。旧 22e 的 SHA 对不能启用本候选，新的 SO 内容、精确 schema、CUDA registration、实际加载库集合和 adapter 源码身份仍按原规则校验。

Root 后续按原服务启动流程显式提供：

```text
VLLM_FL_BIV150_KONLY_ENCODED_SERVICE=1
VLLM_FL_BIV150_KONLY_NATIVE_DECODE=1
VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY=/workspace/experiments/minicpm-s2/cjj-native-k8-v-overlap-compile-v1-20261028/build/cjj_native_k8_v_overlap_probe.so
VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256=8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5
```

组件证据见合入的 [PR106 设备证据](evidence/native-k8-v-overlap-device-v1-20261028/README.md)：实际 SO 调度/资源检查、新库四 case 全数组 consumer、独立 CPU 收据重放已经完成。Root 已报告完整 writer+producer+reducer ABBA 通过，完整记录由 Root 归档；组件结果不授予此服务候选模型或吞吐资格。

创建候选时的 CPU 检查记录在 `docs/evidence/native-k8-v-overlap-service-20261028/`：

- `service-cpu.json`：实际 8b86 ELF/schema/导出符号，四种 shape/pitch 的 attach、fence、C3、capture 记录、非法指针/epoch/alias/scalar 拒绝、原 private fallback、postlaunch error 不重试，以及身份逆变换通过。
- `fingerprint-cpu.json`：未修改的实际 `94dbbb...` helper 调用新源码的 standalone fingerprint；开关、ABI 和五种拒绝情况通过，包括拒绝原 22e SHA。无框架或设备导入。
- `encoded-cpu.json`：旧通用 `check_biv150_konly_encoded_cpu.py` 在未改的 PR98 `41f0f4c` 和本候选都失败于第 178 行 worker-tail import 列表断言。该历史检查还绑定旧 b3 补丁快照，此次未扩改它，也不把它记录为通过。

CPU 检查可用 `python -B tools/check_biv150_native_k8_service_cpu.py --library /ABS/8b86.so` 及 `python -B tools/check_biv150_native_k8_fingerprint_cpu.py --library /ABS/8b86.so --helper /ABS/UNCHANGED_94db_HELPER.py` 重跑。上述 CPU 阶段没有调用设备；实际服务后续由 Root 独占启动，未修改共享 site-packages。

后续仍需完成模型精度、真实 decode/Graph 请求覆盖，以及固定参数下 4K/16K 吞吐/TTFT 验收；实际 import/C3/库命中已进入下述设备流程。当前 `new_service_qualified=false`、`precision_gate=false`；不得将组件约 1.22× 当作模型收益。

## 2026-10-28 服务验收进度

最新主分支同机对照已完成：[完整原件和独立复核](evidence/main105-throughput-control-20261028/README.md)。4K **6371.24**、16K **4815.31 total tok/s**，TTFT **3307.13 / 104764.17 ms**，无失败。包括4K预热/归档疑点，记录不隐藏。

当前实际受测候选仍为 `b5d1a241f45efd21517d04e25a7dc5d44ac1ca06`，5c/0c66/8b86/v6；tag `cjj-native107-main105-math105-v1-20261028`，2026-10-28 20:15 CST 启动。固定CLI、3条短请求及64路burst已通过，v3诊断已记录新库两种capture certificate和真实64bucket replay。正在跑原始105题并发8，随后独立105题并发64；尚未完成精度或正式吞吐验收，不能用旧22e证书替代。正式吞吐将关闭诊断、每档完整warm后计一轮，完成并清理当前服务后才启动。

## 精度完成与正式吞吐启动（2026-10-28 21:24 CST）

[完整精度原件与独立审查](evidence/native107-math105-v1-20261028/README.md)确认官方100/105、补充64为102/105，均零请求错误，分别有2/1条32768截断；补充105题全部曾真实命中新库64桶，32题另命中32桶。保持记录完整性的限制，不改变controller原字段。

同一b5d新服务已于21:05:58启动正式full-only双场景；无Graph诊断，原件传输在启动前完成。4K单轮6374.44/TTFT3675.71ms，相对main6371.24只有+0.05%，按1%波动区间视为持平，不能称作4K收益。16K尚在运行；未取得整体验收或合并资格。
