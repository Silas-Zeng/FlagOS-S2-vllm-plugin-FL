# PR106 完整组件实际结果

受测源码 `569c2e929b51ab639a7b154d826516fa0b4fac33`，8b86 库；独立 CPU 复核报告 SHA256 `058d5bd0f4d650e08527263d2cee986f905079d0983eadff860f3b4547ec90b5`。192 原件、144868555 字节逐 SHA 通过；完整说明和复跑依赖见 [independent-review.md](independent-review.md)。

| 场景 | BF16 ms/step | K8 ms/step | BF16/K8 |
|---|---:|---:|---:|
| 4K、64/S2 | 0.416383967 | 0.341390297 | 1.219671358 |
| 16K、32/S4 | 0.805618048 | 0.657919794 | 1.224492795 |

包含真实 append、producer、reducer；旧 aace 是组件对照，**不是 main105 的 0c66**。原 34276 页、ABBA、warm128/replay64、数值容差保持。144 数组硬门、Graph/side、guard/poison 均通过；原24个 ideal-partial 诊断失败保留，未改成通过。

完整大数组未放入 Git。本目录分享 summary、逐文件 manifest、后验审查代码与报告；不能把它称作完整原始数组归档。原件保留在个人实例 `/workspace/experiments/minicpm-s2/cjj-native-k8-v-overlap-fullstep-v1-20261028` 及 Root 下载目录 `/private/tmp/cjj-native-k8-v-overlap-fullstep-v1-raw-20261028`。首次大文件传输超时后仅重试读取，未重跑 GPU。

独立重建 FP64 golden 的范围是 phase0 两 arm 全行，其余 phase 依据保存参考重算全部数组门；各 phase 的 cache 预期 chunk 均重建。GPU cache/input body 没有导出，原运行时全字节比较账本不等于本次重新读取 GPU 内存。此结果不授予模型精度、真实服务或正式吞吐资格；服务验证在 PR107 进行。

原始 `fullstep-summary.json` 为2.18MB，完整封装于 `fullstep-summary.tar.gz`（508663B，SHA256 `4ade6083c17ce41fa99b8d27561abc953593d4fa4acea5ccd4f5eaba296d0bbb`），避免在PR diff中展开六万行重复记录。解压后内容与下载manifest原SHA一致。完整大数组还持久归档于本机 `repos/experiment-raw-archives/2026-10-28/cjj-native-k8-v-overlap-fullstep-v1-raw-20261028.tar.gz`（67282702B，SHA256 `1a05892c24ae3de50f8bd90774e9e6b00d6508190af55947f442804920c6fef7`）；该大归档未推送Git。
