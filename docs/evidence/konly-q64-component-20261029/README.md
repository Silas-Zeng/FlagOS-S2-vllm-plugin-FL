# Q64 组件负结果归档

被测源码 `522c0b0617a8d2d862f60ebb03f05f6e08ccfcfe`，团队基线 `a455a3d`。本分支的 Q64 在当前服务相关的 chunked prefill 场景中没有收益，关闭 Draft，不合入主干。这里只公开精选元数据；完整数组、IR、缓存、日志和二进制继续保留在私有实验目录。

| 模式 | Q32 GPU 均值 ms | Q64 GPU 均值 ms | Q32/Q64 |
| --- | ---: | ---: | ---: |
| reader | 2.2202560902 | 2.3254314661 | 0.95477167 |
| append + reader | 2.2419283390 | 2.3437935114 | 0.95653833 |

固定 2048 query 行、4096 prefix、6144 总 KV、writer 行步长 2560。每模式仅一次 ABBA，顺序 Q32/Q64/Q64/Q32；capture 4 steps，每臂预热 8 次 Graph replay，每计时位 16 replay/64 steps。Q64 时间分别增加约 4.74% 和 4.54%。这是组件筛选结果，不是正式 4K/16K 服务成绩，也没有证明统计稳定性或所有长上下文均无收益。

独立复核覆盖 88 件原始文件、194,024,251 字节及其完整 SHA；核对实际加载 cubin、完整 typed ABI/IR、7 件缓存子文件和 writer 源码/产物。两种输入的完整 12,664,832 字节 cache 从保存的 CPU BF16 输入重新编码后逐字节相同；16 份完整输出/cache、Graph-before-fresh、输入切换、非默认流、canary 和只读校验均通过。Q32/Q64 的完整输出逐位相同。

原 FP64 参考范围不扩大：pattern0 全 2048 行，pattern1 原 10 行（含 63/64 边界）。保存 metrics 重算差为 0；独立参考没有新增容差。pattern1 的 BF16 diagnostic relative RMSE 为 0.0096286734，量化操作数参考为 0.0022729766。原 BF16 diagnostic 失败和 `precision_gate=false` 保留，不把组件校验写成模型精度验收，也不将两者共同的诊断误差归因于 Q64。

此前两次停止均未发射 reader/writer/Graph：v1 只是诊断 ASM 文本不同，被旧整体快照门拒绝；v2 真正生成了不同的 Q32 cubin，被执行身份门拒绝。v3 通过完整 reviewed-cache 命中，实际 Q32 cubin 为 `d1017266...`，Q64 为 `50a75f05...`；v2 的 `1100a3a3...` 是未准入的 Q32 产物，不是本轮被测 Q64。

资源字段分开记录：Q32/Q64 runtime registers 为 122/237、shared bytes 为 33792/67584；runtime `n_spills=0` 不能替代 ELF SGPR spill 字段 96/485，也不能单凭这些数值断言退化原因。

完整精选值和校验摘要见 [result.json](result.json)。独立审阅报告 SHA 为 `4b12678708abaf0f3fea6d988af063e124d1a6648cdf37c9f1fc9d2b47c320ff`，原始压缩包 SHA 为 `c291478908edc556e9ff4b9005ef8fd0687aafd11643b8e41a90bbbad44e55be`。长 KV 或其他 Q64 调度仍可作为另一个独立实验；本结果仅关闭当前实现与当前筛选配置。
