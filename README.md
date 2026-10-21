# FlagOS 第二季 MiniCPM5-2B 插件协作

这是团队共用的 `vllm-plugin-FL` 代码库，用于天数 BI-V150 和沐曦－曦云 C500 的框架代码协作。
代码库不要求按平台拆分目录或固定分支；提交需写明适用平台与验证范围，通用改动由双方审阅。
**当前已集成候选和本文全部实测结果仅来自天数 BI-V150，尚无 C500 验证结论。**
上游说明原文保存在 [README.upstream.md](README.upstream.md)，源码许可证和原始文件保持不变。

实验、分析和报告在[团队记录库](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput)按平台整理，
现有 BI-V150 原始路径保留；C500 资料由队友维护。
算子库使用团队共用的 [FlagGems 私有仓库](https://github.com/Silas-Zeng/FlagGems)。

## 当前代码身份

- 比赛基点：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。
- 此基点上已应用完整 SQL+C3 补丁，SHA256：
  `4aba147aa0bd3ec8dd1c12b6ae94bf39537a4cbad449e68e2fcdbfbdc0f8def6`。
- 中文说明加入前的候选 Git tree：
  `5c14cfb8d3a452e97a67758873bf18039e1bf989`。除两份 README 外，其余文件与该 tree 完全一致。
- runner SHA256：
  `e155a16669e69b7eb7aff370dbcbcf225a4f593fa42b2399722e96999b53a1f1`。

完整补丁包含保留的 prefill/top-p、SQLite 目标表反射、3D decode launch 和天然前缀 mixed Attention 分流。
当前代码已经应用这些改动，**不要再叠加完整补丁或历史增量**。
bucket、growth、hidden-gather、双 stream overlap、prefix-mask 等探索原型没有并入本候选。

## BI-V150 已验结果

固定环境为比赛镜像、vLLM `0.24.0+empty` 与 FlagGems `v5.3.5`。
独立官方 Level 3 准确率 run `20261021T090505Z-sql-mixed-level3` 为
**103/105（98.10%）**，105 请求成功、零请求错误、零 token 上限截断。

候选官方吞吐 run `20261021T092944Z-sql-mixed-official` 保持原脚本和服务参数，
每档四轮，汇总取第2–4轮均值：

| 输入 / 输出 token | 并发 / 请求数 | 耗时（秒） | 总吞吐（tok/s） |
|---|---|---|---|
| 4096 / 1024 | 64 / 256 | 355.88 | 3683.65 |
| 16384 / 1024 | 64 / 128 | 816.48 | 2729.07 |

这些是已验收的候选绝对结果。原版对照、数据库/KV条件和团队复现的最新状态，以
[技术报告](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/main/docs/technical-report.md)
和[当前交接](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/main/CURRENT_WORK.md)为准；
本页不据此声明总体加速比或 C500 成绩。

## 安装与依赖隔离

在固定比赛环境中，从本仓库安装：

```bash
python3 -m pip install --no-build-isolation --no-deps .
```

**原版基线测量期间不要安装或导入此候选平台模块。**
保留运行时会持久改写已安装的 Attention，切换插件 checkout 不能自动恢复依赖。
原版 Attention 应为 `bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0`；
BI-V150 候选使用 `54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f`。
仅在原测量结束并释放设备后，或在独立干净环境中按
[安装与验证说明](https://github.com/Silas-Zeng/FlagOS-S2-Minicpm-throughput/blob/main/docs/candidate-usage.md)操作。
不得通过变更官方参数、量化或推测解码开关代替实现。

本地准备副本保留了可用的13eb浅历史基点；不宣称包含该基点之前缺失的历史。
当前仅准备代码交接，性能探索冻结。原始证据、失败记录和未集成原型继续保存在团队记录库。
