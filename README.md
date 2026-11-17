# MiniCPM5-2B 框架优化协作

本仓库以比赛指定的 vllm-plugin-FL `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` 为起点。新增代码与维护文档通过 PR 提交，按依赖合并；官方上游及原公开 fork 保持只读。

## 天数 BI-V150 启动

在比赛镜像中从本仓库安装插件，然后执行官方启动命令。安装包内含验收使用的四份原生库。符合比赛模型、设备和 KV 配置时，服务会加载这些库并默认启用 K-only INT8 KV、纯 Decode NativeK8、混合 NativeK8 和 Q32 复用。

```sh
python3 -m pip install --no-build-isolation .
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

需要复现 BF16 KV 对照时，在启动前设置 `VLLM_FL_BIV150_KONLY_ENCODED_SERVICE=0`。库来源、SHA256 和构建入口见 [BI-V150 复现说明](docs/biv150_qualified_service_reproduction_20261029.md)。

## 早期服务优化

| PR | 内容 |
|---|---|
| [#6](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/6) | prefill Attention 启动策略 |
| [#7](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/7) | 直方图 zoom top-p 过滤 |
| [#8](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/8) | SQLite 目标表反射 |
| [#9](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/9) | 3D decode 的 w2s1 启动 |
| [#10](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/10) | Attention 内部天然单 token 前缀拆分 |
| [#1](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/1) | CPU metadata 复用 |
| [#2](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/2) | 异步 token 的整数 gather/scatter 更新 |
| [#22](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/22) | computed H2D 与计算顺序的兼容组合 |
| [#23](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/23) | prefill Attention 的 Q fragment 跨 K 循环保持 |

#1/#2 的冲突采用已经组合验证过的 runner，保留两个 helper 和各自回退路径。代码合并提交为 `0f2b925588114d47c478707afa41e58c769c2ec4`；其全部 425 个文件与既有组件的预期组合一致，runner SHA256 为 `6f918ae6e2f32a1244bb5ac7411a5cffdd5071b1636e899e61ab6f0f20df2141`。本次复用已有验证，未重复运行测试。

## 最新 BI-V150 验收记录

2026-10-29，PR118 的冻结候选 `6982725` 已通过两档配对验收：**4K 6961.16、16K 5265.06 total tok/s**，官方 Level 3 精度 **101/105**，补充64并发为102/105。对照团队主干 `56f88737` 本轮为6509.84 / 5207.00，候选分别高 **6.93% / 1.12%**，两档TTFT均达标且请求全部成功。

这是各一轮充分预热后的正式结果。4K缓存与调优变化、预热成绩接近且排序相反的现象已保留；16K计分窗口没有观测到缓存或调优变化，但1.12%的差距仍不足以证明稳定收益。完整边界见[PR118 两档精选验收记录](docs/evidence/native118-final-paired-selected-20261029/README.md)。[PR117历史验收](docs/evidence/native117-model-acceptance-20261029/README.md)的6610.25 / 5207.72及原负结果继续保留。

这些历史成绩对应明确启用的功能、已校验的库和初始调优快照。完整环境见[受测服务的复现条件与入口](docs/biv150_qualified_service_reproduction_20261029.md)。

## 历史基线：2026-10-22，四轮弃首后的均值

`PR15+PR21+Q fragment` 已完成原口径官方 4K/16K 各四轮，跳过首轮并使用第 2–4 轮均值：

| 负载 | Total tok/s | Mean TTFT |
|---|---:|---:|
| 4K | 5313.59 | 4182.55 ms |
| 16K | 3764.67 | 137191.04 ms |

相对此前的 PR15+PR21，4K 为 +3.80%，16K 为 +9.56%。完整 Math500 Level 3 为 100/105（95.24%），105 请求成功、无空最终答案；2 题达到原输出上限。Q fragment 在实际服务中选中且 fallback 为 0，方法见 [prefill Q fragment](docs/07-prefill-q-fragment-biv150.md)。

两次官方吞吐使用相同起始 DB 种子，但候选 KV 为 `521840`，旧基线为 `521936`；因此不把差异写成严格等 KV 的单项因果拆分。仅验证天数 BI-V150，沐曦－曦云 C500 的适用性由对应平台另行验证。

## 验证与协作

- [实验总表](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/main/team/experiments.md)：尝试、改动、测法、结果及是否保留；按硬件区分。
- [方法与原始证据](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/tree/main/team)：同一方法的短记集中查阅。
- [FlagGems](https://github.com/Silas-Zeng/FlagGems)：固定比赛版本，实际库内算子改动另开 PR。
- [上游原始说明](README.upstream.md)：保留原文与许可证。

后续实现按方法提 PR，中文说明依赖、适用输入、验证及收益边界。已验证且未变的部分复用结论；新冲突或新增行为针对性核验。小输入和大输入可分别评估适用策略，不改变官方评测参数。失败方法进入实验记录，不因每轮实验重复开 PR。旧快照及历史证据继续保留。
