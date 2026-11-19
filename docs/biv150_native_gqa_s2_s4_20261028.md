# BI-V150 BF16 GQA原生页循环

MiniCPM5-2B的合法decode路径中，以CoreX原生页循环实现替换原Triton GQA producer，保留原page16、8Q共享、QK/PV计算顺序、BF16概率舍入、partial／max／sum和原reducer。使用SME及TCU直接消费页内数据，减少原页循环的指令与布局管理成本；KV格式及容量预算、采样和服务参数不改。

仅支持BI-V150 16SM、BF16、N64/S2和N32/S4、D128，以及已验证的Q stride2048或2560；实际服务post-RoPE为2048。其余合法输入沿原producer，已开始的原生launch错误直接传播。没有新增常驻KV副本或packed权重。运行时只加载成品，须在启动服务前构建：

```bash
python tools/build_biv150_native_gqa.py
```

默认从build-native-gqa/corex_native_gqa_s2_probe.so加载；独立部署可用BIV150_NATIVE_GQA_LIBRARY指定成品路径。VLLM_FL_BIV150_NATIVE_GQA默认1（联动原GQA开关），设0可隔离；源码、构建源码及实际SO内容均纳入编译缓存身份。INSTALL只表示安装，正式验证另核每shape实际SELECTED/CAPTURED。

共同起点是main dac1b8fb（PR84）、原W8A8 native14f、初始DB207f。本PR九个生产文件与受测生产包c75db297逐字一致，实际GQA SO为f72064dc（144747离线构建）。

| 原官方负载 | 基线弃首暖均 | 候选弃首暖均 | 观察差异 | KV tokens（基线→候选） |
|---|---:|---:|---:|---:|
| 4K，各四轮 | 5971.44 | 6481.226667 | +8.5371% | 548288→548336 |
| 16K，各四轮 | 4529.786667 | 4826.446667 | +6.5491% | 548432→548288 |

4K四轮5349.02／6301.93／6445.80／6695.95，1024请求成功；16K四轮4250.66／4828.60／4826.82／4823.92，512请求成功。两档均实际命中S2/S4，原栈、源码／DB／两份native库恢复与清理通过。KV分别差48和144 tokens，因此保留整模正信号，**不将这些百分比声称为严格相同显存条件的单因子因果收益或稳定承诺**。

复用两shape、两已验Q布局的完整数值、全部partial位型、FP64、换值Graph、side stream及生产／ABI／入口独立审阅，不为发布重测未变代码。原官方105已完成102/105（97.142857%），valid/pass，达到95%门；105条stop、105条success，请求错误／截断／空最终答案／空提取均0，原栈与清理通过。服务初始化的S2/S4捕获不证明105请求逐次走该分支；request_coverage_proven仍为false。分支数学资格来自严格组件，官方105只证明该提交服务变体通过原评测，不混淆二者。

[方法与原件导航](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/team/methods/corex-native-gqa-page16.md)；[4K结果与原件清单](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/experiments/official/20261027T145950Z-native-gqa-abi-v2-4k-four-rounds/summary.json)、[16K结果与原件清单](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/experiments/official/20261027T152326Z-native-gqa-abi-v2-16k-four-rounds/summary.json)；[原官方105与原件清单](https://github.com/Silas-Zeng/FlagOS-Minicpm-S2/blob/codex/pr24-kv-verify-20261024/experiments/accuracy/20261027T161121Z-native-gqa-abi-v2-native14f-math500/summary.json)。

该方法独立基于PR84后的main dac1b8，不包含未合并K-only格式实验。PR81／89也修改编译身份和compute-order，未来叠加须单独合并身份并验证，不能把本次BF16资格授给encoded cache。
