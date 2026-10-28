# PR107 b5d Math105 / Native Graph 独立 CPU 审查

原件：`/private/tmp/cjj-native107-main105-math105-v1-raw-20261028`。55文件 / 44,782,164B 全部 SHA与大小一致；未修改原件、PR或生产源码，没有SSH/GPU。复用旧 Math105逐题审计和实际 FLGraphWrapper 证书/请求联结检查。

| 评测 | 正确/唯一题数 | 精确准确率 | 请求错误/空输出 | 32768 token截断 |
|---|---:|---:|---:|---:|
| 官方并发8 | 100/105 | .9523809524 | 0/0 | 2 |
| 独立补充并发64 | 102/105 | .9714285714 | 0/0 | 1 |

各105 predictions/reviews/index/question_id完整且唯一；相同题集与输入提示、固定temperature1/top_p.95/max_tokens32768、原官方CLI/配置和token账本均已独立核对。准确率来自保存的逐题 evaluator grades 求和，不是重新人工判题；两次采样输出不要求相同，也不因100/102之差声称并发或内核造成准确率变化。两次指标均过原.95门，补充64仍不替代官方8。截断原值保留，零failed不等于无截断。

测试源严格绑定 `b5d1a241f45efd21517d04e25a7dc5d44ac1ca06` 的252文件（按不可变Git对象核对，包含1个真实symlink解引用）；共享开发WT已前进，不拿其新HEAD当测试源。controller558579、bootstrap v3/905aaf、Native模块90c960、库8b86/operator `cjj_native_k8_v_overlap_probe::produce`、stock库0c66及实际worker fingerprint均吻合；没有借用22e证据。

两份实际32/64桶证书各有42个独立cache epoch1..42，payload逐项匹配原Native CAPTURED日志。所有2568 real replay均绑定同PID/wrapper/entry/graph及对应epoch集合，capture不计真实请求。

| 补充105实际观测 | replay下界 | 有效decode行下界 | 至少一次命中的题 |
|---|---:|---:|---:|
| 64桶 | 1828 | 115032 | 105 |
| 32桶 | 726 | 20803 | 32（105题子集） |

2554条补充评测 replay 的真实engine request_id均唯一联到本次评测 `model_output.id` 前缀，输入token数与已完成prompt一致，computed_tokens在合法区间并严格前进，且位于保存的supplemental64 HTTP字节窗口。官方8未观测32/64 Native replay。另14条属于先行64短burst（13×64+1×63=895行），不计入Math覆盖。

没有observer_error记录，但没有Finalizer worker-summary/exit-summary，**不能把最终error_count或overflow写成0，也不能宣称完整频率/记录完整性**。这里只授保存记录的覆盖下界和105题均曾命中64桶，不声称每个token/所有路由均Native。controller `native_quality_covered=false` 原值保留，没有人工翻转。清理与owner21979/start353729022终态一致，state completed/errors[]、API返回0，日志没有EngineDeadError。正式吞吐/merge资格不由本报告授予。

完整报告：`/private/tmp/cjj-native107-main105-math105-independent-review-v2-20261028.json`

SHA256：`6b860fce0c330c68229a71f0d97b468444cbb1d5d1d985d0861655df92214723`

复跑到全新路径：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error \
 /private/tmp/check_cjj_native107_math105_actual_20261028.py \
 --raw-root /private/tmp/cjj-native107-main105-math105-v1-raw-20261028 \
 --out /private/tmp/cjj-native107-main105-math105-independent-review-v3-20261028.json
```

使用Python3.12。v1审查输出保留：首次适配过强地比较了两次评测的完整 messages（包含随机message id和生成assistant文本），因此工具一致性检查失败；v2恢复原审计的输入提示比较，并核对去除运行期id/耗时后的完整输入消息。原始成绩、数据和.95门均未改动。
