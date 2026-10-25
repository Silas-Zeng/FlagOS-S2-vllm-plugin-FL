v2 两窗整轮诊断已完成。collector 最终为 Z，日志明确 Report saved，SQLite quick_check 通过；原 driver 的超时失败状态保留。本报告只使用 CPU 读取 trace，没有额外启动 GPU 任务。

| 指标 | 4K | 16K |
|---|---:|---:|
| 去除诊断区间后的活跃时间 | 231.999898 s | 555.886648 s |
| GPU union | 227.414830 s / 98.0237% | 549.859611 s / 98.9158% |
| Graph GPU | 122.964475 s | 180.669046 s |
| 非 Graph kernel GPU | 104.055926 s | 368.937863 s |
| 非 Graph MM | 34.031612 s | 68.861543 s |
| 非 Graph normalized prefill | 27.390315 s | 209.497097 s |
| 非 Graph GQA producer | 12.268440 s | 42.855412 s |
| host wait scope | 103.245889 s | 158.759080 s |
| 其中与 GPU 重叠 | 102.697311 s | 158.175952 s |

host wait、launch API 耗时是主机经过时间，包含等待，不能称为 CPU 开销。记录中没有 GPU 活动的时间分别为 4.585068 s / 6.027037 s，不能直接归因于 CPU。Graph / kernel / memcpy 类别在这份保留时间线上没有重叠，类别 union 之和与整体 union 一致。

每窗 4 个 biv150_diag.sample 外层区间、16 次额外/诊断 Graph 均按 enqueue correlation 核对；外层时间分别为 2.284908775 s / 2.354278597 s，所有诊断 GPU 工作都在外层内，没有异步越界。相同排除 mask 应用于 GPU、NVTX、host API 与分母。它也裁去与采样重叠的正常异步 Graph 尾部 0.105501637 s / 0.191596278 s，以及少量普通 kernel/memcpy；因此这是裁剪后的诊断时间线，不是无采样运行的反事实结果。

新增 token 位置账本确认：4K 无抢占或重计算。16K 有 8 次抢占，5 个请求重新进入 worker（新请求事件 136 次，独立请求仍为 128）。prompt 实际计算位置为 2146032，其中独立位置 2097152，重复计算 48880；output 反馈位置为 133102，其中独立位置 130944，重复计算 2158。独立 output 位置加每个请求最后一个未反馈的输出 token，即 130944+128=131072，与 benchmark 输出数一致。不能把 scheduled token 总数当独立生成量，也未按 CPU 时间窗口把重计算硬归属到 GPU 耗时。

Graph 内节点事件的样本分析保持独立；这里没有将少量样本均值乘 Graph 次数推导整轮 kernel 总量。所有数值均为诊断结果，不能作正式 score。

原 trace 位于个人实例 /workspace/experiments/minicpm-s2/baseline-profiling-v2-20261025/baseline.ixsys，1050116096 bytes，SHA256 11394a8b0cbdc0b74802dd466aad6fa4884cde6916bc9a1b9510cabbba150a85。本地仅取小报告，未下载 1GB trace。
