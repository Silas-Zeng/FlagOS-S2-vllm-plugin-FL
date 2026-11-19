# PR43 v1：4K 正式负载的诊断时间线

这轮诊断的 4K 负载完整结束（256/256，失败 0），但 Graph 节点采样验收失败：实际使用插件 GraphWrapper，v1 错 patch 了 SDK wrapper，因此 samples=0。该失败不抹去已经完整保存的 CPU/NVTX、整图 GPU 和非图 kernel 时间线。本报告只使用这些可验证数据，不把它当正式性能分数或 Graph 内部 kernel 分析。

## 主要事实

- 活跃 EngineCore 窗口 **231.396551 s**，GPU 活动区间并集 **227.546097 s，覆盖 98.3360%**。未被记录 GPU 区间覆盖的时间为 **3.850455 s**；这不能直接解释为 CPU 导致的空等。
- **3,718 个 CUDA Graph** 的 GPU 时间合计 **123.321551 s**；**377,213 个非图 kernel** 合计 **103.832997 s**；memcpy **0.391548 s**。按 GPU 区间取并集，包含整图跨度，不把图内缺失节点当空白。
- host `wait_for_output_plus_sampling` 为 **104.028510 s**，其中 **103.514580 s（99.51%）与实际 GPU 活动重叠**。它不是 104 秒可消除的 CPU 开销。
- host preprocess 为 **11.611176 s**，但其中只有 **0.087277 s** 未与 GPU 活动重叠；scheduler.schedule 为 **5.505854 s**，其自身 exclusive 部分只有 **0.011238 s** 未与 GPU 活动重叠。当前证据不支持优先把常规 CPU 调度开销视为主要瓶颈。

## Host 范围耗时

单位均为秒。inclusive 包含嵌套范围；exclusive 只扣除同线程直接子 NVTX 范围，仍是 host **墙钟时间**，不是 CPU 执行周期。“exclusive 未覆盖”指该 exclusive 时间内没有记录 GPU 活动，不是 CPU 归因。

| 范围 | 次数 | inclusive | exclusive | exclusive 未被 GPU 覆盖 |
|---|---:|---:|---:|---:|
| engine.step_with_batch_queue | 4247 | 231.263637 | 0.501152 | 0.003497 |
| worker.execute_model | 4246 | 105.495372 | 0.456820 | 0.002430 |
| engine.wait_for_output_plus_sampling | 4246 | 104.028510 | 104.028510 | 0.513931 |
| gpu_model_runner: forward | 4244 | 89.261059 | 89.261059 | 2.088856 |
| gpu_model_runner: preprocess | 4246 | 11.611176 | 11.611176 | 0.087277 |
| worker.sample_tokens | 4244 | 13.557529 | 0.255251 | 0.001340 |
| gpu_model_runner: sample | 4244 | 10.646304 | 10.646304 | 0.665713 |
| scheduler.schedule | 4246 | 5.505854 | 1.846932 | 0.011238 |
| schedule: allocate_slots | 262409 | 2.922761 | 2.922761 | 0.015214 |
| scheduler.update_from_output | 4246 | 2.126616 | 2.126616 | 0.322614 |
| gpu_model_runner: postprocess | 4244 | 3.846005 | 3.846005 | 0.115364 |

底层 CUDA API 交叉核对：`cudaEventSynchronize` 8,490 次，host 合计 103.845728 s，与等待范围的量级一致；`cuLaunchKernel` 35.054781 s、`cudaLaunchKernel` 5.600277 s、`cudaGraphLaunch` 0.226036 s。这些是被 profiler 观测的 host API 墙钟时长，会与上述范围及 GPU 执行重叠，不能再相加。

## 非图 kernel 热点

这些 kernel 时间不能代表整图内部的同名算子。GPU 总和与 host forward 等范围是不同维度，不应相加。

| kernel | 次数 | GPU 秒 | 非图 kernel 时间占比 |
|---|---:|---:|---:|
| mm_kernel | 70484 | 33.987885 | 32.73% |
| kernel_biv150_prefill_normalized | 22092 | 27.484011 | 26.47% |
| _gqa_decode_attention | 21672 | 12.174832 | 11.73% |
| ixBLAS INT8 Gemm_tcu_mr_kernel | 17884 | 8.306239 | 8.00% |
| linear_kernel | 4244 | 4.774569 | 4.60% |
| native silu_epilogue | 17884 | 4.082640 | 3.93% |
| triton_red_fused_fused_add_rms_norm_1 | 17884 | 1.490097 | 1.44% |
| triton_poi_fused_2 | 17884 | 1.456391 | 1.40% |

BF16 mm 与 normalized prefill attention 合计 **61.471896 s**，约为活跃窗口的 **26.57%**；它们是当前已可直接定位的热点。decode Graph 占窗口 **53.29%**，其内部如何分配必须等修正后的采样证据，v1 无法给出完整模型 kernel 排名。

## 请求阶段与交叉核对

根据每个 request 的 prompt 长度、num_computed_tokens 与 num_scheduled_tokens 分类，不能只凭 new/cached 标记判断 decode。4,246 条 worker metadata 与 NVTX 按顺序一一对应，metadata 单调时钟时间均落在对应范围内（允许 10µs 时钟标定容差），没有未知 request：

| 阶段 | worker 次数 | prefill tokens | decode tokens | worker host 秒 |
|---|---:|---:|---:|---:|
| prefill | 2 | 4096 | 0 | 0.247287 |
| mixed | 524 | 1044480 | 28082 | 92.424618 |
| decode | 3718 | 0 | 233806 | 12.823155 |
| empty | 2 | 0 | 0 | 0.000312 |

Prefill tokens 总和恰为 1,048,576；decode tokens 261,888，加上 256 个请求在 prefill 后产生的首 token，等于 benchmark 的 262,144 输出 tokens。纯 decode 步数 3,718 与 GPU Graph 数量相同。非图全模型步骤为 2+524=526；kernel 计数与模型结构一致：normalized prefill attention 22,092=526×42；BF16 mm 70,484=526×(42×3+8)；INT8 GEMM 17,884=526×34。最后这些等式是对计数的结构性交叉核对，不是 Graph 内部测量。

## 范围、完整性与复现

- 源码 `9ac118e09cf0984af3a358aa3edf9754c6650fb4`；vllm_fl/csrc 与主线 0bf 一致；原生 SO 为已接受 control 的 5cbb SHA。
- 服务在 9031、FULL_DECODE_ONLY、max-model-len 131072、memory-utilization 0.85；4K 输入/1024 输出/256 prompts/concurrency64/seed0；三个固定 warmup 已先完成。所有数据均为 `diagnostic_only=true`、`official_score=false`。
- 只取 worker PID 282216 的活跃 NVTX 窗口 `[3258942222728870, 3259173619280354)` ns；启动、warmup、profile 前后空档不计入。完整 worker profile 单调时钟 envelope 为 231.853524 s，比活跃窗口多 0.457 s；若这段前后空档也计入，GPU 覆盖率仍为 98.1422%，未覆盖 4.307428 s。所有 GPU kernel/Graph/memcpy 均落在活跃窗口内。
- `biv150_diag*` NVTX 范围数量为 **0**，本轮没有额外 Graph clone replay；scope errors=0，callback 未禁用。Graph clone 数值验证仍为失败/缺失，绝不能宣传通过。
- collector 最终已退出，日志显示 Report file saved，SQLite `PRAGMA quick_check` 返回 `ok`。state 中保留的 flush_pending 是此前 180 秒超时留下的旧状态。原始文件没有修改。
- SQLite 使用 `cuda_nvtx`、`cuda_graph`、`cuda_kernel`、`cuda_memcpy` 视图（实际 underlying `*_data` 表和 function_name_map 已核对）；时间戳/时长单位 ns。同线程 NVTX 栈无交叉范围，exclusive 扣除可验证。
- GPU union 含完整 opaque Graph span。剩余 uncovered 时间不能单独归因为 CPU；单个最大 uncovered 区间 606.704 ms，需要进一步结合具体 API/阶段定位，不能靠此总体统计解释。

本地原始 trace：`/private/tmp/biv150-baseline-profile-v1-evidence/baseline.ixsys`（372912128 bytes）。SHA256：`3c52ddb6b3aa82266bd26b5181510648d946a72b23cc38ebb2cd2376b5ddd66a`。

远端原始 trace：`/workspace/experiments/minicpm-s2/baseline-profiling-v1-20261025/baseline.ixsys`。

分析脚本：`/private/tmp/analyze_biv150_profile_v1.py`。只读 SQLite 与 JSON，不导入 GPU 模块。复现命令：

```sh
python3 /private/tmp/analyze_biv150_profile_v1.py   --root /private/tmp/biv150-baseline-profile-v1-evidence   --out /private/tmp/biv150-baseline-profile-v1-analysis-rerun
```

共享小文件位于 `/private/tmp/biv150-baseline-profile-v1-analysis/`：summary.json、cpu_scopes.json、non_graph_kernel_hotspots.json、cuda_api_host_durations.json、raw_trace_identity.json。worker_phase_steps.json 为每步审计数据，可以选择不放入小报告。原始 trace 与 63MB worker JSON 不应直接提交进 Git。
