# Ragged head：baseline 背衬与首次 launch 前置检查

父提交00213保持不动。本补丁仅改组件harness与CPU检查；生产head、GPU kernel、compile-only入口、Sampler、FP64 `.001/.01`、所有原逐位/RNG门和ABBA timing helper不变。31行也纳入，不能只修41/42/43。

原 `Capture` 先调用实际 `original[grid]`，然后才检返回constants，所以旧检查不是内存安全的前置门。实际linear源SHA `d431a4fa…` 使用`ram % M`连续提示、输入load无行mask；已取回R31原IR显示整tile读取。实际LibEntry `414e8a3f…` 冷路径先调用tuner/JIT，再返回constants。原配置池还含BM128/256，不能凭期望32/64假定首次autotune安全。

这里选择严格缓存准入，不扩大/替换配置池：首次head调用前，检查实际LibTuner为NORMAL、无动态FlagTune、已知selector table、真实`get_key`，必须命中原DB中的精确配置。缺失/未知/不同配置/未审hook立即停止，不调用原head kernel，也不强插配置。随后将同一grid函数包一层检查，在实际runtime评估grid时再次核BM/BN/BK、EVEN_K/SPLIT_K和grid；原返回后的检查仍保留。真实新进程的前置配置记录写入result，CPU模拟通过不等于目标机已通过。

已保存111实际v6 DB中该table为`linear_kernel_91df7045e2de13a4c456e8d13079eb7c-0f015d35c97293819201f206510edb40`：key32/130560/2048/BF16×4对应BM32/BN64/BK64/w8/s1；key64对应BM64/BN256/BK32/w16/s2，二者num_ctas1。31按真实align32 key32；41/42/43按key64。记录只证明该归档，设备必须读取自己的实际相同缓存门；不能因文件名叫v6就通过。

| 逻辑行 | 准入BM | 物理总行（含前后guard） | 逻辑行之后全部canary行 |
|---|---:|---:|---:|
|31|32|34|2|
|41|64|66|24|
|42|64|66|23|
|43|64|66|22|

两臂共享同一个logical `[M,2048]` view、stride `(2048,1)`、输入值和地址；并未变成逻辑32/64行。只增加hidden物理tail，全部填原有限值71。全storage复制回CPU后，检查完整prefix和**逻辑M之后的整个tail**并保存全部边界字节，包含tile内被允许读取的padding与tile后的guard；不是仅检查末行。其它buffer范围不变。输入初始化的CPU RNG、所有数学调用和计时循环不改。前置selector检查只发生在未计时首次capture，之后恢复原module入口，原baseline计时路径没有新wrapper。

CPU验收包括：四行数正例；48项坏配置/cache miss/动态策略/短背衬/实际grid变化/hook拒绝，opaque device调用计数为0；每个tail行首/中/末字节故障都能被发现；00213整个worker在逆换buffer、guard和一条receipt后AST一致；原factory launch/PreScaledSampler AST、timing与compile入口字节不变；生产和GPU源码零diff。模拟器不声称实际LibEntry导入或数值资格。

复跑：

```sh
python3 -B tools/check_biv150_head_temperature_ragged_component_cpu.py
python3 -B tools/check_biv150_head_temperature_ragged_cpu.py
python3 -B tools/check_biv150_head_temperature_backing_cpu.py \
  --linear-source /Users/chenjunjie/Documents/flagos/repos/FlagGems-v5.3.5/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py \
  --libentry-source /private/tmp/cjj-flaggems-tuner-source-live-20261028/flag_gems/utils/libentry.py \
  --tuning-db /private/tmp/cjj-native111-q32-n48-pinfix-throughput-v2-raw-20261029/out/flaggems-cache/config_cache/TunedConfig_iluvatar_triton_3_6.db
```

实际设备仍需Root安排；发生配置门拒绝应停止并保留原因，不能重新打开autotune绕过。fresh baseline IR/加载资源、原十门和完整Sampler费用待核；未授服务、105或吞吐资格。这里只保存源码、精选CPU报告与SHA摘要；原SDK文件/DB/数组/日志/二进制与18件历史原件留本地，不上传。
