# PR114 K-pair：原对齐四 case 数值入口

新入口 `tools/probe_biv150_native_k8_compact_n48_kpair_consumer.py` 默认只输出 CPU plan。
实际 ef73 数值/Graph/side 尚未运行。它沿用 c91 的可逆身份适配方式，固定 112 原 driver
SHA `5a9e6aaa51b12daae4e1767ca2ae01cbd290c4b7c25d6a610b0176692c8a5ea3`，仅有六处
可逆替换：candidate factory、114 专属 admission、候选 namespace、归档新增源码、保存
实际执行文本，以及调用一项默认 cache 对齐记录/断言。逆变换逐字返回原 driver。

四 case 仍为 41/42 live × Q pitch2048/2560，29 页/maxctx193，64 backing。
原六阶段、Graph B 先于 fresh eager B、side C、全部 O/M/L/final 的 FP64 与逐位门、
完整 guards/readonly/cache/LiveEpoch 和两个 host 拒绝均保留；.005 容差不改。
zero-length producer 保留 poison、reducer 输出正零；候选未 launch 的后 16 行仍保持
poison，对照额外 16 padding 行按原两次 N32 语义验收。BF16 proxy 与 ideal partial
仍独立记录，不把诊断失败改成通过，不授 Math105 或模型资格。

数值控制仍是原 7892 库的两次 N32/S4，而不是性能分母。candidate 只接实际
`ef732ef832b6399bf156d24c705b15e2e9e766947c0a456745673953ffbfb8fd`。
新 admission 保留原完整 compile_binding/source/IR/SASS/SO artifact 门，之后直接读
Root 原凭据 `5fc79ec23dc50108eea06ee34e3c56d7f55bd2ca5d6f3774b2531919e71b1f61`。
不把 6 KiB 凭据翻译成 114，也不改 Root 原件：实际 49d 审查中的 host-admitted S4
资源为 VGPR97 / SGPR69 / shared8192，核 full/tail 分支、K vector/scalar 互斥与回到
barrier、QK/PV 数学和等待顺序。未使用的 S2 不授准入；source_proof 历史 pending 字段
保持原样，实际编译/审查身份另行绑定。

首轮保持原 whole-cache allocation、layer offset128、page stride12544、V offset4352。
新增断言记录实际 key/scale/value 地址且要求模8为0，禁止悄悄引入额外 4B 整块偏移。
这不改 C++ 的 4B ABI，也不是已测硬件对齐结论；实际地址只有 Root 执行时才取得。
**whole-cache 4B 非8B 另列待测**，旧 2ad 和新 ef73 都未获此资格。静态保留 scalar K
路径不证明 V-SME 在该偏移下兼容。未来该项需要单独记录两库整个 cache 一起偏移的
实际地址、正确性及错误结果，不能把首轮向量路径成功转授给 scalar/V-SME 场景。

CPU 已验证真实 ef73 编译、5fc Root receipt、原 7892 consumer/codec 全部前置原件；
十个窄拒绝包括错库、错/缺/改凭据、真实 113 凭据、原阈值变化、4B 偏移和错 V offset。
对齐正反例仅为 CPU 假指针，不代表设备已查询。报告与未启动的参数模板位于
`docs/evidence/native114-kpair-consumer-cpu-20261029/`。没有复制或修改任何 GPU kernel、
原 driver、生产源码、正式服务/精度/benchmark 函数，也没有合入 PR113 的 shared6k 改动。

Root 后续冻结代码、核独占并给出新空输出目录后，才可用 `device-spec.json` 显式 `--gpu`
执行。它会保存候选所有实际输入、完整输出、Graph/side 快照及新增源码。
当前实际 consumer summary、独立数值 review、fullstep 的 pin 均为 null，fullstep 未启用。
之后完整 step 应直接 old2ad/newEf73 同输入同形状，保持原 44769 页、stride2560、dc5
writer/reducer 和 ABBA 协议；不借此处两次 N32 数值控制做性能结论。

本地默认 plan 及窄检查：

```sh
python3 -B tools/probe_biv150_native_k8_compact_n48_kpair_consumer.py
python3 -B tools/check_biv150_native_k8_compact_n48_kpair_consumer_cpu.py \
  --raw /private/tmp/cjj-native114-n48-kpair-compile-v1-raw-20261029 \
  --root-review /private/tmp/cjj-root-native114-numeric-admission-20261029.json \
  --wrong-library /private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so \
  --foreign-review /private/tmp/cjj-root-native113-numeric-admission-20261029.json
```

`cpu-preflight-argv.json` 保存此次真实本地完整前置命令；用列表直接交给 subprocess 即可复跑，
重定向到新报告文件。默认 plan 和 preflight 均断言未导入 Torch/Triton/NumPy。
