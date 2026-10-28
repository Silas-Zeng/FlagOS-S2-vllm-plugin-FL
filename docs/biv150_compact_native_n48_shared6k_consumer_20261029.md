# PR113 6 KiB：复用 N48 四 case 数值入口

新入口 `tools/probe_biv150_native_k8_compact_n48_shared6k_consumer.py` 默认只输出 CPU plan。
实际 GPU 尚未运行。它读取并固定原 112 consumer SHA `5a9e6aaa…c8a5ea3`，仅做 5 处
可逆身份/留档适配：候选 factory、admission、operator，以及归档新增源码/实际执行文本。
完整逆变换回原字节；不复制或重写数值实现。

原 41/42 live × pitch2048/2560 四 case、29 页/maxctx193、64 backing、六 phases、
两个 host 拒绝、所有 O/M/L/output FP64 与逐位检查、Graph B 先于 eager B、side C、
整输入/cache/guard/epoch 和 .005 容差保持不变。原 BF16 proxy 与 ideal partial 诊断
仍独立记录，失败不会被转成通过。两次 N32/7892 只作原数值对照，不作性能分母。

新 admission 复用原 compile_binding，核全部实际编译源、SO、IR/SASS 与 artifact manifest；
随后精确读取 Root 已发出的 `b481…` 原始完整审查，并与 `1dc / 5cf / S4 6144B` 绑定。
旧 8192B/S2+S4 模板不适用于此库，因此未伪造旧字段或改写 Root 原件。硬件对齐、GPU
数值、Graph、side、occupancy、完整步骤、服务资格在此 receipt 中仍为 false；Root 批准
只允许接下来的隔离数值测试。原 reference 合同、codec 资格、LiveEpoch 均复用原实现。

CPU 验证已完成：实际新编译/Root receipt 与旧 7892/codec/consumer 原件完整 preflight；
7 个窄拒绝，包括错库、错/缺/改 receipt、修改原数值阈值；原 fixtures/reference 对象相同。
报告见 `docs/evidence/native113-shared6k-consumer-cpu-20261029/`。这不是 GPU 数值资格。
`device-spec.json` 是不能自动启动的参数模板；只有 Root 在当前服务结束后，核实独占并
填入冻结代码路径、原 b481 路径、新空输出目录，才可执行。所有原始数组和新增适配源码
都会保存于完整 artifact manifest。

下一 fullstep 将直接比较 `2ad old8k / 5cf new6k`，两臂同 44769 页、stride2560、dc5
writer、reducer、原 warm128/timing64/unroll8 ABBA；不对旧 TL 比值作跨运行相减。
新 consumer 的实际 summary、原件与独立 review 尚不存在，相关 pin 为 null，当前没有
可执行 fullstep 入口，也不继承原 493e/c079 的新候选资格。详见前一证据提交的
`next-device-stages.md`。

本地窄 CPU 检查：

```sh
python3 -B tools/check_biv150_native_k8_compact_n48_shared6k_consumer_cpu.py \
  --raw /private/tmp/cjj-native113-n48-shared6k-compile-v1-raw-20261029 \
  --root-review /private/tmp/cjj-root-native113-numeric-admission-20261029.json \
  --reference-library /private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so
```
