# Ragged head 显式行 mask 修正

原 b606 的 31 行 compile-only 已完成，实际 LLVM 暴露了原行提示的风险：仅 tile 基行对 M 取模，输入 SME descriptor 的 length 为 -1，仍按连续 32 行读取；不能把标量 `rm % M` 枚举当成实际生成代码的逐行回读。该进程没有执行 head，不能声称已经观察到设备越界或数值失败。

原身份为 b6066b0899df81605ca1a19867459954e88d8699、owner79187/start357210613 同 start Z、kernel SHA4c800693…2537。18 原件共 483723 B，完整 gzip131925 B/SHA317cff6bf6d32306596490924634017729924470ddf2c5e0dc9511f955161ccb 留在本地及 Root 已保存的 workspace artifacts。**本提交不上传该 gzip、cache、log、cubin 或完整 IR**，只保存完整 SHA 索引、源码绑定和最小摘录/报告；没有换方式上传被审批拒绝的原件。

`original-compile-review.json` 给出实际 LLVM 行号和 `%0 → %53 → %.pn846 → %106 → %110 → SME load` 输入地址链，以及 descriptor 的 -1。原 `gpu_executed=true` 表示 compile-only 初始化 CUDA/分配；head/attention/writer/encoder launch 与输入初始化均为0。prelaunch n_regs/n_spills 为 null，不能称 loaded 资源或零 spill。

本次只增 `RAGGED_M: tl.constexpr=False`。32/64 生产调用传 False；31/41/42/43、compile-only plan 和 component factory 传 True。True 分支使用真实 rm；EVEN_K 和普通 K 循环的 A load 均为 `mask=rm[:,None]<M, other=0`；K 尾段合并原 K mask，other 保持原零张量。ragged 行不再带 modulo 后的连续/对齐提示。W 的提示、dot 次序、K 循环、bias、store、acc→BF16→FP32÷temperature 不变。

CPU 检查把 False 分支投影还原为原 kernel AST；True 分支仅允许真实 rm 和上述三处 A-load mask 改动，除此以外 AST 必须相同。**整个 kernel 的字节已经改变**，新 SHA15a599098fa4bc5ff61c28cff143e9459e2eca592c0beafeda7189246b87025c；不声称新编译机器码与原32/64相同。旧候选实际字节与两份原历史 receipt 的 SHA、历史证据目录保持不变。

生产 fake-call 协议检查涵盖所有新行 True、32/64 False；组件 fake-call 与 compile plan 同为 True，其余原配置保持。worker、FP64 `.001/.01`、完整 scaled/filtered/ID/RNG 零容差、Graph/side、canary 与 ABBA 原门没有改。默认开关仍关闭。`source-cpu.json`、`component-cpu.json` 都只证明本地源/协议，不授任何 GPU、数值、采样、服务或105资格。

下一步先由 Root 对新 R31 实际编译进行 IR 审查，确认 mask 在实际读路径生效，然后才运行原数值门。41/42/43 的 stock F.linear baseline 不在本修正范围；其完整 SME panel 可能超过现有 hidden backing 行数，需单独核实际 baseline 读取边界，不能因为候选已 mask 就默认 baseline 安全。此次没有改参考路径、扩大其 backing 或放宽原门。

本地复核（输出目录外的文件名必须尚不存在）：

```sh
python3 -B -W error tools/check_biv150_head_temperature_ragged_cpu.py --out /private/tmp/head-mask-source-new.json
python3 -B -W error tools/check_biv150_head_temperature_ragged_component_cpu.py --out /private/tmp/head-mask-component-new.json
python3 -B docs/evidence/head-temperature-ragged-mask-fix-20261029/review_original_compile.py --raw /private/tmp/cjj-head115-ragged-r31-compile-v1-raw-20261029 --out /private/tmp/head-old-compile-new.json
```

Root 后续编译命令仍为 `tools/compile_biv150_head_temperature_ragged.py --compile-only --root-owned-exclusive --commit <冻结commit> --row 31 --out <全新绝对目录>`；本地没有执行。原 gzip 和重复源码备份还保存在 `/private/tmp/cjj-head115-mask-fix-local-originals-20261029`，不属于待发布内容。
