# PR112 N48/S4 实际四 case 数值复核

冻结 `09e502134fcf9fa7168c9f342c2a3d33754cee1c`（consumer 代码与 `1c0de575` 相同）
使用实际库 `2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`，
完成 N48/S4 的四个实际设备 case：live rows 41、42 × Q pitch 2048、2560。
参考为 PR110 的实际 `7892a6aa…`，每步两次 N32 producer 和两次 N32 reducer。
该参考只用于数值对照，不作为性能分母。

本次独立 CPU 重放报告 `independent-review.json` SHA256 为
`c07937c3d6c3ffc78a5c7b213271e4f864a9463775fe932f9f2aadbff08ef4d0`。
实际 consumer summary SHA256 为
`493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272`。
170 个原件、59,569,068 字节全部重新核对 SHA，owner PID 62050 / start ticks 355994499
在下载 manifest 中记录为同 start、状态 Z。没有重跑设备。

| 从完整保存数组重新计算的检查 | 结果 |
| --- | ---: |
| 原 FP64 硬门（O/M/L/final） | 176 / 176 |
| 原 BF16 final proxy | 32 / 32 |
| 两 arm 完整 first48 O/M/L/final 逐位比较 | 96 / 96 |
| 重新生成的原 FP64 golden 数组 | 168 |
| 完整 Q/cache/length/table 只读比较 | 336 |
| live epoch 保存元数据重放 | 24 |
| N48/S2、N32/S4 host 拒绝与完整 buffer 不变 | 2 / 2 |
| 原 ideal-partial diagnostic | **48 项均未通过，原样保留** |

硬门仍是逐点 `absdiff <= .005 + .005*abs(reference)` 且 relative RMSE ≤ .005。
ideal partial 与 pagewise BF16-P partial 的差异不是硬门，通过数量不包含这 48 项。
CPU 使用原 v2 fixture、完整 128D I64 dot/FP64 golden 和原 metrics；新适配器从冻结源码
AST 提取完整 `check`、`core_bytes` 函数体，仅将 snapshot 回调接到保存的完整字节。
没有另写数值参考、改 `.005` 容差或更换 codec。

Golden 的 inactive NaN mask 单独精确比较，有限值使用既有 CPU 重放容差
`rtol=1e-12, atol=1e-11`；实际最大 host 差为 `1.4210854715202004e-14`。
这只是跨 CPU 的参考重算一致性，不替换设备硬门。保存 metrics 与重算 metrics 另核对
`rtol=1e-12, atol=4e-14`，所有通过/失败布尔值必须完全一致。

两 arm 的完整 64 行 backing 语义分别检查：candidate 只 launch 前 48 行，后 16 行的
O/M/L/final 必须保持 poison；reference launch 全 64 行，后 16 行长度为零，O/M/L
仍 inactive poison，final 由 reducer 写为正零。所有 16 个 scratch segment 的物理布局、
segments 4..15 guard、外保护区和 producer-only output poison 都参与比较。
Graph B 在 fresh eager B 前运行，八份完整 input/output backing 与 fresh eager B 逐字一致；
A/B reducer 未改 O/M/L，side C 由原 FP64 门独立覆盖。保存的 before snapshots 全部从完整
预期 guard/poison 重建核对，不能靠 after 相等掩盖未执行。

准入重新执行冻结 `checked_admissions`：N48 实际 compile/SO/IR/SASS 与 Root `14c8f26f…`
批准、参考 `7892`/Root `90f6e18e…`/完整 `6f719fe8…` consumer，以及原 codec 全部原门。
完整 cache readback 与三阶段预期 compact 字节相同，CPU 用既有 `LiveEpoch` 对实际观察到的
allocation/role/offset 元数据重放，检查 0/32 producer 切片和 Graph 捕获指针；它不会重建远程
进程内的 token。两 arm 的 reducer 实际 source-IR、TTIR、TTGIR、LLVM、ASM、cubin 都与原件
逐字绑定，实际 metadata 为 11 registers、0 spills、1024 shared bytes。这里 `.source` 是带
Python 行号的 Triton IR，不能冒称是 Python 文件；原完整 Python 模块 hash 与运行前后
`handle.src.fn` 身份通过冻结前端的实际检查绑定。

范围仍是 29 页、最大 context 193、12544B/page、layer offset 128B、N48/S4、两个 Q pitch。
本次没有 writer/encoder、长 context fullstep、42 层容量、服务、Math105 或吞吐资格。
Graph/side 顺序来自冻结控制流及实际保存的 readback，没有额外流 profiler。
候选后续完整步骤和模型测试必须另外保存证据，不能借此记录标为通过。

## 原件与重放

`original-metadata-only.tar.gz` 只含 26 个非 blob 原件及 download manifest，27 个成员。
339,006 字节，SHA256 `c68256cdfa58331bad0f71e1a2e31a157d1dd1f011f6acf45d475aaf0920146e`。
完整 index 在 `metadata-package.json`。每个共享文件小于 2 MiB；没有重复复制大数组。

完整原包 56,749,692 字节，SHA256
`b031e5a3ed8903a190fa2d4ac8d114ab49392b48147714f16335b0d2206370c3`，Root 持久副本
已再次重 hash，位置见 `full-archive-location.json`。其中包括所有压缩完整数组；小包不能独立
执行数值重放。历史 7892/2ad compile 原包已共享，codec 与完整 PR110 consumer 原件继续作为
外部依赖，不复制大型依赖、不通过网络取回、不自动批准任何 receipt。

从仓库根目录做轻量核包（不重新算 FP64）：

```sh
python3 -B docs/evidence/native112-n48-consumer-20261029/verify_shared_evidence.py
```

完整重放使用已有冻结源码和完整原件，输出必须是新文件。若从此 docs child 重放，可将
`--source` 改为该仓库根目录；入口先核对全部冻结工具/内核/reducer 字节，并拒绝不一致。
下面是本次实际命令的同字节共享入口版本：

```sh
python3 -B docs/evidence/native112-n48-consumer-20261029/replay_consumer.py \
  --source /private/tmp/flagos-cjj-native112-consumer-validation-20261029 \
  --raw /private/tmp/cjj-native112-n48-consumer-v1-raw-20261029 \
  --candidate-compile /private/tmp/cjj-native112-n48-compile-v1-raw-20261029 \
  --reference-compile /private/tmp/cjj-native110-compile-v1-raw-20261029 \
  --reference-consumer /private/tmp/cjj-native110-consumer-v1-raw-20261029 \
  --codec /private/tmp/cjj-native-k8-codec-v1-raw-20261028 \
  --out /private/tmp/cjj-native112-new-independent-review.json
```

入口只读 Git 对象和本地文件，导入 NumPy，不导入 Torch/Triton、不加载 SO、不使用 GPU。
Root 可独立复跑；报告不包含本地输出目录或时间，因此相同 CPU 数学结果可逐字比较。
