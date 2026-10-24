# BI-V150 小 batch BF16 GEMM 与 FP32 Split-K 组件实验

状态：Draft，首轮全部停在JIT编译，尚无设备数值或性能结果；修正后待重验，未接入真实服务。

首轮ff188ca生成640条记录，但0条成功编译、0条数值通过、0条性能结果，16条tail也均为CompilationError。
旧completed仅表示记录矩阵写完，不能当设备测试通过，也不能判定所有结构都慢。
完整原始结果按gzip/base64无损归档，附原文SHA，见`evidence/bf16-small-splitk-20261024/initial-compile-failure/`。
原异常只保存repr，未保留内层原因，现阶段不能断言编译根因。

下一版将chunk和循环界改为纯constexpr整数运算、显式dot精度/输出类型，保持同一数学切分；
CPU同时执行kernel实际chunk表达式核对42种分区。新增异常链与完整traceback，
首个tail几何所有配置均失败即停止，不再重复生成640条同类编译错误。
未通过tail的配置在主矩阵明确跳过，记录不冒充实际测试；兼容修改是否修复编译仍待设备验证。

## 问题与独立性

清单 A01/A02/A03/A04、A05、G01 尚未覆盖的方向是原始 BF16 矩阵乘法。
此前 W8A16、W8A8 head 和 gate 双路原型的负结果不能替代 QKV、O、down、
边界层 gate、BF16 head 的检查。本分支基于团队 main
`31a0a7f9205d90904afac0ba19e9e65856c79867`，已包含 PR34 的 Metax 更新；
没有合入被赛题禁止的官方最新框架分支。`vllm_fl/` 和 `csrc/` 保持团队基线。

本次并不证明实际服务每个形状都命中了 FlagGems mm；组件基准会核验直接调用
`flag_gems.mm` 的真实定义文件及其 SHA，而实际 Llama 路径覆盖、模型收益仍待后续验证。

## 实现

- 保持权重 `[N,K]` BF16 原布局，通过 stride 读取，不增加常驻权重副本。
- 直接路径使用 BM16/32、BN128/256；Split-K2/4 将 partial 保存为 FP32，
  第二阶段固定归约后只转换一次 BF16。没有 BF16 atomic add。
- 八个配置比较完整调用：分配、producer、FP32 workspace 和 reducer 全部计时，
  不把 producer 单独加速算作整个 GEMM 加速。
- 当前 Iluvatar `mm.py` SHA 为
  `4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090`；
  `tune_configs.yaml` 为
  `dd1c7432a84b72dde9fe7d48527f4b1744383fa188a6feb69adb1d8fe5748bc0`。
  只核验这两个文件与本地 v5.3.5 副本相同，不声称完整安装包已逐文件核对。
  当前 YAML 仅配置 split_k=1，通用内核的 split>1 是转换到输出 dtype 后 atomic，
  因此不是本实验两阶段 FP32 确定性归约。

## 测试矩阵与验收

M=1/8/16/31/32/40/63/64，pitch=K/K+32；不改 prefill2048 的服务路径。
五个几何为 QKV(N2560,K2048)、O(2048,2048)、down(2048,6144)、
边界 BF16 gate(12288,2048)、LM head(130560,2048)。总计 640 个组件配置记录。
当前使用固定随机种子的合成权重，不是真实模型权重测试，不能替代模型精度验收。

每条记录检查 zero/tiny/normal/large/outlier，完整输出与当前 FlagGems mm 按 .005/.005 比较，
另用 CPU FP64 sampled 点积校验；CPU 转换顺序固定 `.cpu().double()`。
Split workspace 预填 NaN，验证有效输出全部写入。另测非整除 K/N 的两个 tail 几何，
并在 Graph replay 间改变输入，避免只验证同一份 capture 输入。
初版 tail 检查 16 条，设备结果尚未取得。
数值失败保存最多 16 个违反位置对应的输入/权重切片、输出和 SHA，避免为每条失败
复制整份大词表权重；CPU 检查同时确认 kernel store 只指向 output/scratch。

每配置三轮交替顺序的充分预热 Graph 整调用计时。`completed` 仅代表探索矩阵完成，
不代表所有变体通过；每条保留 passed、失败阶段、资源、原始三轮时长。
组件候选必须正确、通过对应 tail 检查、三轮无倒退且中位收益至少 3%。
仍需后续真实模型路由、Level3≥.95、4K/16K 吞吐及 TTFT 验收才能合并服务改动。
热 Graph 组件结果不是冷 cache 或真实服务加速证据。

记录每配置 FP32 工作区字节：例如 M64/head/Split4 为 127.5 MiB，
即使 GEMM 变快，也不能据此忽略服务 KV 容量影响。当前没有安装任何服务 hook，
没有更改正式 serve/benchmark 参数、site-packages 或共享依赖。

## 当前证据

`docs/evidence/bf16-small-splitk-20261024/cpu-check.json`：42 个真实切分函数覆盖用例
无遗漏或重叠，320 个几何/配置指针范围检查通过，无 atomic，生产源码未改。
这些均为 CPU 检查，不能替代设备数值、Graph 或模型验收。

在个人实例的已有串行队列结束后才允许启动；队列核对前驱 PID/start_ticks，
并在 9031 端口及 GPU 空闲后再次校验，不停止任何已有进程。
