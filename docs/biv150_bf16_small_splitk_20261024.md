# BI-V150 小 batch BF16 GEMM 与 FP32 Split-K 组件实验

状态：Draft，f5ff432完整设备矩阵收齐，347/640通过原检查、27条组件线索；未接入真实服务。

## 最新完整结果与独立参考问题

运行f5ff4322e4433fe582fd26f2890bdaf8a8bc5420：两个非整除tail几何共16/16通过；
640个主配置均实际运行，347通过原全输出/五模式/CPU采样/changedGraph检查，
27配置满足完整三轮中位>=1.03且每轮不退。QKV/O/down及边界gate有局部线索，
大词表head没有满足该门槛的配置。逐配置正确性与三轮原始时长见`evidence/bf16-small-splitk-20261024/v2-final/`。
这是合成组件，不是实际服务提速；失败配置不能安装。

对一份O/M8/pitch2080正常输入失败夹具（原native vs候选58/16384超差）进行CPU FP64仲裁：
保存的16个原违反位置均候选与CPU结果逐位一致，而native对应16点均超差，maxabs0.109375。
这只证明该保存切片，不能将所有293失败改标通过，也未证明native错误的根因。
已增加独立CPU工具逐项核验全部保存夹具SHA，报告候选/native各自偏差和原失败坐标。
全部293份保存夹具已用CPU FP64仲裁：候选通过/native失败106份、候选失败/native通过33份、两者均通过5份、两者均失败149份。
详见`evidence/bf16-small-splitk-20261024/cpu-fixture-audit/`，仅覆盖保存的行/列切片，不是全输出证明；原矩阵347/640通过数不改。
下一步须新全量对照、更完整CPU oracle或真实权重验证，不放宽.005/.005，也不能用只测赢的形状代替验收。

已准备完整输出oracle探针：实际第4/20/37层的QKV合并、O、down权重（逐项SHA），
M1/8/16、unsplit/Split2/Split4共81条配置；五模式中每个输出元素均用CPU FP64点积转BF16核对。
native与候选分别评分、另保留直接对照；不会因native超差自动将候选原失败改标成功。
候选CPU全过后还做改变输入的Graph与完整CPU对照，并验证计时输入；三轮完整调用时长包含工作区/归约/分配。
这是三个代表层与合成激活的组件，不是全部42层或模型精度；大batch已保留首轮结果，不在此反复扫。
任何准备接入的选择仍须补真实调用覆盖、按所选范围验证及完整Level3/4K16K。
新探针排在PR37模型验收之后，GPU与CPU参考均不与正在跑的正式压测并发。

团队main已前进到0bf9d2e（PR35编译缓存语义身份），本分支用906927f非重写同步；
受测f5ff432冻结目录不改，CPU源码保持检查转为最新main，生产仍无新增hook。
当前6c19e63只加CPU仲裁，不改变受测BF16 kernel。

## 历史编译拒绝

首轮ff188ca的640条记录均CompilationError，0条数值/性能结果，16条tail也失败。
完整原始结果按gzip/base64及原SHA无损保留在`evidence/bf16-small-splitk-20261024/initial-compile-failure/`。
修正constexpr界、显式dot及完整异常链后f5ff432正常编译；没有将首轮错误当性能负结果。

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


## 39d63d5：真实权重、每个输出CPU FP64的81记录终态

全部81条完成；五模式candidate CPU通过41/81，changed-input Graph CPU通过38/81，native CPU五模式通过42/81。
直接native比较通过49/81；这些是各自不同的证据门，不能混同或把native与CPU不一致反向改写为candidate自动通过。
QKV/O/down的M1 split4在layer4/20/37均有完整CPU/Graph与正完整调用时序，代表层中约1.38–1.80x；
layer37的QKV/O M8 split4约2.14x/2.13x且对应完整门过，但不能泛化其他层。
下一步优先补完整42层与实际路由，不能按这3层就安装。部分M8/16候选CPU过而native不过，仍需分清误差来源及模型门。
完整JSON、日志、每个失败fixture的逐文件SHA与可解码原始archive在real-oracle-final/，
原先347/640随机矩阵记录不改写，不宣称模型吞吐已提升。


## 下一轮：对齐真实out基线，随后补42层

已读当前生成图：O/down/QKV使用extern_kernels.mm(...,out=buf)。SDK Inductor的该choice是torch.mm / aten.mm.out；
worker调用FlagGems enable默认策略，FlagGems候选表含mm与mm.out，iluvatar两者共享同一_launch_mm；实际还可能受vendor默认排除/dispatch key影响，不能仅由候选表推断最终注册。
不能只据文件名extern或旧default分配版时间认定服务baseline；新探针先按当前worker相同的默认FlagGems enable策略，用torch.mm(...,out=caller_buffer)测完整三代表层81记录，
记录enable前后真实aten::mm.out dispatch table、mm/mm_out源码身份，保留原returning版81记录不改写。权重N×K与实际out stride不变，使用同storage的as_strided view避免额外转置copy。
直接FlagGems.mm_out和torch.mm的最终注册选择不能只靠默认配置假定；新结果以实际dispatch table为准。

仅当三代表层81记录中至少一个case通过原全输出CPU/Graph/计时输入门、三轮完整调用均有正收益时，
再运行42层×QKV/O/down×M1/8/16×split1/2/4=1134记录。不是只测旧成功层或裁掉失败配置。
继续原五模式、每输出CPU FP64→BF16、native单独误差、NaN scratch、变输入Graph与三轮完整成本计时，门槛不变。
尚无新设备结果或服务hook；此队列只允许接在当前PR40完整服务后，防止并发干扰。

新out参考还启用与正常服务相同的inference_mode，避免Autograd dispatch key差异；returning旧模式默认行为保留。
