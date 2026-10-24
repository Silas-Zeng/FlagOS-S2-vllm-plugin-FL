# Prefill Attention 的 Q fragment 生命周期

MiniCPM 的 BF16 prefill Attention 在 TTGIR 中先把 Q 放入 shared memory，随后在每个 K tile 的 `scf.for` 内重新 `ttg.local_load`。计算顺序和 HBM 读取不变，但循环会反复读取同一个 Q fragment。

本方法只在已精确核对源码、shape、dtype、编译器和目标平台的 BI-V150 MiniCPM prefill 路径中，把该 Q `ttg.local_load` 移到唯一的 K tile 循环之前并保持 fragment 活跃。变换会执行逐行反变换校验；普通 kernel 先完成真实编译，候选只替换这一条 operand lifetime，不修改模型、KV、调度、采样、服务参数或 benchmark。

## 接入与回退

- 默认由提交代码安装，不依赖环境变量或外部 IR artifact；保留 `VLLM_FL_BIV150_Q_FRAGMENT=0` 仅作排障关闭开关。
- 仅接受 `normalization-over-gqa` 或直接 GQA 的精确已审计绑定；C2 只用于直接 GQA，normalized clone 使用自己的核源码、GQA wrapper、Attention、backend、JIT、compiler 和 MiniCPM shape 校验。
- 候选使用私有 JIT cache 和内容寻址的临时 TTGIR；原 JIT cache、原 options、ABI、constants、attrs、异步编译和 stream capture 行为保持不变。
- 不满足资格、源码变化、IR 形态变化或候选编译失败时返回已编译的 normal kernel，不重编译或重复 launch。

## 实际命中

官方大测日志出现一次 `BIV150_Q_FRAGMENT_SELECTED`，fallback 为 0。选中候选满足：

- Q allocation 唯一，`ttg.local_load` 位于唯一 `scf.for` 前；
- LLVM 中与 Q 源码位置关联的 shared memory read 共 32 次，循环内为 0；
- normal/candidate shared memory 为 `65536/32768` bytes；
- 原始 ABI、constants 和 attrs 在候选编译后保持一致。

## 官方吞吐

比较对象是此前的 `PR15+PR21` 官方结果；两轮测试均跳过首轮，使用第 2–4 轮均值：

| 负载 | PR15+PR21 | +Q fragment | 变化 |
|---|---:|---:|---:|
| 4K | 5118.99 tok/s | 5313.59 tok/s | +3.80% |
| 16K | 3436.17 tok/s | 3764.67 tok/s | +9.56% |

原始证据位于实验仓库：`experiments/official/20261024T061556Z-pr15-pr21-qfragment-official/`。

## 边界

- 仅验证天数 BI-V150；沐曦曦云 C500 未验证，不能外推。
- 两次官方测试使用相同起始 DB 种子 `1907e03…`，但候选 KV 为 `521840`，旧基线为 `521936`。因此文档不把差异包装成严格等 KV 的因果分解，只报告实际绝对结果和保守边界。
- 完整 Math500 Level 3 为 `100/105 = 95.24%`，105 请求成功、0 transport error、0 空最终答案；2 题达到输出上限（index 83、101）。该结果满足 `>=100/105` 门，但单次随机评测不解释为精度提升或稳定等价证明。
