# Native K8 packed-V8：仅 N64/S2 编译筛选

基于团队 main `899ce77470b0b539aebc108fd122f668c2d5894d` 的已验收 5898 K-lookahead reader，新增独立 namespace `cjj_native_k8_packed_v8_probe`。生产 `vllm_fl/`、`csrc/`、服务参数、调度、writer、reducer 均无改动；没有生产开关/安装路径。只有一个 N64/S2 实例，Q pitch2048/2560沿原声明，所有设备资格均未验证。

假设是：在当前 V issue 处读取每线程8个 uint32 packed words，尝试保持 packed 至 P 的 BF16 打包之后，随后逐小组反量化，并将相邻 token 的同一维合成一个 b32 shared store。保留现有 PV shared 坐标、Q/K codec、QK、softmax、P 舍入、rho/FMA、PV 和 scratch 输出。编译器是否兑现晚解包、寄存器是否恶化，由最终 SO 决定。

## 与已关闭方案的具体差别

| 旧方案 | 已有组件结论 | 本候选的差别及仍有的成本 |
| --- | --- | --- |
| PR73 `4a1f7ddd` | 4K 0.722609×，16K 0.682247×，ragged失败 | 原式为 BF16(P×scale)×BF16(code)；这里是 BF16(P)×BF16(F32(code×scale))，不是同一舍入数学。利用 Native 原 V shared 坐标显式 paired b32 store；旧方案也已有 packed load，不宣称首次使用 packed。 |
| PR75 `ac6f7c23` | 4K 0.622819×；实际只兑现 after-QK，未兑现完整 softmax 后 | 本次以最终 SO 的 P packing→解包/写入为停止条件；仅源码摆后不算通过。 |
| PR78 `a6980ddf`/`11ed97dc` | 相对原 BF16 0.656564×，相对旧V8 0.908698× | 不新增 byte shared stage、b8 store/reload 或 K/V shared 复用。仍增加 signed-byte 重建、F32乘法和 BF16 舍入，不能因寄存器少就授收益。 |

这里替换原四条 BF16 V-SME issue，**不保留该异步搬运能力**。单 head/page 的有用 V 读为2048B code+64B scale，原为4096B BF16；新 shared 写仍4096B。每线程逻辑上8个packed load、32次反量化、16个b32 store、8次scale shuffle；V侧 scale 乘法的逻辑元素数是旧PR73 P侧的16倍。上述不是 HBM 流量或吞吐结果。

## 私有数据合同：仍分配16384B/page

| 页内字节 | 内容 |
| --- | --- |
| `[0,4096)` | 原K signed-I8 code，strides `(16384,256,128,1)` |
| `[4096,4224)` | 原K F32 token/head scale；K/Q仍原 RN32/RNE |
| `[4224,8192)` | 原padding，未使用 |
| `[8192,12288)` | 私有V signed-I8 code，指针已加8192一次，strides `(16384,256,128,1)` |
| `[12288,12416)` | 私有V F32 token/head scale，FULL-base索引 `page*4096+3072+slot*2+head` |
| `[12416,16384)` | padding，未使用 |

C++ 仅接受上述 int8 V角色，拒绝原 BF16 V/compact dtype和pitch；与 K、FULL F32 view 同 backing 的原别名/范围门保留。私有格式不能交给现有 BF16 V 服务 reader/fallback。没有释放页空间或恢复容量。

本次**没有 V encoder/writer**。编译不分配、编码或读取任何 fixture。未来隔离数值输入必须由独立拥有者提供有限正F32 scale和有效code，并核整个私有arena；可先复用 PR73 已保存的实际 code/scale 字节重新映射，不能自行把当前 K 的 RN32/RNE 当成旧 V codec。旧 V writer 数学为 per-token/head maxabs/127、zero→1、min-normal clamp、half-away/clamp到[-127,127]，其 reciprocal 降低/边界误差仍保留。reader 的 signed-byte primitive表达全部256位型，不据此宣称 writer允许-128。

新消费式为 `Vhat=BF16_RNE(F32(F32(code)*scale))`；P仍 `BF16_RNE(p)`。这是量化近似，不等于原BF16 V，也不能借PR73/现有main精度。之后最短数值对照应为原5898读独立重建的BF16 shadow Vhat，对本reader读相同code/scale，核全O/M/L及原参考；再单列相对原V精度，不复制新数学框架。本提交不实现这一步。

## 同步：CPU可证明部分与明确缺口

K0的普通load→shared store→原barrier、Knext在当前V-ready barrier之后读取、PV后发布Knext→原页尾barrier，源文逐字保留。shared仍8192B，K在前区，V在`[4096,8192)`；没有新增barrier或byte staging。

新 V publisher 每个 shared word 只有一个线程拥有，成对写入有效值/零尾。旧 `clear_v_tail` 的 word-owner 不同；其odd-tail RMW若直接接在新store后，不能靠单warp假设证明跨lane顺序。因此仅删除该冗余调用（函数保留），CPU穷尽valid0…16证明原低/高半字清零语义相同，之后沿用原V-ready barrier。

**原 wait8 暂作为待审编译指令保留，绝不当作普通global load完成证明。** 5898在该窗口有V-SME任务且没有新Knext global；本版移除了V-SME、增加packed V普通global，原5898等待准入已不适用。本地已保存的SDK头文件只有声明/入口，没有足以证明该新任务计数的正式语义。正确的数据流意图是 `global→code/scale→BF16→shared store→barrier→PV`；最终 SO 必须展示依赖/CFG，并独立解决 retained wait8 的任务语义与K完成条件。无法证明就停在compile-only，不用旧数值或一次小跑补证明。

## CPU检查与编译计划

```sh
python3 -B -S -W error tools/check_biv150_native_k8_packed_v8_cpu.py
python3 -B -S -W error tools/compile_biv150_native_k8_packed_v8.py
# 仅Root在独占窗口、确认冻结commit后运行：
python3 -B tools/compile_biv150_native_k8_packed_v8.py --compile-only \
  --root-owned-exclusive --expected-source-commit FULL_FROZEN_COMMIT \
  --out /workspace/experiments/cjj-native-k8-packed-v8-scout-compile-v1/out
```

CPU只做main899有限源变换的完整逆证、原K/PV/同步片段检查、完整V地址/word-owner/valid0…16零尾、I64页地址与signed byte/paired bit映射。没有镜像Torch验收、GPU数值或性能计时。编译入口窄适配原Native compiler并对父字节逆证；保存真实最终SO、IR/SASS、编译日志、文件SHA，检查Torch CUDA上下文未初始化；0 reader/writer/codec/reducer launch。不会安装依赖或修改site-packages。

停止条件：实际提前展开大批标量、b16 store/额外byte stage、新同步或spill、Klookahead被破坏、PV坐标/依赖/等待语义不能证明。资源对照是实际5898 S2的85 VGPR/70 SGPR/8192B shared/两类spill0，并非旧8b的100reg。新资源、数值、Graph/side、完整writer成本、模型105和服务吞吐都保持未知；不准备性能或服务入口。
