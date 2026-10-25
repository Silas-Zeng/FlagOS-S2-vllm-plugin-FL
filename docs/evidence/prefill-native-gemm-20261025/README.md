# M2048 native BF16 GEMM experiment — Draft PR44

Base: team main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`, verified again through the GitHub API before service acceptance. No official upstream upgrade. PR43's CPU enqueue correlation finds 99.24% of non-Graph BF16 MM time at M2048. The earlier PR42 negative tests cover M32/M64 and do not test this shape.

## Implementation and scope

The opt-in service hook calls a functional C++ custom-op. At M2048 it invokes the installed Corex ixblasGemmEx with BF16 input/output and FP32 compute; other rows call the existing registered FlagGems mm.out. The 34 middle INT8 gate projections retain their existing route. The hook defaults off and its flag, source and library identities enter compile-cache semantics. An opaque operator avoids a Python M==2048 guard conflicting with vLLM's explicitly dynamic row dimension.

The installed legacy GemmEx ABI uses IX_R_32F=0, not the separate compute-type enum value68. The current Torch BLAS handle is bound to the current stream; pointer mode is saved/restored, math mode unchanged. The out operator allocates and synchronizes nothing; the functional operator owns its output allocation. The actual service must still verify performance, accuracy and memory effects.

## Hot-weight component evidence (fe3abc9)

`v1-component-result.json` records three alternating timing rounds on real weights: QKV1.059618x, O1.243085x, down1.338432x, edge gate/up1.214994x (median baseline/candidate duration). Twenty direct numerical comparisons across four geometries and five inputs are identical, and twelve changed-input Graph checks match eager. Strict CPU FP64 tolerance fails for both implementations with equal error statistics; those failures remain in the evidence. Timing inputs have aggregate CPU metrics only, not a direct equality comparison. This is component evidence, not model accuracy or throughput.

The initial queue stopped before GPU work when the profiler export exceeded its timeout. The resumed queue verified collector exit, Report saved, SQLite quick_check and both profiling windows. Original failed state remains preserved.

## Functional, dynamic-M and real-weight-ring evidence (0118a13)

`functional-v1-result.json` is the original unmodified result. Its probe is saved as `tools/check_biv150_native_bf16_functional.py` (original SHA1ef32fd54e315791a6aecc2bcc3d0f0209fe5c933c641c7b6b05fc720221dad4). Library SHA: cba804a1bdd3f7e514443043f6f11c5ce6864af4a7f79e8244728d7f1d3af9d7. Source and model-weight identities are embedded in the result.

- Four actual projection geometries with closed-over nn.Parameter each compile one graph for M32,64,2048,2030,8; all outputs equal eager.
- All 134 model-order BF16 weights (2,252,341,248 bytes) at M2048 give numerically identical FlagGems, native-out and functional outputs. Changed-input Graph endpoints also match; endpoints are the last writer per reused output width, not a claim to retain all intermediate Graph outputs.
- Three alternating AB/BA/AB rounds, ten complete weight-ring replays each: 64.4617/51.7504ms, 64.4597/51.7390ms, 64.4518/51.7389ms. Median baseline/native ratio1.245712. This is a cold-weight component result, not end-to-end throughput.
- All28 sampled non-default-stream Graph/eager cases match. However, the first eager O-projection M8 input has63/16384 differing elements,60 beyond tolerance, max_abs0.3515625. They occupy row1 columns128..191 (one element is equal). The second input and all later Graph comparisons match. This anomaly is under investigation; do not call all fallback tests passed or assume ordinary rounding.

## Service acceptance

The candidate/control runner and16 CPU fault checks are prepared at c56c4d3. It preserves the fixed competition CLI, full105-question accuracy and three warmups before one scored round per length. A99/105 near miss permits diagnostic throughput only. Actual105-question accuracy, fixed4K/16K throughput/TTFT and matched latest control are still pending. No service gain, precision pass, or merge is claimed.

## M8 O回退问题与窄范围替代

重复AB/BA与调用间同步验证显示，原FlagGems mm.out和functional中的同一回退都会间歇产生约64值块异常，不能再描述为仅首次调用。选中配置BM8/BN64/BK128、SPLIT_K1、16warps/2stages；根本硬件/编译机制尚未确定。CPU按原随机序列重建的63个异常坐标，functional63/63等于FP64舍入BF16、原baseline0/63；原始输入未保存，RNG路径假设明确保留。随后完整输入重复测试独立重现两条原MM路线的误差。

现有native_out在非默认stream的64次eager+16次变化输入Graph输出稳定，CPU容差违规0，Graph/eager相同；热组件计时约16.50us，原mm.out约34.03us。首个三路线probe误将默认stream用于Graph capture，默认stream的Graph阶段失败，不能视为全通过；修复后的独立capture验证待完成。

本次代码将native路径额外限定到M8/N2048/K2048的O投影，其他小行形状仍使用原回退，M2048计算实现不变。新SO、动态编译与完整服务精度/吞吐仍待验收。

修正capture后的v2已完成：default/side各64次eager+16次变化输入Graph，native_out全部finite、CPU容差违规0、Graph/eager相同、每输入仅1种输出。native计时16.442/16.620us；同轮原baseline分别2/4次CPU违规、functional原回退4/10次违规。v2无异常退出，完整原件m8-routes-v2-result.json；工具check_biv150_m8_o_routes.py。原v1失败保持不变。这里证明M8窄替代的组件稳定性，仍不代表全模型精度或吞吐通过。
