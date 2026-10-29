# NativeK8：单次校验复用 backing-storage 描述

> 下文原a455/8b86记录保留为历史。当前兼容修订见末节：正常合入main56，继续固定5898，公开二参API恢复；旧8b86工具不作为新版本资格。

基线为团队 main `a455a3de364051fbea80226870c239de3a48e716`，独立分支 `codex/cjj-native-storage-observation-20261029`。本候选只减少 Python metadata 调用；未运行 GPU、SSH 或服务测试，也没有吞吐结论。与正在验证的117/119源分开，不合入 lookahead、统计或 closure 优化。

在正常 eager/mixed 输入上，`Native.validate` 的9次 `bounds` 和24次 `overlaps` 共调用 `storage(t)` 57次。现在每次 `bounds(t)` 在原位置完成全部检查后返回刚观察的 `(storage_id, base, end)`，`validate` 仅在局部字典保存这些结果，随后24次重叠检查复用它们。所有9次 bounds仍按原顺序执行，即使两个操作数是同一个Tensor，也不跳过任何检查。强引用列表 `tensors` 在校验期间一直持有所有对象，字典不保存到Adapter、Registry或任何跨调用状态；有效调用实测CPU fixture中 storage helper读取为9次。

生产改动只有 `biv150_konly_encoded_native.py` 与 manifest中的adapter SHA：`7af8110d77a84a98c23aa740ab20464cddd86790300a0314f3f39937e8a4bb9f`。原 `storage(t)`、dtype/shape/stride/offset检查、owner/epoch/role观察、完整backing区间别名规则及其检查顺序保留。`try_produce`、verify、attach、统计、KernelFacade/Registry、GPU源、库8b86和服务参数均不改。Native的形状筛选仍在validate之后；不支持N也先完成安全检查，合法时才走原private Triton fallback。launch后的错误仍不重试。

范围是服务输入在这段同步host校验中不被外部并发换storage的既有调用生命周期；校验本身没有主动Tensor数据操作或storage修改。未宣称支持自定义Tensor accessor有副作用，或另一线程在同一次校验内部修改metadata，也没有新增对此类并发的原子性保证。每次调用边界都重新读storage，不能因为一次通过而接受下次变化。

## CPU验证

复用原 `check_biv150_native_k8_service_cpu.py` 的实际8b86 ELF/schema读取、四个shape/stride组合、attach/fence/C3、owned别名拒绝、private TL fallback及postlaunch不重试矩阵。原identity inverse前仅增加本次七处源码变换的精确逆变换；不将本候选再标为identity-only。

新增 `check_biv150_native_storage_observation_cpu.py` 使用同一fixture，加载Git中原a455 adapter做直接对照：32组、每实现64次调用。连续调用之间更换storage、pointer、stride、offset、shape、epoch、owner、role、完整storage别名与同Tensor别名；保留合法新storage/offset/stride的接受行为；同时两个错误时原错误顺序一致。30/31/61/63均覆盖合法fallback与坏role/alias先拒绝。所有返回/异常类型及文本、Native/TL调用数与原实现相同；正常第二次调用继续57→9，无跨调用缓存。这里的fake dispatcher只记录调用，不是GPU数值、Graph数值或性能资格。

另外原 `check_biv150_native_k8_fingerprint_cpu.py` 通过：实际94db helper、完整manifest、新实现digest、8b86内容SHA、standalone C3、ABI/ON/OFF与错误身份拒绝。

```bash
python3 -B -S tools/check_biv150_native_storage_observation_cpu.py --library /absolute/path/to/cjj_native_k8_v_overlap_probe.so
python3 -B -S tools/check_biv150_native_k8_fingerprint_cpu.py --helper /absolute/path/to/actual94db.py --library /absolute/path/to/cjj_native_k8_v_overlap_probe.so
```

本地完整CPU报告只保留在 `/private/tmp`：

- `cjj-agent-native-storage-observation-cpu-20261029.json`：`ed09745428056200a1fcf304bfc5cf235cde3336e6bb55fae59f040ab71e693d`。
- `cjj-agent-native-storage-fingerprint-cpu-20261029.json`：`b8469f55bbc950e317c272d15ee70d6151ff7178f23b35710575bdb61b7f3977`。

没有新增GPU验证框架。后续等119提供当前host调用计数，再由Root用既有服务流程独立验收和A/B；计数不是时间，旧profile频率不能外推当前main。已经捕获的Graph replay不重新执行本Python校验，因此不应把57→9解释为纯decode Graph replay的吞吐收益。冻结119 launcher故意只接受原A/B source，不直接允许本候选；后续接入须明确更新独立source身份，不能冒充a455生产树相等。

## main56 / 5898 与118宿主接口兼容修订

由原候选 `ec70e30cf77963929caa6556350b91eb98a4cf3e` 正常merge已确认main `56f88737fcf3b940760b9ca3027a925cd5a7d77d`，保留main的5898 library、klookahead namespace/operator与GPU源。生产差异相对main仅adapter和其manifest SHA。公开 `overlaps(a,b)` 恢复为main逐字原函数；只在一次 `validate` 内定义读取局部backing的 `observed_overlap(a,b)`，9次bounds保序、强引用和跨调用freshness不变。实际SDK/GPU/服务仍未测试本修订。

新检查器复用原32组双调用fixture，并执行冻结118@698的原始mixed `forward`/runtime/producer-reducer宿主AST及其原CPU fixture。原mixed的二参 `overlaps` 调用正常，N61/pitch2048连续两次调用中，全部Native storage-helper读取由94/91降至46/43；其中一次validate为57→9。计数只证明CPU调用变化，不是耗时或模型收益。旧8b CPU qualification函数未调用；只读取实际5898 ELF字节/SHA，dispatcher是明确标记的CPU recorder。另用实际94db standalone helper验证C3和拒绝8b/未知库身份。

```bash
python3 -B -S tools/check_biv150_native_storage_main56_cpu.py \
  --library /absolute/path/to/cjj_native_k8_klookahead_probe.so \
  --helper /absolute/path/to/actual94db.py
```

需仓库已有main56和698 Git对象；698仅作冻结参考读取，不修改或合入其生产源。新报告覆盖storage/pointer/stride/offset/shape/epoch/owner/role/alias变化、双错误先后次序，以及unsupported shape先安全检查再原TL fallback。删除本次局部观察的精确逆变换后adapter全字节等于main56。纯Graph replay不执行这段Python；mixed成功Native返回早于TL `Runtime.note`，但仍执行本validate。writer/prefill/residual-TL的note是另一独立候选121范围。

本修订本地CPU原报告：`/private/tmp/cjj-pr120-main56-cpu-20261029.json`，SHA `fd4e0c694973b45bdc3c4654dafef4cc225363828e360367d943b6a0538d82e3`；两次重放逐字一致。此前只读路径结论另存 `/private/tmp/cjj-pr120-main56-host-path-review-20261029.md`（私有短报告，不外发）。
