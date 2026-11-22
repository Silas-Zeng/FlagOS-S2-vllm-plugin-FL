# PR108 compact fullstep 独立 CPU 复核

受测源码 `fa92060af9399e60938ddee6dd7e0a603610cd8c`；compact reader SO `40f9b17c…`，实际 writer 为dc5的 `encode_append`。只读原件 `/private/tmp/cjj-native-k8-compact-fullstep-v1-raw-20261028` 的181文件 / 145,937,040B 全SHA通过；冻结WT保持干净，无GPU、SSH、PR或生产修改。

复用原3eb完整后验数值检查，作明确的布局/源码身份适配。既有compact consumer真实138原件被重新完整审查，四case/176数组硬门、96跨arm完整位比较、32个小context BF16 final proxy检查通过。这是40f9实际资格，没有借旧22e或8b86资格。

fullstep两档36个完整buffer快照、144个O/M/L/output硬门、首尾guard/未用segment poison、Graph B和side C先于fresh eager以及同arm位对照均过。原逐元素 `.005+.005*abs(ref)` 和RRMSE `.005`门不变；partial沿用逐页BF16 P实现参考，final分别对原BF16/量化QK理想FP64。ideal partial诊断两档仍各12/12 false，保留原值。

缓存参考重新核对：每页Kcode `[0,4096)`，FP32 scale `[4096,4224)`，reserved `[4224,4352)`，BF16 V `[4352,12544)`；全phase的初始/append canonical oracle均按此映射生成并核对所有chunk digest。完整冷counter/编码oracle、phase0两arm全行FP64也已重建；其余phase的全部保存FP64数组已重算GPU误差门。没有声称重建了phase1/2 FP64 golden。

实际dc5 writer/views保存源码逐字等于冻结Git对象；初始编码和逐步append均绑定该writer源与实际TTIR/TTGIR/LLVM/ASM/cubin，签名6-pointer/6-I64、9 regs/0 spill。V consumer偏移4352，writer三个参数均为各自typed full-page0 base，避免二次加offset。实际scratch维持width16。所有初始线性K/V/identity-slot的整buffer chunk、guard/readonly账本重算通过；K/V线性打包、slots构造、上传、初始full encode、JIT成本均独立保留，未移入或抹掉成本。

| 固定组件 | BF16 aace ms/step | compact40f9 ms/step | 事件BF16/compact | host wall比 |
|---|---:|---:|---:|---:|
| N64 / 4K / S2 / Qpitch2048 | .416410223 | .396827564 | 1.049348031 | 1.049392367 |
| N32 / 16K / S4 / Qpitch2560 | .805491537 | .768954337 | 1.047515436 | 1.047502513 |

原ABBA、warm128/replay64/unroll8、34276页/8192列table、newKV pitch256与append+producer+reducer计时正文均保持；单synthetic arena物理字节是BF16 561,577,984 vs compact429,958,144。同页数下的组件分配变化**不是42层容量/allocator回收/模型资格**。旧aace对照也不是最新main，未成对对比当前8b86，不作相对其收益或稳定效应结论。真实服务pitch2560写入尚非该计时覆盖。

清理证据：owner38381/start354308715同start Z；两case源码删除分配并调用empty_cache后才到completed/errors[]。归档没有额外postrun GPUidle/其他进程观测，故未补造全机清理结论。完整GPU cache/input正文未保存；本次重建的是CPU预期chunk并对照原运行时全字节/readonly断言账本，不是新的GPU内存读回。保存的完整输出guard/poison则已独立逐字节审查。归档含的pyc仅校验传输SHA，未执行。

完整报告：`/private/tmp/cjj-native-k8-compact-fullstep-independent-review-v1-20261028.json`

SHA256：`96262cd2414ce85a3648f0d45e1727f93b243e754310e31e1160fc30639ebdb0`

复跑使用Python3.12和全新输出文件：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error \
 /private/tmp/check_cjj_native_k8_compact_fullstep_actual_20261028.py \
 --out /private/tmp/cjj-native-k8-compact-fullstep-independent-review-v2-20261028.json \
 --regenerate
```

计时字段随本机CPU复跑变化；原件保持只读。没有新增GPU数值框架、阈值或验收门。
