# stage2 Q读提前与共享内存生命周期探针

独立实验基于团队main `241220723a09061c182764f8f738e5ec01190367`。开始前git fetch尝试超时；GitHub实时main API确认仍为2412207，与本地已有精确对象一致。生产vllm_fl/csrc无改动，不修改当前PR63冻结受测源码。诊断协议复用PR63已共享91ee316的工具，再增加候选专属IR变换及当前默认基线适配，不依赖其未合生产代码。

## 真实依据与待证假设

PR63 stage2组件在两档慢约15–19%，实际shared81920B；main Q32/stage1为65536B。捕获的原Q-hoist override（不是本方案输出）具有：Qglobal load254、Q local_alloc255，K mutable buffer alloc289、K async-copy332/commit333、Vglobal load340，Q local_load341、主loop342。原变换只把Q读放到loop前，Q64KiB和K16KiB的逻辑生命周期仍重叠。

本探针将**同一条Q local_load**再提前到Qalloc后、K预取之前；希望共享内存颜色分配可复用这64KiB。提前延长Q寄存器生命周期也可能增压或使编译器重新下沉，因此没有宣称共享峰值已下降、编译已通过或有性能收益。

## 当前实际完成

- 固定原件input SHA5210527f9d84f0c6e0a0152d57a3dbce910fe6c6944cdbe9b12a4a967f0c77bd，来源archive7caecdc...中的真实stage2 override；原件和provenance已保留。输出提案SHAa215cd327d65db0f777ecf08ef7c140e4a4750dddb9478be884ddb65c09b0483可用工具重建。
- 只移动一行，生产者前缀及其他所有行逐字不变、精确逆变换；Q descriptor仅有已知alloc/read，唯一SSA依赖已先定义，同region、无跨显式同步/未知store/control flow，识别到独立K预取buffer。
- 15项变异拒绝覆盖重复alloc/read、descriptor额外使用、barrier/async-wait、store/control flow、新SSA依赖、错误dot operand/producer/buffer、错误tile、非法UTF8、重复提前和变更manager源码。重命名SSA也通过逆变换。
- 候选manager仅克隆固定SHA38e44e6...源码的私有globals，先调用原严格Q-hoist、再执行该变换；原baseline全局transform不变。工厂布线用标明的**synthetic normal fixture**检查，它不是捕获的normal TTGIR。未import Torch、未运行目标编译器或GPU。
- 组件工具默认`--baseline current`，支持现有Q32及若PR63合入后的K128，tap放实际内层manager，拒绝当前K128安装失败或依赖缺失时静默削弱baseline。四种模拟路由及两类拒绝的CPU契约通过；目标真实默认安装仍须实测。

CPU proof与故障结果在`docs/evidence/prefill-early-q-20261027/`。这是文本/SSA准备，不是完整内存可见性、编译lowering或数值证明。

## 后续GPU门槛与协作

**当前没有GPU队列、未上传个人实例，也不会在PR63完整配对中做目标编译或测试。** 待其真实结束和GPU/9031空闲，再fresh同步团队main，核对实际导入和真实默认政策；若main已变，先集成再测。不能重复用旧BM128或旧Q32冒充新K128对照。

第一项仅BM256/w16/K64/stage2/early-Q：

```bash
python tools/bench_biv150_prefill_bm64_qhoist.py --run-gpu \
  --baseline current --bm 256 --warps 16 --kv-tile 64 --stages 2 \
  --early-q --case chunk4k --case chunk16k --out NEW_UNIQUE_PATH
```

必须先看实际lowered Q位置、shared、regs/spill与driver capacity，保留原件。不能因源码早移就宣称shared64KiB；不能用寄存器升压/新spill换名义共享内存下降。原数值/FP64参考、非默认stream和changed-input Graph门槛不放松，随后才交错计时及必要边界覆盖。没有机制或组件收益则记录关闭；明确有价值才设计生产选择并重做完整精度/TTFT/固定CLI配对。没有新的服务参数或benchmark调优。
