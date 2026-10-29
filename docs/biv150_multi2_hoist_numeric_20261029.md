# PR123：独立 multi2 数值筛选，未执行设备

基于已共享 f667，在新 `codex/cjj-multi2-hoist-numeric-20261029` 分支正常合并团队 main `899ce77470b0b539aebc108fd122f668c2d5894d`。`vllm_fl/` 与 `csrc/` 逐字等于 main899；已有 117/118 收益、服务资格、CLI 和路由保持原样。f667/585 冻结树未改。入口默认只输出 CPU plan。

实际编译证据是 f667 的44件原件（3,818,887B）及独立审查 `8737facf61e20be7afcb265e7df113059a5eebc18c4cb536014079668b9e1729`，Root 已实际逐字重放。该原报告的 `Root_admission_granted=false`、数值/服务未验证字段保留；本次新源码和设备启动仍须 Root 审查。寄存器135→122、动态 shared57344→33792、runtime spill0只用于筛选；ELF scalar spill98→96分列，不声称零spill、occupancy或吞吐。

复用已设备执行过的 PR116 `522c0b0617a8d2d862f60ebb03f05f6e08ccfcfe` reader driver、严格 reviewed-cache loader 和执行身份检查；文件逐SHA锁定，并用可逆编辑生成窄入口。删除 writer 和计时段，保留原完整输入/cache/输出/poison/guards、逐位、FP64数值和Graph换值检查。相应原 helper 的 844a 依赖是522c祖先，运行/源码 bundle 必须明确携带522c链，不能假定目标机已有。

两臂都是实际 multi2：原 unhoisted cubin `4991ac0a6279f6d89686a8f2ffe1e50e7129a87d7b8eb7c68f17ec73a3152cd9` 与 hoisted `b2fc60868dd0635ddf40de258996be5d21b5ed47c0215afff86359c7a1f041c8`。七个缓存子文件、metadata、原Python source/typed JIT ABI及真实 CompiledKernel 路径严格绑定。仅重定位缓存索引的路径；原 override 文件路径/字节保持，unhoisted 无 override。两次 warmup 窗口禁止 backend 新编译，未命中就停止，随后恢复原 backend。没有替换 cubin 或修改旧编译产物。

固定 N=2、总Q2048、Q/output pitch2048、773个16KiB页、table pitch8192、BM256/BQ32/K64/w16/s1、grid66×2。

| 阶段 | qlens | klens | 数据与执行 |
|---|---|---|---|
| A | 1024,1024 | 3072,3072 | 原fixture pattern0零Q；全部2048行均匀softmax参考；eager-A后捕获并重放Graph-A |
| B | 31,2017 | 2079,4063 | pattern1随机；换Q/cache/table/lengths/cu_q；相同Graph先消费B，再fresh eager-B |
| C | 2017,31 | 4063,2079 | pattern2再换值并反转请求边界；独立且不同于capture/default的side stream |

B/C末KV页均有15个有效slot，K64尾31；保持原fixture未使用页和43值tail、随机不重叠物理页。三阶段都完整保留原CPU数组/编码字节、所有输入指纹、两臂完整输出前后及整cache。B/C必须有变化，Graph-B必须逐位等于fresh-B；每阶段两臂必须全输出逐位一致。

参考使用 e4dd 原 RN32/RNE、INT64 QK、FP64 scales/closed softmax、BF16 V以及原 `.005+.005*abs(reference)` 点误差与 `.005` BF16 relative RMSE。只把原单序列参考的行号、长度和因果prefix作为参数，分序列运行后按原行顺序拼接；所有量化、矩阵和softmax表达式不变。A复用原FP64累计均值捷径；B/C按原fixture每序列行选择（短31行全测，长序列首/尾/15/16/31/32/中点）。因此完整数组逐位并不等于随机数据每个元素都做FP64参考。原 explicit `--numeric-diagnostic` 只允许记录BF16质量失败，不能越过任何硬门，也不授精度/模型/服务资格。

CPU检查验证来源逆变换、坐标覆盖/拒绝、原数学与最终装载检查AST、实际44原件和review绑定，以及明确标记为fake的handle/损坏cache拒绝。它没有调用Torch、JIT或GPU。

```sh
python3 -B -S tools/check_biv150_konly_multi2_numeric_cpu.py \
  --raw /absolute/actual123-raw \
  --review /absolute/root-replayed-8737.json
python3 -B -S tools/probe_biv150_konly_multi2_numeric.py
python3 -B -S tools/probe_biv150_konly_multi2_numeric.py --cpu-preflight \
  --raw /absolute/actual123-raw --review /absolute/root-replayed-8737.json \
  --references /absolute/e4dd-original-four-reference-files
```

Root后续单独批准才使用 `--gpu --root-owned-exclusive --commit <精确冻结HEAD> --out <新绝对目录>` 和同一组raw/review/references参数。source原路径与override原路径必须仍存在且逐字相同，否则首个reader前失败；不自动复制到其它路径后假称同一JIT绑定。

完整reader性能对照只留计划：数值、Graph-B与side-C全部通过后，复用原116的4-step Graph、8次warm replay、16次timed replay，对相同输入做unhoisted/hoisted/hoisted/unhoisted；同时记录event和wall时间、完整输出/guards/输入/cache后验。所有Q量化、加载/staging、QK、softmax、PV和epilogue必须计费。本入口没有计时执行，亦没有writer/append、其它pitch、长上下文或实际服务资格。
