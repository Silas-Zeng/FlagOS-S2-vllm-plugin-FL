# PR128 M2048 Native INT8 gate：受限数值入口

从 `fbc6858bdccf7688936f9d0887b65e855762ab83` 独立建分支并普通 merge
最新团队 main `5f0086be50f1d6d115c1ebb25b9d3c7250470372`。生产 `vllm_fl/`
和 `csrc/` 与该 main 逐字一致；候选 CUDA/CPP 与原 fbc 编译输入逐字一致。
本提交只有工具和本文，不注册生产路由、不安装服务、不计时。

实际已编译库固定为 `8eefb1b8012b62e6849eeaf5fc43f9600746a39f19354532255ba899d1c9e592`。
它仍归属于 fbc 编译，而非新工具 HEAD。Root 已独立重放实际静态审查，
本后继绑定报告 `6939c47aa9e572ccc8cc4206cb703b3a64ad8a9daffef7d00dc1eb242a779fc9`
与仅准受限数值的批准 `0419ec33452ba1272e43cab98b03ed1fd2162610d3d3d1563a48ebb8db3f410a`。
数值/模型/性能/服务仍未通过，串行设备安排仍须 Root 显式授权；
不能传任意命令行 hash 放行，缺失或错误收据在导入 Torch 前拒绝。

## 参考与范围

只做 `M2048 / K2048 / N12288`，完整 BF16 输出 `[2048,6144]`。参考为已
接受 `5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe`
库的 `biv150_native.gate_silu_cached_probe(x,wq,cols,marker,256)`：内部仍是
原 `_int_mm`、先 `row_scale*col_scale`、独立 gate/up 投影 BF16 舍入，及原
FP32 SiLU×up 和最终 BF16 舍入。没有 Python 重写 GEMM/SiLU 参考。

候选 AQ/SA 直接取同库 `quantize_rows_probe(x,256)`，首个输入再与原 0 配置
量化器完整逐位核对。固定 BF16 synthetic 权重使用 main 中原
`quantize_weight` 函数的精确 AST 源执行；没有改量化规则。所有 epoch 固定
同一 wq/cols/marker 地址及值，只改变 x/AQ/SA，避免夹带厂商权重 packing
缓存失效语义。三个确定 activation epoch 含零行、小幅值和有限大幅值。
不宣称真实34层模型权重覆盖。

五阶段为 eager A、Graph A、**换值 Graph B 先于 fresh eager B**、fresh eager B、
独立 side stream C。两臂实际 Graph 独立捕获；B 保持全部输入/输出地址与
布局；side C 用 wait-stream/event 完成交接。各阶段完整输出以原 gate_auto
route 的 `atol=rtol=0` 和 BF16 位比较为硬门，同时记录完整误差统计。发生
差异时先保存完整输出/输入/摘要再停止，不能把相近误差当成通过。

所有阶段检查完整输入只读、固定权重、NaN 输出 poison 被覆盖和两端 guard。
保存固定原 BF16 权重、WQ/scales、三组实际 activation/AQ/SA 及五阶段两臂完整
输出 backing；相同字节内容按 SHA 去重。它们足够后验逐位比较，完整原件只
留本地/个人机，不外发 GitHub。Graph/side 数值通过也不代表模型、性能或服务
验收。

## CPU 与后续运行

```sh
python3 -B -S -W error tools/check_biv150_native_i8_gate2048_numeric_cpu.py
python3 -B -S -W error tools/probe_biv150_native_i8_gate2048_numeric.py
```

CPU 重用原完整 panel/fragment 检查，仅将其本次内存中的 host 基线由899换成
已经正常合入的5f；原 checker/数学/编译代码没有改。另检查真实参考调用、
固定权重、Graph 顺序以及未知收据反例、两份真实收据及真实 CLI 的排他 tripwire。CPU通过不是设备资格。

待 Root 安排独占窗口后，先运行以下命令加
`--preflight-only`（将 `--run-numeric` 替换掉），再由 Root 显式启动：

```sh
python3 -B tools/probe_biv150_native_i8_gate2048_numeric.py --run-numeric \
  --root-owned-exclusive --expected-source-commit EXACT_NEW_FRONTEND_HEAD \
  --compile-summary ORIGINAL_FBC_COMPILE_OUT/compile-summary.json \
  --library ORIGINAL_FBC_COMPILE_OUT/build/cjj_native_i8_gate2048_probe.so \
  --reference-library EXACT_5CBB_NATIVE_LIBRARY.so \
  --static-review ROOT_REVIEWED_ACTUAL_STATIC_REPORT.json \
  --root-admission ROOT_NUMERIC_ONLY_APPROVAL.json --out UNUSED_ABSOLUTE_DIR
```

原完整 compile-out 的工件必须齐全且 SHA 匹配；不重新编译、不更新依赖、不修改
原编译路径或 site-packages。上述是待填的协议模板；静态审查只允许这项隔离数值，当前串行调度仍 pending。
CPU 可附 `--static-review REAL_REPORT.json --root-admission REAL_APPROVAL.json`
验证两份真实文件；该检查不加载库、不执行设备。
