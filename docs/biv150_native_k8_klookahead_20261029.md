# Native K8：V 完成之后的一页 packed K lookahead

当前 Native8b86 每页完成 PV 后，下一页才执行 K global load / shared scatter / ready barrier。本私有候选只尝试把下一完整页的 packed K global read 放在当前 V 已完成之后、PV之前，把依赖的 shared store 放在当前 PV之后、原页尾 barrier之前。默认不接生产，仅提供 CPU 合同与显式 compile-only；没有新数值框架、自动队列或设备资格。

固定 base `a455a3de364051fbea80226870c239de3a48e716`，原设备源 `a1eb7deee5e8ef99136128984256762096a3657f` / SO `8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5`。此次普通 fetch 尝试因 `Could not resolve host: github.com` 失败，未称成功；a455 新鲜度采用 Root 已完成的 fetch/明确指定基线。没有官方升级。原工具、原 CUDA/C++/codec 与生产文件保持。

## 唯一变体与所有权

- 仅 N64/S2、N32/S4，Q pitch2048/2560，PAGE16、D128、2 KVhead、8Q；原16KiB K-only format、width16 scratch/host ABI不动。新库/operator namespace `cjj_native_k8_klookahead_probe`，不能被旧服务自动加载。
- K0原global/store/barrier作为完整页loop prologue。每次当前 QK/softmax/P打包、原V wait8/tail-clear/ready-barrier后，才允许读取下一完整页的8个uint32/线程；没有I8展开、Kpair、6KiB shared或V8组合。
- 当前原BF16 PV之后，才把同8个packed word按原 `k_i8_index` 写到 `[0,2048)`。原页尾barrier发布这些依赖的global→shared写，下一QK在该barrier之后。V始终在 `[4096,8192)`，shared8192B不增；待用K不占新的shared bank。
- lookahead仅 `position+16 < full_upper`，不预读tail/下个segment。最后完整页无prefetch；原tail调用逐字保留，并使用默认 `KReady=false` 执行原masked K装载/ready-barrier。空段原early-return不动。
- 当前页数学、FP32 FMA顺序、Q/K codec/RNE、Kscale乘法、BF16 P舍入、PV、M/L/log单位和reducer不变。factory逆变换恢复实际8b源全字节，C++/codec仅namespace重命名。

这比初始离线研究的“softmax/PV重叠”更保守：**只尝试PV重叠**。预计有8个新增packed寄存器/线程跨PV存活，实际寄存器分配、等待与收益未知；不能借8b资格，也不能从减少loop内K-ready barrier推速度。旧Triton stages2已测负，原Kpair完整步骤无明确收益，均不作为正证。

## wait8 边界：实际 SO 未证即停止

本地没有足以证明“新增global未完成时wait8仍必定等完旧V SME”的CoreX正式语义，不猜counter编码。因此源码把所有新global read放在**原V wait8与V-ready barrier之后**，让旧等待窗口不新增操作。下次V SME issue之前，原页尾barrier必须已完成依赖Knext global→shared store，这与原K global/store/barrier的发布方式相同。

上述是待验证的条件论证，CPU源码顺序不是实际机器码保证。目标编译必须核最终SO的S2/S4完整页、最后页和原tail：

1. 当前V SME issue、原wait8/ready-barrier、Knext global read、当前PV、Knext shared store、end-barrier、下一QK依此在真实CFG成立；不能按反汇编文本行号排序判断。
2. 新K读取不能被编译器前移进原V未完成窗口。若前移，需要额外正式CoreXcounter语义/完整等待证明；本版没有该许可，应保持false停止，不用小数值或旧8b结果代替。不能加假依赖、恒零算式或伪memory屏障。
3. 每个packed read依赖到相同原K地址/字节与shared image，store完成由end-barrier支配下一QK；当前QK所有shared读者已经结束。原V/PV、tail清零与页间overwrite支配关系保留，无多余/越界/重复read、无新spill。
4. 导出并把实际LLVM/SASS指令字与最终SO/内嵌ELF绑定；独立`--cuda-device-only`文本的flags与extension不同，不能替代最终SO。记录实际资源、barrier和等待指令，不用低regs授资格。

编译只收证，不授数值许可。`assembly-review-template.json`所有审核布尔值初始false，S2/S4资源unknown；没有消费者会因为模板被手改就自动运行。若不能证明，保守停在compile-only。没有新增运行中ignore/降门开关。

## CPU 复跑

```sh
python3 -B -S tools/check_biv150_native_k8_klookahead_cpu.py
python3 -B -S tools/compile_biv150_native_k8_klookahead.py
```

只用stdlib：3072个word地址检查（含int32-max页、两head）、438个segment case/120个empty segment，覆盖0..65、127/128/129/193/257/4K/16K，证明逻辑页读取一次与stage所有权模型；12个边界/算术/等待反例拒绝。它不是CoreX内存模型模拟器或设备数值。报告在 `docs/evidence/native-k8-klookahead-20261029/`。Python3.14本地执行，`-S`不加载site-packages；默认无Torch/Triton/NumPy。

## 仅供 Root 后续安排的 compile-only CLI

当前没有执行以下命令。Root需先核旧owner终态并串行，输出用新绝对目录，源码必须clean冻结：

```sh
python3 -B tools/compile_biv150_native_k8_klookahead.py \
  --compile-only --root-owned-exclusive \
  --compiler /usr/local/corex/bin/clang++ \
  --objdump /usr/local/corex/bin/ixobjdump \
  --out /ABS/NEW_KLOOKAHEAD_COMPILE
```

该显式分支才导入Torch并编译/注册新op，不调用attention/codec/reducer，不分配输入、不初始化CUDA。保存源/SHA、compiler/flags、standalone LLVM/ASM、实际Ninja/object/SO与SHA，以及对**最终SO**运行 `ixobjdump --IR/--sass` 的全部stdout/stderr/returncode；失败也保留。CUDA初始化前后若变true则拒绝完成。`frontend_explicit_kernel_launches=0`仅描述本前端，非无编译工作。

若未来静态门真正成立，Root可再用既有8b consumer/fullstep做明确新身份适配：先一个N64/S2当前main对照、所有OML/final位一致、原FP64/guard/readonly/Graph-B-before-eager/side门，再一次完整writer+producer+reducer ABBA。此commit不实现、不运行数值/计时。没有模型、服务、容量或吞吐收益声明。

按本轮“不要上传”要求，本分支仅本地冻结；Draft正文已备于 `docs/draft_native_k8_klookahead_20261029.md`，未创建外部PR。
