# PR123：默认关闭的 multi2 Q 操作数驻留接缝

本提交从 `ce905346a0bd6938cd8478cde34c1bcc52eaeed7` 追加，包含团队 main
`899ce77470b0b539aebc108fd122f668c2d5894d`。不改任何 GPU Python/CUDA 源、
codec、writer、reducer、调度、CLI、已有 Native5898/mixedebbc/C3 身份。

`VLLM_FL_BIV150_KONLY_SERVICE_MULTI2_HOIST` 默认 `0`。只有显式 `1` 才在
encoded service 创建独立 multi2 Adapter；原 q32 实例、schema 和资格保留。
共享 proposal 的私有 `_multi2=False` 为严格 bool；True 仅选择既有实验的固定
multi2 schema/profile、num_seqs=2 和 grid+2。共享 Adapter 仅把 profile、binding
schema 和日志前缀提为类常量，新 27 行模块复用原实现；未复制编译、缓存或审查框架。

实际入口仍是 `biv150_konly_encoded_multi2.py::kernel_cjj_konly_multi2`，由
`encoded_service.install` 正常导入。`encoded_runtime.KernelFacade` 原 n=2、
Q>=1024、非 capture 的路由、BM256/BQ32/K64/w16/s1/grid+2 全部不变。
新增的 prepare 与 after_launch 使用服务实际 JIT/CompiledKernel/ABI；只有一次 reader
launch，prepare 失败保留同一私有 unhoisted reader，launch 后失败禁止重试。
原 capture 回退、强引用缓存所有权/生命周期/别名检查不变。

开关关闭时原 Q32 默认 AST、构造、实际 proposal 输出和 runtime stats/log 输出保持相同。
新 helper 字节进入 provenance/encoded implementation digest；新开关进入编译语义键。
因此**内容身份改变，旧 source/控制器 receipt 不能自动认证新提交**；默认行为等价不等于
继承旧整机资格。旧 private compile/numeric 工具读取当前 helper 的固定 SHA，因 helper
已修改会在此新提交拒绝；旧 f667/d39/ce905 冻结工作区和原件完整保留，不放宽它们的门。

CPU 检查（不导入 Torch/Triton）：

```sh
python3 -B -S -W error tools/check_biv150_konly_multi2_service_cpu.py \
  --compile-raw /private/tmp/cjj-native123-multi2-hoist-compile-v1-raw-20261029 \
  --out /private/tmp/native123-service-cpu.json
python3 -B -S -W error tools/check_biv150_konly_encoded_hoist_cpu.py
```

第一项可省略 `--compile-raw`，只做可移植的 host/default 检查。指定原件时核对其固定
44 文件 manifest，并证明新 multi2 proposal 在真实旧 TTGIR/metadata 上与七处字面
适配版本返回逐字相同 override 和完整 proof。Mock binder 检查仅为 host 协议，绝非
实际 multi2 编译/数值凭据。另覆盖严格 bool/num_seqs/grid 拒绝、捕获绕过、独立实例、
单次调用及 launch 后不可重试；Registry/原路由 AST 完全相同。

原 `check_biv150_konly_encoded_hoist_cpu.py` 在父提交及本提交的完整输出逐字相同。
较老 `check_biv150_konly_encoded_cpu.py` 在两者均于第178行旧 worker-tail import
列表断言失败；它还带有更早整份 production patch 的固定比较。本提交不把该历史检查
改成通过，也不据此否定已经通过的当前定向检查。

下一步按顺序验证，当前全部 pending：

1. 用正常生产模块的实际 multi2 fn、新 Adapter.prepare 和现有 fixture 做独立
   compile-only；只 warmup，不 reader/writer。原 `_preserve` 保存两臂完整
   TTIR/TTGIR/LLVM/ASM/cubin、ABI、metadata、handle resources 及精确 cache groups。
   新 compile 前端须从冻结 ce905 helper/fixture 取输入，不修改旧当前文件 SHA 门。
2. 新两臂 TTIR/typed ABI 相同，TTGIR 仅一处不可变 Q local_load 位置变化；审查新实际
   native operand lifetime/资源。新 cubin SHA 未产生，不能填旧 b2fc 或借旧 cache 身份。
3. 用原 d39/ce905 完整 fixture/reference/数组门对这组新缓存做严格 cache-hit 数值检查，
   拒绝 fresh 编译；保留 eager A、换值 Graph B 先于 fresh B、side C、全 cache/guards。
   不更改任何容差或原 BF16 proxy diagnostic 边界。
4. 再运行实验 service smoke、官方105/supplemental、固定参数充分预热 A/B。收集实际
   multi2 postlaunch binding、Q/cache/table/pitch；静默 fallback 只算未覆盖。host 调用
   计数不是 Graph replay 次数或 GPU 时间。

此前 ce905 reader 组件全数组复核 SHA `4af8992dfac2670a394f3319a0393a1f0e2bf6e29f7ee3a5350368cea14c12f0`：
两序列 totalQ2048/pitch2048/773个16KiB页，event 2.134540796→1.465148449ms，1.456876809×。
两臂原 BF16 proxy 失败保留。该结果不含 writer，不是本提交的新执行身份/模型精度/服务收益。
