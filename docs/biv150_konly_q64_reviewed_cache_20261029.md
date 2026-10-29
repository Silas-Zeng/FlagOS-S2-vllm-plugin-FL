# PR116：固定装载原审核缓存，避免重新编译产生另一份 cubin

本提交以冻结的 `5ce042f` 为父，仅修复隔离组件的装载。生产代码、原编译源码、数学、完整缓存检查、数值阈值、Graph-B-first、writer 和 ABBA 时序不变；CPU 检查将整个 frontend 的三处适配逆回后，必须与 5ce 的原文件 SHA `0dbe005b40936548ebfdc30b5db8c4d761663bc677fe5c27728fc373efbbfe9a` 一致。

两次前置失败均未执行 reader/writer/Graph。第二次的最终 Q32 cubin 从 `d101726655eeb9d815ebc748d310051a8fd3c98b299dd24ec09567738807aa2b` 变成 `1100a3a36aa84607645570481016abc33fe36e2fb8e43e2cd327a36abe82fa19`。ELF 仅 `.text` 内 29 字节不同，落在 13 个对齐 8 字节窗口；ELF 资源、非执行段、metadata JSON 与已记录 LLVM/ABI 相同。这是真实可执行变化，继续严格拒绝，未证明新二进制等价或其变化根因。没有数值或吞吐负结论。

## 实际实现依据

已读取的安装源码 SHA：

- spec `compiler/compiler.py`：`a97f067023873acf2e348c28f61494e59170ca84c3ee2314f20e491d3e833dfd`。
- `backends/iluvatar/compiler.py`：`a0d2bbda2f779b528147a238ff6fc558d026478f72d3d12d51d27f48f1aa570a`。
- 实际通用 `runtime/cache.py`：`faf84021196925228e20fdab44b3372c5ac40a57f653c212c0655e8fb97757d0`；spec 对应路径不存在。

实际 cache 的 95–112 行按 group 内绝对路径读取，并静默省略缺失成员。故仅复制目录不足：先以原 80 文件 manifest 验证输入，把两组原七个子文件逐字复制到新的独占 cache，重写唯一的 group 路径索引；要求实际 manager 返回精确完整路径集合，并再次逐文件核 SHA。

两个目录来自原 hash 的 base32，与实际 `get_cache_manager` 268–270 行一致：

- Q32：`L22WXDOCWX6P7HWDIASA6M6M2BI67CB2KVSKBBSL2UKPUYOK3BQA`。
- Q64：`HP4S4Z4JRCDBMMD32C2UGXIMXKLCRIYPE4JORC2MAKHJA23JZMOQ`。

每组七个子文件是 source、TTIR、TTGIR、LLVM、ASM、cubin、metadata JSON。所有字节保持原样；仅索引中的绝对路径改变。原 source 路径和已审核 `ir_override` 路径/内容仍按旧 frontend 严格验证；它们属于原编译身份，和禁止全局 `knobs.compilation.override` 是不同事项。

正常 JIT binder / `_do_compile` 继续创建真实 typed ASTSource；不重建 ABI，不手动调换 kernel bytes。实际 spec compiler 301–312 行在 cache hit 时直接构造 `CompiledKernel(src, metadata_group, hash)` 并返回；324 行才进入 `backend.add_stages`，之后才创建 IR。

每次候选 reader warmup 前验证实际 compiler/backend/cache 模块、cache root、完整 group，拒绝 `always_compile`、全局 override、自定义/远端 cache manager、异步编译和预先存在的 JIT 内存缓存。在这一次 warmup 窗口临时让实际 backend 类的 `add_stages` 抛错，`finally` 恢复原方法。改变 key 或其他 cache miss 必须在任何新后端 stage 前失败。两次 reader 窗口结束后，原 writer 可以按原代码正常编译；不修改 SDK 文件或全进程后续编译政策。

返回值还需通过原 kernel snapshot/full cubin/typed ABI/实际 CompiledKernel/`_init_handles` 资源门。Q64 仍精确要求 237 registers、67584B 动态 shared、runtime spill 0；ELF SGPR spill 485 单列，不放宽旧资源检查，也不把 ELF spill 写成 0。

## CPU 验证与未通过项

`check_biv150_konly_q64_reviewed_cache_cpu.py` 从固定 SHA 的真实 `cache.py` 提取原 FileCacheManager 方法，并逐字执行真实 compiler 284–324 行的连续 cache 前缀。两组命中都取回原完整 cubin；43 个反例包括各子文件缺失/改字节、索引重定向、错误根目录、错误 manager、always_compile/global override、真实 cache miss/错误 hash，以及本次实际改变 cubin 的拒绝。成功和抛错后均核实原 backend 方法恢复。原 screen CPU 检查的 48 个反例继续通过。

CPU 仅检查原源码控制流、文件和模拟调用，不冒充实际设备 cache hit、launcher、数值、Graph 或性能。真实设备仍待 Root 串行运行；原 e4dd reference、63/64 边界、全量输出/缓存存档及所有原门保持不变。原二次失败归档只保留本地，本提交不携带原数组、日志或二进制。

本地重放参数保存在 `/private/tmp/cjj-agent-konly116-reviewed-cache-argv-20261029.json`；它不自动启动设备任务。第一次 CPU stub 少了 `ASTSource.language`，已保留失败 stub/log 后补齐该模拟属性；没有改变真实实现或放宽原门。
