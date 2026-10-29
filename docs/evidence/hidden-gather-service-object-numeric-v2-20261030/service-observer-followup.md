# 仅取证修正的服务后继（设备验证待执行）

服务快照的 `launcher_sha256` 现在来自 `PyCFunction_GetFunction → dladdr` 的实际 C 函数 DSO，新增 `launcher_binding` 保存地址、DSO 偏移及原 module 路径标签。只在原两次 smoke 边界读取，不调用 C 函数、CUDA、同步、编译或重置计数。旧模块标签不再作为函数归属；仍原样作为诊断呈现。

新增 `biv150_hidden_gather_sme1_pins.json` 独立绑定 `73470bb`／摘要 `15209402…`／独立审计 `e8d90e57…`。旧 SME5 pins 与 d74、旧服务失败均未修改。新 bridge 分开保留生产数学来源 `a101` 和数值组件来源 `7347`；批准两旗默认 false，模型、服务和吞吐资格仍 false。原 helper、kernel、Plan guard、controller、163 smoke、105、正式 CLI、warmup 与清理字节均未改。

observer 要求实际 C 函数为 H7／5124 的 `launch`、METH_VARARGS、偏移 5216，并核确切 SME1 GPU hash、六件 source/IR/ASM/cubin、ABI、options、resources。C 地址的 ASLR 基址可以变化，地址减基址须吻合符号偏移。旧无 C 指针的 ACK 不获新资格。若冷编译字节不同，原 ACK 仍先保存，随后拒绝，不扩大白名单、不强制 SME、不覆盖历史记录。

CPU 测试明确以实际保存的 C 地址记录替换取证函数边界；不会在本机加载 Linux ELF 或宣称服务 C 指针已被再次观测。它覆盖字段错误拒绝、旧 module 标签变化不影响 C 归属、保留 119 transport、默认关闭和真实 run 到首 idle 前的准入。后续实际 smoke 仍须取得新 worker 的前后 ACK，ordinary/Plan host return 计数仍不等于 GPU 完成。构造 Plan 数值通过不意味着服务 Plan 自然命中。
