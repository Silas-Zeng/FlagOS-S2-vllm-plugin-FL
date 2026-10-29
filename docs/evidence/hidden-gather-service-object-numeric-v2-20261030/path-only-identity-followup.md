# 冷编译目录变化的限定后继（尚未设备验证）

c41 的完整原始 SHA 门在上线前发现跨 checkout 的确定性问题：实际 source/IR/ASM 和 ELF 的 debug comp_dir 带原目录。c41、其 CPU 报告和旧失败原件保留；本后继仅修改取证比较，不改 helper、kernel、Plan guard、控制器、正式参数和数学门。

`biv150_hidden_gather_path_identity.py` 固定真实 4a9f 的 12 section、5 program header、原始对齐及零 padding；新 root 只能来自已经核对的实际 Python JIT code/module 位置。将候选 ELF 的唯一 `.debug_info` comp_dir 恢复成旧 root，并重算其 DWARF2 unit_length、必需的 section 文件偏移/e_shoff/零 padding，随后核完整 ELF SHA `4a9f93b8…`。其他三个 debug section 也逐字保留，任何非路径 payload、符号、note、descriptor、指令、flags、VA 或段内容变化都不能通过。

五件文本仅替精确 old/new root；source/TTIR/TTGIR/LLVM/ASM 路径计数固定为37/34/34/1/2。ASM 唯一 Length of Unit 须先等于实际 ELF debug_info.size−4，再恢复171；其余字节严格比较。全部原始 SHA 与大小继续原样保存，新增 `source_path_identity` 记录规范后摘要。C函数/5124 DSO、ABI/options/resources、b0fa hash 均不变。失败 proof 仍随原有效 ACK 保存，再由原 Session 拒绝。

CPU 正例包括原件自比、等长和变长目录的明确合成副本；副本不是设备编译。负例覆盖 text/descriptor/note/symbol/其他debug、padding、ALLOC/PH/偏移、尾部、文本和实际源路径。真实 run 仅走到首 idle，HTTP/worker 为明确 CPU fixture。未加载旧 cache、未编译、未修改 co_filename、未强制 SME；新冷编译是否满足该规则仍待原件，bridge 两批准旗及服务/模型/性能资格继续为 false。
