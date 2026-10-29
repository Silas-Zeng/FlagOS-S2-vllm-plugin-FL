# PR120：同步已合并PR118后的兼容复核

独立分支已正常合入团队main `899ce77470b0b539aebc108fd122f668c2d5894d`，保留旧fdc源码与main56记录。相对新主干，生产改动仍仅为NativeK8每次validate复用storage观察及其manifest摘要；公共两参数overlaps接口保持原字节，不跨调用缓存。

复用原CPU检查，将父adapter与mixed forward/producer AST的来源同时明确改为main899。32组两次调用的布局、所有权、跨调用变化、别名和错误顺序检查通过；每次validate的storage helper读取仍为57→9，真实mixed生产AST在CPU fixture内的两次调用为94/91→46/43。实际5898成品SHA/ELF和94db C3检查通过，原错库拒绝门保留。它们是CPU fixture计数，不是设备或模型性能数据。

运行：`python3 -B -S tools/check_biv150_native_storage_main899_cpu.py --library /absolute/verified-5898.so --helper /absolute/verified-94db-helper.py`。新入口仅重用既有检查并修正报告中的历史来源标签；原main56入口不改。

本次报告SHA为 `f35605209fea3fb08a9e5a62a871787df3b5ebb0a0e7030ab62612e83f82270b`。GPU数学未变，未加载CUDA算子、未验证服务吞吐；PR继续保持Draft，下一步需要实际host调用成本及新主干服务验收。
