# PR113 实际 SO：6 KiB shared，设备数值待验

冻结源码 `1dc7ce5c2dd9050712aa9d5e7cba59fe3699f8d4`，实际库
`5cf698e8a60272d86b2545364f580faf4c40114e0044e91557c1a137f2521bed`。
独立 CPU 报告 SHA256 为 `48b8e9d8ca1ee2a226727727dcedafa2663086952dcfa37578209bc7a54e5654`；
Root Python 3.12 重放逐字相同，并另发 `b481ccf7ec5da72b3036b91d212ebdf02c97d906d93f0914d083df7b35d2d87f`
准入记录，仅允许独占的数值、Graph、side stream 测试。两份原记录均保留，独立报告中的
`manual_Root_approval=false` 未改写。编译前后 Torch CUDA initialized 均 false，前端 kernel 计数均零。

S4 实际 ELF 为 **100 VGPR / 66 SGPR / 6144 B shared / 0 B private / 两类 spill 均 0**。
原 2ad 是相同寄存器和 spill、8192 B shared。S4 仍 1339 条指令；全部 kernel 的 2764 条
机器字均绑定到最终 SO 的嵌入 ELF，另存对象的完整 GPU ELF 也与最终 SO 一致。
保留但不准入的 S2 同样核对。完整 `.args` ABI 仅发生名称变化。

每个 producer 仅 18 条地址或地址寄存器操作数变化；其余机器字逐字相同，分支目标和
支配关系不变。实际 IR 函数体仅允许以下差异，保留全部语义标记：

- V shared 由 `[4096,8192)` 平移到 `[2048,6144)`；K 保留 `[0,2048)`。
- 16 个 V shared GEP 和 8 个 full/tail SME 目的地址减 2048。
- 尾清零的 `or disjoint offset,1024` 变成 `add nuw nsw offset,512`。完整 1024 个
  word 的 offset 是 0…1023 的排列，两者恰差 512 word；不能把新操作误写为 OR 512。
- full 分支 scale load 从 s55 改读 s59；新 s59 仍是 `0x1000`，global scale **没有**平移。

候选最终 SASS：S4 行 1348…2686；地址常量变更行 1928/1929/1931/1933，scale load
行 2078；tail SME 描述符寄存器复用行 2289…2298，tail clear 基址行 2440，PV 基址行
2490/2504/2523/2542。所有精确原/新机器字及 PC 在报告中。继承的 CFG 明细行号仍指向
7892 原件，已显式标注，不能当成本次原件行号。K 完成与 barrier 先于 V issue；QK、
P softmax/packing 先于 V wait；wait 支配 tail clear 和 PV。部分 rho exp/running-L FMA
允许留在 wait 后。codec `.text` SHA `936ce115…ba19cf` 与 2ad 完全相同。

实际 host dynsym 中 S4 弱导出为
`_Z58__device_stub__cjj_native_k8_compact_n48_shared6k_producerILj4EEvPK13__ix_bfloat16PKaPKfS2_PKiS8_fllllllllPfS9_S9_`。
它的地址为 `0x189c0`、大小 381，GOT `0x2de58` 是对此符号的 GLOB_DAT。构造器
`0x187f5` 取此 GOT，`0x18812` 把它与完整设备名注册；stub 内 `0x18b1c` 取同一 GOT，
`0x18b30` 调用 ixLaunchKernel。脚本从最终 ELF 解码并核验相对位移、重定位和设备名。
后续只能在正常加载此精确 SO 后，用 dlsym/dladdr 核对真实导出，再以 runtime API 查询
64 threads / dynamic 0；它不是 CUfunction，不可直接送给 Triton 的 driver occupancy API。
尚未实际取得 handle 或查询属性。该可选查询失败不阻塞同输入 ABBA。

静态资源变化不证明 occupancy、驻留 CTA 或收益；2 KiB shared SME 目的地址的硬件接受性、
所有输出数值、Graph/side、完整步骤、模型和吞吐仍待实测。旧 BF16 K helper 在源中保留但
没有调用，实际 IR 无它们；若以后调用，可能与新 V 范围冲突。ixobjdump 原始长签名的
换行/截断照存；不声称打印 IR 可重新编译，ABI 使用完整 ELF 元数据核对。

完整原包 356885 B，SHA256 `59b24488811ea6f96086427cb01b11661e56ad2935d669e0feb42b7c1ab47733`，
包含全部 33 原件及 download manifest，共 34 个成员，含实际 SO/IR/SASS 和编译源。
`archive-index.json` 记录每个路径/字节数/SHA。参考 2ad 包和旧审查 helper 复用已共享 Git
对象，见 `existing-dependencies.json`，没有复制无关大文件，也不自动联网。

在仓库根目录轻量核包：

```sh
python3 -B docs/evidence/native113-shared6k-actual-so-20261029/replay.py
```

完整 CPU 重放（输出目录必须不存在；不 import Torch/NumPy/Triton、不加载 SO、不编译）：

```sh
python3 -B docs/evidence/native113-shared6k-actual-so-20261029/replay.py --run --out /private/tmp/native113-new-review
```
