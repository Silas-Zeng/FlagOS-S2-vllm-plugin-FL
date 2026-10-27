# 私有 V8 2D prefill source factory：CPU 阶段

条件依赖[PR73](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/pull/73)冻结
`4a1f7ddd38e4bcade30e702faba2e9a7cdb5441d`与PR75实际完整组件性能余量。
本分支从main2412独立，只包含tools/docs，不改66/73/75、不注册服务、不导入
Torch/Triton、不JIT/SSH/GPU，不新增队列。完整模型精度与吞吐未通过，不能合并。
旧E02/E03/E04/E10已包含混合精度、直接消费与scale布局方向；本项补具体格式消费者。

`tools/biv150_v8_prefill_factory.py`仅从逐SHA核对的SDK54c495、helper59d9d3与
生产normalizer957720派生私有kernel字符串。先严格重建已审stride/zero-initial-mass
source fa6ddc，再只增加私有类型/模式guard、一次Vscale load和NONE分支PV：
`BF16(P * Vscale) × BF16(Vcode)`。K cast、QK、score、mask、softmax、累加/输出
和原死mode2/3分支原样；移除新增内容并逆回函数名后逐字及AST相等。raw ABI不变。
source factory没有调用SDK原内置量化功能，KV_QUANT_MODE必须为NONE/0；原per-token
head分支仅作索引/scale参考，不使用mode2、Kscale=1或官方量化flag/CLI作为实现。

## 实际ABI陷阱与显式binding

SDK原kernel已有`v_scale_cache_ptr`及`stride_vs_blk/slot/head:tl.int64`，所以私有
调用无需增加参数；`physical_block_idx`由原block table gather后转int64，位于
tile loop内、支配新增Vscale load。新增索引为physical-page*scale-block-stride
+(seq_offset%16)*slot-stride+kv_head*head-stride，mask=tile_mask、other=1.0。

**不能只替换kernel再使用原mode0 wrapper。** SDK wrapper在mode0会把scale ptr
替换为V本体、stride_vs全设0。`bind_private`只为显式私有raw调用绑定真实FP32
scale view及元素strides，所有非V kwargs保持原样；拒绝无格式声明、BF16伪Vcode、
dtype/offset/stride错误、dummy pointer、非NONE/2D和未经审计配置。它是CPU
metadata合同，不是从真实设备tensor提取metadata的生产安装器；未来调用方必须证明
View/handle确实对应实际tensor，不能把自报dtype当实际指针证明。

当前格式仍16384B/page：BF16 K byte0；I8 Vcode byte8192，stride
`(16384,256,128,1)`；FP32 scale byte12288、stride`(4096,2,1)`。
同page cache三个typed view共享原始storage，但彼此有效字节不重叠；scale offset
是3072个FP32元素，不能把12288当FP32元素偏移。支持描述性page pitch及Q/O stride
padding，无容量回收声明。新增scale索引仍int64；不以逻辑页号代替物理页。

保持原SDK2D配置资格：max_query_len>1为BM128/BQ16/K64/8warps/1stage；纯qlen1
只有SDK本来选2D时才研究BM16/BQ2/K32且不覆盖backend默认warps/stages。现有GQA
3D、Q32/MULTI2 widened/hoist不在本factory的设备已验范围。保留完整SDK grid
`(qrows//BLOCK_Q+num_seqs,2)`，空batch不launch。

## 必须完成的服务集成前置

V8及prefix7目前只是decode组件；编码V后，现有SDK BF16 2D prefill、短多query、
generic fallback都不能直接读它。原normalization._eligible只查dtype/shape/mode0，
其BF16 view仍可能被接纳，不能识别payload已改变。标准库反例：I8[-127,127]
的bytes81 7f被旧BF16 reader当成0x7f81，即NaN；I8[127,0]被当作BF16 0x007f，
也不等于Vcode×Vscale重建。mixed forward的decode/prefill共享同一kv_cache。
这是未来V8集成风险，**不是e08原BF16服务的错误**。

未来服务必须让所有消费者理解同一格式，覆盖prefill/短多query/2D/3D/decode、
profile/warmup/capture/replay、preempt/recompute、prefix共享/页复用、append、raw
copy/offload/transfer和cache sizing。有限回退必须证明仍使用正确格式，或在launch前
完成显式格式转换且计入所有代价；绝不能把encoded BF16 view交回旧reader，不能只
hook decode。任何绑定/编译失败都不能对同一encoded cache重试旧BF16 kernel。
prefix7额外BF16保存区也需要自己的reader和全部writer/copy合同，当前factory只支持
73的全V8格式，不支持prefix7，不暗中选择某评测题或benchmark。

原table gather没有tile mask，完全沿用：未来fixtures必须有rounded TILE_SIZE页表
padding，尾部仍指向有效或可安全计算的dummy页，K/V/scale load分别保留tile mask；
不能由CPU mask算术推出真实设备越界安全。CPU阶段还不能证明资源、指令支持、
数值容差或性能。只有4a/75完整实测有余量，才考虑一个由root串行安排的首个设备
falsifier；当前没有可运行GPU验收或自动调度入口，不会因生成源码而启动设备任务。
