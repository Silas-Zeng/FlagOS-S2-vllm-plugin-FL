# S19/S12/AD18：隔离 primitive 的内存归因准备

不修改生产算子或服务；当前PR37→PR36串行任务优先，内存诊断只能作为后继。
基于实时团队main 0bf9d2e。探针使用明确传入的私有native SO，记录SHA，禁止在9031服务或GPU占用时运行。
每种primitive/shape在新子进程执行，避免把前一个GEMM的lazy workspace带入下一个基线。

矩阵：main 0bf9d2e的native gate M32/256/2048、BF16 head M256 primitive、历史INT8 head M256 primitive。
不把native gate M32标成PR37的独立fused M32候选；该候选不在本探针中。
真实layer20 gate/up与head权重，逐tensor SHA；输入为合成BF16。BF16 head只是F.linear primitive，
未加载完整vLLM/FlagGems dispatch，不当作完整生产head或profile复现。历史INT8 head不是当前已部署路径。

分别记录context、释放量化临时张量后的权重、输入就绪、冷调用输出活跃/释放、清理非活跃缓存、两次暖调用。
区分Torch active/reserved/peak与device_used采样，按真实storage pointer去重（marker只占一个标量），
保存每次调用的alloc/free trace与snapshot，不将42层串行scratch相加。调用峰值delta以已存在输入/权重为基线。
mem_get_info调用后差值只说明残留分配，不能证明非Torch瞬态峰值；trace也可能受容量/平台支持限制。
本探针不测吞吐、不进行精度验收；最终因果仍需原生产profile阶段和KV tokens的对照。

历史证据不能忽略：分块head d5f5f58曾令M256 Torch峰值192.751→128.750MiB，
但服务KV504688对旧504784，没有恢复容量；不能再次只减同一INT32张量便预言KV增长。
当前PR37启动日志是KV547632。原生gate M2048显式INT32为96MiB；原生head M256为127.5MiB，
M2048/1020MiB仅是假设大head行数，不是当前sampler profile已发生的事实。

CPU仅核对tensor几何/字节与语法，GPU及真实trace待验证。没有性能/内存改善结论，也不能据此合入生产改动。


## 01:24后：已串行排队，尚无GPU结果

冻结源码0cb108a，个人实例独立目录code/memory-ownership-20261025。
队列188322等待PR36队列185093/start320829287；后者等待PR37队列178221/start320757084。
原服务源/依赖未改，队列与子进程均检查9031/GPU空闲。复用明确SHA为3e114fc0201dd8e4ef31b5aa7a3c75364469a6d69e54d3ae2d6ac929e254830d的私有SO，
这是当前a19ff5c服务manifest的build，不是先前组件build。尚未开始任何内存测量，不把排队当结果。


## 实测结果与下一项候选（当前仍Draft）

0cb108a五个新进程案例全部结束并保存snapshot。实际main-native gate M32的24MiB wq位于96MiB分配段；
除该权重以外该大段其余72MiB空闲但无法归还。head的255MiB wq位于1020MiB段；输入就绪且empty_cache后
仍有约765MiB大段空闲分片被该权重钉住。原权重已释放，不能称为双份BF16常驻；调用后allocator/device样本
也没有证实一个额外的多GiB vendor常驻工作区。此为隔离primitive证据，不证明完整模型加载的碎片形态或历史KV下降根因，
不能把单层72MiB直接乘34（后续层可能复用空洞）。

候选仅调整quantize_weight分配顺序：先分配最终KxN INT8 packed buffer，保留原来的FP32 abs/amax/clamp/div/round，
最后copy_原q.t()。原contiguous同样需要一次转置copy，不新增权重常驻副本；保证最后wq布局/每元素INT8和scale位模式一致。
不改native CPP/CUDA/量化算法、模型选择、head部署范围、服务参数或内存计账。目标是避免最终权重分配进FP32临时大段，
而非把真实峰值移出profile或改allocator环境变量。

新增35个真实weight检查（34 gate/up+一个仅用于历史head探针的head）比较所有INT8/FP32 scale位与输入不变；
内存探针现在从候选生产函数提取精确AST执行并记录源码/AST SHA。先验证数值合同与真实snapshot是否改善，
再做完整服务profile/KV、105题和两档充分预热，现阶段不宣称容量或吞吐收益。


首轮724846b在packing-equivalence保护门即退出1，未生成任何数值比较或进入memory/service。
原因是本机Python3.14 ast.dump默认省略空字段，远端3.12保留，导致同一baseline AST摘要不同。
修复只更换工具摘要序列化为保留实际语义字段的canonical JSON，baseline合同仍锁定原函数；不更改量化数学或放宽数值门。
旧失败目录weight-packing-20261025保留，新冻结版本与新输出目录单独重试。
