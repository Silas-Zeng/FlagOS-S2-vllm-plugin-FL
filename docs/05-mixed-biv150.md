# Attention内部天然单token前缀拆分

加入原a129模块与原footer，builder构造天然连续单token前缀计划，forward拆原Attention的decode/prefill范围，不排序请求、不拆整个模型。

硬依赖已安装原e5 DecodeLaunch以及Attention54c/backend1219/helper59d9。mixed把部分原2D转3D，归约顺序变化；准确率属于对应完整组合，不能把它归入SQL。

本材料只是历史实现拆包。仅BI-V150证据，C500未验证；不改官方服务参数、模型、量化或benchmark。所有旧源/失败/局限保留，不能把同一完整组合成绩分摊到各PR。原代码函数体复用，只有01/02为拆开共享入口而机械移除/加入原top-p函数及调用；没有新算法或本轮上卡。
