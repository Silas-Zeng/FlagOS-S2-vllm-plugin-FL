# 3D decode的w2s1启动

加入原e5模块与原footer，在严格BF16/3D/shape/feature守卫下把4warps3stages改为2warps1stage，kernel/grid/reducer不改。

真正源码依赖prefill产生的Attention54c和helper59d9；SQL不是算法依赖，只为保持runner footer与历史073309字节及推荐堆叠顺序。此处不能混入mixed拆分。

本材料只是历史实现拆包。仅BI-V150证据，C500未验证；不改官方服务参数、模型、量化或benchmark。所有旧源/失败/局限保留，不能把同一完整组合成绩分摊到各PR。原代码函数体复用，只有01/02为拆开共享入口而机械移除/加入原top-p函数及调用；没有新算法或本轮上卡。
