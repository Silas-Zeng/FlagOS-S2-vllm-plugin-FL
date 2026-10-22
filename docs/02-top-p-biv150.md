# 直方图zoom top-p过滤

加入原_patch_top_p_sampler和原bounded histogram实现，并在共享入口中按原顺序先top-p后Attention；top-p-only、float32 logits、vocab130560的路由及其余fallback逐byte保留。

此堆叠依赖是共享biv150_runtime文件的拆包关系，top-p算法不依赖prefill计算。01与02中原算法函数体不改，02完成后共享runtime精确回受测a0c。

本材料只是历史实现拆包。仅BI-V150证据，C500未验证；不改官方服务参数、模型、量化或benchmark。所有旧源/失败/局限保留，不能把同一完整组合成绩分摊到各PR。原代码函数体复用，只有01/02为拆开共享入口而机械移除/加入原top-p函数及调用；没有新算法或本轮上卡。
