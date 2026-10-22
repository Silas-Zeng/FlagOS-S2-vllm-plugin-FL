# MiniCPM prefill Attention启动策略

从原runtime机械抽出原_patch_generic_attention及两段原字符串变换；目标GQA8/head128/multi-token时BM128、BQ16、TILE64、warps8/stages1。vendor入口沿原函数调用，只执行Attention部分。

包装使用00后的共享vendor文件；算法不依赖诊断开关。此安装器会修改site-packages Attention源码、清pycache、重绑backend，须隔离已安装源；不能把Git回到13eb等同安装源也恢复。

本材料只是历史实现拆包。仅BI-V150证据，C500未验证；不改官方服务参数、模型、量化或benchmark。所有旧源/失败/局限保留，不能把同一完整组合成绩分摊到各PR。原代码函数体复用，只有01/02为拆开共享入口而机械移除/加入原top-p函数及调用；没有新算法或本轮上卡。
