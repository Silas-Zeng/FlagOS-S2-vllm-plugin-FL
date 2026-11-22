# 实际31／41／42／43行的随机采样head候选

基线为团队main `a455a3de364051fbea80226870c239de3a48e716`。本候选独立于compact／N48 attention；没有改变服务或benchmark参数。

原head温度融合只接真实请求数32／64。107和109实际计分客户端未设置temperature，使用新版`lib/endpoint_request_func.py`，服务日志确认模型默认temperature1.0／top_p0.95；不能套用旧backend客户端的temperature0。首次head SELECTED／SAMPLED来自暖身，只证明实际出现，不能作为正式计分的命中频率或41／42行损失证据。

## 默认关闭的最小接口

`VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED=1`仅增加真实整数行数31／41／42／43；原`VLLM_FL_BIV150_HEAD_TEMPERATURE=0`仍关闭整个head融合。未设置或值不为`1`均不启用新增行数。

| 真实请求／hidden行数 | 复用的原配置对象 | grid |
|---|---|---|
| 31 | R32：BM32／BN64／BK64／w8／s1 | 2040×1 |
| 41／42／43 | R64：BM64／BN256／BK32／w16／s2 | 510×1 |

32／64配置表、数学及原采样条件保留；kernel 已增加独立 ragged mask 特化，False 分支 AST 与原 kernel 相同（见本次修正说明），不能称整个 kernel 字节未变；资格检查先保留原32／64判断，新增helper不读取GPU标量、不读文件或重新审查源码。其它行数保持原head／Sampler。新增开关进入正式编译语义指纹，未设置与显式0相同，1与0隔离；主head关闭时新增开关不生效。当前CPU测试只证明这些协议，不代表实际SDK／JIT／资源／数学／RNG资格。

## 旧十门的实际来源

已用经核验的CJJ1008 `gh api`只读取回团队方法记录所指两组Git对象，未写团队分支。`docs/evidence/head-temperature-ragged-20261029/legacy/manifest.json`保留实际API路径、Git blob SHA及内容SHA；本提交保存其中worker／factory／helper／candidate、两组result和archive-receipt子集，不宣称保存完整历史bundle。

- R64：`20261027T050730877973Z-lm-head-temperature-epilogue-component`，原worker `b529db313cd657b00d14bd116ea901ab518194c83ed2dace04a3cbd66df955ab`。
- R32：`20261027T083835962824Z-lm-head-temperature-r32-epilogue-component`，原worker `297a77c2894d3a87cc497757da1ab67a48d17538d39604aec24dbd774ea0bc9f`。
- 两组candidate相同：`4c800693a2ae134fc580382d4166091b4c48e96dc2b7838ab417da7094cb2537`，与main kernel逐字相同。

原结果各10门：normal完整scaled／filtered／采样ID逐位、CPU／默认CUDA／逐请求generator RNG、输入不变；独立FP64样本；changed／zero／extreme；逐请求generators；side；两次换值prefix Graph；timing后复核。旧独立FP64门为`atol=.001, rtol=.01`，全scaled／filtered／ID为零容差。Graph仅捕确定性head／cast／temperature，完整原Sampler留在eager计时。旧完整head＋Sampler单ABBA含实际分配、原统计／RNG与Python提交；不得以prefix Graph替代完整成本。

这两组历史记录只授32／64；31／41／42／43未执行。下一工具提交将参数化原R32 worker而非冒充重新构造的旧十门，并为非整tile增加完整CPU权威边界快照。旧entry绑定main56／native14f／原DB，且用空ixsmi进程表作排他证明；不能直接复用其外层启动／写回DB逻辑。

## 尾行与准入限制

以下是 b606 原方案的地址模型及已暴露风险，不能作为修正后的读取方式。标量地址模型对31行的最后一个tile lane回读row0；41／42／43分别有23／22／21个重复回读lane。实际有效输出仅真实M行；temperature tail有显式mask、other1；N130560整除BN64／256，K2048整除BK64／32。CPU枚举证明该地址公式的范围，**不能证明`tl.multiple_of`／`tl.max_contiguous`在新的非整M特化下被编译器安全消费**。b606 kernel当时未改这些hint；原实际31行IR已暴露输入SME整32行读取风险。当前已改为显式mask，仍须核验修正后的fresh实际IR／机器码、全输入／输出CPU权威canary和原十门；失败停止，不继承32／64资源或性能资格。

新实际配置需以原F.linear捕获的真实配置核对，完整head＋temperature＋同Sampler比较，保留acc→BF16→FP32→除温度的顺序。支持的metadata、一次温度票据、grammar／新metadata回退、RNG和异常传播不放宽。实际请求数不能替换成pad48桶；新开关每call检查有host成本，不能称零开销。任何后续局部正结果均不等于整机、105或正式吞吐。

## CPU复放

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_head_temperature_ragged_cpu.py --out /private/tmp/head-ragged-cpu-new.json
```

需在该独立worktree运行；输出路径必须不存在。检查完整真实模块加载／旧launcher逆变换、kernel False特化AST及True特化允许改动、历史kernel字节、0..130行和坏类型、实际参数identity与grid、greedy／penalty／top-k／prompt-logprob／capture拒绝、metadata／grammar／开关失效回退、prelaunch拒绝和launch异常传播。GPU对象是明确的opaque替身；未执行实际SDK初始准入、GPU数学或采样RNG。另以独立无父包subprocess加载compile_semantics，禁止framework导入，验证开关指纹。

当前已增加下面的独立组件入口，尚无新形状实际设备结果。该接口默认关闭且未安装服务，不能合并为已验收优化。

## 独立组件入口（仅Root后续执行）

默认两个命令只输出stdlib计划；GPU参数若不齐在framework import前拒绝：

```sh
python3 -B tools/compile_biv150_head_temperature_ragged.py
python3 -B tools/bench_biv150_head_temperature_ragged.py
python3 -B -W error tools/check_biv150_head_temperature_ragged_component_cpu.py --out /private/tmp/head-ragged-component-cpu-new.json
```

Root确认上一owner终态、端口关闭并独占设备后，可先只编译31行，再短数值；`HEAD`替换为本工具冻结commit，输出目录必须全新。

```sh
python3 -B tools/compile_biv150_head_temperature_ragged.py --compile-only --root-owned-exclusive --commit HEAD --row 31 --out /workspace/experiments/minicpm-s2/head-ragged-compile-NEW
python3 -B tools/bench_biv150_head_temperature_ragged.py --run-gpu --root-owned-exclusive --commit HEAD --row 31 --model /workspace/MiniCPM5-2B --tuning-snapshot /ABSOLUTE/v6-SNAPSHOT --out /workspace/experiments/minicpm-s2/head-ragged-numeric-NEW
```

不传`--row`才依次执行31／41／42／43；任一失败即停止，不重试、不换tile。私有tuning副本要求输入目录`manifest.json`中的`databases`逐SHA匹配；不写回源DB。不加载无关W8A8／GQA SO；显式记录并拒绝变化的旧15份实际Sampler／FlagGems source pins，以及已取回的LibEntry414e／JIT66a源身份。没有借用旧native14f库身份。

编译入口用实际JIT `warmup`，分配未初始化Q／W／output／temperature，允许CUDA context／allocator／编译，**零head／attention／writer／encoder launch、零输入初始化**。保存每个真实M的完整TTIR／TTGIR／LLVM／ASM／cubin及signature／constants／metadata；prelaunch字段不能称loaded资源或occupancy。实际hint是否安全仍须Root审真实产物及数值，源码地址枚举不授该资格。

数值worker沿原R32十门参数化为真实M，另加一组逐行不同的正temperature检查，防止广播／索引错误被统一温度掩盖。9个原数值／RNG／observer helper AST及原PreScaledSampler类保持；check仅增加canary；原timed_batch字节不变。独立FP64原4行扩至所有真实行×同5列，阈值仍`.001/.01`；完整scaled／filtered／ID／RNG门仍零容差。新真实原F.linear配置必须恰好落在上述复用配置，否则首次候选launch前拒绝。

额外guard在非零offset视图前后保留marker，保证数据指针16B对齐。完整storage先CPU读取、按物理行重排，再取边界；1D temperature／top_p检查完整4字节而非单字节。边界tensor保存前clone，防止PyTorch视图序列化意外保存0.5GB整W。逐阶段记录完整CPU storage SHA与前后guard；整W初始GPU字节须与CPU真权重一致，其完整storage SHA后续只读。没有保存全部中间logits／W原始大数组，不能把这些hash／程序检查声称事后独立全数组重算。

canary检查、快照与哈希在ROI外；timing mode仍按真实M分配生产大小output，保留原完整eager head＋Sampler单ABBA／8调用及原6CPU quota检查。候选多一个诊断模式分支，其host成本仍计入，不能称零host成本。Graph仅确定性prefix；完整随机Sampler保持eager／原RNG。Graph新值先replay再核新值eager，readonly／边界在CPU核；不把prefix Graph当完整采样Graph资格。

当前仅CPU源码／协议通过，真正SDK导入、JIT、尾行hint、loaded资源、10门、性能、服务、105均未验。Root可据独立编译或首case反例另提交精确mask／hint修复，原32／64与原硬门不动。

## 本次显式行 mask 修正

原 b606 的31行 compile-only 已完成，但零head launch，不能当作数值或内存安全通过。完整原件与修正说明见 [mask修正证据](evidence/head-temperature-ragged-mask-fix-20261029/README.md)。新增 RAGGED_M 默认False；生产32/64传False，31/41/42/43及其compile-only/组件factory传True。True分支用真实rm、所有A load显式rm<M和other0，K尾段再合并原K mask；不再对ragged行作modulo后连续/对齐提示。False分支投影及True分支允许改动经CPU AST核验，数学循环、dot与温度舍入保留。实际编译/硬件读取行为、数值/RNG仍需重新验证，不转授旧结果。
