# 实际31／41／42／43行的随机采样head候选

基线为团队main `a455a3de364051fbea80226870c239de3a48e716`。本候选独立于compact／N48 attention；没有改变服务或benchmark参数。

原head温度融合只接真实请求数32／64。107和109实际计分客户端未设置temperature，使用新版`lib/endpoint_request_func.py`，服务日志确认模型默认temperature1.0／top_p0.95；不能套用旧backend客户端的temperature0。首次head SELECTED／SAMPLED来自暖身，只证明实际出现，不能作为正式计分的命中频率或41／42行损失证据。

## 默认关闭的最小接口

`VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED=1`仅增加真实整数行数31／41／42／43；原`VLLM_FL_BIV150_HEAD_TEMPERATURE=0`仍关闭整个head融合。未设置或值不为`1`均不启用新增行数。

| 真实请求／hidden行数 | 复用的原配置对象 | grid |
|---|---|---|
| 31 | R32：BM32／BN64／BK64／w8／s1 | 2040×1 |
| 41／42／43 | R64：BM64／BN256／BK32／w16／s2 | 510×1 |

32／64配置表、kernel、数学及原采样条件不变；资格检查先保留原32／64判断，新增helper不读取GPU标量、不读文件或重新审查源码。其它行数保持原head／Sampler。新增开关进入正式编译语义指纹，未设置与显式0相同，1与0隔离；主head关闭时新增开关不生效。当前CPU测试只证明这些协议，不代表实际SDK／JIT／资源／数学／RNG资格。

## 旧十门的实际来源

已用经核验的CJJ1008 `gh api`只读取回团队方法记录所指两组Git对象，未写团队分支。`docs/evidence/head-temperature-ragged-20261029/legacy/manifest.json`保留实际API路径、Git blob SHA及内容SHA；本提交保存其中worker／factory／helper／candidate、两组result和archive-receipt子集，不宣称保存完整历史bundle。

- R64：`20261027T050730877973Z-lm-head-temperature-epilogue-component`，原worker `b529db313cd657b00d14bd116ea901ab518194c83ed2dace04a3cbd66df955ab`。
- R32：`20261027T083835962824Z-lm-head-temperature-r32-epilogue-component`，原worker `297a77c2894d3a87cc497757da1ab67a48d17538d39604aec24dbd774ea0bc9f`。
- 两组candidate相同：`4c800693a2ae134fc580382d4166091b4c48e96dc2b7838ab417da7094cb2537`，与main kernel逐字相同。

原结果各10门：normal完整scaled／filtered／采样ID逐位、CPU／默认CUDA／逐请求generator RNG、输入不变；独立FP64样本；changed／zero／extreme；逐请求generators；side；两次换值prefix Graph；timing后复核。旧独立FP64门为`atol=.001, rtol=.01`，全scaled／filtered／ID为零容差。Graph仅捕确定性head／cast／temperature，完整原Sampler留在eager计时。旧完整head＋Sampler单ABBA含实际分配、原统计／RNG与Python提交；不得以prefix Graph替代完整成本。

这两组历史记录只授32／64；31／41／42／43未执行。下一工具提交将参数化原R32 worker而非冒充重新构造的旧十门，并为非整tile增加完整CPU权威边界快照。旧entry绑定main56／native14f／原DB，且用空ixsmi进程表作排他证明；不能直接复用其外层启动／写回DB逻辑。

## 尾行与准入限制

标量地址模型对31行的最后一个tile lane回读row0；41／42／43分别有23／22／21个重复回读lane。实际有效输出仅真实M行；temperature tail有显式mask、other1；N130560整除BN64／256，K2048整除BK64／32。CPU枚举证明该地址公式的范围，**不能证明`tl.multiple_of`／`tl.max_contiguous`在新的非整M特化下被编译器安全消费**。kernel未改这些hint；必须先保存fresh实际IR／机器码、全输入／输出CPU权威canary和原十门；失败停止，不继承32／64资源或性能资格。

新实际配置需以原F.linear捕获的真实配置核对，完整head＋temperature＋同Sampler比较，保留acc→BF16→FP32→除温度的顺序。支持的metadata、一次温度票据、grammar／新metadata回退、RNG和异常传播不放宽。实际请求数不能替换成pad48桶；新开关每call检查有host成本，不能称零开销。任何后续局部正结果均不等于整机、105或正式吞吐。

## CPU复放

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_head_temperature_ragged_cpu.py --out /private/tmp/head-ragged-cpu-new.json
```

需在该独立worktree运行；输出路径必须不存在。检查完整真实模块加载／旧源逆变换、kernel字节、0..130行和坏类型、实际参数identity与grid、greedy／penalty／top-k／prompt-logprob／capture拒绝、metadata／grammar／开关失效回退、prelaunch拒绝和launch异常传播。GPU对象是明确的opaque替身；未执行实际SDK初始准入、GPU数学或采样RNG。另以独立无父包subprocess加载compile_semantics，禁止framework导入，验证开关指纹。

当前没有可执行的device入口／实际设备结果；组件前端随后正常提交。该接口默认关闭且未安装服务，不能合并为已验收优化。
