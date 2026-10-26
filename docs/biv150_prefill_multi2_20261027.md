# 双序列预填充的大tile组件试验（仅CPU准备）

基于实际 fetch 的团队 main `241220723a09061c182764f8f738e5ec01190367`，独立分支 `codex/biv150-prefill-multi2-20261027`。生产 `vllm_fl/`、`csrc/` 不变；当前没有上传个人实例、没有 GPU 任务或自动队列，不影响 PR63 的完整配对。

现有16K暖场日志出现 normalized `(total2033,maxQ1945,numSeq2)`、`(total2033,maxQ1994,numSeq2)`。当前 Q32 和 PR63 K128 生产策略都只选择 `num_seqs==1`，这些双序列输入仍由原 BM128/w8/K64 处理。这是覆盖缺口的源码依据，**出现次数不是耗时占比**，也不意味着扩大tile一定会赢。完整调度/每序列KV长度尚未捕获。

组件准备四组代表输入：Q `[1945,88]` 及其逆序、`[1994,39]` 及其逆序；前两组各KV4096，后两组各KV16384。只从日志总量推导Q长度组合，序列顺序和KV长度为测试选择，不能冒称原始捕获step。覆盖非对齐Q边界及长短序列顺序变化。

`tools/bench_biv150_prefill_multi2.py` 的 baseline 安装完整当前 Q-fragment/Q32，未来源码含K128时也要求安装实际K128，保留真实外层policy。仅在Q32的原 BM128 fallback 上放透传tap；单序列父路径不被关闭，双序列按父策略自然回退。必须观察实际BM128编译/选择和资源，父manager仍active、未disabled、绑定未变且两序列未误选父路径。K128存在却失活、依赖异常或链路变化会拒绝运行，不伪造“当前baseline”。

candidate使用私有normalized函数/JIT，BM256/w16，先K64/stage1，另可有界测试K128/maxnreg128。使用生产原严格Q-hoist/lowered proof，不使用PR64的early-Q变换。数值协议源自已共享PR64 `032d6d3`，复用原CPU FP64/reference_guard、changed-input非默认stream Graph、真实加载资源/driver capacity和三轮交错计时；保留IR和失败原件。Graph计时直接重放已验证核，组件数据不能冒充实际服务命中或模型吞吐。

`python3 tools/check_biv150_prefill_multi2_cpu.py` 通过：现有Q32/未来K128两种模拟完整policy链共八次双序列自然BM128回退，失活K128/断链/其他依赖丢失、父路径误选和四类scope错误均拒绝。只是显式假CPU fixture，没有导入Torch、没有运行目标编译器；不证明真实安装、C3闭包、数值、资源或速度。

仅在PR63控制器及所有owned进程实际终止、cleanup完成、GPU和9031空闲后，先fresh同步团队main，再考虑以下独立组件测试：

```bash
python tools/bench_biv150_prefill_multi2.py --run-gpu \
  --kv-tile 64 --out NEW_UNIQUE_PATH
```

若第一配置有机制和收益，再单独比较 `--kv-tile 128 --maxnreg 128`；资源升高或某档无益时保留负结果。原baseline必须仍是该输入的真实BM128 fallback；若未来团队改变双序列默认策略，应停止并更新本探针，不能对过时fallback宣称收益。只有真实组件可行后才设计生产二序列选择，并重证严格C3绑定和完整105题/两档吞吐/TTFT；此Draft不包含任何放宽生产scope的改动。
