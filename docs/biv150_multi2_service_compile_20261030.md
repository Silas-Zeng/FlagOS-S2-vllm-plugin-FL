# PR123：实际生产 multi2 的 prepare-only 编译入口

生产目录逐字保持 `d5d9c56f51d731e03b78a7087ffdb1a64de93c11`。
新入口复用冻结 `ce905346a0bd6938cd8478cde34c1bcc52eaeed7` 的 driver、
`allocate(torch)` 和 `c.kwargs`，9处有计数及完整字节逆变换的适配。
须提供当前 frontend Git对象及 `844a69f10539c551500e29e2304061aef7d8671b`
依赖对象（其中两个旧 helper 的 SHA 原样校验）；ce905 是本分支祖先。
不得为缺对象而跳过 hash 或改用当前已改变的旧 helper。

默认只输出 CPU plan：

```sh
python3 -B -S -W error tools/probe_biv150_konly_multi2_service_compile.py
python3 -B -S -W error tools/check_biv150_konly_multi2_service_compile_cpu.py
```

Root 在 owner/端口/GPU 串行门闭合后，才执行：

```sh
python3 -B -W error tools/probe_biv150_konly_multi2_service_compile.py \
  --compile-only --root-owned-exclusive --commit <frontend完整提交> \
  --out /workspace/experiments/minicpm-s2/cjj-native123-service-compile-v1-20261030
```

继承原 source/干净Git/新绝对out/独占确认/ixsmi进程/9031/SDK/JIT检查。
生产模块按真实 `vllm_fl.patches.biv150_konly_encoded_multi2` 正常导入并核路径/源；
不安装 encoded service。使用真实 service `implementation_digest()` 创建独立 Adapter，
仅调用一次 `prepare`，零 reader/writer/encoder launch，原 allocation 不初始化内容。
`binding=None`（包括正常 prelaunch fallback）必须失败；未知 warmup 次数保留 None，
只有完整双臂成功才记2次已完成 warmup。

原 `_preserve` 在 `bindings/<pid>/<spec>/` 保存双臂 IR/cubin/source/record；另保存原
compiler metadata、binding proof、真实 handle resources 和实际执行 driver。
逐项核对并记录两个私有 cache group 的7个原件、metadata hash与每个最终handle artifact SHA，
缓存全留在新out内，便于后续原严格cache-hit loader复用。新cubin/cache身份只由实际结果填写。

编译成功仅允许继续独立实际产物审查；不授新数值、Graph、模型或服务资格。
此caller未加timing，也未复制数学/数值driver。下一步仍用原完整数组参考与BF16诊断边界。
CPU检查只检查源码、参数拒绝及模拟cache格式，不运行SDK/设备或授编译资格。
