# PR120：真实 Tensor 的完整 host validate 组件

从已共享 `8554750` 独立准备，基线为 main `899ce77`。仅新增工具和本文，不改冻结分支、生产 adapter、CUDA、codec、比赛 CLI 或 benchmark。

main899 的 `biv150_konly_encoded_native.py::Adapter.try_produce` 与 mixed adapter 都调用父 `validate`。此完整函数包含 dispatcher/code binding、Registry epoch/owner/recipe/roles、9个bounds、完整storage别名及scalar/stride检查。PR120保留这些检查及顺序，仅复用本次bounds观察，把有效调用的storage helper调用从57降至9。既有32组CPU fixture通过，但不是实际Tensor计时。旧119 A窗口证明host路径存在；没有当前899的validate墙钟占比，不能把它先称为主热点。

本入口在真实 Torch2.10/ABI false、CUDA0 BI-V150 上加载实际5898库，并直接构造两份未经修改的完整Adapter；参考模块来自固定main899 Git blob。两臂共享同一个原Runtime/Registry/broker、同一组真实Tensor和实际dispatcher对象，没有fake storage或fake op。调用完整validate，检查返回的9个对象身份；两臂接受/异常类型/错误文本一致。覆盖真实storage/offset替换、owner/epoch/role改变、whole-backing别名、同scratch对象、dtype/device错误与双错误顺序。Torch拒绝构造的越界metadata不伪造为Tensor，原CPU负例仍保留。

创建42个真实Registry条目，每层仅29个16KiB页；4个形状分别访问首/中/末条目：64/S2/pitch2048，32/S4/pitch2560，61/S2/pitch2560，30/S4/pitch2560。pitch2560的Q/K/V共享实际readonly backing。计时固定128次/臂warmup、3轮ABBA、每sample512次；每个shape分别在真实default/side stream上下文计完整host `perf_counter_ns`。不在每次validate额外加stream查询；原函数本身没有该查询。计时中不patch方法、不计序列化或fixture准备，不减去空循环耗时。

这些数据只回答**完整host校验成本是否下降**。无Native producer/writer/encoder调用，无Tensor内容初始化/读回，无模型加载。真实CUDA context、storage分配、stream及SO加载仍需要Root排他设备窗口；不能把0attention称作0CUDA API。29页/42个metadata entry不等于42层模型容量测试；Graph replay不执行这段Python，default/side host正确性也不是Graph数值资格。

默认只打印CPU计划：

```sh
python3 -B -S tools/probe_biv150_native_storage_actual_validate.py
```

本地冻结后CPU准入使用已保存真实5898库的字节，包含true-ready入口tripwire，绝不导入Torch或调用设备：

```sh
python3 -B -S -W error tools/check_biv150_native_storage_actual_validate_cpu.py \
  --library /absolute/actual5898.so --report /private/tmp/pr120-actual-validate-cpu.json
```

Root检查前驱已结束、GPU独占及端口关闭后，才可对新的out运行：

```sh
python3 -B tools/probe_biv150_native_storage_actual_validate.py \
  --actual-cuda --root-owned-exclusive --commit EXACT_FROZEN_HEAD \
  --library /absolute/actual5898.so --out /workspace/experiments/minicpm-s2/NEW_TAG
```

此准备尚无设备结果。若完整validate无明显收益，就不凭57→9升级整机；即使host成本降低，也需后续独立服务桥绑定新adapter/manifest/C3与完整源，保留当前5898及mixed ebbc，不冒充原controller接受的生产身份。已有source/94db CPU检查不能替代自然服务导入或吞吐验收。组件不修改现有C3或controller，也不输出服务资格。
