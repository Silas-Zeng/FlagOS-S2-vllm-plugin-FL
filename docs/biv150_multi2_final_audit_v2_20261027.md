# PR66 final auditor v2：仅修正合法grid表示

后续root目标CPU完整v2审计已通过，本地133 raw重放一致；最终报告与原v2 JSON见
`docs/biv150_prefill_multi2_final_qualification_20261027.md`。以下保留修复时的原因和范围。

原CEA auditor对完整pair返回 `AuditError('Manifest selection events changed')`。
失败原JSON逐字保留在
`docs/evidence/prefill-multi2-20261027/completed-pair-independent-v1-failure.json`，
630B，SHA256 `8cf451c91e7dbb04f6f2d9465335d86b89280ca891c4af9c07c2b47d548d0c4f`。
原 `tools/audit_biv150_multi2_pair.py` 仍为SHA
`cea62aeaea0915ec9a4b04c05879888014517ae70308130b07d995aee945b2dc`，
原baseline-only CEA证据、成绩、e08生产与所有raw文件均不改。

原因是实际service `print(dict)`产生Python repr，grid为tuple；manifest经JSON保存，
grid为list。v1先严格验证日志scope/grid/compiler proof，随后把整个selection数组直接
与manifest比较，tuple/list差异造成误报。初版CPU fixture全部json.dumps写日志，未覆盖
真实repr；这是审计工具的表示处理bug，不是新发现的kernel数值错误。

新增 `tools/audit_biv150_multi2_pair_v2.py`，SHA256
`2c79015c8d58ab410261d1eb3333900b1836f6de89c2a8452a7e89cda970f480`。
仅在原runtime严格验证之后，把日志selection中类型恰为tuple的grid复制成JSON list，
再比较整个dict/list。原字段、hash、额外字段、顺序、数值与baseline仍严格原样比较；
不筛字段、不改raw、不改容差，不改任何其他gate。除v2说明字符串与这处表示转换，
将差异逆回后与CEA源码逐字相同。

CPU回归保留原16类fault和CLI行为：合法JSON退出0，99诊断仍退出1。新增真实Python
repr tuple正例及7类负例：log grid/hash/extra、manifest grid/hash/extra/单独字段改动。
v1对合法repr仍重现相同错误，v2通过。摘要为
`docs/evidence/prefill-multi2-20261027/final-auditor-v2-cpu.json`。

另只读核对root下载的真实candidate原件：manifest SHA
`8ca944da28bf9a9a7906b91b3033664fc778d7071a1ae7b28a486820e4af6dba`，
server.log SHA `5036196e7211db93f651c485fd4f1c77cfa45b5706d08793d3f7dc766ae52241`。
原runtime严格检查通过；其1条日志selection的grid确为tuple、manifest为list。
执行两版源码原比较片段，v1误报重现、v2整个数组相等，无其他字段差异。原件前后SHA
相同。此项只验证runtime及表示修正，**不是完整pair audit通过**。

root接下来在原pair终态执行纯CPU完整审计，输出需为新文件：

```bash
python tools/audit_biv150_multi2_pair_v2.py \
  --root /workspace/experiments/minicpm-s2/cjj-multi2-main2412-pair-v1-20261027 \
  --out NEW_UNIQUE_AUDIT_V2_JSON
```

root提供的状态是原工作负载已结束并cleanup；当前先交付v2冻结供其审计。
完整105、原始语义、TTFT/counts、source/library/snapshot/defaults、C3实际选择及cleanup
等其余门保持原样。新的完整v2输出、133个raw文件归档和最终PR结论待root完成审计后
再补；不把原始分数、终态completed或本次单点回归写成验收passed，不merge、不重跑GPU。
