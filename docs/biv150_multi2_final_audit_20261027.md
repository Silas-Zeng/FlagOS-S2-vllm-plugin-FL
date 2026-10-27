# PR66 独立终态审计入口

本工具提交只包含 tools/docs，`vllm_fl/` 与 `csrc/` 必须逐字保持模型冻结
`e08c79456f9b943dbc4d9d02d0a4e030b4eceac1`。运行中的模型 pair 不使用本工具新
commit 作为模型版本。基线固定 main2412，正式 driver/controller 仍为已冻结 e08
的 `6f741f12...` / `eccb67a6...`，不在 GPU 运行期间修改或重测模型。

等 root 确认 pair 终态及完整原件可读后，再执行纯 CPU 审计：

```bash
python tools/audit_biv150_multi2_pair.py \
  --root /workspace/experiments/minicpm-s2/cjj-multi2-main2412-pair-v1-20261027 \
  --baseline-commit 241220723a09061c182764f8f738e5ec01190367 \
  --candidate-commit e08c79456f9b943dbc4d9d02d0a4e030b4eceac1 \
  --out NEW_UNIQUE_AUDIT_JSON
```

不读取模型权重，不导入 Torch，不访问网络，不启动 GPU/服务，不改输入原件。
输出文件必须不存在。0 表示本次固定 pair 满足全部审计门且两档实测 TPS 均高于
其基线，可进入 merge review；1 表示拒绝或证据不完整，仍输出诊断 JSON。
工具不执行 merge，单 pair 不代表统计显著或稳定收益。

## 硬审计内容

- 每 arm 的原始 report/review/prediction：105个唯一同题、binary准确率、成功状态、
  无失败预测；从原始 reviews 重新汇总正确数，与 metric/verdict/precision flags
  一致。99/105 即使 driver正常完成仍不能accept，不重抽、不调用新grader。
- initial/post-score 的 raw SSE、decode-response与prefill-response独立重算1..128
  和85，检查 transport、原始结束标记、HTTP时间重叠及已存summary一致。请求记录
  的固定fixture SHA要吻合；HTTP重叠不解释成模型/GPU重叠。
- 逐项读取3短暖场、1完整暖场、每档唯一score的原始 command/result；固定
  CLI、请求/并发/token计数、有限正吞吐/TTFT与TPS-duration-token算术一致。
- 独立按冻结 research limits 重算working_gates，要求与子state逐项一致，并把
  precision/throughput/TTFT全部纳入formal gate；不把controller completed/exit0
  或`label_validated`当作验收。
- 正式serve/default CLI、native5cbb、同initial snapshot、冻结driver/controller
  SHA、source commit/fingerprint、默认feature env缺省、controller嵌套arm原件
  与最终comparison一致。
- baseline没有MULTI2，但允许原有Q32_SELECTED；candidate必须实际MULTI2_SELECTED、
  num_sequences2/rows>=1024、完整grid和BM256/w16/K64配置，hash关联实际
  Q-fragment lowered proof；两arm有真实C3正常选择，无Q32/MULTI2 fallback或
  scope reject。
- 两arm都报告完整owned cleanup、9031关闭、无remaining，owner身份非空。
  这是完成原件检查；root仍需独立确认当前进程/GPU空闲，工具不检查实时机器。

source/library/snapshot的启动时内容核验由冻结driver执行，本工具核对其固定身份、
实际server fingerprint日志与完整原始记录；不重新载入SO或重算依赖Torch ABI的
live指纹。应连同root已有prepared manifest、SDK/C3 audit和最终逐文件SHA归档。

## CPU 故障证明

`python tools/check_biv150_multi2_final_audit_cpu.py` 生成明确标记为synthetic的最小
合法pair，再注入：99/105、伪造precision=true、TTFT、working gate、请求token、
少一grid、缺SELECTED、baseline误接管、源fingerprint、手工feature env、cleanup、
105重复题、raw SSE、snapshot、serve CLI和comparison错误。合法fixture允许
baseline Q32，全部16类错误拒绝。此检查不是模型精度/吞吐证据；摘要写入
`docs/evidence/prefill-multi2-20261027/final-auditor-cpu.json`。

## 模型结果回来后一次性补证索引

| 原件 | 结果/归档SHA |
| --- | --- |
| root prepared manifest、SDK/C3 off/default-on audit | 待原件 |
| e936 18组scope result/actual IR | 已逐SHA复验/归档，见completed evidence文档；含254b四例 |
| main2412/e08两arm的105正确数与raw文件 | baseline-only独立audit102/105已存；candidate尚未审计 |
| 4K/16K唯一score TPS/TTFT/完整计数 | baseline5806.1431/4479.5676及TTFT/计数通过；candidate待终态 |
| defaults、source fingerprint、native/snapshot、MULTI2/C3 proof | baseline-only原audit保留完整判定；candidate待终态 |
| raw initial/post mixed、owned cleanup | baseline-only原audit通过；模型原件未额外复制到组件包 |
| 本auditor输出与脚本SHA、最终逐文件归档manifest | 待审计 |

当前只交付组件及baseline单臂证据，详见 `docs/biv150_multi2_completed_evidence_20261027.md`。
没有candidate模型通过、pair验收或新的服务吞吐收益结论。
