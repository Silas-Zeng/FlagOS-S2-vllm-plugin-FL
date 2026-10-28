# PR107 真实精度与 Native 请求覆盖

受测生产源码 `b5d1a241f45efd21517d04e25a7dc5d44ac1ca06`，8b86新库、90c native模块、v3 FLGraph观察器。独立CPU报告SHA256 `6b860fce0c330c68229a71f0d97b468444cbb1d5d1d985d0861655df92214723`。原件、grade与controller字段没有修改。

| 测试 | 正确/总数 | 准确率 | 请求错误/空输出 | 达32768上限 |
|---|---:|---:|---:|---:|
| 官方105题、并发8 | 100/105 | 0.9523809524 | 0/0 | 2 |
| 独立补充105题、并发64 | 102/105 | 0.9714285714 | 0/0 | 1 |

原精度门0.95均通过，保留截断及错误题；补充64不替代官方8。逐题grade求和、105唯一题集及真实输入、固定采样参数、CLI/token账本均独立核对；没有重新人工判题，也不把两次随机采样差异归因内核。

新库32/64证书各绑定42层epoch。补充评测2554个真实replay按响应ID联到105题：64桶1828次/115032有效decode行/全部105题，32桶726次/20803行/其中32题。另14次属于此前64burst，不计入Math。官方8没有观测32/64 Native replay。证明的是实际参与与保存的覆盖下界；不声称每个token均Native。缺最终observer摘要，不能宣称零overflow、最终零observer错误或完整频率；这与已完成105题且逐题有Native参与记录是两个不同结论。controller原保守quality字段仍为false。

完整55原件44782164B及下载manifest分在三个归档中，文件互不重复，归档SHA见 `index.json`。解压三个包至同一目录可还原全部记录，私有生成cache按原collector约定排除。`actual-bootstrap-v3.tar.gz`另保存实际4文件诊断bootstrap；它不属于55个输出原件，也不进入正式性能计分。大单包的持久本机副本位于 `repos/experiment-raw-archives/2026-10-28/cjj-native107-main105-math105-v1-raw-20261028.tar.gz`。

详见[独立审查范围与复跑依赖](independent-review.md)。审计脚本按不可变b5d Git对象验证252源文件及symlink；共享开发分支已增加docs，不取新HEAD冒充受测源码。此证据不单独授予正式吞吐或合并资格，需结合同源双场景实际成绩。
