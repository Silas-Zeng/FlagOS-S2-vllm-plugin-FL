# PR120：main899 对照的严格服务入口准备

已完成的实际 Tensor 组件是 main899 `4a2f…` adapter 对071/855 `ce5f…` adapter：8组、96个512次完整调用样本；中位数310–314us→262–265us，单次省48.3–49.2us。原performance/C3/model标记仍false。host调用成本不是Graph replay或模型吞吐收益；原件和完整审计留本地。

此工具子分支不改071生产、数学、CUDA、cache格式、CLI、benchmark、105、warmup或owner清理。没有独立storage env；“默认关闭”只指原整个Native入口。A必须用main899生产及Native5898+mixedebbc ON，B同配置用071生产，不能给A传 `--baseline` 或关闭Native来冒充storage OFF。

新 `tools/run_biv150_native120_service.py` 只有相对原637d controller的5处桥接/入口身份字符串变化，逐字逆回原118全文。原118控制器、桥、helper94db保持原样。新增桥复用原118的环境、scope、观察、smoke和清理解析函数；不调用旧桥的source admission，不把旧c14 receipt授给新代码。

新source admission严格要求main899是祖先，生产差异仅native adapter及manifest两文件；固定本次实测SHA并精确逆回main899，保留全树mode/link-text/read-through摘要。其余生产（包括mixed、Q32、compile_semantics、所有CUDA）必须相同。新receipt绑定当前实际Git source、071组件summary7f1339…/独立报告bd03aa…及原5898/ebbc；Root两批准旗默认false，服务/模型/吞吐旗始终false。旧c14和任何源/manifest/库错误都会拒绝。

准备新bridge：

```text
python -B tools/biv150_native120_service_bridge.py --source <本worktree> --commit <冻结HEAD> --component-review <独立bd03报告> --component-summary <原7f13摘要> --parent-library <实际5898SO> --mixed-library <实际ebbcSO> --out <新绝对JSON>
```

CPU入口 `tools/check_biv150_native120_service_cpu.py` 接受相同两库/组件报告与摘要，加 `--helper <94db>`、`--old-bridge <原c14>`、`--production-ref <冻结HEAD>`、`--out <新JSON>`。检查新pending拒绝、临时正例（不保存批准件）、旧receipt/错误身份拒绝、两个ON环境、diagnostic OFF清理、A/B真实C3不同与协议逆变换。CPU C3 fixture仅启用5898+ebbc，明确关闭其他两库；真实controller维持5c/0c66，并重新生成、比对真实完整服务指纹。

后续由Root独占执行：A取main899现有118入口；B取此新入口及新Root bridge，两臂同helper94db、5c/0c66/5898/ebbc、Q32 ON和v6 snapshot，stats/Graph额外观察关闭。先B原smoke/实际加载身份与请求窗口，然后两臂同4K三短+完整warm+单score、16K完整warm+单score；固定正式CLI不变。计分缓存新增和tuning变动继续留证。若吞吐值得继续，再按原官方105与补充流程验收；不能因组件加速或旧main精度直接声称新服务通过。

FULL_DECODE_ONLY回放可能绕过Python validate；mixed的非Graph路径才可能在每次forward受益。无当前profile占比，且host/GPU可能重叠，所以不按42层放大48us预测模型收益。缺FINAL_STATS时沿原118保存false及观察缺口，已有请求成功与终末汇总不足分别报告。

2026-10-30 修正：冻结3328首次smoke在启动服务、设备检查前因两处 `git show` 路径仍指向118而退出（owner181596/start363700321，终态Z）。旧3处逆变换检查只覆盖import及本地Path，未覆盖运行时Git对象路径；旧40项CPU结果不能证明 `run()` 身份前置可达。此次只修正这两处路径，并将逆变换扩为5处。新增 `tools/check_biv150_native120_run_preflight_cpu.py` 用真实冻结源、四库、94db、完整初始v6 snapshot和按新源生成的临时CPU准入件执行实际 `run()`；仅在首次helper `idle()`处主动停止，禁止服务/设备调用。临时两旗仅用于CPU控制流测试，不生成Root批准文件。另分别注入两个旧路径验证拒绝，生产、原118和所有协议字节保持不变。
