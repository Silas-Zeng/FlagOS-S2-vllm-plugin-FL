# 条件研究卡：V8 页内固定保留前七个 slot 的 BF16 V

仅在 PR73 `4a1f7ddd38e4bcade30e702faba2e9a7cdb5441d` 的诊断证明完整组件
确有吞吐余量、同时精度需要恢复时，再考虑这个混合精度布局。本卡没有 kernel、
服务注册或新 GPU 队列，不承诺吞吐或精度；PR66 模型源码仍冻结 e08。SSH 不可用
期间，现有服务/owner 状态未知，本轮未访问或改变目标机。

## 先前工作与边界

- 旧《天数BI-V150_算子优化调研清单_2026-10-24.md》E02 已提出“最近尾段 BF16、
  历史完整块低比特”，E03 是低比特解包进入矩阵布局，E04/E10 已涵盖 scale 折叠
  与页内联合布局。本卡属于这些方向下的具体权衡，不称为新方向或创新。
- [PR55 冻结文档](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/4767ca134d84e42533eb6d68eb04dab7d3c929fd/docs/biv150_int8_pv_20261026.md)
  已有相同 K/code/FP32-scale 基础布局，但 padding 全未使用；其 P8×V8/K32
  integer-dot 完整组件已慢，不能复用为性能依据。
- [PR73 冻结文档](https://github.com/Silas-Zeng/FlagOS-S2-vllm-plugin-FL/blob/4a1f7ddd38e4bcade30e702faba2e9a7cdb5441d/docs/biv150_v8_bf16p_20261027.md)
  与对应 kernel/harness 使用 `BF16(P*scale) × BF16(code)`，12416 后仍是 padding。
  本轮检查上述源码/文档未见“在 padding 保存每页首7个完整 BF16 V”的同一精确布局；
  这不是全面文献检索或新颖性判断。固定页首7个不等于 E02 的最近尾段：历史页首
  也永久保留，新追加的 slot7..15 仍量化。

旧调研清单本地 SHA256 为 `660e378cda0596252d4f8a9fd18396dd668cfd8af053ac91656ba88bf8a6a758`；
扩展调研 SHA256 为 `6e9c813fd2eee225e3fd4a88a679e0b79b1d7db0fcefa102c154a05e77ea65e7`。
本轮只读复核，不修改这些旧调研或 PR55/73。

## 物理布局：16 KiB 配额不变

每页16slot、2 KV head、head_dim128。下表使用半开 byte 区间：

| 区间 | 字节 | 类型与内容 |
| --- | ---: | --- |
| [0,8192) | 8192 | 原 BF16 K |
| [8192,12288) | 4096 | 原 INT8 Vcode，保留完整16slot地址范围 |
| [12288,12416) | 128 | 原 FP32 scale，slot-major/head-minor |
| [12416,16000) | 3584 | 7×2×128 个原 BF16 V |
| [16000,16384) | 384 | padding |

page base 为 `p*16384`，slot=s、head=h、dim=d：
K 地址 `base+s*512+h*256+d*2`；code `base+8192+s*256+h*128+d`；
scale `base+12288+s*8+h*4`；saved `base+12416+s*512+h*256+d*2`，只允许 s<7。
saved 最后元素占[15998,16000)，区间和跨页均不重叠。
FP32 scale 地址满足4B对齐，BF16满足2B；saved 起点128B对齐、模256为128。
它不是256B对齐，不能从 dtype 正确推断宽加载无拆分。

若用 typed view，saved BF16 的 storage_offset 为6208，strides 为
`(8192,256,128,1)`、shape `(pages,7,2,128)`；scale FP32 offset3072、
strides `(4096,2,1)`；INT8 code offset8192、strides `(16384,256,128,1)`。
不同 dtype 的 offset/stride 单位不可混用。所有 reader/writer 必须同意此格式。
仍分配16384B/page；prototype 若保留原 BF16 control cache，则双份占用仍在。
没有回收物理容量，不能声称增加 KV slot/请求驻留数。

## 有用载荷、事务与写入要分别核算

完整页消费的理想有用读载荷是
`K8192 + saved3584 + code(9*256)2304 + scale(9*8)72 = 14152B`，
较 BF16 K+V16384B 少13.6230%，但较现 V8 的12416B 多1736B。
前提是三种 V 数据用独立 load mask：saved 仅 s<7、code/scale 仅 s>=7，
再各自与有效列 mask 相与。`where` 本身不使两个 load 分支惰性执行。
若仍读取完整code/scale再读saved，有用载荷已是16000B。
单个PV避免第二次dot，但双路masked load、where、地址运算和布局搬运仍可能增加
活跃寄存器/共享内存或降低占用；字节节省不能证明这些代价足够小。

按绝对地址取唯一 cache line 的**理想覆盖模型**：128B粒度为14208B（scale
要覆盖完整128B line），256B粒度为14336B（scale 与 saved 共用首个 line）。
如果把不同region分别按256B取整且共享line不能复用，则是14592B。
真实 head/program 分发、指令合并、cache命中、重复事务、写分配/RMW和尾页会改变
实际流量；这些数值不是天数事务粒度或 DRAM 实测。单看14152B最多只能形成
带宽假设，不能推导完整组件加速比。

steady append 的缓存写载荷有两个不同选择：

- 条件 writer：s<7 写原K512B+saved V512B=1024B/token，不写其死code/scale；
  s>=7 写K512B+code256B+scale8B=776B/token。完整页为14152B。
- 保留完整现 V8 codec、再额外保存7slot：`8192+4096+128+3584=16000B/page`，
  首7slot的quant/reduce工作也没有省掉。不能给它套用14152B写量。

两者还要读原新token K/V共1024B/token；原框架writer与诊断writer的时间不能等同。
现 V8 writer为776B/token；恢复精度付出读写字节和选择/布局成本。
初始转换若从已有 BF16 V 生成新页，源V至少读8192B/page，所有新目的V相关写入也
要计算；copy/双buffer/workspace另计。源BF16 V与新code/scale/saved区间重叠，
不能把当前双buffer bulk codec机械改成并发原地转换，避免覆盖尚未读取的源值。
页共享、复用、append、prefill消费、copy/offload和Graph都需同格式；encoded
字节中的 BF16/FP32解释可能为NaN，carry/copy需要保留原始bits。
若未来实施，原73的3968B padding不变检查必须改为末384B，并独立检查saved写入；
changed-input Graph poison也须覆盖saved，不能让旧的全V8 codec自洽检查掩盖漏写。

## 单个 BF16 PV 的代数与舍入边界

保持原 BF16 QK、FP32 softmax概率P、running_sum/max、segment/reducer不变，只令
`Vop=where(s<7,V_saved_BF16,code.toBF16)`，
`Pop=BF16(P*where(s<7,1,scale_FP32))`，再做一次原 BF16 PV。
分母仍累加原P，不能改成P*scale。两个where的V分支都显式为BF16；所有[-127,127]
整数都可精确转BF16。失效列的K/V/scale访问分别mask，失效P/V置零，不能让未初始化
scale或越界saved地址参与运算。

对有限FP32 P，乘1不改变P；保留slot的Pop恰为原路径`RN_BF16(P)`，Vop恰为原
BF16 V。这个论证是两个PV操作数的一致性，不是混合dot的累加顺序、GPU全输出
bitwise一致或模型通过保证。CPU只模拟有限标量RNE，未模拟设备FTZ、FP32 dot累加。

具体边界：P=0.501953125是0.5与0.50390625的正中间，ties-even舍入为0.5；
其上一个FP32步长`+2^-24`则舍入到0.50390625，首7slot与原路径均如此。
未保留slot仍有codec和P*scale舍入：某原BF16向量同时含2.0和0.5，scale=
FP32(2/127)=0.015748031437397003，0.5的code=32；上述P下原项=0.25，
新项=`BF16(FP32(P*scale))*32=0.25390625`，并不相等。
令a=RN_BF16(P)、b=RN_BF16(FP32(P*s))，其误差可分解为
`b*c-a*V = c*(b-P*s)+P*(s*c-V)+(P-a)*V`；非保留slot的两种近似仍在。
零vector的scale1/code0需沿用，极小scale、half-away codec tie、FP32 reciprocal
一ULP和设备下溢/FTZ边界也未由保留7slot解决。

通常从page slot0开始的初始<=7token，其全部有效V自然保留BF16；每页尾r<=7
也全部保留。r=8才出现首个量化slot7，r=16为7保留+9量化。对r有效token，
理想载荷为`512*r+512*min(r,7)+264*max(r-7,0)`；r<=7并无载荷节省。
这是通用物理slot规则，不按105题、benchmark名或已知输出分流。
7/16的token保真不代表7/16注意力质量或单调改善，概率可能集中在另九个slot，
混合累加也可能发生抵消。b8的唯一超阈误差尚无failed-row归因，不能说该布局会修复它。

## CPU 留证与进入条件

`python3 tools/check_biv150_v8_prefix7_research_cpu.py` 枚举布局/地址、17种尾页、
理想line覆盖、560个保留操作数例和一个非保留反例。结果在
`docs/evidence/prefill-multi2-20261027/v8-prefix7-research-cpu.json`，明确
`precision_gate=false / gpu_executed=false / kernel_implemented=false / queued=false`。
通过仅表示本卡的地址/代数成立。

先等4a原有诊断提供完整writer+consumer+reducer吞吐、未放宽的数值守卫和failed-row
证据；只有它证明有可交换的性能余量、确需恢复精度，才决定是否值得实施/测这个
变体。若无余量则不启动。当前不在测试队列，不新增105或组件任务，不修改服务参数。
