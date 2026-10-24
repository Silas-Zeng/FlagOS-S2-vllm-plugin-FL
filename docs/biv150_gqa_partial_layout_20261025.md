# C03：GQA partial输出布局，独立准备阶段

状态：Draft，只有CPU地址覆盖/源码保持与语法检查；没有加入GPU队列、没有设备或模型收益。
生产基线main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`，GQA源SHA2596cc09…；生产源码未改。

固定当前producer与reducer以及shipping split选择，仅把partial vector缓存的物理布局
由[B,head,segment,dim]换为[B,segment,head,dim]，通过逻辑view的stride传入现有两核。
不增加transpose copy，仍保持builder16的相同容量。scalar max/sum保持原布局，因为
当前producer明确假定scalar segment stride=1；没有将该限制漏掉或未经证明改变scratch生命周期。
潜在收益来自producer写入与reducer读取布局共同变化，可能一端赢另一端输，必须计完整两核。
本轮没有实现跨核归约融合、persistent工作队列或更少scratch容量，不能冒充这些方向完成。

准备的设备矩阵为13个batch×4个KV长度×uniform/ragged=104记录，采用packed K/V与随机页表，
空长度/尾页、NaN scratch、改变query/长度的Graph重放。比较全输出逐位一致，并对少量实际行
做CPU FP64 attention参考，三轮交替充分预热的完整producer+reducer时长，不只测reducer。
CPU已证明32768个单序列地址双射、五个segment前缀合法无重叠、生产源码保持；它不证明设备收益。

用户新增71卡清单后，当前PR37模型验收、PR36真实权重/全输出oracle安排保持；
下一批优先AD02/AD03/S11真实编译图与S19等内存证据门。本项仅共享可审查代码准备，
待已有证据支持再加入串行设备队列，不并发或挤占现有任务。
