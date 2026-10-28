# PR113 shared6k：独立数值与运行时容量结果摘要

c91 / 实际5cf完成原四case消费测试，Root独立重放175原件和完整数组，与作者报告逐字一致（eabf9148）。全部176硬FP64门、32 BF16最终proxy、96完整first48逐位对照、168 golden、24 epoch和两项host拒绝通过。Graph-B先于Eager-B，side及完整输入/cache只读检查通过。48项未舍入ideal诊断失败仍完整保留，没有放宽硬门。

范围限于N48/S4、live41/42、Q pitch2048/2560、29物理页/maxctx193。它准许进一步试验，不授完整步骤、42层、模型精度或吞吐资格。完整原始Root报告已随前置82db提交的direct-native-abba-admission目录保存，本目录只汇总结果。

独立运行时API在64线程/dynamic0下返回旧2ad 8KiB容量16块/SM、新5cf 6KiB容量21块/SM；numRegs仍100。Root对六份原件的CPU复核与作者逐字一致（356a1c28）。这是运行时容量，不是实测驻留、利用率或性能。

consumer-results.json和runtime-capacity-results.json只含实验参数、计数和校验摘要。完整数组、cache、日志及二进制原件保存在执行者本地，不随此提交公开；可用原gzip和manifest SHA核对版本。replay_consumer.py为实际f2bed脚本，保持原c079数学段逐字一致，需要相应原件和冻结c91工具树才能重放。

受测82db完整步骤独立比较旧2ad/新5cf：同44769页、Q/K/V pitch2560、live41/42、相同dc5 writer和reducer，warm128、计时64 replay、unroll8、ABBA顺序。该试验结果待独立核实，不从旧TL比值推算新收益。
