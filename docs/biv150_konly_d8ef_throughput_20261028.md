# d8ef 原始精度与完整吞吐记录

实际受测源d8ef055；证据head保留相同生产实现1cfc3334。固定官方CLI、相同5c/aace/初始v6，三短warm+完整几何warm后各单score，未修改max-num-batched-tokens或并发64。

| 场景 | total tok/s | mean TTFT ms | 成功/失败 |
| --- | ---: | ---: | --- |
| 4K | 5799.656646 | 3520.517340 | 256/0 |
| 16K | 4919.028536 | 102522.337130 | 128/0 |

同源原Level3完整精度102/105，105个stop，无空final/提取或请求错误；两档原性能工作门和TTFT通过。相对本机b3正式5655.19/4808.18的单点差异+2.55%/+2.31%；4K稳定增益未建立，继续Draft不合。main完整warm6449.65→score5655.19，本候选6467.85→5799.66均下降，不能拿低main分数宣称稳定收益。16K计分无新metadata/selected SQLite变化；4K仅9个新index、无新kernel JSON，SQLite聚合hash变化不证明GPU autotune耗时。

64件原始记录2,802,069B的压缩包、逐件SHA及Root独立审计在同名evidence目录。生成缓存排除并保留实例，未重跑模型。实际worker指纹2条、writer/reader选择12事件，encoded未选择stock BF16 native；事件不证明全部请求覆盖。PR93新C++ K8及PR91容量回收均未包含在该服务资格中。
