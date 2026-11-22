# Head baseline：逻辑表名与 SQLite 物理表名修正

父提交 d29b 的首次 R31 尝试在 factory 的 prelaunch 身份检查拒绝，尚未调用底层 head kernel，cases/timings 无数值结果。该过程曾初始化设备、准备模型权重/输入；“没有 head kernel”不等于“没有任何 CUDA 活动”。失败保持 failed，不转为通过。原状态和后续诊断原件仅在本地保存；完整 SHA 索引见 `evidence/head-temperature-logical-table-20261029/source-index.json`。

实际 selector 诊断 owner80529/start357750095 同启动时间 Z，completed=true/errors=[]，head_calls=0/autotune_called=false，源私有 DB 前后 SHA 相同。原 worker 至实际 linear module 的 import/patch 次序不变，只复制已结束测试的私有 DB 作诊断。该结果只观察 selector，不接收未知配置、不运行 head 或模型。

## 命名闭环

- 实际 LibTuner 逻辑名是 `linear_kernel_91df7045e2de13a4c456e8d13079eb7c`；keys 为 M/N/K、NORMAL、FlagTune=false，benchmark protocol 为 `(triton_do_bench,5,10)`。
- 当前 cache_key `64dc42fe84bd8451deb282a291c7675376208ac69e413d06c8e7f1e67445e85d` 与 configs_hash `13bf3aa8ea3bab9ab34ca2c4bf7b18a0` 的拼接 MD5 是 `91df7045e2de13a4c456e8d13079eb7c`。
- 已核 SDK libentry `414e8a3f...`：ConfigCache.get 调 model.get_config(self.table_name,key)。保存的 SQL model 源 SHA `6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7`：get_config 先构建 key_0…key_6 字典，get_sql_model 将字典列名拼接取 MD5，追加到逻辑名后。
- `md5('key_0key_1key_2key_3key_4key_5key_6')` 恰为 `0f015d35c97293819201f206510edb40`，故真实物理表是 `linear_kernel_91df7045e2de13a4c456e8d13079eb7c-0f015d35c97293819201f206510edb40`。当前诊断确有该物理表及原 M32/M64 配置。这不是已证 cache miss；此前把后缀差异猜作 SDK 换表的解释在此更正。

首次诊断没有单独报告实际导入 SQL model 文件的 SHA；CPU 推导使用已有安装源码归档，且与真实物理名吻合。修正没有以该离线推导替代实际 cache.get：下一设备进程仍必须真实读到相同 tile，否则原门继续拒绝。

## 最小修正与验收

factory 仅将 CONFIG_TABLE 从物理名改为精确逻辑名。逆替换该一个字面量即可逐字还原 d29 整个 factory；cache.get、NORMAL/FlagTune/key/config/hook、物理 backing、grid、原后置检查全不变。SQL 只读查询仍查原精确物理名，未改缓存/配置池/benchmark/timing/数学/十个数值门。

四行数正例及 52 个负例通过，新增四个“把物理名当逻辑 selector 名”的拒绝。已有全部 tail canary 与 AST 逆变换检查保留；worker、timing、compile 入口、生产和 GPU 源不变。SQL 命名的 CPU 检查从 SHA 固定的真实 get_sql_model 抽取 full_name 表达式求值，不导入 GPU 框架或重造数据库实现。

串行复跑命令沿 `biv150_head_temperature_baseline_backing_20261029.md` 三条 CPU 入口；第三条额外只读取 `--libentry-source` 同目录的 `models/sql.py`，SHA 不符即拒绝。精选结果：source-cpu `0548114c...`、component-cpu `807f9d57...`、kernel-cpu `4402d155...`；完整 SHA 在索引中。原 d29 报告、失败状态和诊断原件不改写；新设备数值/资源/完整费用、服务/105/吞吐资格仍待。
