# BI-V150 已验收服务：复现条件与入口

本文记录 PR117 / main `56f88737fcf3b940760b9ca3027a925cd5a7d77d` 的受测配置。完整成绩、精度、不同预热协议和原负结果见[验收记录](evidence/native117-model-acceptance-20261029/README.md)。PR118尚未完成双档验收，不能把它的单档成绩当成主干默认成绩。

## 功能与环境

比赛框架和 FlagGems 保持指定版本。受测环境是 BI-V150、CoreX 4.5、Torch 2.10.0（CXX11 ABI=false）及已经固定的 Triton/CoreX 后端；模型为 `/workspace/MiniCPM5-2B`，精度数据为 `/workspace/evalscope-datasets/math_500`。

Encoded K-only 服务、其 Q32 hoist 和 NativeK8 分别默认关闭。受测控制器显式打开这些功能，并使用5898库；compact保持关闭。原 W8A8 和 stock GQA 默认路径仍依赖成品库，不会自动生成全部受测配置。不要给当前主干对照加 `--baseline`，该开关会关闭已经接受的 encoded 功能。

源码以独立干净checkout为准，必须核对实际导入路径。测试前确认GPU和端口空闲；每个运行使用新的输出目录和私有缓存，不能复用其他任务的活动缓存或停止别人的服务。

## 必需的成品

下列SHA来自已审运行。仓库普通源码文件不包含可直接部署的这些 `.so` / `.db` 文件；需要在目标环境定位、保存并校验成品。初始缓存必须是v6快照，不得用计分后改变的缓存替换。

| 依赖 | SHA256 |
| --- | --- |
| W8A8 library | `5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe` |
| stock NativeGQA library | `0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8` |
| PR117 NativeK8 library | `5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b` |
| 初始v6 `TunedConfig_iluvatar_triton_3_2.db` | `5764f125233211d8ed8369f0bcdc6a08a58a00c9a3470bfaa36be0af7665ca55` |
| 初始v6 `TunedConfig_iluvatar_triton_3_6.db` | `2c5aa448065453188f1ff1794d8e929bacb63dcad84a600e086ae05165821314` |

构建入口分别是 `tools/build_biv150_native_w8a8.py`、`tools/build_biv150_native_gqa.py`、`tools/compile_biv150_native_k8_klookahead.py`。重建不保证得到相同ELF SHA；不同成品需要重新检查实际代码与数值，不能只改校验常量继承资格。`tools/snapshot_biv150_tuning_cache.py` 也不会把任意当前缓存变成受测v6。

## 从已共享源码恢复原控制器和 helper

以下仅在CPU读取已共享源码，输出到一个全新目录；不会启动服务或GPU。将两个绝对路径替换成实际源码目录和新的准备目录。控制器固定为2cfc版本，不开启统计RPC；helper只读取已共享归档的唯一指定成员，不解包其他内容。

```sh
python - /absolute/source /absolute/new-preparation-dir <<'PYCODE'
from pathlib import Path
import hashlib, importlib.util, sys, tarfile
repo, out = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location(
    'controller_builder', repo / 'tools/build_biv150_runtime_stats_controller.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
controller = builder.inverse((repo / 'tools/run_biv150_native117_stats_service.py').read_bytes())
with tarfile.open(repo / 'docs/evidence/konly-encoded-service-20261027/smoke-records-v1/encoded-smoke-records-v1.tar.gz') as archive:
    helper = archive.extractfile('root/stdlib-helper-a48.py').read()
expected = {
    'controller.py': (controller, '2cfc74c98e366015fe7bf3d6227ad06f8c1bcf87a341b72bd68fbf42062dcb7c'),
    'helper.py': (helper, '94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'),
}
for raw, digest in expected.values():
    assert hashlib.sha256(raw).hexdigest() == digest
out.mkdir(parents=True, exist_ok=False)
for name, (raw, digest) in expected.items():
    with (out / name).open('xb') as stream:
        stream.write(raw)
    print(name, digest)
PYCODE
```

`tools/build_biv150_runtime_stats_controller.py` 的命令行是正向生成接口；上面直接调用已有 `inverse()` 恢复原受测文件，没有另造反向CLI。

## 运行参数如何对应验收

控制器需提供实际已校验的绝对路径和源码commit：`--source`、`--commit`、`--base`、`--out`、`--helper`、`--gate-library`、`--tuning-snapshot`、`--native-gqa-library`、`--native-gqa-sha256`、`--native-k8-library`、`--native-k8-sha256`。参数中的SHA须对应上表，不能用旧8b86 NativeK8库替换5898。

吞吐运行共同使用 `--run-service --q32-hoist --throughput --full-workload-warmup`：

- 4K追加 `--warmup-plan legacy --throughput-case 4096`：三次64请求/128输出短预热，一次256请求/1024输出完整预热，一次正式计分。
- 16K追加 `--warmup-plan full-only --throughput-case 16384`：一次128请求/1024输出完整预热，一次正式计分。

这描述当前无统计RPC的对照流程。PR117原16K验收使用已共享的 `run_biv150_native117_stats_service.py` 和其固定边界统计入口；需按该记录保留观察配置，不能冒称与2cfc原控制器逐字相同。当前主干复核16K为5207.00 tok/s，只是新的单轮对照，不覆盖原验收记录。

控制器固定的正式服务命令仍为：

```sh
vllm serve /workspace/MiniCPM5-2B --port 9031   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'   --served-model-name minicpm --gpu-memory-utilization 0.85   --max-model-len 131072
```

正式benchmark函数SHA为 `e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d`，并发64，输出1024；4K/16K分别256/128请求。预热不计入成绩。不得为追成绩修改正式CLI、`max_num_batched_tokens`或精度采样设置。只看到命令正常退出不足以验收，还要核对成功请求数、精度、TTFT和实际代码/库身份。
