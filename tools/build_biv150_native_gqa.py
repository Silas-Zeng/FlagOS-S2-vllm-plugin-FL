"""在启动服务前构建 BI-V150 BF16 GQA；服务进程只加载成品，不执行 JIT。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MAX_JOBS", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.1")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", type=Path)
args = parser.parse_args()
build = args.build_dir or root / "build-native-gqa"
build.mkdir(parents=True, exist_ok=True)
sources = [root / "csrc/iluvatar/native_gqa_s2.cpp", root / "csrc/iluvatar/native_gqa_s2.cu"]
result = load(name="corex_native_gqa_s2_probe", sources=list(map(str, sources)),
              build_directory=str(build), extra_cflags=["-O3", "-std=c++17"],
              extra_cuda_cflags=["-O3", "-std=c++17", "-D__ILUVATAR__", "-D__MR__",
                                "-DCOREX_NATIVE_GQA_V_OVERLAP=0", "-x", "ivcore",
                                "--cuda-gpu-arch=ivcore11", "--cuda-path=/usr/local/corex"],
              is_python_module=False, verbose=True)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
print(json.dumps(dict(library=str(result), library_sha256=sha(result),
                      source={p.name: sha(p) for p in sources}, overlap_v=False,
                      note="仅构建成品；没有GPU数值或性能资格"), sort_keys=True), flush=True)
