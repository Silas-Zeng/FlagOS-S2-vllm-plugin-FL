"""仅构建独立 BI-V150 N30/S8 GQA producer；不重编 S2/S4 库。"""
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
parser.add_argument("--build-dir", type=Path, required=True)
args = parser.parse_args()
args.build_dir.mkdir(parents=True, exist_ok=True)
sources = [root / "csrc/iluvatar/native_gqa_s8.cpp",
           root / "csrc/iluvatar/native_gqa_s8.cu"]
result = load(name="corex_native_gqa_s8_component", sources=list(map(str, sources)),
              build_directory=str(args.build_dir),
              extra_cflags=["-O3", "-std=c++17"],
              extra_cuda_cflags=["-O3", "-std=c++17", "-D__ILUVATAR__", "-D__MR__",
                                 "-DCOREX_NATIVE_GQA_V_OVERLAP=1", "-x", "ivcore",
                                 "--cuda-gpu-arch=ivcore11", "--cuda-path=/usr/local/corex"],
              is_python_module=False, verbose=True)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
print(json.dumps({"library": str(result), "library_sha256": sha(result),
                  "source": {path.name: sha(path) for path in sources},
                  "grid": [30, 2, 8], "q_stride": [2048, 128, 1],
                  "overlap_v": 1, "note": "仅构建新 S8 SO；S2/S4 库不变"},
                 sort_keys=True), flush=True)
