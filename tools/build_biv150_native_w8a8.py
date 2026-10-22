import json
import os
import argparse
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.1")
parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path)
args = parser.parse_args()
build = args.build_dir or Path(os.environ.get("BIV150_NATIVE_BUILD", str(root / "build-native-w8a8")))
build.mkdir(parents=True, exist_ok=True)
result = load(name="biv150_native_w8a8", sources=[str(root / "csrc/iluvatar/native_w8a8.cpp"), str(root / "csrc/iluvatar/native_w8a8.cu")],
              build_directory=str(build), extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-x", "ivcore"],
              is_python_module=False, verbose=True)
print(json.dumps({"library": str(result), "build": str(build)}), flush=True)
