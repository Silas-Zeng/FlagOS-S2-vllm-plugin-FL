"""Build an isolated BF16 BLAS probe; never install or patch the service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as port:
        assert port.connect_ex(('127.0.0.1', 9031)) != 0, 'Service is running'
    assert 'No running processes found' in subprocess.check_output(
        ['/usr/local/corex/bin/ixsmi'], text=True), 'GPU is busy'
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / 'csrc/iluvatar/native_bf16_probe.cpp'
    os.environ.setdefault('MAX_JOBS', '2')
    from torch.utils.cpp_extension import load
    library = load(name='biv150_native_bf16_probe', sources=[str(source)],
        build_directory=str(args.out.resolve()),
        extra_include_paths=['/usr/local/corex/include'],
        extra_cflags=['-O3', '-std=c++17'],
        extra_ldflags=['-L/usr/local/corex/lib64', '-lcublas', '-ltorch_cuda', '-lc10_cuda'],
        with_cuda=False, is_python_module=False, verbose=True)
    record = dict(library=str(library), source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        production_service_installed=False)
    (args.out/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
