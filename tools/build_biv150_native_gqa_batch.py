"""Default CPU source emission; compilation requires explicit Root opt-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from biv150_native_gqa_batch_factory import emit,render,CFLAGS,CUDAFLAGS,NAMESPACE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--emit',type=Path)
    p.add_argument('--root-build',type=Path);a=p.parse_args()
    if a.root_build is None:
        print(json.dumps(emit(a.emit) if a.emit else render()[1],indent=2));return
    directory=a.root_build.resolve();proof=json.loads((directory/'source-proof.json').read_text())
    current=render()[0]
    for name,text in current.items():
        data=(directory/name).read_bytes()
        assert data==text.encode() and hashlib.sha256(data).hexdigest()==proof['generated'][name]
    assert proof['cflags']==CFLAGS and proof['cuda_flags']==CUDAFLAGS
    import torch
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','7.1')
    library=load(name=NAMESPACE,sources=[str(directory/x) for x in ('batch.cpp','batch.cu')],
        build_directory=str(directory),extra_cflags=CFLAGS,extra_cuda_cflags=CUDAFLAGS,
        is_python_module=False,verbose=True)
    result=dict(source_proof_sha256=hashlib.sha256((directory/'source-proof.json').read_bytes()).hexdigest(),
        library=str(library),library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),torch_module=torch.__file__,
        kernel_launches=0,numeric_verified=False,performance_verified=False)
    (directory/'build-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
