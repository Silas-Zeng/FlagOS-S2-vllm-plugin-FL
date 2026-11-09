"""CPU source factory: private N48/N56 S4 wrapper; shipping source is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MAIN='b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd'
INTEGRATION='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
NAMESPACE='cjj_native_gqa_batch_s4_20261028'
SYMBOLS={
 'corex_native_gqa_producer_draft':'cjj_batch_gqa_producer_20261028',
 'launch_corex_native_gqa_s2_draft':'launch_cjj_batch_gqa_s2_unused_20261028',
 'launch_corex_native_gqa_s4_draft':'launch_cjj_batch_gqa_s4_20261028'}
CFLAGS=['-O3','-std=c++17']
CUDAFLAGS=['-O3','-std=c++17','-D__ILUVATAR__','-D__MR__','-DCOREX_NATIVE_GQA_V_OVERLAP=0',
           '-x','ivcore','--cuda-gpu-arch=ivcore11','--cuda-path=/usr/local/corex']


def sha(data):return hashlib.sha256(data).hexdigest()
def source(path):return subprocess.check_output(['git','-C',str(ROOT),'show',MAIN+':'+path]).decode()


def render():
    original=source('csrc/iluvatar/native_gqa_s2.cu');cuda=original
    for old,new in SYMBOLS.items():
        assert old in cuda and new not in cuda;cuda=cuda.replace(old,new)
    inverse=cuda
    for old,new in reversed(tuple(SYMBOLS.items())):inverse=inverse.replace(new,old)
    assert inverse==original
    full=source('csrc/iluvatar/native_gqa_s2.cpp')
    cpp=full[full.index('#include <cstdint>',full.index('TORCH_LIBRARY_IMPL')+1):]
    expected=cpp
    changes=[('launch_corex_native_gqa_s4_draft',SYMBOLS['launch_corex_native_gqa_s4_draft']),
      ('produce_s4','produce_batch_s4'),('corex_native_gqa_s4_probe',NAMESPACE),
      ('{32,16,128}','{rows,16,128}'),('{32}','{rows}'),('table.size(0)==32','table.size(0)==rows'),
      ('partial.size(0)>=32','partial.size(0)>=rows'),('t.size(0)>=32','t.size(0)>=rows'),
      ('table.data_ptr<int>(),32,','table.data_ptr<int>(),static_cast<int>(rows),')]
    for old,new in changes:
        assert old in cpp and new not in cpp
        cpp=cpp.replace(old,new)
    anchor=' TORCH_CHECK(q.is_cuda()'
    assert cpp.count(anchor)==1
    scope=' TORCH_CHECK(q.dim()==3);\n const auto rows=q.size(0);\n TORCH_CHECK(rows==48 || rows==56);\n'
    cpp=cpp.replace(anchor,scope+anchor)
    marker=' c10::cuda::CUDAGuard guard(q.device());'
    guards=''' // Additional diagnostic ABI guards; no producer arithmetic changes.
 TORCH_CHECK(std::isfinite(scale));
 for(const auto& t:{q,k,v,lengths,table,partial,maximum,summation})TORCH_CHECK(!t.requires_grad());
 TORCH_CHECK(reinterpret_cast<std::uintptr_t>(k.data_ptr())%64==0 && reinterpret_cast<std::uintptr_t>(v.data_ptr())%64==0);
 TORCH_CHECK((partial.stride(1)==512 || partial.stride(1)==2048) && partial.stride(0)==16*partial.stride(1));
 TORCH_CHECK((maximum.stride(1)==4 || maximum.stride(1)==16) && maximum.stride(0)==16*maximum.stride(1));
 TORCH_CHECK(partial.stride(1)==128*maximum.stride(1));
'''
    assert cpp.count(marker)==1;cpp=cpp.replace(marker,guards+marker)
    cpp='#include <cmath>\n'+cpp
    restored=cpp.removeprefix('#include <cmath>\n').replace(guards,'').replace(scope,'')
    for old,new in reversed(changes):restored=restored.replace(new,old)
    assert restored==expected
    return {'batch.cpp':cpp,'batch.cu':cuda},dict(main=MAIN,integration_main=INTEGRATION,namespace=NAMESPACE,rows=[48,56],segments=4,
      CUDA_inverse_byte_exact=True,CUDA_changes='three private symbol renames only',
      CPP_inverse_exact_after_scope_and_extra_ABI_guards=True,
      source_sha256={'cpp':sha(full.encode()),'cuda':sha(original.encode())},
      native32_64_unchanged=True,N40_shipping_segments=16,N40_new_coverage=False,
      S16_static_assert_still_rejects=True,service_activation=False,device_numeric_verified=False,
      performance_verified=False,frequency_known=False)


def emit(folder):
    data,proof=render();folder.mkdir(parents=True,exist_ok=False)
    for name,text in data.items():(folder/name).write_bytes(text.encode())
    proof['generated']={name:sha(text.encode()) for name,text in data.items()}
    proof['cflags']=CFLAGS;proof['cuda_flags']=CUDAFLAGS
    (folder/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--emit',type=Path);a=p.parse_args()
    print(json.dumps(emit(a.emit) if a.emit else render()[1],indent=2))
