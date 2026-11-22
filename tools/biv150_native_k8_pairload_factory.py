"""One-variable, byte-invertible K paired-load variant of frozen 377/3eb native K8."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT='3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
MAIN='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
PINS={'biv150_native_k8_qk.cpp': 'bc6d9d3c289d49435f7598e6b5c2325cfc3f867ab915ab424ffb289bffb20b29', 'biv150_native_k8_qk.cu': 'bf84c82ff7e73684c3923ca8944c6564f37c21fb636134e106d0542f5ba47cd9', 'biv150_native_k8_codec.cu': '7b744ba7112083add31ccb49c35055d4b573d718df4343ccf9d10ab998abaad2', 'biv150_native_k8_codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79'}
NAMESPACE='cjj_native_k8_pairload_probe'
LIBRARY=NAMESPACE+'.so'
NAMES={name:name.replace('native_k8_', 'native_k8_pairload_',1) for name in PINS}
SYMBOLS=(('corex_native_k8_gqa','corex_native_k8_pairload_gqa'),
         ('cjj_native_k8_qk_probe',NAMESPACE),
         ('cjj_launch_native_k8_codec','cjj_launch_native_k8_pairload_codec'),
         ('cjj_native_k8_codec_probe','cjj_native_k8_pairload_codec_probe'),
         ('cjj_k8_scale_rn32','cjj_k8_pairload_scale_rn32'),
         ('cjj_k8_code_rne','cjj_k8_pairload_code_rne'),
         ('"biv150_native_k8_codec.cuh"','"biv150_native_k8_pairload_codec.cuh"'))


def sha(raw):return hashlib.sha256(raw).hexdigest()

def originals():
    result={name:(ROOT/'tools/kernels'/name).read_text() for name in PINS}
    for name,source in result.items():
        if sha(source.encode())!=PINS[name]:raise ValueError('Frozen parent source changed: '+name)
    return result


def k_blocks():
    source=originals()['biv150_native_k8_qk.cu']
    start=source.index('__device__ __forceinline__ void issue_k_i8(')
    end=source.index('__device__ __forceinline__ int signed_i32_word',start)
    old=source[start:end]
    new='__device__ __forceinline__ void issue_k_i8(unsigned* stage,const signed char* key,int page,unsigned head,unsigned valid,unsigned lane){\n const auto* base=key+static_cast<std::int64_t>(page)*16384+head*128;\n struct WordPair{unsigned first,second;};\n static_assert(sizeof(WordPair)==8 && alignof(WordPair)==4,"Preserve original 4-byte source ABI");\n // Pair adjacent global words; shared destinations are independently swizzled.\n for(unsigned pair=lane;pair<256;pair+=64){\n  const unsigned word=2*pair,slot=word/32,dim=(word%32)*4;\n  WordPair bits{0,0};\n  if(slot<valid){\n   const auto* source=reinterpret_cast<const unsigned*>(base+slot*256+dim);\n   __builtin_memcpy(&bits,source,sizeof(bits));\n  }\n  stage[k_i8_index(slot,dim)/4]=bits.first;\n  stage[k_i8_index(slot,dim+4)/4]=bits.second;\n }\n}\n'
    return old,new


def build():
    result=originals();old,new=k_blocks()
    name='biv150_native_k8_qk.cu'
    if result[name].count(old)!=1:raise ValueError('Unique original K block')
    result[name]=result[name].replace(old,new,1)
    renamed={}
    for name,source in result.items():
        for before,after in SYMBOLS:
            if after in source:raise ValueError('Variant symbol already present')
            source=source.replace(before,after)
        renamed[NAMES[name]]=source
    if inverse(renamed)!=originals():raise AssertionError('Whole source inverse differs')
    return renamed


def inverse(result):
    if set(result)!=set(NAMES.values()):raise ValueError('Exact four variant source files required')
    reverted={}
    for name,target in NAMES.items():
        source=result[target]
        for before,after in reversed(SYMBOLS):source=source.replace(after,before)
        reverted[name]=source
    old,new=k_blocks();name='biv150_native_k8_qk.cu'
    if reverted[name].count(new)!=1:raise ValueError('Unique exact transformed K block required')
    reverted[name]=reverted[name].replace(new,old,1)
    for name,source in reverted.items():
        if sha(source.encode())!=PINS[name]:raise ValueError('Non-K/symbol source changed: '+name)
    return reverted


def verify():
    sources=build();actual={name:(ROOT/'tools/kernels'/name).read_text() for name in sources}
    if actual!=sources:raise ValueError('Committed generated source differs')
    inverse(actual)
    return dict(parent=PARENT,team_main=MAIN,parent_source_sha256=PINS,
      variant_source_sha256={name:sha(text.encode()) for name,text in actual.items()},
      whole_source_inverse_byte_exact=True,original_377_and_3eb_files_unchanged=True,
      Q_codec_PV_QK_waits_ABI_segments_source_unchanged=True,
      namespace=NAMESPACE,library=LIBRARY,K_source_alignment_bytes=4,
      source_word_reads_per_entered_full_CTA=512,pair_copies_per_entered_full_CTA=256,
      source_bytes_per_entered_full_CTA=2048,shared_word_stores_per_CTA=512,
      source_issue_groups_old=8,source_issue_groups_new=4,
      old_full_wave_request='2 slots * 32 lanes * 4 bytes = 256 bytes',
      new_full_wave_request='4 slots * 16 lanes * 8 bytes = 512 bytes',
      actual_vector_load_or_register_reduction_verified=False,
      actual_compile_or_numeric_or_performance_verified=False)



if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.write:
        for name,text in build().items():(ROOT/'tools/kernels'/name).write_text(text)
    print(json.dumps(verify(),sort_keys=True,indent=2))
