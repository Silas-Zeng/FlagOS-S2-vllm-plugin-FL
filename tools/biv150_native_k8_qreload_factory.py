"""One-variable, byte-invertible Q reload variant of frozen 377/3eb native K8."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT='3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
MAIN='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
PINS={'biv150_native_k8_qk.cpp': 'bc6d9d3c289d49435f7598e6b5c2325cfc3f867ab915ab424ffb289bffb20b29', 'biv150_native_k8_qk.cu': 'bf84c82ff7e73684c3923ca8944c6564f37c21fb636134e106d0542f5ba47cd9', 'biv150_native_k8_codec.cu': '7b744ba7112083add31ccb49c35055d4b573d718df4343ccf9d10ab998abaad2', 'biv150_native_k8_codec.cuh': '37d5c7548c0301ee522f50fb0d4d2da53300676e71b03bb806c2d88779801d79'}
NAMESPACE='cjj_native_k8_qreload_probe'
LIBRARY=NAMESPACE+'.so'
NAMES={name:name.replace('native_k8_', 'native_k8_qreload_',1) for name in PINS}
SYMBOLS=(('corex_native_k8_gqa','corex_native_k8_qreload_gqa'),
         ('cjj_native_k8_qk_probe',NAMESPACE),
         ('cjj_launch_native_k8_codec','cjj_launch_native_k8_qreload_codec'),
         ('cjj_native_k8_codec_probe','cjj_native_k8_qreload_codec_probe'),
         ('cjj_k8_scale_rn32','cjj_k8_qreload_scale_rn32'),
         ('cjj_k8_code_rne','cjj_k8_qreload_code_rne'),
         ('"biv150_native_k8_codec.cuh"','"biv150_native_k8_qreload_codec.cuh"'))


def sha(raw):return hashlib.sha256(raw).hexdigest()

def originals():
    result={name:(ROOT/'tools/kernels'/name).read_text() for name in PINS}
    for name,source in result.items():
        if sha(source.encode())!=PINS[name]:raise ValueError('Frozen parent source changed: '+name)
    return result


def q_blocks():
    source=originals()['biv150_native_k8_qk.cu']
    start=source.index(' AI8 q[4];');end=source.index(' __shared__ __align__(16) unsigned stage[2048];',start)
    old=source[start:end]
    edits=(
      (' float values0[8],values1[8],peak0=0.0f,peak1=0.0f;', ' float peak0=0.0f,peak1=0.0f;'),
      ('  values0[k]=','  const float value0='),('  values1[k]=','  const float value1='),
      ('fabsf(values0[k])','fabsf(value0)'),('fabsf(values1[k])','fabsf(value1)'),
      ('   const unsigned i=2*k+component;',
       '   const unsigned i=2*k+component;\n'
       '   const auto* base=query+static_cast<std::int64_t>(seq)*q_token_stride;\n'
       '   const float value0=__bfloat162float(base[(h*8+2*g)*q_head_stride+16*i+col]);\n'
       '   const float value1=__bfloat162float(base[(h*8+2*g+1)*q_head_stride+16*i+col]);'),
      ('cjj_k8_code_rne(values0[i],input_scale0)','cjj_k8_code_rne(value0,input_scale0)'),
      ('cjj_k8_code_rne(values1[i],input_scale1)','cjj_k8_code_rne(value1,input_scale1)'))
    new=old
    for before,after in edits:
        if new.count(before)!=1:raise ValueError('Unique Q transform anchor: '+before)
        new=new.replace(before,after,1)
    return old,new


def build():
    result=originals();old,new=q_blocks()
    name='biv150_native_k8_qk.cu'
    if result[name].count(old)!=1:raise ValueError('Unique original Q block')
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
    old,new=q_blocks();name='biv150_native_k8_qk.cu'
    if reverted[name].count(new)!=1:raise ValueError('Unique exact transformed Q block required')
    reverted[name]=reverted[name].replace(new,old,1)
    for name,source in reverted.items():
        if sha(source.encode())!=PINS[name]:raise ValueError('Non-Q/symbol source changed: '+name)
    return reverted


def verify():
    sources=build();actual={name:(ROOT/'tools/kernels'/name).read_text() for name in sources}
    if actual!=sources:raise ValueError('Committed generated source differs')
    inverse(actual)
    return dict(parent=PARENT,team_main=MAIN,parent_source_sha256=PINS,
      variant_source_sha256={name:sha(text.encode()) for name,text in actual.items()},
      whole_source_inverse_byte_exact=True,original_377_and_3eb_files_unchanged=True,
      original_K_then_V_issue_order_source_unchanged=True,PV_QK_waits_ABI_segments_source_unchanged=True,
      namespace=NAMESPACE,library=LIBRARY,Q_readonly_required=True,
      requested_extra_Q_BF16_reads_per_entered_CTA=1024,requested_extra_Q_bytes_per_entered_CTA=2048,
      requested_extra_Q_bytes_full_64S2_or_32S4=524288,
      actual_Q_global_reload_verified=False,actual_live_range_or_register_reduction_verified=False,
      actual_compile_or_numeric_or_performance_verified=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.write:
        for name,text in build().items():(ROOT/'tools/kernels'/name).write_text(text)
    print(json.dumps(verify(),sort_keys=True,indent=2))
