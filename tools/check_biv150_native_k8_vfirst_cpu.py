#!/usr/bin/env python3
"""No frameworks: prove one source-order edit plus isolated names, not scheduling."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import biv150_native_k8_vfirst_factory as f

def run():
    proof=f.verify();text=f.path('qk.cu').read_text();rejected=[]
    for name,bad in [('wait_count',text.replace('__ivcorex_sl_waitcnt(8ULL)','__ivcorex_sl_waitcnt(7ULL)')),
                     ('P_round',text.replace('__float2bfloat16_rn(p0)','__float2bfloat16_rn(p0*.99f)')),
                     ('scale_base',text.replace('page)*4096+1024+col','page)*4096+2048+col')),
                     ('restore_original_issue_order',text.replace(f.NEW,f.OLD)),
                     ('K_byte_layout',text.replace('panel*1024+bank*256','panel*1024+bank*252'))]:
        assert bad!=text
        try:f.inverse(bad,'qk.cu')
        except AssertionError:rejected.append(name)
        else:raise AssertionError('mutant accepted: '+name)
    adapters={kind:f.host_source(kind)[1] for kind in ('compile','numeric','fullstep')}
    for kind in adapters:ast.parse(f.host_source(kind)[0])
    import probe_biv150_native_k8_vfirst as frontend
    # A small, explicitly simulated compiler binding tests CPU admission only;
    # no fake artifact is used for compilation, numerical or performance claims.
    with tempfile.TemporaryDirectory(prefix='cjj-vfirst-cpu-') as directory:
        d=Path(directory);rows=[]
        for name in ('biv150_native_k8_vfirst_qk.ll','biv150_native_k8_vfirst_qk.s'):
            (d/name).write_bytes(b'CPU metadata fixture, not target IR\n')
            rows.append(dict(relative=name,bytes=(d/name).stat().st_size,sha256=f.sha((d/name).read_bytes())))
        for local in [f.path(k) for k in f.FILES]+[f.ROOT/'tools/kernels/biv150_native_k8_codec.cuh']:
            (d/local.name).write_bytes(local.read_bytes())
            rows.append(dict(relative=local.name,bytes=local.stat().st_size,sha256=f.sha(local.read_bytes())))
        library=d/'fixture.so';library.write_bytes(b'CPU fixture, never loaded')
        summary=dict(completed=True,compile_passed=True,errors=[],vfirst_source_proof=f.verify(),
                     artifact_manifest=rows,library=dict(sha256=f.sha(library.read_bytes())),
                     explicit_cuda_flags=frontend.load_module('compile_biv150_native_k8_qk.py').CUDA_FLAGS,
                     **{k:0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations')})
        path=d/'compile-summary.json';path.write_text(json.dumps(summary))
        review=frontend.review_template(path);rp=d/'review.json';rp.write_text(json.dumps(review))
        a=SimpleNamespace(root_owned_exclusive=True,compile_summary=path,assembly_review=rp,
                          assembly_review_sha256=f.sha(rp.read_bytes()),variant_library=library)
        def refused(name):
            try:frontend.admitted(a)
            except (AssertionError,KeyError):rejected.append(name)
            else:raise AssertionError('admission mutant accepted: '+name)
        refused('actual_schedule_unreviewed')
        for key in ('actual_V_issue_before_K_packing','actual_wait_barrier_tail_preserved','original_I8_QK_BF16_PV_and_codec_preserved'):review[key]=True
        review['notes']='CPU admission fixture only, never target qualification';rp.write_text(json.dumps(review));a.assembly_review_sha256=f.sha(rp.read_bytes())
        assert frontend.admitted(a)['variant_library_sha256']==summary['library']['sha256']
        a.root_owned_exclusive=False;refused('no_Root_exclusive');a.root_owned_exclusive=True
        review['qk_ASM_sha256']='0'*64;rp.write_text(json.dumps(review));a.assembly_review_sha256=f.sha(rp.read_bytes());refused('different_actual_ASM')
        review['qk_ASM_sha256']=rows[1]['sha256'];rp.write_text(json.dumps(review));a.assembly_review_sha256='0'*64;refused('review_digest_mismatch')
        a.assembly_review_sha256=f.sha(rp.read_bytes());library.write_bytes(b'changed');refused('library_digest_mismatch')
    default=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_vfirst.py')],text=True,capture_output=True,check=True)
    assert json.loads(default.stdout)==frontend.plan()
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,source_proof=proof,host_adapters=adapters,rejected=rejected,
                actual_compilation_or_IR_or_device=False,precision_gate=False)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
