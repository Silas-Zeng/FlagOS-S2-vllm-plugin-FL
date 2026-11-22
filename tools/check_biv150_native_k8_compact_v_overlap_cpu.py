#!/usr/bin/env python3
"""Bounded source and shared-tail checks; no GPU or numerical qualification."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import biv150_native_k8_compact_v_overlap_factory as f
import compile_biv150_native_k8_compact_v_overlap as c


def check():
    proof=f.verify()
    source=f.path('qk.cu').read_text()
    body=source[source.index('__device__ __forceinline__ void consume_page('):
                source.index('\n}\n}\n#endif',source.index('__device__ __forceinline__ void consume_page('))]
    anchors=['issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();',
             'issue_v(dst,descriptor(value,page,head));', 'CI8 scores;',
             'nvcuda::wmma::mma_sync_tcu(scores,q[k],b,scores);',
             'A<PVShape> probability;probability[0]=packed;probability[1]=packed;',
             '__ivcorex_sl_waitcnt(8ULL);', 'if(valid<16)clear_v_tail(stage,valid,lane);',
             'BRow<PVShape> b;nvcuda::wmma::load_matrix_sync_tcu(b,stage+1024,n,0);',
             '__syncthreads(); // All shared fragment reads precede the next page overwrite.']
    positions=[body.index(a) for a in anchors]
    assert positions==sorted(positions) and len(set(positions))==len(positions)
    assert body.count('__ivcorex_sl_waitcnt(8ULL);')==1
    assert body.count('__syncthreads();')==3
    assert 'if(valid<16)clear_v_tail(stage,valid,lane);\n  __syncthreads();' in body
    # Exhaust the unchanged per-word V clear: odd cuts retain the valid low
    # half; QK uses only the disjoint 2048-byte K image until this late clear.
    kbytes=set()
    for slot in range(16):
        for dim in range(128):
            panel,within=divmod(dim,64);quartet,byte=divmod(within,4);bank,subslot=divmod(slot,4)
            kbytes.add(panel*1024+bank*256+(quartet%4)*64+((quartet//4)^bank)*16+subslot*4+byte)
    assert kbytes==set(range(2048))
    seen=set()
    for i in range(1024):
        slot=(i//128)*2;dim=i%128;panel,vc=divmod(dim,16);r=slot//2
        half=panel*256+(r//4)*128+((r*32+vc*2+(slot&1)+panel*64)&127)
        index=1024+half//2
        assert index not in seen;seen.add(index)
        assert not set(range(index*4,index*4+4))&kbytes
        for valid in range(17):
            word=0xBEEFCAFE
            got=0 if slot>=valid else word&0xffff if slot+1>=valid else word
            expected=(0xCAFE if slot<valid else 0)|((0xBEEF if slot+1<valid else 0)<<16)
            assert got==expected
    assert seen==set(range(1024,2048))
    # Whole-file inverse rejects any accidental arithmetic or synchronization edit.
    faults=[]
    for name,old,new in (
        ('missing_K_ready_barrier',' issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();',' issue_k_i8(stage,key,page,head,valid,lane);'),
        ('missing_late_wait','  __ivcorex_sl_waitcnt(8ULL);',''),
        ('tail_clear_change','if(valid<16)clear_v_tail(stage,valid,lane);','if(valid<15)clear_v_tail(stage,valid,lane);'),
        ('PV_FMA','running_l0=fmaf(running_l0,rho0,page_l0)','running_l0=running_l0*rho0+page_l0'),
        ('BF16_P','__float2bfloat16_rn(p0)','__float2bfloat16_rz(p0)'),
        ('K_page','page)*12544+head','page)*16384+head'),
        ('V_page','page)*6272+head','page)*8192+head'),
        ('scale_page','page)*3136+1024','page)*4096+1024'),
    ):
        assert source.count(old)==1
        try:f.inverse(source.replace(old,new),'qk.cu')
        except AssertionError:faults.append(name)
        else:raise AssertionError('Unexpected mutation accepted: '+name)
    host=f.path('qk.cpp').read_text()
    for name,old,new in (
        ('host_unapproved_N40','segments==2?64:32','segments==2?64:40'),
        ('host_old_V_offset','kb+4352','kb+8192'),
        ('host_old_span','kb+pages*12544','kb+pages*16384'),
    ):
        assert host.count(old)==1
        try:f.inverse(host.replace(old,new),'qk.cpp')
        except AssertionError:faults.append(name)
        else:raise AssertionError('Unexpected host mutation accepted: '+name)
    # The build body is exactly the original compile-only body, except for the
    # independent extension name. Compiler flags differ only in macro 0 -> 1.
    text=(f.ROOT/'tools/compile_biv150_native_k8_compact_v_overlap.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='compile_only')
    fn=ast.get_source_segment(text,node).replace("name='cjj_native_k8_compact_v_overlap_probe'","name='cjj_native_k8_qk_probe'")
    assert f.sha(fn.encode())=='8c8842c393f2799e5c446b45e6e864a14f331671b38e8e47374adfd18fec7bbb'
    assert c.CUDA_FLAGS==['-O3','-std=c++17','-D__ILUVATAR__','-D__MR__',
                         '-DCOREX_NATIVE_GQA_V_OVERLAP=1','-x','ivcore',
                         '--cuda-gpu-arch=ivcore11','--cuda-path=/usr/local/corex']
    default=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/compile_biv150_native_k8_compact_v_overlap.py')],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout)==c.cpu_plan()
    rejected=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/compile_biv150_native_k8_compact_v_overlap.py'),
                             '--compile-only','--out','/private/tmp/cjj-agent-unlaunched-compact-v-overlap-cpu-negative'],capture_output=True,text=True)
    assert rejected.returncode==2 and 'Root must confirm prior owner terminal' in rejected.stderr
    assert not Path('/private/tmp/cjj-agent-unlaunched-compact-v-overlap-cpu-negative').exists()
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(source=proof,wait_order_checked=True,K_V_shared_regions_disjoint=True,
                V_tail_valid_counts=list(range(17)),V_unique_word_checks=1024*17,
                rejected_source_mutants=faults,parent_compile_body_inverse=True,
                default_CPU_only=True,missing_Root_ack_rejected=True,
                compiled=False,GPU_executed=False,precision_gate=False)


if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
