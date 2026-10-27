#!/usr/bin/env python3
"""Bounded stdlib address/codec/source checks for the new compact device entry."""
import ast
import json
from pathlib import Path
import struct
import sys
from biv150_konly_compact_device_contract import ROOT,BASE,PARENT,FORMAT,PAGE,GUARD,CASES,refs,append,mapping,load,plan,bootstrap_receipt


def run():
    import types
    package=types.ModuleType('vllm_fl');package.__file__=str(ROOT/'vllm_fl/__init__.py')
    assert bootstrap_receipt({'vllm_fl':package},{})['package_imported']
    assert not bootstrap_receipt({}, {})['package_imported']
    service=types.ModuleType('vllm_fl.patches.biv150_konly_compact_service')
    service.__file__=str(ROOT/'vllm_fl/patches/biv150_konly_compact_service.py');service._STATE=None
    bootstrap_receipt({'vllm_fl':package,service.__name__:service},{})
    rejections=0
    for modules,env in [({'vllm_fl':types.SimpleNamespace(__file__='/tmp/wrong-package/__init__.py')},{}),
                        ({'vllm_fl':package},{'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1'}),
                        ({'vllm_fl':package},{'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1'}),
                        ({'vllm_fl':package},{'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'})]:
        try:bootstrap_receipt(modules,env)
        except AssertionError:rejections+=1
        else:raise AssertionError('bad bootstrap accepted')
    service._STATE=object()
    try:bootstrap_receipt({service.__name__:service},{})
    except AssertionError:rejections+=1
    else:raise AssertionError('active private service accepted')
    assert rejections==5
    proof=plan();reference=refs();views=load(ROOT/'vllm_fl/patches/biv150_konly_compact_views.py','compact_real_views_cpu')
    checks=0
    for pages in (1,3,2**30+7):
        base=2**40;offset=GUARD//2
        layer=views.View('bfloat16',(pages,2,16,2,128),(6272,2176,256,128,1),19,base,pages*PAGE+2*GUARD,
                         offset,base+GUARD,'cuda:0')
        for role in views.ROLES:
            p=views.parameters(layer,role,format_id=FORMAT);assert views.validate_binding(layer,role,p,format_id=FORMAT)
            for page in {0,pages-1}:
                for slot in (0,15):
                    for head in (0,1):
                        for dim in (0,127):
                            actual=views.element_addresses(layer,role,page,slot,head,dim)
                            expected=(base+GUARD+page*PAGE+slot*256+head*128+dim,
                                      base+GUARD+page*PAGE+4096+4*(slot*2+head),
                                      base+GUARD+page*PAGE+4352+2*(slot*256+head*128+dim))
                            assert actual==expected;checks+=1
            # Old physical strides / offset scale subviews must reject.
            key='code_full' if role=='writer83' else 'key_cache_ptr';original=p[key]
            from dataclasses import replace
            bad=dict(p);bad[key]=replace(original,pointer=original.pointer+16)
            try:views.validate_binding(layer,role,bad,format_id=FORMAT)
            except ValueError:pass
            else:raise AssertionError('wrong pointer accepted')
    vectors=reference['codec_vectors']();assert len(vectors)==141
    code=[]
    for v in vectors:
        c,s=reference['quantize_v2'](v);assert len(c)==128 and all(-127<=x<=127 for x in c) and s>0;code.append((c,s))
    keys=[[vectors[t],vectors[t+1]] for t in range(4)]+[[[float('nan')]*128]*2 for _ in range(4)]
    values=keys;slots=[0,15,16,47,-1,48,2**63-1,-2**63];before=bytes([0xa5])*(3*PAGE)
    result=append(before,slots,keys,values,reference['quantize_v2'])
    allowed=set()
    for t,s in enumerate(slots[:4]):
        page,col=divmod(s,16)
        for h in (0,1):
            c,scale=code[t+h];k=page*PAGE+col*256+h*128;v=page*PAGE+4352+col*512+h*256;sc=page*PAGE+4096+4*(col*2+h)
            assert result[k:k+128]==bytes(x&255 for x in c) and result[sc:sc+4]==struct.pack('<f',scale)
            assert result[v:v+256]==b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',x))[0]>>16) for x in values[t][h])
            allowed.update(range(k,k+128));allowed.update(range(sc,sc+4));allowed.update(range(v,v+256))
    assert all(a==b for i,(a,b) in enumerate(zip(before,result)) if i not in allowed)
    for name,(batch,length,segments) in CASES.items():
        tab,pages=mapping(batch,length);flat=[x for row in tab for x in row]
        assert len(set(flat))==len(flat)==batch*((length+15)//16) and max(flat)<pages
        assert (batch,segments) in ((64,2),(32,4))
        # All three control geometries and empty/full/tail append mappings.
        for n in (0,1,15,16,17,63,100,length-1,length):
            for row in tab:
                if n:assert 0<=row[(n-1)//16]*16+(n-1)%16<pages*16
    source=(ROOT/'tools/bench_biv150_konly_compact_device.py').read_text();tree=ast.parse(source)
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.split('.')[0] in ('torch','numpy','triton','vllm') for a in n.names) for n in tree.body)
    # Syntax-check every new source only. Never import the JIT probe/kernel here.
    for p in [ROOT/'tools/bench_biv150_konly_compact_device.py',ROOT/'tools/kernels/biv150_compact_codec_probe.py']:
        ast.parse(p.read_text())
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    run_node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    # Exact record/witness call must precede the unchanged FP64 numerical assert.
    statements=list(ast.walk(run_node))
    gate=next(n for n in statements if isinstance(n,ast.Assert) and 'baseline_hard_passed' in ast.unparse(n))
    append_row=next(n for n in statements if isinstance(n,ast.Expr) and "record['numeric'].append" in ast.unparse(n))
    npz=next(n for n in statements if isinstance(n,ast.Expr) and 'np.savez_compressed(witness' in ast.unparse(n))
    assert npz.lineno<append_row.lineno<gate.lineno

    return dict(passed=True,CPU_only=True,base91=BASE,parent89=PARENT,physical_page_bytes=PAGE,
                independent_byte_addresses=checks,nonzero_storage_offsets=True,large_I64_page_indices=True,
                codec_vectors=141,real_package_bootstrap_allowed_with_exact_source=True,bootstrap_rejections=5,witness_saved_before_numeric_assert=True,complete_writer_edge_bytes=True,invalid_nan_sources_skipped=True,
                reserved_and_unmapped_bytes_exact=True,source_inverse=proof['source_inverse'],
                GPU_or_JIT_executed=False,service_capacity_or_performance_verified=False,
                actual_numeric_Graph_side_typed_ABI_and_loaded_resources_pending=True,old_Q32_hoist_still_rejected=True)

if __name__=='__main__':print(json.dumps(run(),indent=2))
