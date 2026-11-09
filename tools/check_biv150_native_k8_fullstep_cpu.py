#!/usr/bin/env python3
"""Small CPU pack/reference/geometry falsifiers; never constructs the full arena."""
import ast
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np
import biv150_native_k8_fullstep_fixture as f
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_native_k8_consumer_receipt import require_consumer


def check():
    f.pins();book=f.codebook();vectors=0
    for start,count in ((0,2),(1234,1),(34275,1)):
        cold,*aux=f.chunk_sources(start,count);packed=f.encode_chunk(start,cold,*aux,book)
        for p in range(count):
            for slot in range(16):
                for head in range(2):
                    logical=(start+p)*16+slot
                    assert cold[p,0,slot,head].tolist()==f.vector(1,0,logical,head)
                    assert cold[p,1,slot,head].tolist()==f.vector(2,0,logical,head)
                    code,scale=quantize_v2([f.bf16_value(int(x)) for x in cold[p,0,slot,head]])
                    assert tuple(packed[p,slot*256+head*128:slot*256+(head+1)*128].view(np.int8))==code
                    assert struct.unpack_from('<f',packed[p],4096+slot*8+head*4)[0]==scale
                    assert packed[p,8192+slot*512+head*256:8192+slot*512+(head+1)*256].tobytes()==cold[p,1,slot,head].tobytes()
                    vectors+=1
        assert bool((packed[:,4224:8192]==165).all())
    cold,packed,_=f.build_initial(6,2);old_cold=cold.tobytes();old_packed=packed.tobytes()
    c=f.control(2,33,2560,0);c['table'][:]=-1;c['table'][0,:3]=[0,1,2];c['table'][1,:3]=[3,4,5]
    c['lengths'][:]=[17,33];c['slots'][:]=[1*16,5*16]
    gold={}
    for arm,initial in [('bf16',cold),('k8',packed)]:
        changed=f.patched_chunk(arm,0,initial,c)
        for row,physical in enumerate(c['slots']):
            page,slot=divmod(int(physical),16)
            assert changed[page,8192+slot*512:8192+(slot+1)*512].tobytes()==c['new_v'][row].tobytes()
        assert cold.tobytes()==old_cold and packed.tobytes()==old_packed
        for segments in (2,4):
            g=f.golden(cold,packed,c,segments,arm);gold[arm,segments]=g
            assert f.metrics(g['output'],g['output'])['passed']
            for field in ('partial','maximum','summation'):
                mask=np.broadcast_to(g['active'][:,None,:,None] if field=='partial' else g['active'][:,None,:],g[field].shape)
                assert bool(np.isfinite(g[field][mask]).all()) and bool(np.isnan(g[field][~mask]).all())
                bad=g[field][mask].copy();bad[0]+=1;assert not f.metrics(bad,g[field][mask])['passed']
    geometries=[]
    for name,(n,L,S,pitch) in f.CASES.items():
        control=f.control(n,L,pitch,0);active=control['table'][:,:L//16].reshape(-1)
        assert len(set(active.tolist()))==n*(L//16) and int(active.min())>=0 and int(active.max())<f.PAGES
        assert control['table'].shape==(n,8192) and tuple(control['new_k'].strides)==(512,256,2)
        assert np.all(control['slots']>=0)
        geometries.append(dict(case=name,active_pages=len(active),table_sha256=f.sha(control['table'].tobytes()),builder_width=16,segments=S,q_pitch=pitch))
    with tempfile.TemporaryDirectory(prefix='cjj-fullstep-precondition-') as d:
        path=Path(d)/'failed-consumer.json';path.write_text(json.dumps(dict(reference_version='pagewise_BF16_P_v2',thresholds_changed=False,final_output_hard_reference='unchanged ideal quantized-score FP64',completed=False)))
        try:require_consumer(path,f.sha(path.read_bytes()))
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Unqualified consumer accepted')
    source=(f.ROOT/'tools/probe_biv150_native_k8_fullstep.py').read_text();ast.parse(source)
    assert source.index('pre=require_consumer')<source.index('import torch')
    assert source.index("graphs[arm].replay()")<source.index("'new_eager_or_repeat'")
    assert source.index("'actual83 typed ABI'")<source.index("rec['warm']")
    assert "(n,16,16,128)" in source and 'wide.narrow(2,0,s)' in source
    result=subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_fullstep.py')],capture_output=True,text=True,check=True);assert json.loads(result.stdout)==json.loads(json.dumps(f.plan()))
    assert not {'torch','triton'}.intersection(sys.modules)
    return dict(plan=f.plan(),scalar_full_D128_codec_vectors_verified=vectors,small_reference_variants=4,
      geometries=geometries,failed_consumer_refused=True,full_arena_built=False,device_executed=False,precision_gate=False)

if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
