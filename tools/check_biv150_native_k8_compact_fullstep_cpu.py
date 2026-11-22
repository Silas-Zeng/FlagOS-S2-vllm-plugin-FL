#!/usr/bin/env python3
"""Bounded address/adapter falsifiers; no GPU or full-sized cache allocation."""
import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import numpy as np
import probe_biv150_native_k8_compact_fullstep as c


def check(parent):
    old=c.prepare(parent);f=old.f
    source,proof=c.adapted_source();assert proof['inverse_byte_exact']
    assert 'support.encode_initial' not in source
    assert source.count("support.append_kv[(n,2)](nk,nv,slots,z['key'],dummy,z['value'],f.PAGES*16,False") == 1
    for text in ("graphs[arm].replay()",'side.wait_stream(torch.cuda.current_stream())',
                 'for arm in f.ORDER:', 'timed(timed_graph[arm].replay,f.WARM)',
                 'timed(timed_graph[arm].replay,f.REPLAYS)', 'gold[arm,p]=f.golden(cold,packed,controls[p],s,arm)'):
        assert text in source
    count=0;book=f.codebook()
    for start,pages in ((0,2),(1234,1),(34275,1)):
        cold,*aux=f.chunk_sources(start,pages);packed=f.encode_chunk(start,cold,*aux,book)
        mapped=c.compact_chunk(packed)
        assert mapped.shape == (pages,12544)
        assert np.array_equal(mapped[:,:4224],packed[:,:4224])
        assert np.array_equal(mapped[:,4352:],packed[:,8192:])
        assert np.all(mapped[:,4224:4352] == 165)
        for page in range(pages):
            for slot in (0,15):
                for head in (0,1):
                    offset=4352+slot*512+head*256
                    assert mapped[page,offset:offset+256].tobytes() == cold[page,1,slot,head].tobytes()
                    count+=1
    cold,packed,_=f.build_initial(6,2)
    control=f.control(2,33,2560,0);control['slots'][:]=[16,-1]
    changed=f.patched_chunk('k8',0,packed,control);mapped=c.compact_chunk(changed)
    for head in (0,1):
        offset=4352+head*256
        assert mapped[1,offset:offset+256].tobytes()==control['new_v'][0,head].tobytes()
        code,scale=f.quantize_v2([f.bf16_value(int(v)) for v in control['new_k'][0,head]])
        assert mapped[1,head*128:(head+1)*128].view(np.int8).tolist()==list(code)
        assert struct.unpack_from('<f',mapped[1],4096+head*4)[0]==scale
    assert np.array_equal(mapped[2:],c.compact_chunk(packed)[2:])
    assert np.all(mapped[:,4224:4352]==165)
    original=old.module(Path(parent)/'tools/biv150_konly_strided_writer_contract.py','cjj_compact_fullstep_cpu_contract')
    contract=c.compact_contract(original);V=contract.View
    key=V('bfloat16',(2,2,128),(256,128,1),0x100000,1536,256)
    value=V('bfloat16',(2,2,128),(256,128,1),0x200000,1536,256)
    slots=V('int64',(2,),(1,),0x300000,528,256)
    cache=V('uint8',(6*12544,),(1,),0x400000,6*12544+512,256)
    snapshot=contract.SlotSnapshot((16,-1),'component-owned-cpu-preupload',0)
    binding=contract.qualify(key,value,slots,cache,6,snapshot)
    assert binding['format']==c.FORMAT and binding['key_row_stride']==256
    refused=0
    for badcache,badslots in [
        (V('uint8',(6*16384,),(1,),0x400000,6*16384+512,256),snapshot),
        (cache,contract.SlotSnapshot((16,16),'component-owned-cpu-preupload',0)),
        (cache,contract.SlotSnapshot((96,-1),'component-owned-cpu-preupload',0)),
        (V('uint8',(6*12544,),(1,),0x100000,6*12544+512,256),snapshot)]:
        try:contract.qualify(key,value,slots,badcache,6,badslots)
        except ValueError:refused+=1
        else:raise AssertionError('Invalid compact writer binding accepted')
    assert original.PAGE==16384 and original.FORMAT!='cjj-konly-compact12544-rn32-rne-v1'
    with tempfile.TemporaryDirectory() as d:
        out=Path(d);paths={k:c.materialize(out,k) for k in c.FROZEN}
        assert all(c.sha(p.read_bytes())==c.FROZEN[k][1] for k,p in paths.items())
        views=old.module(paths['views'],'cjj_compact_fullstep_cpu_views')
        layer=views.View('bfloat16',(6,2,16,2,128),(6272,2176,256,128,1),1,0x400000,6*12544+512,128,0x400100,'cuda:0')
        writer=views.parameters(layer,'writer83',format_id=c.FORMAT)
        decode=views.parameters(layer,'decode79',format_id=c.FORMAT)
        assert {writer[k].pointer for k in ('code_full','scale_full','value_full')}=={0x400100}
        assert decode['value_cache_ptr'].pointer==0x400100+4352
    assert not {'torch','triton'}.intersection(sys.modules)
    return dict(CPU_only=True,parent_source=str(parent),plan=c.plan(),small_page_field_boundaries=count,
        append_bytes_and_reserved_padding_checked=True,writer_binding_negative_cases=refused,
        frozen_writer_views_materialized_exactly=True,original_contract_unchanged=True,
        GPU_executed=False,full_arena_built=False,device_numeric_verified=False,
        saved_array_consumer_replay_executed=False,timing_executed=False,precision_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    a=p.parse_args();print(json.dumps(check(a.parent_source),sort_keys=True,indent=2))
