"""One dense reader fixture; original fullstep counter, codec and golden reused."""
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import biv150_native127_v8_fixture as small

PAGES=34276
COLS=8192
LENGTH=4096
CASES=((64,2,2048),)
CHUNK_PAGES=128
CACHE_BYTES=128+PAGES*16384+128


def layout():
    return dict(cases=CASES,pages=PAGES,page_bytes=16384,cache_bytes_per_arm=CACHE_BYTES,
      table_pitch=COLS,lengths=[LENGTH]*64,active_unique_pages=64*256,
      mapping='original3eb fixed_table seed20261028; B rotates each row by1',
      changed_B=['Q','table'],immutable_B=['cache','all64 lengths4096'],
      scratch='original consumer compactS2, not width16',writer=False,
      cache_evidence='all GPU cache bytes compared; reconstructible128-page chunk SHA, guards included',
      cache_raw_bytes_not_archived_on_success=2*CACHE_BYTES,
      expected_success_raw_upper_bytes=40*1024**2,
      failure_extra_per_first_bad_cache_chunk_upper_bytes=2*CHUNK_PAGES*16384)


def vbook(quantize):
    import numpy as np
    codes=np.empty((64,127),dtype=np.int8);scales=np.empty(64,dtype='<f4')
    for peak in range(64):
        values=[peak/64.]+[d/64. if abs(d)<=peak else 0. for d in range(-63,64)]
        q,s=quantize(values);codes[peak]=q[1:];scales[peak]=s
    return codes,scales


def pack_v(cold,packed,book):
    """No new quantizer: finite source alphabet indexes original rational codec."""
    import numpy as np
    values=((cold[:,1].astype(np.uint32)<<16).view('<f4')*np.float32(64)).astype(np.int16)
    assert bool(((values>=-63)&(values<=63)).all())
    peaks=np.abs(values).max(axis=-1)
    code=book[0][peaks[...,None],values+63];scale=book[1][peaks]
    product=code.astype(np.float32)*scale[...,None]
    bits=product.view(np.uint32)
    dequant=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')
    control=packed.copy();private=packed.copy()
    control[:,8192:]=dequant.reshape(len(cold),4096).view(np.uint8).reshape(len(cold),8192)
    private[:,8192:]=165
    private[:,8192:12288]=code.reshape(len(cold),4096).view(np.uint8)
    private[:,12288:12416]=scale.reshape(len(cold),32).view(np.uint8).reshape(len(cold),128)
    return control,private


def cache_chunks(raw,expected,out,recipe):
    """Like3eb fullstep: compare every byte; archive chunks only on failure."""
    assert len(raw)==len(expected)==CACHE_BYTES
    digest=hashlib.sha256(raw).hexdigest();chunks=[]
    for offset in range(0,len(raw),CHUNK_PAGES*16384):
        actual=raw[offset:offset+CHUNK_PAGES*16384]
        wanted=expected[offset:offset+len(actual)]
        if actual!=wanted:
            folder=out/'failed-cache';folder.mkdir(exist_ok=True)
            (folder/(str(offset)+'-actual.bin')).write_bytes(actual)
            (folder/(str(offset)+'-expected.bin')).write_bytes(wanted)
            raise AssertionError('complete cache/guard readonly mismatch at byte'+str(offset))
        chunks.append(dict(offset=offset,bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest()))
    evidence=dict(kind='regenerable_cache_all_bytes_compared_v1',bytes=len(raw),sha256=digest,
      recipe=recipe,guards_exact=raw[:128]==raw[-128:]==b'\x5a'*128,chunks=chunks)
    assert evidence['guards_exact']
    data=(json.dumps(evidence,indent=2,sort_keys=True)+'\n').encode()
    folder=out/'cache-evidence';folder.mkdir(exist_ok=True);path=folder/(digest+'.json')
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    return dict(kind=evidence['kind'],bytes=len(raw),sha256=digest,
      evidence=dict(path=str(path.relative_to(out)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))


def namespace(parent,role,shared,load,plan):
    assert role in ('control','packed_v8')
    original=load(parent,'biv150_native_k8_consumer_fixture','cjj127_dense_original_consumer')
    full=load(parent,'biv150_native_k8_fullstep_fixture','cjj127_dense_original_fullstep')
    assert (full.PAGES,full.TABLE_COLS)==(PAGES,COLS)
    recipe=dict(parent_commit='3ebdbebf61e0e90735cd8ad83eb1605627b6ef46',
      functions={name:hashlib.sha256((parent/'tools'/name).read_bytes()).hexdigest() for name in
        ('biv150_native_k8_fullstep_fixture.py','biv150_native_tl_pair_contract.py','biv150_konly_decode_quant_reference.py')},
      dense_fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),role=role,
      source='fullstep chunk_sources+encode_chunk+codebook; V original rational codec lookup; no append')
    def initialize():
        if 'dense' in shared:return shared['dense']
        import numpy as np
        import time
        begin=time.monotonic();cold,packed,costs=full.build_initial();book=vbook(full.quantize_v2)
        control=np.empty_like(packed);private=np.empty_like(packed)
        for start in range(0,PAGES,CHUNK_PAGES):
            end=min(start+CHUNK_PAGES,PAGES)
            control[start:end],private[start:end]=pack_v(cold[start:end],packed[start:end],book)
        del packed
        raw={r:b'\x5a'*128+a.tobytes()+b'\x5a'*128 for r,a in [('control',control),('packed_v8',private)]}
        del control,private
        shared['dense']=dict(cold=cold,raw=raw,cases={},gold=None,
          initial_CPU_costs=dict(costs,total_counter_and_CPU_V_pack_seconds=time.monotonic()-begin,
            excluded_from_timing=True))
        return shared['dense']
    def fixture(n,s,pitch,phase):
        assert (n,s,pitch) in CASES and phase in (0,1)
        item=initialize()
        if phase not in item['cases']:
            import numpy as np
            ctrl=full.control(n,LENGTH,pitch,phase);ctrl['lengths'][:]=LENGTH
            del ctrl['slots'] # Never install an append operation in this reader.
            assert len(set(map(int,ctrl['table'][:,:256].reshape(-1))))==64*256
            query=[original.BF16_GUARD]*64+ctrl['query'].reshape(-1).tolist()+[original.BF16_GUARD]*64
            qcodes=[];qscales=[]
            for row in range(n):
                pairs=[full.quantize_v2([original.bf16_value(int(b)) for b in head]) for head in ctrl['query'][row,:2048].reshape(16,128)]
                qcodes.append([x[0] for x in pairs]);qscales.append([x[1] for x in pairs])
            item['cases'][phase]=dict(rows=n,segments=s,pitch=pitch,phase=phase,query_bits=query,
              lengths=[-1234567]*64+[LENGTH]*64+[-1234567]*64,
              table=[-1]*64+ctrl['table'].reshape(-1).tolist()+[-1]*64,qcodes=qcodes,qscales=qscales,control=ctrl)
        return dict(item['cases'][phase],cache=item['raw'][role])
    def golden(case):
        item=initialize()
        if item['gold'] is None:
            import numpy as np
            assert case['phase']==1
            cold=item['cold'];ctrl=dict(case['control'])
            ids=ctrl['table'][:,255] # Last active column, not the8192 tail.
            ctrl['new_k']=cold[ids,0,15].copy();ctrl['new_v']=cold[ids,1,15].copy()
            packed=np.ndarray((PAGES,16384),np.uint8,buffer=item['raw']['control'],offset=128)
            # The reused fullstep golden replaces the last KV. Make that operation
            # provably idempotent, rather than quietly adding a writer to this test.
            for row,p in enumerate(ids):
                for head in range(2):
                    code,scale=full.quantize_v2([original.bf16_value(int(b)) for b in ctrl['new_k'][row,head]])
                    assert bytes(c&255 for c in code)==packed[int(p),15*256+head*128:15*256+(head+1)*128].tobytes()
                    assert struct.pack('<f',scale)==packed[int(p),4096+15*8+head*4:4100+15*8+head*4].tobytes()
            proxy=full.golden(cold,packed,ctrl,2,'bf16')['output']
            dequant=np.ndarray((PAGES,16,2,128),'<u2',buffer=item['raw']['control'],offset=128+8192,strides=(16384,512,256,2))
            cold[:,1]=dequant;ctrl['new_v']=cold[ids,1,15].copy()
            result=full.golden(cold,packed,ctrl,2,'k8');result['original_BF16_output']=proxy
            result['last_KV_reference_replacement_proved_idempotent']=True
            item['gold']=result;del item['cold']
        return item['gold']
    f=SimpleNamespace(**vars(original))
    f.PAGES=PAGES;f.COLS=COLS;f.CASES=CASES;f.DENSE_CACHE_BYTES=CACHE_BYTES
    f.fixture=fixture;f.golden=golden;f.plan=lambda:plan(role)
    f.cache_evidence=lambda raw,out:cache_chunks(raw,initialize()['raw'][role],out,recipe)
    f.initial_metadata=lambda:dict(recipe=recipe,**initialize()['initial_CPU_costs'])
    return f
