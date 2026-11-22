#!/usr/bin/env python3
"""Small stdlib native-I8 QK layout/ABI/RN32-RNE model; no Torch/JIT/GPU."""
import hashlib,json,random,sys
from pathlib import Path
import biv150_native_k8_qk_cpu as m


def check():
    root=Path(__file__).resolve().parent.parent
    pins=json.loads((root/'docs/evidence/native-k8-qk-feasibility-20261028/source-identities.json').read_text())
    assert pins['base']==m.BASE and pins['SDK_HPP']==m.SDK_HPP
    for name,expected in pins['files'].items():
        raw=(root/name).read_bytes()
        assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],name
    rng=random.Random(20261028);checks=0
    assert sorted(m.key_byte(d,s) for d in range(128) for s in range(16))==list(range(2048))
    assert sorted(m.query_byte(r,d) for r in range(16) for d in range(128))==list(range(2048))
    coordinates={(lane,c):m.score_coordinate(lane,c) for lane in range(64) for c in range(4)}
    assert set(coordinates.values())=={(r,s) for r in range(16) for s in range(16)}
    scenarios=[];wrong_input_pair_scale_disagrees=False
    for pattern in range(4):
        q=[[(rng.randrange(-120,121)*2.**(h-4)) for _ in range(128)] for h in range(8)]
        if pattern==0:q=[[0.]*128 for _ in range(8)]
        if pattern==1:q=[[(-1 if d%2 else 1)*(h+1)*.5 for d in range(128)] for h in range(8)]
        if pattern==3:q=[[(-1 if d%2 else 1)*2.**-130 for d in range(128)] for h in range(8)]
        qs=[m.quantize_v2(v) for v in q];qcode=[v[0] for v in qs];qscale=[v[1] for v in qs]
        if pattern==1:
            wrong_input_pair_scale_disagrees=any(qscale[2*g]!=qscale[g] for g in range(4))
            assert wrong_input_pair_scale_disagrees
        key=[m.quantize_v2([rng.randrange(-100,101)*.125 for _ in range(128)]) for _ in range(16)]
        for valid in (0,1,7,15,16):
            image=bytearray(8192);image[4096:]=b'\x71'*4096
            for slot in range(16):
                for dim in range(0,128,4):
                    word=m.pack4(key[slot][0][dim:dim+4] if slot<valid else (0,0,0,0))
                    start=m.key_byte(dim,slot);image[start:start+4]=word.to_bytes(4,'little')
            assert image[2048:4096]==bytes(2048) and image[4096:]==b'\x71'*4096
            aq=bytearray(2048)
            for row in range(16):
                for dim in range(128):aq[m.query_byte(row,dim)]=qcode[row%8][dim]&255
            decoded_a={};decoded_b={}
            for lane in range(64):
                g,col=divmod(lane,16)
                for k in range(4):
                    for component in (0,1):
                        start=m.a_word_index(lane,k,component);word=int.from_bytes(aq[start:start+4],'little')
                        assert word==m.fragment_a(qcode,lane,k,component)
                        for byte,value in enumerate(m.unpack4(word)):decoded_a[4*g+byte,32*k+16*component+col]=value
                        start=m.b_word_index(lane,k,component);word=int.from_bytes(image[start:start+4],'little')
                        for byte,value in enumerate(m.unpack4(word)):decoded_b[32*k+16*component+4*g+byte,col]=value
                        checks+=1
            assert len(decoded_a)==len(decoded_b)==2048
            for lane in range(64):
                for c in range(4):
                    row,slot=coordinates[lane,c]
                    dot=sum(decoded_a[row,d]*decoded_b[d,slot] for d in range(128))
                    expected=sum(a*b for a,b in zip(qcode[row%8],key[slot][0])) if slot<valid else 0
                    assert dot==expected and abs(dot)<=2064512 and m.signed32(dot&0xffffffff)==dot
                    assert m.score_scale(qscale,lane,c)==qscale[row%8]
                    checks+=1
            scenarios.append(dict(pattern=pattern,valid_slots=valid))
    addresses=set()
    for page in (0,1,2):
        for slot in range(16):
            for head in (0,1):
                s=m.scale_address(page,slot,head,3,128);assert s%4==0
                for dim in range(128):
                    address=m.key_address(page,slot,head,dim,3,128);assert address not in addresses;addresses.add(address)
                    assert address<128+page*16384+4096
    assert len(addresses)==12288
    assert all(m.signed32(v&0xffffffff)==v for v in (-2064512,-127,-1,0,1,2064512))
    good=dict(format=m.FORMAT,key_dtype='i8',scale_dtype='f32',value_dtype='bf16',query_dtype='bf16',page_bytes=16384,
        k_strides=(16384,256,128,1),v_strides=(8192,256,128,1),scale_full_base=True,scale_byte_offset=4096,
        head_dim=128,queries_per_kv=8,page_size=16,shape=(64,2),query_stride=(2048,128,1))
    assert m.admit(good);faults=[]
    for name,bad in [('wrong_dtype',dict(good,key_dtype='bf16')),('offset_scale',dict(good,scale_full_base=False)),
        ('compact_unverified',dict(good,page_bytes=12544)),('wrong_Qstride',dict(good,query_stride=(2560,256,1))),
        ('wrong_geometry',dict(good,shape=(64,4))),('bool_scale',dict(good,scale_byte_offset=True)),
        ('bool_stride',dict(good,query_stride=(2048,128,True))),('float_shape',dict(good,shape=(64.,2)))]:
        assert not m.admit(bad);faults.append(name)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(plan=m.plan(),logical_and_fragment_word_checks=checks,scenarios=len(scenarios),
        unique_key_addresses=len(addresses),fragment_images_bijective=True,negative_I32_bit_interpretation=True,
        independent_C_head_scale_ownership=True,Q_global_read_reuse_pair_shuffles=True,rejected=faults,
        wrong_input_pair_scale_negative_control=wrong_input_pair_scale_disagrees,source_pins_verified=True,
        note='CPU describes SDK ABI; builtin MMA execution/layout and CUDA codec remain unvalidated')

if __name__=='__main__':print(json.dumps(check(),sort_keys=True,indent=2))
