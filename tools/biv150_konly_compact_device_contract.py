"""Stdlib-only compact writer/decode device contract; never imports GPU runtimes."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE='8babed1957f3477bfa158ecdadc8b7d020b1147c'
MAIN='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
PARENT='099f38aa382581645420699c3f6291604e7bc147'
FORMAT='cjj-konly-compact12544-rn32-rne-v1'
CASES={'4k':(64,4096,2),'16k':(32,16384,4)}
PAGE=12544
GUARD=256

def sha(b):return hashlib.sha256(b).hexdigest()
def blob(path,commit=BASE):return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def refs():
    directory=ROOT/'tools/compact_reference';pins=json.loads((directory/'source-pins.json').read_text())
    for n,v in pins['paths'].items():assert sha((directory/n).read_bytes())==v
    # These names are the unchanged imports inside the frozen stdlib helpers.
    sys.path.insert(0,str(directory))
    from biv150_konly_decode_quant_reference import quantize_v2
    from biv150_konly_decode_numeric_contract import codec_vectors,reference_gates
    from biv150_konly_decode_numeric_oracle import quantize_cpu,attention_references
    from biv150_konly_decode_ir import audit
    return dict(quantize_v2=quantize_v2,codec_vectors=codec_vectors,reference_gates=reference_gates,
                quantize_cpu=quantize_cpu,attention_references=attention_references,audit=audit)

def packed(codes,scales,value_bits,np):
    p=codes.shape[0];assert codes.shape==(p,16,2,128) and scales.shape==(p,16,2)
    assert value_bits.shape==codes.shape and value_bits.dtype==np.uint16
    result=np.full((p,PAGE),0xa5,dtype=np.uint8)
    result[:,:4096]=codes.view(np.uint8).reshape(p,4096)
    result[:,4096:4224]=scales.view(np.uint8).reshape(p,128)
    result[:,4352:]=value_bits.view(np.uint8).reshape(p,8192)
    return result.tobytes()

def append(before,slots,keys,values,quantize):
    assert len(before)%PAGE==0 and len(slots)==len(keys)==len(values)
    p=len(before)//PAGE;valid=[s for s in slots if 0<=s<p*16];assert len(valid)==len(set(valid))
    result=bytearray(before)
    for t,s in enumerate(slots):
        if not 0<=s<p*16:continue
        page,col=divmod(s,16)
        for h in range(2):
            code,scale=quantize(keys[t][h]);base=page*PAGE
            result[base+col*256+h*128:base+col*256+h*128+128]=bytes(v&255 for v in code)
            struct.pack_into('<f',result,base+4096+4*(col*2+h),scale)
            bits=[struct.unpack('<I',struct.pack('<f',v))[0]>>16 for v in values[t][h]]
            struct.pack_into('<128H',result,base+4352+col*512+h*256,*bits)
    return bytes(result)

def mapping(batch,length):
    n=(length+15)//16;pages=batch*n+3;ids=list(range(pages));random.Random(20261028+batch+length).shuffle(ids)
    return [ids[t*n:(t+1)*n] for t in range(batch)],pages

def plan():
    import biv150_konly_compact_kv_bridge as bridge
    manifest=json.loads((ROOT/'vllm_fl/patches/biv150_konly_compact_sources.json').read_text())
    assert manifest['parent89']=='d8ef055422d9d5c45049f85468b486b47b993406' and not manifest['q32_hoist_supported']
    proofs={};private={}
    for role in ('writer','decode','q32','multi2','raw3d','normalized2d128'):
        p='vllm_fl/patches/biv150_konly_encoded_'+role+'.py'
        generated,proof=bridge.kernel_source(blob(p).decode(),role)
        compact='vllm_fl/patches/biv150_konly_compact_'+role+'.py'
        assert generated.encode()==(ROOT/compact).read_bytes()==blob(compact)
        proofs[role]=proof
    for p in (ROOT/'vllm_fl/patches').glob('biv150_konly_encoded_*'):
        assert p.read_bytes()==blob(str(p.relative_to(ROOT))),p
        receipt=json.loads((ROOT/'docs/evidence/konly-compact-kv-bridge-20261027/device-v1/parent89-production.json').read_text())
        assert sha(p.read_bytes())==receipt['private_files'][p.name]
        private[p.name]=sha(p.read_bytes())
    for p in (ROOT/'vllm_fl/patches').glob('biv150_konly_compact_*'):
        assert p.read_bytes()==blob(str(p.relative_to(ROOT))),p
    assert not subprocess.check_output(['git','-C',str(ROOT),'diff',BASE,'--','vllm_fl','csrc'])
    return dict(CPU_only=True,base91=BASE,main=MAIN,parent89=PARENT,physical_page_bytes=PAGE,
        source_inverse=proofs,parent89_private_bytes=private,
        cases={k:dict(batch=b,length=l,segments=s,grid=[b,2,s],pages=mapping(b,l)[1]) for k,(b,l,s) in CASES.items()},
        compact_roles=dict(K_byte_stride=12544,scale_F32_stride=3136,V_BF16_stride=6272,V_byte_offset=4352,
                           scale_fullbase=True,K_fullbase=True,guarded_layer_byte_offset=GUARD),
        kernel_math_changed=False,old_16KiB_or_hoist_qualification_used=False,
        compile_executed=False,GPU_executed=False,service_or_capacity_verified=False,
        following_scopes=['SDK raw3D','2D','Q32','multi2','actual service allocation/Graph/lifecycle/model/performance'])
