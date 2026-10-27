"""Stdlib BF16-bit fixtures and independent PR83 rational RN32/RNE oracle."""
import hashlib
import json
import random
import struct
from biv150_konly_decode_quant_reference import quantize_v2

INPUT_GUARD=0x4255
CODE_GUARD=0x53
SCALE_GUARD=0x428e0000
SCALE_POISON=0x7fc12345
LIBRARY_SHA='22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def bf16_value(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
def f32_bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def bf16_bits(value):
    bits=f32_bits(value)
    return ((bits+0x7fff+((bits>>16)&1))>>16)&65535

def fixtures():
    result=[]
    def add(name,bits):
        assert len(bits)==128 and all(type(x) is int and 0<=x<=65535 and x&0x7f80!=0x7f80 for x in bits)
        result.append(dict(name=name,bits=bits))
    add('positive_zero',[0]*128);add('negative_zero',[0x8000]*128)
    add('mixed_signed_zero',[0x8000 if i%2 else 0 for i in range(128)])
    for name,bits in [('minimum_subnormal',1),('maximum_subnormal',0x7f),('minimum_normal',0x80),
                      ('normal_successor',0x81),('maximum_finite',0x7f7f),
                      ('tiny_normal',0x100),('half',0x3f00),('one',0x3f80),('mantissa_boundary',0x3fff),
                      ('two',0x4000),('huge_power',0x7f00)]:
        add(name,[(bits|(0x8000 if i%2 else 0)) if i%5 else 0 for i in range(128)])
    for exponent in (-133,-130,-127,-126,-125,-64,-8,0,8,64,120):
        values=[127.*2.**exponent]+[(i-63.5)*2.**exponent for i in range(127)]
        add('half_integer_family_exp'+str(exponent),[bf16_bits(v) for v in values])
    rng=random.Random(20261028)
    for row in range(8):
        add('finite_BF16_mixed_exp'+str(row),[(rng.randrange(2)<<15)|(rng.randrange(255)<<7)|rng.randrange(128) for _ in range(128)])
    a=[row['bits'] for row in result]
    b=[a[(row+7)%len(a)][13:]+a[(row+7)%len(a)][:13] for row in range(len(a))]
    b[0]=[0x3f80]+[0]*127
    c=[[value^0x8000 for value in b[(row+11)%len(a)]] for row in range(len(a))]
    c[0]=[0x42fe]+[bf16_bits((i-63.5)) for i in range(127)]
    return dict(names=[row['name'] for row in result],A=a,B=b,C=c)

def oracle(rows):
    code=[];scale=[]
    for bits in rows:
        values=[bf16_value(v) for v in bits];codes,s=quantize_v2(values)
        code.extend(codes);scale.append(f32_bits(s))
    return dict(codes=code,scale_bits=scale)

def guarded_input(rows):return struct.pack('<'+str((len(rows)+2)*128)+'H',*([INPUT_GUARD]*128+[v for row in rows for v in row]+[INPUT_GUARD]*128))
def guarded_codes(expected):return bytes([CODE_GUARD]*128+[v&255 for v in expected['codes']]+[CODE_GUARD]*128)
def guarded_scales(expected):return struct.pack('<'+str(len(expected['scale_bits'])+2)+'I',SCALE_GUARD,*expected['scale_bits'],SCALE_GUARD)

def full_check(rows,expected,arrays):
    authority=dict(input_full=guarded_input(rows),code_full=guarded_codes(expected),scale_full=guarded_scales(expected))
    checks={name:arrays[name]==value for name,value in authority.items()}
    mismatches={name:dict(actual_bytes=len(arrays[name]),expected_bytes=len(value),
      first_bytes=[dict(byte=i,actual=a,expected=b) for i,(a,b) in enumerate(zip(arrays[name],value)) if a!=b][:16])
      for name,value in authority.items() if not checks[name]}
    return checks,mismatches

def plan():
    data=fixtures();refs={key:oracle(data[key]) for key in ('A','B','C')}
    assert refs['A']['codes']!=refs['B']['codes'] and refs['A']['scale_bits']!=refs['B']['scale_bits']
    assert refs['B']['codes']!=refs['C']['codes'] and refs['B']['scale_bits']!=refs['C']['scale_bits']
    return dict(component='native K8 codec only',library_sha256=LIBRARY_SHA,rows=len(data['A']),width=128,
      categories=data['names'],fixture={k:dict(input_full_sha256=sha(guarded_input(data[k])),
        code_full_sha256=sha(guarded_codes(refs[k])),scale_full_sha256=sha(guarded_scales(refs[k]))) for k in refs},
      exact_code_bytes=True,exact_F32_scale_bits=True,tolerance=None,
      oracle='existing independent rational RN32 divide plus integer RNE matching PR83; not torch.round',
      phases=['eager A','capture A','changed B + poison + graph replay before eager B','eager B','side-stream changed C'],
      full_CPU_storage_authority=True,negative_GPU_advanced_indexing_used=False,
      compile_requested=False,default_imports_framework=False,attention_launches=0,encoder_launches=0,
      actual_codec_verified=False,Graph_verified=False,side_stream_verified=False,
      numerical_attention_verified=False,model_verified=False,precision_gate=False,performance_measured=False)

if __name__=='__main__':print(json.dumps(plan(),sort_keys=True,indent=2))
