"""Private CPU V codec/operand adapter; original 3eb numerical functions retained."""
import ast
import math
import struct
from types import FunctionType, SimpleNamespace

PAGES=389
COLS=384
CASES=((64,2,2048),(64,2,2560))
SHORT=(0,1,15,16,17,31,32,33,63,64,65,127,128,129,191,192,193)


def dequant_bf16(code,scale):
    assert type(code) is int and -128<=code<=127
    assert type(scale) is float and math.isfinite(scale) and scale>0 and struct.unpack('<f',struct.pack('<f',scale))[0]==scale
    # I8*F32 is exact in Python binary64, then explicitly RN32 before BF16 RNE.
    f32=struct.unpack('<I',struct.pack('<f',float(code)*scale))[0]
    return ((f32+0x7fff+((f32>>16)&1))>>16)&65535


def lengths(phase):
    assert type(phase) is int and phase in (0,1,2)
    rows=list(SHORT)+[6143,6144]
    rows += [SHORT[i%len(SHORT)] for i in range(64-len(rows))]
    shift=phase*5
    return rows[shift:]+rows[:shift]


def mapping(lens,phase):
    # All sources are readonly: sharing physical pages across requests is legal.
    # Logical page384 is exclusive to the 6143 last page, so lane15 can be poison.
    out=[-1]*(64*COLS)
    for row,length in enumerate(lens):
        for col in range((length+15)//16):
            logical=(384 if length==6143 and col==383 else col if length>=6143 else (row*3+col*7)%29)
            out[row*COLS+col]=(logical*7+phase*5)%PAGES
    return out


def prepare_v(original,quantize,bf16_value,pages=PAGES):
    """Same logical payload for two ABIs; full allocation remains16KiB/page."""
    control=bytearray(original['cache']);private=bytearray(original['cache'])
    dequant=[]
    for page in range(pages):
        base=128+page*16384
        private[base+8192:base+16384]=b'\xa5'*8192
        page_values=[]
        for slot in range(16):
            slot_values=[]
            for head in range(2):
                codes,scale=quantize([bf16_value(x) for x in original['value_bits'][page][slot][head]])
                assert len(codes)==128 and all(-127<=c<=127 for c in codes) and scale>0
                bits=[dequant_bf16(int(c),scale) for c in codes]
                private[base+8192+slot*256+head*128:base+8192+slot*256+(head+1)*128]=bytes(c&255 for c in codes)
                struct.pack_into('<f',private,base+12288+slot*8+head*4,scale)
                struct.pack_into('<128H',control,base+8192+slot*512+head*256,*bits)
                slot_values.append(bits)
            page_values.append(slot_values)
        dequant.append(page_values)
    return bytes(control),bytes(private),dequant


def flatten_bits(values):
    return struct.pack('<'+str(len(values)*16*2*128)+'H',*(v for page in values for slot in page for head in slot for v in head))


def pagewise_source(parent):
    text=(parent/'tools/biv150_native_k8_pagewise_reference.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='from_inputs')
    source=ast.get_source_segment(text,node)
    assert source.count('.reshape(rows,16)')==1
    changed=source.replace('.reshape(rows,16)','.reshape(rows,COLS)')
    assert changed.replace('.reshape(rows,COLS)','.reshape(rows,16)')==source
    return source,changed


def namespace(parent,role,shared,load,plan):
    assert role in ('control','packed_v8')
    original=load(parent,'biv150_native_k8_consumer_fixture','cjj127_original_fixture')
    # Code objects are unchanged. Only private fixture globals/extent differ.
    globals_=dict(vars(original),PAGES=PAGES,COLS=COLS,CASES=CASES,BOUNDARIES=SHORT)
    original_fixture=FunctionType(original.fixture.__code__,globals_)
    ideal=FunctionType(original.golden.__code__,globals_)
    f=SimpleNamespace(**vars(original));f.PAGES=PAGES;f.COLS=COLS;f.CASES=CASES
    def logical(phase):
        if phase not in shared:
            case=original_fixture(64,2,2048,phase)
            lens=lengths(phase);table=mapping(lens,phase)
            case['lengths']=[-1234567]*64+lens+[-1234567]*64
            case['table']=[-1]*64+table+[-1]*64
            control,private,value_bits=prepare_v(case,original.quantize_v2,original.bf16_value)
            # The uniquely mapped last-page invalid lane must never affect output.
            last=(384*7+phase*5)%PAGES
            arrays=[bytearray(control),bytearray(private)]
            for a in arrays:
                base=128+last*16384
                a[base+15*256:base+16*256]=b'\x80'*256
                struct.pack_into('<2I',a,base+4096+15*8,0x7fc12345,0x7fc12345)
            base=128+last*16384
            struct.pack_into('<256H',arrays[0],base+8192+15*512,*([0x7fc1]*256))
            arrays[1][base+8192+15*256:base+8192+16*256]=b'\x80'*256
            struct.pack_into('<2I',arrays[1],base+12288+15*8,0x7fc12345,0x7fc12345)
            raw=dict(case);case=dict(case,value_bits=value_bits,cache=bytes(arrays[0]))
            shared[phase]=dict(case=case,original=raw,private=bytes(arrays[1]),gold=None)
        return shared[phase]
    def fixture(n,s,pitch,phase):
        assert (n,s,pitch) in CASES
        item=logical(phase);case=dict(item['case'])
        if pitch!=2048:
            q=[f.BF16_GUARD]*(128+n*pitch)
            for row in range(n):q[64+row*pitch:64+row*pitch+2048]=case['query_bits'][64+row*2048:64+(row+1)*2048]
            case['query_bits']=q
        case.update(pitch=pitch,cache=item['case']['cache'] if role=='control' else item['private'])
        return case
    def golden(case):
        item=logical(case['phase'])
        if item['gold'] is None:
            # The original ideal final/M/L and pagewise-BF16-P partial are reused.
            result=ideal(item['case'])
            pagewise=load(parent,'biv150_native_k8_pagewise_reference','cjj127_original_pagewise')
            source,changed=pagewise_source(parent)
            ns=dict(vars(pagewise),COLS=COLS);exec(compile(changed,'<127-original-pagewise-table-extent-only>','exec'),ns)
            result['ideal_partial_diagnostic']=result['partial']
            result['partial']=ns['from_inputs'](original.input_bytes(item['case']),64,2,2048)['partial']
            # Preserve the original, UNQUANTIZED BF16 Q/K/V proxy independently.
            result['dequantV_original_QK_diagnostic']=result['original_BF16_output']
            result['original_BF16_output']=ideal(item['original'])['original_BF16_output']
            item['gold']=result
        return item['gold']
    def cpu_oracles(cases,blob):
        result={}
        for phase in cases:
            item=logical(phase)
            result[str(phase)]=dict(original_BF16_K_bits=blob(flatten_bits(item['original']['key_bits'])),
                original_BF16_V_bits=blob(flatten_bits(item['original']['value_bits'])),
                control_dequant_BF16_cache=blob(item['case']['cache']),private_V8_cache=blob(item['private']),
                V_codec='original rational RN32/RNE; not a GPU writer',V_operand='BF16_RNE(F32(I8code*F32scale))')
        return result
    f.fixture=fixture;f.golden=golden;f.cpu_oracles=cpu_oracles;f.plan=lambda:plan(role)
    return f
