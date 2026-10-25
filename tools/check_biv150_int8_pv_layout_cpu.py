"""Independent byte budget and two-page segment-address checks, no GPU execution."""
import ast
import hashlib
import json
from pathlib import Path
from biv150_int8_pv_reference import blocks

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'vllm_fl/patches/biv150_gqa_decode.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
    kernel=ROOT/'tools/kernels/biv150_int8_pv.py';tree=ast.parse(kernel.read_text())
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_integer_pv')
    dots=[n for n in ast.walk(helper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dot']
    assert len(dots)==1 and [n.id for n in dots[0].args]==['codes','values']
    for n in ast.walk(helper):
        assert not (isinstance(n,ast.Attribute) and n.attr in ['bfloat16','float16']), 'wide V reconstruction introduced'
    for pages in [1,3]:
        key=set();value=set();scale=set();padding=set()
        for page in range(pages):
            for slot in range(16):
                for head in range(2):
                    for dim in range(128):
                        offset=page*16384+((slot*2+head)*128+dim)*2
                        key.update([offset,offset+1])
                        value.add(page*16384+8192+(slot*2+head)*128+dim)
            scale.update(range(page*16384+12288,page*16384+12416))
            padding.update(range(page*16384+12416,(page+1)*16384))
        assert len(key)==pages*8192 and len(value)==pages*4096
        assert not (key&value or key&scale or value&scale or key&padding or value&padding or scale&padding)
        assert key|value|scale|padding==set(range(pages*16384))
    cases=0
    for length in range(1,258):
        page_count=(length+15)//16;table=list(reversed(range(page_count)))
        for segments in [1,2,4,8,16]:
            observed=[]
            for _,spans in blocks(length,segments):
                for start,end in spans:
                    first=table[start//16]
                    second=table[start//16+1] if start+16<end else None
                    for offset in range(end-start):
                        physical=first if offset<16 else second
                        assert physical is not None
                        logical=start+offset
                        assert physical==table[logical//16]
                        assert offset%16==logical%16
                        observed.append(logical)
            assert observed==list(range(length));cases+=1
    print(json.dumps({'passed':True,'gpu_executed':False,'mapping_cases':cases,'direct_int8_dot_ast':True,
                      'physical_page_bytes':16384,'payload_bytes_per_page':12416,'embedded_scale_bytes_per_page':128,'scale_float32_stride':[4096,2,1],'scale_byte_offset':12288,
                      'probe_capacity_reclaimed':False,'limitation':'CPU convention/static checks, not runtime address/Graph/precision proof'},indent=2))


if __name__=='__main__':main()
