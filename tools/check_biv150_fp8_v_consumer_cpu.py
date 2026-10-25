"""No-GPU proof of bounded producer edits and mixed-unit packed V addresses."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'vllm_fl/patches/biv150_gqa_decode.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
    original=source.read_text();tree=ast.parse(original)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gqa_decode_attention')
    expected=ast.get_source_segment(original,node)
    own=(ROOT/'tools/kernels/biv150_fp8_v_attention.py').read_text();tree=ast.parse(own)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gqa_decode_fp8_v')
    restored=ast.get_source_segment(own,node).replace('def gqa_decode_fp8_v(','def _gqa_decode_attention(',1)
    restored=restored.replace('    value_lut_ptr,\n','',1).replace('    USE_LUT: tl.constexpr,\n','',1)
    full='''        codes = tl.load(value_cache_ptr + v_offsets)
        if USE_LUT:
            V = decode_e4m3_lut(codes, value_lut_ptr)
        else:
            V = decode_e4m3_bits(codes)'''
    tail='''        codes = tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0)
        if USE_LUT:
            V = decode_e4m3_lut(codes, value_lut_ptr)
        else:
            V = decode_e4m3_bits(codes)'''
    assert restored.count(full)==restored.count(tail)==1
    restored=restored.replace(full,'        V = tl.load(value_cache_ptr + v_offsets)')
    restored=restored.replace(tail,'        V = tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0.0)')
    assert restored==expected,'QK/softmax/loop/output math changed beyond V reconstruction'
    for pages in [1,3]:
        key_bytes=set();value_bytes=set();padding=set()
        for block in range(pages):
            for slot in range(16):
                for head in range(2):
                    for dim in range(128):
                        key=(block*8192+slot*256+head*128+dim)*2
                        value=block*16384+8192+slot*256+head*128+dim
                        assert key not in key_bytes and key+1 not in key_bytes
                        assert value not in value_bytes
                        key_bytes.update((key,key+1));value_bytes.add(value)
            padding.update(range(block*16384+12288,(block+1)*16384))
        assert len(key_bytes)==pages*8192 and len(value_bytes)==pages*4096
        assert not key_bytes&value_bytes and not (key_bytes|value_bytes)&padding
        assert key_bytes|value_bytes|padding==set(range(pages*16384))
    print(json.dumps({'passed':True,'gpu_executed':False,'producer_inverse_matches_shipping_source':True,
                      'baseline_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'K':'BF16 unchanged','V':'one-byte software E4M3',
                      'physical_bytes_per_page':16384,'used_payload_bytes_per_page':12288,
                      'capacity_reclaimed':False,'packed_addresses_unique_disjoint_bounded':True,
                      'remaining':'GPU codec/attention/writer/Graph/timing and model qualification'},indent=2))


if __name__=='__main__':main()
