#!/usr/bin/env python3
"""Finite source inverse and V address/owner checks; no mirrored GPU framework."""
import ast
import json
from pathlib import Path
import sys
import subprocess
import biv150_native_k8_packed_v8_factory as f


def vh(s, d):
    panel, c, r = d//16, d%16, s//2
    return panel*256+(r//4)*128+((r*32+c*2+(s&1)+panel*64)&127)


def main():
    source = f.verify()
    generated = f.frontend_source()
    ast.parse(generated)
    cu = f.path('qk.cu').read_text()
    parent = f.original('qk.cu')
    # Three unchanged CTA barrier sites in consume_page plus the K0 prologue.
    assert cu.count('__syncthreads();') == parent.count('__syncthreads();') == 4
    assert '__ivcorex_sl_waitcnt' not in cu
    # The shared scout is preserved. Its only executable delta is removing wait8.
    previous = subprocess.check_output(['git','-C',str(f.ROOT),'show',
        '0c97589ba902696cddd9619fd6fdee3cb0bc1371:tools/kernels/biv150_native_k8_packed_v8_qk.cu']).decode()
    assert f.sha(previous.encode()) == '5de8bcd494e58a7a2ccbf43be807ca96b701d23105242e30f3c3ee5d28b89a41'
    old_wait = ('  // Kept only as a scout instruction, NOT proof of ordinary-load completion.\n'
                '  // No V SME task exists now: final-SO counter safety is an open gate.\n'
                '  __ivcorex_sl_waitcnt(8ULL);\n')
    ordinary = ('  // No explicit G2S wait: ordinary V values feed dependent pair stores.\n'
                '  // Final SO must prove their memory dependencies and the original PV barrier.\n')
    assert cu.count(ordinary) == previous.count(old_wait) == 1
    assert cu.replace(ordinary,old_wait) == previous
    assert '__ivcorex_sme_load_16x1b64_rowxfb16' not in cu
    assert 'if(valid<16)clear_v_tail(stage,valid,lane);' not in cu
    assert cu.count('producer<2><<<') == 1 and 'producer<4><<<' not in cu
    begin, end = ' int next_page=0;', '\n}\n}\n#endif'
    assert cu[cu.index(begin):cu.index(end)] == parent[parent.index(begin):parent.index(end)]
    # Exhaust each byte pattern; signed reconstruction and packing are bitwise.
    assert [(b^128)-128 for b in range(256)] == list(range(128))+list(range(-128,0))
    for bits in range(65536):
        other = bits ^ 0xa55a
        packed = bits | (other << 16)
        assert packed & 65535 == bits and packed >> 16 == other
    total_reads = total_stores = 0
    for valid in range(17):
        for head in range(2):
            reads, stores = set(), {}
            for lane in range(64):
                for i in range(4):
                    w = lane+64*i
                    slot, dim = 2*(w//32), 4*(w%32)
                    for s in (slot, slot+1):
                        if s < valid:
                            address = 8192+s*256+head*128+dim
                            assert address % 4 == 0
                            reads.update(range(address,address+4))
                    for j in range(4):
                        d = dim+j
                        assert vh(slot+1,d) == vh(slot,d)+1
                        destination = 1024+vh(slot,d)//2
                        assert destination not in stores
                        # Distinct arbitrary BF16 bit patterns, including sign/specials.
                        lo = (slot*128+d)*31 & 65535
                        hi = ((slot+1)*128+d)*31 & 65535
                        original = lo | hi << 16
                        old_clear = (0 if slot >= valid else original & 65535
                                     if slot+1 >= valid else original)
                        pair = (lo if slot < valid else 0) | ((hi if slot+1 < valid else 0) << 16)
                        assert pair == old_clear
                        stores[destination] = lane
            assert set(stores) == set(range(1024,2048))
            assert reads == {8192+s*256+head*128+d for s in range(valid) for d in range(128)}
            total_reads += len(reads)
            total_stores += len(stores)
            # Physical scale sources are precisely active lanes0..15, once per head.
            scales = {12288+lane*8+head*4 for lane in range(64) if lane < valid}
            assert scales == {12288+s*8+head*4 for s in range(valid)}
            assert scales.isdisjoint(reads)
    for page in (0,1,2147483647):
        for base in (0,4,16):
            low = base+page*16384
            assert 0 <= low < 2**63
            assert low+8192+15*256+128+127 < low+12288
            assert low+12288+15*8+4+4 <= low+12416 <= low+16384
    cpp = f.path('qk.cpp').read_text()
    assert 'TORCH_CHECK(segments==2,' in cpp and 'const std::int64_t rows=64;' in cpp
    assert 'v.scalar_type()==at::kChar' in cpp and 'private V8 +8192B role; BF16/compact forbidden' in cpp
    assert 'm.impl("codec"' not in cpp and f.NAMESPACE in cpp
    # Exercise actual CPU plan using generated frontend, never compile_only().
    ns = dict(__name__='cpu_only',__file__=str(Path(__file__).with_name('compile_biv150_native_k8_packed_v8.py')))
    exec(compile(generated,'<pinned-frontend>','exec'),ns)
    plan = ns['cpu_plan']()
    assert len(ns['SOURCES']) == 2 and plan['frontend_explicit_kernel_launches'] == 0
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    result = dict(passed=True,source=source,source_inverse_includes_all_original_math=True,
        compile_frontend_sha256=f.sha(generated.encode()),
        check_script_sha256=f.sha(Path(__file__).read_bytes()),
        tail_lengths=list(range(17)),head_cases=2,shared_words_per_case=1024,
        byte_reads_across_tail_cases=total_reads,word_writes_across_tail_cases=total_stores,
        signed_byte_patterns=256,paired_bf16_bit_patterns=65536,
        invalid_slot_no_source_or_scale_read=True,old_tail_zero_semantics_equal=True,
        new_unique_word_owner_no_tail_RMW=True,original_Knext_PV_end_barrier_text_exact=True,
        CTA_barrier_sites=4,explicit_G2S_waits=0,private_parent_wait_only_inverse_byte_exact=True,
        actual_ordinary_load_store_PV_and_Knext_dependencies_verified=False,
        actual_compiled_resources=None,device_compile=False,numeric_verified=False,
        writer_verified=False,performance_verified=False,service_enabled=False)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
