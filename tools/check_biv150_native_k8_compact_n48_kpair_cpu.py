#!/usr/bin/env python3
"""CPU-only source inverse and exhaustive K transport address/bit-image audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import biv150_native_k8_compact_n48_kpair_factory as f


def ki(slot, dim):
    panel, within = divmod(dim, 64)
    quartet, byte = divmod(within, 4)
    bank, subslot = divmod(slot, 4)
    return panel*1024 + bank*256 + (quartet % 4)*64 + ((quartet//4) ^ bank)*16 + subslot*4 + byte


def payload(slot, dim, salt, pattern):
    if pattern == 0:
        return (slot*128 + dim + salt) & 255
    if pattern == 1:
        return (0, 255, 128, 127, 1, 254, 85, 170)[(slot*3 + dim + salt) % 8]
    return hashlib.sha256(f'{salt}:{slot}:{dim//32}'.encode()).digest()[dim % 32]


def image(base, valid, salt, pattern, paired):
    # Integer virtual addresses permit high page IDs without huge allocations.
    before = bytes((i*29+0xa5) & 255 for i in range(8192+64))
    stage = bytearray(before)
    reads, writes = [], []

    def read(address, width):
        relative = address-base
        slot, dim = divmod(relative, 256)
        assert 0 <= slot < valid and 0 <= dim and dim+width <= 128
        assert address % width == 0
        reads.append((address, width))
        return bytes(payload(slot, dim+i, salt, pattern) for i in range(width))

    def store(slot, dim, bits):
        target = ki(slot, dim)
        assert target % 4 == 0 and 0 <= target <= 2044 and len(bits) == 4
        writes.append((target, bits))
        stage[32+target:36+target] = bits

    vector_path = paired and base % 8 == 0
    for lane in range(64):
        if vector_path:
            for pair in range(lane, 256, 64):
                word = pair*2
                slot, q = divmod(word, 32)
                dim = q*4
                bits = read(base+slot*256+dim, 8) if slot < valid else bytes(8)
                store(slot, dim, bits[:4])
                store(slot, dim+4, bits[4:])
        else:
            for word in range(lane, 512, 64):
                slot, q = divmod(word, 32)
                dim = q*4
                bits = read(base+slot*256+dim, 4) if slot < valid else bytes(4)
                store(slot, dim, bits)
    assert len(writes) == 512 and sorted(t for t, _ in writes) == list(range(0, 2048, 4))
    assert stage[:32] == before[:32] and stage[2080:] == before[2080:]
    actual_reads = Counter(a+i for a, n in reads for i in range(n))
    expected_reads = Counter(base+s*256+d for s in range(valid) for d in range(128))
    assert actual_reads == expected_reads  # complete multiset, not just bounds
    for slot in range(16):
        for dim in range(128):
            expected = payload(slot, dim, salt, pattern) if slot < valid else 0
            assert stage[32+ki(slot, dim)] == expected
    return bytes(stage), reads, writes, vector_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    proof = f.verify()
    assert f.sha(f.blob('qk.cu')) == '1aca8c4be8d8d8e8d818cad529c0716d16b1e30a871ee3c8ec48b12d62338ab7'
    assert sorted(ki(s, d) for s in range(16) for d in range(128)) == list(range(2048))
    assert {ki(s, d+4)-ki(s, d) for s in range(16) for d in range(0, 128, 8)} == {64}
    bases = [0x100000000 + r for r in (0, 4, 8, 12)]
    pages = [0, 1, 32767, 2147483646]
    heads = [0, 1]
    counts = Counter()
    digest = hashlib.sha256()
    for arena in bases:
        for page in pages:
            for head in heads:
                base = arena+page*12544+head*128
                assert base < 2**63 and base % 8 == arena % 8
                for valid in range(17):
                    for pattern in range(3):
                        salt = page*101+head*17+valid*7
                        old, old_reads, old_writes, _ = image(base, valid, salt, pattern, False)
                        new, new_reads, new_writes, vector = image(base, valid, salt, pattern, True)
                        assert old == new and sorted(old_writes) == sorted(new_writes)
                        assert len(old_reads) == valid*32
                        assert len(new_reads) == valid*(16 if vector else 32)
                        assert sorted(a+i for a, n in old_reads for i in range(n)) == sorted(a+i for a, n in new_reads for i in range(n))
                        if not vector:
                            assert old_reads == new_reads and old_writes == new_writes
                        counts['image_pairs'] += 1
                        counts['paired_path_cases' if vector else 'scalar_fallback_cases'] += 1
                        counts['shared_word_pairs_checked'] += 512
                        counts['shared_byte_pairs_checked'] += 2048
                        counts['old_global_read_transactions'] += len(old_reads)
                        counts['candidate_global_read_transactions'] += len(new_reads)
                        digest.update(json.dumps([arena, page, head, valid, pattern]).encode())
                        digest.update(old)
                        digest.update(new)
    cuda = f.path('qk.cu').read_text()
    negatives = []
    for label, old, new in (
        ('4B branch incorrectly admitted as uint2', '(base)&7u)==0', '(base)&3u)==0'),
        ('second vector word reordered', 'bits.y;', 'bits.x;'),
        ('vector tail read made unconditional', 'if(slot<valid)bits=*reinterpret_cast<const uint2*>', 'bits=*reinterpret_cast<const uint2*>'),
        ('shared stage changed', 'stage[2048];', 'stage[1536];'),
        ('V wait changed', '__ivcorex_sl_waitcnt(8ULL);', '__ivcorex_sl_waitcnt(0ULL);'),
    ):
        assert cuda.count(old) == 1, label
        try:
            f.inverse(cuda.replace(old, new), 'qk.cu')
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError('undeclared mutation accepted: ' + label)
    assert not {'torch', 'triton', 'numpy'} & set(sys.modules)
    sources = [Path(__file__), f.ROOT/'tools/biv150_native_k8_compact_n48_kpair_factory.py',
               f.ROOT/'tools/compile_biv150_native_k8_compact_n48_kpair.py']
    result = dict(passed=True, source=proof,
        checker_source_pins={str(p.relative_to(f.ROOT)): f.sha(p.read_bytes()) for p in sources},
        counts=dict(counts), arena_base_mod16=[0, 4, 8, 12], logical_pages=pages,
        KV_heads=heads, valid_slots=list(range(17)), payload_patterns=3,
        all_byte_values_exercised=True, all_512_unique_shared_words=True,
        all_global_read_byte_multisets_equal=True, all_shared_write_address_bit_multisets_equal=True,
        source_read_mask_precedes_scalar_or_vector_load=True,
        scalar_fallback_read_and_write_sequences_exact=True,
        stage_middle_V_and_guards_unchanged=True,
        paired_shared_word_byte_delta=64, comparison_digest_sha256=digest.hexdigest(),
        source_mutations_rejected=negatives, uint2_size_alignment_offsets_checked_by_future_compiler=True,
        whole_cache_4B_non8B_device_compatibility_verified=False,
        pending='Original 2ad and candidate must both test the full cache/V SME alignment boundary.',
        device_executed=False, compile_verified=False, actual_vector_lowering_verified=False,
        actual_resources_verified=False, numeric_verified=False, performance_verified=False)
    text = json.dumps(result, indent=2, sort_keys=True)+'\n'
    if args.out:
        assert not args.out.exists()
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
