#!/usr/bin/env python3
"""CPU byte/source contract only; never imports a framework or installs a hook."""
import argparse
import ast
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/konly-cache-lifecycle-20261027'
PAGE, CODES, SCALES, VBASE = 16384, 4096, 4096, 8192
FORMAT = 'cjj-konly-rn32-rne-v2'
ARCHIVES = {
    'service-io-target-source-v1.tar.gz': 'd9915caff7883cdcae6d04a9bf8df5212c82864be3c8af299b66f3c8f4414688',
    'attention-context-source-v1.tar.gz': 'eac530fbc7a1d10453144c01764283d848e41d762566dd3883b76e26f7130781',
    'lifecycle-target-source-v1.tar.gz': 'f154e9e38ece8244b5b5cf233fccacc6d1a37e4840b19d13288473ea84b2ee2e',
    'rope-fusion-target-source-v1.tar.gz': '4b5a47a4c172e33e64eea8167b28a6c4303e52e14eb024adb40c113e7a3cbb3b',
    'zeroer-platform-source-v1.tar.gz': '9cc96740eef71bc0e27d43c8e98184c2aede94d54b18f2d2f5fa58fc72b1206a',
}
REPO_SOURCES = {
    'vllm_fl/platform.py': 'aaf14ee641edb2d02f0c9d6ec52593d73086cae5d05df463404129a7fb9af773',
    'vllm_fl/worker/model_runner.py': '0161c595ff2f1b115039f03980fbb6de0447f1edf0d55fba1793b6a8055c230c',
    'vllm_fl/patches/biv150_mixed_decode.py': 'a129361f1ef1babe1ce1e6238f6b1a725050612c235201064ddeba676b420c42',
    'vllm_fl/patches/biv150_compute_order.py': 'e60909d8832505566f172ecfe69c63728a5bfdfe3e08f40582fc771ce92cf17b',
    'vllm_fl/dispatch/backends/flaggems/impl/attention.py': 'f86e1cc67a1a3316c6d437ba96a6e0dcf19ebe29b22178d0f4f34fa360d65c57',
    'vllm_fl/compilation/graph.py': 'ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


@dataclass(frozen=True)
class Layout:
    """Explicit hypothetical NHD contract; not an observed Tensor descriptor."""
    pages: int = 3
    shape_tail: tuple = (2, 16, 2, 128)
    bf16_strides: tuple = (8192, 4096, 256, 128, 1)
    storage_byte_offset: int = 0
    dtype: str = 'bfloat16'
    quant_mode: int = 0
    format: str = FORMAT

    def validate(self):
        need(type(self.pages) is int and 0 < self.pages <= 2**31-1, 'physical page count')
        need(type(self.shape_tail) is tuple and all(type(x) is int for x in self.shape_tail) and self.shape_tail == (2, 16, 2, 128), 'shape')
        need(type(self.bf16_strides) is tuple and all(type(x) is int for x in self.bf16_strides) and self.bf16_strides == (8192, 4096, 256, 128, 1), 'NHD strides')
        need(type(self.storage_byte_offset) is int and self.storage_byte_offset == 0, 'offset')
        need(type(self.dtype) is str and self.dtype == 'bfloat16' and type(self.quant_mode) is int and self.quant_mode == 0, 'stock wrapper')
        need(type(self.format) is str and self.format == FORMAT, 'explicit format')


def slot_spans(slot):
    need(type(slot) is int and 0 <= slot < 16, 'slot')
    return ((slot*256, (slot+1)*256),
            (SCALES+slot*8, SCALES+(slot+1)*8),
            (VBASE+slot*512, VBASE+(slot+1)*512))


def append_bytes(storage, layout, physical_slot, code, scale_bits, value):
    """Model an already-encoded input write. No codec/rotation is implemented."""
    layout.validate()
    need(type(storage) is bytearray and len(storage) == layout.pages*PAGE, 'allocation')
    need(type(physical_slot) is int, 'physical slot type')
    if physical_slot < 0:
        return False  # Skip before touching padded/unmapped token payloads.
    need(physical_slot < layout.pages*16, 'physical slot range')
    need(type(code) is bytes and len(code) == 256 and 128 not in code, 'signed code')
    need(type(scale_bits) is tuple and len(scale_bits) == 2, 'two scales')
    need(all(type(b) is int and 0x00800000 <= b <= 0x7f7fffff for b in scale_bits), 'positive finite normal scale')
    need(type(value) is bytes and len(value) == 512, 'original V bytes')
    page, slot = divmod(physical_slot, 16)
    # All validation precedes the first store. Source payloads are distinct bytes.
    for (start, end), payload in zip(slot_spans(slot), (code, struct.pack('<2I', *scale_bits), value)):
        storage[page*PAGE+start:page*PAGE+end] = payload
    return True


def checked_sources():
    result, sources, copied_sources = [], {}, []
    for name, digest in ARCHIVES.items():
        data = (EVIDENCE/name).read_bytes()
        need(hashlib.sha256(data).hexdigest() == digest, 'source archive SHA')
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            members = tar.getmembers()
            need(all(m.isfile() and 0 <= m.size < 600000 for m in members), 'source members')
            need(len({m.name for m in members}) == len(members), 'duplicate source')
            files = {m.name: tar.extractfile(m).read() for m in members}
        manifest = json.loads(files['manifest.json'])
        rows = manifest['files']
        if type(rows) is dict:
            rows = [dict(relative=k, **v) for k, v in rows.items()]
        need(set(files) == {'manifest.json'} | {r['relative'] for r in rows}, 'source inventory')
        for row in rows:
            body = files[row['relative']]
            need(len(body) == row['bytes'] and hashlib.sha256(body).hexdigest() == row['sha256'], 'source SHA')
            ast.parse(body)  # Parse only: no imports or evaluation of annotations.
            if row['relative'] in sources:
                need(sources[row['relative']] == body, 'same path different source')
            sources[row['relative']] = body
            copied_sources.append((row['relative'], row['sha256']))
        result.append(dict(file=name, sha256=digest, bytes=len(data), sources=len(rows),
                           source_bytes=sum(r['bytes'] for r in rows)))
    # Duplicate installation trees are retained as evidence, not counted as
    # distinct active implementations. Canonical and nested copies must agree.
    canonical = {}
    for name, digest in copied_sources:
        if name.startswith('vllm/vllm/'):
            name = name.replace('vllm/vllm/', 'vllm/', 1)
        elif name.startswith(('v1/', 'model_executor/')) or name == 'forward_context.py':
            name = 'vllm/'+name
        need(name not in canonical or canonical[name] == digest, 'canonical alias differs')
        canonical[name] = digest
    inventory = dict(downloaded_path_copies=len(copied_sources),
                     unique_content_sha256=len(set(d for _, d in copied_sources)),
                     canonical_paths=len(canonical),
                     source_bytes_with_copies=sum(r['source_bytes'] for r in result))
    for name, digest in REPO_SOURCES.items():
        data = (ROOT/name).read_bytes()
        need(hashlib.sha256(data).hexdigest() == digest, 'main source '+name)
        sources[name] = data
    anchors = {}
    wanted = {
        'v1/attention/backends/triton_attn.py': ('get_kv_cache_shape', 'do_kv_cache_update', 'do_rope_and_kv_cache_update', 'fused_rope_kvcache_supported'),
        'v1/worker/block_table.py': ('swap_row', '_compute_slot_mapping_kernel'),
        'v1/core/kv_cache_manager.py': ('get_computed_blocks', 'free', 'reset_prefix_cache'),
        'v1/core/kv_cache_utils.py': ('hash_block_tokens', 'generate_block_hash_extra_keys'),
        'model_executor/layers/attention/attention.py': ('get_attention_context', 'unified_kv_cache_update', 'unified_attention_with_output'),
        'vllm/v1/attention/ops/triton_reshape_and_cache_flash.py': ('reshape_and_cache_kernel_flash', 'triton_reshape_and_cache_flash'),
        'vllm/v1/worker/gpu/attn_utils.py': ('_reshape_attention_kv_cache',),
        'vllm/distributed/kv_transfer/kv_connector/utils.py': ('copy_kv_blocks', 'kv_postprocess_layout_on_receive', 'kv_postprocess_blksize_on_receive', 'kv_postprocess_blksize_and_layout_on_receive'),
        'vllm/v1/worker/kv_connector_model_runner_mixin.py': ('_get_kv_connector_output', 'finalize_kv_connector'),
        'vllm/v1/core/block_pool.py': ('get_new_blocks', 'touch', 'free_blocks', 'cache_full_blocks', 'cache_partial_block'),
        'vllm/v1/core/single_type_kv_cache_manager.py': ('add_local_computed_blocks', 'allocate_new_blocks', 'free'),
        'vllm/_custom_ops.py': ('swap_blocks', 'gather_and_maybe_dequant_cache'),
        'flag_gems/fused/reshape_and_cache_flash.py': ('reshape_and_cache_flash_kernel', 'reshape_and_cache_flash'),
        'vllm_fl/platform.py': ('insert_blocks_to_device', 'swap_out_blocks_to_host'),
        'vllm_fl/worker/model_runner.py': ('_get_slot_mappings', '_reshape_kv_cache_tensors', 'initialize_kv_cache_tensors', '_zero_block_ids'),
        'vllm_fl/compilation/graph.py': ('__call__', 'clear_graphs'),
        'vllm/compilation/passes/fusion/rope_kvcache_fusion.py': ('fused_rope_and_unified_kv_cache_update_impl', '_mk_pattern_with_layer_name_input', '_mk_pattern_with_layer_name_closure', 'is_applicable_for_range'),
        'vllm/_aiter_ops.py': ('is_aiter_found_and_supported', 'if_aiter_supported', 'triton_rope_and_cache'),
        'vllm/v1/worker/utils.py': ('_zero_kv_blocks_kernel', 'zero_block_ids'),
        'vllm/platforms/interface.py': ('is_rocm',),
    }
    for name, methods in wanted.items():
        tree = ast.parse(sources[name])
        for method in methods:
            found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method]
            need(bool(found), 'missing actual symbol '+method)
            anchors[name+':'+method] = [[n.lineno, n.end_lineno] for n in found]
    tree = ast.parse(sources['vllm/compilation/passes/fusion/rope_kvcache_fusion.py'])
    registrations = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'direct_register_custom_op']
    need(len(registrations) == 1, 'fused op registration')
    kws = {k.arg:k.value for k in registrations[0].keywords}
    need(ast.literal_eval(kws['op_name']) == 'fused_rope_and_unified_kv_cache_update', 'fused op name')
    need(ast.literal_eval(kws['mutates_args']) == ['query','key'], 'fused mutated arguments')
    implementations = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'fused_rope_and_unified_kv_cache_update_impl']
    need(len(implementations) == 1, 'fused implementation')
    last = implementations[0].body[-1]
    need(isinstance(last, ast.Return) and isinstance(last.value, ast.Call)
         and ast.unparse(last.value.func) == 'torch.empty' and ast.literal_eval(last.value.args[0]) == 0,
         'fused return dependency')
    pattern = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RopeReshapeKVCachePattern']
    need(len(pattern) == 1, 'reshape fusion pattern')
    replacements = [n for n in ast.walk(pattern[0]) if isinstance(n, ast.FunctionDef) and n.name == 'replacement']
    need(len(replacements) == 2 and all(ast.unparse(n.body[-1]) == 'return (results[0], results[1], results[2], v)'
                                      for n in replacements), 'functionalized Q/K outputs')
    tree = ast.parse(sources['vllm/_aiter_ops.py'])
    wrappers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'triton_rope_and_cache']
    need(len(wrappers) == 1, 'AITER wrapper')
    calls = [n for n in ast.walk(wrappers[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'fused_qk_rope_reshape_and_cache']
    need(len(calls) == 1, 'AITER called implementation')
    bindings = {k.arg:ast.unparse(k.value) for k in calls[0].keywords}
    need(bindings['q_out'] == 'query' and bindings['k_out'] == 'key', 'AITER inplace outputs')
    decorators = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'if_aiter_supported']
    need(len(decorators) == 1, 'AITER decorator')
    wrapped = [n for n in decorators[0].body if isinstance(n, ast.FunctionDef) and n.name == 'wrapper']
    need(len(wrapped) == 1 and ast.unparse(wrapped[0].body[-1]) == 'return None', 'AITER unsupported return')
    tree = ast.parse(sources['vllm_fl/platform.py'])
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PlatformFL']
    need(len(cls) == 1, 'FL platform class')
    enums = [n for n in cls[0].body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_enum' for t in n.targets)]
    need(len(enums) == 1 and ast.unparse(enums[0].value) == 'PlatformEnum.OOT'
         and not any(isinstance(n, ast.FunctionDef) and n.name == 'is_rocm' for n in cls[0].body), 'FL platform enum')
    tree = ast.parse(sources['vllm/platforms/interface.py'])
    rocm = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'is_rocm']
    need(len(rocm) == 1 and ast.unparse(rocm[0].body[-1]) == 'return self._enum == PlatformEnum.ROCM', 'inherited ROCm test')
    fusion = dict(custom_op_mutates=['query','key'], return_is_dummy=True,
                  aiter_explicit_outputs={k:bindings[k] for k in ('q_out','k_out')},
                  unsupported_aiter_decorator_returns=None, pinned_fl_source_is_rocm=False,
                  actual_platform_binding_observed=False, actual_fusion_observed=False,
                  underlying_aiter_device_implementation_verified=False)
    return result, anchors, inventory, fusion


def run():
    archives, anchors, inventory, fusion = checked_sources()
    layout = Layout()
    layout.validate()
    # Exact disjoint physical regions and per-slot occupancy, independent of values.
    all_spans = [span for s in range(16) for span in slot_spans(s)]
    occupied = set()
    for lo, hi in all_spans:
        need(not occupied.intersection(range(lo, hi)), 'overlap')
        occupied.update(range(lo, hi))
    need(len(occupied) == 12416 and occupied == set(range(4224)) | set(range(8192, 16384)), 'payload geometry')
    # Relative code offsets are128B aligned; relative scale offsets are4B aligned.
    need(all((s*256+h*128) % 128 == 0 and (4096+s*8+h*4) % 4 == 0 for s in range(16) for h in range(2)), 'alignment')
    legal_large_page = 2**31-1
    need(legal_large_page*PAGE+16383 == 2**45-1, 'i64 legal int32 page product')
    checks = 0
    for slot in range(layout.pages*16):
        before = bytes((i*37+11) % 256 for i in range(layout.pages*PAGE))
        memory = bytearray(before)
        code = bytes((i % 127)+1 for i in range(256))
        value = bytes((i*29+slot) % 256 for i in range(512))
        scales = (0x3f800000, 0x00800000)
        need(append_bytes(memory, layout, slot, code, scales, value), 'append skipped')
        page, local = divmod(slot, 16)
        touched = {page*PAGE+i for lo, hi in slot_spans(local) for i in range(lo, hi)}
        need(all(a == b for i, (a, b) in enumerate(zip(memory, before)) if i not in touched), 'adjacent slot/padding modified')
        for (lo, hi), payload in zip(slot_spans(local), (code, struct.pack('<2I', *scales), value)):
            need(memory[page*PAGE+lo:page*PAGE+hi] == payload, 'wrong append payload')
        checks += 1
    # Negative/padded slots must not inspect payloads or mutate the cache.
    memory = bytearray(PAGE*3)
    for slot in (-1, -17):
        need(not append_bytes(memory, layout, slot, object(), object(), object()) and not any(memory), 'invalid slot read/write')
    # Old BF16 writer at slot8 overwrites all32 scale words; patching only slot8
    # restores two, leaving the scales of earlier live prefix tokens corrupted.
    before = bytearray(b'\x5a'*PAGE)
    before[4096:4224] = struct.pack('<32I', *([0x3f800000]*32))
    broken = bytearray(before)
    broken[8*512:9*512] = b'\x11'*512
    append_bytes(broken, Layout(pages=1), 8, bytes(256), (0x3f800000,)*2, bytes(512))
    corrupt_scales = [s for s in range(16) if broken[4096+s*8:4104+s*8] != before[4096+s*8:4104+s*8]]
    need(corrupt_scales == [s for s in range(16) if s != 8], 'legacy writer counterexample')
    # The old HND->NHD BF16 word permutation preserves each word but does not
    # preserve this private NHD scale region. No arithmetic is needed to break it.
    old_words = struct.unpack('<4096H', before[:8192])
    rearranged = tuple(old_words[h*16*128+s*128+d] for s in range(16) for h in range(2) for d in range(128))
    repacked = struct.pack('<4096H', *rearranged)
    need(sorted(old_words) == sorted(rearranged) and repacked[4096:4224] != before[4096:4224], 'logical-copy counterexample')
    # Test all65536 BF16 bit patterns as opaque words, including NaN payloads.
    # This proves the CPU transfer model, not a device implementation of copy_.
    patterns = struct.pack('<65536H', *range(65536))
    need(struct.pack('<65536H', *struct.unpack('<65536H', patterns)) == patterns, 'same-width bit copy')
    raw = bytes((i*17+3) % 256 for i in range(PAGE*16))
    raw = bytearray(raw)
    for p in range(16):
        raw[p*PAGE+VBASE:(p+1)*PAGE] = patterns[p*8192:(p+1)*8192]
    reordered = b''.join(raw[p*PAGE:(p+1)*PAGE] for p in reversed(range(16)))
    need(b''.join(reordered[p*PAGE:(p+1)*PAGE] for p in reversed(range(16))) == raw, 'complete page copy')
    field_copy = b''.join(bytes(raw[p*PAGE:p*PAGE+8192])+bytes(raw[p*PAGE+8192:(p+1)*PAGE]) for p in range(16))
    need(field_copy == raw, 'full K/V field copy')
    # Mirror only the reviewed fixed-layout zeroer's address enumeration:
    # one segment per layer,4096 I32 words/page, four1024-word chunks.
    zeroed = bytearray(raw)
    seen = set()
    for chunk in range(4):
        for column in range(1024):
            byte = (3*4096+chunk*1024+column)*4
            need(byte not in seen, 'duplicate zero word')
            seen.add(byte)
            zeroed[byte:byte+4] = bytes(4)
    need(len(seen) == 4096 and not any(zeroed[3*PAGE:4*PAGE])
         and zeroed[:3*PAGE] == raw[:3*PAGE] and zeroed[4*PAGE:] == raw[4*PAGE:], 'full-page zero bounds')
    # A QKV split view may keep the wider fused-projection row pitch.
    # The production wrapper passes these strides; the private control's
    # hardcoded token*256 cannot be assumed to implement that source ABI.
    source_pitch_cases = []
    for pitch in (256, 2560):
        source = struct.pack('<'+str(pitch*3+256)+'H', *range(pitch*3+256))
        expected = [source[t*pitch*2:t*pitch*2+512] for t in range(3)]
        old = [source[t*256*2:t*256*2+512] for t in range(3)]
        need((old == expected) == (pitch == 256), 'source pitch counterexample')
        source_pitch_cases.append(dict(bf16_row_stride=pitch, old_contiguous_control_matches=old == expected))
    # Arithmetic may quiet a signalling NaN; no arithmetic is needed by bit-copy.
    bf16 = 0x7f81  # Valid I8 bytes81,7f also look like a BF16 NaN.
    need(struct.unpack('<H', bytes((129, 127)))[0] == bf16, 'valid-code BF16 alias')
    fp = struct.unpack('<f', struct.pack('<I', bf16 << 16))[0]
    converted = struct.unpack('<I', struct.pack('<f', fp+0.0))[0] >> 16
    need(converted != bf16 and converted & 0x7f80 == 0x7f80, 'arithmetic-copy counterexample')
    old_read = struct.unpack('<f', struct.pack('<I', 0x007f << 16))[0]
    need(old_read != 127.0, 'codes127,0 falsely treated as original K')
    # Existing platform helper's axis1 indexing is not a page index for[P,2,...].
    triton_axes = [[(page, field) for field in range(2)] for page in range(3)]
    try:
        [fields[2] for fields in triton_axes]
    except IndexError:
        pass
    else:
        raise AssertionError('Triton cache accepted axis1 page2')
    faults = []
    def reject(name, f):
        try:
            f()
        except (ValueError, TypeError):
            faults.append(name)
        else:
            raise AssertionError('accepted fault '+name)
    for name, changed in [('HND', dict(bf16_strides=(8192,4096,128,2048,1))),
                          ('padding', dict(bf16_strides=(16384,4096,256,128,1))),
                          ('offset', dict(storage_byte_offset=4)), ('old-format', dict(format='BF16')),
                          ('mode2', dict(quant_mode=2)), ('bad-dtype', dict(dtype='int8')),
                          ('bool-page-count', dict(pages=True)), ('wrong-shape', dict(shape_tail=(2,32,2,128)))]:
        reject(name, lambda changed=changed: replace(layout, **changed).validate())
    for name, slot, code, scales, value in [
            ('out-of-bounds', 48, bytes(256), (0x3f800000,)*2, bytes(512)),
            ('bool-slot', True, bytes(256), (0x3f800000,)*2, bytes(512)),
            ('code-minus128', 0, b'\x80'*256, (0x3f800000,)*2, bytes(512)),
            ('scale-NaN', 0, bytes(256), (0x7fc00000,)*2, bytes(512)),
            ('wrong-V-size', 0, bytes(256), (0x3f800000,)*2, bytes(2))]:
        old = bytes(memory)
        reject(name, lambda: append_bytes(memory, layout, slot, code, scales, value))
        need(bytes(memory) == old, 'rejected write mutated destination')
    need(not {'torch','triton','numpy'}.intersection(sys.modules), 'framework imported')
    return dict(cpu_only=True, passed=True, base_main='56f1838192abc940b3fb9accef490022ac125000',
                source_archives=archives, source_inventory=inventory, fusion_contract=fusion,
                main_sources=REPO_SOURCES, source_anchors=anchors, explicit_layout=layout.__dict__,
                page_bytes=PAGE, active_payload_bytes=len(occupied), capacity_reclaimed_bytes=0,
                append_slots=checks, preserved_words=65536, opaque_copy_pages=16,
                legacy_writer_slot8_corrupted_scale_slots=corrupt_scales,
                logical_copy_counterexample=dict(word_bits_preserved=True, private_scale_region_preserved=False),
                numeric_copy_counterexample={'original_bf16_bits':bf16,'after_add_zero':converted},
                source_row_pitch_counterexamples=source_pitch_cases,
                zeroer_fixed_nhd_contract=dict(segments_per_layer=1, int32_words_per_page=4096,
                                              chunks_per_page=4, clear_before_valid_writer=True,
                                              actual_device_verified=False),
                rejected=faults, source_io_closed=False, real_tensor_strides_observed=False,
                codec_implemented=False, service_hook_installed=False, device_copy_verified=False,
                graph_verified=False, active_platform_copy_observed=False, precision_gate=False, gpu_work=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True)+'\n'
    if args.out:
        need(not args.out.exists(), 'output exists')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    else:
        print(result, end='')
