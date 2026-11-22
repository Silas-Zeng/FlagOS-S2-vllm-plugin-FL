"""CPU-only source/ABI/pointer/math falsification; no Torch, JIT, device or model."""
import argparse
import ast
from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import biv150_v8_prefill_factory as f

ROOT = Path(__file__).resolve().parents[1]
CASES = [([], []), ([0], [0]), ([1], [1]), ([1], [17]), ([2], [2]), ([7], [31]),
         ([15], [16]), ([16], [17]), ([17], [17]), ([31], [63]), ([33], [65]),
         ([1, 16], [17, 31]), ([16, 1], [31, 17]), ([0, 33], [0, 65]),
         ([33, 0], [65, 0]), ([513, 513], [513, 1025]), ([1, 1025], [17, 1025]),
         ([1025, 1], [1025, 17])]


class Opaque:
    def __eq__(self, other):
        raise AssertionError('A raw pointer/tensor must never be elementwise compared')


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset+offset)


def cpu_resolver(helper):
    """Execute only pinned scalar lookup AST as Python, with checked list loads."""
    class Scalar:
        int32 = int

        @staticmethod
        def load(pointer):
            assert 0 <= pointer.offset < len(pointer.values)
            return pointer.values[pointer.offset]
    nodes = [n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef)
             and n.name in ('find_seq_idx','resolve_seq_and_query_len')]
    for node in nodes:
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
    namespace = {'tl':Scalar}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-scalar-lookup>','exec'),namespace)
    return namespace['resolve_seq_and_query_len']


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', f32(value)))[0]
    return struct.unpack('<f', struct.pack('<I', ((word + 0x7fff + ((word >> 16) & 1)) >> 16) << 16))[0]


def views(rows, pages, pitch, padded):
    qstride = (4096, 256, 1) if padded else (2048, 128, 1)
    cache_bytes = pages * pitch + 256
    return dict(q=f.View('bf16', (rows, 16, 128), qstride, 0, rows*qstride[0]*2, 'q'),
                out=f.View('bf16', (rows, 16, 128), qstride, 0, rows*qstride[0]*2, 'out'),
                k=f.View('bf16', (pages, 16, 2, 128), (pitch//2, 256, 128, 1), 128, cache_bytes, 'cache'),
                v=f.View('i8', (pages, 16, 2, 128), (pitch, 256, 128, 1), 256+8192, cache_bytes, 'cache'),
                scales=f.View('f32', (pages, 16, 2), (pitch//4, 2, 1), (256+12288)//4, cache_bytes, 'cache'))


def raw_call(v, rows, sequences, maxq):
    bm, bq, tile, warps, stages = (128, 16, 64, 8, 1) if maxq > 1 else (16, 2, 32, None, None)
    raw = {name: False for name in ('USE_TD','USE_TD_QO','Q_IS_FP8','USE_FP8','USE_ALIBI_SLOPES',
        'USE_ALIBI_SQRT','USE_QQ_BIAS','USE_SOFTCAP','USE_SINKS','USE_PER_SEQ_CAUSAL','USE_MM_PREFIX')}
    raw.update(KV_QUANT_MODE=0, IS_3D=False, USE_CAUSAL=True, SLIDING_WINDOW=0, CHUNK_LOOKBACK=-1,
        BLOCK_M=bm, BLOCK_Q=bq, TILE_SIZE=tile, num_warps=warps, num_stages=stages,
        num_seqs=sequences, BLOCK_SIZE=16, HEAD_SIZE=128, HEAD_SIZE_PADDED=128,
        num_queries_per_kv=8, num_query_heads=16, value_cache_ptr=Opaque(), query_ptr=Opaque(),
        output_ptr=Opaque(), key_cache_ptr=Opaque(), v_scale_cache_ptr=Opaque(), stride_vs_blk=0,
        stride_vs_slot=0, stride_vs_head=0)
    raw['v_scale_cache_ptr'] = raw['value_cache_ptr']  # Actual SDK mode0 dummy binding.
    for prefix, name, count in (('stride_k_cache_','k',4),('stride_v_cache_','v',4),
                               ('query_stride_','q',2),('output_stride_','out',2)):
        raw.update({prefix+str(i): v[name].strides[i] for i in range(count)})
    return raw, dict(rows=rows, pages=v['k'].shape[0], num_seqs=sequences, max_query_len=maxq,
                     sdk_selected_2d=True, grid=(rows//bq+sequences, 2), cache_format=f.FORMAT)


def reject(action):
    try:
        action()
    except (f.ContractError, ValueError):
        return
    raise AssertionError('Mutant was accepted')


def main(args):
    built = f.build(args.attention_source, args.helper_source, ROOT/'vllm_fl/patches/biv150_prefill_normalized.py')
    tree = ast.parse(built['source'])
    index = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'private_v_scale_idx' for t in n.targets))
    index = compile(ast.Expression(body=index), '<actual-private-scale-expression>', 'eval')
    need_guard = 'tl.static_assert(output_ptr.dtype.element_ty == tl.bfloat16'
    assert need_guard in built['source']
    resolve = cpu_resolver(f.pinned(args.helper_source,f.HELPER_SHA))
    geometry = []; indices = 0; math_cases = 0; routed_rows = 0; max_algebra_error = 0.
    for qlens, klens in CASES:
        rows, sequences, maxq = sum(qlens), len(qlens), max(qlens, default=0)
        counts = [(x+15)//16 for x in klens]
        pages = max(1, sum(counts)+1)
        for pitch, padded in ((16384, False), (32768, True)):
            v = views(rows, pages, pitch, padded)
            raw, meta = raw_call(v, rows, sequences, maxq)
            scale_handle = Opaque()
            bound, contract = f.bind_private(raw, v, scale_handle, **meta)
            assert contract['no_launch'] is (rows == 0) and not contract['ordinary_bf16_fallback_allowed']
            assert bound['v_scale_cache_ptr'] is scale_handle
            assert all(bound[k] is raw[k] for k in raw if not k.startswith('stride_vs_') and k != 'v_scale_cache_ptr')
            covered = []; starts = [0]
            for qlen in qlens:
                starts.append(starts[-1]+qlen)
            if rows:
                for program in range(meta['grid'][0]):
                    seq, local, start, qlen, kvlen = resolve(Pointer(starts),Pointer(klens),program,sequences,raw['BLOCK_Q'])
                    assert qlen == qlens[seq] and kvlen == klens[seq]
                    if local*raw['BLOCK_Q'] >= qlen:
                        continue
                    covered.extend(range(start+local*raw['BLOCK_Q'],start+min(qlen,(local+1)*raw['BLOCK_Q'])))
            assert sorted(covered) == list(range(rows))
            routed_rows += len(covered)
            used = list(reversed(range(1, pages))); offset = 0; tables = []
            for count in counts:
                width = ((count + raw['TILE_SIZE']//16 - 1)//(raw['TILE_SIZE']//16))*(raw['TILE_SIZE']//16)
                tables.append(used[offset:offset+count] + [0]*(width-count)); offset += count
            for seq, length in enumerate(klens):
                for position in range(0, len(tables[seq])*16):
                    physical = tables[seq][position//16]
                    if position >= length:
                        # Original table gather exists; added scale load must not dereference this lane.
                        if position//16 >= counts[seq]:
                            assert physical == 0
                        continue
                    for head in range(2):
                        expected_byte = 256 + physical*pitch + 12288 + (position%16)*8 + head*4
                        actual_element = eval(index, {}, dict(physical_block_idx=physical, stride_vs_blk=pitch//4,
                            seq_offset=position, BLOCK_SIZE=16, stride_vs_slot=2, kv_head_idx=head, stride_vs_head=1))
                        assert (v['scales'].offset+actual_element)*4 == expected_byte
                        code_byte = v['v'].offset + physical*v['v'].strides[0] + (position%16)*256 + head*128 + 127
                        key_byte = (v['k'].offset + physical*v['k'].strides[0] + (position%16)*256 + head*128 + 127)*2
                        assert 256+physical*pitch+8192 <= code_byte < 256+physical*pitch+12288
                        assert 256+physical*pitch <= key_byte < 256+physical*pitch+8192
                        indices += 1
            if not padded:
                # Global normalized FP64 reference, independent of any streaming/segment recurrence.
                for seq, (qlen, kvlen) in enumerate(zip(qlens, klens)):
                    for query in sorted({0, qlen-1}) if qlen else []:
                        visible = kvlen-qlen+query+1
                        for head in range(2):
                            scores = [((tables[seq][t//16]*7+t%16+head)%17-8)/8 for t in range(visible)]
                            scales = [f32((1+(t+head)%7)/127) for t in range(visible)]
                            codes = [((t*11+head*7)%255)-127 for t in range(visible)]
                            maximum = max(scores); weights = [math.exp(s-maximum) for s in scores]
                            denominator = math.fsum(weights)
                            original = math.fsum(p/denominator*(c*s) for p,c,s in zip(weights,codes,scales))
                            folded = math.fsum((p*s)*c for p,c,s in zip(weights,codes,scales))/denominator
                            error = abs(original-folded); assert error <= 1e-12
                            max_algebra_error = max(max_algebra_error,error); math_cases += 1
            geometry.append(dict(qlens=qlens, klens=klens, page_pitch_bytes=pitch,
                                 q_o_padded=padded, grid=meta['grid'], no_launch=contract['no_launch']))
    # The actual source expression retains 64-bit physical addressing beyond int32.
    large = eval(index, {}, dict(physical_block_idx=2**30+3, stride_vs_blk=8192,
        seq_offset=31, BLOCK_SIZE=16, stride_vs_slot=2, kv_head_idx=1, stride_vs_head=1))
    assert large == (2**30+3)*8192+31 and large > 2**43
    assert all(bf16(c) == c for c in range(-127,128))
    assert math.isnan(struct.unpack('<f', bytes([0,0,0x81,0x7f]))[0])
    legacy_small = struct.unpack('<f', bytes([0,0,0x7f,0]))[0]
    assert legacy_small != 127.
    p = f32(.5+2**-9); s = f32(2/127); code = 32
    assert bf16(p)*.5 == .25 and bf16(f32(p*s))*code == .25390625
    assert f32(0.) == 0. and bf16(0.) == 0.  # Zero-vector codec requires scale1/code0.
    v = views(18, 5, 32768, True); raw, meta = raw_call(v,18,2,17)
    failures = []
    for label in ('bf16-v','scale-dtype','scale-offset','zero-scale-stride','cache-stride','q-raw-stride','out-raw-stride',
                  'dummy-scale','missing-scale','out-dtype','none-mode','3d','changed-tile','wrong-grid','unknown-format','wrong-dispatch','aliases'):
        vv, rr, mm, handle = dict(v), dict(raw), dict(meta), Opaque()
        if label == 'bf16-v': vv['v'] = replace(vv['v'],dtype='bf16')
        elif label == 'scale-dtype': vv['scales'] = replace(vv['scales'],dtype='bf16')
        elif label == 'scale-offset': vv['scales'] = replace(vv['scales'],offset=12288)
        elif label == 'zero-scale-stride': vv['scales'] = replace(vv['scales'],strides=(0,2,1))
        elif label == 'cache-stride': rr['stride_v_cache_0'] = 16384
        elif label == 'q-raw-stride': rr['query_stride_0'] = 2048
        elif label == 'out-raw-stride': rr['output_stride_1'] = 128
        elif label == 'dummy-scale': handle = rr['value_cache_ptr']
        elif label == 'missing-scale': handle = None
        elif label == 'out-dtype': vv['out'] = replace(vv['out'],dtype='f32')
        elif label == 'none-mode': rr['KV_QUANT_MODE'] = 2
        elif label == '3d': rr['IS_3D'] = True
        elif label == 'changed-tile': rr['num_warps'] = 16
        elif label == 'wrong-grid': mm['grid'] = (2,2)
        elif label == 'unknown-format': mm['cache_format'] = 'bf16'
        elif label == 'wrong-dispatch': mm['sdk_selected_2d'] = False
        elif label == 'aliases': vv['out'] = replace(vv['out'],storage_id='q')
        reject(lambda: f.bind_private(rr,vv,handle,**mm)); failures.append(label)
    with tempfile.TemporaryDirectory(prefix='v8-prefill-cpu-') as tmp:
        for label, original, helper in (('sdk-sha',args.attention_source,False),('helper-sha',args.helper_source,True)):
            path = Path(tmp)/label; path.write_bytes(Path(original).read_bytes()+b'\n')
            reject(lambda: f.build(args.attention_source if helper else path,
                path if helper else args.helper_source,ROOT/'vllm_fl/patches/biv150_prefill_normalized.py'))
            failures.append(label)
    reject(lambda: f.restore(built['source'].replace('S += score_scale * tl.dot(Q, K)', 'S += 2 * score_scale * tl.dot(Q, K)')))
    failures.append('non-V-QK-edit')
    with tempfile.TemporaryDirectory(prefix='v8-prefill-normalizer-') as tmp:
        path = Path(tmp)/'normalizer.py'
        path.write_bytes((ROOT/'vllm_fl/patches/biv150_prefill_normalized.py').read_bytes()+b'\n')
        reject(lambda: f.build(args.attention_source,args.helper_source,path))
        failures.append('normalizer-sha')
    with tempfile.TemporaryDirectory(prefix='v8-prefill-cli-') as tmp:
        out = Path(tmp)/'generated'
        command = [sys.executable,str(ROOT/'tools/biv150_v8_prefill_factory.py'),
                   '--attention-source',str(args.attention_source),'--helper-source',str(args.helper_source),
                   '--out',str(out)]
        child = subprocess.run(command,capture_output=True,text=True)
        assert child.returncode == 0, child.stderr
        assert f.digest((out/'private_v8_prefill.py').read_text()) == built['metadata']['candidate_sha256']
        ast.parse((out/'private_v8_prefill.py').read_text())
        assert subprocess.run(command,capture_output=True,text=True).returncode != 0
    assert f.digest(f.restore(built['source'])) == f.NORMALIZED_SHA
    assert not {'torch','triton','vllm'} & set(sys.modules)
    result = dict(cpu_only=True, checks_passed=True, gpu_executed=False, jit_executed=False,
        production_installed=False, precision_gate=False, queued=False, model_evaluated=False,
        sources=built['metadata'], geometries=geometry, pointer_indices_checked=indices, actual_helper_query_rows_checked=routed_rows,
        ideal_scale_fold_cases=math_cases, ideal_scale_fold_max_abs=max_algebra_error,
        rejected=failures, opaque_raw_handle_equality_never_called=True,
        source_generation_cli_passed=True, existing_source_output_rejected=True,
        legacy_bf16_nan_counterexample=True, legacy_bf16_small_value=legacy_small,
        nonpreserved_original_term=.25, nonpreserved_weighted_term=.25390625,
        limitation='Source and CPU scalar contracts only, not device compilation/FP32 dot/FTZ, resources, numeric tolerance or model acceptance.')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('cpu_only','checks_passed','pointer_indices_checked','ideal_scale_fold_cases','precision_gate')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attention-source',type=Path,required=True); p.add_argument('--helper-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
