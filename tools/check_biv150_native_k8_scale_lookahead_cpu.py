#!/usr/bin/env python3
"""Bounded CPU address/carry check; not a GPU or instruction scheduling model."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import biv150_native_k8_scale_lookahead_factory as f


def scale_address(page, head, col):
    return page*16384+4096+(col*2+head)*4


def carry_trace(full, tail, valid, head, col, *, stale=False, overread=False):
    """Only address identities are values here; no numerical GPU result claimed."""
    reads, consumed = [], []
    carried = None
    if full:
        carried = scale_address(full[0], head, col)
        reads.append(carried)
    for index, page in enumerate(full):
        consumed.append(carried)
        if index+1 < len(full):
            next_scale = scale_address(full[index+1], head, col)
            reads.append(next_scale)
            if not stale:
                carried = next_scale
        elif overread:
            reads.append(scale_address(page+1, head, col))
    if tail is not None and col < valid:
        address = scale_address(tail, head, col)
        reads.append(address); consumed.append(address)
    expected = [scale_address(p, head, col) for p in full]
    if tail is not None and col < valid:
        expected.append(scale_address(tail, head, col))
    assert reads == expected and consumed == expected
    return len(reads)


def reject(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Unsafe mutation accepted')


def check():
    proof = f.verify()
    source = f.path('qk.cu').read_text()
    parent = f.original('qk.cu')
    tail = lambda s: s.split(' if(full_upper<upper){', 1)[1].split('\n }', 1)[0]
    assert tail(source) == tail(parent)
    assert source.count('float* scale_carry=nullptr') == 1
    assert source.count('acc,has_next,row,next_position,&kscale);') == 1
    assert source.count('*scale_carry=next_kscale;') == 1
    assert 'KReady?*scale_carry:(col<valid?' in source
    body = source.split('__device__ __forceinline__ int consume_page(', 1)[1]
    body = body.split('\n}\n}', 1)[0]
    ordered = [
        '__ivcorex_sl_waitcnt(8ULL);', 'if(valid<16)clear_v_tail(stage,valid,lane);',
        '  __syncthreads();', 'if(has_next){', 'next_page=next_row[next_position/16];',
        'read_next_k(next,key,next_page,head,lane);', 'next_kscale=key_scale_full[',
        'nvcuda::wmma::mma_sync_tcu(acc[n],probability,b,acc[n]);',
        'publish_next_k(stage,next,lane);', '*scale_carry=next_kscale;',
        '__syncthreads(); // Publish dependent K stores;',
    ]
    positions = [body.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert body.count('if(has_next){') == 2
    full = source.split(' if(position<full_upper){', 1)[1].split(' while(position<full_upper){', 1)[0]
    assert full.index('int page=row[position/16];') < full.index('float kscale=key_scale_full[')
    assert full.index('float kscale=key_scale_full[') < full.index('issue_k_i8(')
    # All scale lanes/heads stay in the same exact 128B page-scale region,
    # including maximum I32 page ID. This proves index arithmetic, not allocation.
    addresses = 0
    for page in (0, 1, 2**31-1):
        unique = set()
        for head in (0, 1):
            for lane in range(64):
                col = lane % 16
                actual = 4*(page*4096+1024+col*2+head)
                assert actual == scale_address(page, head, col)
                assert page*16384+4096 <= actual < actual+4 <= page*16384+4224
                assert actual % 4 == 0 and actual+4 < 2**63
                unique.add(actual); addresses += 1
        assert len(unique) == 32
    segments_checked = scale_reads = empty = 0
    for length in (*range(66),127,128,129,193,257,4095,4096,4097,16383,16384,16385):
        for splits in (2, 4):
            steps = (length+splits*16-1)//(splits*16)
            for segment in range(splits):
                lower = segment*steps*16
                upper = min(lower+steps*16, length)
                live = max(upper-lower, 0)
                count, valid = divmod(live, 16)
                # Noncontiguous physical pages distinguish stale carries; max
                # I32 page is included, but never used to size an allocation.
                physical = lambda i: 2**31-1-(lower//16+i)*3
                pages = [physical(i) for i in range(count)]
                tail_page = physical(count) if valid else None
                assert count+(valid != 0) == (live+15)//16
                for head in (0, 1):
                    for col in range(16):
                        scale_reads += carry_trace(pages, tail_page, valid, head, col)
                segments_checked += 1; empty += live == 0
    reject(lambda: carry_trace([7,3], None, 0, 0, 0, stale=True))
    reject(lambda: carry_trace([7], None, 0, 0, 0, overread=True))
    mutations = [
        ('last_full_overread', 'next_position<full_upper', 'next_position<=full_upper'),
        ('wrong_scale_page', 'static_cast<std::int64_t>(next_page)*4096', 'static_cast<std::int64_t>(next_page)*2048'),
        ('wrong_head', '+1024+col*2+head', '+1024+col*2'),
        ('null_full_carry', 'next_position,&kscale);', 'next_position,nullptr);'),
        ('tail_unmasked', '(col<valid?key_scale_full', '(true?key_scale_full'),
        ('changed_math', 'qscale0*kscale*scale', 'qscale0*scale'),
    ]
    for name, old, new in mutations:
        assert old in source
        reject(lambda: f.inverse(source.replace(old, new), 'qk.cu'))
    entry = f.ROOT/'tools/compile_biv150_native_k8_scale_lookahead.py'
    namespace = dict(__name__='cpu_frontend_check', __file__=str(entry))
    exec(compile(f.frontend_source(), str(entry)+'::<cpu-check>', 'exec'), namespace)
    plan = namespace['cpu_plan']()
    proc = subprocess.run([sys.executable, '-B', '-W', 'error', '-S', str(entry)],
                          capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout) == plan
    with tempfile.TemporaryDirectory(prefix='cjj-scale-cpu-') as temporary:
        out = Path(temporary)/'never-created'
        for extra, message in (([], 'Root must confirm'),
                               (['--root-owned-exclusive'], 'Explicit full expected source commit')):
            proc = subprocess.run([sys.executable, '-B', '-W', 'error', '-S', str(entry),
                                   '--compile-only', '--out', str(out), *extra],
                                  capture_output=True, text=True)
            assert proc.returncode == 2 and message in proc.stderr and not out.exists()
    assert not subprocess.check_output(['git', '-C', str(f.ROOT), 'diff', f.BASE, '--', 'vllm_fl', 'csrc'])
    for filename in (f.__file__, entry, __file__):
        ast.parse(Path(filename).read_text())
    assert not {'torch', 'triton', 'numpy'}.intersection(sys.modules)
    return dict(source=proof, source_tail_call_byte_exact=True,
        source_order_only=True, scale_address_checks=addresses, segment_cases=segments_checked,
        empty_segments=empty, logical_scale_reads_checked=scale_reads,
        carry_model_only=True, rejected=['stale_carry', 'last_page_read']+[x[0] for x in mutations],
        default_plan_stdlib_only=True, Root_ack_and_exact_commit_required=True,
        production_diff_empty=True, actual_instruction_order_verified=False,
        compiled=False, kernel_launches=0, numeric_verified=False, performance_verified=False)


if __name__ == '__main__':
    print(json.dumps(check(), sort_keys=True, indent=2))
