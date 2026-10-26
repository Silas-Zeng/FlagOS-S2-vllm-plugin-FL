#!/usr/bin/env python3
"""Component-only BF16 GQA page-load pipeline; no serving installation.

Preserve the audited math, paged addressing, tail, scratch and reducer. Replace
one full-page while loop with an exactly inverse-checked tl.range and explicit
pipeline stages. Actual IR/resources and device measurements decide viability.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys

MAIN = '241220723a09061c182764f8f738e5ec01190367'
JIT = '/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py'
JIT_SHA = '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
MODEL_4K_HASH = '641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff'


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def pipeline_source(text, stages=2):
    tree = ast.parse(text)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gqa_decode_attention']
    assert len(found) == 1
    original = found[0]
    assert sum(isinstance(n, ast.While) for n in ast.walk(original)) == 1
    candidate = copy.deepcopy(original)
    indices = [i for i, n in enumerate(candidate.body) if isinstance(n, ast.While)]
    assert len(indices) == 1
    i = indices[0]; loop = candidate.body[i]
    canonical = lambda value: ast.dump(ast.parse(value).body[0], include_attributes=False)
    assert ast.dump(candidate.body[i-1], include_attributes=False) == canonical('position = lower')
    assert ast.dump(loop.test, include_attributes=False) == ast.dump(ast.parse('position < full_upper', mode='eval').body, include_attributes=False)
    assert not loop.orelse
    assert ast.dump(loop.body[-1], include_attributes=False) == canonical('position += PAGE_SIZE')
    assert stages in (1, 2)
    replacement = ast.For(target=ast.Name(id='position', ctx=ast.Store()),
        iter=ast.Call(func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()), attr='range', ctx=ast.Load()),
            args=[ast.Name(id=x, ctx=ast.Load()) for x in ('lower','full_upper','PAGE_SIZE')],
            keywords=[ast.keyword(arg='num_stages',value=ast.Constant(value=stages))]),
        body=loop.body[:-1], orelse=[])
    candidate.body[i-1:i+1] = [replacement]
    inverse = copy.deepcopy(candidate)
    restored = inverse.body[i-1]
    assert isinstance(restored, ast.For)
    inverse.body[i-1:i] = [ast.parse('position = lower').body[0],
        ast.While(test=ast.parse('position < full_upper', mode='eval').body,
            body=restored.body+[ast.parse('position += PAGE_SIZE').body[0]],orelse=[])]
    assert ast.dump(inverse, include_attributes=False) == ast.dump(original, include_attributes=False)
    ast.fix_missing_locations(candidate)
    generated = ('import triton\nimport triton.language as tl\n'
                 'LN2 = tl.constexpr(0.6931471805599453)\n\n' + ast.unparse(candidate) + '\n')
    compile(generated, '<gqa-page-pipeline>', 'exec')
    return generated, dict(original_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generated_source_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        inverse_ast_equal=True, only_full_page_loop_changed=True, pipeline_stages=stages,
        tail_and_math_preserved=True)


def reuse_invariant_query(raw):
    """CSE one readonly duplicate Q load against its existing dominator."""
    lines = raw.splitlines(keepends=True)
    allocations = [(i, re.search(r'(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<8x128xbf16', x))
                   for i, x in enumerate(lines)]
    allocations = [(i, m.group(1)) for i, m in allocations if m]
    if len(allocations) != 1:
        raise ValueError('Unique original Q allocation required')
    alloc_index, allocation = allocations[0]
    token = re.compile(re.escape(allocation) + r'(?![A-Za-z0-9_])')
    loads = []
    for i, line in enumerate(lines):
        if token.search(line) and 'ttg.local_load ' in line:
            match = re.fullmatch(r'\s*(%[A-Za-z0-9_]+) = (ttg.local_load .*?)\n?', line)
            if match is None or 'dot_op<{opIdx = 0' not in line:
                raise ValueError('Unknown Q operand load')
            loads.append((i, match.group(1), match.group(2)))
    loops = [i for i, line in enumerate(lines) if '= scf.for ' in line]
    if len(loads) != 2 or len(loops) != 1:
        raise ValueError('Expected exactly one existing outside and one duplicate inside Q read')
    first, second = loads
    begin = loops[0]
    indent = re.match(r'\s*', lines[begin]).group()
    end = next((i for i in range(begin+1, len(lines)) if lines[i].startswith(indent+'}')), None)
    if end is None or not alloc_index < first[0] < begin < second[0] < end:
        raise ValueError('Q dominance/loop boundaries not established')
    if first[2] != second[2]:
        raise ValueError('Q pointer/type/layout/source location differ')
    allowed = {alloc_index, first[0], second[0]}
    for i, line in enumerate(lines):
        if token.search(line) and i not in allowed:
            if i <= end or 'ttg.local_dealloc ' not in line:
                raise ValueError('Unknown Q descriptor use or potential write')
    old_token = re.compile(re.escape(second[1]) + r'(?![A-Za-z0-9_])')
    new_token = re.compile(re.escape(first[1]) + r'(?![A-Za-z0-9_])')
    uses = [i for i, line in enumerate(lines) if i != second[0] and old_token.search(line)]
    if not uses or any(not second[0] < i < end or new_token.search(lines[i]) for i in uses):
        raise ValueError('Duplicate SSA must be used only inside the same loop')
    changed = lines.copy()
    for i in uses:
        changed[i] = old_token.sub(first[1], changed[i])
    del changed[second[0]]
    restored = changed.copy()
    restored.insert(second[0], lines[second[0]])
    for i in uses:
        restored[i] = new_token.sub(second[1], restored[i])
    if restored != lines:
        raise ValueError('Exact byte inverse failed')
    text = ''.join(changed)
    return text, dict(exact_byte_inverse=True, readonly_q_descriptor=True,
        existing_dominating_value=first[1], removed_duplicate_value=second[1],
        q_descriptor=allocation,
        replaced_uses=len(uses), input_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        explicit_sync_operations_unchanged=True,
        limits='Readonly typed SSA equivalence; actual lowering and device tests remain required')


class StagedKernel:
    def __init__(self, kernel, stages, *, reuse_query=False, out=None, label=None,
                 lowering_checker=None, abi_checker=None, q_source_lines=None):
        self.kernel = kernel
        self.stages = stages
        self.reuse_query = reuse_query
        self.out, self.label = out, label
        self.lowering_checker, self.q_source_lines = lowering_checker, q_source_lines
        self.abi_checker = abi_checker
        self.launch_options = None
        self.proof = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        def call(**kw):
            kw['num_stages'] = self.stages
            if self.reuse_query:
                if self.launch_options is None:
                    normal = self.kernel.warmup(grid=grid, **kw)
                    normal._init_handles()
                    resources(normal,self.out,self.label+'-normal-before-reuse')
                    changed, proof = reuse_invariant_query(normal.asm['ttgir'])
                    path = self.out / (self.label+'-query-reuse.ttgir')
                    path.write_text(changed)
                    replacement = dict(kw, ir_override=str(path))
                    candidate = self.kernel.warmup(grid=grid, **replacement)
                    candidate._init_handles()
                    resources(candidate,self.out,self.label+'-candidate-before-launch')
                    if candidate.hash == normal.hash:
                        raise ValueError('Query reuse did not create a distinct compilation')
                    if self.abi_checker(candidate.src) != self.abi_checker(normal.src):
                        raise ValueError('Query reuse changed ABI/constants/attributes')
                    lowered = self.lowering_checker(candidate.asm['ttgir'], candidate.asm['llir'],
                                                    self.q_source_lines, block_m=8)
                    self.proof = dict(**proof, normal_hash=normal.hash,
                                      candidate_hash=candidate.hash, lowered=lowered)
                    atomic(self.out/(self.label+'-query-reuse-proof.json'),self.proof)
                    if not lowered['mechanism_pass']:
                        raise ValueError('LLVM rematerialized query reads in cyclic blocks')
                    if hashlib.sha256(path.read_bytes()).hexdigest() != proof['output_sha256']:
                        raise ValueError('Override content changed')
                    self.launch_options = {'ir_override':str(path)}
                kw.update(self.launch_options)
            return launch(**kw)
        return call


def resources(kernel, out, label):
    metadata = kernel.metadata
    record = dict(hash=kernel.hash, n_regs=kernel.n_regs, n_spills=kernel.n_spills,
                  metadata={k: getattr(metadata, k, None) for k in
                            ('shared', 'num_warps', 'num_stages', 'use_sme', 'warp_size')}, files={})
    for name, value in kernel.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        data = value.encode() if isinstance(value, str) else value
        path = out / (label + '.' + name)
        path.write_bytes(data)
        record['files'][name] = dict(path=path.name, bytes=len(data),
                                    sha256=hashlib.sha256(data).hexdigest())
    return record


def main(args):
    source = args.source.resolve()
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == args.commit
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', MAIN, '--', 'vllm_fl', 'csrc'], check=True)
    assert hashlib.sha256(Path(JIT).read_bytes()).hexdigest() == JIT_SHA
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 9031)) != 0, 'Existing service; nothing stopped'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True)
    args.out.mkdir(parents=False, exist_ok=False)
    import torch
    from types import SimpleNamespace
    from biv150_benchmark_utils import inputs, load_module, main_call, reduce_call, graph_time
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    assert 'BI-V150' in props.name and props.multi_processor_count == 16
    module = load_module(source / 'vllm_fl/patches/biv150_gqa_decode.py', 'pointer_view_frozen_gqa')
    generated, proof = pipeline_source((source / 'vllm_fl/patches/biv150_gqa_decode.py').read_text(), args.pipeline_stages)
    generated_path = args.out / 'generated-gqa-page-pipeline.py'
    generated_path.write_text(generated)
    atomic(args.out / 'source-transform-proof.json', proof)
    candidate_module = load_module(generated_path, 'gqa_page_pipeline_private')
    q_guard = load_module(source/'vllm_fl/patches/biv150_q_fragment_service.py','gqa_query_lowering_guard')
    q_source_lines = q_guard._q_debug_lines(generated_path)
    result = dict(component_only=True, official_score=False, production_unchanged=True,
                  source_commit=args.commit, production_baseline=MAIN, sdk_jit_sha256=JIT_SHA, source_transform=proof,
                  kernel_source_sha256=hashlib.sha256((source / 'vllm_fl/patches/biv150_gqa_decode.py').read_bytes()).hexdigest(),
                  records=[])
    atomic(args.out / 'result.json', result)
    for batch, length in ((64, 4096), (32, 16384)):
        row = dict(batch=batch, kv=length, passed=False)
        result['records'].append(row)
        try:
            c = inputs(batch, length)
            packed = torch.empty((c['k'].shape[0], 2, 16, 2, 128), device='cuda', dtype=torch.bfloat16)
            k, v = packed.unbind(1)
            k.copy_(c['k']); v.copy_(c['v']); c.update(k=k, v=v)
            table = torch.full((batch, 8192), -1, device='cuda', dtype=torch.int32)
            table[:, :c['table'].shape[1]].copy_(c['table'])
            c['table'] = table[:, :c['table'].shape[1]]
            segments = module._pick_segments(batch, 2, 16)
            for key in ('scratch', 'maxima', 'sums'):
                c[key] = c[key].narrow(2, 0, segments)
            proxy = SimpleNamespace(LOG2E=module.LOG2E,
                _gqa_decode_attention=StagedKernel(candidate_module._gqa_decode_attention,args.pipeline_stages,
                    reuse_query=args.reuse_query,out=args.out,label=f'b{batch}',
                    lowering_checker=q_guard.lowered_proof,abi_checker=q_guard._abi,
                    q_source_lines=q_source_lines))
            def run(selected):
                kernel = main_call(selected, c, segments)
                reduce_call(module, c, segments, 4)
                return kernel
            row.update(segments=segments, packed_shape=list(packed.shape), kv_stride=list(k.stride()),
                       pipeline_stages=args.pipeline_stages, same_allocation_and_strides=True)
            old = run(module); gold = c['out'].clone()
            new = run(proxy); torch.cuda.synchronize()
            row['baseline'] = resources(old, args.out, f'b{batch}-baseline')
            row['candidate'] = resources(new, args.out, f'b{batch}-candidate')
            row['query_reuse_proof'] = proxy._gqa_decode_attention.proof
            if batch == 64:
                assert old.hash == MODEL_4K_HASH, 'Fixture differs from actual current model kernel'
            assert row['baseline']['metadata']['use_sme'] == 14
            assert row['candidate']['metadata']['use_sme'] == row['baseline']['metadata']['use_sme']
            assert row['candidate']['metadata']['num_stages'] == args.pipeline_stages
            row['uniform_exact'] = bool(torch.equal(c['out'], gold))
            row['numeric_checks'] = []
            for ragged in (False, True):
                c['q'].normal_()
                c['lengths'].fill_(length)
                if ragged:
                    for index, n in enumerate((0, 1, 15, 16, 17, 63, 100, length - 1)):
                        c['lengths'][index] = n
                c['out'].fill_(float('nan')); run(module); baseline = c['out'].clone()
                c['out'].fill_(float('nan')); run(proxy); torch.cuda.synchronize()
                assert torch.isfinite(c['out']).all().item()
                for index in (0, 1, batch - 1):
                    n = int(c['lengths'][index].cpu())
                    if n:
                        ids = c['table'][index, :(n + 15)//16].long()
                        kk = c['k'][ids].reshape(-1, 2, 128)[:n].cpu().double()
                        vv = c['v'][ids].reshape(-1, 2, 128)[:n].cpu().double()
                        qq = c['q'][index].cpu().double().reshape(2, 8, 128)
                        pp = torch.einsum('hgd,thd->hgt', qq, kk).mul(128**-.5).softmax(-1)
                        ref = torch.einsum('hgt,thd->hgd', pp, vv).reshape(16, 128)
                    else:
                        ref = torch.zeros((16, 128), dtype=torch.float64)
                    denom = max(float(ref.square().mean().sqrt()), 1e-30)
                    errors = [float((t[index].cpu().double()-ref).square().mean().sqrt())/denom
                              for t in (baseline, c['out'])]
                    row['numeric_checks'].append(dict(ragged=ragged, row=index, length=n,
                        baseline_relative_rmse=errors[0], candidate_relative_rmse=errors[1],
                        passed=errors[1] <= max(.005, 1.2*errors[0])))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(proxy)
                c['q'].normal_(); c['out'].fill_(float('nan')); graph.replay(); torch.cuda.synchronize()
                changed = c['out'].clone(); run(module); torch.cuda.synchronize()
                row['numeric_checks'].append(dict(changed_input_graph=True, ragged=ragged,
                    finite=bool(torch.isfinite(changed).all().item()), exact=bool(torch.equal(changed, c['out'])),
                    passed=bool(torch.isfinite(changed).all().item()) and bool(torch.allclose(changed, c['out'], rtol=.005, atol=1e-5))))
                del graph, changed, baseline
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run(proxy)
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            row['side_stream_finite'] = bool(torch.isfinite(c['out']).all().item())
            c['lengths'].fill_(length)
            row['timings'] = []
            for iteration in range(3):
                choices = [('baseline_ms', module), ('candidate_ms', proxy)]
                if iteration % 2:
                    choices.reverse()
                timing = {name: graph_time(lambda selected=selected: run(selected), repeats=15, unroll=8)[0]
                          for name, selected in choices}
                timing['speedup'] = timing['baseline_ms']/timing['candidate_ms']
                row['timings'].append(timing)
            row.update(median_speedup=statistics.median(x['speedup'] for x in row['timings']),
                       passed=all(x['passed'] for x in row['numeric_checks']) and row['side_stream_finite'])
            del c, proxy, packed, k, v, table, gold
            torch.cuda.empty_cache()
        except BaseException as error:
            row['error'] = repr(error)
            atomic(args.out / 'result.json', result)
            raise
        atomic(args.out / 'result.json', result)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pipeline-stages', type=int, choices=(1,2), default=2)
    parser.add_argument('--reuse-query', action='store_true')
    parser.add_argument('--check-source', action='store_true')
    args=parser.parse_args()
    if args.check_source:
        _,proof=pipeline_source((args.source/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),args.pipeline_stages)
        print(json.dumps(dict(**proof,torch_imported='torch' in sys.modules,gpu_executed=False)))
    else:
        if not args.commit or not args.out: parser.error('--commit and --out required for GPU experiment')
        main(args)
