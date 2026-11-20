#!/usr/bin/env python3
"""CPU-only source/bounds/receipt checks; fake tensors never qualify a device."""
import argparse
import ast
from array import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sources(helper):
    order = load('_gather_order_source', ROOT / 'vllm_fl/patches/biv150_compute_order.py')
    runner = (ROOT / 'vllm_fl/worker/model_runner.py').read_bytes()
    runner_digest = sha(runner)
    check('runner matches the compute-order source', runner_digest == order.EXPECTED_RUNNER)
    tree = ast.parse(runner)
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    check('runner prepares and consumes local gather receipt', {
          '_biv150_prepare_hidden_gather', '_biv150_try_hidden_gather'} <= calls)
    kernel = ROOT / 'vllm_fl/patches/biv150_dynamic_hidden_gather_kernel.py'
    kernel_bytes = kernel.read_bytes()
    check('kernel SHA bound in helper', sha(kernel_bytes) == helper._KERNEL_SHA)
    tree = ast.parse(kernel_bytes)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    call = function.decorator_list[0]
    check('runtime row extents not specialized', isinstance(call, ast.Call)
          and ast.literal_eval(call.keywords[0].value) == ['input_rows', 'selected_rows'])
    annotations = {a.arg: ast.unparse(a.annotation) for a in function.args.args if a.annotation}
    check('row extents runtime int32; tiles only constexpr', annotations == {
        'input_rows': 'tl.int32', 'selected_rows': 'tl.int32', 'BLOCK_R': 'tl.constexpr', 'BLOCK_D': 'tl.constexpr'})
    text = kernel_bytes.decode()
    check('no autotune/generated source/clamping', all(x not in text for x in ('autotune', 'exec(', 'minimum(', 'maximum(')))
    check('both pointers use uint16 bits', text.count('tl.pointer_type(tl.uint16)') == 2)
    check('index row/range and column masks explicit', all(x in text for x in (
        'mask=rows < selected_rows', '(index >= 0)', '(index < input_rows)', '(columns[None, :] < 2048)')))
    helper_tree = ast.parse((ROOT / 'vllm_fl/patches/biv150_dynamic_hidden_gather.py').read_bytes())
    launch = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_launch')
    launch_text = ast.unparse(launch)
    check('fixed tile/grid/warps/stages', all(x in launch_text for x in (
        '(selected_rows + 7) // 8', '16', 'BLOCK_R=8', 'BLOCK_D=128', 'num_warps=4', 'num_stages=1')))
    consume = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'try_gather')
    launches = [n for n in consume.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)]
    after_checks = [ast.unparse(n.value.func) for n in launches]
    check('allocation and launch are top-level after fallback checks',
          'torch.empty' in after_checks and '_launch' in after_checks
          and after_checks.index('torch.empty') < after_checks.index('_launch'))
    io = ast.parse((ROOT / 'vllm_fl/dispatch/io_common.py').read_bytes())
    decorator = next(n for n in io.body if isinstance(n, ast.FunctionDef) and n.name == 'managed_inference_mode')
    check('real managed inference wrapper exposes wrapped function', '@functools.wraps(fn)' in ast.unparse(decorator))
    semantics = load('_gather_semantics_cpu', ROOT / 'vllm_fl/compile_semantics.py')
    config = NS(model_config=NS(hf_config=NS(**semantics._GEOMETRY)))
    captures = []
    digest = semantics._digest
    semantics._digest = lambda value: (captures.append(value), digest(value))[1]
    def fingerprint(env):
        return semantics.build_fingerprint(config, 'iluvatar', environ=env, abi={'CPU': True})
    default = fingerprint({})
    on = fingerprint({helper.ENV: '1'})
    off = fingerprint({helper.ENV: '0'})
    check('gather defaults on and off gets a separate cache', default == on and on != off
          and captures[0]['features'][helper.ENV] and not captures[2]['features'][helper.ENV])
    check('both new sources enter fingerprint', all('patches/' + name in captures[0]['source'] for name in (
        'biv150_dynamic_hidden_gather.py', 'biv150_dynamic_hidden_gather_kernel.py')))
    return runner_digest


def bounds_and_bits(helper):
    for n in (1, 2, 7, 8, 9, 31, 41, 42, 43, 59, 60, 61, 62, 63, 64):
        for total in sorted({n, min(2048, n + 1), 1782, 2048}):
            scheduled = (total - n + 1,) + (1,) * (n - 1)
            for permutation in (None, tuple(reversed(range(n)))):
                prefix, values = helper.cpu_indices(scheduled, total, permutation)
                p = tuple(range(n)) if permutation is None else permutation
                expected = [sum(scheduled[j] for j in p[:i + 1]) - 1 for i in range(n)]
                check(f'CPU producer n{n}/total{total}/permuted{permutation is not None}',
                      prefix[-1] == total and all(values[p[i]] == expected[i] for i in range(n)))
    for scheduled, total, permutation in (((), 0, None), ((0,), 1, None), ((-1, 2), 1, None),
            ((1,), 2, None), ((2049,), 2049, None), ((1,) * 65, 65, None),
            ((1.0,), 1, None), ((True,), 1, None), ((1,), True, None),
            ((1, 1), 2, (0, 0)), ((1, 1), 2, (0, 2)), ((1, 1), 2, (0.0, 1))):
        try:
            helper.cpu_indices(scheduled, total, permutation)
        except ValueError:
            CHECKS.append('reject integer producer boundary ' + str(len(CHECKS)))
        else:
            raise AssertionError('invalid CPU producer accepted')
    # Exhaust every BF16 bit pattern, including NaN payloads, in input rows.
    hidden = array('H', range(65536)) * 3
    for n in (1, 7, 8, 9, 31, 41, 42, 43, 59, 60, 61, 62, 63, 64):
        indices = [(n - i * 3) % 96 for i in range(n)]  # shuffled and repeated valid indices
        guard = 0x3A5C
        output = array('H', [guard]) * (n * 2048 + 2)
        writes = 0
        for tile in range((n + 7) // 8):
            for local in range(8):
                row = tile * 8 + local
                if row >= n:
                    continue
                for col_tile in range(16):
                    start = col_tile * 128
                    src = indices[row] * 2048 + start
                    dst = 1 + row * 2048 + start
                    output[dst:dst + 128] = hidden[src:src + 128]
                    writes += 128
        expected = array('H')
        for index in indices:
            expected.extend(hidden[index * 2048:(index + 1) * 2048])
        check('CPU tile bit mapping and complete canaries n' + str(n),
              output[0] == output[-1] == guard and output[1:-1] == expected and writes == n * 2048)


def fake_receipts(helper):
    import numpy as np
    order = load('_gather_order_cpu', ROOT / 'vllm_fl/patches/biv150_compute_order.py')
    class Device:
        type = 'cuda'
        def __str__(self): return 'cuda:0'
    device = Device()
    class Tensor:
        def __init__(self, shape, dtype, dev=device, pointer=0x100000, offset=0, stride=None, nbytes=None):
            self.shape, self.dtype, self.device, self.offset = tuple(shape), dtype, dev, offset
            self.steps = tuple(stride or ((2048, 1) if len(shape) == 2 else (1,)))
            size = 4 if dtype == 'int32' else 2
            self.pointer = pointer
            self.storage = NS(_cdata=pointer, data_ptr=lambda: pointer - offset * size,
                nbytes=lambda: nbytes if nbytes is not None else (offset + __import__('math').prod(shape)) * size)
        @property
        def ndim(self): return len(self.shape)
        def stride(self): return self.steps
        def storage_offset(self): return self.offset
        def data_ptr(self): return self.pointer
        def untyped_storage(self): return self.storage
        def is_pinned(self): return True
        def cpu(self): raise AssertionError('GPU readback forbidden')
        def item(self): raise AssertionError('GPU readback forbidden')
        def tolist(self): raise AssertionError('GPU readback forbidden')
    class Buffer: pass
    class Runner:
        def _prepare_inputs(self): pass
        def _get_cumsum_and_arange(self): pass
        def execute_model(self, before=None, after=None, **options):
            if before: before(self)
            self.receipt = helper.prepare(self, self.indices, self.scheduled, self.total, self.plan,
                                          options.get('spec'))
            if after: after(self)
            output = helper.try_gather(self, self.hidden, self.indices, self.receipt,
                                       options.get('graph_none', True), options.get('spec'))
            return output
    allocations, launches = [], []
    cuda = NS(capture=False, stream=17)
    cuda.is_current_stream_capturing = lambda: cuda.capture
    cuda.current_stream = lambda dev: NS(cuda_stream=cuda.stream)
    cuda.synchronize = lambda: (_ for _ in ()).throw(AssertionError('no synchronize'))
    def empty(shape, dtype, device):
        result = Tensor(shape, dtype, device, pointer=0x900000 + len(allocations) * 0x100000)
        allocations.append(result)
        return result
    fake_torch = NS(Tensor=Tensor, int32='int32', bfloat16='bf16', cuda=cuda, empty=empty)
    def setup(permuted=False):
        cuda.capture, cuda.stream = False, 17
        runner = Runner()
        runner.speculative_config = None
        runner.input_batch = NS(num_reqs=3)
        runner.device, runner.prepare_inputs_event = device, object()
        runner.scheduled, runner.total = np.array([5, 1, 2], dtype=np.int32), 8
        permutation = np.array([1, 2, 0], dtype=np.int64) if permuted else np.arange(3, dtype=np.int64)
        prefix, values = helper.cpu_indices((5, 1, 2), 8, tuple(int(x) for x in permutation))
        buffer = Buffer()
        buffer.np = np.array(prefix, dtype=np.int32)
        buffer.cpu = Tensor((4,), 'int32', NS(type='cpu'), pointer=buffer.np.__array_interface__['data'][0])
        runner.query_start_loc = buffer
        runner.indices, runner.hidden = Tensor((3,), 'int32'), Tensor((8, 2048), 'bf16', pointer=0x200000)
        runner.plan = None
        if permuted:
            values_array = np.array(values, dtype=np.int32)
            state = {'logits_np': values_array, 'logits_cpu': Tensor((3,), 'int32', NS(type='cpu'),
                     pointer=values_array.__array_interface__['data'][0])}
            runner.plan = order.Plan(state, permutation, np.argsort(permutation), runner.scheduled[permutation],
                                     values_array + 1, 8, 0, 0)
        setattr(runner, helper._STATE, dict(codes=helper._codes(runner), scope=lambda r, t: True,
                order=order, buffer_type=Buffer, query_buffer=buffer, event=runner.prepare_inputs_event))
        return runner
    old_torch, old_launch, old_cold = sys.modules.get('torch'), helper._launch, helper._cold
    old_env = os.environ.get(helper.ENV)
    try:
        sys.modules['torch'] = fake_torch
        helper._cold = lambda *a: (_ for _ in ()).throw(AssertionError('fake tests cannot qualify cold SDK'))
        helper._launch = lambda *args: launches.append(args)
        os.environ[helper.ENV] = '0'
        check('explicit OFF does not touch tensor metadata', helper.prepare(object(), object(), object(), 8, None, None) is None)
        os.environ[helper.ENV] = '1'
        for permuted in (False, True):
            runner = setup(permuted)
            result = runner.execute_model()
            check('same original source accepts ' + str(permuted), result is not None and result is not runner.hidden
                  and result.shape == (3, 2048) and runner.receipt.consumed and launches[-1][-2:] == (8, 3))
            first = result
            runner.indices = Tensor((3,), 'int32', pointer=0x500000)
            result = runner.execute_model()
            check('next call revalidates new index storage and output ' + str(permuted),
                  result is not first and runner.receipt.indices is runner.indices)
        cases = {
            'dtype': lambda r: setattr(r.indices, 'dtype', 'bf16'),
            'pointer': lambda r: setattr(r.indices, 'pointer', r.indices.pointer + 4),
            'stride': lambda r: setattr(r.indices, 'steps', (2,)),
            'offset': lambda r: setattr(r.indices, 'offset', 1),
            'backing': lambda r: setattr(r.indices.storage, '_cdata', 123),
            'replaced indices': lambda r: setattr(r, 'indices', Tensor((3,), 'int32')),
            'stream': lambda r: setattr(cuda, 'stream', 18),
            'capture': lambda r: setattr(cuda, 'capture', True),
            'event': lambda r: setattr(r, 'prepare_inputs_event', object()),
            'method override': lambda r: setattr(r, '_prepare_inputs', lambda: None),
            'request count': lambda r: setattr(r.input_batch, 'num_reqs', 2),
            'hidden rows': lambda r: setattr(r.hidden, 'shape', (7, 2048)),
            'hidden width': lambda r: setattr(r.hidden, 'shape', (8, 1024)),
            'hidden stride': lambda r: setattr(r.hidden, 'steps', (2049, 1)),
            'hidden dtype': lambda r: setattr(r.hidden, 'dtype', 'fp16'),
            'hidden span': lambda r: setattr(r.hidden.storage, 'nbytes', lambda: 2),
            'disabled': lambda r: os.environ.__setitem__(helper.ENV, '0'),
        }
        for name, mutate in cases.items():
            runner = setup()
            os.environ[helper.ENV] = '1'
            before = len(allocations), len(launches)
            check('prelaunch rejects changed ' + name, runner.execute_model(after=mutate) is None
                  and before == (len(allocations), len(launches)))
        os.environ[helper.ENV] = '1'
        for name, mutate in {
            'prefix': lambda r: r.query_start_loc.np.__setitem__(1, 4),
            'zero scheduled': lambda r: r.scheduled.__setitem__(0, 0),
            'prefix alias': lambda r: setattr(r.query_start_loc.cpu, 'pointer', 7),
            'float permutation': lambda r: setattr(r.plan, 'permutation', r.plan.permutation.astype(np.float64)),
            'plan ends': lambda r: r.plan.ends_by_original.__setitem__(0, 0),
            'logits staging': lambda r: r.plan.state['logits_np'].__setitem__(0, -1),
        }.items():
            runner = setup(True)
            before = len(allocations), len(launches)
            check('producer rejects ' + name, runner.execute_model(before=mutate) is None
                  and before == (len(allocations), len(launches)))
        runner = setup()
        check('external caller cannot certify arbitrary indices', helper.prepare(
            runner, runner.indices, runner.scheduled, runner.total, None, None) is None)
        check('Graph NONE required before allocation', setup().execute_model(graph_none=False) is None)
        check('spec path untouched', setup().execute_model(spec=object()) is None)
        runner = setup()
        delattr(runner, helper._STATE)
        helper._cold = lambda *a: (_ for _ in ()).throw(ValueError('unknown producer source'))
        check('unknown source returns before allocation', runner.execute_model() is None)
        runner = setup()
        fake_torch.empty = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('sentinel allocation failure'))
        before = len(launches)
        try:
            runner.execute_model()
        except RuntimeError as exc:
            check('allocation error propagates without fallback/retry', str(exc) == 'sentinel allocation failure'
                  and runner.receipt.consumed and len(launches) == before)
        else:
            raise AssertionError('allocation error retried')
        fake_torch.empty = empty
        runner = setup()
        helper._launch = lambda *args: (_ for _ in ()).throw(RuntimeError('sentinel launch failure'))
        try:
            runner.execute_model()
        except RuntimeError as exc:
            check('launch error propagates with consumed receipt', str(exc) == 'sentinel launch failure' and runner.receipt.consumed)
        else:
            raise AssertionError('launch error retried')
        try:
            helper.try_gather(runner, runner.hidden, runner.indices, runner.receipt, True, None)
        except RuntimeError as exc:
            check('consumed receipt cannot retry', 'already consumed' in str(exc))
        else:
            raise AssertionError('receipt reused')
    finally:
        if old_torch is None: sys.modules.pop('torch', None)
        else: sys.modules['torch'] = old_torch
        helper._launch, helper._cold = old_launch, old_cold
        if old_env is None: os.environ.pop(helper.ENV, None)
        else: os.environ[helper.ENV] = old_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', action='store_true', help='NumPy plus fake Tensor metadata; never Torch')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if 'torch' in sys.modules or 'triton' in sys.modules:
        raise RuntimeError('run in a fresh CPU-only interpreter')
    helper = load('_gather_cpu_helper', ROOT / 'vllm_fl/patches/biv150_dynamic_hidden_gather.py')
    runner_digest = sources(helper)
    bounds_and_bits(helper)
    if args.arrays: fake_receipts(helper)
    check('Torch/Triton never imported', 'torch' not in sys.modules and 'triton' not in sys.modules)
    report = dict(completed=True, errors=[], CPU_only=True, GPU_executed=False,
                  device_numeric_verified=False, service_verified=False, performance_measured=False,
                  runner_sha256=runner_digest,
                  checks=CHECKS, count=len(CHECKS), arrays=args.arrays,
                  script_sha256=sha(Path(__file__).read_bytes()),
                  helper_sha256=sha(Path(helper.__file__).read_bytes()), kernel_sha256=helper._KERNEL_SHA)
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.out: args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
