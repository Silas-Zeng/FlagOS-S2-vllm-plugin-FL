"""Scoped gather at the exact non-spec, common Graph-NONE runner seam.

The local receipt proves bounds from the existing CPU producer, not GPU data.
No stream, H2D, event, request order, or sampled-token buffer is replaced.
"""
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
import sys

ENV = 'VLLM_FL_BIV150_DYNAMIC_HIDDEN_GATHER'
_STATE = '_biv150_dynamic_hidden_gather_runtime'
DIAGNOSTICS = 'CJJ_HIDDEN_GATHER_DIAGNOSTICS'
_KERNEL_SHA = 'c5613dc8645e862fee10be013f2238d436aa5760d3f5501be560c257b8d3ece5'
_SEAL = object()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _validate_sizes(scheduled, total):
    """The pre-cold acceptance gate; no discarded prefix/index construction."""
    _require(type(total) is int and 0 < total <= 2048, 'input rows')
    _require(0 < len(scheduled) <= 64 and all(type(x) is int and x > 0 for x in scheduled), 'positive scheduled rows')
    _require(sum(scheduled) == total, 'scheduled total')


def cpu_indices(scheduled, total, permutation=None):
    """Small integer-only copy of the two existing producer index expressions."""
    _require(type(total) is int and 0 < total <= 2048, 'input rows')
    _require(0 < len(scheduled) <= 64 and all(type(x) is int and x > 0 for x in scheduled), 'positive scheduled rows')
    _require(sum(scheduled) == total, 'scheduled total')
    n = len(scheduled)
    if permutation is None:
        permutation = tuple(range(n))
    _require(len(permutation) == n and all(type(x) is int for x in permutation)
             and sorted(permutation) == list(range(n)), 'request permutation')
    result = [None] * n
    prefix = [0]
    for request in permutation:
        prefix.append(prefix[-1] + scheduled[request])
        result[request] = prefix[-1] - 1
    _require(all(0 <= index < total for index in result), 'index bounds')
    return tuple(prefix), tuple(result)


def _codes(runner):
    names = ('execute_model', '_prepare_inputs', '_get_cumsum_and_arange')
    _require(not any(name in vars(runner) for name in names), 'instance method override')
    return tuple(inspect.unwrap(getattr(type(runner), name)).__code__ for name in names)


def _cold(runner, torch):
    from vllm_fl.patches import biv150_compute_order as order
    from vllm_fl.patches import biv150_cpu_metadata as metadata
    module = sys.modules.get('vllm_fl.worker.model_runner')
    _require(module is not None and type(runner) is module.ModelRunnerFL, 'exact runner class')
    runner_path = Path(module.__file__).resolve()
    _require(hashlib.sha256(runner_path.read_bytes()).hexdigest() == order.EXPECTED_RUNNER, 'runner source identity')
    _require(all(Path(code.co_filename).resolve() == runner_path for code in _codes(runner)), 'runner code origin')
    _require(metadata._scope(runner, torch), 'fixed non-spec text scope')
    try:
        metadata._require_dependencies()
    except RuntimeError as error:
        raise ValueError('fixed CPU producer dependency identity') from error
    _require(sys.modules['vllm.platforms'].current_platform.vendor_name == 'iluvatar', 'platform')
    properties = torch.cuda.get_device_properties(runner.device)
    _require('BI-V150' in properties.name and properties.multi_processor_count == 16, 'device')
    kernel = Path(__file__).with_name('biv150_dynamic_hidden_gather_kernel.py')
    _require(hashlib.sha256(kernel.read_bytes()).hexdigest() == _KERNEL_SHA, 'kernel source identity')
    return dict(codes=_codes(runner), scope=metadata._scope, order=order,
                buffer_type=sys.modules['vllm.v1.utils'].CpuGpuBuffer,
                query_buffer=runner.query_start_loc, event=runner.prepare_inputs_event,
                diagnostics=({'original_query_start_loc_minus_one': 0,
                              'original_compute_order_logits_copy': 0,
                              'compiled': {}, 'overflow': 0}
                             if os.environ.get(DIAGNOSTICS, '0') == '1' else None))


def _description(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.dtype), str(tensor.device), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.data_ptr(), storage._cdata, storage.data_ptr(), storage.nbytes())


def _index_description(tensor, n, device, torch):
    _require(type(tensor) is torch.Tensor and tensor.dtype == torch.int32
             and tensor.device == device and tensor.device.type == 'cuda'
             and tuple(tensor.shape) == (n,) and tuple(tensor.stride()) == (1,), 'int32 CUDA index vector')
    d = _description(tensor)
    _require(type(d[4]) is int and d[4] >= 0 and d[5] == d[7] + d[4] * 4
             and d[4] * 4 + n * 4 <= d[8] and d[5] % 4 == 0, 'index backing span')
    return d


@dataclass
class _Receipt:
    owner: object
    indices: object
    description: tuple
    total: int
    values: tuple
    source: str
    runtime: object
    stream: int
    seal: object
    consumed: bool = False


def prepare(runner, indices, scheduled, total, compute_order, spec_metadata):
    """After original preparation/dispatch, inside the existing input event domain."""
    if os.environ.get(ENV, '1') != '1' or spec_metadata is not None:
        return None
    torch, np = sys.modules.get('torch'), sys.modules.get('numpy')
    if torch is None or np is None:
        return None
    try:
        _require(runner.speculative_config is None and not torch.cuda.is_current_stream_capturing(), 'non-spec prepare')
        n = runner.input_batch.num_reqs
        _require(type(n) is int and type(scheduled) is np.ndarray and scheduled.dtype == np.int32
                 and scheduled.shape == (n,) and scheduled.flags.c_contiguous, 'original scheduled CPU array')
        sizes = tuple(int(x) for x in scheduled)
        # Reject the shape/range before the one-time source/SDK scope audit.
        _validate_sizes(sizes, total)
        state = getattr(runner, _STATE, None)
        if state is None:
            state = _cold(runner, torch)
            setattr(runner, _STATE, state)
        _require(state['codes'] == _codes(runner) and state['scope'](runner, torch)
                 and sys._getframe(1).f_code is state['codes'][0]
                 and runner.query_start_loc is state['query_buffer']
                 and runner.prepare_inputs_event is state['event'], 'same runner/event scope')
        permutation = None
        if compute_order is not None:
            _require(type(compute_order) is state['order'].Plan and compute_order.number == n
                     and compute_order.total == total, 'original compute-order Plan')
            _require(type(compute_order.permutation) is np.ndarray
                     and compute_order.permutation.dtype == np.int64
                     and compute_order.permutation.shape == (n,)
                     and type(compute_order.scheduled) is np.ndarray
                     and compute_order.scheduled.dtype == np.int32
                     and compute_order.scheduled.shape == (n,)
                     and type(compute_order.ends_by_original) is np.ndarray
                     and compute_order.ends_by_original.dtype == np.int32
                     and compute_order.ends_by_original.shape == (n,), 'original integer Plan arrays')
            permutation = tuple(int(x) for x in compute_order.permutation)
        prefix, values = cpu_indices(sizes, total, permutation)
        buffer = runner.query_start_loc
        _require(type(buffer) is state['buffer_type'] and 'copy_to_gpu' not in vars(buffer), 'original prefix buffer')
        _require(type(buffer.np) is np.ndarray and buffer.np.dtype == np.int32
                 and buffer.np.ndim == 1 and buffer.np.flags.c_contiguous
                 and tuple(int(x) for x in buffer.np[:n + 1]) == prefix, 'CPU prefix matches original GPU subtraction source')
        _require(buffer.cpu.device.type == 'cpu' and buffer.cpu.dtype == torch.int32
                 and buffer.np.__array_interface__['data'][0] == buffer.cpu.data_ptr(), 'prefix CPU alias')
        source = 'original_query_start_loc_minus_one'
        if compute_order is not None:
            staged = compute_order.state
            _require(tuple(int(x) for x in compute_order.scheduled) == tuple(sizes[i] for i in permutation)
                     and tuple(int(x) - 1 for x in compute_order.ends_by_original) == values, 'compute-order original-request ends')
            cpu, array = staged['logits_cpu'], staged['logits_np']
            _require(type(array) is np.ndarray and array.dtype == np.int32 and array.ndim == 1
                     and array.flags.c_contiguous and tuple(int(x) for x in array[:n]) == values
                     and cpu.device.type == 'cpu' and cpu.dtype == torch.int32 and cpu.is_pinned()
                     and array.__array_interface__['data'][0] == cpu.data_ptr(), 'original logits H2D staging')
            source = 'original_compute_order_logits_copy'
        description = _index_description(indices, n, runner.device, torch)
        return _Receipt(runner, indices, description, total, values, source, state,
                        torch.cuda.current_stream(runner.device).cuda_stream, _SEAL)
    except (AttributeError, KeyError, TypeError, ValueError):
        # This function only observes metadata; no gather/allocation was issued.
        return None


def _launch(hidden, indices, output, input_rows, selected_rows):
    from .biv150_dynamic_hidden_gather_kernel import gather_hidden_bits
    return gather_hidden_bits[((selected_rows + 7) // 8, 16)](
        hidden, indices, output, input_rows, selected_rows,
        BLOCK_R=8, BLOCK_D=128, num_warps=4, num_stages=1)


def try_gather(runner, hidden, indices, receipt, graph_none, spec_metadata):
    """None means prelaunch unsupported; launch/allocation failures propagate."""
    if receipt is None:
        return None
    if type(receipt) is not _Receipt or receipt.seal is not _SEAL:
        return None
    if receipt.consumed:
        raise RuntimeError('dynamic hidden gather receipt already consumed')
    receipt.consumed = True
    torch = sys.modules.get('torch')
    try:
        _require(os.environ.get(ENV, '1') == '1' and graph_none is True and spec_metadata is None
                 and runner.speculative_config is None and receipt.owner is runner
                 and receipt.indices is indices and torch is not None, 'same non-spec common call')
        state = receipt.runtime
        _require(getattr(runner, _STATE, None) is state and state['codes'] == _codes(runner)
                 and sys._getframe(1).f_code is state['codes'][0]
                 and runner.prepare_inputs_event is state['event'] and state['scope'](runner, torch)
                 and not torch.cuda.is_current_stream_capturing()
                 and torch.cuda.current_stream(runner.device).cuda_stream == receipt.stream, 'same eager stream/lifetime')
        n = len(receipt.values)
        _require(n == runner.input_batch.num_reqs, 'same request count')
        _require(_index_description(indices, n, runner.device, torch) == receipt.description, 'index identity/geometry changed')
        _require(type(hidden) is torch.Tensor and hidden.dtype == torch.bfloat16
                 and hidden.device == runner.device and hidden.device.type == 'cuda'
                 and hidden.ndim == 2 and hidden.shape[1] == 2048
                 and receipt.total <= hidden.shape[0] <= 2048
                 and tuple(hidden.stride()) == (2048, 1), 'contiguous BF16 hidden scope')
        d = _description(hidden)
        _require(type(d[4]) is int and d[4] >= 0 and d[5] == d[7] + d[4] * 2
                 and (d[4] + hidden.shape[0] * 2048) * 2 <= d[8] and d[5] % 2 == 0, 'hidden backing span')
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    # Preserve new-output and current-stream allocation semantics. No try/retry
    # surrounds allocation, JIT or launch; the local receipt is already consumed.
    output = torch.empty((n, 2048), dtype=hidden.dtype, device=hidden.device)
    compiled = _launch(hidden, indices, output, hidden.shape[0], n)
    # Optional host-return observation only; never GPU completion or correctness.
    observed = state.get('diagnostics')
    if observed is not None:
        observed[receipt.source] += 1
        key = id(compiled)
        if key in observed['compiled'] or len(observed['compiled']) < 4:
            observed['compiled'][key] = compiled
        else:
            observed['overflow'] += 1
    return output
