"""固定main3283672b加PR15、PR21的局部计算序；持久请求/采样序不动，未知描述在首次写前回原路。"""

# ruff: noqa: E501

import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path

EXPECTED_RUNNER = "fa6f2387d258804021e53130ef8ae362ea91a9a23d9b1dcaf1c3d37cb4e0ee9f"
_STATE = "_biv150_compute_order_buffers"
SOURCES = {
    "vllm.v1.cudagraph_dispatcher": "51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63",
    "vllm.v1.worker.block_table": "b3a6649b968fba0b48c333d7ac96e65cc713a86133cc7eef0a8399f5339ec600",
    "vllm.v1.worker.gpu_input_batch": "de1ea4fc9d43fc6079dd375e407c2b6d93f5e1a11f49732d13654c41eaf30a65",
    "vllm.v1.utils": "4e4cda209e10d4477d251003779dec26ea5ef47541c3310995c007c7ee5fdcdb",
    "vllm.v1.kv_cache_interface": "7e3c220998267190a622e216b2b125306807cf8be3ec1e9a8e84fe6e7a293003",
    "vllm.v1.attention.backends.triton_attn": "1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605",
    "vllm_fl.patches.biv150_cpu_metadata": "0e96152454a9197f65a34c2fe8805bd5073e1fa473070d59d4ec4f20a6d8846f",
    "vllm_fl.patches.biv150_mixed_decode": "a129361f1ef1babe1ce1e6238f6b1a725050612c235201064ddeba676b420c42",
    "vllm_fl.patches.biv150_gqa_decode": "1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a",
    "vllm_fl.patches.biv150_prefill_normalized": "24a421fa25e0ce7e21b54deb760a5bc893d3899a86ced4849bd3f3bfab50ed42",
    "vllm_fl.patches.biv150_q_fragment_service": "ea361e6b604fbd3c7e3a6b95776892a7bc9d22e9988927db3f7876412c6e310f",
    "vllm.v1.attention.ops.triton_unified_attention": "54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f",
    "vllm.v1.attention.ops.triton_attention_helpers": "59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1",
}


def _function_cells(function, module, qualname):
    if (
        not isinstance(function, types.FunctionType)
        or function.__globals__ is not vars(module)
        or function.__code__.co_qualname != qualname
        or Path(function.__code__.co_filename).resolve()
        != Path(module.__file__).resolve()
    ):
        raise ValueError("非固定main函数绑定：" + qualname)
    return dict(
        zip(
            function.__code__.co_freevars,
            (cell.cell_contents for cell in (function.__closure__ or ())),
        )
    )


def _main_stack(torch):
    attention = sys.modules["vllm.v1.attention.ops.triton_unified_attention"]
    backend = sys.modules["vllm.v1.attention.backends.triton_attn"]
    normalized = sys.modules["vllm_fl.patches.biv150_prefill_normalized"]
    gqa = sys.modules["vllm_fl.patches.biv150_gqa_decode"]
    mixed = sys.modules["vllm_fl.patches.biv150_mixed_decode"]
    top = attention.unified_attention
    if (
        backend.unified_attention is not top
        or getattr(attention, normalized._INSTALLED, False) is not True
        or getattr(attention, gqa._INSTALLED_ATTR, False) is not True
    ):
        raise ValueError("main normalized/GQA安装或双统一入口不符")
    cells = _function_cells(top, normalized, "install.<locals>.wrapper")
    if set(cells) != {"normalized", "original", "signature"}:
        raise ValueError("normalization闭包不符")
    previous = cells["original"]
    if getattr(top, "__wrapped__", None) is not previous:
        raise ValueError("normalization原路不是本GQA")
    old = _function_cells(previous, gqa, "_make_wrapper.<locals>.patched")
    if (
        set(old) != {"original"}
        or getattr(previous, "__wrapped__", None) is not old["original"]
    ):
        raise ValueError("GQA原路存在未知包装")
    generic = old["original"]
    if _function_cells(generic, attention, "unified_attention"):
        raise ValueError("generic存在未知闭包")
    clone = cells["normalized"]
    if (
        not isinstance(clone, types.FunctionType)
        or clone.__code__ is not generic.__code__
        or getattr(clone, "__wrapped__", None) is not generic
        or clone.__globals__ is generic.__globals__
        or clone.__defaults__ != generic.__defaults__
        or clone.__kwdefaults__ != generic.__kwdefaults__
        or clone.__closure__ is not generic.__closure__
        or clone.__globals__.get("reduce_segments") is not attention.reduce_segments
        or not isinstance(cells["signature"], inspect.Signature)
        or cells["signature"] != inspect.signature(previous)
    ):
        raise ValueError("normalized统一函数不是原generic的独立global克隆")
    source = inspect.getsource(attention.kernel_unified_attention.fn)
    old_mass = "L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)"
    old_name = "def kernel_unified_attention("
    if source.count(old_mass) != 1 or source.count(old_name) != 1:
        raise ValueError("pinned原kernel源码形态不符")
    expected = source.replace(old_mass, "L = tl.zeros([BLOCK_M], dtype=tl.float32)")
    expected = expected.replace(old_name, "def kernel_biv150_prefill_normalized(", 1)
    kernel = clone.__globals__.get("kernel_unified_attention")
    fn = getattr(kernel, "fn", None)
    digest = hashlib.sha256(expected.encode()).hexdigest()
    if (
        kernel is attention.kernel_unified_attention
        or not isinstance(fn, types.FunctionType)
        or fn.__code__.co_filename != "<biv150-prefill-normalized-" + digest + ".py>"
        or fn.__code__.co_name != "kernel_biv150_prefill_normalized"
        or inspect.getsource(fn) != expected
    ):
        raise ValueError("normalized kernel不是本次已核L=0克隆")
    for function, name, original_name, qualname in (
        (
            backend.TritonAttentionMetadataBuilder.build,
            "build",
            "original_build",
            "TritonAttentionMetadataBuilder.build",
        ),
        (
            backend.TritonAttentionImpl.forward,
            "forward",
            "original_forward",
            "TritonAttentionImpl.forward",
        ),
    ):
        captured = _function_cells(function, mixed, "_wrap_backend.<locals>." + name)
        original = captured.get(original_name)
        if (
            set(captured) != {"backend", original_name, "torch"}
            or captured["backend"] is not backend
            or captured["torch"] is not torch
            or getattr(function, mixed._MARKER, False) is not True
            or getattr(function, "__wrapped__", None) is not original
            or _function_cells(original, backend, qualname)
        ):
            raise ValueError("C3原build/forward闭包或动态backend不符")


def _cold(runner, torch):
    for name, pin in {**SOURCES, type(runner).__module__: EXPECTED_RUNNER}.items():
        module = sys.modules.get(name)
        if (
            pin is None
            or module is None
            or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != pin
        ):
            raise ValueError("未自然加载的固定compute-order源：" + name)
    _main_stack(torch)
    md = sys.modules["vllm_fl.patches.biv150_cpu_metadata"]
    backend = sys.modules["vllm.v1.attention.backends.triton_attn"]
    dispatcher_module = sys.modules["vllm.v1.cudagraph_dispatcher"]
    mode = backend.CUDAGraphMode
    dispatcher = runner.cudagraph_dispatcher
    config = runner.vllm_config
    platform = sys.modules["vllm.platforms"].current_platform
    spec_module = sys.modules["vllm.v1.kv_cache_interface"]
    spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    if (
        not md._scope(runner, torch)
        or config.model_config.is_encoder_decoder
        or platform.vendor_name != "iluvatar"
        or torch.cuda.current_device() != 0
        or type(spec) is not spec_module.FullAttentionSpec
        or (spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype)
        != (16, 2, 128, torch.bfloat16)
        or len(runner.kv_cache_config.kv_cache_groups) != 1
        or config.kv_transfer_config is not None
        or config.ec_transfer_config is not None
        or runner.num_prompt_logprobs
        or runner.routed_experts_initialized
        or runner.shared_kv_cache_layers
        or runner.cross_layers_kv_cache is not None
        or runner._kernel_block_sizes != [16]
        or config.compilation_config.cudagraph_mode != mode.FULL_DECODE_ONLY
        or type(dispatcher) is not dispatcher_module.CudagraphDispatcher
        or "dispatch" in dispatcher.__dict__
        or not dispatcher.keys_initialized
        or dispatcher.cudagraph_mode != mode.FULL_DECODE_ONLY
        or dispatcher.uniform_decode_query_len != 1
        or runner.uniform_decode_query_len != 1
        or len(runner.attn_groups) != 1
        or len(runner.attn_groups[0]) != 1
        or runner.attn_groups[0][0].backend is not backend.TritonAttentionBackend
        or backend.envs.VLLM_BATCH_INVARIANT
    ):
        raise ValueError("非固定text/async/单组FULL_DECODE_ONLY scope")
    properties = torch.cuda.get_device_properties(0)
    if "BI-V150" not in properties.name or properties.multi_processor_count != 16:
        raise ValueError("仅已测BI-V150 16SM，不继承C500行为")
    full = dispatcher.cudagraph_keys[mode.FULL]
    piecewise = dispatcher.cudagraph_keys[mode.PIECEWISE]
    if (
        piecewise
        or not full
        or any(key.uniform is not True or key.has_lora for key in full)
    ):
        raise ValueError("混合批不能在写前证明NONE且不padding")
    group = runner.attn_groups[0][0]
    builder = group.get_metadata_builder()
    table = runner.input_batch.block_table[0]
    if (
        type(builder) is not backend.TritonAttentionMetadataBuilder
        or len(runner.input_batch.block_table.block_tables) != 1
        or table.block_size != 16
        or table.blocks_per_kv_block != 1
        or table.use_hybrid_blocks
        or table.dcp_world_size != 1
        or table.pcp_world_size != 1
        or not table.block_table.cpu.is_contiguous()
        or not table.block_table.gpu.is_contiguous()
        or table.block_table.cpu.dtype != torch.int32
        or table.block_table.gpu.dtype != torch.int32
        or table.block_table.gpu.device != runner.device
        or table.block_table.cpu.shape != table.block_table.gpu.shape
        or table.block_table.cpu.shape[1] != 8192
    ):
        raise ValueError("不是原完整stride单组页表")

    # 只增加host staging；原GPU页表/positions/seq地址和原event不替换。
    def pinned(shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    state = dict(
        table=table,
        table_buffer=table.block_table,
        dispatcher=dispatcher,
        mode=mode,
        full=full,
        full_count=len(full),
        piecewise=piecewise,
        threshold=builder.seq_threshold_3D,
        positions_gpu=runner.positions,
        seq_gpu=runner.seq_lens,
        event=runner.prepare_inputs_event,
        table_cpu=pinned(tuple(table.block_table.cpu.shape), torch.int32),
        positions_cpu=pinned((runner.max_num_tokens,), torch.int64),
        seq_cpu=pinned(tuple(runner.seq_lens.shape), torch.int32),
        computed_cpu=pinned(tuple(runner.seq_lens.shape), torch.int32),
        prompt_cpu=pinned(tuple(runner.seq_lens.shape), torch.int32),
        logits_cpu=pinned(tuple(runner.seq_lens.shape), torch.int32),
        selected=0,
    )
    for key in ("table", "positions", "seq", "computed", "prompt", "logits"):
        state[key + "_np"] = state[key + "_cpu"].numpy()
    return state


class Plan:
    """只由本次execute局部变量持有；所有写发生在原prepare_inputs_event域内。"""

    def __init__(
        self, state, permutation, inverse, scheduled, ends, total, prefix, outside
    ):
        self.state, self.permutation, self.inverse = state, permutation, inverse
        self.scheduled, self.ends_by_original = scheduled, ends
        self.number, self.total = len(permutation), total
        self.prefix, self.outside = prefix, outside
        self.dall = prefix + outside

    def copy_block_table(self, runner):
        s, n = self.state, self.number
        # P由全行互斥mask构建且范围已验，clip不会实际改变索引，避免raise模式额外缓冲。
        import numpy as np

        np.take(
            s["table_buffer"].np,
            self.permutation,
            axis=0,
            out=s["table_np"][:n],
            mode="clip",
        )
        s["table_buffer"].gpu[:n].copy_(s["table_cpu"][:n], non_blocking=True)

    def copy_metadata(self, runner, positions, req_indices):
        s, n, total = self.state, self.number, self.total
        s["positions_np"][:total] = positions
        runner.req_indices.np[:total] = req_indices
        runner.num_scheduled_tokens.np[:n] = self.scheduled
        runner.positions[:total].copy_(s["positions_cpu"][:total], non_blocking=True)
        runner.seq_lens.copy_(s["seq_cpu"], non_blocking=True)

    def logits_indices(self, runner):
        # 替代原query_start_loc[1:]-1 enqueue；原hidden[N] gather和sampler不变。
        return self.state["logits_cpu"][: self.number].to(
            runner.device, non_blocking=True
        )

    def cpu_metadata(self, number):
        if number != self.number:
            raise RuntimeError("写后出现未知padding，禁止原路重试")
        s = self.state
        return (
            s["computed_cpu"][:number],
            s["prompt_cpu"][:number],
            s["seq_cpu"][:number],
        )

    def check_dispatch(self, mode, descriptor, should_ubatch, tokens_across_dp):
        if (
            mode != self.state["mode"].NONE
            or descriptor.num_tokens != self.total
            or descriptor.num_reqs not in (None, self.number)
            or should_ubatch
            or tokens_across_dp is not None
        ):
            raise RuntimeError("compute-order写后dispatcher不符；不回退或重复提交")

    def forward_returned(self):
        try:
            s = self.state
            s["selected"] += 1
            if s["selected"] == 1:
                print(
                    "BIV150_COMPUTE_ORDER_FIRST_SELECTED "
                    + json.dumps(
                        dict(
                            pid=os.getpid(),
                            N=self.number,
                            M=self.total,
                            Dprefix=self.prefix,
                            Dall=self.dall,
                            outside=self.outside,
                            sampler_rows=self.number,
                            original_model_forward_returned=True,
                            runtime_mode="NONE",
                            inverse_logits_order="original_request_rows",
                            token_bitwise_equivalence_claimed=False,
                        )
                    ),
                    flush=True,
                )
        except Exception:
            pass  # 观察错误不能把已成功的forward改成失败。


def make_plan(runner, output, scheduled, max_query_len):
    if max_query_len <= 1:
        return None
    platforms = sys.modules.get("vllm.platforms")
    if (
        getattr(getattr(platforms, "current_platform", None), "vendor_name", None)
        != "iluvatar"
    ):
        return None
    import numpy as np
    import torch

    try:
        n, total = runner.input_batch.num_reqs, output.total_num_scheduled_tokens
        if (
            type(scheduled) is not np.ndarray
            or scheduled.dtype != np.int32
            or scheduled.shape != (n,)
            or not scheduled.flags.c_contiguous
            or n < 3
            or type(total) is not int
            or not 0 < total <= runner.max_num_tokens
            or n > 64
            or scheduled[0] != 1
            or scheduled.min() <= 0
            or int(scheduled.sum()) != total
            or output.scheduled_spec_decode_tokens
            or output.scheduled_encoder_inputs
            or output.scheduled_cached_reqs.resumed_req_ids
            or output.__dict__.get("_biv150_kv_protocol_v1_resume") is not None
            or output.has_structured_output_requests
            or output.pending_structured_output_tokens
            or runner.num_prompt_logprobs
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        mask = scheduled == 1
        prefix, dall = int(np.flatnonzero(~mask)[0]), int(mask.sum())
        if not 0 < prefix < dall < n:
            return None
        state = getattr(runner, _STATE, None)
        if state is False:
            return None
        if state is None:
            state = _cold(runner, torch)
            setattr(runner, _STATE, state)
        mode, dispatcher = state["mode"], runner.cudagraph_dispatcher
        if (
            dall > state["threshold"]
            or dispatcher is not state["dispatcher"]
            or n > state["seq_cpu"].shape[0]
            or n > state["table_buffer"].cpu.shape[0]
            or dispatcher.cudagraph_mode != mode.FULL_DECODE_ONLY
            or dispatcher.cudagraph_keys[mode.FULL] is not state["full"]
            or len(state["full"]) != state["full_count"]
            or dispatcher.cudagraph_keys[mode.PIECEWISE] is not state["piecewise"]
            or state["piecewise"]
            or runner.input_batch.block_table[0] is not state["table"]
            or state["table"].block_table is not state["table_buffer"]
            or runner.positions is not state["positions_gpu"]
            or runner.seq_lens is not state["seq_gpu"]
            or runner.prepare_inputs_event is not state["event"]
        ):
            return None
        permutation = np.concatenate(
            (np.flatnonzero(mask), np.flatnonzero(~mask))
        ).astype(np.int64, copy=False)
        inverse = np.empty(n, dtype=np.int64)
        inverse[permutation] = np.arange(n, dtype=np.int64)
        compute_q = scheduled[permutation].copy()
        ends = np.cumsum(compute_q, dtype=np.int32)[inverse]
        # 私有host staging写仍在原event保护域；输入批和optimistic/discard均不原地P。
        np.take(
            runner.input_batch.num_computed_tokens_cpu,
            permutation,
            out=state["computed_np"][:n],
            mode="clip",
        )
        np.take(
            runner.input_batch.num_prompt_tokens_cpu_tensor.numpy(),
            permutation,
            out=state["prompt_np"][:n],
            mode="clip",
        )
        state["seq_np"].fill(0)
        np.add(state["computed_np"][:n], compute_q, out=state["seq_np"][:n])
        state["logits_np"][:n] = ends - 1
        return Plan(
            state, permutation, inverse, compute_q, ends, total, prefix, dall - prefix
        )
    except Exception as error:
        # 尚未提交或改原buffer；未知scope回原。提交方法不在这个except中。
        if getattr(runner, _STATE, None) is None:
            setattr(runner, _STATE, False)
            try:
                print(
                    "BIV150_COMPUTE_ORDER_SCOPE_REJECTED "
                    + json.dumps(
                        dict(
                            pid=os.getpid(),
                            error_type=type(error).__name__,
                            reason=str(error)[:160],
                        )
                    ),
                    flush=True,
                )
            except Exception:
                pass
        return None
