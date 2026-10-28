"""仅BI-V150 MiniCPM真实R32/R64 head的原舍入/温度epilogue；显式调用票据。"""
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
import weakref

_FLAG = 'VLLM_FL_BIV150_HEAD_TEMPERATURE'
_RAGGED_FLAG = 'VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED'
_STATE = '_biv150_head_temperature_runtime'
# 仅两项实际完整head+Sampler组件已过的配置；无新形状泛化。
_CONFIGS = {
    32: dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=1, grid=(2040, 1)),
    64: dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=32, num_warps=16, num_stages=2, grid=(510, 1)),
}
# Only explicit experimental rows; keep the qualified 32/64 mappings intact.
_RAGGED_BASE_ROWS = {31: 32, 41: 64, 42: 64, 43: 64}


def _config_for_rows(rows):
    config = _CONFIGS.get(rows)
    if config is not None:
        return config
    if type(rows) is int and os.environ.get(_RAGGED_FLAG, '0') == '1':
        base_rows = _RAGGED_BASE_ROWS.get(rows)
        if base_rows is not None:
            return _CONFIGS[base_rows]
    return None


_PINS = {
    'vllm.model_executor.models.llama': '1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384',
    'vllm.model_executor.layers.vocab_parallel_embedding': '90bc72a4758c44fd631dda30a5e8695369bf6e3fe5e1d9b7edc80356abfcd110',
    'vllm.model_executor.layers.logits_processor': 'd6dfec7287020d587f4a4d475ce113c383b2eb55afe5ae9b0e037df2e2d13310',
    'vllm.model_executor.layers.utils': '061adffc6af55ef3b001557c0fad738e474c5e4134fa3dc35cb967bdbc90a121',
    'vllm.v1.sample.sampler': '1a2f18ddc91c89f3744ab5a93624be056bf20b7a8b2e6e13d0236182718a2253',
    'vllm.v1.sample.logits_processor.builtin': 'd8967b2358debb3ba34e0c5fc49b43c3a880e3d408d83f517575485701ad5cc9',
}


@dataclass(frozen=True)
class HeadTemperatureState:
    """随原ExecuteModelState一次生产/消费；没有全局pending或隐式张量标记。"""
    sampling_metadata: object
    temperature: object
    request_ids: tuple
    hidden: object
    runtime: object


class _AlreadyScaledSampler:
    """仅委托原forward/sample代码；原统计、softmax与噪声对象不复制。"""
    def __init__(self, sampler):
        self._sampler = weakref.ref(sampler)

    def __getattr__(self, name):
        return getattr(self._sampler(), name)

    def __call__(self, *args, **kwargs):
        original = self._sampler()
        return type(original).forward(self, *args, **kwargs)

    def sample(self, *args, **kwargs):
        original = self._sampler()
        return type(original).sample(self, *args, **kwargs)

    @staticmethod
    def apply_temperature(logits, temperature, all_random):
        # 只由显式HeadTemperatureState且资格仍有效的入口调用。
        return logits


def _non_argmax_noops(processors):
    from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor, LogitBiasLogitsProcessor
    for processor in processors:
        if type(processor) is MinTokensLogitsProcessor and not processor.min_toks:
            continue
        if type(processor) is LogitBiasLogitsProcessor and not processor.biases:
            continue
        return False
    return True


def _metadata_supported(runner, scheduler_output):
    """只读当前CPU资格/张量元信息，不做GPU标量回读或改变采样参数。"""
    sm = runner.input_batch.sampling_metadata
    if (getattr(scheduler_output, 'has_structured_output_requests', None) is not False
            or runner.num_prompt_logprobs or runner.speculative_config is not None
            or runner.broadcast_pp_output or runner.use_aux_hidden_state_outputs
            or runner.is_pooling_model or runner.lora_config is not None
            or getattr(runner, 'routed_experts_initialized', False)
            or (runner.input_batch.num_reqs not in _CONFIGS
                and _config_for_rows(runner.input_batch.num_reqs) is None)
            or not sm.all_random or sm.all_greedy or not sm.no_penalties
            or sm.max_num_logprobs is not None or sm.logprob_token_ids
            or sm.allowed_token_ids_mask is not None or sm.bad_words_token_ids
            or any(sm.spec_token_ids or ())
            or sm.top_k is not None or sm.top_p is None
            or not _non_argmax_noops(sm.logitsprocs.non_argmax_invariant)):
        return False
    holder = sm.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return False
    for req_id in runner.input_batch.req_ids:
        params = runner.requests[req_id].sampling_params
        if not hasattr(params, 'structured_outputs') or params.structured_outputs is not None:
            return False
    return True


class _Runtime:
    def __init__(self, runner):
        import importlib
        import torch
        from vllm import envs
        from vllm.platforms import current_platform
        from vllm.model_executor.layers import utils
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.models.llama import LlamaForCausalLM
        from vllm.v1.sample.sampler import Sampler
        from vllm_fl.compile_semantics import native_model_matches
        if current_platform.vendor_name != 'iluvatar' or not native_model_matches(runner.model_config.hf_config):
            raise ValueError('unsupported platform/model')
        p = runner.parallel_config
        if any(getattr(p, name, None) != 1 for name in (
                'tensor_parallel_size', 'pipeline_parallel_size',
                'decode_context_parallel_size', 'data_parallel_size')):
            raise ValueError('TP/PP/DCP/DP1 required')
        prop = torch.cuda.get_device_properties(runner.device)
        if 'BI-V150' not in prop.name or prop.multi_processor_count != 16:
            raise ValueError('BI-V150 16SM required')
        model = runner.get_model()
        head, processor = model.lm_head, model.logits_processor
        if (type(model) is not LlamaForCausalLM or type(head) is not ParallelLMHead
                or type(head.quant_method) is not UnquantizedEmbeddingMethod
                or type(processor) is not LogitsProcessor
                or type(runner.sampler) is not Sampler
                or envs.VLLM_BATCH_INVARIANT or envs.VLLM_COMPUTE_NANS_IN_LOGITS
                or utils.dispatch_unquantized_gemm() is not utils.default_unquantized_gemm
                or processor.logits_as_input or processor.soft_cap is not None
                or processor.scale != 1.0 or processor.org_vocab_size != 130560
                or processor.vocab_size != 130560
                or processor._forward_hooks or processor._forward_pre_hooks
                or getattr(head, 'bias', None) is not None
                or hasattr(head, '_biv150_native_wq')
                or tuple(head.weight.shape) != (130560, 2048)
                or head.weight.dtype != torch.bfloat16 or not head.weight.is_contiguous()
                or runner.sampler.logprobs_mode != 'raw_logprobs'
                or runner.sampler._forward_hooks or runner.sampler._forward_pre_hooks):
            raise ValueError('unsupported actual head/processor/sampler')
        for name, expected in _PINS.items():
            module = importlib.import_module(name)
            if hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest() != expected:
                raise ValueError('installed SDK identity changed: ' + name)
        import flag_gems
        entry = inspect.unwrap(flag_gems.linear)
        if hashlib.sha256(Path(inspect.getsourcefile(entry)).read_bytes()).hexdigest() != 'd431a4fa144ed67dc3deab46dd5beace3612858075f85c39e602447ba3bf768a':
            raise ValueError('FlagGems linear changed')
        table = torch._C._dispatch_dump_table('aten::linear')
        if not any(line.startswith('CUDA:') and 'flag_gems/' in line.replace('\\', '/') for line in table.splitlines()):
            raise ValueError('actual linear dispatch changed')
        self.model, self.weight, self.sampler = weakref.ref(model), weakref.ref(head.weight), weakref.ref(runner.sampler)
        self.proxy = _AlreadyScaledSampler(runner.sampler)
        self.selected, self.sampled = set(), set()
        print('BIV150_HEAD_TEMPERATURE_READY', 'rows32/BM32/BN64/BK64/w8/s1;rows64/BM64/BN256/BK32/w16/s2', flush=True)

    def supported(self, runner, hidden):
        import torch
        model, weight = self.model(), self.weight()
        temp = runner.input_batch.sampling_metadata.temperature
        rows = hidden.shape[0] if hidden.ndim == 2 else None
        return (model is not None and runner.get_model() is model and model.lm_head.weight is weight
                and self.sampler() is runner.sampler
                and (rows in _CONFIGS or _config_for_rows(rows) is not None)
                and runner.input_batch.num_reqs == rows
                and tuple(hidden.shape) == (rows, 2048) and hidden.dtype == torch.bfloat16
                and hidden.is_contiguous() and hidden.device.type == 'cuda'
                and weight.device == hidden.device and not hidden.requires_grad
                and temp is not None and tuple(temp.shape) == (rows,) and temp.dtype == torch.float32
                and temp.is_contiguous() and temp.device == hidden.device
                and tuple(runner.input_batch.sampling_metadata.top_p.shape) == (rows,)
                and runner.input_batch.sampling_metadata.top_p.dtype == torch.float32
                and runner.input_batch.sampling_metadata.top_p.is_contiguous()
                and runner.input_batch.sampling_metadata.top_p.device == hidden.device
                and not torch.cuda.is_current_stream_capturing())

    def compute(self, hidden, metadata):
        import torch
        from vllm_fl.patches.biv150_lm_head_temperature_kernel import head_temperature_kernel
        weight = self.weight()
        rows = hidden.shape[0]
        config = _CONFIGS[rows] if rows in _CONFIGS else _config_for_rows(rows)
        if config is None:
            raise ValueError('head row scope changed before launch')
        out = torch.empty((rows, 130560), dtype=torch.float32, device=hidden.device)
        # 保留原W descriptor、K循环和温度舍入；ragged输入mask由独立constexpr控制。
        kernel = head_temperature_kernel[config['grid']](
            hidden, weight, weight, out, rows, 130560, 2048,
            2048, 1, 2048, 1, 130560, 1, 0, metadata.temperature,
            BIAS=False, GROUP_M=8, SPLIT_K=1, BLOCK_M=config['BLOCK_M'], BLOCK_N=config['BLOCK_N'],
            BLOCK_K=config['BLOCK_K'], EVEN_K=True, num_warps=config['num_warps'],
            num_stages=config['num_stages'], num_ctas=1, RAGGED_M=rows not in _CONFIGS)
        if rows not in self.selected:
            self.selected.add(rows)
            print('BIV150_HEAD_TEMPERATURE_SELECTED',
                  {'rows': rows, 'grid': list(config['grid']), 'hash': getattr(kernel.metadata, 'hash', None),
                   'shared': getattr(kernel.metadata, 'shared', None),
                   'n_regs': getattr(kernel, 'n_regs', None), 'n_spills': getattr(kernel, 'n_spills', None)}, flush=True)
        return out


def compute_logits(runner, hidden, scheduler_output, spec_decode_metadata):
    """唯一正式execute common入口；不安装/修改模型compute_logits。"""
    if os.environ.get(_FLAG, '1') != '1':
        return runner.model.compute_logits(hidden), None
    from vllm.platforms import current_platform
    if (getattr(current_platform, 'vendor_name', None) != 'iluvatar'
            or spec_decode_metadata is not None
            or not _metadata_supported(runner, scheduler_output)):
        return runner.model.compute_logits(hidden), None
    runtime = getattr(runner, _STATE, None)
    if runtime is None:
        try:
            runtime = _Runtime(runner)
        except (AttributeError, ValueError, ImportError, OSError) as error:
            print('BIV150_HEAD_TEMPERATURE_REJECTED', str(error), flush=True)
            runtime = False
        setattr(runner, _STATE, runtime)
    if runtime is False or not runtime.supported(runner, hidden):
        return runner.model.compute_logits(hidden), None
    metadata = runner.input_batch.sampling_metadata
    out = runtime.compute(hidden, metadata)  # launch错误传播；不重试另一实现。
    state = HeadTemperatureState(metadata, metadata.temperature, tuple(runner.input_batch.req_ids), hidden, runtime)
    return out, state


def _state_valid(runner, scheduler_output, state):
    return (state.sampling_metadata is runner.input_batch.sampling_metadata
            and state.temperature is runner.input_batch.sampling_metadata.temperature
            and state.request_ids == tuple(runner.input_batch.req_ids)
            and state.runtime.supported(runner, state.hidden)
            and _metadata_supported(runner, scheduler_output))


def prepare_sampling(runner, logits, state, scheduler_output, grammar_output):
    """mask前确认资格；未知消费者必须恢复原raw logits，不能把scaled当raw。"""
    if state is not None and (grammar_output is not None or not _state_valid(runner, scheduler_output, state)):
        print('BIV150_HEAD_TEMPERATURE_RAW_FALLBACK', 'consumer changed before sample', flush=True)
        return runner.model.compute_logits(state.hidden), None
    return logits, state


def sample_scaled(runner, logits, sampling_metadata, state):
    """_sample原async token更新后调用；调用票据显式消费，无跨call可变flag。"""
    if (state.sampling_metadata is not sampling_metadata
            or state.temperature is not sampling_metadata.temperature
            or state.request_ids != tuple(runner.input_batch.req_ids)):
        return runner.sampler(logits=runner.model.compute_logits(state.hidden), sampling_metadata=sampling_metadata)
    output = state.runtime.proxy(logits=logits, sampling_metadata=sampling_metadata)
    rows = state.hidden.shape[0]
    if rows not in state.runtime.sampled:
        state.runtime.sampled.add(rows)
        print('BIV150_HEAD_TEMPERATURE_SAMPLED', {'rows': rows, 'chain': 'original stats/top-p/softmax/noise; temperature consumed once'}, flush=True)
    return output
