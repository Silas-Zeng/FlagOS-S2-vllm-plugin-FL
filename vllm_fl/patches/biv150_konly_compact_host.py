"""Cold get-spec/reshape/lifecycle hooks; no import-time mutation or GPU imports."""
from dataclasses import fields
import functools
import importlib
import json
from pathlib import Path
import sys
import types

from .biv150_konly_compact_identity import FORMAT, PAGE_BYTES, CORE_PINS, need, verify_core, verify_package


class Host:
    def __init__(self, runner_class, torch, runtime, platform):
        verify_package()
        core = verify_core()
        self.verified_core_pins = dict(CORE_PINS)
        spec_module = importlib.import_module('vllm_fl.patches.biv150_konly_compact_spec')
        self.spec_class = spec_module.CompactFullAttentionSpec
        self.ensure_registered = spec_module.ensure_registered
        self.full_class = core['vllm.v1.kv_cache_interface'].FullAttentionSpec
        self.runner_class, self.torch, self.runtime = runner_class, torch, runtime
        self.platform = platform
        need(platform.vendor_name == 'iluvatar' and platform.is_cuda() is False,
             'unqualified CUDA temporary-profile-cache lifecycle')
        self.module = sys.modules[runner_class.__module__]
        self.parent_get_spec = runner_class.get_kv_cache_spec
        self.parent_allocate = runner_class._allocate_kv_cache_tensors
        self.parent_initialize_tensors = runner_class.initialize_kv_cache_tensors
        self.parent_reshape = self.module._reshape_attention_kv_cache
        need(self.parent_reshape is core['vllm.v1.worker.gpu.attn_utils']._reshape_attention_kv_cache,
             'unknown parent reshape hook')
        need(not getattr(self.parent_get_spec, '_compact_hook', False), 'get-spec already wrapped')
        for name, fn in (('get_kv_cache_spec', self.parent_get_spec),
                         ('_allocate_kv_cache_tensors', self.parent_allocate),
                         ('initialize_kv_cache_tensors', self.parent_initialize_tensors)):
            need(isinstance(fn, types.FunctionType) and fn.__globals__ is vars(self.module)
                 and fn.__code__.co_qualname == runner_class.__qualname__ + '.' + name
                 and Path(fn.__code__.co_filename).resolve() == Path(self.module.__file__).resolve()
                 and not fn.__closure__ and not hasattr(fn, '__wrapped__'), 'unknown parent method: ' + name)
        need(self.module.torch is torch
             and self.module.bind_kv_cache is core['vllm.v1.worker.utils'].bind_kv_cache,
             'parent Torch/bind global alias changed')
        self.original_code = (self.parent_get_spec.__code__, self.parent_allocate.__code__,
                              self.parent_initialize_tensors.__code__, self.parent_reshape.__code__)
        self.records = {}; self.runner = None; self.initialized = False; self.started = False
        self.layer_names = None
        self.get_spec_hook = self._get_spec()
        self.reshape_hook = self._reshape()
        self.initialize_hook = None

    def scope(self, runner):
        config = runner.vllm_config
        need(self.platform.is_cuda() is False, 'platform cache profiling policy changed')
        need(not any(name in vars(runner) for name in ('get_kv_cache_spec', 'initialize_kv_cache',
             '_allocate_kv_cache_tensors', 'initialize_kv_cache_tensors')), 'instance cache hook override')
        need(getattr(config, 'speculative_config', None) is None, 'speculative decoding unsupported')
        need(getattr(config, 'kv_transfer_config', None) is None
             and getattr(config, 'ec_transfer_config', None) is None
             and getattr(config, 'kv_offload_config', None) is None, 'connector/offload unsupported')
        parallel = config.parallel_config
        need(all(getattr(parallel, name, None) == 1 for name in
                 ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size',
                  'decode_context_parallel_size', 'prefill_context_parallel_size')),
             'only single-device UniProc metadata path')
        need(parallel.distributed_executor_backend == 'uni', 'non-UniProc executor unsupported')
        need(config.model_config.dtype == self.torch.bfloat16 and not config.model_config.is_encoder_decoder,
             'only BF16 decoder model')
        model = config.model_config.hf_config
        need(tuple(getattr(model, n, None) for n in ('num_hidden_layers', 'num_attention_heads',
             'num_key_value_heads', 'hidden_size')) == (42, 16, 2, 2048), 'MiniCPM cache geometry')
        need(config.cache_config.cache_dtype in ('auto', 'bfloat16')
             and config.cache_config.block_size == 16, 'stock quantization/block size unsupported')
        # Defaults are None for KV offload, 0GiB for UVA and group_size0 for
        # prefetch. offload_num_in_group defaults1 and is not an enable flag.
        need(config.cache_config.kv_offloading_size is None
             and config.offload_config.uva.cpu_offload_gb == 0
             and config.offload_config.prefetch.offload_group_size == 0, 'offload size unsupported')
        need(not runner.shared_kv_cache_layers and runner.cross_layers_kv_cache is None,
             'shared/cross-layer KV unsupported')

    def spec_owned(self, spec):
        return (type(spec) is self.spec_class and spec.format_id == FORMAT
                and spec.page_size_bytes == PAGE_BYTES and spec.page_size_padded is None
                and (spec.block_size, spec.num_kv_heads, spec.head_size, spec.head_size_v, spec.dtype)
                == (16, 2, 128, 128, self.torch.bfloat16))

    def _get_spec(self):
        @functools.wraps(self.parent_get_spec)
        def get_spec(runner):
            self.scope(runner)
            need(not self.started, 'get-spec after allocation epoch began')
            self.ensure_registered(runner.vllm_config)
            original = self.parent_get_spec(runner)
            self.scope(runner)
            need(type(original) is dict and len(original) == 42
                 and all(type(spec) is self.full_class for spec in original.values()),
                 'all 42 stock full-attention specs required')
            names = tuple(sorted(original))
            need(self.layer_names is None or self.layer_names == names, 'layer set changed')
            result = {name: self.spec_class(**{f.name: getattr(spec, f.name) for f in fields(self.full_class)})
                      for name, spec in original.items()}
            need(all(self.spec_owned(spec) for spec in result.values()), 'incomplete compact spec conversion')
            self.layer_names = names
            return result
        get_spec._compact_hook = True
        return get_spec

    def _reshape(self):
        @functools.wraps(self.parent_reshape)
        def reshape(raw, spec, shape, order, blocks, packing):
            # The hook is installed only in a dedicated compact worker. No stock
            # interpretation is admitted after that worker starts allocation.
            need(self.started and not self.initialized and self.spec_owned(spec), 'unknown allocation/spec epoch')
            need(packing is None and tuple(order) == (0, 1, 2, 3, 4)
                 and tuple(shape) == (blocks, 2, 16, 2, 128), 'only independent NHD page16 layout')
            need(type(blocks) is int and 0 < blocks <= 2**31-1, 'physical block count')
            need(raw.dtype == self.torch.int8 and raw.ndim == 1 and raw.is_contiguous()
                 and raw.numel() == blocks*PAGE_BYTES and raw.storage_offset() == 0, 'raw compact byte allocation')
            storage = raw.untyped_storage(); identity = int(storage._cdata)
            need(storage.nbytes() == blocks*PAGE_BYTES and raw.data_ptr() % 256 == 0,
                 'retained old backing or unaligned compact storage')
            need(str(raw.device) == 'cuda:0' and identity not in self.records, 'shared/foreign cache allocation')
            carrier = self.torch.as_strided(raw.view(self.torch.bfloat16),
                size=(blocks, 2, 16, 2, 128), stride=(6272, 2176, 256, 128, 1))
            self.records[identity] = (raw, carrier, blocks)
            return carrier
        return reshape

    def initialize(self, parent_initialize):
        @functools.wraps(parent_initialize)
        def initialize(runner, *args, **kwargs):
            config = args[0] if args else kwargs.get('kv_cache_config')
            need(config is not None, 'KV cache configuration argument missing')
            need((len(args) < 2 or args[1] is False) and kwargs.get('is_profiling', False) is False,
                 'temporary profiling allocation unsupported')
            self.scope(runner)
            need(not self.started and self.runner is None, 'reinitialization requires a fresh worker')
            need(self.layer_names is not None and len(config.kv_cache_groups) == 1,
                 'compact get-spec/group path missing')
            group = config.kv_cache_groups[0]
            need(self.spec_owned(group.kv_cache_spec) and tuple(sorted(group.layer_names)) == self.layer_names,
                 'old/unknown group format')
            need(len(config.kv_cache_tensors) == 42 and type(config.num_blocks) is int
                 and config.num_blocks > 0, 'independent layer allocation plan')
            names = []
            for tensor in config.kv_cache_tensors:
                need(tensor.block_stride == 0 and len(tensor.shared_by) == 1
                     and tensor.size == config.num_blocks*PAGE_BYTES, 'packed/old allocation plan')
                names.extend(tensor.shared_by)
            need(tuple(sorted(names)) == self.layer_names, 'allocation layer coverage')
            self.started = True; self.runner = runner
            result = parent_initialize(runner, *args, **kwargs)
            need(len(self.records) == 42, 'compact reshape did not cover all layers')
            context = runner.compilation_config.static_forward_context
            for name in names:
                cache = context[name].kv_cache
                identity = int(cache.untyped_storage()._cdata)
                need(identity in self.records and self.records[identity][1] is cache,
                     'bound cache differs from compact reshape')
                need(cache.untyped_storage().nbytes() == config.num_blocks*PAGE_BYTES,
                     'bound cache storage extent changed')
            self.initialized = True
            print('BIV150_KONLY_COMPACT_ALLOCATION',json.dumps(self.stats(),sort_keys=True),flush=True)
            return result
        self.initialize_hook = initialize
        return initialize

    def receipt(self, spec):
        need(self.initialized and self.runtime.registry.armed and self.runner is not None,
             'compact epoch not initialized')
        need(self.spec_owned(spec), 'unowned/old cache spec')
        self.scope(self.runner)
        need(self.runner_class.get_kv_cache_spec is self.get_spec_hook
             and self.module._reshape_attention_kv_cache is self.reshape_hook
             and self.runner_class.initialize_kv_cache is self.initialize_hook,
             'compact host binding changed')
        need(self.runner_class._allocate_kv_cache_tensors is self.parent_allocate
             and self.runner_class.initialize_kv_cache_tensors is self.parent_initialize_tensors,
             'parent allocator/initializer changed')
        need(self.original_code == (self.parent_get_spec.__code__, self.parent_allocate.__code__,
             self.parent_initialize_tensors.__code__, self.parent_reshape.__code__), 'parent host code changed')
        need(len(self.records) == 42 and self.ensure_registered(self.runner.vllm_config).__module__
             == 'vllm_fl.patches.biv150_konly_compact_manager', 'compact manager identity')
        return dict(compact_host_registered=True, format=FORMAT, page_bytes=PAGE_BYTES,
                    allocation_metadata_layers=42, device_numeric_verified=False, capacity_reclaimed=False)

    def stats(self):
        """Storage metadata only, including partial evidence after startup failure."""
        rows = [dict(storage_bytes=raw.untyped_storage().nbytes(), physical_pages=blocks,
                     carrier_shape=list(carrier.shape), carrier_strides=list(carrier.stride()),
                     pointer_offset_bytes=carrier.data_ptr()-raw.data_ptr())
                for raw, carrier, blocks in self.records.values()]
        return dict(format=FORMAT, page_bytes=PAGE_BYTES, initialized=self.initialized,
                    configured_num_blocks=getattr(getattr(self.runner,'kv_cache_config',None),'num_blocks',None),
                    recorded_layers=len(rows), records=rows,
                    raw_storage_bytes=sum(row['storage_bytes'] for row in rows),
                    verified_core_source_sha256=self.verified_core_pins,
                    current_scope_needs_kv_zeroing=False, actual_data_readback=False,
                    device_numeric_verified=False, capacity_reclaimed=False,
                    performance_verified=False)
