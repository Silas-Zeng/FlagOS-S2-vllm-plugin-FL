"""Private encoded-cache runtime. Stdlib import; explicit injected GPU dependencies.

No data readback, writer in forward, cache compression, allocator change, or
stock fallback for an owned encoded allocation. Device qualification is pending.
"""
from dataclasses import dataclass
import functools
import json
import hashlib
import os
from pathlib import Path
import time

FORMAT = 'cjj-konly-rn32-rne-v2'


def require(value, message):
    if not value: raise RuntimeError('K-only encoded: ' + message)


def storage_id(tensor): return int(tensor.untyped_storage()._cdata)


def descriptor(tensor, broker):
    storage = tensor.untyped_storage()
    result = broker.View(str(tensor.dtype).removeprefix('torch.'), tuple(tensor.shape), tuple(tensor.stride()),
        int(storage._cdata), int(storage.data_ptr()), int(storage.nbytes()), int(tensor.storage_offset()),
        int(tensor.data_ptr()), str(tensor.device))
    result.validate(); return result


@dataclass
class Entry:
    owner: object
    observation: object
    roles: dict
    generation: int


class Registry:
    """Strong ownership for one cold worker epoch; no address-only cache keys.

    Keeping the base tensors alive prevents storage reuse. Reallocation of a
    claimed impl is refused; a fresh worker/capture/prefix lifetime is required.
    """
    def __init__(self, torch, broker):
        self.torch, self.broker = torch, broker
        self.entries = {}; self.impl_owner = {}; self.impl_references = {}; self.by_pointer = {}; self.owned_storages = set()
        self.generation = 0; self.armed = False

    def observe(self, tensor): return descriptor(tensor, self.broker)

    def claim(self, impl, layer, key, value):
        require(self.armed, 'cache writer before qualified cold initialize_kv_cache')
        seen = self.observe(layer); self.broker.layer_pages(seen)
        identity = (seen.storage_id, seen.storage_offset, seen.pointer, seen.shape, seen.strides)
        previous = self.impl_owner.get(id(impl))
        require(previous is None or previous == identity, 'cache reallocation/view change in encoded lifetime')
        self.broker.reject_cache_alias(seen, (self.observe(key), self.observe(value)))
        if identity in self.entries: return self.entries[identity]
        require(not self.torch.cuda.is_current_stream_capturing(), 'new cache views during capture')
        roles = {}
        for role in ('decode79', 'prefill80', 'writer83'):
            params = self.broker.parameters(seen, role, format_id=FORMAT)
            actual = {}
            for name, view in params.items():
                if type(view) is self.broker.View:
                    tensor = self.torch.empty(0, dtype=getattr(self.torch, view.dtype), device=layer.device).set_(
                        layer.untyped_storage(), view.storage_offset, view.shape, view.strides)
                    require(self.observe(tensor) == view, 'constructed pointer/stride/offset differs: ' + name)
                    actual[name] = tensor
                else: actual[name] = view
            roles[role] = actual
        self.generation += 1
        entry = Entry(layer, seen, roles, self.generation)
        self.entries[identity] = entry; self.impl_owner[id(impl)] = identity; self.impl_references[id(impl)] = impl
        self.by_pointer[(seen.storage_id, seen.pointer, seen.pointer + 8192)] = entry
        self.owned_storages.add(seen.storage_id)
        return entry

    def lookup(self, key, value):
        k, v = self.observe(key), self.observe(value)
        entry = self.by_pointer.get((k.storage_id, k.pointer, v.pointer)) if k.storage_id == v.storage_id else None
        if entry is not None:
            root = entry.observation; expected = (root.shape[0], 16, 2, 128)
            require(k.dtype == v.dtype == 'bfloat16' and k.shape == v.shape == expected, 'unknown cache view dtype/shape')
            require(k.strides == v.strides == (8192, 256, 128, 1), 'unknown cache view strides')
            return entry
        require(k.storage_id not in self.owned_storages and v.storage_id not in self.owned_storages,
                'unrecognized alias of encoded storage')
        return None

    def owns(self, value):
        if hasattr(value, 'untyped_storage'):
            return storage_id(value) in self.owned_storages
        if isinstance(value, dict): return any(self.owns(v) for v in value.values())
        if isinstance(value, (tuple, list)): return any(self.owns(v) for v in value)
        return False


class Runtime:
    def __init__(self, torch, broker, kernels, gqa):
        self.torch, self.kernels, self.gqa = torch, kernels, gqa
        self.registry = Registry(torch, broker)
        self.counts = dict(writer=0, selected={}, encoded_to_stock_fallback=0, profile_fallback={},
                           quantization_in_forward=0, host_counts_not_graph_replays=True,
                           encoded_stock_native_bypass_attempts=0, stock_parent_producer_host_calls=0)
        # Canonical source routes exist; arbitrary service geometry is unproved.
        # Disabled M2 uses encoded2D; disabled raw3D rejects before launch.
        self.raw3d_enabled = True; self.mixed_pair_enabled = False; self.multi2_enabled = True
        self.q32_enabled = True
        self.shape_records = {}; self.kernel_records = {}; self.shape_overflow = 0
        self.stats_path = None; self.final_export = None
        self.q32_hoist = None
        self.multi2_hoist = None
        self.native_decode = None; self.native_decode_identity = None
        self.mixed_native = None; self.mixed_native_identity = None

    @staticmethod
    def typed(value):
        # Never repr a Tensor or read its data merely to serialize metadata.
        if value is None or type(value) in (bool,str,int,float):
            return dict(type=type(value).__name__,value=value)
        if isinstance(value,int): return dict(type=type(value).__name__,value=int(value))
        if type(value) is dict:
            return dict(type='dict',items=[[Runtime.typed(k),Runtime.typed(v)] for k,v in value.items()])
        if type(value) in (list,tuple):
            return dict(type=type(value).__name__,items=[Runtime.typed(v) for v in value])
        return dict(type=type(value).__module__+'.'+type(value).__name__,unrepresented=True)

    @staticmethod
    def geometry(value):
        return dict(dtype=str(value.dtype),shape=list(value.shape),strides=list(value.stride()),
                    storage_offset=value.storage_offset(),device=str(value.device))

    def note(self, profile, handle, launch=None, grid=None):
        launch = {} if launch is None else launch
        self.counts['selected'][profile] = self.counts['selected'].get(profile,0)+1
        kernel_hash = str(handle.hash) if getattr(handle,'hash',None) else None
        query = launch.get('query_ptr'); lengths = launch.get('seq_lens_ptr')
        pointers = {k:self.geometry(v) for k,v in launch.items() if hasattr(v,'untyped_storage')}
        scalar_names = ('num_seqs','BLOCK_M','BLOCK_Q','TILE_SIZE','num_warps','num_stages','IS_3D',
            'KV_QUANT_MODE','HEAD_SIZE','HEAD_SIZE_PADDED','num_query_heads','num_queries_per_kv',
            'NUM_SEGMENTS_PER_SEQ','ROWS','PAGE_SIZE','SEGMENTS','key_row_stride','value_row_stride',
            'key_rows','value_rows','slot_count','cache_pages')
        scalars = {k:self.typed(launch[k]) for k in scalar_names if k in launch}
        record = dict(profile=profile,hash=kernel_hash,grid=list(grid) if grid is not None else None,
            capture=self.torch.cuda.is_current_stream_capturing(),pointers=pointers,selected_scalars=scalars)
        key = json.dumps(record,sort_keys=True,separators=(',',':'))
        if key in self.shape_records: self.shape_records[key]['host_calls'] += 1
        elif len(self.shape_records)<4096: self.shape_records[key] = dict(**record,host_calls=1)
        else: self.shape_overflow += 1
        identity = (profile,kernel_hash)
        if identity not in self.kernel_records:
            src = getattr(handle,'src',None); fn = getattr(src,'fn',None)
            source = getattr(fn,'src',None); asm = getattr(handle,'asm',{})
            metadata = getattr(handle,'metadata',None)
            resources = dict(n_regs=getattr(handle,'n_regs',None),n_spills=getattr(handle,'n_spills',None),
                **{k:getattr(metadata,k,None) for k in ('shared','num_warps','num_stages','warp_size','use_sme')})
            known = (type(resources['n_regs']) is int and resources['n_regs']>0
                and all(type(resources[k]) is int and resources[k]>=0 for k in ('n_spills','shared','use_sme'))
                and all(type(resources[k]) is int and resources[k]>0 for k in ('num_warps','num_stages','warp_size')))
            item = dict(profile=profile,hash=kernel_hash,
                actual_function_name=getattr(metadata,'name',None),source_function_name=getattr(fn,'__name__',None),
                source_digest=hashlib.sha256(source.encode()).hexdigest() if isinstance(source,str) else None,
                typed_ABI={k:self.typed(getattr(src,k,None)) for k in ('signature','constants','attrs')},
                actual_artifact_sha256={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for k,v in asm.items() if isinstance(v,(str,bytes))},resources=resources,
                post_launch_resource_fields_known=known,target=getattr(self,'actual_target',None),
                verified_installed_sources=getattr(self,'verified_installed_sources',None))
            if self.q32_hoist and kernel_hash in self.q32_hoist.by_hash:
                binding = self.q32_hoist.by_hash[kernel_hash]
                item['fresh_Q32_hoist'] = dict(artifact_directory=binding['artifact_directory'],
                    override_sha256=binding['override_sha256'],actual_native_verified=True,
                    complete_actual_ABI_bound=True,numerical_verified=False,service_verified=False)
            if self.multi2_hoist and kernel_hash in self.multi2_hoist.by_hash:
                binding = self.multi2_hoist.by_hash[kernel_hash]
                item['fresh_multi2_hoist'] = dict(artifact_directory=binding['artifact_directory'],
                    override_sha256=binding['override_sha256'],actual_native_verified=True,
                    complete_actual_ABI_bound=True,numerical_verified=False,service_verified=False)
            self.kernel_records[identity] = item
            print('BIV150_KONLY_ENCODED_SELECTED',json.dumps(dict(profile=profile,hash=kernel_hash,
                implementation_sha256=getattr(self,'installed_digest',None),format=FORMAT,
                query_rows=None if query is None else query.shape[0],
                sequences=None if lengths is None else lengths.shape[0],
                I8_fragment_hoist_verified=bool((self.q32_hoist and kernel_hash in self.q32_hoist.by_hash)
                    or (self.multi2_hoist and kernel_hash in self.multi2_hoist.by_hash)),
                hoist_numerical_verified=False,host_event_not_Graph_replay_count=True)),flush=True)
        if kernel_hash:
            self.counts.setdefault('actual_kernel_hashes',{}).setdefault(profile,set()).add(kernel_hash)

    def finalize(self, reason):
        if self.final_export is not None: return self.final_export
        try:
            ticks = None
            stat = Path('/proc/self/stat')
            if stat.is_file(): ticks = int(stat.read_text().rsplit(')',1)[1].split()[19])
            payload = dict(schema='cjj-konly-encoded-final-stats-v1',reason=reason,pid=os.getpid(),
                start_ticks=ticks,unix_time=time.time(),implementation_sha256=getattr(self,'installed_digest',None),
                stats=self.stats(),host_shapes=list(self.shape_records.values()),
                shape_records_complete=self.shape_overflow==0,omitted_shape_host_calls=self.shape_overflow,
                actual_kernel_records=list(self.kernel_records.values()),
                graph_kernel_replay_counts_available=False,service_numeric_verified=False,precision_gate=False)
            data = (json.dumps(payload,indent=2)+'\n').encode(); digest = hashlib.sha256(data).hexdigest()
            if self.stats_path is not None:
                path = Path(str(self.stats_path).replace('{pid}',str(os.getpid())))
                require(path.is_absolute(),'absolute private stats path required')
                path.parent.mkdir(parents=True,exist_ok=True)
                require(not path.exists(),'refuse overwriting previous final stats')
                temp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
                with temp.open('xb') as stream: stream.write(data)
                temp.replace(path)
                print('BIV150_KONLY_ENCODED_FINAL_STATS',json.dumps(dict(path=str(path),sha256=digest,
                    bytes=len(data),pid=os.getpid(),reason=reason,shape_records=len(self.shape_records),
                    graph_kernel_replay_counts_available=False)),flush=True)
            else:
                print('BIV150_KONLY_ENCODED_FINAL_STATS_LOG_ONLY',json.dumps(payload),flush=True)
            self.final_export = payload
            return payload
        except Exception as error:
            # Statistics failure must not mask a model/shutdown result. A missing
            # export is nevertheless an evidence gap for the external controller.
            print('BIV150_KONLY_ENCODED_FINAL_STATS_ERROR',type(error).__name__,str(error)[:160],flush=True)
            return None

    def shutdown(self, original):
        @functools.wraps(original)
        def wrapper(runner,*args,**kwargs):
            try: return original(runner,*args,**kwargs)
            finally: self.finalize('model_runner_shutdown')
        return wrapper

    def fallback(self, reason):
        values = self.counts['profile_fallback']; values[reason] = values.get(reason, 0) + 1

    def stats(self):
        result = dict(self.counts); result['actual_kernel_hashes'] = {
            k: sorted(v) for k, v in self.counts.get('actual_kernel_hashes', {}).items()}
        result.update(format=FORMAT, claimed_allocations=len(self.registry.entries),
            dynamic_I8_hoist_verified=bool((self.q32_hoist and self.q32_hoist.by_hash)
                or (self.multi2_hoist and self.multi2_hoist.by_hash)), all_service_shapes_verified=False,
            raw3d_enabled=self.raw3d_enabled, mixed_pair_enabled=self.mixed_pair_enabled,
            multi2_enabled=self.multi2_enabled, capacity_reclaimed=False, precision_gate=False)
        result['native_decode'] = dict(enabled=False) if self.native_decode is None else self.native_decode.stats()
        result['mixed_native'] = dict(enabled=False) if self.mixed_native is None else self.mixed_native.verify(self)
        result['q32_hoist'] = dict(enabled=False) if self.q32_hoist is None else self.q32_hoist.stats()
        if self.multi2_hoist is not None: result['multi2_hoist'] = self.multi2_hoist.stats()
        return result

    def initialize(self, original):
        @functools.wraps(original)
        def wrapper(runner, *args, **kwargs):
            require(not self.registry.entries, 'reinitialize requires fresh worker/prefix/capture lifetime')
            config = runner.vllm_config
            require(getattr(config, 'kv_transfer_config', None) is None, 'KV connector/offload not qualified')
            require(getattr(config, 'speculative_config', None) is None, 'speculative KV ownership not qualified')
            parallel = config.parallel_config
            require(parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1,
                    'only single-device private epoch qualified')
            require(config.cache_config.cache_dtype in ('auto', 'bfloat16'), 'built-in KV quantization forbidden')
            kv_config = args[0] if args else kwargs['kv_cache_config']
            groups = kv_config.kv_cache_groups
            require(len(groups) == 1, 'one fixed full-attention cache group')
            spec = groups[0].kv_cache_spec
            require((spec.block_size,spec.num_kv_heads,spec.head_size) == (16,2,128)
                    and spec.dtype == self.torch.bfloat16, 'resolved KV spec geometry/dtype')
            require(config.model_config.dtype == self.torch.bfloat16, 'BF16 model source')

            self.registry.armed = True
            try: return original(runner, *args, **kwargs)
            except BaseException:
                self.registry.armed = False; raise
        return wrapper

    def writer(self, original):
        @functools.wraps(original)
        def wrapper(impl, layer, key, value, cache, slots):
            require((impl.num_heads, impl.num_kv_heads, impl.head_size) == (16, 2, 128), 'model head geometry')
            require(str(impl._kv_quant_mode) in ('0', 'KVQuantMode.NONE') and not impl._is_per_token_head_quant,
                    'only NONE private format')
            require(impl.kv_sharing_target_layer_name is None, 'shared writer ownership not qualified')
            entry = self.registry.claim(impl, cache, key, value)
            require(str(slots.dtype) == 'torch.int64' and slots.ndim == 1 and slots.stride(0) == 1,
                    'actual int64 slot ABI')
            require(slots.device == key.device == value.device == cache.device, 'writer device')
            count = slots.shape[0]
            require(count <= min(key.shape[0], value.shape[0]), 'writer source extent')
            require(not self.registry.owns(slots), 'slot/cache alias')
            if count:
                views = entry.roles['writer83']
                handle = self.kernels['writer'][(count, 2)](key, value, slots,
                    views['code_full'], views['scale_full'], views['value_full'],
                    key_row_stride=key.stride(0), value_row_stride=value.stride(0),
                    key_rows=key.shape[0], value_rows=value.shape[0], slot_count=count,
                    cache_pages=entry.observation.shape[0], num_warps=1, num_stages=1)
                self.note('writer83',handle,dict(key_source=key,value_source=value,slots=slots,
                    key_row_stride=key.stride(0),value_row_stride=value.stride(0),key_rows=key.shape[0],
                    value_rows=value.shape[0],slot_count=count,cache_pages=entry.observation.shape[0]),(count,2))
            self.counts['writer'] += 1
            # Preserve original do_kv_cache_update's None result/custom-op dummy.
            return None
        return wrapper

    def facade(self, original, family):
        return KernelFacade(self, original, family)

    def launch_producer(self, original, private_reader):
        """Intercept format ownership BEFORE main's native BF16 eligibility.

        Before the encoded allocation epoch, unowned stock profiling tensors
        keep main's entire native/Triton path. Once armed, this private worker
        admits only its registered encoded caches; other formats are unknown.
        """
        @functools.wraps(original)
        def wrapper(grid, **kwargs):
            key, value = kwargs.get('key_cache_ptr'), kwargs.get('value_cache_ptr')
            require(key is not None and value is not None, 'producer cache pointers absent')
            if self.registry.owns((key,value)):
                # The private facade validates the exact pair/view/ABI before
                # launch. Even an unrecognized owned alias never reaches main
                # native eligibility or its nominal BF16 pointer interpretation.
                self.counts['encoded_stock_native_bypass_attempts'] += 1
                return private_reader[grid](**kwargs)
            require(not self.registry.armed, 'unknown producer cache format after encoded epoch')
            self.counts['stock_parent_producer_host_calls'] += 1
            return original(grid, **kwargs)
        return wrapper

    def forbid_transfer(self, original):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            require(not self.registry.owns((args, kwargs)), 'encoded copy/swap/layout conversion not qualified')
            return original(*args, **kwargs)
        return wrapper


class KernelFacade:
    def __init__(self, runtime, original, family):
        self.runtime, self.original, self.family = runtime, original, family

    def __getattr__(self, name): return getattr(self.original, name)

    def __getitem__(self, grid):
        fallback = self.original[grid]
        def invoke(*args, **kwargs):
            runtime = self.runtime
            key, value = kwargs.get('key_cache_ptr'), kwargs.get('value_cache_ptr')
            if key is None or value is None:
                require(not runtime.registry.owns((args, kwargs)), 'encoded launch lacks typed cache arguments')
                return fallback(*args, **kwargs)
            entry = runtime.registry.lookup(key, value)
            if entry is None:
                require(not runtime.registry.armed, 'untagged cache after encoded epoch initialized')
                return fallback(*args, **kwargs)
            require(not args and 'ir_override' not in kwargs, 'only exact private kwargs/no foreign IR')
            selected = dict(kwargs)
            require(str(kwargs['query_ptr'].dtype) == 'torch.bfloat16', 'query dtype')
            require(tuple(kwargs['query_ptr'].shape[1:]) == (16, 128), 'query heads/dim')
            if self.family != 'gqa':
                require(kwargs.get('HEAD_SIZE') == kwargs.get('HEAD_SIZE_PADDED') == 128
                        and kwargs.get('BLOCK_SIZE') == 16 and kwargs.get('num_query_heads') == 16
                        and kwargs.get('num_queries_per_kv') == 8, 'private SDK geometry')
                require(kwargs.get('USE_CAUSAL') is True and kwargs.get('SLIDING_WINDOW') == 0
                        and kwargs.get('CHUNK_LOOKBACK') == kwargs.get('CHUNK_SIZE') == -1, 'private causal/window semantics')
                require(all(kwargs.get(k) is False for k in ('USE_TD','USE_TD_QO','Q_IS_FP8','USE_FP8',
                    'USE_PER_SEQ_CAUSAL','USE_MM_PREFIX','USE_ALIBI_SLOPES','USE_ALIBI_SQRT','USE_QQ_BIAS',
                    'USE_SOFTCAP','USE_SINKS')), 'unsupported encoded SDK feature')
                require(str(kwargs.get('KV_QUANT_MODE')) in ('0','KVQuantMode.NONE'), 'private mode0')
            if self.family == 'gqa':
                require(kwargs.get('HEAD_SIZE') == 128 and kwargs.get('PAGE_SIZE') == 16
                        and kwargs.get('ROWS') == kwargs.get('queries_per_kv') == 8, 'private GQA geometry')
                views = entry.roles['decode79']
                selected.update(views, query_code_ptr=views['key_cache_ptr'], query_scale_ptr=views['key_scale_ptr'])
                profile = 'decode'; target_grid = grid
                if runtime.mixed_native is not None and runtime.mixed_native.try_produce(runtime,entry,selected,target_grid):
                    return None
                if runtime.native_decode is not None and runtime.native_decode.try_produce(runtime,entry,selected,target_grid):
                    return None
            else:
                views = entry.roles['prefill80']; selected.update(views)
                n = kwargs['num_seqs']; qrows = kwargs['query_ptr'].shape[0]
                if kwargs.get('IS_3D'):
                    require(runtime.raw3d_enabled, 'unqualified raw3D reached kernel')
                    profile = 'raw3d'; target_grid = grid
                else:
                    capture = runtime.torch.cuda.is_current_stream_capturing()
                    if n == 1 and qrows >= 1024 and runtime.q32_enabled and not capture:
                        profile, bm, bq, warps = 'q32', 256, 32, 16
                    elif n == 2 and qrows >= 1024 and runtime.multi2_enabled and not capture:
                        profile, bm, bq, warps = 'multi2', 256, 32, 16
                    else:
                        profile, bm, bq, warps = 'normalized2d128', 128, 16, 8
                        if n == 2 and qrows >= 1024: runtime.fallback('multi2_to_encoded_2d_pending_numeric')
                    selected.update(BLOCK_M=bm, BLOCK_Q=bq, TILE_SIZE=64, num_warps=warps, num_stages=1)
                    target_grid = (qrows // bq + n, 2)
            fn = runtime.kernels[profile]
            override, binding = ({}, None)
            hoist = runtime.q32_hoist if profile == 'q32' else runtime.multi2_hoist if profile == 'multi2' else None
            if hoist is not None:
                override, binding = hoist.prepare(fn, selected, target_grid,
                    capture=runtime.torch.cuda.is_current_stream_capturing())
            # Exactly one reader launch. No postlaunch retry/fallback.
            handle = fn[target_grid](**selected, **override)
            if binding is not None: hoist.after_launch(handle, fn, binding)
            runtime.note(profile, handle, selected, target_grid)
            return handle
        return invoke
