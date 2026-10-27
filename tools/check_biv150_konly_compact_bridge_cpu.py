"""Exercise pinned KV host allocation/zeroing methods with CPU metadata mocks."""
import argparse
import ast
import builtins
import copy
from dataclasses import dataclass, fields, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import biv150_konly_compact_kv_bridge as b


def digest(data): return hashlib.sha256(data).hexdigest()


def git_bytes(repo, commit, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':' + path])


PINS = {
    'vllm/v1/kv_cache_interface.py': '7e3c220998267190a622e216b2b125306807cf8be3ec1e9a8e84fe6e7a293003',
    'vllm/v1/attention/backends/triton_attn.py': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    'vllm/v1/core/kv_cache_utils.py': '3ca561818019cef84c43c6d3839627145fd049f9479d40fc733ae142d6d71df3',
    'vllm/v1/core/single_type_kv_cache_manager.py': '22e3677acf0e769461bdbf278df031d6e362c97aa1ba7b862c071b76d47d8914',
    'vllm/v1/worker/utils.py': '36cae980896d5607ce238606ec6c053eff17b7597a0c9185549f94583c1b9de6',
    'vllm/v1/worker/gpu/attn_utils.py': 'a351178df66fb5364cbe663ba7f25346c6317927fcc29a8c00f8972a23440ace',
    'vllm/v1/kv_cache_spec_registry.py': '16b3f6458538910f72cdac48554c20432c7ac0e5122e91c8a6219a1ec25f44fe',
    'vllm/v1/worker/kv_connector_model_runner_mixin.py': 'd140e79e93b8670a2100b8b365606d3af402430a59c990ef65820da987ef4974',
}

ARCHIVED = {
    'vllm/v1/kv_cache_interface.py': '/private/tmp/cjj-target-kv-cache-service-sources-20261027/v1/kv_cache_interface.py',
    'vllm/v1/attention/backends/triton_attn.py': '/private/tmp/cjj-target-kv-cache-service-sources-20261027/v1/attention/backends/triton_attn.py',
    'vllm/v1/core/kv_cache_utils.py': '/private/tmp/cjj-target-kv-cache-service-sources-20261027/v1/core/kv_cache_utils.py',
    'vllm/v1/core/single_type_kv_cache_manager.py': '/private/tmp/cjj-target-konly-cache-lifecycle-sources-v1-20261027/vllm/v1/core/single_type_kv_cache_manager.py',
    'vllm/v1/worker/utils.py': '/private/tmp/cjj-target-kv-zeroer-platform-sources-v1-20261027/vllm/v1/worker/utils.py',
    'vllm/v1/worker/gpu/attn_utils.py': '/private/tmp/cjj-target-konly-cache-lifecycle-sources-v1-20261027/vllm/v1/worker/gpu/attn_utils.py',
    'vllm/v1/kv_cache_spec_registry.py': '/private/tmp/cjj-target-kv-spec-registry-20261028.py',
    'vllm/v1/worker/kv_connector_model_runner_mixin.py': '/private/tmp/cjj-target-konly-cache-lifecycle-sources-v1-20261027/vllm/v1/worker/kv_connector_model_runner_mixin.py',
}


def execute(nodes, scope):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), '<CPU-pinned-host-source>', 'exec'), scope)


class Tensor:
    def __init__(self, shape, dtype='int8', strides=None, size=None, base=4096):
        self.shape, self.dtype, self.device = tuple(shape), dtype, 'cuda:0'
        self.ndim = len(shape); self._bytes = math.prod(shape) * {'int8': 1, 'bfloat16': 2}[dtype] if size is None else size
        self._strides = tuple(strides) if strides else (1,)
        self._base = base
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self.ndim == 1 and self._strides == (1,)
    def storage_offset(self): return 0
    def data_ptr(self): return self._base
    def untyped_storage(self): return SimpleNamespace(nbytes=lambda: self._bytes)
    def element_size(self): return {'int8': 1, 'bfloat16': 2}[self.dtype]
    def stride(self, dim=None): return self._strides if dim is None else self._strides[dim]
    def view(self, dtype): return Tensor((self._bytes // {'int8': 1, 'bfloat16': 2}[dtype],), dtype, size=self._bytes,base=self._base)


def run(a):
    assert not {'torch', 'triton', 'vllm', 'numpy'}.intersection(sys.modules)
    source = {}
    for path, pin in PINS.items():
        target = a.vllm_root / path if a.vllm_root else Path(ARCHIVED[path])
        raw = target.read_bytes(); assert digest(raw) == pin
        source[path] = raw.decode()
    module = ModuleType('_cjj_compact_bridge_cpu'); sys.modules[module.__name__] = module
    scope = module.__dict__
    @dataclass(frozen=True)
    class Mode:
        is_nvfp4: bool = False
        is_per_token_head: bool = False
    mode = Mode()
    allocations = []
    def zeros(size, *, dtype, device):
        value = Tensor((size,), dtype,base=4096+len(allocations)*65536); allocations.append(value); return value
    torch = SimpleNamespace(int8='int8', bfloat16='bfloat16', float32='float32',
        int64='int64', uint64='uint64', Tensor=Tensor, zeros=zeros,
        as_strided=lambda raw, *, size, stride: Tensor(size, raw.dtype, stride, raw._bytes,raw._base),
        empty=lambda *args, **kwargs: SimpleNamespace(),
        tensor=lambda values, **kwargs: tuple(values))
    scope.update(dataclass=dataclass, fields=fields, replace=replace, copy=copy, FORMAT=b.FORMAT,
                 KVQuantMode=SimpleNamespace(NONE=mode), torch=torch,
                 get_dtype_size=lambda dt: 4 if dt == 'float32' else 2,
                 cdiv=lambda x, y: (x + y - 1) // y,
                 MLAAttentionSpec=type('MLAAttentionSpec', (), {}))
    tree = ast.parse(source['vllm/v1/kv_cache_interface.py'])
    execute([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('KVCacheSpec', 'AttentionSpec', 'FullAttentionSpec')], scope)
    execute(ast.parse(b.SPEC_SOURCE).body, scope)
    full, compact = scope['FullAttentionSpec'], scope['CompactFullAttentionSpec']
    stock = full(block_size=16, num_kv_heads=2, head_size=128, dtype='bfloat16')
    specs = b.convert_specs({f'layer.{i}': stock for i in range(42)}, full_class=full, compact_class=compact)
    spec = next(iter(specs.values())); assert stock.page_size_bytes == 16384 and spec.page_size_bytes == 12544
    assert type(compact.merge(list(specs.values()))) is compact
    # Execute the actual unchanged DAC1 byte allocator, not a reimplementation.
    runner_raw = git_bytes(a.parent_repo, b.BASE, 'vllm_fl/worker/model_runner.py')
    assert digest(runner_raw) == '42cdf64d8af025ecd2762e586ca592daf3d760a69e204e01ce930729f74f2c6e'
    runner_ast = ast.parse(runner_raw)
    runner_cls = next(n for n in runner_ast.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunnerFL')
    allocate = next(n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name == '_allocate_kv_cache_tensors')
    execute([allocate], scope)
    config = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=2*b.PAGE, block_stride=0, shared_by=[name]) for name in specs],
        kv_cache_groups=[SimpleNamespace(layer_names=list(specs), kv_cache_spec=spec)])
    runner = SimpleNamespace(device='cuda:0', runner_only_attn_layers=set())
    buffers = scope['_allocate_kv_cache_tensors'](runner, config)
    assert len(allocations) == 42 and sum(x._bytes for x in allocations) == 42*2*b.PAGE
    carrier = b.reshape_compact(next(iter(buffers.values())), spec, (2,2,16,2,128),
        (0,1,2,3,4), 2, None, torch=torch, spec_class=compact)
    assert carrier.stride() == (6272,2176,256,128,1) and carrier._bytes == 25088
    # Replay the actual target zeroer metadata constructor. No zero kernel call.
    zero_ast = ast.parse(source['vllm/v1/worker/utils.py'])
    zero_cls = next(n for n in zero_ast.body if isinstance(n, ast.ClassDef) and n.name == 'KVBlockZeroer')
    scope.update(FullAttentionSpec=full, iprod=itertools.product,
                 largest_power_of_2_divisor=lambda n: n & -n)
    execute([zero_cls], scope)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0,
        backend=SimpleNamespace(get_kv_cache_block_dim=lambda *args, **kw: 0), layer_names=list(buffers))
    carriers={name:b.reshape_compact(raw,spec,(2,2,16,2,128),(0,1,2,3,4),2,None,
             torch=torch,spec_class=compact) for name,raw in buffers.items()}
    zero = scope['KVBlockZeroer']('cuda:0', False, [group], [16], 'auto',
                                {name:SimpleNamespace(kv_cache=kv) for name,kv in carriers.items()})
    assert zero._meta == (tuple(x.data_ptr() for x in allocations), 3136, 64, 42)
    assert 3136//64*64*4 == b.PAGE
    manager = b.manager_source(source['vllm/v1/core/single_type_kv_cache_manager.py'])
    scope.update(FullAttentionManager=type('FullAttentionManager', (), {}),
                 TQFullAttentionSpec=type('TQFullAttentionSpec', (), {}),
                 HiddenStateCacheSpec=type('HiddenStateCacheSpec', (), {}))
    execute(ast.parse(manager).body, scope)
    class Pool:
        next_id=10
        def get_new_blocks(self,count):
            result=[SimpleNamespace(block_id=i) for i in range(self.next_id,self.next_id+count)]
            self.next_id+=count
            return result
    pool=Pool()
    mgr = scope['CompactFullAttentionManager']()
    mgr.kv_cache_spec=spec; mgr.block_size=16; mgr.block_pool=pool; mgr.req_to_blocks={'r': []}; mgr.new_block_ids=[]
    added = mgr.allocate_new_blocks('r',32,32); assert len(added)==2 and mgr.new_block_ids==[10,11]
    mgr.get_num_skipped_tokens=lambda _:0
    mgr.allocate_external_computed_blocks('r',32,16); assert mgr.new_block_ids==[10,11,12]
    # Replay the actual registry with one narrowly mocked lazy-import endpoint.
    # No vLLM/framework package is imported; the callback records initialization.
    regmod=ModuleType('_cjj_target_registry_cpu');sys.modules[regmod.__name__]=regmod
    regscope=regmod.__dict__;registrations={};bootstrap=[]
    def register_defaults(config):
        bootstrap.append(config)
        regscope['KVCacheSpecRegistry'].register(full,scope['FullAttentionManager'],full)
    def only_lazy_import(name,*args,**kwargs):
        if name=='__future__':return builtins.__import__(name,*args,**kwargs)
        assert name=='vllm.v1.core.single_type_kv_cache_manager'
        return SimpleNamespace(register_all_kvcache_specs=register_defaults)
    builtin_scope=dict(vars(builtins));builtin_scope['__import__']=only_lazy_import
    regscope.update(__builtins__=builtin_scope,dataclass=dataclass,_REGISTRY_KVCACHESPEC_LIST=registrations)
    regtree=ast.parse(source['vllm/v1/kv_cache_spec_registry.py'])
    execute([n for n in regtree.body if isinstance(n,ast.ClassDef)
             and n.name in ('KVCacheSpecMetadata','KVCacheSpecRegistry')],regscope)
    reg=regscope['KVCacheSpecRegistry'];cfg=SimpleNamespace()
    reg._ensure_registered(cfg)
    reg.register(compact,scope['CompactFullAttentionManager'],compact)
    assert len(bootstrap)==1 and set(registrations)=={full,compact}
    assert reg.get_manager_class(spec) is scope['CompactFullAttentionManager']
    assert reg.get_uniform_type_base_spec(spec) is compact
    reg.check_kv_cache_spec_registry(specs)
    registrations.clear();bootstrap.clear()
    reg.register(compact,scope['CompactFullAttentionManager'],compact)
    reg._ensure_registered(cfg)
    assert not bootstrap and full not in registrations
    parent = json.loads(git_bytes(a.parent_repo,b.PARENT89,'vllm_fl/patches/biv150_konly_encoded_sources.json'))
    host_proofs={};broker=None
    for role in ('views','runtime'):
        raw=git_bytes(a.parent_repo,b.PARENT89,'vllm_fl/patches/biv150_konly_encoded_'+role+'.py')
        generated,proof=b.parent_host_source(raw.decode(),role)
        host_proofs[role]=dict(parent_sha256=digest(raw),compact_sha256=digest(generated.encode()),**proof)
        if role=='views':
            broker=ModuleType('_cjj_actual_compact_broker_cpu');sys.modules[broker.__name__]=broker
            exec(compile(generated,'<pinned-PR89-compact-broker>','exec'),broker.__dict__)
    observation=broker.View('bfloat16',(3,2,16,2,128),(6272,2176,256,128,1),
                            1,4096,3*b.PAGE,0,4096,'cuda:0')
    assert broker.layer_pages(observation)==3
    for role in ('writer83','decode79','prefill80'):
        actual=broker.parameters(observation,role,format_id=b.FORMAT)
        recipes=b.role_parameters(3,role,format_id=b.FORMAT)
        assert set(actual)==set(recipes)
        for key,value in recipes.items():
            if type(value) is b.View:
                seen=actual[key]
                assert (seen.dtype,seen.shape,seen.strides,seen.pointer-4096)==(
                    value.dtype,value.shape,value.strides,value.byte_offset)
            else:assert actual[key]==value
        assert broker.validate_binding(observation,role,actual,format_id=b.FORMAT)
    kernel_proofs = {}; generated_sources={}
    for role in ('writer','decode','normalized2d128','q32','multi2','raw3d'):
        entry = parent['generated'][role]; raw = git_bytes(a.parent_repo,b.PARENT89,entry['path'])
        assert digest(raw)==entry['sha256']
        generated, proof = b.kernel_source(raw.decode(),role)
        generated_sources[role]=generated
        kernel_proofs[role] = dict(parent_sha256=digest(raw), compact_sha256=digest(generated.encode()), **proof)
    cold = git_bytes(a.parent_repo,b.PARENT89,'vllm_fl/patches/biv150_compute_order.py')
    adapted_cold = b.c3_source(cold.decode()); ast.parse(adapted_cold)
    contract_raw = git_bytes(a.parent_repo,b.CONTRACT88,'tools/biv150_konly_compact_page_contract.py')
    contract_module = ModuleType('_cjj_compact88_cpu'); sys.modules[contract_module.__name__]=contract_module
    exec(compile(contract_raw,'<frozen-PR88-contract>','exec'),contract_module.__dict__)
    layout = contract_module.CompactPage(); addresses=0
    def expression(node):return compile(ast.Expression(body=node),'<compact-address-expression>','eval')
    writer_expr={}
    for node in ast.walk(ast.parse(generated_sources['writer'])):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.store':
            text=ast.unparse(node.args[0]);name=text.split(' ',1)[0]
            if name in ('code_full','scale_full','value_full'):writer_expr[name]=expression(node.args[0])
    assert set(writer_expr)=={'code_full','scale_full','value_full'}
    scale_expr=[]
    for node in ast.walk(ast.parse(generated_sources['decode'])):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.load' and ast.unparse(node.args[0]).startswith('key_scale_ptr +'):
            scale_expr.append(node.args[0])
    assert len(scale_expr)==2 and ast.dump(scale_expr[0])==ast.dump(scale_expr[1])
    decode_scale=expression(scale_expr[0])
    prefill=[]
    for role in ('normalized2d128','q32','multi2','raw3d'):
        nodes=[n.value for n in ast.walk(ast.parse(generated_sources[role])) if isinstance(n,ast.Assign)
               and len(n.targets)==1 and ast.unparse(n.targets[0])=='offset']
        nodes=[n for n in nodes if 'pages.to(tl.int64)' in ast.unparse(n)]
        assert len(nodes)==1;prefill.append(nodes[0])
    assert all(ast.dump(x)==ast.dump(prefill[0]) for x in prefill)
    prefill_scale=expression(prefill[0])
    class I(int):
        def to(self,_):return self
    tl=SimpleNamespace(arange=lambda a,z:0,int64='int64')
    for page,slot,head,dim in itertools.product(range(3),range(16),range(2),range(128)):
        for role in ('writer83','decode79','prefill80'):
            p=b.role_parameters(3,role,format_id=b.FORMAT)
            if role=='writer83':
                env=dict(page=page,column=slot,head=head,dim=dim,tl=tl,code_full=0,scale_full=0,value_full=0)
                actual=tuple(eval(writer_expr[k],{},env)*width for k,width in
                             [('code_full',1),('scale_full',4),('value_full',2)])
            else:
                k,v=p['key_cache_ptr'],p['value_cache_ptr']
                scale=(eval(decode_scale,{},dict(key_scale_ptr=0,page=page,column_slot=slot,kv_head_idx=head))
                       if role=='decode79' else eval(prefill_scale,{},dict(pages=I(page),slots=I(slot),head=I(head),tl=tl)))
                actual=(k.byte_offset+sum(i*s for i,s in zip((page,slot,head,dim),k.strides)),
                        scale*4,v.byte_offset+2*sum(i*s for i,s in zip((page,slot,head,dim),v.strides)))
            expected=tuple(layout.address(k,page,slot,head,0 if k=='scale' else dim) for k in ('code','scale','value'))
            assert actual==expected; addresses+=3
    rejected=0
    assert b.views(2**31-1)['key'].strides[0]*(2**31-2)>2**32
    tests=[lambda:b.views(True),lambda:b.views(0),lambda:b.views(2**31),lambda:b.role_parameters(1,'stock',format_id=b.FORMAT),
        lambda:b.role_parameters(1,'decode79',format_id='cjj-konly-rn32-rne-v2'),
        lambda:b.convert_specs({'only':stock},full_class=full,compact_class=compact),
        lambda:compact.merge([spec,stock]),lambda:spec.copy_with_new_block_size(32),
        lambda:b.reshape_compact(Tensor((32768,)),spec,(2,2,16,2,128),(0,1,2,3,4),2,None,torch=torch,spec_class=compact),
        lambda:b.kernel_source('x = page * 4095','decode')]
    for call in tests:
        try:call()
        except (ValueError,AssertionError):rejected+=1
        else:raise AssertionError('Rejected-case unexpectedly accepted')
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(base=b.BASE,contract88=b.CONTRACT88,parent89=b.PARENT89,format=b.FORMAT,
        cpu_only=True,source_hashes=PINS,source_provenance='Existing local target-source archives, no fresh machine access',
        target_registry_source_replayed=True,builtin_then_private_registration_verified=True,
        private_first_skips_builtin_mutant_reproduced=True,
        runtime_installed_hashes_require_future_check=True,actual_DAC_allocator_method_replayed=True,
        mock_raw_allocations=42,mock_total_requested_bytes=42*2*b.PAGE,physical_page_bytes=b.PAGE,
        carrier_shape=[2,2,16,2,128],carrier_strides=list(carrier.stride()),
        nominal_BF16_K_is_not_data=True,target_zeroer_metadata_replayed=True,
        zeroer_page_i32=3136,zeroer_tile_i32=64,zeroer_tiles_per_page=49,
        zero_notification_guard_inverse=True,manager_new_block_ids_verified=True,
        compact_C3_guard_AST_inverse=True,kernel_source_proofs=kernel_proofs,
        parent_host_source_proofs=host_proofs,actual_parent_typed_broker_replayed=True,
        address_comparisons=addresses,rejected_cases=rejected,
        installed=False,activation_ready=False,RPC_module_export_verified=False,
        actual_tensor_views_verified=False,actual_IR_verified=False,device_numeric_verified=False,
        capacity_reclaimed=False,performance_verified=False,precision_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vllm-root',type=Path,help='Optional complete source root; default uses eight existing target archives')
    p.add_argument('--parent-repo',type=Path,default=Path('/private/tmp/flagos-cjj-konly-encoded-service-20261027'))
    print(json.dumps(run(p.parse_args()),indent=2)+'\n',end='')
