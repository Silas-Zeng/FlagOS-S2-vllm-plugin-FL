"""CPU metadata execution of the actual compact installer and UniProc spec path.

Target method bodies are extracted from hash-pinned originals. Framework modules,
Tensor storage and JIT objects are explicit mocks; no device work is performed.
"""
import argparse
import ast
import builtins
import collections
import contextlib
import copy
from dataclasses import dataclass, fields, field, replace
from enum import IntEnum
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as S
from typing import Any, cast

from check_biv150_konly_compact_bridge_cpu import ARCHIVED

ROOT=Path(__file__).resolve().parents[1]
ENV='VLLM_FL_BIV150_KONLY_COMPACT_SERVICE'
PREFIX='vllm_fl.patches.biv150_konly_compact_'


def execute(nodes, namespace, filename='<actual-target-method-CPU>'):
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),filename,'exec'),namespace)


def functions(path,names,namespace):
    tree=ast.parse(Path(path).read_bytes())
    execute([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],namespace,str(path))


class Storage:
    count=0
    def __init__(self,size):
        Storage.count+=1;self._cdata=Storage.count;self.size=size;self.address=0x100000+Storage.count*0x100000
    def data_ptr(self):return self.address
    def nbytes(self):return self.size


class Tensor:
    widths={'bfloat16':2,'float32':4,'int8':1,'int64':8,'int32':4,'uint64':8}
    def __init__(self,dtype,shape,strides=None,storage=None,offset=0):
        self.dtype=DTYPES[str(dtype).removeprefix('torch.')];self.shape=tuple(shape)
        self.ndim=len(shape);self.offset=offset;self.device='cuda:0'
        if strides is None:
            strides=[];s=1
            for n in reversed(shape):strides.insert(0,s);s*=n
        self.strides=tuple(strides);self.storage=storage or Storage(self.numel()*self.element_size())
    def numel(self):return math.prod(self.shape)
    def element_size(self):return self.widths[str(self.dtype).removeprefix('torch.')]
    def untyped_storage(self):return self.storage
    def data_ptr(self):return self.storage.address+self.offset*self.element_size()
    def storage_offset(self):return self.offset
    def stride(self,dim=None):return self.strides if dim is None else self.strides[dim]
    def is_contiguous(self):return self.strides==Tensor(self.dtype,self.shape).strides
    def view(self,dtype):return Tensor(dtype,(self.storage.size//Tensor.widths[str(dtype).removeprefix('torch.')],),storage=self.storage)
    def set_(self,storage,offset,shape,strides):
        self.storage,self.offset,self.shape,self.strides=storage,offset,tuple(shape),tuple(strides);self.ndim=len(shape);return self
    def cpu(self):raise AssertionError('device readback')
    def item(self):raise AssertionError('device scalar')


DTYPES={name:'torch.'+name for name in Tensor.widths}


class Kernel:
    def __init__(self):self.device_caches={};self.pre_run_hooks=[];self.calls=[]
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):self.calls.append((grid,args,kwargs));return S(hash='CPU-metadata-mock')
        return invoke


def module(name,path=None,package=False):
    value=ModuleType(name);value._CPU_mock=True
    if path:value.__file__=str(path)
    if package:value.__path__=[str(path)] if path else []
    sys.modules[name]=value
    return value


def foundation():
    module('vllm_fl',ROOT/'vllm_fl',True);module('vllm_fl.patches',ROOT/'vllm_fl/patches',True)
    for name in ('vllm','vllm.v1','vllm.v1.core','vllm.v1.worker','vllm.v1.worker.gpu',
                 'vllm.v1.executor','vllm.v1.engine','vllm.config','vllm.utils','vllm.v1.attention',
                 'vllm.v1.attention.backends','vllm.v1.attention.ops'):
        module(name,package=True)
    torch=module('torch');torch.__dict__.update(DTYPES);torch.Tensor=Tensor;torch.cuda=S(is_current_stream_capturing=lambda:False)
    torch.empty=lambda n,*,dtype,**kw:Tensor(dtype,(n,))
    allocations=[]
    def zeros(n,*,dtype,device):
        value=Tensor(dtype,(n,));allocations.append(value);return value
    torch.zeros=zeros
    torch.as_strided=lambda value,*,size,stride:Tensor(value.dtype,size,stride,value.storage,value.offset)
    kv=module('vllm.v1.kv_cache_interface',ARCHIVED['vllm/v1/kv_cache_interface.py'])
    kv.__dict__.update(dataclass=dataclass,field=field,fields=fields,replace=replace,copy=copy,torch=torch,IntEnum=IntEnum,
        cdiv=lambda a,b:(a+b-1)//b,get_dtype_size=lambda dt:Tensor.widths[str(dt).removeprefix('torch.')],
        MLAAttentionSpec=type('MLAAttentionSpec',(),{}),MambaSpec=type('MambaSpec',(),{}),Counter=collections.Counter)
    names=('KVQuantMode','KVCacheSpec','AttentionSpec','FullAttentionSpec','UniformTypeKVCacheSpecs',
           'KVCacheTensor','KVCacheGroupSpec','KVCacheConfig')
    execute([n for n in ast.parse(Path(kv.__file__).read_bytes()).body if isinstance(n,ast.ClassDef) and n.name in names],vars(kv))
    for name in ('TQFullAttentionSpec','HiddenStateCacheSpec'):setattr(kv,name,type(name,(),{}))
    manager=module('vllm.v1.core.single_type_kv_cache_manager',ARCHIVED['vllm/v1/core/single_type_kv_cache_manager.py'])
    manager.FullAttentionManager=type('FullAttentionManager',(),{})
    reg=module('vllm.v1.kv_cache_spec_registry',ARCHIVED['vllm/v1/kv_cache_spec_registry.py'])
    reg.__dict__.update(dataclass=dataclass,_REGISTRY_KVCACHESPEC_LIST={})
    execute([n for n in ast.parse(Path(reg.__file__).read_bytes()).body if isinstance(n,ast.ClassDef)
             and n.name in ('KVCacheSpecMetadata','KVCacheSpecRegistry')],vars(reg))
    def register_all(config):reg.KVCacheSpecRegistry.register(kv.FullAttentionSpec,manager.FullAttentionManager,kv.FullAttentionSpec)
    manager.register_all_kvcache_specs=register_all
    maths=module('vllm.utils.math_utils');maths.cdiv=lambda a,b:(a+b-1)//b
    utils=module('vllm.v1.worker.utils',ARCHIVED['vllm/v1/worker/utils.py'])
    utils.__dict__.update(defaultdict=collections.defaultdict,extract_layer_index=lambda name,n:int(name.split('.')[2]))
    functions(utils.__file__,('bind_kv_cache',),vars(utils))
    attn=module('vllm.v1.worker.gpu.attn_utils',ARCHIVED['vllm/v1/worker/gpu/attn_utils.py'])
    functions(attn.__file__,('_reshape_attention_kv_cache',),vars(attn))
    for path in ('vllm/v1/core/kv_cache_utils.py','vllm/v1/worker/kv_connector_model_runner_mixin.py'):
        name=path[:-3].replace('/','.');mod=module(name,ARCHIVED[path])
        if name.endswith('kv_cache_utils'):
            mod.__dict__.update(vars(kv));mod.__name__=name;mod.__file__=ARCHIVED[path]
            functions(mod.__file__,('create_kv_cache_group_specs','get_uniform_page_size','get_num_blocks',
                'may_override_num_blocks','_use_packed_kv_cache_config','get_kv_cache_config_from_groups'),vars(mod))
    for group,folder,names in [
        ('uni','/private/tmp/cjj-target-uniproc-spec-sources-20261028',
         ('vllm/v1/executor/uniproc_executor.py','vllm/v1/executor/abstract.py','vllm/v1/engine/core.py',
          'vllm/v1/worker/gpu_worker.py','vllm/v1/serial_utils.py')),
        ('config','/private/tmp/cjj-target-compact-config-sources-20261028',
         ('vllm/config/cache.py','vllm/config/offload.py','vllm/v1/worker/worker_base.py'))]:
        for path in names:module(path[:-3].replace('/','.'),Path(folder)/path)
    return torch,kv,reg,allocations


def setup_install(torch,kv):
    service=importlib.import_module(PREFIX+'service')
    parent_modules={}
    for name,pin in service.PINS.items():
        if name.startswith('vllm_fl.'):
            path=ROOT/(name.replace('.','/')+'.py')
        else:
            suffix=name.replace('.','/')+'.py'
            path=Path(ARCHIVED.get(suffix,'/private/tmp/flagos-sdk-triton-unified-attention-54c495-20261027.py'))
            if name.endswith('triton_attention_helpers'):path=Path('/private/tmp/flagos-sdk-triton-attention-helpers-59d9d3-20261027.py')
        mod=module(name,path);parent_modules[name]=mod
    backend=parent_modules['vllm.v1.attention.backends.triton_attn']
    attention=parent_modules['vllm.v1.attention.ops.triton_unified_attention']
    # Locate exact SDK snapshots by their required hash if older aliases moved.
    for name,pin in service.PINS.items():
        mod=parent_modules[name]
        if not Path(mod.__file__).exists() or hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()!=pin:
            candidates=list(Path('/private/tmp').glob('*helpers*59d9*'))+list(Path('/private/tmp').glob('*unified*54c495*'))
            match=[p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==pin]
            assert len(match)==1,(name,pin);mod.__file__=str(match[0])
    attention.current_platform=S(vendor_name='iluvatar',is_cuda=lambda:False);attention.is_batch_invariant=False
    attention.unified_attention=lambda **kw:None;backend.unified_attention=attention.unified_attention
    class Impl:
        def do_kv_cache_update(self,*args):raise AssertionError('stock writer')
        def fused_rope_kvcache_supported(self):return True
        def do_rope_and_kv_cache_update(self,*args):raise AssertionError('fused writer')
    backend.TritonAttentionImpl=Impl
    class Backend:
        @staticmethod
        def get_kv_cache_shape(n,p,h,d,**kw):return(n,2,p,h,d)
        @staticmethod
        def get_kv_cache_stride_order():return(0,1,2,3,4)
        @staticmethod
        def indexes_kv_by_block_stride():return True
    backend.TritonAttentionBackend=Backend
    gqa=parent_modules['vllm_fl.patches.biv150_gqa_decode'];gqa._state={'calls':0,'fallbacks':0};gqa._gqa_decode_attention=Kernel()
    native=parent_modules['vllm_fl.patches.biv150_native_gqa']
    class Native:
        def __init__(self):self.calls=[]
        def try_produce(self,grid,kwargs):
            # Poison eligibility: accepts every dtype/stride. Owned storage must
            # never arrive here, even when compact strides cease to reject it.
            self.calls.append((grid,kwargs));return True
    native._Runtime=Native;gqa._native_s2=Native()
    functions(gqa.__file__,('_launch_producer',),vars(gqa))

    fragment=parent_modules['vllm_fl.patches.biv150_q_fragment_service']
    base=S(active=True,holder={},kernel_key='kernel_unified_attention',wrapper=Kernel(),raw=Kernel());fragment._MANAGER=base
    q32=parent_modules['vllm_fl.patches.biv150_prefill_q32'];m2=parent_modules['vllm_fl.patches.biv150_prefill_multi2']
    class QPolicy(Kernel):pass
    class MPolicy(Kernel):pass
    q32._Policy=QPolicy;q32._MANAGER=S(active=True)
    q=QPolicy();q.manager=q32._MANAGER;q.base=base.wrapper
    m2._Policy=MPolicy;m2._MANAGER=S(active=True)
    outer=MPolicy();outer.manager=m2._MANAGER;outer.base=q;m2._POLICY=outer;base.holder[base.kernel_key]=outer
    attention.kernel_unified_attention=Kernel()
    custom=module('vllm._custom_ops');custom.swap_blocks=lambda *args:None;custom.copy_blocks=lambda *args:None
    platform=module('vllm_fl.platform',ROOT/'vllm_fl/platform.py')
    class Platform:copy_kv_blocks=staticmethod(lambda *args:None)
    platform.PlatformFL=Platform
    triton=module('triton');triton.runtime=S(driver=S(active=S(get_current_target=lambda:S(backend='corex',arch=71,warp_size=64))))
    generated=json.loads((ROOT/'vllm_fl/patches/biv150_konly_compact_sources.json').read_text())['generated']
    kernels={}
    for role in ('writer','decode','normalized2d128','q32','multi2','raw3d'):
        path=ROOT/generated[role]['path'];mod=module('vllm_fl.patches.'+path.stem,path);kernels[role]=Kernel()
        name='encode_append' if role=='writer' else '_gqa_decode_konly_i8' if role=='decode' else 'kernel_cjj_konly_'+role
        setattr(mod,name,kernels[role])
    model=S(num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2,hidden_size=2048,model_type='minicpm')
    config=S(model_config=S(dtype=torch.bfloat16,is_encoder_decoder=False,hf_config=model),
        parallel_config=S(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,
            decode_context_parallel_size=1,prefill_context_parallel_size=1,distributed_executor_backend='uni'),
        cache_config=S(cache_dtype='auto',block_size=16,kv_offloading_size=None,num_gpu_blocks_override=None),
        offload_config=S(uva=S(cpu_offload_gb=0),prefetch=S(offload_group_size=0,offload_num_in_group=1)),
        speculative_config=None,kv_transfer_config=None,ec_transfer_config=None,kv_offload_config=None)
    class Attention:
        kv_sharing_target_layer_name=None
        def get_kv_cache_spec(self,config):return kv.FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.bfloat16)
        def get_attn_backend(self):return Backend
    layers={f'model.layers.{i}.self_attn.attn':Attention() for i in range(42)}
    runner_mod=module('vllm_fl.worker.model_runner',ROOT/'vllm_fl/worker/model_runner.py')
    runner_mod.__dict__.update(torch=torch,Attention=Attention,AttentionLayerBase=Attention,AttentionSpec=kv.AttentionSpec,
        MambaSpec=type('MambaSpec',(),{}),cast=cast,Any=Any,replace=replace,
        has_ec_transfer=lambda:False,get_layers_from_vllm_config=lambda *args:layers,
        set_current_vllm_config=lambda cfg:contextlib.nullcontext(),bind_kv_cache=sys.modules['vllm.v1.worker.utils'].bind_kv_cache,
        _reshape_attention_kv_cache=sys.modules['vllm.v1.worker.gpu.attn_utils']._reshape_attention_kv_cache)
    tree=ast.parse(Path(runner_mod.__file__).read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunnerFL')
    names={'get_kv_cache_spec','_allocate_kv_cache_tensors','_reshape_kv_cache_tensors',
           'initialize_kv_cache_tensors','_kv_cache_spec_attn_group_iterator'}
    body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    child=ast.ClassDef(name='ModelRunnerFL',bases=[],keywords=[],body=body,decorator_list=[])
    execute([child],vars(runner_mod),runner_mod.__file__);Runner=runner_mod.ModelRunnerFL
    def initialize(self,kv_cache_config):
        self.kv_cache_config=kv_cache_config;self._kernel_block_sizes=[16]
        self.attn_groups=[[S(kv_cache_spec=kv_cache_config.kv_cache_groups[0].kv_cache_spec,
            backend=Backend,kv_cache_group_id=0,layer_names=list(layers))]]
        return self.initialize_kv_cache_tensors(kv_cache_config,[16])
    Runner.initialize_kv_cache=initialize;Runner.shutdown=lambda self:None;Runner.use_uniform_kv_cache=lambda *args:False
    runner=Runner();runner.vllm_config=config;runner.model_config=config.model_config;runner.cache_config=config.cache_config
    runner.shared_kv_cache_layers={};runner.cross_layers_kv_cache=None;runner.runner_only_attn_layers=set();runner.device='cuda:0'
    runner.compilation_config=S(static_forward_context=layers);runner.kv_caches=[]
    return service,Runner,runner,kernels,backend,attention,base


def reject(call):
    try:call()
    except (RuntimeError,ValueError,AssertionError):return
    raise AssertionError('invalid metadata admitted')


def run(export_only=False):
    torch,kv,registry,allocations=foundation()
    service=importlib.import_module(PREFIX+'service')
    if export_only:
        spec=importlib.import_module(PREFIX+'spec');manager=spec.ensure_registered(S())
        assert spec.CompactFullAttentionSpec.__module__==PREFIX+'spec' and manager.__module__==PREFIX+'manager'
        return dict(module_export_smoke=True,actual_RPC_test=False,frameworks_mocked=True)
    calls=[];old=service.importlib.import_module
    service.importlib.import_module=lambda name:calls.append(name) or (_ for _ in ()).throw(AssertionError(name))
    try:assert service.install(object,environ={})==dict(installed=False,reason='disabled') and not calls
    finally:service.importlib.import_module=old
    worker_tree=ast.parse((ROOT/'vllm_fl/worker/model_runner.py').read_bytes())
    tails=[node for node in worker_tree.body if isinstance(node,ast.If) and ENV in ast.unparse(node.test)]
    assert len(tails)==2
    for flags,vendor,expected in [({},'iluvatar',0),({ENV:'1'},'iluvatar',1),
        ({ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1'},'iluvatar',-1),
        ({ENV:'1','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'},'iluvatar',1),({ENV:'1'},'other',-1)]:
        received=[];imports=[]
        def importer(name,*args,**kw):
            imports.append(name)
            if name=='os':return S(environ=flags)
            assert name==PREFIX+'service'
            return S(install=lambda cls:received.append(cls) or dict(CPU_mock=True))
        ns=dict(current_platform=S(vendor_name=vendor),ModelRunnerFL=object,
                __builtins__={**vars(builtins),'__import__':importer,'print':lambda *a,**kw:None})
        def run_tail():exec(compile(ast.Module(body=tails,type_ignores=[]),'<actual-worker-tail>','exec'),ns)
        if expected<0:reject(run_tail)
        else:run_tail();assert len(received)==expected
        if expected<=0:assert not received and all(name=='os' for name in imports)
    service,Runner,runner,kernels,backend,attention,base=setup_install(torch,kv)
    reject(lambda:service.install(Runner,environ={ENV:'1','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'}))
    reject(lambda:service.install(Runner,environ={ENV:'1','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1'}))
    parent_get=Runner.get_kv_cache_spec;parent_alloc=Runner._allocate_kv_cache_tensors
    installed=service.install(Runner,environ={ENV:'1'});assert installed['installed']
    host=service._STATE.compact_host
    gqa=sys.modules['vllm_fl.patches.biv150_gqa_decode']
    native=gqa._native_s2;installed_producer=gqa._launch_producer
    ordinary=Tensor(torch.bfloat16,(2,16,2,128),(8192,256,128,1))
    assert gqa._launch_producer((1,),key_cache_ptr=ordinary,value_cache_ptr=ordinary) is None
    assert len(native.calls)==1 and gqa._native_s2 is native

    for obj,key,bad in [(runner.vllm_config,'speculative_config',S()),
        (runner.vllm_config,'kv_transfer_config',S()),
        (runner.cache_config,'kv_offloading_size',0),
        (runner.vllm_config.offload_config.uva,'cpu_offload_gb',1),
        (runner.vllm_config.offload_config.prefetch,'offload_group_size',1),
        (runner.vllm_config.parallel_config,'distributed_executor_backend','mp')]:
        old=getattr(obj,key);setattr(obj,key,bad);reject(lambda:host.scope(runner));setattr(obj,key,old)
    spec_dict=runner.get_kv_cache_spec();assert len(spec_dict)==42 and Runner._allocate_kv_cache_tensors is parent_alloc
    spec_module=importlib.import_module(PREFIX+'spec');Spec=spec_module.CompactFullAttentionSpec
    assert all(type(spec) is Spec for spec in spec_dict.values())
    # Actual UniProc/get-spec/run_method bodies preserve the same dict/spec objects.
    serial=sys.modules['vllm.v1.serial_utils'];functions(serial.__file__,('run_method',),vars(serial))
    uni=sys.modules['vllm.v1.executor.uniproc_executor'];abstract=sys.modules['vllm.v1.executor.abstract']
    for mod,clsname,fnname in ((uni,'UniProcExecutor','collective_rpc'),(abstract,'Executor','get_kv_cache_specs')):
        node=next(n for n in ast.parse(Path(mod.__file__).read_bytes()).body if isinstance(n,ast.ClassDef) and n.name==clsname)
        execute([n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==fnname],vars(mod))
    uni.__dict__.update(run_method=serial.run_method,AsyncModelRunnerOutput=type('AsyncModelRunnerOutput',(),{}))
    worker_mod=sys.modules['vllm.v1.worker.gpu_worker'];proxy_mod=sys.modules['vllm.v1.worker.worker_base']
    for mod,fnname in ((worker_mod,'get_kv_cache_spec'),(proxy_mod,'__getattr__')):
        matches=[n for cls in ast.parse(Path(mod.__file__).read_bytes()).body if isinstance(cls,ast.ClassDef)
                 for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==fnname]
        assert len(matches)==1;execute(matches,vars(mod))
    Worker=type('CPUActualWorkerMethod',(),{'get_kv_cache_spec':worker_mod.get_kv_cache_spec})
    worker=Worker();worker.model_runner=S(get_kv_cache_spec=lambda:spec_dict)
    Proxy=type('CPUActualWorkerProxy',(),{'__getattr__':proxy_mod.__getattr__});proxy=Proxy();proxy.worker=worker
    executor=S(driver_worker=proxy);executor.collective_rpc=lambda *a,**kw:uni.collective_rpc(executor,*a,**kw)
    result=abstract.get_kv_cache_specs(executor);assert result[0] is spec_dict and result[0][next(iter(spec_dict))] is next(iter(spec_dict.values()))
    utils=sys.modules['vllm.v1.core.kv_cache_utils']
    groups=utils.create_kv_cache_group_specs(spec_dict,[list(spec_dict)])
    config=utils.get_kv_cache_config_from_groups(runner.vllm_config,groups,42*2*12544)
    assert config.num_blocks==2 and all(t.size==25088 for t in config.kv_cache_tensors)
    assert not config.needs_kv_cache_zeroing
    runner.initialize_kv_cache(config)
    assert len(allocations)==42 and sum(t.untyped_storage().nbytes() for t in allocations)==1053696
    assert host.receipt(groups[0].kv_cache_spec)['compact_host_registered']
    assert host.stats()['raw_storage_bytes']==1053696 and host.stats()['recorded_layers']==42
    reject(lambda:runner.initialize_kv_cache(config));reject(lambda:runner.get_kv_cache_spec())
    first=next(iter(runner.compilation_config.static_forward_context.values())).kv_cache
    source=Tensor(torch.bfloat16,(1,2,128));slots=Tensor(torch.int64,(1,))
    impl=backend.TritonAttentionImpl();impl.num_heads=16;impl.num_kv_heads=2;impl.head_size=128
    impl._kv_quant_mode=0;impl._is_per_token_head_quant=False;impl.kv_sharing_target_layer_name=None
    impl.do_kv_cache_update(None,source,source,first,slots)
    assert kernels['writer'].calls and service._STATE.registry.entries
    entry=next(iter(service._STATE.registry.entries.values()))
    assert entry.roles['decode79']['key_cache_ptr'].stride(0)==12544
    assert entry.roles['decode79']['value_cache_ptr'].data_ptr()==first.data_ptr()+4352
    # Actual private facade, actual ownership lookup, explicit Tensor/JIT mocks.
    # Cache K/V have nominal BF16 carriers but byte-packed private K payloads.
    key=Tensor(torch.bfloat16,(2,16,2,128),(6272,256,128,1),first.storage)
    value=Tensor(torch.bfloat16,(2,16,2,128),(6272,256,128,1),first.storage,2176)
    query=Tensor(torch.bfloat16,(1,16,128))
    kw=dict(key_cache_ptr=key,value_cache_ptr=value,query_ptr=query,
            HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,queries_per_kv=8)
    gqa._launch_producer((1,2,1),**kw)
    assert len(kernels['decode'].calls)==1 and len(native.calls)==1
    changed=dict(kw,key_cache_ptr=Tensor(torch.bfloat16,key.shape,(8192,256,128,1),first.storage))
    reject(lambda:gqa._launch_producer((1,2,1),**changed))
    changed=dict(kw,value_cache_ptr=Tensor(torch.bfloat16,value.shape,value.strides,first.storage,2177))
    reject(lambda:gqa._launch_producer((1,2,1),**changed))
    reject(lambda:gqa._launch_producer((1,),key_cache_ptr=ordinary,value_cache_ptr=ordinary))
    assert len(native.calls)==1 and len(kernels['decode'].calls)==1

    receipt=service.compute_order_receipt(attention,base.holder[base.kernel_key]);assert receipt['parent_fn_is_interface_only']
    assert receipt['producer_boundary_registered'] and receipt['stock_native_parent_preserved']
    gqa._native_s2=object();reject(lambda:service.compute_order_receipt(attention,base.holder[base.kernel_key]));gqa._native_s2=native
    gqa._launch_producer=installed_producer.__wrapped__
    reject(lambda:service.compute_order_receipt(attention,base.holder[base.kernel_key]));gqa._launch_producer=installed_producer

    old_hook=Runner.get_kv_cache_spec;Runner.get_kv_cache_spec=parent_get
    reject(lambda:service.compact_spec_is_owned(groups[0].kv_cache_spec));Runner.get_kv_cache_spec=old_hook
    fakeold=kv.FullAttentionSpec(block_size=16,num_kv_heads=2,head_size=128,dtype=torch.bfloat16)
    reject(lambda:service.compact_spec_is_owned(fakeold))
    old_buffer=Tensor(torch.bfloat16,(2,2,16,2,128),(8192,4096,256,128,1))
    before=len(kernels['writer'].calls)
    reject(lambda:impl.do_kv_cache_update(None,source,source,old_buffer,slots))
    assert len(kernels['writer'].calls)==before
    identity=importlib.import_module(PREFIX+'identity')
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='cjj-compact-wrong-source-') as folder:
        path=Path(folder)/'wrong.py';path.write_text('wrong source\n')
        mod=sys.modules['vllm.v1.kv_cache_spec_registry'];old=mod.__file__;mod.__file__=str(path)
        reject(lambda:identity.verify_module('vllm.v1.kv_cache_spec_registry',identity.CORE_PINS['vllm.v1.kv_cache_spec_registry']))
        mod.__file__=old
    c3={};functions(ROOT/'vllm_fl/patches/biv150_compute_order.py',('_compact_or_stock_spec',),c3)
    c3.update(os=os,sys=sys)
    assert c3['_compact_or_stock_spec'](groups[0].kv_cache_spec,kv)
    os.environ[ENV]='0';assert c3['_compact_or_stock_spec'](fakeold,kv);os.environ[ENV]='1'
    sem=ModuleType('_compact_semantics_cpu');sem.__file__=str(ROOT/'vllm_fl/compile_semantics.py')
    exec(compile(Path(sem.__file__).read_bytes(),sem.__file__,'exec'),vars(sem))
    runner.vllm_config.model_config.hf_config=S(**sem._GEOMETRY)
    common=dict(vendor_name='iluvatar',abi={'CPU_only':True})
    with TemporaryDirectory(prefix='cjj-compact-native-fingerprint-') as folder:
        library=Path(folder)/'CPU-metadata-not-executable.so';library.write_bytes(b'CPU-only-native-content-1')
        flags={'VLLM_FL_BIV150_NATIVE_W8A8':'0','BIV150_NATIVE_GQA_LIBRARY':str(library)}
        off=sem.build_fingerprint(runner.vllm_config,environ=flags,**common)
        on=sem.build_fingerprint(runner.vllm_config,environ={**flags,ENV:'1'},**common)
        library.write_bytes(b'CPU-only-native-content-2')
        changed=sem.build_fingerprint(runner.vllm_config,environ={**flags,ENV:'1'},**common)
        assert changed['sha256']!=on['sha256']

    assert off['sha256']!=on['sha256'] and on['konly_compact']['page_bytes']==12544
    assert on['konly_compact']['implementation_sha256']==service.implementation_digest()
    child=subprocess.check_output([sys.executable,'-B',__file__,'--export-only'],env=dict(os.environ,**{ENV:'1'}))
    assert json.loads(child)['module_export_smoke']
    import atexit
    atexit.unregister(service._STATE.finalize)
    return dict(cpu_passed=True,actual_installer_metadata_path=True,frameworks_mocked=True,
        default_off_no_import_or_bind=True,actual_worker_tail_switch_verified=True,
        uniproc_get_spec_object_identity_preserved=True,
        spec_transport='UniProc direct string run_method; no serialization',RPC_serializer_test=False,
        independent_child_module_export=True,builtin_then_private_registration=True,
        actual_config_allocator_reshape_bind_methods_replayed=True,raw_buffers=42,raw_bytes=1053696,
        page_bytes=12544,compact_receipt_and_parent_hooks_verified=True,allocation_metadata_export_checked=True,
        wrong_format_missing_hoist_cache_reinit_late_getspec_rejected=True,no_spec_connector_offload_scope_verified=True,
        CUDA_temporary_profile_cache_scope_excluded=True,
        wrong_installed_source_rejected=True,pure_full_attention_zeroer_not_activated=True,
        compile_fingerprint_bound=True,native_library_content_fingerprint_retained=True,
        owned_compact_before_native_verified=True,unknown_or_alias_cannot_reach_native=True,
        cold_unowned_main_native_preserved=True,native_receipt_mutants_rejected=True,
        actual_Tensor_verified=False,JIT_executed=False,GPU_executed=False,actual_IR_verified=False,
        numeric_verified=False,model_verified=False,capacity_reclaimed=False,performance_verified=False,precision_gate=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export-only',action='store_true');args=parser.parse_args()
    os.environ[ENV]='1'
    import io
    with contextlib.redirect_stdout(io.StringIO()):result=run(args.export_only)
    print(json.dumps(result,indent=2))
