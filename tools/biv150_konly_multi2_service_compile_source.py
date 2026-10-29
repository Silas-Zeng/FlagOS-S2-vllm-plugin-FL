"""Compile the actual production multi2 fn through its new Adapter.prepare only."""
import ast
import base64
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
BASE='d5d9c56f51d731e03b78a7087ffdb1a64de93c11'
FROZEN='ce905346a0bd6938cd8478cde34c1bcc52eaeed7'
FROZEN_SOURCE_SHA='a474f21f67e8ca74a9110b8e27d918677c39c9cc76ef7a5c85c717458b36917e'
SOURCES={'vllm_fl/patches/biv150_konly_encoded_multi2.py': '7046cf538bf06278982041d04b129c65c74bdb65531e528366fc62626c6adb19', 'vllm_fl/patches/biv150_konly_encoded_multi2_adapter.py': '57f8d5925efc5e2aa8c8c56a9f22ede85cc32c2edb817a08c466c71799918070', 'vllm_fl/patches/biv150_konly_encoded_q32_adapter.py': '058b791435c196e02fe34cf4b1cd374496358493ee077e9603e3807b648564cd', 'vllm_fl/patches/biv150_konly_encoded_hoist_proposal.py': '1a3f165590dcb9ee5c012d242691477bc9739fbc6867b162c27b05b77c911688', 'vllm_fl/patches/biv150_konly_encoded_hoist_native.py': '2027e1ce4a489f6b47c97a5253c1482d2dc767e62e9b10ec029e3f192c559ef2', 'vllm_fl/patches/biv150_konly_encoded_service.py': 'dc76638af2208724c5189183860e7e66c7610564320ddca2c19f66d3bda37ece', 'vllm_fl/patches/biv150_konly_encoded_runtime.py': 'b7394ef3b04dcc6329180328e273254420eb596959a5751ac9c334157d28a64c', 'vllm_fl/patches/biv150_konly_encoded_sources.json': '5b0101e78e0a0c77e48567b479d1bc211531b9227aaad320ce29a1dcf68f66fa', 'vllm_fl/compile_semantics.py': 'c90a9d44a92d9d78fb8cbd9148a9039d81ee57887da5f5d783c5a61d11d39893'}
NAME='kernel_cjj_konly_multi2'
FUNCTION_SHA='39ebc5ce6ee4546f45932026f935c670e8c568aa94c6a3bc26e94b9ddab50e18'

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(value):return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()

def frozen():
    body=subprocess.check_output(['git','-C',str(ROOT),'show',FROZEN+':tools/biv150_konly_multi2_hoist_source.py'])
    need(sha(body)==FROZEN_SOURCE_SHA,'frozen ce905 helper source')
    result=types.ModuleType('_cjj_service_compile_frozen');result.__file__=str(ROOT/'tools/frozen-ce905.py')
    exec(compile(body,'ce905-frozen-compile-helper','exec'),result.__dict__)
    return result

def kwargs(p,t,candidate):return frozen().kwargs(p,t,candidate)

def proposal(bq):
    need(type(bq) is int and bq==32,'only original BQ32')
    return frozen().module_from_source('_cjj_service_compile_pins',(ROOT/'vllm_fl/patches/biv150_konly_encoded_hoist_proposal.py').read_text())

def load_production(folder):
    result={}
    for key in ('multi2','multi2_adapter','q32_adapter','hoist_proposal','hoist_native','service'):
        name='vllm_fl.patches.biv150_konly_encoded_'+key
        mod=importlib.import_module(name);expected=ROOT/'vllm_fl/patches'/('biv150_konly_encoded_'+key+'.py')
        need(Path(mod.__file__).resolve()==expected.resolve(),'actual production import '+key)
        need(sha(expected.read_bytes())==SOURCES[str(expected.relative_to(ROOT))],'actual source '+key)
        result[key]=mod
    for path,digest in SOURCES.items():
        body=(ROOT/path).read_bytes();need(sha(body)==digest,'production source '+path)
        (folder/Path(path).name).write_bytes(body)
    return result

def cache_identity(cache,record):
    # Same seven-file format/base32 key as the ce905 reviewed-cache loader; readonly.
    directory=cache/base64.b32encode(bytes.fromhex(record['hash'])).decode().rstrip('=')
    name=record['function_name'];index=directory/('__grp__'+name+'.json')
    need(index.is_file() and not index.is_symlink(),'actual private cache index')
    paths=json.loads(index.read_text())['child_paths']
    need(set(paths)=={name+'.'+ext for ext in ('source','ttir','ttgir','llir','asm','cubin','json')},'complete seven-file cache group')
    children={}
    for filename,path in paths.items():
        p=Path(path);need(p==directory/filename and p.is_file() and not p.is_symlink(),'actual private cache child')
        body=p.read_bytes();ext=filename.rsplit('.',1)[1]
        if ext in record['artifacts']:need(sha(body)==record['artifacts'][ext],'cache/final handle artifact '+ext)
        if ext=='json':need(json.loads(body)['hash']==record['hash'],'cache metadata hash')
        children[filename]=dict(path=str(p),bytes=len(body),sha256=sha(body))
    return dict(hash=record['hash'],name=name,index=str(index),index_sha256=sha(index.read_bytes()),children=children)

BODY = r"""        folder = args.out/'sources'; folder.mkdir()
        modules = c.load_production(folder)
        fn = getattr(modules['multi2'], c.NAME)
        adapter, native = modules['multi2_adapter'], modules['hoist_native']
        need_module = modules['q32_adapter']
        c.need(c.sha(fn.src) == c.FUNCTION_SHA,
            'actual normal production function/source')
        c.need(not fn.device_caches and not fn.pre_run_hooks, 'fresh production JIT object')
        c.need(modules['service']._STATE is None, 'component must not install an encoded service')
        pins = adapter.Adapter.installed_pins(fn, modules['multi2'].__file__, {
            'vllm.v1.attention.ops.triton_unified_attention':h32.PINS['sdk'],
            'vllm.v1.attention.ops.triton_attention_helpers':h32.PINS['helpers']})
        c.need(pins == h32.PINS, 'actual production codec/JIT pins')
        for key, module_name in (('sdk','vllm.v1.attention.ops.triton_unified_attention'),
                                 ('helpers','vllm.v1.attention.ops.triton_attention_helpers')):
            c.need(Path(sys.modules[module_name].__file__).resolve() == installed[key].resolve(), 'actual import '+key)
        c.need(Path(inspect.getfile(type(fn))).resolve() == installed['jit'].resolve(), 'actual JIT path')
        result['context_or_allocations_attempted'] = True; checkpoint()
        tensors, owner = allocate(torch)
        result['tensor_descriptors'] = {k:dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
            storage_offset=t.storage_offset(), storage_bytes=t.untyped_storage().nbytes()) for k,t in tensors.items()}
        config = dict(BLOCK_M=256, BLOCK_Q=32, TILE_SIZE=64, IS_3D=False); grid=(66,2)
        launch = kwargs(dict(profile='multi2', sequences=2, segments=1, config=config), tensors, True)
        launch.update(num_warps=16,num_stages=1)
        live = adapter.Adapter(adapter.Proposal,native,directory=args.out/'bindings',target=target,
            pins=pins,implementation_digest=modules['service'].implementation_digest(),
            jit=inspect.getmodule(type(fn)),device=triton.runtime.driver.active.get_current_device(),fn=fn)
        result.update(prepare_invocations=1,JIT_executed=True); checkpoint()
        opts,binding = live.prepare(fn,launch,grid)
        result['adapter_stats'] = live.stats(); checkpoint()
        c.need(binding is not None and set(opts)=={'ir_override'}, 'prepare retained unhoisted: no valid compiled pair')
        c.need(binding['schema']=='cjj-encoded-multi2-live-binding-v1', 'actual multi2 binding identity')
        c.need(live.counts['unhoisted_warmups']==live.counts['cold_audits']==live.counts['prepared']==1,
            'exact single fresh prepare pair')
        record={}; result['records']['multi2']=record
        for role,kernel in (('unhoisted',binding['_unhoisted_kernel']),('hoisted',binding['_kernel'])):
            actual=need_module.kernel_snapshot(kernel,fn)
            c.need(actual==binding[role], 'actual final handle differs from prepared snapshot')
            artifact=Path(binding['artifact_directory'])/role
            (artifact/'compiler-metadata.txt').write_text(repr(kernel.metadata)+'\n')
            record[role]=actual
            record.setdefault('cache_groups',{})[role]=c.cache_identity(args.out/'triton-cache',actual)
        c.need(record['unhoisted']['hash']!=record['hoisted']['hash'],'distinct compiled group identities')
        record.update(binding_path=str(Path(binding['artifact_directory'])/'binding.json'),
            proposal=binding['proposal'],native=binding['native'],handle_resources=binding['resources'],
            actual_backend_hoist_and_resource_cost_review_pending=True)
        result.update(JIT_warmups_attempted=2,JIT_warmups_completed=2,
            normal_production_module=modules['multi2'].__name__,normal_production_source=modules['multi2'].__file__,
            new_final_cubin_sha256={r:record[r]['artifacts']['cubin'] for r in ('unhoisted','hoisted')},
            new_cache_identity=record['cache_groups'],
            attention_launches=0,writer_launches=0,encoder_launches=0)
        checkpoint()
"""

def driver_source():
    original=text=frozen().driver_source();edits=[]
    def edit(a,b):
        nonlocal text
        need(text.count(a)==1,'unique driver adapter '+a[:80]);offset=text.index(a)
        text=text[:offset]+b+text[offset+len(a):];edits.append((offset,a,b))
    edit('Root-owned private multi2 warm compiles','Root-owned production multi2 Adapter.prepare compiles')
    edit('import biv150_konly_multi2_hoist_source as c','import biv150_konly_multi2_service_compile_source as c')
    edit('from biv150_konly_multi2_hoist_source import kwargs','from biv150_konly_multi2_service_compile_source import kwargs')
    edit('production must equal main56','production must equal frozen d5d service source')
    edit('args.out.mkdir(parents=True)',"args.out.mkdir(parents=True)\n    (args.out/'actual-service-compile-driver.py').write_text(c.driver_source()[0])")
    edit('JIT_warmups_attempted=0','JIT_warmups_attempted=None')
    edit("'VLLM_FL_BIV150_KONLY_NATIVE_DECODE','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'):","'VLLM_FL_BIV150_KONLY_NATIVE_DECODE','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST',\n                    'VLLM_FL_BIV150_KONLY_SERVICE_MULTI2_HOIST'):")
    edit('        import torch\n','        sys.path.insert(0,str(c.ROOT))\n        import torch\n')
    start=text.index("        folder = args.out/'sources'");stop=text.index('        result.update(completed=True',start)
    edit(text[start:stop],BODY)
    inverse=text
    for offset,a,b in reversed(edits):
        need(inverse[offset:offset+len(b)]==b,'driver exact inverse span');inverse=inverse[:offset]+a+inverse[offset+len(b):]
    need(inverse==original,'complete original driver inverse')
    return text,dict(parent_sha256=sha(original),actual_sha256=sha(text),exact_byte_inverse=True,edits=len(edits))

def driver():return frozen().module_from_source('_cjj_production_multi2_compile_driver',driver_source()[0])

def plan():
    for path,digest in SOURCES.items():need(sha((ROOT/path).read_bytes())==digest,'production source '+path)
    need(not subprocess.check_output(['git','-C',str(ROOT),'diff',BASE,'--','vllm_fl','csrc']),'production freeze d5d')
    old=frozen();code,proof=driver_source()
    return dict(schema='cjj-production-multi2-prepare-compile-v1',base_commit=BASE,frozen_fixture_commit=FROZEN,
        compile_driver_dependency=old.DEPENDENCY,frozen_helper_sha256=FROZEN_SOURCE_SHA,production_sources=SOURCES,
        source_digest=FUNCTION_SHA,driver=proof,profile='multi2',num_seqs=2,total_query_rows=2048,
        logical_pages=773,logical_page_bytes=16384,query_pitch=2048,BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,
        grid=[66,2],num_warps=16,num_stages=1,normal_production_import=True,prepare_invocations=0,
        attention_launches=0,writer_launches=0,encoder_launches=0,initialized_tensor_elements=0,
        non_none_binding_required=True,GPU_executed=False,service_activated=False,
        numeric_verified=False,Graph_verified=False,performance_verified=False,service_verified=False,
        new_final_cubin_sha256=None,new_cache_identity=None,old_b2fc_qualification_inherited=False)
