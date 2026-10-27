"""Private diagnostic driver: CPU default; explicit compile proof precedes GPU mode."""
import argparse
import ast
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from biv150_v8_decode_factory import ROOT, FIXTURE, MAIN_GQA_SHA, make_vdecode, sha
from biv150_v8_bias_probe import (CASES, OPTION_KEYS, adapted_probe,
    descriptor_functions, mechanism_check, pinned_sources)

MAIN_HASHES = {'ragged':'c44ed4caa048314f475e533a47d072c8460987e9c4036a3217fbd955f644ff19',
 'ragged-long':'c44ed4caa048314f475e533a47d072c8460987e9c4036a3217fbd955f644ff19',
 'steady4k':'641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff',
 'steady16k':'e29a0c84d6b99f1ad44189fbdb74434b2d7da5675af2e2e2422941739916adb9'}
ROLES = {'gqa_main','gqa_old','gqa_A','reducer',
         'old_writer_False','old_writer_True','A_writer_False','A_writer_True','old_encoder','A_encoder'}


def head():
    return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def preflight():
    pinned_sources()
    assert sha((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_bytes())==MAIN_GQA_SHA
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip(), 'Dirty source'
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',9031))!=0, 'Service active'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True), 'GPU busy'


def loaded_helper_paths():
    for name in ('biv150_benchmark_utils','biv150_v8_bf16p_reference','biv150_v8_bf16p_validation'):
        if name in sys.modules:
            assert Path(sys.modules[name].__file__).resolve()==ROOT/'tools'/(name+'.py'), ('Wrong imported helper',name)


def options(compiled):
    return {key:getattr(compiled.metadata,key,None) for key in OPTION_KEYS}


def abi(args, kwargs, grid):
    def describe(value):
        if hasattr(value,'data_ptr'):
            return dict(dtype=str(value.dtype),shape=list(value.shape),stride=list(value.stride()),
                        storage_offset=value.storage_offset(),contiguous=value.is_contiguous())
        assert value is None or type(value) in (str,int,float,bool)
        return value
    return dict(args=[describe(x) for x in args],kwargs={k:describe(v) for k,v in kwargs.items()},grid=list(grid))


def export(compiled, directory, call):
    compiled._init_handles();directory.mkdir(parents=True,exist_ok=False)
    files = {}
    for name,value in compiled.asm.items():
        if isinstance(value,(str,bytes)):
            raw = value.encode() if isinstance(value,str) else value
            path = directory/('actual.'+name);path.write_bytes(raw)
            files[name] = dict(path=str(path.resolve()),bytes=len(raw),sha256=sha(raw))
    return dict(hash=str(compiled.hash),abi=call,options=options(compiled),
        n_regs=compiled.n_regs,n_spills=compiled.n_spills,
        metadata={key:getattr(compiled.metadata,key,None) for key in
                  ('shared','num_warps','num_stages','use_sme','warp_size','triton_version')},files=files)


def read_ir(record):
    result = {}
    for name in ('ttgir','llir','asm'):
        item = record['files'][name];raw = Path(item['path']).read_bytes()
        assert sha(raw)==item['sha256'] and len(raw)==item['bytes'], 'IR changed'
        result[name] = raw.decode()
    return result


def known_resources(kernel):
    assert type(kernel['n_regs']) is int and kernel['n_regs']>0, 'Unknown/invalid register count'
    assert type(kernel['n_spills']) is int and kernel['n_spills']==0, 'Unknown/nonzero spill count'
    metadata=kernel['metadata']
    for key in ('shared','use_sme'):
        assert type(metadata[key]) is int and metadata[key]>=0, 'Unknown/invalid '+key
    for key in ('num_warps','num_stages','warp_size'):
        assert type(metadata[key]) is int and metadata[key]>0, 'Unknown/invalid '+key
    assert type(metadata['triton_version']) is str and metadata['triton_version'].strip(), 'Unknown compiler version'
    return dict(n_regs=kernel['n_regs'],n_spills=kernel['n_spills'],metadata=dict(metadata))


def inspect(row):
    assert set(row['kernels'])==ROLES, 'Incomplete consumer compilation proof'
    old,new = row['kernels']['gqa_old'],row['kernels']['gqa_A']
    verdict = mechanism_check(read_ir(old),read_ir(new))
    if old['hash']==new['hash']:
        verdict['reasons'].append('A/frozen compilation hash unchanged')
    if old['abi']!=new['abi'] or old['options']!=new['options']:
        verdict['reasons'].append('V8 ABI/options changed')
    for key in ('num_warps','num_stages','warp_size','triton_version'):
        if old['metadata'][key]!=new['metadata'][key]:
            verdict['reasons'].append('V8 metadata changed: '+key)
    verdict['evidence']['resources']={}
    for role,kernel in row['kernels'].items():
        try:verdict['evidence']['resources'][role]=known_resources(kernel)
        except (AssertionError,KeyError) as error:verdict['reasons'].append('Unknown/unreviewed resources: '+role+': '+str(error))
        if any(kernel['options'][key] is None for key in OPTION_KEYS if key!='maxnreg'):
            verdict['reasons'].append('Compiler option metadata unavailable')
    verdict['mechanism_passed'] = not verdict['reasons']
    return verdict


class Warmup:
    def __init__(self,kernel,callback):self.kernel,self.callback=kernel,callback
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            compiled = self.kernel.warmup(*args,grid=grid,**kwargs)
            self.callback(compiled,abi(args,kwargs,grid))
            return compiled
        return call


class Scaled:
    def __init__(self,kernel,scales):self.kernel,self.scales=kernel,scales
    def __getitem__(self,grid):
        def call(**kwargs):
            return self.kernel[grid](**kwargs,value_scale_ptr=self.scales,
                scale_stride_block=self.scales.stride(0),scale_stride_slot=self.scales.stride(1),scale_stride_head=self.scales.stride(2))
        return call


def compile_only(args):
    preflight();args.out.mkdir(parents=True,exist_ok=False)
    source_dir = args.out/'sources';source_dir.mkdir()
    generated,source_proof = make_vdecode()
    paths = {'old':source_dir/'frozen_v8.py','A':source_dir/'bias_A.py'}
    for name,raw in [('old',FIXTURE.read_bytes()),('A',generated)]:paths[name].write_bytes(raw)
    import torch
    import biv150_benchmark_utils as bench
    loaded_helper_paths()
    base = bench.load_module(ROOT/'vllm_fl/patches/biv150_gqa_decode.py','v8_bf16p_shipping_gqa')
    modules = {v:bench.load_module(p,'v8_bias_'+v) for v,p in paths.items()}
    _,construct = descriptor_functions()
    template = ast.parse((ROOT/'tools/fixtures/biv150_v8_probe_4a.py.txt').read_bytes())
    append_fn = next(n for n in ast.walk(template) if isinstance(n,ast.FunctionDef) and n.name=='append')
    encoder = next(n for n in ast.walk(template) if isinstance(n,ast.Expr) and ast.unparse(n).startswith('own.encode_cache_v['))
    record = dict(completed=False,compile_only=True,kernel_launches=0,source_commit=head(),pins=pinned_sources(),
        source_proof=source_proof,sources={v:dict(path=str(p.resolve()),sha256=sha(p.read_bytes())) for v,p in paths.items()},
        cases=[],requested_cases=args.case,errors=[],precision_gate=False,official_score=False,
        private_consumer_wired=True,consumer_device_tested=False)
    destination = args.out/'compile-proof.json';save(destination,record)
    try:
        assert base.TARGET_PROGRAMS==256
        for name in args.case:
            batch,length,segments = CASES[name]
            assert base._pick_segments(batch,2,16)==segments
            c,plan = construct(torch,batch,length,segments)
            other,_ = construct(torch,batch,length,segments)
            c['out']=torch.empty_like(c['q'])
            # Separate source and destination cache allocations, as in the real
            # encoder/writer harness. No values are initialized or kernels run.
            storage_base = lambda t:t.data_ptr()-t.storage_offset()*t.element_size()
            assert storage_base(c['k'])!=storage_base(other['k'])
            quant = dict(c,k=other['k'],v=other['v8'],vscale=other['scale'])
            row = dict(case=name,segments=segments,descriptors=plan,separate_cache_allocations=True,
                       compile_allocation_upper_bytes=2*plan['unique_allocation_bytes'],kernels={})
            record['cases'].append(row);save(destination,record)
            def wrap(role,kernel):
                def captured(obj,call):
                    row['kernels'][role]=export(obj,args.out/name/role,call)
                    save(destination,record)
                return Warmup(kernel,captured)
            compiled = bench.main_call(SimpleNamespace(LOG2E=base.LOG2E,_gqa_decode_attention=wrap('gqa_main',base._gqa_decode_attention)),c,segments)
            assert str(compiled.hash)==MAIN_HASHES[name], 'Current-main BF16 specialization differs'
            bench.reduce_call(SimpleNamespace(_reduce_partials=wrap('reducer',base._reduce_partials)),c,segments,4)
            for variant,module in modules.items():
                bench.main_call(SimpleNamespace(LOG2E=base.LOG2E,_gqa_decode_attention=Scaled(wrap('gqa_'+variant,module.gqa_decode_v8_bf16p),other['scale'])),quant,segments)
                scope = dict(own=SimpleNamespace(append_token_kv=wrap(variant+'_writer',module.append_token_kv)),
                    new_key=torch.empty(batch,2,128,device='cuda',dtype=torch.bfloat16),
                    new_value=torch.empty(batch,2,128,device='cuda',dtype=torch.bfloat16),
                    slots=torch.empty(batch,device='cuda',dtype=torch.int64),batch=batch,
                    packed=SimpleNamespace(shape=plan['packed_shape']),scales=other['scale'])
                exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(append_fn)],type_ignores=[])),'frozen_append','exec'),scope)
                # ENCODE is a specialization; freeze both roles separately.
                for encode in (False,True):
                    scope['own'].append_token_kv=wrap(variant+'_writer_'+str(encode),module.append_token_kv)
                    scope['append'](quant if encode else c,encode)
                scope.update(own=SimpleNamespace(encode_cache_v=wrap(variant+'_encoder',module.encode_cache_v)),
                             value=c['v'],encoded_v=other['v8'])
                exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(encoder)],type_ignores=[])),'frozen_encoder','exec'),scope)
                scope.pop('append',None)  # break the temporary function/globals cycle
            row['mechanism'] = inspect(row)
            record['mechanism_passed']=all(r['mechanism']['mechanism_passed'] for r in record['cases'])
            save(destination,record)
            assert row['mechanism']['mechanism_passed'], 'Unknown/unchanged decode mechanism: stop before further cases or device execution'
            del scope,c,other,quant
        record['completed']=True
        record['mechanism_passed']=all(r['mechanism']['mechanism_passed'] for r in record['cases'])
        save(destination,record)
        assert record['mechanism_passed'], 'Unknown/unchanged decode mechanism: stop before numeric execution or timing'
    except Exception as error:
        record['errors'].append(repr(error));save(destination,record);raise


def validate_proof(path,chosen):
    record = json.loads(path.read_text())
    assert record['completed'] and record['compile_only'] and record['kernel_launches']==0
    assert record['source_commit']==head() and record['pins']==pinned_sources()
    assert not record['errors'] and record['mechanism_passed'] is True
    generated,source_proof = make_vdecode();assert source_proof==record['source_proof']
    for variant,raw in [('old',FIXTURE.read_bytes()),('A',generated)]:
        entry=record['sources'][variant];assert Path(entry['path']).read_bytes()==raw and sha(raw)==entry['sha256']
    rows={row['case']:row for row in record['cases']};assert rows and set(chosen)<=rows.keys()
    plan,_=descriptor_functions()
    for name,row in rows.items():
        assert row['descriptors']==plan(*CASES[name]) and row['separate_cache_allocations'] is True
        assert row['kernels']['gqa_main']['hash']==MAIN_HASHES[name]
        assert inspect(row)['mechanism_passed']
    return record


class Proven:
    def __init__(self,kernel,role,proof):
        self.kernel,self.role=kernel,role
        self.allowed=[row['kernels'][role] for row in proof['cases']]
        self.checked=set()
    def __getitem__(self,grid):
        launch=self.kernel[grid]
        def call(*args,**kwargs):
            call_abi=abi(args,kwargs,grid)
            signature=(json.dumps(call_abi,sort_keys=True),tuple(x.data_ptr() for x in [*args,*kwargs.values()] if hasattr(x,'data_ptr')))
            if signature not in self.checked:
                compiled=self.kernel.warmup(*args,grid=grid,**kwargs);compiled._init_handles()
                expected=next((item for item in self.allowed if item['hash']==str(compiled.hash) and item['abi']==call_abi),None)
                assert expected is not None, ('Unproven actual specialization',self.role)
                assert options(compiled)==expected['options'], 'Actual compiler options differ'
                actual=dict(n_regs=compiled.n_regs,n_spills=compiled.n_spills,
                    metadata={key:getattr(compiled.metadata,key,None) for key in expected['metadata']})
                assert known_resources(actual)==known_resources(expected), 'Actual resources/metadata differ'
                for name in ('ttgir','llir','asm'):
                    value=compiled.asm[name];raw=value.encode() if isinstance(value,str) else value
                    assert sha(raw)==expected['files'][name]['sha256'], ('Actual IR differs',self.role,name)
                self.checked.add(signature)
            return launch(*args,**kwargs)
        return call


def timing_label(token):
    return 'frozen_v8' if token=='old-v8' else 'bias_A' if token else 'main_bf16'


class NumericScope:
    def __init__(self,proof,destination,*,tensor_backend=None,module_loader=None):
        self.proof,self.destination=proof,destination;self.old={};self.passed=set();self.records=[]
        self.tensor_backend,self.module_loader=tensor_backend,module_loader
    def tensors(self):
        if self.tensor_backend is None:
            import torch
            self.tensor_backend=torch
        return self.tensor_backend
    def snapshots(self,c):
        torch=self.tensors()
        return {name:dict(shape=list(c[name].shape),dtype=str(c[name].dtype),
                         bytes=c[name].cpu().contiguous().view(torch.uint8).clone())
                for name in ('scratch','maxima','sums','out')}
    def equal(self,a,b):
        torch=self.tensors()
        if isinstance(a,torch.Tensor):
            return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(
                a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(self.equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(self.equal(x,y) for x,y in zip(a,b))
        return type(a)==type(b) and a==b
    def assert_snapshots(self,a,b):
        if not self.equal(a,b):
            torch=self.tensors()
            torch.save(dict(graph=a,eager=b),self.destination/'graph-eager-bitwise-failure.pt')
            raise AssertionError('Graph/eager scratch/max/sum/output bytes differ')
    def capture(self,args,fixture,artifact,key,value):
        identity=(artifact['case'],artifact['pattern'])
        inputs=(key,value,fixture['table'],fixture['lengths'],fixture['slot_mapping'],fixture['patterns'][artifact['pattern']])
        if args.variant=='old':self.old[identity]=(inputs,artifact['bitwise_fields'])
        else:
            self.passed.discard(identity)
            expected=self.old[identity]
            input_equal=self.equal(inputs,expected[0]);fields_equal=self.equal(artifact['bitwise_fields'],expected[1])
            self.records.append(dict(case=identity[0],pattern=identity[1],full_cpu_inputs_bitwise=input_equal,
                scratch_max_sum_output_bitwise=fields_equal,precision_gate=False))
            save(self.destination/'A-vs-old-bitwise.json',self.records)
            assert input_equal and fields_equal, 'A/frozen V8 bitwise mismatch: timing forbidden'
            self.passed.add(identity)
    def require(self,args,name):
        assert args.variant=='A' and all((name,p) in self.passed for p in (0,1)), 'Bitwise equivalence not proven'
    def wrap_base(self,base,args):
        loaded_helper_paths()
        base._gqa_decode_attention=Proven(base._gqa_decode_attention,'gqa_main',self.proof)
        base._reduce_partials=Proven(base._reduce_partials,'reducer',self.proof)
    def make_gate(self,own,args):
        old=own
        if args.variant!='old':
            loader=self.module_loader
            if loader is None:
                import biv150_benchmark_utils as bench
                loader=bench.load_module
            old=loader(Path(self.proof['sources']['old']['path']),'v8_bias_old')
        scope=self
        class Selector:
            mode=args.variant
            def select(self,token):
                self.mode='old' if args.variant=='old' or token=='old-v8' or token is False else 'A'
                return bool(token)
            def __getitem__(self,grid):return producers[self.mode][grid]
        selector=Selector()
        producers={'old':Proven(old.gqa_decode_v8_bf16p,'gqa_old',self.proof)}
        if args.variant=='A':producers['A']=Proven(own.gqa_decode_v8_bf16p,'gqa_A',self.proof)
        class SelectedWriter:
            def __getitem__(self,grid):
                def call(*values,**kwargs):
                    return writers[selector.mode][kwargs['ENCODE']][grid](*values,**kwargs)
                return call
        writers={}
        for label,module in [('old',old),*([('A',own)] if args.variant=='A' else [])]:
            writers[label]={flag:Proven(module.append_token_kv,label+'_writer_'+str(flag),scope.proof) for flag in (False,True)}
        own.append_token_kv=SelectedWriter()
        own.encode_cache_v=Proven(own.encode_cache_v,args.variant+'_encoder',self.proof)
        own.gqa_decode_v8_bf16p=selector
        return selector


def run_gpu(args):
    proof=validate_proof(args.ir_proof,args.case);preflight()
    args.out.mkdir(parents=True,exist_ok=False)
    scope=NumericScope(proof,args.out)
    tree,adaptation=adapted_probe()
    summary=dict(completed=False,source_commit=head(),compile_proof_sha256=sha(args.ir_proof.read_bytes()),
        adaptation=adaptation,phases=[],errors=[],precision_gate=False,official_score=False,
        service_tested=False,capacity_reclaimed=False)
    try:
        for name in args.case:
            for variant in ('old','A'):
                output=args.out/(name+'-'+variant)
                phase=SimpleNamespace(run_gpu=True,numeric_diagnostic=args.numeric_diagnostic,warps=1,case=[name],
                    out=output,variant=variant,numeric_only=variant=='old',
                    kernel_source=Path(proof['sources'][variant]['path']),kernel_module='v8_bias_'+variant)
                namespace=dict(__file__=str(Path(__file__).resolve()),__name__='private_frozen4a_bias_consumer',
                    wrap_base=scope.wrap_base,make_gate=scope.make_gate,snapshot_fields=scope.snapshots,
                    assert_snapshots_equal=scope.assert_snapshots,capture_evidence=scope.capture,
                    require_equivalence=scope.require,timing_label=timing_label)
                exec(compile(tree,'frozen4a_private_bias_probe','exec'),namespace)
                namespace['main'](phase)
                summary['phases'].append(dict(case=name,variant=variant,result=str(output/'result.json')))
            for pattern in (0,1):scope.old.pop((name,pattern))
        summary['completed']=True
    except Exception as error:
        summary['errors'].append(repr(error));raise
    finally:save(args.out/'run-summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--compile-only',action='store_true')
    mode.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--out',type=Path);parser.add_argument('--ir-proof',type=Path)
    parser.add_argument('--numeric-diagnostic',action='store_true')
    parser.add_argument('--case',action='append',choices=sorted(CASES))
    args=parser.parse_args();args.case=args.case or ['ragged','steady4k','steady16k']
    if len(set(args.case))!=len(args.case):parser.error('Duplicate cases are not supported')
    if (args.compile_only or args.run_gpu) and args.out is None:parser.error('--out required')
    if args.run_gpu and args.ir_proof is None:parser.error('--ir-proof required')
    if args.compile_only and args.numeric_diagnostic:parser.error('Numeric mode does not apply to compilation')
    if args.compile_only:compile_only(args)
    elif args.run_gpu:run_gpu(args)
    else:
        _,proof=make_vdecode();_,adaptation=adapted_probe()
        print(json.dumps(dict(source=proof,adaptation=adaptation,cases=args.case,
            compile_and_device_opt_in_required=True,actual_ir_available=False,
            private_consumer_wired=True,
            consumer_device_tested=False,precision_gate=False,official_score=False),indent=2))
