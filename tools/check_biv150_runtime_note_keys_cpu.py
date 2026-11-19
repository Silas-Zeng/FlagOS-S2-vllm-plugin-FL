#!/usr/bin/env python3
"""Compare actual Runtime.note against a455 using CPU metadata only."""
import argparse
import ast
import contextlib
import enum
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
from unittest.mock import patch
from check_biv150_konly_encoded_cpu import Tensor,Storage

ROOT=Path(__file__).resolve().parents[1]
BASE='a455a3de364051fbea80226870c239de3a48e716'
FILE='vllm_fl/patches/biv150_konly_encoded_runtime.py'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def tensor(shape,strides,dtype='bfloat16',offset=0):
    return Tensor(Storage(1,0x100000,2**40),dtype,shape,strides,offset)


def fixture(profile):
    if profile=='writer83':
        return dict(key_source=tensor((2048,2,128),(2560,128,1),offset=2048),value_source=tensor((2048,2,128),(2560,128,1),offset=2304),slots=tensor((2048,),(1,),'int64'),key_row_stride=2560,value_row_stride=2560,key_rows=2048,value_rows=2048,slot_count=2048,cache_pages=34276),(2048,2)
    n=31 if profile=='decode' else 2048
    p=dict(query_ptr=tensor((n,16,128),(2560,128,1)),key_cache_ptr=tensor((34276,16,2,128),(16384,256,128,1),'int8'),key_scale_ptr=tensor((34276,4096),(4096,1),'float32'),value_cache_ptr=tensor((34276,16,2,128),(8192,256,128,1),offset=4096),seq_lens_ptr=tensor((31 if profile=='decode' else 1,),(1,),'int32'),block_table_ptr=tensor((31 if profile=='decode' else 1,8192),(8192,1),'int32'),segm_output_ptr=tensor((n,16,4,128),(32768,2048,128,1),'float32'),segm_max_ptr=tensor((n,16,4),(256,16,1),'float32'),segm_expsum_ptr=tensor((n,16,4),(256,16,1),'float32'),HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,SEGMENTS=4,num_warps=1,num_stages=1)
    p.update(query_code_ptr=p['key_cache_ptr'],query_scale_ptr=p['key_scale_ptr'])
    if profile=='q32':p.update(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,num_seqs=1,num_query_heads=16,num_queries_per_kv=8,KV_QUANT_MODE=0,IS_3D=False,num_warps=16)
    return p,(n,2,4) if profile=='decode' else (65,2)


def runtime(module):
    capture=[False]
    state=module.Runtime(NS(cuda=NS(is_current_stream_capturing=lambda:capture[0])),NS(),{},NS())
    return state,capture


def public(state):
    # Both final export and119 retain the original JSON-string mapping keys.
    return json.dumps(dict(stats=state.stats(),shapes=state.shape_records,overflow=state.shape_overflow,
                           kernels=list(state.kernel_records.values())),sort_keys=True)


def invoke(state,profile,handle,launch,grid):
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        try:state.note(profile,handle,launch,grid);error=None
        except (TypeError,ValueError,RuntimeError,RecursionError,IndexError) as e:error=(type(e).__name__,str(e))
    return error,output.getvalue(),public(state)


def correctness(old,new):
    checks={}
    def same(label,calls):
        a,ac=runtime(old);b,bc=runtime(new)
        for profile,handle,launch,grid,capture in calls:
            ac[0]=bc[0]=capture
            assert invoke(a,profile,handle,launch,grid)==invoke(b,profile,handle,launch,grid),label
        assert len(b.shape_json_keys)<=min(4096,len(b.shape_records)),label
        checks[label]=True
        return a,b
    for profile in ('writer83','decode','q32'):
        launch,grid=fixture(profile);h=NS(hash=profile)
        same(profile+'_repeat_and_capture',[(profile,h,launch,grid,c) for c in (False,False,True,True,False)])
        same(profile+'_keyword_order',[(profile,h,launch,grid,False),(profile,h,dict(reversed(list(launch.items()))),grid,False)])
    class Mode(enum.IntEnum):NORMAL=0
    unusual=[None,True,1,Mode.NORMAL,'unicode-张',0.0,-0.0,float('nan'),float('inf'),-float('inf'),[1,True],(1,True),{'b':2,'a':1},object()]
    for index,value in enumerate(unusual):
        launch,grid=fixture('writer83');launch['cache_pages']=value;h=NS(hash='typed')
        same('typed_'+str(index),[('writer83',h,launch,grid,False)]*3)
    # Canonical JSON distinguishes bool/int, negative zero, and list/tuple tags.
    for label,values in [('float_zero',(0.0,-0.0,0.0)),('typed_bool',(True,1,True)),('typed_sequence',([1],(1,),[1]))]:
        calls=[]
        for value in values:
            launch,grid=fixture('writer83');launch['cache_pages']=value;calls.append(('writer83',NS(hash='typed'),launch,grid,False))
        same(label,calls)
    for label,grid in [('grid_none',None),('grid_empty',()),('grid_bool',(True,2)),('grid_float',(1.0,2)),('grid_noniterable',5),('grid_object',(object(),2))]:
        launch,_=fixture('writer83');same(label,[('writer83',NS(hash='grid'),launch,grid,False)]*2)
    for label,alter in [('shape_object',lambda t:setattr(t,'shape',(object(),))),('shape_bool',lambda t:setattr(t,'shape',(True,2))),('shape_float',lambda t:setattr(t,'shape',(1.0,2))),('offset_object',lambda t:setattr(t,'offset',object()))]:
        launch,grid=fixture('writer83');alter(launch['key_source']);same(label,[('writer83',NS(hash='geometry'),launch,grid,False)]*2)
    launch,grid=fixture('writer83');launch[0]=launch['key_source']
    same('nonstring_pointer_name_original_JSON_error',[('writer83',NS(hash='bad-key'),launch,grid,False)]*2)
    cycle=[];cycle.append(cycle);launch,grid=fixture('writer83');launch['cache_pages']=cycle
    same('recursive_scalar_error',[('writer83',NS(hash='cycle'),launch,grid,False)])
    launch,grid=fixture('decode');launch['query_ptr'].shape=()
    same('post_record_first_log_error',[('decode',NS(hash='empty-query'),launch,grid,False)]*2)
    launch,grid=fixture('writer83')
    for cap in (1,None,object()):same('capture_'+type(cap).__name__,[('writer83',NS(hash='capture'),launch,grid,cap)]*2)
    for profile in ('other',7,None):same('profile_'+str(profile),[(profile,NS(hash=None),launch,None,False)]*2)
    # Same live metadata object changes between calls; never cache Tensor/ID.
    a,ac=runtime(old);b,bc=runtime(new);launch,grid=fixture('writer83');h=NS(hash='mutating')
    for field,value in [('shape',(1024,2,128)),('strides',(2048,128,1)),('offset',7),('device','cuda:1'),('dtype','torch.float32')]:
        setattr(launch['key_source'],field,value)
        assert invoke(a,'writer83',h,launch,grid)==invoke(b,'writer83',h,launch,grid)
    checks['live_metadata_changes']=True
    def leaves(value):
        if isinstance(value,tuple):
            for x in value:yield from leaves(x)
        else:yield value
    assert not any(isinstance(x,Tensor) for key in b.shape_json_keys for x in leaves(key))
    checks['index_retains_no_Tensors']=True
    # Direct export consumer compatibility plus values mutated outside note.
    key=next(iter(a.shape_records))
    for state in (a,b):state.shape_records[key]['pointers']['key_source']['shape'][0]=999
    assert invoke(a,'writer83',h,launch,grid)==invoke(b,'writer83',h,launch,grid)
    assert json.dumps(a.shape_records,sort_keys=True,allow_nan=False)==json.dumps(b.shape_records,sort_keys=True,allow_nan=False)
    checks['original_string_keys_and_mutable_record_values']=True
    a,ac=runtime(old);b,bc=runtime(new);launch,grid=fixture('writer83');h=NS(hash='cap')
    with contextlib.redirect_stdout(io.StringIO()):
        for n in range(4100):
            launch['cache_pages']=50000+n;a.note('writer83',h,launch,grid);b.note('writer83',h,launch,grid)
        launch['cache_pages']=50000;a.note('writer83',h,launch,grid);b.note('writer83',h,launch,grid)
    assert public(a)==public(b) and len(b.shape_records)==len(b.shape_json_keys)==4096 and b.shape_overflow==4
    checks['cap_overflow_and_existing_hit_after_full']=True
    with tempfile.TemporaryDirectory(prefix='cjj-note-export-cpu-') as temp,patch.object(time,'time',return_value=1234.5),contextlib.redirect_stdout(io.StringIO()):
        a.stats_path=Path(temp)/'a.json';b.stats_path=Path(temp)/'b.json'
        assert a.shutdown(lambda *args:'original')(object())==b.shutdown(lambda *args:'original')(object())=='original'
        assert a.stats_path.read_bytes()==b.stats_path.read_bytes()
        assert a.finalize('again')==b.finalize('again') and a.final_export is not None
    checks['shutdown_final_export_bytes_and_one_shot']=True
    a,ac=runtime(old);b,bc=runtime(new)
    with tempfile.TemporaryDirectory(prefix='cjj-note-export-error-cpu-') as temp,contextlib.redirect_stdout(io.StringIO()):
        path=Path(temp)/'exists.json';path.write_text('existing')
        a.stats_path=b.stats_path=path
        assert a.finalize('no-overwrite') is b.finalize('no-overwrite') is None
        assert path.read_text()=='existing'
    checks['final_export_error_preserved']=True
    # Count record serialization itself; the first kernel artifact/print paths
    # remain unchanged and are excluded by the exact record-field predicate.
    original=json.dumps;counts={}
    for name,module in [('A',old),('B',new)]:
        r,_=runtime(module);launch,grid=fixture('decode');h=NS(hash='json-count');counter=[0]
        def dumps(value,*args,**kwargs):
            if type(value) is dict and set(value)=={'profile','hash','grid','capture','pointers','selected_scalars'}:counter[0]+=1
            return original(value,*args,**kwargs)
        with patch.object(json,'dumps',dumps),contextlib.redirect_stdout(io.StringIO()):
            for _ in range(10):r.note('decode',h,launch,grid)
        counts[name]=counter[0]
    assert counts=={'A':10,'B':1};checks['repeat_10_record_serializations_become_1']=True
    return dict(checks=checks,check_count=len(checks),record_serialization_calls=counts)


def bench(old,new,iterations,rounds,templates=None):
    result=[]
    cases=templates if templates is not None else [(profile,kind,*fixture(profile),NS(hash='CPU-template-'+profile),False)
        for profile,kind in [('writer83','hit'),('decode','hit'),('q32','hit'),('writer83','miss')]]
    for profile,kind,template,grid,h,capture in cases:
        rows=[];number=iterations if kind=='hit' else min(iterations,1024)
        for trial in range(rounds):
            for arm,module in [('A',old),('B',new),('B',new),('A',old)]:
                r,flag=runtime(module);flag[0]=capture;launch=dict(template)
                with contextlib.redirect_stdout(io.StringIO()):r.note(profile,h,launch,grid)
                if kind=='hit':
                    for _ in range(300):r.note(profile,h,launch,grid)
                wall=time.perf_counter_ns();cpu=time.process_time_ns()
                for n in range(number):
                    if kind=='miss':launch['cache_pages']=100000+n
                    r.note(profile,h,launch,grid)
                rows.append(dict(arm=arm,trial=trial,cpu_ns_per_call=(time.process_time_ns()-cpu)/number,
                                 wall_ns_per_call=(time.perf_counter_ns()-wall)/number))
        a=statistics.median(x['cpu_ns_per_call'] for x in rows if x['arm']=='A');b=statistics.median(x['cpu_ns_per_call'] for x in rows if x['arm']=='B')
        result.append(dict(profile=profile,kind=kind,calls_per_arm=number,ABBA=rows,median_CPU_A_ns=a,median_CPU_B_ns=b,A_div_B=a/b))
    return result


def saved_records(old,new,before_path,after_path,benchmark,iterations,rounds):
    """Reconstruct metadata only; snapshot counters contain no call ordering."""
    before=json.loads(before_path.read_bytes());after=json.loads(after_path.read_bytes())
    assert before['run_id']==after['run_id']
    assert before['phase']=='warm-after-4096' and after['phase']=='score-after-4096'
    a,ac=runtime(old);b,bc=runtime(new);templates={};selected={};capture_calls={};fresh_calls=0
    eligible=0;delta_calls=0;active=0;new_keys=0
    for key,row in after['shape_records'].items():
        record={k:v for k,v in row.items() if k!='host_calls'}
        assert json.dumps(record,sort_keys=True,separators=(',',':'))==key
        launch={}
        for name,value in record['pointers'].items():
            assert set(value)=={'dtype','device','shape','strides','storage_offset'}
            assert value['dtype'].startswith('torch.')
            t=tensor(value['shape'],value['strides'],value['dtype'].removeprefix('torch.'),value['storage_offset'])
            t.device=value['device'];launch[name]=t
        for name,value in record['selected_scalars'].items():
            # This saved window has only exact int/bool; reject unknown data.
            assert set(value)=={'type','value'} and value['type'] in ('int','bool')
            assert type(value['value']).__name__==value['type'];launch[name]=value['value']
        profile=record['profile'];h=NS(hash=record['hash']);grid=record['grid']
        ac[0]=bc[0]=record['capture']
        for repeat in (1,2):
            old_result=invoke(a,profile,h,launch,grid);new_result=invoke(b,profile,h,launch,grid)
            assert old_result==new_result and old_result[0] is None
            assert b.shape_records[key]==dict(record,host_calls=repeat)
        assert len(b.shape_json_keys)==len(b.shape_records)
        eligible+=1
        delta=row['host_calls']-before['shape_records'].get(key,{}).get('host_calls',0)
        assert type(delta) is int and delta>=0
        delta_calls+=delta;active+=int(delta>0)
        selected[profile]=selected.get(profile,0)+delta
        capture_calls[str(record['capture'])]=capture_calls.get(str(record['capture']),0)+delta
        if key not in before['shape_records']:
            new_keys+=1;fresh_calls+=delta
        if delta>0 and (profile not in templates or delta>templates[profile][0]):
            templates[profile]=(delta,(profile,'hit',launch,grid,h,record['capture']),key)
    assert set(before['shape_records'])<=set(after['shape_records'])
    assert before['shape_overflow']==after['shape_overflow']==0
    cases=[value[1] for _,value in sorted(templates.items())]
    writer=templates['writer83'][1];cases.append((writer[0],'miss',*writer[2:]))
    return dict(passed=True,before_sha256=hashlib.sha256(before_path.read_bytes()).hexdigest(),
        after_sha256=hashlib.sha256(after_path.read_bytes()).hexdigest(),run_id=after['run_id'],
        snapshot_records=len(after['shape_records']),metadata_records_reconstructed=eligible,
        original_and_candidate_calls_each=2*eligible,all_canonical_JSON_keys_and_records_equal=True,
        window_host_note_calls=delta_calls,window_active_records=active,window_new_records=new_keys,
        window_calls_to_new_records=fresh_calls,window_profile_host_counts=selected,
        window_capture_host_counts=capture_calls,overflow=after['shape_overflow'],
        templates={name:dict(record_key_sha256=hashlib.sha256(value[2].encode()).hexdigest(),
            key_chars=len(value[2]),score_delta_host_calls=value[0]) for name,value in sorted(templates.items())},
        benchmark=bench(old,new,iterations,rounds,cases) if benchmark else None,
        scope='Saved metadata reconstructed on CPU, not tensor data, GPU execution, request ordering, complete A acceptance, Graph replay counts or timing. All-miss varies cache_pages synthetically on the saved writer geometry.')


def inverse(source):
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runtime')
    helper=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='shape_key')
    start=min(helper.lineno,*(d.lineno for d in helper.decorator_list));lines=source.splitlines(True)
    chunk=''.join(lines[start-1:helper.end_lineno+1]);assert chunk.endswith('\n\n')
    assert source.count(chunk)==1;source=source.replace(chunk,'',1)
    changes=[
        ('        self.shape_json_keys = {}\n',''),
        ("        fast_key = self.shape_key(record)\n        key = self.shape_json_keys.get(fast_key) if fast_key is not None else None\n        missing = key is None\n        if missing: key = json.dumps(record,sort_keys=True,separators=(',',':'))","        key = json.dumps(record,sort_keys=True,separators=(',',':'))"),
        ("        if missing and fast_key is not None and key in self.shape_records and len(self.shape_json_keys)<4096:\n            self.shape_json_keys[fast_key] = key\n",'')]
    for new,old in changes:
        assert source.count(new)==1,new
        source=source.replace(new,old,1)
    return source


def run(benchmark=False,iterations=5000,rounds=3,before=None,after=None):
    original=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+FILE]);candidate=(ROOT/FILE).read_bytes()
    assert inverse(candidate.decode()).encode()==original
    old_ast,new_ast=ast.parse(original),ast.parse(candidate)
    for tree in (old_ast,new_ast):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runtime')
        cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name not in ('__init__','note','shape_key')]
    assert ast.dump(old_ast,include_attributes=False)==ast.dump(new_ast,include_attributes=False)
    with tempfile.TemporaryDirectory(prefix='cjj-note-reference-') as folder:
        p=Path(folder)/'reference.py';p.write_bytes(original);old=load('cjj_note_reference',p);new=load('cjj_note_candidate',ROOT/FILE)
        checked=correctness(old,new);timing=bench(old,new,iterations,rounds) if benchmark and before is None else None
        saved=saved_records(old,new,before,after,benchmark,iterations,rounds) if before is not None else None
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,base=BASE,source_sha256=hashlib.sha256(candidate).hexdigest(),
        reference_sha256=hashlib.sha256(original).hexdigest(),checks=checked,benchmark=timing,saved_metadata=saved,
        platform=platform.platform(),machine=platform.machine(),python=sys.version,
        fixture_scope='Synthetic metadata. Q32 2048/grid65/constants correspond to saved117 evidence; full geometry is representative, not a complete actual shape-record replay.',
        source_inverse_byte_exact=True,all_other_runtime_methods_and_Registry_AST_identical=True,Graph_replay_time_measured=False,
        GPU_SSH_or_network_executed=False,actual_service_speedup_measured=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',action='store_true')
    p.add_argument('--iterations',type=int,default=5000);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--snapshot-before',type=Path);p.add_argument('--snapshot-after',type=Path);a=p.parse_args()
    assert a.iterations>0 and a.rounds>0
    assert (a.snapshot_before is None)==(a.snapshot_after is None)
    print(json.dumps(run(a.benchmark,a.iterations,a.rounds,a.snapshot_before,a.snapshot_after),indent=2)+'\n',end='')
