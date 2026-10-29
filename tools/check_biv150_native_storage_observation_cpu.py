#!/usr/bin/env python3
"""Same Native metadata fixture and reject matrix; no Torch, GPU, or timing claim."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
from check_biv150_konly_encoded_cpu import Tensor, Storage, load
from check_biv150_native_k8_service_cpu import fixture, check, SO, restore_storage_observation

ROOT=Path(__file__).resolve().parents[1]
BASE='a455a3de364051fbea80226870c239de3a48e716'
MODULE='vllm_fl/patches/biv150_konly_encoded_native.py'


def mutate(label,runtime,entry,kw):
    q=kw['query_ptr']
    if label=='unchanged':return
    if label=='storage_too_small':q.storage=Storage(30,0x11000000,1)
    elif label=='valid_new_storage':q.storage=Storage(30,0x11000000,q.storage.size)
    elif label=='pointer':
        old=q.data_ptr;q.data_ptr=lambda:old()+2
    elif label=='stride':q.strides=(2048,128,1)
    elif label=='valid_stride':q.strides=(2048,128,1);kw['query_stride_token']=2048
    elif label=='offset':q.offset=q.storage.size//2
    elif label=='valid_offset':q.offset+=1
    elif label=='shape':q.shape=(q.shape[0],8,128)
    elif label=='epoch':entry.generation=runtime.registry.generation+1
    elif label=='owner_storage':entry.owner.storage=Storage(31,0x70000000,entry.owner.storage.size)
    elif label=='owner_pointer':
        old=entry.owner.data_ptr;entry.owner.data_ptr=lambda:old()+4
    elif label=='role':entry.roles['decode79']['key_cache_ptr'].offset+=1
    elif label=='whole_storage_source_alias':
        # The length view is before the cache view's offset, but their complete
        # backing overlaps. Preserve the original conservative rejection.
        kw['seq_lens_ptr'].storage=entry.owner.storage
    elif label=='scratch_alias':kw['segm_max_ptr'].storage=kw['segm_output_ptr'].storage
    elif label=='same_object_alias':kw['segm_expsum_ptr']=kw['segm_max_ptr']
    elif label=='two_errors':
        old=q.data_ptr;q.data_ptr=lambda:old()+2
        kw['segm_max_ptr'].storage=kw['segm_output_ptr'].storage
    else:raise AssertionError(label)


def sequence(native,path,batch,segments,pitch,label):
    runtime,entry,kw,producer,kernels,ops,capture,service,mod=fixture(native,batch,segments,pitch,path)
    reads=[];original=Tensor.untyped_storage
    def counted(t):
        # Count actual storage(t) calls without replacing any bound production
        # helper. Registry observe() remains unchanged and is not this count.
        if sys._getframe(1).f_code is native.storage.__code__:reads.append(id(t))
        return original(t)
    records=[]
    with patch.object(Tensor,'untyped_storage',counted),contextlib.redirect_stdout(io.StringIO()):
        for invocation in range(2):
            if invocation:mutate(label,runtime,entry,kw)
            before=(len(reads),len(ops.op.calls),len(kernels['decode'].calls))
            try:
                result=producer((batch,2,segments),**kw)
                outcome=dict(accepted=True,result_type=type(result).__name__)
            except (RuntimeError,ValueError) as error:
                outcome=dict(accepted=False,error_type=type(error).__name__,error=str(error))
            outcome.update(storage_helper_reads=len(reads)-before[0],native_calls=len(ops.op.calls)-before[1],
                           private_TL_calls=len(kernels['decode'].calls)-before[2])
            records.append(outcome)
    assert 'backing' not in vars(runtime.native_decode), 'no per-adapter cache'
    return records


def run(path):
    original=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+MODULE])
    candidate=(ROOT/MODULE).read_bytes()
    assert restore_storage_observation(candidate.decode()).encode()==original
    for filename in ('vllm_fl/compile_semantics.py','vllm_fl/patches/biv150_konly_encoded_runtime.py',
                     'vllm_fl/patches/biv150_konly_encoded_views.py'):
        assert (ROOT/filename).read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+filename])
    old_tree,new_tree=ast.parse(original),ast.parse(candidate)
    old_adapter=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='Adapter')
    new_adapter=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='Adapter')
    for method in ('__init__','verify','try_produce','stats'):
        select=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==method)
        assert ast.dump(select(old_adapter),include_attributes=False)==ast.dump(select(new_adapter),include_attributes=False)
    native=load('native')
    with contextlib.redirect_stdout(io.StringIO()):wiring=check(path)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='cjj-storage-reference-cpu-') as folder:
        reference_path=Path(folder)/'native.py';reference_path.write_bytes(original)
        spec=importlib.util.spec_from_file_location('native_storage_reference',reference_path)
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        requests=[(n,s,p,'unchanged') for n,s in ((64,2),(32,4)) for p in (2048,2560)]
        requests += [(32,4,2560,label) for label in ('storage_too_small','valid_new_storage','pointer','stride',
            'valid_stride','offset','valid_offset','shape','epoch','owner_storage','owner_pointer','role',
            'whole_storage_source_alias','scratch_alias','same_object_alias','two_errors')]
        requests += [(n,s,2560,label) for n,s in ((30,4),(31,4),(61,2),(63,2))
                     for label in ('unchanged','whole_storage_source_alias','role')]
        for n,s,p,label in requests:
            with contextlib.redirect_stdout(io.StringIO()):
                a=sequence(reference,path,n,s,p,label);b=sequence(native,path,n,s,p,label)
            for old,new in zip(a,b):
                assert {k:v for k,v in old.items() if k!='storage_helper_reads'}=={k:v for k,v in new.items() if k!='storage_helper_reads'},(n,s,p,label,a,b)
            assert a[0]['storage_helper_reads']==57 and b[0]['storage_helper_reads']==9
            should_accept=label in ('unchanged','valid_new_storage','valid_stride','valid_offset')
            assert a[1]['accepted'] is should_accept,(n,s,p,label,a)
            if should_accept:
                assert a[1]['storage_helper_reads']==57 and b[1]['storage_helper_reads']==9
                key='native_calls' if n in (32,64) else 'private_TL_calls'
                assert a[1][key]==b[1][key]==1
            else:assert a[1]['native_calls']==a[1]['private_TL_calls']==b[1]['native_calls']==b[1]['private_TL_calls']==0
            rows.append(dict(batch=n,segments=s,pitch=p,second_call_change=label,reference=a,candidate=b))
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,baseline=BASE,reference_source_sha256=hashlib.sha256(original).hexdigest(),
        candidate_source_sha256=hashlib.sha256(candidate).hexdigest(),source_inverse_exact=True,
        original_service_wiring_and_rejections=wiring,sequence_count=len(rows),invocations_per_implementation=2*len(rows),
        sequences=rows,unchanged_actual_storage_helper_calls_per_valid_invocation=dict(reference=57,candidate=9),
        scope='Only per-call Native backing storage observations; owner/role observations and all other metadata checks remain.',
        storage_changes_between_calls_observed=True,whole_backing_alias_policy_preserved=True,
        unsupported_shapes_validate_before_private_fallback=True,try_produce_AST_unchanged=True,
        cross_call_cache=False,GPU_or_SSH_executed=False,performance_verified=False,service_qualification=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',type=Path,default=SO)
    args=parser.parse_args();print(json.dumps(run(args.library),indent=2)+'\n',end='')
