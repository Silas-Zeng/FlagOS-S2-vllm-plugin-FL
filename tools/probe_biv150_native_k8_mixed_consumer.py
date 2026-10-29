#!/usr/bin/env python3
"""Default CPU; Root-only short six-N consumer with exact accepted8b padded reference."""
import argparse
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import traceback
import biv150_native_k8_mixed_numeric_contract as c

def dispatch_binding(torch,args):
    paths={'reference':(args.main_library,c.MAIN_SHA,c.MAIN_NAMESPACE),
           'candidate':(args.mixed_library,args.mixed_sha256,c.factory.NAMESPACE)}
    actual={}
    for role,(path,digest,namespace) in paths.items():
        c.need(c.sha(path)==digest,'library changed before load '+role)
        torch.ops.load_library(str(path))
        op=getattr(torch.ops,namespace).produce
        schema=namespace+'::produce(Tensor query, Tensor key_code, Tensor scale_full, Tensor value_offset, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale_log2e, int segments) -> ()'
        c.need(str(op.default._schema)==schema and torch._C._dispatch_has_kernel_for_dispatch_key(namespace+'::produce','CUDA'),'actual namespace/schema '+role)
        c.need(str(path.resolve()) in {str(Path(x).resolve()) for x in torch.ops.loaded_libraries},'actual library set '+role)
        c.need(c.sha(path)==digest,'library changed during load '+role)
        actual[role]=dict(path=str(path.resolve()),sha256=digest,namespace=namespace,schema=schema,loaded_library_set_verified=True)
    c.need(paths['reference'][2]!=paths['candidate'][2],'distinct dispatcher namespaces')
    return getattr(torch.ops,args.namespace).produce,actual

def execute(a):
    root,binding=c.admission(a)
    original,proof=c.consumer_source(root)
    line='torch.ops.load_library(str(args.library));op=getattr(torch.ops,args.namespace).produce'
    source,loadproof=c.edit(original,[(line,"op,dispatch=dispatch_binding(torch,args);result['actual_dual_library_load']=dispatch",1)])
    result=dict(plan=c.plan(),binding=binding,completed=False,errors=[],cases=[],
        all_real_OML_final_bitwise=False,implementation_hard_passed=False,precision_gate=False,
        performance_verified=False,service_verified=False)
    a.out.mkdir();out=a.out
    def save():(out/'mixed-consumer-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    save()
    try:
        for n,s,pitch in c.CASES:
            key=f'n{n}s{s}q{pitch}';folder=out/key;folder.mkdir()
            rec=dict(case=[n,s,pitch],arms={});result['cases'].append(rec);save()
            summaries={};folders={}
            # Reference first; the first run loads both actual libraries and
            # rejects dispatcher/schema mismatch before any attention call.
            for role,physical,digest,namespace,path in [
                ('reference',32 if s==4 else 64,c.MAIN_SHA,c.MAIN_NAMESPACE,a.main8b_library),
                ('candidate',n,binding['library']['sha256'],c.factory.NAMESPACE,a.library)]:
                module=c.load(root,'probe_biv150_native_k8_consumer_v2','cjj118_consumer_'+key+'_'+role)
                fixture=c.fixture(module,physical,n,s,pitch)
                def role_plan():
                    p=module.f.plan();p.update(cases=fixture.CASES,library_sha256=digest,
                        mixed_consumer_binding=binding,mixed_case=[n,s,pitch],role=role,
                        physical_rows=physical,valid_rows=n,reference_zero_length_rows=physical-n,
                        source_adapter=proof,load_adapter=loadproof,
                        scratch_physical_width=16,table_physical_pitch=8192,table_columns=8192,logical_reference_table_columns=16,
                        kernel_or_dtype_changed=True,numeric_expression_or_dtype_changed=False,
                        original_numerical_gates_unchanged=True,qualification_inherited=False)
                    return p
                args=SimpleNamespace(library=path,role_sha256=digest,namespace=namespace,
                    main_library=a.main8b_library,mixed_library=a.library,mixed_sha256=binding['library']['sha256'],
                    valid_rows=n,out=folder/role,codec_summary=a.codec_summary,codec_proof_sha256=a.codec_proof_sha256)
                ns=dict(vars(module),f=fixture,role_plan=role_plan,dispatch_binding=dispatch_binding)
                exec(compile(source,'<PR118-original-consumer-width16-role>','exec'),ns)
                with (folder/(role+'-stdout.txt')).open('w') as log,contextlib.redirect_stdout(log):status=ns['run'](args)
                summary=args.out/'consumer-summary.json';rec['arms'][role]=dict(summary=str(summary.relative_to(out)),sha256=c.sha(summary));save()
                c.need(status==0,'original full consumer refused '+key+'/'+role)
                j=json.loads(summary.read_text());c.need(j['completed'] and not j['errors'],'actual consumer terminal')
                c.need(j['consumer_numeric_verified'] and j['Graph_verified'] and j['side_stream_verified'],'original full gates')
                c.need(j['library_sha256']==digest and j['mixed_consumer_binding']==binding,'actual role provenance')
                summaries[role]=j;folders[role]=args.out
            rec['bitwise']=c.compare_records(folders['candidate'],summaries['candidate'],folders['reference'],summaries['reference'],n,pitch)
            rec['completed']=True;save()
        result.update(completed=True,all_real_OML_final_bitwise=True,implementation_hard_passed=True)
    except Exception as e:
        result['errors'].append(repr(e));(out/'failure.txt').write_text(traceback.format_exc())
    save();print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['completed'] else 1

def arguments():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    for x in ('parent-source','compile-summary','assembly-review','main8b-library','library','codec-summary','out'):p.add_argument('--'+x,type=Path)
    for x in ('assembly-review-sha256','codec-proof-sha256'):p.add_argument('--'+x)
    return p

def main():
    p=arguments();a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k!='gpu'):p.error('Root-only: actual six-row mixed compile/review, accepted8b/codec, frozen3eb and fresh out required')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Actual device arguments require explicit --gpu')
    c.need(not {'torch','triton','numpy'}.intersection(sys.modules),'stdlib CPU default')
    print(json.dumps(c.plan(),sort_keys=True,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
