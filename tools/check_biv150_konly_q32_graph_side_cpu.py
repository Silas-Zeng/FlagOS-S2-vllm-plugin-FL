#!/usr/bin/env python3
"""Small CPU host-contract checks; no Graph/device execution or framework imports."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import bench_biv150_konly_q32_graph_side as graph


def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Invalid private Graph/side boundary accepted')


def run():
    host,reference,proof=graph.transforms()
    assert graph.RESOURCE_GUARD not in host and 'original_numeric_gates_unchanged=True' in host
    assert ';length=int(cpu[\'lengths\'][0])' in reference
    table=list(range(384))+[772]*(8192-384);changed,length=graph.changed_metadata(table,6144)
    assert length==6143 and changed[:384]==[383]+list(range(383)) and changed[384:]==table[384:]
    assert len(set(changed[:384]))==384 and len(changed)==8192
    reject(lambda:graph.changed_metadata(table,6143));reject(lambda:graph.changed_metadata(table[:384],6144))
    for row in (0,1,16,2047): assert 0<length-2048+row+1<=length
    scope=graph.Scope(Path('/unused'))
    compiled=SimpleNamespace(hash='actual',n_regs=None,n_spills=3,metadata=SimpleNamespace(shared=33792,num_warps=16,num_stages=1,use_sme=229379,warp_size=64))
    scope.warmups['hoisted']=compiled;scope.observe('hoisted','eager0',compiled)
    assert scope.resources[0]['n_regs'] is None and scope.resources[0]['n_spills']==3
    assert not scope.resources[0]['zero_spill_required'] and scope.resources[0]['same_object_as_warmup']
    admission=SimpleNamespace(state='active',token='fresh',owner=dict(pid=7,start_ticks=11),roots=dict(commit='frozen'),native=701,
        identity=lambda:dict(pid=7,start_ticks=11),binding=lambda:dict(commit='frozen'),closed=lambda:True,query=lambda:[701])
    scope.admission=admission;scope.guard('hoisted','eager0')
    admission.query=lambda:[7];reject(lambda:scope.guard('hoisted','graph1'))
    admission.query=lambda:[701,702];reject(lambda:scope.guard('hoisted','graph1'))
    admission.query=lambda:[701];admission.token='stale';reject(lambda:scope.guard('hoisted','graph1'))
    with tempfile.TemporaryDirectory(prefix='q32-graph-cpu-') as directory:
        path=Path(directory);sentinel=path/'numeric-summary.json';sentinel.write_text('original')
        unowned=graph.Scope(path);unowned.save();assert not (path/'graph-side-summary.json').exists()
        reject(lambda:graph.run(SimpleNamespace(out=path)))
        assert sentinel.read_text()=='original'
    loader,inputs,reference_fn=graph.frozen.parent.module,graph.fixture.cpu_inputs,graph.numeric.references
    try:
        with scope.adapt(SimpleNamespace(),reference): raise RuntimeError('restore test')
    except RuntimeError: pass
    assert graph.frozen.parent.module is loader and graph.fixture.cpu_inputs is inputs and graph.numeric.references is reference_fn
    assert not any(name in sys.modules for name in ('torch','triton','numpy'))
    return dict(cpu_only=True,host_transform=proof,changed_metadata_tested=True,native_token_foreign_namespace_refused=True,
                existing_output_unmodified=True,private_hooks_restored=True,unknown_nonzero_resource_counts_observed_without_gate=True,
                runtime_lengths=[6144,6143],capacity_query_rows=2048,capacity_prior_context=4096,
                actual_Graph_verified=False,actual_side_stream_verified=False,actual_numeric_executed=False,
                precision_gate=False,timing_attempted=False,performance_verified=False,service_verified=False)


if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
