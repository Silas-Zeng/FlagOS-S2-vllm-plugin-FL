#!/usr/bin/env python3
"""Synthetic CPU forwarding/failure fixtures for the fixed override frontend."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import bench_biv150_konly_q32_hoist_compile as front
from check_biv150_konly_prefill_compile_cpu import EmptyOnlyTorch


def reject(call):
    try: call()
    except (ValueError,AssertionError): return
    raise AssertionError('Unknown/unsafe override fixture accepted')


class WarmupOnly:
    def __init__(self): self.calls=[]
    def warmup(self,**kwargs): self.calls.append(kwargs);return 'synthetic_CPU_compile_return'
    def __getitem__(self,grid): raise AssertionError('No launch interface allowed')


def run():
    payload=front.preflight();p,bundle,changed,proof,summary,reference=payload
    rejects=0
    reject(lambda:front.preflight('multi2'));rejects+=1
    reject(lambda:front.preflight('raw3d'));rejects+=1
    original_functions=(front.parent.module,front.parent.preserve,front.parent.save)
    with tempfile.TemporaryDirectory(prefix='cjj-q32-hoist-frontend-CPU-') as temporary:
        root=Path(temporary);out=root/'fresh';out.mkdir();(out/'sources').mkdir()
        for name,value in [('baseline.py',bundle[1]),('candidate.py',bundle[2])]:
            (out/'sources'/name).write_text(value)
        t,owners=front.contract.allocate(EmptyOnlyTorch(),p)
        values=front.contract.kwargs(p,t,True);kwargs=dict(values,grid=p['grid'],**p['requested_options'])
        session=front.Session(out,payload,'synthetic_CPU_frontend')
        raw=WarmupOnly();unadmitted=front.CandidateWarmup(raw,session)
        reject(lambda:unadmitted.warmup(**kwargs));rejects+=1
        assert not raw.calls
        baseline=WarmupOnly();candidate=WarmupOnly()
        def loader(path,name):
            return SimpleNamespace(**({'kernel_biv150_prefill_stride':baseline} if path.name=='baseline.py' else
                                      {front.parent.factory.name('q32'):candidate}))
        front.parent.module=loader
        try:
            with session.adapt():
                front.parent.save(out/'compile-summary.json',dict(driver_commit='synthetic_CPU_frontend',compile_only=True))
                b=front.parent.module(out/'sources/baseline.py','cjj_konly_compile_baseline')
                c=front.parent.module(out/'sources/candidate.py','cjj_konly_compile_candidate')
                assert b.kernel_biv150_prefill_stride is baseline
                base_kwargs=dict(front.contract.kwargs(p,t,False),grid=p['grid'],**p['requested_options'])
                b.kernel_biv150_prefill_stride.warmup(**base_kwargs)
                assert baseline.calls==[base_kwargs] and 'ir_override' not in baseline.calls[0]
                wrapper=getattr(c,front.parent.factory.name('q32'))
                for patch in ({'ir_override':'foreign.ttgir'},{'num_warps':8},{'num_stages':2},{'grid':(128,2)},
                              {'KV_QUANT_MODE':2},{'scale':1.0},{'extra_option':1}):
                    reject(lambda patch=patch:wrapper.warmup(**dict(kwargs,**patch)));rejects+=1
                assert wrapper.warmup(**kwargs)=='synthetic_CPU_compile_return'
                assert len(candidate.calls)==1
                observed=candidate.calls[0]
                assert set(observed)-set(kwargs)=={'ir_override'}
                assert all(observed[k] is v or observed[k]==v for k,v in kwargs.items())
                assert front.hoist.sha(Path(observed['ir_override']).read_bytes())==front.OVERRIDE_SHA
                reject(lambda:wrapper.warmup(**kwargs));rejects+=1
                assert not session.state['actual_I8_hoist_verified']
        finally:
            front.parent.module=original_functions[0]
        assert (front.parent.module,front.parent.preserve,front.parent.save)==original_functions
        exported=[]
        def exporter(kernel,directory,label,requested,grid,kwargs,tensors,descriptors):
            value=deepcopy(next(x for x in summary['records'] if x['label']==label))
            value['requested_options']=requested.copy();exported.append((label,requested.copy()))
            return value
        front.parent.preserve=exporter
        try:
            with session.adapt():
                front.parent.preserve(None,out,'baseline',p['requested_options'],p['grid'],{}, {},{})
                front.parent.preserve(None,out,'candidate',p['requested_options'],p['grid'],{}, {},{})
            assert exported[0]==('baseline',p['requested_options'])
            assert exported[1][1]==dict(p['requested_options'],ir_override=str(session.override_path))
            assert session.state['baseline_recorded'] and session.state['candidate_recorded']
            assert not session.state['exported_TTGIR_exact_CPU_override'] and not session.state['actual_I8_hoist_verified']
        finally: front.parent.preserve=original_functions[1]
        assert (front.parent.module,front.parent.preserve,front.parent.save)==original_functions
        # Existing output is rejected before a parent call and never overwritten.
        marker=out/'do-not-touch';marker.write_bytes(b'original evidence')
        before={str(x.relative_to(out)):x.read_bytes() for x in out.rglob('*') if x.is_file()}
        reject(lambda:front.compile_only(SimpleNamespace(commit='synthetic',out=out),payload));rejects+=1
        after={str(x.relative_to(out)):x.read_bytes() for x in out.rglob('*') if x.is_file()}
        assert before==after
        # A failed d6 invocation that never owned output cannot write a sidecar.
        other=root/'not-owned';s=front.Session(other,payload,'synthetic')
        with s.adapt():
            other.mkdir();(other/'foreign').write_text('foreign')
        s.save();assert sorted(x.name for x in other.iterdir())==['foreign']
        # Exceptions restore private host adapters, not framework/global JIT state.
        try:
            with s.adapt(): raise RuntimeError('synthetic_CPU_error')
        except RuntimeError: pass
        assert (front.parent.module,front.parent.preserve,front.parent.save)==original_functions
    data=json.loads(subprocess.check_output([sys.executable,front.__file__],text=True))
    assert data['cpu_only'] and not data['JIT_executed'] and data['override_sha256']==front.OVERRIDE_SHA
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    return dict(cpu_only=True,passed=True,rejected_fixtures=rejects,fixed_override_sha256=front.OVERRIDE_SHA,
        parent_driver_commit=front.hoist.DRIVER,parent_pins=front.PARENT_PINS,
        baseline_forwarding_unchanged=True,candidate_difference_only_ir_override=True,
        existing_output_unchanged=True,unowned_output_not_written=True,host_adapters_restored=True,
        actual_override_compile_completed=False,actual_I8_hoist_verified=False,ASM_hoist_auditor_supported=False,
        runtime_binding_qualified=False,precision_gate=False,performance_verified=False,service_installed=False,
        kernel_launches=0,JIT_executed=False,torch_imported=False,triton_imported=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path)
    args=parser.parse_args();text=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if args.out:
        with args.out.open('x') as out: out.write(text)
    print(text,end='')


if __name__=='__main__':main()
