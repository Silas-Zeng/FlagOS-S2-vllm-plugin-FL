#!/usr/bin/env python3
"""CPU preflight by default; root opt-in adds one fixed candidate ir_override to d6 warmup."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import bench_biv150_konly_prefill_compile as parent
import biv150_konly_prefill_compile_contract as contract
import biv150_konly_q32_i8_hoist as hoist

OVERRIDE_SHA='ef347cd98b1603f9deec35b9ac3e0407efde022e399972b14ca5a60a4226891e'
PARENT_PINS={
    'bench_biv150_konly_prefill_compile.py':'a9f0a6462717017e8180363cfa3c78dc3a246091cc03abb507bbec771eaf56be',
    'biv150_konly_prefill_compile_contract.py':'4d3b256bb270963cd5848aafc255fdbd6d0a946d44df9570c6e3ae7ba94c4fd4',
}


def preflight(profile='q32'):
    if profile!='q32': raise ValueError('Only the witnessed candidate Q32 override is admitted')
    for name,digest in PARENT_PINS.items():
        if hoist.sha((parent.ROOT/'tools'/name).read_bytes())!=digest:
            raise ValueError('Original d6 driver/descriptor source changed')
    files,summary,record=hoist.original()
    changed,proof=hoist.transform(files['out/ir/candidate/actual.ttgir'].decode())
    if hoist.sha(changed)!=OVERRIDE_SHA: raise ValueError('CPU override bytes changed')
    p=contract.plan('q32');bundle=parent.source_bundle('q32')
    for name,value in [('baseline.py',bundle[1]),('candidate.py',bundle[2])]:
        if value.encode()!=files['out/sources/'+name]:
            raise ValueError('Source differs from actual d6 compile')
    if parent.freeze(p)!=summary['plan']:
        raise ValueError('d6 geometry/options/descriptor plan differs')
    return p,bundle,changed,proof,summary,record


class CandidateWarmup:
    """No launch interface. Forward unchanged kwargs plus the single fixed option."""
    def __init__(self,kernel,session): self.kernel,self.session=kernel,session
    def warmup(self,*args,**kwargs):
        s=self.session
        if not s.owned or not s.out.is_dir():
            raise ValueError('d6 must own a fresh admitted compile output first')
        if args or s.state['candidate_warmup_attempts']:
            raise ValueError('One keyword-only candidate warmup required')
        allowed=set(s.reference['arguments'])|set(s.reference['requested_options'])|{'grid'}
        if set(kwargs)!=allowed or 'ir_override' in kwargs:
            raise ValueError('Unknown/overridden candidate ABI or options')
        if tuple(kwargs['grid'])!=tuple(s.reference['grid']):
            raise ValueError('Candidate grid changed')
        for key,value in s.reference['requested_options'].items():
            if type(kwargs[key]) is not type(value) or kwargs[key]!=value:
                raise ValueError('d6 candidate warp/stage options changed')
        for key,value in s.reference['arguments'].items():
            if not isinstance(value,dict) and parent.freeze(kwargs[key])!=value:
                raise ValueError('d6 scalar argument changed: '+key)
        with s.override_path.open('x') as stream: stream.write(s.override)
        if hoist.sha(s.override_path.read_bytes())!=OVERRIDE_SHA:
            raise ValueError('Override file changed before compile')
        s.state['candidate_warmup_attempts']=1
        s.state['override_file']=dict(path=str(s.override_path),sha256=OVERRIDE_SHA)
        s.save()
        result=self.kernel.warmup(**dict(kwargs,ir_override=str(s.override_path)))
        if hoist.sha(s.override_path.read_bytes())!=OVERRIDE_SHA:
            raise ValueError('Override file changed during compile')
        s.state['candidate_warmup_returned']=True
        return result


class Session:
    def __init__(self,out,payload,commit=None):
        self.out=out;_,_,self.override,self.proof,self.summary,self.reference=payload
        self.owned=False
        self.override_path=(out/'fixed-q32-i8-operand-hoist.ttgir').resolve()
        self.state=dict(frontend_commit=commit,parent_driver_source_commit=hoist.DRIVER,
            reader_source_commit=hoist.READER,original_candidate_hash=hoist.KERNEL_HASH,
            original_debug_source_root='original d6 compile-v1 sources retained in fixed TTGIR locations',
            parent_source_pins=PARENT_PINS,CPU_transform=self.proof,
            candidate_warmup_attempts=0,candidate_warmup_returned=False,baseline_overridden=False,
            baseline_recorded=False,candidate_recorded=False,errors=[],
            actual_override_compile_completed=False,actual_I8_hoist_verified=False,
            override_application_verified=False,
            ASM_hoist_auditor_supported=False,runtime_binding_qualified=False,
            precision_gate=False,performance_verified=False,service_installed=False,
            ready_for_numeric_or_timing=False,kernel_launches=0)
    def save(self):
        if self.owned: parent.save(self.out/'override-frontend.json',self.state)
    @contextmanager
    def adapt(self):
        load,export,save=parent.module,parent.preserve,parent.save
        def save_hook(path,value):
            if path==self.out/'compile-summary.json':
                if value.get('driver_commit')!=self.state['frontend_commit'] or value.get('compile_only') is not True:
                    raise ValueError('Unknown d6 output ownership')
                self.owned=True
            return save(path,value)
        def module(path,name):
            result=load(path,name)
            if path.name=='candidate.py':
                if name!='cjj_konly_compile_candidate' or hoist.sha(path.read_bytes())!=self.summary['source_files']['sources/candidate.py']['sha256']:
                    raise ValueError('Unknown candidate module/source')
                key=parent.factory.name('q32');setattr(result,key,CandidateWarmup(getattr(result,key),self))
            return result
        def preserve(kernel,directory,label,requested,grid,kwargs,tensors,descriptors):
            if label not in ('baseline','candidate'): raise ValueError('Unexpected non-Q32 role')
            actual_requested=dict(requested)
            if label=='candidate': actual_requested['ir_override']=str(self.override_path)
            record=export(kernel,directory,label,actual_requested,grid,kwargs,tensors,descriptors)
            original=next(x for x in self.summary['records'] if x['label']==label)
            if record['typed_ABI']!=original['typed_ABI']:
                raise ValueError('Actual ABI/constants/attrs differ; exports retained for review')
            if record['files']['source']['sha256']!=original['files']['source']['sha256']:
                raise ValueError('Actual JIT source differs; exports retained for review')
            if label=='baseline':
                if requested!=original['requested_options'] or record['actual_options'].get('ir_override'):
                    raise ValueError('Baseline override/options changed')
                self.state['baseline_recorded']=True
            else:
                if not self.state['candidate_warmup_returned'] or requested!=original['requested_options']:
                    raise ValueError('Candidate warmup/options contract not witnessed')
                self.state.update(candidate_recorded=True,actual_override_compile_completed=record['complete_source_and_ir'],
                    actual_candidate_hash=record['hash'],actual_exported_TTGIR_sha256=record['files']['ttgir']['sha256'],
                    exported_TTGIR_exact_CPU_override=record['files']['ttgir']['sha256']==OVERRIDE_SHA,
                    actual_typed_ABI_matches_original=True,actual_JIT_source_matches_original=True)
            self.save()
            return record
        parent.module,parent.preserve,parent.save=module,preserve,save_hook
        try: yield
        finally: parent.module,parent.preserve,parent.save=load,export,save


def compile_only(args,payload):
    if not args.commit or args.out is None: raise ValueError('Explicit freeze/fresh output required')
    if args.out.exists(): raise ValueError('Fresh output required; existing evidence is never modified')
    p,bundle,*_=payload
    session=Session(args.out,payload,args.commit)
    try:
        with session.adapt(): result=parent.compile_only(args,p,bundle)
        if not session.state['candidate_recorded'] or not session.state['baseline_recorded']:
            raise ValueError('Expected independent baseline/candidate archives missing')
        return dict(original_d6_compile_summary=result,override_frontend=session.state)
    except Exception as error:
        session.state['errors'].append(dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        session.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('q32',),default='q32')
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--commit');parser.add_argument('--out',type=Path)
    args=parser.parse_args();payload=preflight(args.profile)
    if args.compile_only:
        if not args.commit or args.out is None: parser.error('Explicit compile requires exact --commit and fresh --out')
        result=compile_only(args,payload)
    else:
        if args.commit or args.out: parser.error('--commit/--out require explicit --compile-only')
        result=dict(cpu_only=True,plan=payload[0],CPU_transform=payload[3],override_sha256=OVERRIDE_SHA,
                    parent_source_pins=PARENT_PINS,parent_driver_source_commit=hoist.DRIVER,
                    kernel_launches=0,JIT_executed=False,actual_override_compile_completed=False,
                    actual_I8_hoist_verified=False,ASM_hoist_auditor_supported=False,
                    runtime_binding_qualified=False,precision_gate=False,performance_verified=False,
                    service_installed=False,ready_for_numeric_or_timing=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
