"""Opt-in diagnostic compiler timing; install before JIT binders are initialized.

Import is inert. Records host compile-call intervals without touching CUDA or
reading tensors. A compile call can load a binary cache entry; its duration is
not necessarily code-generation time. Already-bound JIT instances are not
retrofitted. Observed benchmarks are diagnostics, not final acceptance scores.
"""
from __future__ import annotations
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import threading
import time

JIT_SHA='6e9ce85c877b84807d97191211e53fbc7547328dd22dc6b8d9185c7bafc67210'
COMPILER_SHA='f9646f23963160af7856e6f725df7af01c2bf0fc0ee98afeb7987f552507e5fd'
_HANDLE=None


class Handle:
    def __init__(self,package,implementation,destination,provenance):
        self.package=package;self.implementation=implementation
        self.destination=Path(destination);self.destination.mkdir(parents=True,exist_ok=True)
        self.original=package.compile;self.errors=0;self.provenance=provenance
        original=self.original
        @functools.wraps(original)
        def wrapper(*args,**kwargs):
            begin=time.monotonic_ns();failed=None
            try:return original(*args,**kwargs)
            except BaseException as error:
                failed=type(error).__name__;raise
            finally:
                end=time.monotonic_ns()
                try:
                    source=args[0] if args else kwargs.get('src')
                    name=getattr(source,'name',None)
                    if name is None:name=getattr(getattr(source,'fn',None),'__name__',None)
                    options=kwargs.get('options') or {}
                    self.emit({'event':'compile_call','pid':os.getpid(),'thread':threading.get_ident(),
                               'start_ns':begin,'end_ns':end,'elapsed_ns':end-begin,
                               'kernel':str(name or type(source).__name__)[:200],
                               'num_warps':options.get('num_warps'),'num_stages':options.get('num_stages'),
                               'error_type':failed})
                except Exception:
                    self.errors+=1
                    if self.errors<=3:
                        try:os.write(2,b'BIV150_COMPILE_OBSERVER_RECORD_ERROR\n')
                        except OSError:pass
        wrapper._biv150_compile_observer=self
        self.wrapper=wrapper
        # Validate aliases before either mutation.
        if package.compile is not implementation.compile:raise RuntimeError('Compiler aliases differ')
        package.compile=implementation.compile=wrapper
        try:
            self.emit({'event':'installation','pid':os.getpid(),'installed_ns':time.monotonic_ns(),
                       'provenance':provenance,'limitation':'Only compile calls bound after installation; includes cache loads, no GPU timing.'})
        except BaseException:
            package.compile=implementation.compile=original
            raise

    def emit(self,event):
        data=(json.dumps(event,separators=(',',':'))+'\n').encode()
        if len(data)>4096:raise ValueError('Compile event exceeds atomic record budget')
        fd=os.open(self.destination/f'compile-{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            if os.write(fd,data)!=len(data):raise OSError('Short event write')
        finally:os.close(fd)

    def uninstall(self):
        restored=[]
        for module in (self.package,self.implementation):
            if module.compile is self.wrapper:module.compile=self.original;restored.append(True)
            else:restored.append(False)
        return {'aliases_restored':restored,'record_errors':self.errors}


def install(destination):
    global _HANDLE
    if _HANDLE is not None:return _HANDLE
    package=importlib.import_module('triton.compiler')
    implementation=importlib.import_module('triton.compiler.compiler')
    jit=importlib.import_module('triton.runtime.jit')
    paths={'jit':Path(jit.__file__),'compiler':Path(inspect.getsourcefile(package.compile))}
    actual={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    if actual!={'jit':JIT_SHA,'compiler':COMPILER_SHA}:raise RuntimeError({'unsupported_compiler_source':actual})
    _HANDLE=Handle(package,implementation,destination,actual)
    return _HANDLE



def verify_worker_coverage(directory,server_log):
    import re
    workers={int(pid) for pid in re.findall(r'EngineCore[^)\n]*?pid=(\d+)',server_log)}
    if not workers:raise ValueError('No actual EngineCore PID evidence')
    records=[json.loads(line) for path in Path(directory).glob('compile-*.jsonl') for line in path.read_text().splitlines()]
    installed={r['pid'] for r in records if r['event']=='installation' and r.get('provenance')=={'jit':JIT_SHA,'compiler':COMPILER_SHA}}
    if not workers<=installed:raise ValueError({'unobserved_workers':sorted(workers-installed)})
    positive={r['pid'] for r in records if r['event']=='compile_call' and '_small_o_kernel' in r.get('kernel','')}
    if not workers<=positive:raise ValueError({'missing_small_kernel_positive_control':sorted(workers-positive)})
    return {'engine_pids':sorted(workers),'early_installation_and_small_kernel_positive_control':True,
            'limitation':'Positive control does not quantify every background alias; compare new metadata coverage too.'}


def summarize(directory,windows):
    events=[];installations=[]
    for path in sorted(Path(directory).glob('compile-*.jsonl')):
        for line in path.read_text().splitlines():
            value=json.loads(line)
            if value['event']=='compile_call':
                if not (value['end_ns']>=value['start_ns'] and value['elapsed_ns']==value['end_ns']-value['start_ns']):raise ValueError('Invalid event interval')
                events.append(value)
            elif value['event']=='installation':installations.append(value)
    if not installations:raise ValueError('No installation evidence; zero events is not proof of no compilation')
    def union(intervals):
        total=0;end=None
        for begin,finish in sorted(intervals):
            if end is None or begin>end:total+=finish-begin;end=finish
            elif finish>end:total+=finish-end;end=finish
        return total
    result=[]
    for window in windows:
        start,finish=window['start_ns'],window['end_ns'];assert finish>start
        selected=[e for e in events if e['end_ns']>start and e['start_ns']<finish]
        clipped=[(max(start,e['start_ns']),min(finish,e['end_ns'])) for e in selected]
        names=sorted({e['kernel'] for e in selected})
        result.append({'label':window['label'],'duration_seconds':(finish-start)/1e9,
                       'compile_calls_overlapping':len(selected),'host_interval_union_seconds':union(clipped)/1e9,
                       'host_interval_sum_seconds':sum(b-a for a,b in clipped)/1e9,
                       'by_kernel':{name:{'calls':sum(e['kernel']==name for e in selected),
                           'host_interval_union_seconds':union([(max(start,e['start_ns']),min(finish,e['end_ns'])) for e in selected if e['kernel']==name])/1e9} for name in names}})
    return {'diagnostic_only':True,'installations':installations,'windows':result,
            'limits':['Host compile-call duration includes cache loads; not GPU kernel time.',
                      'Concurrent intervals use union; category unions may overlap.',
                      'Install before first bind; compare event coverage against new metadata.',
                      'Client-process windows include client startup/test requests/postprocessing; not exact official timer.',
                      'Host compile intervals can overlap other work; do not subtract them to invent recovered TPS.']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--events',type=Path,required=True);parser.add_argument('--windows',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.events,json.loads(args.windows.read_text())),indent=2)+'\n')
