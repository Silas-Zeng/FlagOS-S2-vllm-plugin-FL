"""Opt-in diagnostic timing of the actual Iluvatar compiler function.

Uses per-code CPython entry/return monitoring; compiler aliases, code identities and already
bound JIT callables remain unchanged. Includes binary-cache loads. Import is
inert and observed benchmarks remain diagnostic, never final acceptance scores.
"""
from __future__ import annotations
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

JIT_SHA='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
COMPILER_SHA='a97f067023873acf2e348c28f61494e59170ca84c3ee2314f20e491d3e833dfd'
_HANDLE=None


class Handle:
    def __init__(self,package,implementation,destination,provenance):
        self.package=package;self.implementation=implementation
        self.destination=Path(destination);self.destination.mkdir(parents=True,exist_ok=True)
        self.original=package.compile;self.errors=0;self.provenance=provenance
        if package.compile is not implementation.compile:raise RuntimeError('Compiler aliases differ')
        self.code=self.original.__code__;self.local=threading.local()
        self.origin_pid=os.getpid();self.seen_pids={self.origin_pid}
        self.monitor=getattr(sys,'monitoring',None);self.tool_id=None
        if self.monitor is None:raise RuntimeError('CPython monitoring requires Python3.12+')
        # Reserve only unassigned slots, avoiding standard optimizer/profiler IDs.
        for slot in (4,3):
            if self.monitor.get_tool(slot) is None:
                try:self.monitor.use_tool_id(slot,'biv150_compile_observer')
                except ValueError:continue
                self.tool_id=slot;break
        if self.tool_id is None:raise RuntimeError('No free diagnostic monitoring slot')
        events=self.monitor.events
        self.events=(events.PY_START,events.PY_RETURN,events.PY_UNWIND)
        try:
            for event,callback in zip(self.events,(self._begin,self._returned,self._unwound)):
                self.monitor.register_callback(self.tool_id,event,callback)
            self.monitor.set_local_events(self.tool_id,self.code,self.events[0]|self.events[1])
            # CPython does not allow PY_UNWIND as a local event. Other exception
            # unwinds are filtered by exact code identity without reading frames.
            self.monitor.set_events(self.tool_id,self.events[2])
            self.emit_install(self.origin_pid)
        except BaseException:
            self.uninstall();raise

    def emit_install(self,pid):
        self.emit({'event':'installation','pid':pid,'inherited_from_pid':self.origin_pid if pid!=self.origin_pid else None,
                   'installed_ns':time.monotonic_ns(),'provenance':self.provenance,
                   'method':'Per-code entry/return plus filtered exception-unwind monitoring; compile callable unchanged',
                   'tool_id':self.tool_id,'limitation':'Calls starting after install only; includes cache loads, no GPU timing.'})

    def _stack(self):
        pid=os.getpid()
        if pid not in self.seen_pids:
            self.emit_install(pid);self.seen_pids.add(pid)
        if getattr(self.local,'pid',None)!=pid:
            self.local.pid=pid;self.local.stack=[]
        return self.local.stack

    def _record_error(self):
        self.errors+=1
        if self.errors<=3:
            try:os.write(2,b'BIV150_COMPILE_OBSERVER_RECORD_ERROR\n')
            except OSError:pass

    def _begin(self,code,offset):
        begin=time.monotonic_ns()
        try:
            stack=self._stack();frame=sys._getframe(1)
            while frame is not None and frame.f_code is not self.code:frame=frame.f_back
            if frame is None:raise RuntimeError('No actual monitored compiler frame')
            values=frame.f_locals;source=values.get('src');options=values.get('options') or {}
            name=getattr(source,'name',None)
            if name is None:name=getattr(getattr(source,'fn',None),'__name__',None)
            stack.append({'event':'compile_call','pid':os.getpid(),'thread':threading.get_ident(),
                          'start_ns':begin,'kernel':str(name or type(source).__name__)[:200],
                          'num_warps':options.get('num_warps'),'num_stages':options.get('num_stages')})
        except Exception:
            self._record_error()

    def _finish(self,error):
        end=time.monotonic_ns()
        try:
            stack=self._stack()
            if not stack:return  # In-flight call at installation/fork is not timed.
            value=stack.pop();value.update(end_ns=end,elapsed_ns=end-value['start_ns'],error_type=type(error).__name__ if error else None)
            self.emit(value)
        except Exception:self._record_error()

    def _returned(self,code,offset,value):
        self._finish(None)

    def _unwound(self,code,offset,error):
        if code is self.code:self._finish(error)

    def emit(self,event):
        data=(json.dumps(event,separators=(',',':'))+'\n').encode()
        if len(data)>4096:raise ValueError('Compile event exceeds atomic record budget')
        fd=os.open(self.destination/f'compile-{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            if os.write(fd,data)!=len(data):raise OSError('Short event write')
        finally:os.close(fd)

    def uninstall(self):
        if self.tool_id is not None:
            self.monitor.set_events(self.tool_id,0)
            self.monitor.set_local_events(self.tool_id,self.code,0)
            for event in self.events:self.monitor.register_callback(self.tool_id,event,None)
            self.monitor.free_tool_id(self.tool_id);self.tool_id=None
        return {'aliases_unchanged':[self.package.compile is self.original,self.implementation.compile is self.original],
                'record_errors':self.errors}


def install(destination):
    global _HANDLE
    if _HANDLE is not None:return _HANDLE
    package=importlib.import_module('triton.compiler')
    implementation=importlib.import_module('triton.compiler.compiler')
    jit=importlib.import_module('triton.runtime.jit')
    paths={'jit':Path(jit.__file__),'compiler':Path(inspect.getsourcefile(package.compile))}
    actual={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    if actual!={'jit':JIT_SHA,'compiler':COMPILER_SHA}:raise RuntimeError({'unsupported_actual_backend_source':actual})
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
            'limitation':'Positive control covers this compiler code object; other compiler implementations require separate coverage.'}


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
                      'Per-code monitoring preserves callable identity; compare event coverage against new metadata.',
                      'Client-process windows include client startup/test requests/postprocessing; not exact official timer.',
                      'Host compile intervals can overlap other work; do not subtract them to invent recovered TPS.']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--events',type=Path,required=True);parser.add_argument('--windows',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.events,json.loads(args.windows.read_text())),indent=2)+'\n')
