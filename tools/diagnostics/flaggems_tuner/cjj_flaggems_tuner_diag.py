"""Temporary opt-in diagnostics; stdlib only, no source edits or device calls."""
import atexit
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

PINS={
'flag_gems.utils.libentry':'414e8a3f74e2522e40659e7fccd84e301f4517220837a87af0311f050bbf3025',
'flag_gems.utils.models.sql':'6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7',
'triton.runtime.autotuner':'631052b97f5807f9179fa72f339c0aaef1caae8f6c63e07083b29fcdd54094ed'}
PROTOCOL_PINS={'triton.flagtune.runtime.benchmark_protocol':'0bf6ef0eaba3f50094c99b7e9573fa0a115c89f3ccade968cd40c12b9623c7ce','flag_gems.flagtune.runtime._benchmark_protocol':'15ea6ad8e120764c2ad529528aeec2e52e45792db69eb443ab5d75f05b26c0e5'}
FIELDS=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_code','co_names','co_varnames','co_freevars','co_cellvars','co_filename','co_firstlineno','co_name','co_qualname','co_linetable','co_exceptiontable')

def code_at(code,*names):
    for name in names:code=next(c for c in code.co_consts if isinstance(c,types.CodeType) and c.co_name==name)
    return code

def same_code(left,right):
    if any(getattr(left,k)!=getattr(right,k) for k in FIELDS) or len(left.co_consts)!=len(right.co_consts):return False
    return all(same_code(a,b) if isinstance(a,types.CodeType) and isinstance(b,types.CodeType) else type(a) is type(b) and a==b for a,b in zip(left.co_consts,right.co_consts))

def guarded(module,pin):
    path=Path(module.__file__);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Diagnostic source SHA mismatch: '+module.__name__)
    return path,compile(raw,str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)

def require_code(fn,code):
    if not isinstance(fn,types.FunctionType) or not same_code(fn.__code__,code):
        fields=[k for k in FIELDS if not isinstance(fn,types.FunctionType) or getattr(fn.__code__,k)!=getattr(code,k)]
        raise ValueError('Loaded diagnostic method differs from pinned full-module code: '+getattr(fn,'__qualname__','unknown')+'; fields='+repr(fields or ['co_consts']))

def safe(value):
    if value is None or type(value) in (str,int,float,bool):return value
    if isinstance(value,(tuple,list)):return [safe(x) for x in value[:64]]
    if isinstance(value,dict):return {str(k):safe(v) for k,v in list(value.items())[:64] if type(k) in (str,int)}
    return {'opaque_type':type(value).__module__+'.'+type(value).__qualname__}

class Monitor:
    def __init__(self,out,limit=8192,write=True):
        self.out=Path(out);self.limit=limit;self.write=write;self.restores=[];self.installed={};self.sources={};self.reset()
        if write:self.out.mkdir(parents=True,exist_ok=True);atexit.register(self.flush)
        if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=self.reset)
    def reset(self):
        self.pid=os.getpid();self.started=time.time_ns();self.lock=threading.Lock();self.flush_lock=threading.Lock();self.local=threading.local();self.events=[];self.count=0;self.dropped=0;self.recording_errors=[];self.last_flush=0
    def invoke(self,kind,fn,args,kwargs,meta=None):
        stack=getattr(self.local,'stack',[]);self.local.stack=stack
        with self.lock:
            eid=self.count;self.count+=1
        event=dict(id=eid,parent=stack[-1] if stack else None,kind=kind,pid=os.getpid(),thread=threading.get_native_id(),wall_start_ns=time.time_ns(),start_ns=time.perf_counter_ns(),metadata=safe(meta))
        stack.append(eid)
        try:return fn(*args,**kwargs)
        except BaseException as exc:event['exception_type']=type(exc).__module__+'.'+type(exc).__qualname__;raise
        finally:
            event['end_ns']=time.perf_counter_ns();event['duration_ns']=event['end_ns']-event['start_ns'];stack.pop()
            try:
                if kind=='tuner_run':
                    obj=args[0];event['after']=safe({name:vars(obj).get(name) for name in ('__name__','benchmark_success_count','benchmark_cache_hit_count','bench_time','config_table_name','benchmark_table_name','_benchmark_protocol','_run_mode','_flagtune_mode')})
                    config=vars(obj).get('best_config');event['best_config']=safe(vars(config)) if hasattr(config,'__dict__') else None
                with self.lock:
                    if len(self.events)<self.limit:self.events.append(event)
                    else:self.dropped+=1
                if not stack and time.monotonic()-self.last_flush>.5:self.flush()
            except BaseException as error:
                self.recording_errors.append(type(error).__module__+'.'+type(error).__qualname__)
    def summary(self):
        with self.lock:events=sorted(self.events,key=lambda x:x['id']);dropped=self.dropped
        return dict(python=sys.version,optimize=sys.flags.optimize,diagnostic_only=True,formal_score_eligible=False,pid=self.pid,started_wall_ns=self.started,sources=self.sources,installed=sorted(self.installed),dropped_events=dropped,recording_errors=list(self.recording_errors),required_hooks_installed=all(name in self.installed for name in ('flag_gems.utils.libentry','flag_gems.utils.models.sql')),complete_bounded_trace=dropped==0 and not self.recording_errors and all(name in self.installed for name in ('flag_gems.utils.libentry','flag_gems.utils.models.sql')),events=events,
          clocks='nested inclusive monotonic wall; do not sum parents and children',benchmark_callback_means='callback invoked, not an assertion of successful GPU measurement')
    def flush(self):
        self.last_flush=time.monotonic()
        if not self.write:return
        with self.flush_lock:
            path=self.out/(str(self.pid)+'-'+str(self.started)+'.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(self.summary(),sort_keys=True,indent=2)+'\n');os.replace(tmp,path)
    def wrap(self,kind,fn,metadata=None):
        @functools.wraps(fn)
        def wrapped(*args,**kwargs):
            try:meta=metadata(*args,**kwargs) if metadata else None
            except BaseException as error:self.recording_errors.append('metadata:'+type(error).__qualname__);meta=None
            return self.invoke(kind,fn,args,kwargs,meta)
        return wrapped
    def install_sql(self,module):
        if module.__name__ in self.installed:return
        path,code=guarded(module,PINS[module.__name__]);cls=module.SQLPersistantModel;original=cls.__dict__['build_sql_model_by_db']
        if not isinstance(original,staticmethod):raise ValueError('SQL staticmethod schema changed')
        require_code(original.__func__,code_at(code,'SQLPersistantModel','build_sql_model_by_db'))
        wrapped=staticmethod(self.wrap('sql_reflection',original.__func__,lambda name,engine:dict(physical_table=name)))
        cls.build_sql_model_by_db=wrapped;self.restores.append((cls,'build_sql_model_by_db',original));self.installed[module.__name__]=True;self.sources[module.__name__]=dict(path=str(path),sha256=PINS[module.__name__]);self.flush()
    def install_tuner(self,module,autotuner):
        if module.__name__ in self.installed:return
        path,code=guarded(module,PINS[module.__name__]);apath,acode=guarded(autotuner,PINS[autotuner.__name__]);cls=module.LibTuner;base=autotuner.Autotuner
        resolver=module.resolve_benchmarker;protocol_name=resolver.__module__
        if protocol_name not in PROTOCOL_PINS:raise ValueError('Unknown selected benchmark protocol module')
        protocol=sys.modules[protocol_name];ppath,pcode=guarded(protocol,PROTOCOL_PINS[protocol_name]);require_code(resolver,code_at(pcode,'resolve_benchmarker'))
        run=cls.__dict__['run'];bench=cls._bench;getter=base.__dict__['do_bench']
        require_code(run,code_at(code,'LibTuner','run'));require_code(bench,code_at(acode,'Autotuner','_bench'))
        if not isinstance(getter,functools.cached_property) or 'do_bench' in cls.__dict__:raise ValueError('Unexpected benchmark getter schema')
        require_code(getter.func,code_at(acode,'Autotuner','do_bench'))
        # A data descriptor delegates the original cached_property at the same
        # access point. Its dict cache, laziness, assignment and deletion stay intact.
        mon=self
        class Callback:
            def __get__(self,obj,owner=None):
                if obj is None:return self
                fn=getter.__get__(obj,owner)
                while isinstance(fn,types.FunctionType) and '_cjj_original_benchmark_callback' in vars(fn):fn=fn._cjj_original_benchmark_callback
                wrapped=mon.wrap('benchmark_callback',fn,lambda *a,**kw:dict(module=getattr(fn,'__module__',None),name=getattr(fn,'__qualname__',type(fn).__qualname__)))
                wrapped._cjj_original_benchmark_callback=fn;return wrapped
            def __set__(self,obj,value):
                while isinstance(value,types.FunctionType) and '_cjj_original_benchmark_callback' in vars(value):value=value._cjj_original_benchmark_callback
                obj.__dict__['do_bench']=value
            def __delete__(self,obj):del obj.__dict__['do_bench']
        cls.run=self.wrap('tuner_run',run);cls._bench=self.wrap('bench_entry',bench,lambda *a,**kw:dict(config=safe(vars(kw['config']))));cls.do_bench=Callback()
        self.restores.extend([(cls,'run',run),(cls,'_bench',None),(cls,'do_bench',None)])
        self.installed[module.__name__]=True
        cache=getattr(module,'libcache',None);url=getattr(cache,'db_url',None)
        self.sources['observed_SQLite_URL']=url if isinstance(url,str) and url.startswith('sqlite:///') else 'not SQLite/unknown; no URL credentials logged'
        triton_module=sys.modules.get('triton');self.sources['triton_version']=getattr(triton_module,'__version__',None)
        for name,p in ((module.__name__,path),(autotuner.__name__,apath)):self.sources[name]=dict(path=str(p),sha256=PINS[name])
        self.sources[protocol_name]=dict(path=str(ppath),sha256=PROTOCOL_PINS[protocol_name]);self.flush()
    def uninstall(self):
        for cls,name,original in reversed(self.restores):
            if original is None:delattr(cls,name)
            else:setattr(cls,name,original)
        self.restores.clear();self.installed.clear()

class Finder(importlib.abc.MetaPathFinder):
    def __init__(self,monitor):self.monitor=monitor
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in ('flag_gems.utils.libentry','flag_gems.utils.models.sql'):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None or spec.loader is None:return None
        delegate=spec.loader;mon=self.monitor
        class Loader:
            def __getattr__(self,name):return getattr(delegate,name)
            def create_module(self,spec):return delegate.create_module(spec) if hasattr(delegate,'create_module') else None
            def exec_module(self,module):
                delegate.exec_module(module)
                if fullname.endswith('.sql'):mon.install_sql(module)
                else:mon.install_tuner(module,sys.modules['triton.runtime.autotuner'])
        spec.loader=Loader();return spec

def activate(out):
    if os.environ.get('CJJ_FLAGGEMS_TUNER_DIAGNOSTIC')!='1':raise ValueError('Explicit diagnostic flag required')
    if getattr(sys,'_cjj_flaggems_tuner_diagnostic',None) is not None:return sys._cjj_flaggems_tuner_diagnostic
    monitor=Monitor(out);finder=Finder(monitor);sys.meta_path.insert(0,finder)
    sql=sys.modules.get('flag_gems.utils.models.sql');lib=sys.modules.get('flag_gems.utils.libentry')
    if sql is not None:monitor.install_sql(sql)
    if lib is not None:monitor.install_tuner(lib,sys.modules['triton.runtime.autotuner'])
    sys._cjj_flaggems_tuner_diagnostic=monitor;monitor.flush();return monitor

def plan():return dict(default='stdlib-only plan; no framework import',diagnostic_only=True,formal_score_eligible=False,source_pins=PINS,protocol_pins=PROTOCOL_PINS,hooks=['LibTuner.run','inherited _bench','lazy do_bench callback','SQL build_sql_model_by_db'],device_calls_added=0,cache_clear=False,source_files_changed=False,phase_assignment='join monotonic/wall timestamps to existing fullwarm/score commands',trace_limits='8192 events per process; overflow explicitly invalidates completeness')
