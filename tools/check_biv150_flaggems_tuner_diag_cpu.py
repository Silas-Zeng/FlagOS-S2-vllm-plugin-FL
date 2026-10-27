#!/usr/bin/env python3
"""Pinned real code objects + fake imports, handles, driver and SQL; no frameworks."""
import argparse
import functools
import hashlib
import json
import math
import importlib.machinery
from unittest.mock import patch
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from enum import Enum
sys.path.insert(0,str(Path(__file__).resolve().parent/'diagnostics/flaggems_tuner'))
import cjj_flaggems_tuner_diag as d


def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['files']:
        b=(root/row['relative']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    def fake(name):
        m=types.ModuleType(name);m.__file__=str(root/(name.replace('.','/')+'.py'));return m,compile(Path(m.__file__).read_bytes(),m.__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize)
    lib,lc=fake('flag_gems.utils.libentry');auto,ac=fake('triton.runtime.autotuner');sql,sc=fake('flag_gems.utils.models.sql');protocol,pc=fake('triton.flagtune.runtime.benchmark_protocol')
    protocol.resolve_benchmarker=types.FunctionType(d.code_at(pc,'resolve_benchmarker'),dict(__name__=protocol.__name__))
    lib.resolve_benchmarker=protocol.resolve_benchmarker
    sys.modules[protocol.__name__]=protocol
    counts=dict(launch=0,callback=0,driver=0,reflection=0)
    class Opaque:
        def __eq__(self,other):raise AssertionError('Opaque equality evaluated')
    opaque=Opaque();returned=Opaque()
    def launch(*args,**kwargs):assert args[0] is opaque;counts['launch']+=1;return returned
    def callback(fn,quantiles):counts['callback']+=1;fn();return [.1,.1,.1]
    def driver_bench():counts['driver']+=1;return callback
    errors=types.ModuleType('triton.compiler.errors');errors.CompileTimeAssertionFailure=type('CompileTimeAssertionFailure',(Exception,),{});sys.modules[errors.__name__]=errors
    env=dict(__name__=auto.__name__,__package__='triton.runtime',driver=types.SimpleNamespace(active=types.SimpleNamespace(get_benchmarker=driver_bench)),knobs=types.SimpleNamespace(autotuning=types.SimpleNamespace(print=False,adjust_block_size=False)),OutOfResources=type('OutOfResources',(Exception,),{}),PTXASError=type('PTXASError',(Exception,),{}))
    bench=types.FunctionType(d.code_at(ac,'Autotuner','_bench'),env);getter=types.FunctionType(d.code_at(ac,'Autotuner','do_bench'),env)
    auto.Autotuner=type('Autotuner',(),dict(_bench=bench,do_bench=functools.cached_property(getter)))
    class Mode(str,Enum):NORMAL='normal';FORCE_POLICY='force_policy';EXHAUSTIVE_COLLECTION='exhaustive_collection'
    class Config:
        def __init__(self,b):self.kwargs=dict(BLOCK=b);self.pre_hook=None
        def all_kwargs(self):return dict(self.kwargs,num_warps=4)
    class Store(dict):
        pass
    bc=Store();pool=types.SimpleNamespace()
    class Pool:
        def __getitem__(self,k):return bc
    runenv=dict(__name__=lib.__name__,LibTunerRunMode=Mode,libcache=Pool(),time=time,math=math,os=os)
    run=types.FunctionType(d.code_at(lc,'LibTuner','run'),runenv)
    lib.LibTuner=type('LibTuner',(auto.Autotuner,),dict(run=run));runenv['LibTuner']=lib.LibTuner
    class AutoBase:
        classes={}
        def prepare(self,engine):assert engine is opaque;counts['reflection']+=1
    automap=types.SimpleNamespace(automap_base=lambda:AutoBase())
    dbfn=types.FunctionType(d.code_at(sc,'SQLPersistantModel','build_sql_model_by_db'),dict(__name__=sql.__name__,sqlalchemy=types.SimpleNamespace(ext=types.SimpleNamespace(automap=automap))))
    sql.SQLPersistantModel=type('SQLPersistantModel',(),dict(build_sql_model_by_db=staticmethod(dbfn)))
    with tempfile.TemporaryDirectory(prefix='cjj-gems-diag-cpu-') as tmp:
        mon=d.Monitor(tmp,write=False);mon.install_sql(sql);mon.install_tuner(lib,auto)
        original=lib.LibTuner.run;mon.install_tuner(lib,auto);assert lib.LibTuner.run is original
        t=lib.LibTuner();t._do_bench=None;t.nargs={'x':opaque};t.seen_tuned_metas={};t.fn=types.SimpleNamespace(run=launch);t.base_fn=types.SimpleNamespace(__name__='fake');t.pre_hook=lambda *a,**kw:None;t.post_hook=lambda *a,**kw:None
        assert counts['driver']==0 and 'do_bench' not in vars(t)
        a=Config(1);t._bench(opaque,config=a);t._bench(opaque,config=a)
        assert counts['driver']==1 and counts['callback']==1 and vars(t)['do_bench'] is callback
        assert sum(x['kind']=='bench_entry' for x in mon.events)==2 and sum(x['kind']=='benchmark_callback' for x in mon.events)==1
        previous=t.do_bench;t.do_bench=lambda *a,**kw:None;t.do_bench=previous;assert vars(t)['do_bench'] is callback
        t.do_bench=callback;assert vars(t)['do_bench'] is callback;del t.do_bench;assert 'do_bench' not in vars(t)
        assert sql.SQLPersistantModel.build_sql_model_by_db('fake-table',opaque) is None and counts['reflection']==1
        t.configs=[a,Config(2)];t.arg_names=['x'];t._run_mode=Mode.NORMAL;t.cache={};t.get_key=lambda _:('shape',);t.get_benchmark_key=lambda _:('raw','replay');t.prune_configs=lambda kw:t.configs;t.benchmark_table_name='fake_benchmark';t.config_table_name='fake_config'
        def policy(fn,configs,args,kwargs):
            timings={c:fn(c) for c in configs};return min(timings,key=timings.get),timings
        t.policy=policy
        assert t.run(opaque,grid=(1,)) is returned
        runevents=[x for x in mon.events if x['kind']=='tuner_run'];assert len(runevents)==1
        assert any(x['parent']==runevents[0]['id'] and x['kind']=='bench_entry' for x in mon.events)
        failure=ValueError('original sentinel')
        def raises(*a,**kw):raise failure
        try:mon.wrap('test_error',raises)(opaque,anything=opaque)
        except ValueError as e:assert e is failure
        else:raise AssertionError('Exception swallowed')
        mon.flush=lambda:(_ for _ in ()).throw(OSError('diagnostic writer failed'));mon.last_flush=0
        assert mon.wrap('test_recording_error',lambda *a,**kw:returned)(opaque) is returned
        assert mon.recording_errors and not mon.summary()['complete_bounded_trace']
        try:mon.wrap('error_during_failed_writer',raises)(opaque)
        except ValueError as e:assert e is failure
        else:raise AssertionError('Recording error masked original exception')
        mon.uninstall();assert lib.LibTuner.run is run and '_bench' not in lib.LibTuner.__dict__ and 'do_bench' not in lib.LibTuner.__dict__
        # Fake import loader: forwarding remains intact, hook runs only after
        # module execution; no real package or device import is involved.
        seen=[]
        class Loader:
            def create_module(self,spec):return None
            def exec_module(self,module):seen.append('exec')
            def get_source(self,name):return 'fake-source'
        spec=importlib.machinery.ModuleSpec(sql.__name__,Loader())
        follow=d.Monitor(tmp,write=False)
        with patch.object(importlib.machinery.PathFinder,'find_spec',return_value=spec):
            wrapped_spec=d.Finder(follow).find_spec(sql.__name__,[])
        assert not follow.installed
        wrapped_spec.loader.exec_module(sql);assert seen==['exec'] and sql.__name__ in follow.installed
        assert wrapped_spec.loader.get_source(sql.__name__)=='fake-source';follow.uninstall()

        bad=types.ModuleType(sql.__name__);wrong=Path(tmp)/'bad.py';wrong.write_text('# wrong\n');bad.__file__=str(wrong);bad.SQLPersistantModel=sql.SQLPersistantModel
        try:d.Monitor(tmp,write=False).install_sql(bad)
        except ValueError:pass
        else:raise AssertionError('Bad SHA accepted')
        other=types.FunctionType(run.__code__.replace(co_flags=run.__code__.co_flags^1),run.__globals__)
        try:d.require_code(other,run.__code__)
        except ValueError:pass
        else:raise AssertionError('Code field mutation accepted')
        cap=d.Monitor(tmp,limit=1,write=False);cap.wrap('x',lambda:None)();cap.wrap('x',lambda:None)();assert cap.dropped==1 and not cap.summary()['complete_bounded_trace']
    assert not {'torch','sqlalchemy','triton','flag_gems'}.intersection(sys.modules)
    return dict(source_files_verified=len(manifest['files']),pinned_full_module_code_fields=True,fake_imports_only=True,no_framework_or_device_imports=True,opaque_handle_identity=True,exception_identity=True,recording_error_preserves_return=True,lazy_driver_access=True,cached_property_assignment_delete_and_scoped_restore_preserved=True,bench_entries2_callback1=True,static_sql_reflection=True,fake_import_loader_order_and_forwarding=True,idempotent_install_uninstall=True,bad_source_and_code_refused=True,overflow_invalidates_trace=True,formal_score_eligible=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();print(json.dumps(check(a.source_root.resolve()),sort_keys=True,indent=2))
