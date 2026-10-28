"""Explicit worker-only import observer. Imports standard library only."""
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import traceback
from cjj_native_graph_coverage_probe_20261028 import PINS,Probe

_PROBE=None
_FINDER=None
_FAILED=False
_EXIT_DONE=False

def enabled():
    return os.environ.get('CJJ_NATIVE_GRAPH_COVERAGE')=='1' and os.environ.get('VLLM_FL_BIV150_KONLY_NATIVE_DECODE')=='1'

def emit(event):
    directory=Path(os.environ['CJJ_NATIVE_GRAPH_COVERAGE_OUT']);directory.mkdir(parents=True,exist_ok=True)
    visible=event
    if event.get('kind')=='native_capture_certificate':
        directory=Path(os.environ['CJJ_NATIVE_GRAPH_COVERAGE_OUT']);directory.mkdir(parents=True,exist_ok=True)
        raw=(json.dumps(event,sort_keys=True,indent=2)+'\n').encode();digest=hashlib.sha256(raw).hexdigest();path=directory/(digest+'.json')
        if not path.exists():path.write_bytes(raw)
        visible={key:value for key,value in event.items() if key!='native_payloads'}
        visible.update(full_certificate_file=str(path),full_certificate_sha256=digest,native_payload_sha256=[hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest() for x in event['native_payloads']])
    print('BIV150_NATIVE_GRAPH_COVERAGE',json.dumps(visible,sort_keys=True),flush=True,file=sys.stderr)

def flush_exit_summary():
    global _EXIT_DONE
    if _EXIT_DONE or _PROBE is None:return
    _EXIT_DONE=True
    try:
        value=dict(_PROBE.summary(),kind='worker_exit_summary',pid=os.getpid(),normal_exit_callback_observed=True,abrupt_exit_may_not_flush=True)
        directory=Path(os.environ['CJJ_NATIVE_GRAPH_COVERAGE_OUT']);directory.mkdir(parents=True,exist_ok=True)
        path=directory/('worker-summary-'+str(os.getpid())+'.json');temporary=path.with_suffix('.tmp')
        raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();temporary.write_bytes(raw);temporary.replace(path)
        emit(dict(kind='observer_exit_summary',pid=os.getpid(),summary_file=str(path),summary_sha256=hashlib.sha256(raw).hexdigest(),observer_error_count=value['observer_error_count'],overflow=value['overflow'],qualifying_replay_lower_bound=value['qualifying_replay_lower_bound'],recording_integrity=value['recording_integrity'],formal_score_eligible=False,quality_covered=False))
    except BaseException:
        print('BIV150_NATIVE_GRAPH_OBSERVER_SUMMARY_FAILED',traceback.format_exc(),flush=True,file=sys.stderr)


def maybe_install():
    global _PROBE,_FAILED
    if not enabled() or _PROBE is not None or _FAILED:return
    modules={name:sys.modules.get(name) for name in PINS}
    if any(m is None for m in modules.values()):return
    # A worker's ModelRunner module imports the Native adapter near its end;
    # its complete class is already defined. Do not act on partial classes.
    for name,attr in (('vllm_fl.worker.model_runner','ModelRunnerFL'),('vllm_fl.patches.biv150_konly_encoded_native','Adapter'),('vllm_fl.compilation.graph','GraphWrapper'),('vllm.v1.worker.gpu_input_batch','InputBatch')):
        if attr not in vars(modules[name]):return
    try:
        candidate=Probe(modules,emit=emit)
        instances=getattr(candidate.graph.GraphWrapper,'_all_instances',None)
        if instances is None:raise RuntimeError('Unknown graph instance registry')
        if any(w.runtime_mode is candidate.graph.CUDAGraphMode.FULL and any(e.graph is not None for e in w.concrete_graph_entries.values()) for w in list(instances)):
            raise RuntimeError('Graph observer must install before the first FULL capture')
        _PROBE=candidate.install()
        atexit.register(flush_exit_summary)
        # multiprocessing children normally run util finalizers before os._exit;
        # plain atexit alone does not cover that normal worker-exit path.
        from multiprocessing.util import Finalize
        Finalize(None,flush_exit_summary,exitpriority=0)
        # All source admission and original class wrapping has finished before
        # importlib returns the Native module to native.attach/first capture.
        print('BIV150_NATIVE_GRAPH_OBSERVER_READY',json.dumps(dict(pid=os.getpid(),source_pins=PINS,formal_score_eligible=False,quality_covered=False),sort_keys=True),flush=True,file=sys.stderr)
    except BaseException:
        _FAILED=True
        print('BIV150_NATIVE_GRAPH_OBSERVER_REJECTED',json.dumps(dict(pid=os.getpid(),traceback=traceback.format_exc(),formal_score_eligible=False,quality_covered=False),sort_keys=True),flush=True,file=sys.stderr)
        raise

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in PINS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None or spec.loader is None:return None
        delegate=spec.loader
        class Loader:
            def __getattr__(self,name):return getattr(delegate,name)
            def create_module(self,spec):return delegate.create_module(spec) if hasattr(delegate,'create_module') else None
            def exec_module(self,module):delegate.exec_module(module);maybe_install()
        spec.loader=Loader();return spec

def activate():
    global _FINDER
    if not enabled():return None
    if _FINDER is not None:return _FINDER
    if os.environ.get('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE')!='1' or os.environ.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')=='1':raise RuntimeError('Graph observer requires encoded16KiB Native ON')
    directory=Path(__file__).parent
    manifest=json.loads((directory/'bootstrap-sources.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Graph bootstrap source changed')
    _FINDER=Finder();sys.meta_path.insert(0,_FINDER)
    print('BIV150_NATIVE_GRAPH_OBSERVER_BOOTSTRAP',json.dumps(dict(pid=os.getpid(),waiting_for_loaded_sources=True,source_pins=PINS,formal_score_eligible=False,quality_covered=False),sort_keys=True),flush=True,file=sys.stderr)
    maybe_install();return _FINDER
