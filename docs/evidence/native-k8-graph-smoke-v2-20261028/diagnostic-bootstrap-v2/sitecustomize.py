"""Opt-in only; never silently replace an existing sitecustomize."""
import importlib.machinery
from pathlib import Path
import sys
here=Path(__file__).resolve().parent
other=importlib.machinery.PathFinder.find_spec('sitecustomize',[p for p in sys.path if Path(p or '.').resolve()!=here])
if other is not None:raise SystemExit('Native graph diagnostic refuses existing sitecustomize: '+str(other.origin))
import os
if os.environ.get('CJJ_NATIVE_GRAPH_COVERAGE')=='1' and os.environ.get('VLLM_FL_BIV150_KONLY_NATIVE_DECODE')=='1':
    from cjj_native_graph_coverage_bootstrap import activate
    activate()
