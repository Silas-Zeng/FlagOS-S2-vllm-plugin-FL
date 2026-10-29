"""Exact additive transform of reviewed 558579 controller; never edits parent."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PARENT_SHA='558579df4ebd3ac359167a429e9b3c6905952509de5910ebf843162ba0346600'
OUTPUT='run_biv150_parent107_stats_service.py'
# scope=None selects whole module; named scopes use exact AST line extents.
PATCHES=[
 (None,'import threading\n','import threading\nimport biv150_runtime_stats_controller as stats_hook\n'),
 ('full_workload_only_throughput','def full_workload_only_throughput(out, benchmark, selected_case):','def full_workload_only_throughput(out, benchmark, selected_case, *, snapshot=None):'),
 ('full_workload_only_throughput',"        history = []\n        value = benchmark", "        history = []\n        if snapshot is not None: snapshot(f'warm-before-{length}')\n        value = benchmark"),
 ('full_workload_only_throughput','        validate_score(value, length, 1024, count)\n',"        validate_score(value, length, 1024, count)\n        if snapshot is not None: snapshot(f'warm-after-{length}')\n"),
 ('full_workload_only_throughput','        validate_score(scores[str(length)], length, 1024, count)\n',"        validate_score(scores[str(length)], length, 1024, count)\n        if snapshot is not None: snapshot(f'score-after-{length}')\n"),
 ('run','    warm_plan=warmup_plan_metadata(a)\n','    warm_plan=warmup_plan_metadata(a)\n    stats_plan=stats_hook.plan(a)\n'),
 ('run','    env = dict(os.environ)\n','    env = stats_hook.clean_environment(os.environ, source)\n'),
 ('run',"    shutil.copy2(__file__, out / 'executed-controller.py')\n", "    stats_session=stats_hook.Session(a,__file__,expected) if stats_plan['enabled'] else None\n    manifest['runtime_stats_snapshot']=stats_plan if stats_session is None else stats_session.public()\n    shutil.copy2(__file__, out / 'executed-controller.py')\n"),
 ('run',"    try:\n        m.idle()\n", "    try:\n        if stats_session is not None: env=stats_session.service_environment(env)\n        m.idle()\n"),
 ('run',"            namespace = dict(json=json, subprocess=OwnedBenchmarkAdapter(out, client, owned))\n", "            client=stats_hook.clean_environment(client,source)\n            namespace = dict(json=json, subprocess=OwnedBenchmarkAdapter(out, client, owned))\n"),
 ('run',"                scores = warm_function(out, namespace['benchmark'], a.throughput_case)\n", "                if stats_session is None:\n                    scores = warm_function(out, namespace['benchmark'], a.throughput_case)\n                else:\n                    scores = warm_function(out, namespace['benchmark'], a.throughput_case,\n                        snapshot=lambda phase: stats_session.capture(phase,owned,state['server_owner']))\n                    manifest['runtime_stats_snapshot']=stats_session.finish()\n"),
 (None,"    p.add_argument('--graph-coverage-bootstrap-dir',type=Path)\n", "    p.add_argument('--graph-coverage-bootstrap-dir',type=Path)\n    p.add_argument('--runtime-stats-bootstrap',action='store_true')\n"),
 (None,'    try: warmup_plan_metadata(a)\n','    try: warmup_plan_metadata(a); stats_hook.plan(a)\n'),
 (None,'        print(json.dumps(dict(cpu_plan=True, gpu_executed=False, fixed_service_CLI=True,\n','        print(json.dumps(dict(cpu_plan=True, gpu_executed=False, fixed_service_CLI=True, runtime_stats_snapshot=stats_hook.plan(a),\n'),
]


def replace(text,scope,old,new):
    if scope is None:start,end=0,len(text)
    else:
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==scope)
        lines=text.splitlines(keepends=True);start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]))
    block=text[start:end]
    if block.count(old)!=1:raise ValueError('Nonunique/missing exact controller seam: '+str(scope))
    return text[:start]+block.replace(old,new,1)+text[end:]


def build(raw):
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Parent controller SHA differs')
    text=raw.decode()
    for scope,old,new in PATCHES:text=replace(text,scope,old,new)
    compile(text,OUTPUT,'exec')
    return text.encode()


def inverse(raw):
    text=raw.decode()
    for scope,old,new in reversed(PATCHES):text=replace(text,scope,new,old)
    value=text.encode()
    if hashlib.sha256(value).hexdigest()!=PARENT_SHA:raise ValueError('Inverse does not recover exact reviewed parent')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    result=build(a.parent.read_bytes());inverse(result)
    if a.check:
        if a.output.read_bytes()!=result:raise ValueError('Generated controller differs')
    else:
        with a.output.open('xb') as stream:stream.write(result)
    print(json.dumps(dict(parent_sha256=PARENT_SHA,generated_sha256=hashlib.sha256(result).hexdigest(),inverse_exact=True,GPU_executed=False)))
