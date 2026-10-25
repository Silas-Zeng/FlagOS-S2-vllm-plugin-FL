"""Source provenance only; emitted GPU instructions and accuracy are unverified."""
import ast
import hashlib
import json
from pathlib import Path
from biv150_original_topology_norm_quant import generated_source,REFERENCE_SHA256,ROOT

source=generated_source();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='original_topology_norm_quant')
assert len([n for n in ast.walk(fn) if isinstance(n,ast.For)])==2
assert len([n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='libdevice.rsqrt'])==1
assert not any(isinstance(n,ast.Call) and ast.unparse(n.func).startswith('tl.atomic') for n in ast.walk(fn))
assert 'num_warps=16' in source and 'num_stages=1' in source
assert 'WRITE_NORM=write_norm' in source
path=ROOT/'docs/evidence/live-carry-service-20261025/paired-codegen-v2/original-norm-cached-topology.json'
evidence=json.loads(path.read_text());record=evidence['records'][0]
assert record['source_sha256']==REFERENCE_SHA256
assert all(record['config'][key]==value for key,value in dict(XBLOCK=2,R0_BLOCK=2048,num_warps=16,num_stages=1).items())
assert record['same_directory_python_files']==1
assert evidence['cached_configuration_not_runtime_frequency'] and not evidence['l3_causality_proven']
print(json.dumps(dict(cpu_source_provenance_passed=True,original_body_roundtrip_identical=True,
    original_source_sha256=REFERENCE_SHA256,generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    gpu_checked=False,production_hook=False,precision_fix_proven=False)))
