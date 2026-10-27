"""Recorded installed frontend evidence; no inference about native division lowering."""
import ast
import hashlib
import json
from pathlib import Path

MATH_SHA='474139317072193ad472f182967e09fa2b46c9b7a4115d4dc3bfba1960f25964'
INIT_SHA='ef5fdb926433bf34fb47c4fef12218e7ba9a96790c010a1f8f7b08bc2b7a0fa1'
INSTALLED_MATH=Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/language/math.py')


def check_frontend(root):
    d=root/'docs/evidence/konly-i8-qk-20261027/div-rn-frontend'
    math=(d/'math.py.txt').read_bytes();init=(d/'top-language-init.py.txt').read_bytes()
    assert hashlib.sha256(math).hexdigest()==MATH_SHA and hashlib.sha256(init).hexdigest()==INIT_SHA
    fn=next(n for n in ast.parse(math).body if isinstance(n,ast.FunctionDef) and n.name=='div_rn')
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='_semantic.builder.create_precise_divf' for n in ast.walk(fn))
    assert any(isinstance(n,ast.ImportFrom) and n.module=='math' and any(a.name=='div_rn' for a in n.names)
               for n in ast.parse(init).body)
    manifest=json.loads((d/'target-manifest.json').read_text())
    assert manifest['installed_cpp_hits']==[]
    return dict(actual_frontend_present=True,math_sha256=MATH_SHA,top_init_sha256=INIT_SHA,
                fp32_precise_divf_builder_present=True,native_lowering_verified=False,
                installed_cpp_source_available=False,device_numeric_verified=False)
