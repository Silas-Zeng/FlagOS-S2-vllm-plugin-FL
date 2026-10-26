"""Captured-IR inverse/SSA fault tests; no Torch, compiler, device or model work."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

from biv150_prefill_early_q_ir import advance_q_read, digest, private_manager_module


def main():
    root=Path(__file__).resolve().parents[1]
    data=root/'docs/evidence/prefill-early-q-20261027'
    raw=(data/'captured-stage2-q-hoisted.ttgir').read_bytes()
    assert digest(raw)=='5210527f9d84f0c6e0a0152d57a3dbce910fe6c6944cdbe9b12a4a967f0c77bd'
    had_torch='torch' in sys.modules
    output,proof=advance_q_read(raw)
    assert proof['allocation_line']==255 and proof['old_read_line']==341
    assert proof['new_read_line']==256 and proof['main_loop_line']==342
    assert digest(output)=='a215cd327d65db0f777ecf08ef7c140e4a4750dddb9478be884ddb65c09b0483'
    assert len(raw)==len(output)
    original=raw.decode().splitlines(keepends=True)
    alloc=proof['allocation_line']-1;load=proof['old_read_line']-1
    loop=proof['main_loop_line']-1
    failures=[]
    def reject(name,mutated,rows=256):
        try:advance_q_read(mutated,rows)
        except (ValueError,UnicodeError):failures.append(name)
        else:raise AssertionError('Accepted '+name)
    def inserted(index,text):
        value=original.copy();value.insert(index,text);return ''.join(value).encode()
    reject('ambiguous allocation',inserted(alloc+1,original[alloc]))
    reject('duplicate shared read',inserted(load,original[load]))
    reject('descriptor alias/deallocation',inserted(load+1,'    ttg.local_dealloc '+proof['descriptor']+'\n'))
    reject('explicit barrier',inserted(alloc+1,'    gpu.barrier\n'))
    reject('explicit async wait',inserted(alloc+1,'    %extra = ttg.async_wait %token {num = 0 : i32}\n'))
    reject('unknown store',inserted(alloc+1,'    ttg.local_store %other, %value\n'))
    reject('unknown region',inserted(alloc+1,'    scf.if %unknown { }\n'))
    value=original.copy();value[load]=value[load].replace(' loc(', ', %foreign loc(')
    reject('extra SSA read operand',''.join(value).encode())
    value=original.copy();value[load]=value[load].replace('opIdx = 0','opIdx = 1')
    reject('wrong dot operand',''.join(value).encode())
    value=original.copy();value[alloc-1]=value[alloc-1].replace('tt.load','unknown.producer')
    reject('unrecognized producer',''.join(value).encode())
    reject('different buffering structure',raw.replace(b'memdesc<1x128x64xbf16',b'memdesc<2x128x64xbf16'))
    reject('unsupported row tile',raw,128)
    reject('non UTF8',b'\xff'+raw)
    reject('already early Q',output)
    # Same legal SSA structure with different compiler-generated symbol names.
    renamed=re.sub(r'%[A-Za-z0-9_]+',lambda m:'%renamed_'+m[0][1:],raw.decode()).encode()
    _,renamed_proof=advance_q_read(renamed)
    assert renamed_proof['exact_inverse'] and renamed_proof['other_lines_byte_identical']
    # This normal-loop fixture is synthetic and used only to check wiring:
    # applying the unmodified production transform must reproduce the exact
    # captured override. It is NOT claimed to be the captured normal TTGIR.
    value=original.copy();qread=value.pop(load)
    current_loop=next(i for i,line in enumerate(value) if '= scf.for ' in line)
    value.insert(current_loop+1,'  '+qread);synthetic=''.join(value).encode()
    path=root/'vllm_fl/patches/biv150_q_fragment_service.py'
    spec=importlib.util.spec_from_file_location('_cpu_original_qfragment',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original_transform=module.transform_ir
    assert original_transform(synthetic,256)==raw
    private,source_sha=private_manager_module(module)
    assert private.transform_ir(synthetic,256)==output
    assert module.transform_ir is original_transform and original_transform(synthetic,256)==raw
    assert private.Manager is not module.Manager
    assert private.Manager._compile.__code__.co_code==module.Manager._compile.__code__.co_code
    assert private.Manager._compile.__globals__['transform_ir'] is private.transform_ir
    assert module.Manager._compile.__globals__['transform_ir'] is original_transform
    with tempfile.TemporaryDirectory() as directory:
        altered=Path(directory)/'changed.py';altered.write_bytes(path.read_bytes()+b'\n')
        try:private_manager_module(SimpleNamespace(__file__=str(altered)))
        except RuntimeError:failures.append('changed production manager pin')
        else:raise AssertionError('Unpinned manager accepted')
    assert ('torch' in sys.modules)==had_torch
    result=dict(passed=True,gpu_executed=False,target_compiler_executed=False,
                actual_input_proof=proof,rejected_mutations=failures,
                renamed_ssa_inverse=True,synthetic_factory_wiring_only=True,
                production_manager_sha256=source_sha,baseline_globals_unchanged=True,
                proposed_output_sha256=digest(output),
                limits='No lowering/resource/shared-byte reduction, numerical, concurrency, model or speed proof.')
    (data/'cpu-contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
