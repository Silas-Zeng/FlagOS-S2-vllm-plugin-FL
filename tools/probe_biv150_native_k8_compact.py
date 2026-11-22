#!/usr/bin/env python3
"""Default CPU contract. Root may explicitly compile; no numeric/perf entry yet."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import biv150_native_k8_compact_factory as f


def build_module():
    path = f.ROOT/'tools/compile_biv150_native_k8_qk.py'
    spec = importlib.util.spec_from_file_location('cjj_compact_parent_compile',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert f.sha(path.read_bytes()) == f.sha(f.blob(f.PARENT,'tools/compile_biv150_native_k8_qk.py'))
    return module


def plan():
    parent = build_module()
    sources = [f.path(k) for k in ('qk.cpp','qk.cu','codec.cu')]
    return dict(component='compact12544 NativeK8 address-only; explicit component compile stage',
        source_transform=f.verify(), compile_adapter=f.compile_adapter()[1],
        private_source={str(p.relative_to(f.ROOT)):f.sha(p.read_bytes()) for p in sources+[parent.HEADER]},
        typed_ABI=dict(query='BF16 N64/S2 or N32/S4, 16 Q heads/D128; stride2048 or2560,128,1',
            key_code='I8 [P,16,2,128] stride12544,256,128,1, layer fullbase',
            scale_full='F32 [P*3136] contiguous, same fullbase; internal +1024F32 once',
            value_offset='BF16 [P,16,2,128] stride6272,256,128,1; fullbase+4352B once',
            lengths='I32 [N], owned CPU content; nonnegative, within table and physical pages',
            table='I32 [N,pitch], owned CPU content; physical IDs in compact allocation',
            partial='F32 width16 storage [N,16,16,128] narrowed to S2/S4, strides32768,2048,128,1',
            maximum_and_sum='F32 width16 storage [N,16,16] narrowed to S2/S4, strides256,16,1',
            final='same original reducer; not compiled/launched here'),
        explicit_cuda_flags=parent.CUDA_FLAGS, CPU_only=True, readonly_CPU_default=True,
        root_owned_exclusive_required_for_compile=True,
        attention_launches=0,encoder_launches=0,codec_launches=0,reducer_launches=0,input_initializations=0,
        actual_compile_verified=False,actual_SO_IR_SASS_verified=False,actual_epoch_binding=False,
        numeric_bitexact_verified=False,precision_gate=False,performance_verified=False,
        service_installed=False,capacity_reclaimed=False,model_verified=False,
        next_gate='Actual final SO instructions/IR and static resources, not standalone same-source only; '
                  'then original22e vs compact full O/M/L/final BITWISE on identical logical cache/Q/table/lengths; '
                  'later numerical frontend must enforce live compact format+epoch provenance before dispatch')


def inspect_SO(out, objdump):
    """Dump the final library itself. Standalone .s is a separate artifact."""
    summary = out/'compile-summary.json'
    original = summary.read_bytes()
    j = json.loads(original)
    library = Path(j['library']['path'])
    assert f.sha(library.read_bytes()) == j['library']['sha256']
    record = dict(compile_summary_sha256=f.sha(original), library=j['library'],
                  objdump=str(objdump), commands=[], errors=[], actual_SO_audit_passed=False,
                  static_resources_known=False, numeric_bitexact_verified=False,
                  attention_launches=0,encoder_launches=0,codec_launches=0,reducer_launches=0)
    try:
        assert objdump.is_file(), 'Explicit CoreX ixobjdump required'
        record['objdump_sha256'] = f.sha(objdump.read_bytes())
        for label in ('sass','IR'):
            command = [str(objdump),'--'+label,str(library)]
            p = subprocess.run(command,capture_output=True,check=False)
            for stream, data in (('stdout',p.stdout),('stderr',p.stderr)):
                (out/('actual-so-'+label+'.'+stream+'.txt')).write_bytes(data)
            record['commands'].append(dict(argv=command,exit_code=p.returncode))
            if p.returncode or not p.stdout:
                record['errors'].append('actual SO '+label+' export failed')
    except Exception as exc:
        record['errors'].append(repr(exc))
    record['completed'] = not record['errors']
    record['artifact_manifest'] = [dict(relative=p.name,bytes=p.stat().st_size,sha256=f.sha(p.read_bytes()))
        for p in sorted(out.glob('actual-so-*.txt'))]
    (out/'actual-so-inspection.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compile-only',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true')
    p.add_argument('--compiler',default='/usr/local/corex/bin/clang++')
    p.add_argument('--objdump',type=Path,default=Path('/usr/local/corex/bin/ixobjdump'))
    p.add_argument('--out',type=Path)
    a = p.parse_args()
    result = plan()
    if not a.compile_only:
        assert not a.out and not a.root_owned_exclusive
        assert not {'torch','triton','numpy'}.intersection(sys.modules)
        print(json.dumps(result,sort_keys=True,indent=2))
        return 0
    assert a.root_owned_exclusive and a.out and a.out.is_absolute(), 'Root exclusive coordination and absolute new output required'
    parent = build_module()
    result.update(CPU_only=False,requested='isolated compact address compile-only, zero kernels or input initialization')
    source,_ = f.compile_adapter()
    ns = dict(vars(parent),SOURCES=[f.path(k) for k in ('qk.cpp','qk.cu','codec.cu')])
    exec(compile(source,'<compact-exact-parent-compile-adapter>','exec'),ns)
    code = ns['compile_only'](a,result)
    if code:
        return code
    inspection = inspect_SO(a.out.resolve(),a.objdump.resolve())
    print(json.dumps(inspection,sort_keys=True,indent=2))
    return 0 if inspection['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
