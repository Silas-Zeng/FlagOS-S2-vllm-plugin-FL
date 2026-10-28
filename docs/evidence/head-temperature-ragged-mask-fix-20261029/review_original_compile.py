#!/usr/bin/env python3
"""Local CPU verification of the original 18-file compile; outputs only a summary."""
import argparse,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();assert not a.out.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
manifest=(a.raw/'download-manifest.json').read_bytes()
assert manifest==(HERE/'original-download-manifest.json').read_bytes()
m=json.loads(manifest);seen=set()
for row in m['files']:
    relative=Path(row['relative']);assert not relative.is_absolute() and '..' not in relative.parts
    assert row['relative'] not in seen;seen.add(row['relative'])
    path=a.raw/relative;assert path.is_file() and not path.is_symlink()
    raw=path.read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
assert len(seen)==m['raw_files']==18 and sum(r['bytes'] for r in m['files'])==m['raw_bytes']==483723
assert m['original_owner']==dict(pid=79187,start_ticks=357210613,same_start=True,state='Z')
s=json.loads((a.raw/'out/compile-summary.json').read_text())
owner=json.loads(next((a.raw/'exports').glob('*-owner.json')).read_text())
assert s['source_commit']==owner['source_commit']=='b6066b0899df81605ca1a19867459954e88d8699'
assert owner['pid']==79187 and owner['start_ticks']==357210613
assert s['completed'] is True and s['errors']==[]
assert s['gpu_executed'] is True and s['CUDA_context_initialized'] is True
assert s['head_attention_writer_encoder_launches']==s['input_initializations']==0
assert '--compile-only' in owner['command'] and owner['command'][owner['command'].index('--row')+1]=='31'
bindings=json.loads((HERE/'original-source-bindings.json').read_text())
assert all(s['source_sha256'][r['original_path']]==r['sha256'] for r in bindings['source_files'])
assert s['source_sha256']['vllm_fl/patches/biv150_lm_head_temperature_kernel.py']=='4c800693a2ae134fc580382d4166091b4c48e96dc2b7838ab417da7094cb2537'
record=s['records'][0];assert len(s['records'])==1 and record['rows']==31
assert record['grid']==[2040,1] and record['resource_phase']=='prelaunch'
assert record['n_regs'] is None and record['n_spills'] is None and record['numerical_qualified'] is False
ir=(a.raw/'out/rows31.llir').read_text()
needles=['%49 = srem i32 %29, %4','%53 = getelementptr bfloat, ptr addrspace(1) %0, i64 %52',
         '%.pn846 = phi ptr addrspace(1) [ %53,','%106 = ptrtoint ptr addrspace(1) %.pn846 to i64',
         '%109 = insertelement <4 x i32> %108, i32 -1, i64 2',
         '@llvm.bi.sme.load.16x1b64.rowxfb16(i32 %60, <4 x i32> %110']
chain=[]
for needle in needles:
    hits=[(i+1,line.strip()) for i,line in enumerate(ir.splitlines()) if needle in line]
    assert len(hits)==1;chain.append(dict(line=hits[0][0],text=hits[0][1]))
report=dict(CPU_only=True,original_compile_complete=True,original_device_numeric_executed=False,
    original_CUDA_context_and_allocations=True,original_head_attention_writer_encoder_launches=0,
    source_commit=s['source_commit'],owner=m['original_owner'],raw_files=18,raw_bytes=483723,
    all_original_file_SHA_verified=True,manifest_sha256=sha(manifest),
    original_kernel_sha256=s['source_sha256']['vllm_fl/patches/biv150_lm_head_temperature_kernel.py'],
    original_archive_sha256='317cff6bf6d32306596490924634017729924470ddf2c5e0dc9511f955161ccb',
    actual_imports=s['actual_imports'],original_record=record,LLVM_chain=chain,
    input_SME_descriptor_unbounded_length_minus_one=True,
    conclusion='Compile-only static evidence exposes ragged row hint risk; it does not record an executed out-of-bounds load or numerical failure.',
    new_mask_compile_or_numeric_verified=False,stock_F_linear_ragged_baseline_boundary_verified=False,
    original_archive_shared=False,script_sha256=sha(Path(__file__).read_bytes()))
a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(sha(a.out.read_bytes()))
