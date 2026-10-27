#!/usr/bin/env python3
"""CPU contracts and recorded-IR replay. --arrays uses existing NumPy on CPU only."""
import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tarfile

from biv150_konly_decode_factory import build
from biv150_konly_decode_ir import audit
from biv150_konly_decode_numeric_contract import (append_bf16_reference,codec_vectors,metrics,reference_gates,timing_policy)
from biv150_konly_decode_numeric_oracle import bf16_from_bits
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_decode_reference import encode_reference,mapped_bytes
from biv150_konly_i8_qk_reference import bf16_bits,f32_bits

ROOT=Path(__file__).resolve().parents[1]
BOUNDARY='ae00f58629ad85850fec1b8b8d2da28757af5f8b'
HARD=('finite','baseline_reference','codec','cache_bytes','sources_unchanged','scratch_canaries','graph_eager','side_eager','actual_ir')


def reject(fn):
    try: fn()
    except (AssertionError,ValueError): return
    raise AssertionError('Expected rejection')


def validate_closure(text):
    tree=ast.parse(text)
    step=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step']
    assert len(step)==1
    def callee(call):
        func=call.func
        if isinstance(func,ast.Subscript): func=func.value
        return ast.unparse(func)
    names=[callee(n) for stmt in step[0].body for n in ast.walk(stmt) if isinstance(n,ast.Call)]
    for name in ('support.prepare_q','support.append_kv','main_call','reduce_call'): assert names.count(name)==1
    assert names.index('support.prepare_q')<names.index('support.append_kv')<names.index('main_call')<names.index('reduce_call')
    conditional=[n for n in step[0].body if isinstance(n,ast.If)]
    assert len(conditional)==1 and ast.unparse(conditional[0].test)=="label == 'preq'"
    assert any(isinstance(n,ast.Call) and callee(n)=='support.prepare_q' for n in ast.walk(conditional[0]))


def replay_compile(mode):
    directory=ROOT/'docs/evidence/konly-i8-qk-20261027'
    p=json.loads((directory/('decode-'+mode+'-compile-v1-provenance.json')).read_text())
    data=(directory/p['archive']).read_bytes();assert len(data)==p['bytes']<=2*1024*1024
    assert hashlib.sha256(data).hexdigest()==p['sha256']
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
        members=tar.getmembers();assert len({m.name for m in members})==len(members)==106
        assert all(m.isfile() for m in members)
        files={m.name:tar.extractfile(m).read() for m in members}
    manifest=files['download-manifest.json'];assert hashlib.sha256(manifest).hexdigest()==p['manifest_sha256']
    m=json.loads(manifest);assert len(m['files'])==105 and sum(f['bytes'] for f in m['files'])==p['raw_bytes']
    assert set(files)=={'download-manifest.json'}|{f['relative'] for f in m['files']}
    for f in m['files']:
        raw=files[f['relative']];assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256']
    result=json.loads(files['out/result.json'])
    rr=p['root_replay'];root_bytes=(directory/rr['path']).read_bytes()
    assert len(root_bytes)==rr['bytes'] and hashlib.sha256(root_bytes).hexdigest()==rr['sha256']
    assert result['source_commit']==BOUNDARY and result['compile_passed'] and not result['errors']
    assert not result['device_numeric_passed'] and not result['complete_cost_measured']
    generated=files['out/generated-producer.py'].decode()
    assert hashlib.sha256(generated.encode()).hexdigest()==result['proof']['generated_sha256']
    assert build((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),mode)[0]==generated
    roles=[]
    for role in result['records']:
        assert role['phase']=='pre_launch' and not role['loaded_resources_known']
        assert role['n_regs'] is None and role['n_spills'] is None
        hashes={}
        for kind,entry in role['files'].items():
            raw=files['out/'+entry['path']]
            assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
            hashes[kind]=entry['sha256']
        if role['label']=='candidate_producer':
            artifacts={k:files['out/'+role['files'][k]['path']].decode() for k in ('ttgir','llir','asm')}
            actual=audit(artifacts,generated);assert actual['passed']
        roles.append(dict(label=role['label'],hash=role['hash'],artifact_hashes=hashes,loaded_resources_known=False))
    assert len(roles)==7
    return dict(passed=True,mode=mode,raw_files_verified=105,raw_bytes_verified=p['raw_bytes'],roles=roles,
                actual_decoder_ir=actual,artifact_files_assumed_identical=False,numeric_passed=False,timing_passed=False)


def array_checks():
    import numpy as np
    from biv150_konly_decode_numeric_oracle import pack_arrays,quantize_bits_cpu
    bits=np.array([i for i in range(65536) if (i&0x7f80)!=0x7f80],dtype=np.uint16).reshape(-1,128)
    codes,scales,proof=quantize_bits_cpu(bits,np)
    for row in range(len(bits)):
        expected,scale=quantize_v2([bf16_from_bits(int(v)) for v in bits[row]])
        assert tuple(map(int,codes[row]))==expected and f32_bits(float(scales[row]))==f32_bits(scale)
    vectors=codec_vectors();vb=np.array([[bf16_bits(v) for v in row] for row in vectors],dtype=np.uint16)
    c,s,_=quantize_bits_cpu(vb,np)
    for i,row in enumerate(vectors):
        expected,scale=quantize_v2(row);assert tuple(map(int,c[i]))==expected and float(s[i])==scale
    from check_biv150_konly_i8_qk_cpu import fixture
    keys,values,_=fixture(2)
    kb=np.array([[[[bf16_bits(v) for v in row] for row in slot] for slot in page] for page in keys],dtype=np.uint16)
    vv=np.array([[[[bf16_bits(v) for v in row] for row in slot] for slot in page] for page in values],dtype=np.uint16)
    c,s,_=quantize_bits_cpu(kb.reshape(-1,128),np)
    actual=pack_arrays(c.reshape(3,16,2,128),s.reshape(3,16,2),vv,np)
    assert actual==encode_reference(keys,values)
    rng=np.random.default_rng(20261027)
    q=rng.integers(-127,128,(2,8,128),dtype=np.int64);k=rng.integers(-127,128,(17,2,128),dtype=np.int64)
    product=np.einsum('hgd,thd->hgt',q,k,dtype=np.int64,optimize=True)
    for head in range(2):
        for row in range(8):
            for col in range(17): assert int(product[head,row,col])==sum(int(a)*int(b) for a,b in zip(q[head,row],k[col,head]))
    return dict(passed=True,finite_bf16_patterns=bits.size,codec_probe_vectors=len(vectors),
                packed_bytes=3*16384,int64_dot_cells=2*8*17,numpy_version=np.__version__,
                rational_boundary_cells=proof['rational_boundary_cells'],torch_bridge_executed=False,gpu_executed=False)


def run(arrays=False):
    count=0
    for bits in range(65536):
        if (bits&0x7f80)==0x7f80: continue
        actual=bf16_from_bits(bits);expected=struct.unpack('<f',struct.pack('<I',bits<<16))[0]
        assert actual==expected and math.copysign(1,actual)==math.copysign(1,expected);count+=1
    for bits in (0x7f80,0x7fff,0xff80,0xffff): reject(lambda bits=bits:bf16_from_bits(bits))
    tiny=metrics([1e-300],[0.]);assert tiny['rms']>0 and tiny['max_abs']>0
    equal=reference_gates([1.,2.],[1.,2.],[1.,2.],[1.,2.]);assert equal['candidate_numeric_passed']
    failed=reference_gates([1.,2.],[1.1,2.],[1.,2.],[1.1,2.]);assert failed['baseline_hard_passed'] and not failed['candidate_numeric_passed']
    reject_count=0
    for mask in range(1<<len(HARD)):
        hard={name:bool(mask&(1<<i)) for i,name in enumerate(HARD)}
        if all(hard.values()):
            assert timing_policy(hard,True,False)['allowed']
            assert not timing_policy(hard,False,False)['allowed']
            assert timing_policy(hard,False,True)['diagnostic_only']
        else:
            reject(lambda hard=hard:timing_policy(hard,False,True));reject_count+=1
    runner=(ROOT/'tools/bench_biv150_konly_decode_numeric.py').read_text();validate_closure(runner)
    for old,new in [('support.append_kv[(batch,2)]','support.omitted_append[(batch,2)]'),
                    ('main_call(proxies[label]','omitted_main(proxies[label]'),
                    ('reduce_call(base,cases[label]','omitted_reduce(base,cases[label]'),
                    ("if label=='preq': compiled['qprep']","if label=='baseline': compiled['qprep']")]:
        assert old in runner;reject(lambda old=old,new=new:validate_closure(runner.replace(old,new,1)));reject_count+=1
    for mode in ('inline','preq'):
        result=json.loads(subprocess.check_output([sys.executable,str(ROOT/'tools/bench_biv150_konly_decode_numeric.py'),'--mode',mode],text=True))
        assert result['cpu_only'] and result['complete_numeric_runner_prepared'] and not result['gpu_executed']
        assert not result['torch_imported'] and not result['triton_imported']
    for path in ('tools/kernels/biv150_konly_codec_falsifier.py','tools/bench_biv150_konly_decode_numeric.py'):
        compile((ROOT/path).read_text(),path,'exec')
    frozen=['tools/bench_biv150_konly_decode.py','tools/biv150_konly_decode_factory.py','tools/biv150_konly_decode_ir.py',
            'tools/biv150_konly_decode_quant_reference.py','tools/biv150_konly_decode_reference.py',
            'tools/biv150_konly_decode_sdk.py','tools/check_biv150_konly_decode_cpu.py',
            'tools/kernels/biv150_konly_decode_support.py','vllm_fl','csrc']
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',BOUNDARY,'--',*frozen],check=True)
    replay={mode:replay_compile(mode) for mode in ('inline','preq')}
    result=dict(cpu_only=True,passed=True,decoder_source_boundary=BOUNDARY,finite_bf16_decode_patterns=count,
                hard_failure_and_cost_closure_rejections=reject_count,complete_closure_has_all_stages=True,
                tiny_rms_not_zero=True,compiled_actual_ir_replays=replay,array_checks=array_checks() if arrays else None,
                ae00_runtime_bytes_unchanged=True,numpy_imported='numpy' in sys.modules,torch_imported='torch' in sys.modules,triton_imported='triton' in sys.modules,
                actual_numeric_runner_executed=False,precision_gate=False,gpu_executed=False)
    assert not result['torch_imported'] and not result['triton_imported']
    assert arrays or not result['numpy_imported']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arrays',action='store_true');parser.add_argument('--out',type=Path);args=parser.parse_args()
    result=run(args.arrays)
    if args.out: args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','cpu_only','finite_bf16_decode_patterns','hard_failure_and_cost_closure_rejections','precision_gate')}))
