#!/usr/bin/env python3
"""Default stdlib CPU; explicit --run-gpu is a frozen component falsifier, no timing."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
from biv150_konly_strided_writer_contract import View, need, oracle, qualify
from biv150_konly_strided_writer_fixtures import GUARD, SPECS, make_case
from check_biv150_konly_strided_writer_cpu import KERNEL, ROOT, run

MAIN = '56f1838192abc940b3fb9accef490022ac125000'
BASE = '/usr/local/lib/python3.12/site-packages/triton/'
FRONTEND = {
    BASE+'spec/iluvatar/triton/runtime/jit.py': '66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775',
    BASE+'spec/iluvatar/triton/language/math.py': '474139317072193ad472f182967e09fa2b46c9b7a4115d4dc3bfba1960f25964',
    BASE+'language/__init__.py': 'ef5fdb926433bf34fb47c4fef12218e7ba9a96790c010a1f8f7b08bc2b7a0fa1',
}
EXPECTED_SIGNATURE = ['*bf16','*bf16','*i64','*i8','*fp32','*bf16']+['i64']*6
ARGUMENTS = ['key_source','value_source','slots','code_full','scale_full','value_full',
             'key_row_stride','value_row_stride','key_rows','value_rows','slot_count','cache_pages']


def signature_record(raw):
    need(type(raw) is dict and len(raw) == 12, 'complete typed signature')
    result = {}
    for key, value in raw.items():
        if type(key) is tuple and len(key) == 1:
            key = key[0]
        if type(key) is str and key in ARGUMENTS:
            key = ARGUMENTS.index(key)
        need(type(key) is int and 0 <= key < 12 and key not in result, 'signature parameter key')
        result[key] = str(value)
    ordered = [result[i] for i in range(12)]
    need(ordered == EXPECTED_SIGNATURE, 'actual typed signature differs: '+repr(ordered))
    return ordered


def signature_self_test():
    for raw in (dict(enumerate(EXPECTED_SIGNATURE)),
                {(i,):v for i,v in enumerate(EXPECTED_SIGNATURE)},
                dict(zip(ARGUMENTS,EXPECTED_SIGNATURE))):
        signature_record(raw)
    rejected=[]
    for name,raw in [('short',{i:v for i,v in enumerate(EXPECTED_SIGNATURE) if i != 11}),
                     ('I32_pitch',dict(enumerate(EXPECTED_SIGNATURE[:6]+['i32']+EXPECTED_SIGNATURE[7:]))),
                     ('BF16_code',dict(enumerate(EXPECTED_SIGNATURE[:3]+['*bf16']+EXPECTED_SIGNATURE[4:]))),
                     ('wrong_key',{**dict(enumerate(EXPECTED_SIGNATURE)),12:'i64'})]:
        try: signature_record(raw)
        except ValueError: rejected.append(name)
        else: raise AssertionError('accepted ABI fault '+name)
    return dict(accepted_key_representations=3,rejected=rejected,actual_device_abi_verified=False)


def atomic(path, value):
    tmp = path.with_suffix('.tmp');tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n');tmp.replace(path)


def preserve(kernel, out, phase):
    digest = str(kernel.hash)
    need(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'compiled hash')
    folder = out/digest;folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, value in kernel.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        need(re.fullmatch('[a-zA-Z0-9_]+', name) is not None, 'compiler artifact name')
        data = value.encode() if isinstance(value, str) else value
        path = folder/('actual.'+name);path.write_bytes(data)
        files[name] = dict(file=str(path.relative_to(out)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    need(all(k in files and files[k]['bytes'] > 0 for k in ('ttir','ttgir','llir','asm','cubin')), 'missing actual compiler artifacts')
    signature = signature_record(dict(kernel.src.signature))
    md = kernel.metadata
    metadata = {k:getattr(md,k,None) for k in ('num_warps','num_stages','warp_size','shared','use_sme')}
    need([metadata[k] for k in ('num_warps','num_stages','warp_size')] == [1,1,64], 'fixed launch geometry')
    regs, spills = getattr(kernel,'n_regs',None), getattr(kernel,'n_spills',None)
    known = all(type(v) is int and v >= 0 for v in (regs,spills,metadata['shared']))
    if phase == 'post_launch':
        need(known and regs > 0 and spills == 0, 'unknown/unsupported loaded resources')
    return dict(hash=digest, phase=phase, signature=signature, metadata=metadata,
                n_regs=regs,n_spills=spills,loaded_resources_verified=phase == 'post_launch' and known,
                files=files)


def describe(tensor):
    storage = tensor.untyped_storage()
    return View(str(tensor.dtype).removeprefix('torch.'), tuple(tensor.shape), tuple(tensor.stride()),
                int(storage.data_ptr()), int(storage.nbytes()), int(tensor.storage_offset())*tensor.element_size(), str(tensor.device))


def run_gpu(args, cpu):
    need(args.commit and args.out, 'explicit device mode requires --commit and fresh --out')
    need(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip() == args.commit, 'frozen commit mismatch')
    need(not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip(), 'dirty source checkout')
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',MAIN,'--','vllm_fl','csrc'],check=True)
    need(not args.out.exists(), 'fresh output required')
    args.out.mkdir(parents=True)
    result = dict(commit=args.commit,cpu=cpu,component_only=True,completed=False,passed=False,
                  errors=[],records=[],performance_measured=False,precision_gate=False,
                  model_or_service_verified=False,live_slot_lifecycle_verified=False)
    def save(): atomic(args.out/'result.json',result)
    save()
    try:
        for path, digest in FRONTEND.items():
            need(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'frontend source '+path)
        smi = subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        (args.out/'prelaunch-ixsmi.txt').write_text(smi)
        need('No running processes found' in smi, 'GPU busy; do not queue')
        with socket.socket() as s:
            need(s.connect_ex(('127.0.0.1',9031)) != 0, 'service9031 active')
        os.environ['TRITON_CACHE_DIR'] = str(args.out/'triton-cache')
        import torch
        import triton
        target = triton.runtime.driver.active.get_current_target()
        result['target'] = dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
        need((target.backend,target.arch,target.warp_size) == ('corex',71,64), 'wrong device target')
        result['modules'] = {name:dict(file=str(module.__file__),version=getattr(module,'__version__',None))
                             for name,module in [('torch',torch),('triton',triton)]}
        result['frontend_sha256'] = FRONTEND
        source = KERNEL.read_bytes();(args.out/'kernel-source.py').write_bytes(source)
        spec = importlib.util.spec_from_file_location('cjj_konly_strided_writer_component',KERNEL)
        module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
        need(torch.cuda.current_device() == 0, 'component requires device0')
        torch.set_num_threads(2)

        def device_bytes(data):
            return torch.tensor(list(data),device='cuda',dtype=torch.uint8)
        def host_bytes(t):
            return bytes(t.detach().cpu().contiguous().view(torch.uint8).tolist())
        def build(case):
            kb = device_bytes(case['buffers']['key'])
            vb = kb if case['shared_readonly_sources'] else device_bytes(case['buffers']['value'])
            key = torch.as_strided(kb.view(torch.bfloat16),case['key'].shape,case['key'].strides,case['key'].byte_offset//2)
            value = torch.as_strided(vb.view(torch.bfloat16),case['value'].shape,case['value'].strides,case['value'].byte_offset//2)
            count = len(case['snapshot'].values)
            sb = device_bytes(b'\x6d'*GUARD+struct.pack('<'+'q'*count,*case['snapshot'].values)+b'\x6d'*GUARD)
            slots = sb.view(torch.int64)[GUARD//8:GUARD//8+count]
            cb = device_bytes(b'\x5a'*GUARD+case['before']+b'\x5a'*GUARD)
            raw = cb[GUARD:-GUARD]
            return dict(kb=kb,vb=vb,key=key,value=value,sb=sb,slots=slots,cb=cb,raw=raw,
                        code=raw.view(torch.int8),scale=raw.view(torch.float32),vfull=raw.view(torch.bfloat16))

        for case_spec in SPECS:
            case = make_case(case_spec,0);folder=args.out/case['name'];folder.mkdir()
            tensors=build(case)
            plan=qualify(describe(tensors['key']),describe(tensors['value']),describe(tensors['slots']),
                         describe(tensors['raw']),case['pages'],case['snapshot'])
            need(len({tensors[k].data_ptr() for k in ('raw','code','scale','vfull')}) == 1, 'full-base pointer mismatch')
            args_raw = (tensors['key'],tensors['value'],tensors['slots'],tensors['code'],tensors['scale'],tensors['vfull'],
                        plan['key_row_stride'],plan['value_row_stride'],plan['key_rows'],plan['value_rows'],plan['slot_count'],plan['cache_pages'])
            def launch():
                return module.encode_append[plan['grid']](*args_raw,num_warps=1,num_stages=1)
            def load(current):
                # Fixture owns every source upload; no GPU-derived slot proof is assumed.
                qualify(describe(tensors['key']),describe(tensors['value']),describe(tensors['slots']),
                        describe(tensors['raw']),current['pages'],current['snapshot'])
                tensors['kb'].copy_(device_bytes(current['buffers']['key']))
                if not current['shared_readonly_sources']:
                    tensors['vb'].copy_(device_bytes(current['buffers']['value']))
                count=len(current['snapshot'].values)
                tensors['slots'].copy_(torch.tensor(current['snapshot'].values,device='cuda',dtype=torch.int64))
                tensors['raw'].copy_(device_bytes(current['before']))
            def check(current,label):
                torch.cuda.synchronize()
                expected=oracle(current['before'],current['snapshot'].values,current['logical_keys'],current['logical_values'])
                actual=host_bytes(tensors['raw'])
                need(actual == expected, label+' complete encoded cache bytes')
                need(host_bytes(tensors['cb']) == b'\x5a'*GUARD+expected+b'\x5a'*GUARD,label+' cache canaries')
                need(host_bytes(tensors['kb']) == current['buffers']['key'] and host_bytes(tensors['vb']) == current['buffers']['value'],label+' readonly source/padding bytes')
                expected_slots=b'\x6d'*GUARD+struct.pack('<'+'q'*len(current['snapshot'].values),*current['snapshot'].values)+b'\x6d'*GUARD
                need(host_bytes(tensors['sb']) == expected_slots,label+' slot snapshot/guards')
                prefix=folder/(label+'-')
                for name,data in [('expected',expected),('actual',actual),('key-source',current['buffers']['key']),
                                  ('value-source',current['buffers']['value']),('slots',expected_slots)]:
                    prefix.with_name(prefix.name+name+'.bin').write_bytes(data)
                return dict(label=label,phase=current['snapshot'].epoch,complete_cache_bytes=len(actual),
                            sha256=hashlib.sha256(actual).hexdigest(),all_bytes_exact=True,
                            source_slot_canaries_unchanged=True,slot_values=list(current['snapshot'].values))
            record=dict(name=case['name'],binding=plan,checks=[],launched=False,
                        actual_views={name:describe(tensors[name]).__dict__ for name in ('key','value','slots','raw')},
                        full_base_pointers={name:int(tensors[name].data_ptr()) for name in ('raw','code','scale','vfull')},
                        shared_readonly_sources=case['shared_readonly_sources'])
            result['records'].append(record);save()
            if not plan['launch']:
                record['checks'].append(check(case,'empty-no-launch'));save();continue
            compiled=module.encode_append.warmup(*args_raw,grid=plan['grid'],num_warps=1,num_stages=1)
            record['pre_launch']=preserve(compiled,folder/'compiler','pre_launch');save()
            kernel=launch();record['launched']=True
            record['checks'].append(check(case,'direct'))
            record['post_launch']=preserve(kernel,folder/'compiler','post_launch')
            need(record['pre_launch']['hash'] == record['post_launch']['hash'],'warm/loaded binary differs')
            if case['name'] in ('tail17_k384_v640','shared_qkv_2560'):
                # Capture warm known addresses, then changed data/slots and cache poison.
                load(case);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): launch()
                changed=make_case(case_spec,1);load(changed);torch.cuda.synchronize()
                graph.replay();record['checks'].append(check(changed,'changed-graph'))
                load(changed);ready=torch.cuda.Event();ready.record()
                side=torch.cuda.Stream();side.wait_event(ready)
                with torch.cuda.stream(side):
                    launch()
                    for t in tensors.values(): t.record_stream(side)
                side.synchronize();record['checks'].append(check(changed,'changed-side'))
            save()
        result.update(completed=True,passed=True,requested_writer_gates_passed=True,
                      numerical_scope='finite_K_codec_and_all_cache_bytes_only',
                      full_service_or_lifecycle_verified=False)
        save();return result
    except Exception as exc:
        result['errors'].append(repr(exc));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);args=p.parse_args()
    cpu=run()
    if args.run_gpu:
        result=run_gpu(args,cpu)
    else:
        result=dict(cpu=cpu,signature_self_test=signature_self_test(),gpu_executed=False,device_entry_prepared=True,performance_measured=False,
                    required_typed_signature=EXPECTED_SIGNATURE,launch_geometry=dict(num_warps=1,num_stages=1),
                    direct_case_count=len(SPECS),changed_graph_and_side_cases=['tail17_k384_v640','shared_qkv_2560'])
        if args.out:
            need(not args.out.exists(),'fresh CPU output');args.out.mkdir(parents=True);atomic(args.out/'cpu-plan.json',result)
    print(json.dumps(dict(passed=cpu['passed'],gpu_executed=bool(args.run_gpu),component_only=True,
                          precision_gate=False,performance_measured=False)))


if __name__ == '__main__': main()
