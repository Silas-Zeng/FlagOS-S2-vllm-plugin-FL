#!/usr/bin/env python3
"""Default CPU plan; Root-only --gpu reuses the exact compiled library, never compiles."""
import argparse
import json
from pathlib import Path
import struct
import sys
import traceback

import biv150_native_k8_codec_fixture as f


def gpu(args):
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('New empty output required')
    out.mkdir(parents=True,exist_ok=True)
    plan=f.plan();(out/'cpu-plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    fixtures=f.fixtures();refs={k:f.oracle(fixtures[k]) for k in ('A','B','C')}
    (out/'fixture-and-independent-oracle.json').write_text(json.dumps(dict(fixtures=fixtures,oracle=refs),sort_keys=True,indent=2)+'\n')
    result=dict(plan,completed=False,errors=[],phases=[],admission_rejections=[],eager_op_calls=0,capture_op_calls=0,graph_replays=0)
    try:
        library=args.library.resolve();assert library.is_file() and f.sha(library.read_bytes())==f.LIBRARY_SHA,'Existing hard-SHA library required'
        import torch
        result['torch']=dict(version=torch.__version__,file=torch.__file__)
        assert torch.__version__=='2.10.0','Use the actual compile environment'
        torch.ops.load_library(str(library));op=torch.ops.cjj_native_k8_qk_probe.codec
        result['library']=dict(path=str(library),sha256=f.sha(library.read_bytes()))
        result['schema']=str(op.default._schema)
        assert result['schema']=='cjj_native_k8_qk_probe::codec(Tensor input, Tensor(a!) codes, Tensor(b!) scales) -> ()'
        n=plan['rows'];device='cuda:0'
        # Integer CPU construction/view and whole-byte upload preserve every BF16
        # signed-zero/subnormal bit without a float conversion on the device.
        def cpu_input(key):
            raw=f.guarded_input(fixtures[key]);values=struct.unpack('<'+str(len(raw)//2)+'h',raw)
            return torch.tensor(values,dtype=torch.int16).view(torch.bfloat16).reshape(n+2,128)
        cpu={key:cpu_input(key) for key in ('A','B','C')}
        input_full=cpu['A'].to(device)
        code_full=torch.full((n+2,128),f.CODE_GUARD,dtype=torch.int8,device=device)
        scale_full=torch.empty(n+2,dtype=torch.float32,device=device)
        scale_full.view(torch.int32).fill_(f.SCALE_GUARD)
        inp=input_full.narrow(0,1,n);codes=code_full.narrow(0,1,n);scales=scale_full.narrow(0,1,n)
        result['binding']=dict(input_pointer=inp.data_ptr(),code_pointer=codes.data_ptr(),scale_pointer=scales.data_ptr(),
          input_shape=list(inp.shape),input_stride=list(inp.stride()),input_storage_offset=inp.storage_offset(),
          code_shape=list(codes.shape),scale_shape=list(scales.shape))
        def raw(tensor):
            # Full CPU copy first. No GPU advanced indexing, negative or positive.
            return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        def save(stage):
            arrays={name:raw(tensor) for name,tensor in [('input_full',input_full),('code_full',code_full),('scale_full',scale_full)]}
            for name,data in arrays.items():(out/(stage+'-'+name+'.bin')).write_bytes(data)
            return arrays
        def poison():
            codes.fill_(-128);scales.view(torch.int32).fill_(f.SCALE_POISON);torch.cuda.synchronize()
        def verify(stage,key):
            arrays=save(stage);checks,mismatches=f.full_check(fixtures[key],refs[key],arrays)
            record=dict(stage=stage,input_pattern=key,checks=checks,mismatches=mismatches)
            result['phases'].append(record);(out/(stage+'-checks.json')).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
            assert all(checks.values()),'Exact codec/full canary/readonly gate: '+json.dumps(record,sort_keys=True)
            return arrays
        poison();before=save('eagerA-before');op(inp,codes,scales);result['eager_op_calls']+=1;torch.cuda.synchronize();verify('eagerA-after','A')
        # All invalid calls fail at host admission before a CUDA launch. Their
        # readonly/output backing is byte-compared after the whole negative set.
        negatives=[('input_dtype',inp.view(torch.int16),codes,scales),('code_dtype',inp,codes.view(torch.uint8),scales),
          ('scale_dtype',inp,codes,scales.view(torch.int32)),('scale_extent',inp,codes,scale_full),
          ('code_alias',inp,inp.view(torch.int8).reshape(-1).narrow(0,0,n*128).reshape(n,128),scales),
          ('scale_alias',inp,codes,inp.view(torch.float32).reshape(-1).narrow(0,0,n))]
        readonly_before=save('admission-before')
        for name,bad_input,bad_code,bad_scale in negatives:
            try:op(bad_input,bad_code,bad_scale)
            except RuntimeError as exc:result['admission_rejections'].append(dict(case=name,error=str(exc)))
            else:raise AssertionError('Invalid codec admission accepted: '+name)
        torch.cuda.synchronize();assert save('admission-after')==readonly_before,'Rejected call changed storage'
        graph=torch.cuda.CUDAGraph();capture_stream=torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph,stream=capture_stream):op(inp,codes,scales)
        result['capture_op_calls']+=1;capture_stream.synchronize()
        # B changes codes AND scales. Poison after capture, then replay before
        # any fresh eager B invocation, preventing eager/capture false positives.
        input_full.copy_(cpu['B']);poison();pre=save('graphB-before')
        assert pre['input_full']==f.guarded_input(fixtures['B'])
        assert pre['code_full'][128:-128]==bytes([128])*(n*128)
        assert pre['scale_full'][4:-4]==struct.pack('<I',f.SCALE_POISON)*n
        graph.replay();result['graph_replays']+=1;torch.cuda.synchronize();graph_b=verify('graphB-after','B')
        poison();save('eagerB-before');op(inp,codes,scales);result['eager_op_calls']+=1;torch.cuda.synchronize();eager_b=verify('eagerB-after','B')
        assert eager_b==graph_b,'Changed-input graph and new eager differ'
        input_full.copy_(cpu['C']);poison();save('sideC-before')
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):op(inp,codes,scales)
        result['eager_op_calls']+=1;side.synchronize();verify('sideC-after','C')
        assert f.sha(library.read_bytes())==f.LIBRARY_SHA
        result.update(completed=True,actual_codec_verified=True,Graph_verified=True,side_stream_verified=True,
          full_guarded_storage_verified=True,readonly_verified=True,requested_codec_gates_passed=True)
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=f.sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='codec-summary.json']
    (out/'codec-summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['completed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--library',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if not args.gpu:
        if args.library or args.out:raise ValueError('--library/--out require explicit --gpu')
        result=f.plan();assert not {'torch','triton','numpy'}.intersection(sys.modules);print(json.dumps(result,sort_keys=True,indent=2));return 0
    if args.library is None or args.out is None:raise ValueError('--gpu needs existing --library and new --out')
    return gpu(args)

if __name__=='__main__':raise SystemExit(main())
