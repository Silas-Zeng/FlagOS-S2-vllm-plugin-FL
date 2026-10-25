"""Guarded standalone software FP8 codec conformance, no model/cache installation."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


def main(a):
    if not a.run_gpu:
        print(json.dumps({'gpu_executed':False,'finite_bf16_inputs':65280,'decode_codes':256,
                          'streams':['default','side'],'patterns':2,'graph_changed_inputs':True,
                          'decoders':['integer_bits','BF16_lookup'],'native_fp8_arithmetic':False}))
        return
    assert a.source and a.commit and a.out
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.commit
    assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    a.out.mkdir(exist_ok=False,parents=True)
    sys.path.insert(0,str(a.source/'tools'))
    import torch
    import triton
    from kernels.biv150_fp8_v_codec import encode_vectors,decode_vectors
    from biv150_fp8_v_reference import (decode_e4m3,encode_nearest_reference,
                                        bf16_bits_of_exact_value)
    torch.set_num_threads(2)
    finite=[bits for bits in range(65536) if ((bits>>7)&255)!=255]
    patterns=[finite,finite[127:]+finite[:127]]
    table_values=[decode_e4m3(code) for code in range(256)]
    table=torch.tensor(table_values,dtype=torch.bfloat16,device='cuda')
    result={'complete':False,'diagnostic_only':True,'source_commit':a.commit,
            'torch':str(torch.__version__),'triton':str(triton.__version__),
            'device':torch.cuda.get_device_name(),'records':[],
            'codec_sha256':hashlib.sha256((a.source/'tools/kernels/biv150_fp8_v_codec.py').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitation':'No attention, throughput, cache-capacity or105-question qualification'}
    def save():
        temp=a.out/'result.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(a.out/'result.json')
    def host(bits):return torch.tensor(bits,dtype=torch.uint16).view(torch.bfloat16)
    def expected(bits):
        codes=[encode_nearest_reference(v) for v in bits]
        values=[decode_e4m3(v) for v in codes]
        decoded=[bf16_bits_of_exact_value(v) for v in values]
        return torch.tensor(codes,dtype=torch.uint8),torch.tensor(decoded,dtype=torch.uint16)
    torch.cuda.synchronize();default=torch.cuda.default_stream()
    for stream_name in ['default','side']:
        stream=default if stream_name=='default' else torch.cuda.Stream()
        stream.wait_stream(default)
        # The second length includes a partial128-element block and canaries.
        for length in [65280,65279]:
            source=torch.empty(length,dtype=torch.bfloat16,device='cuda')
            encoded_storage=torch.full((length+16,),63,dtype=torch.uint8,device='cuda')
            bit_storage=torch.full((length+16,),123,dtype=torch.bfloat16,device='cuda')
            lut_storage=torch.full_like(bit_storage,123)
            encoded=encoded_storage[8:8+length];bit_output=bit_storage[8:8+length];lut_output=lut_storage[8:8+length]
            stream.wait_stream(default)
            row={'stream':stream_name,'length':length,'checks':[]}
            def call():
                encode_vectors[(triton.cdiv(length,128),)](source,encoded,length,BLOCK=128,num_warps=1,num_stages=1)
                decode_vectors[(triton.cdiv(length,128),)](encoded,bit_output,table,length,BLOCK=128,USE_LUT=False,num_warps=1,num_stages=1)
                decode_vectors[(triton.cdiv(length,128),)](encoded,lut_output,table,length,BLOCK=128,USE_LUT=True,num_warps=1,num_stages=1)
            def check(pattern,mode):
                stream.synchronize()
                gold_code,gold_bits=expected(patterns[pattern][:length])
                actual=encoded.cpu();bits=bit_output.cpu().view(torch.uint16);lookup=lut_output.cpu().view(torch.uint16)
                assert torch.equal(actual,gold_code),'encoded bytes differ from independent nearest-value oracle'
                assert torch.equal(bits,gold_bits),'integer decoder differs from mathematical format oracle'
                assert torch.equal(lookup,gold_bits),'lookup decoder differs from mathematical format oracle'
                assert torch.equal(encoded_storage[:8].cpu(),torch.full((8,),63,dtype=torch.uint8))
                assert torch.equal(encoded_storage[-8:].cpu(),torch.full((8,),63,dtype=torch.uint8))
                for storage in [bit_storage,lut_storage]:
                    assert torch.equal(storage[:8].cpu(),torch.full((8,),123,dtype=torch.bfloat16))
                    assert torch.equal(storage[-8:].cpu(),torch.full((8,),123,dtype=torch.bfloat16))
                row['checks'].append({'pattern':pattern,'mode':mode,'encoded_exact':True,'both_decoders_exact':True,'canaries':True,
                                      'encoded_sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
            with torch.cuda.stream(stream):
                for pattern in range(2):
                    source.copy_(host(patterns[pattern][:length]));call();check(pattern,'eager')
                capture=torch.cuda.Stream();capture.wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=capture):call()
                stream.wait_stream(capture)
                for pattern in [0,1,0,1]:
                    source.copy_(host(patterns[pattern][:length]));encoded.fill_(127);bit_output.fill_(float('nan'));lut_output.fill_(float('nan'))
                    graph.replay();check(pattern,'changed-poisoned-graph')
            default.wait_stream(stream)
            row['passed']=True;result['records'].append(row);save();print(json.dumps({'stream':stream_name,'length':length,'passed':True}),flush=True)
            del graph,source,encoded,bit_output,lut_output,encoded_storage,bit_storage,lut_storage
        with torch.cuda.stream(stream):
            codes=torch.arange(256,dtype=torch.int32,device='cuda').to(torch.uint8)
            output=torch.empty(256,dtype=torch.bfloat16,device='cuda')
            for use_lut in [False,True]:
                decode_vectors[(2,)](codes,output,table,256,BLOCK=128,USE_LUT=use_lut,num_warps=1,num_stages=1)
                stream.synchronize();actual=output.cpu()
                for code,value in enumerate(table_values):
                    if code&127==127:assert torch.isnan(actual[code])
                    else:assert int(actual.view(torch.uint16)[code])==bf16_bits_of_exact_value(value)
        result.setdefault('all256_decode_checks',[]).append({'stream':stream_name,'both_decoders':True})
    result['complete']=True;result['passed']=True;save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--source',type=Path);p.add_argument('--commit');p.add_argument('--out',type=Path)
    main(p.parse_args())
