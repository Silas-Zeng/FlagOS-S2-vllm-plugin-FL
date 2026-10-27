"""Unmodified PR79 host IO/artifact helpers, extracted for the explicit micro only."""
import math
import hashlib
import json
import importlib.util
import sys
from biv150_konly_decode_numeric_contract import codec_vectors
from biv150_konly_decode_quant_reference import quantize_v2
GUARD = 64

def host_bits(tensor,torch):
    host=tensor.detach().cpu().contiguous()
    return host.view({1:torch.int8,2:torch.int16,4:torch.int32}[host.element_size()])


def host_bytes(tensor,torch):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def loaded(kernel,out,label):
    record,strings=preserve(kernel,out,label)
    assert all(kind in record['files'] and record['files'][kind]['bytes']>0 for kind in
               ('source','ttir','ttgir','llir','asm','cubin')),'incomplete actual compiler artifacts'
    record['phase']='post_launch'
    record['loaded_resources_known']=all(type(v) is int for v in
            (record['n_regs'],record['n_spills'],record['metadata']['shared']))
    return record,strings


def output_buffers(batch,segments,torch):
    result={};backings={}
    for key,shape,dtype in [('scratch',(batch,16,16,128),torch.float32),('maxima',(batch,16,16),torch.float32),
                            ('sums',(batch,16,16),torch.float32),('out',(batch,16,128),torch.bfloat16)]:
        n=math.prod(shape);full=torch.empty(n+2*GUARD,device='cuda',dtype=dtype)
        full[:GUARD].fill_(42.5);full[-GUARD:].fill_(42.5)
        tensor=full[GUARD:-GUARD].reshape(shape)
        result[key]=tensor if key=='out' else tensor.narrow(2,0,segments)
        backings[key]=(full,tensor)
    return result,backings


def poison_outputs(backings):
    for _,tensor in backings.values(): tensor.fill_(math.nan)


def output_check(c,backings,lengths,segments,torch):
    bits={};finite=True
    for key,(full,tensor) in backings.items():
        guard=torch.full((GUARD,),42.5,dtype=full.dtype)
        assert torch.equal(host_bits(full[:GUARD],torch),host_bits(guard,torch))
        assert torch.equal(host_bits(full[-GUARD:],torch),host_bits(guard,torch))
        host=tensor.detach().cpu();bits[key]=host_bits(c[key],torch)
        if key=='out':
            finite=finite and bool(torch.isfinite(host).all())
            for token,n in enumerate(lengths):
                if n==0: assert bool((host[token]==0).all()),'empty output not exactly zero'
        else:
            for token,n in enumerate(lengths):
                steps=(n+segments*16-1)//(segments*16)
                for segment in range(16):
                    active=segment<segments and n>0 and segment*steps*16<n
                    row=host[token,:,segment]
                    if active: finite=finite and bool(torch.isfinite(row).all())
                    else: assert bool(torch.isnan(row).all()),'inactive/unused scratch overwritten'
    assert finite,'nonfinite valid scratch or output'
    return bits,c['out'].detach().cpu().clone()


def preserve(kernel,out,label):
    target=out/label/str(kernel.hash);target.mkdir(parents=True,exist_ok=True);files={};strings={}
    for name,value in kernel.asm.items():
        if not isinstance(value,(str,bytes)): continue
        raw=value.encode() if isinstance(value,str) else value;path=target/('actual.'+name);path.write_bytes(raw)
        files[name]=dict(path=str(path.relative_to(out)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(value,str): strings[name]=value
    md=kernel.metadata
    return dict(phase='pre_launch',hash=str(kernel.hash),n_regs=getattr(kernel,'n_regs',None),
                n_spills=getattr(kernel,'n_spills',None),loaded_resources_known=False,
                metadata={k:getattr(md,k,None) for k in ('shared','num_warps','num_stages','warp_size','use_sme')},
                files=files),strings


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);loaded=importlib.util.module_from_spec(spec)
    sys.modules[name]=loaded;spec.loader.exec_module(loaded);return loaded


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)



def m16_probe(probe, out, torch, np):
    vectors = codec_vectors()
    host = torch.tensor(vectors, dtype=torch.bfloat16)
    expected = [quantize_v2(row) for row in vectors]
    qc = torch.tensor([v[0] for v in expected], dtype=torch.int8)
    qs = torch.tensor([v[1] for v in expected], dtype=torch.float32)
    n = len(vectors)
    source = host.cuda()
    # Guard a whole accidentally unmasked M16 tile, including N141's last tile.
    cb = torch.full((n * 128 + 4096,), 85, dtype=torch.int8, device='cuda')
    sb = torch.full((n + 128,), -98765.5, dtype=torch.float32, device='cuda')
    codes, scales = cb[2048:-2048].view(n, 128), sb[64:-64]
    codes.fill_(-128)
    scales.fill_(math.nan)
    kernel = probe.probe_stock_m16[((n + 15) // 16,)](source, codes, scales, n, num_warps=1, num_stages=1)
    torch.cuda.synchronize()
    resource, _ = loaded(kernel, out, 'codec-M16')
    np.savez_compressed(out / 'codec-M16-witness.npz', source_bits=host.view(torch.int16).numpy().view(np.uint16),
                        actual_codes=codes.cpu().numpy(), actual_scale_bits=host_bits(scales, torch).numpy().view(np.uint32),
                        expected_codes=qc.numpy(), expected_scale_bits=host_bits(qs, torch).numpy().view(np.uint32))
    atomic(out / 'codec-M16-progress.json', dict(passed=False, resources=resource, witness='codec-M16-witness.npz'))
    assert torch.equal(codes.cpu(), qc) and torch.equal(host_bits(scales, torch), host_bits(qs, torch)), 'M16 codec bits mismatch'
    assert bool((cb[:2048].cpu() == 85).all() and (cb[-2048:].cpu() == 85).all())
    assert bool((sb[:64].cpu() == -98765.5).all() and (sb[-64:].cpu() == -98765.5).all())
    assert torch.equal(host_bits(source, torch), host_bits(host, torch))
    atomic(out / 'codec-M16-progress.json', dict(passed=True, resources=resource, witness='codec-M16-witness.npz'))
    return dict(passed=True, m=16, vectors=n, codes_and_scale_bits_exact=True, source_unchanged=True,
                canaries=True, resources=resource)
