#!/usr/bin/env python3
"""Private max-q2 raw3D metadata/empty-segment extension; CPU by default."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.dont_write_bytecode=True

import bench_biv150_konly_allreader_numeric as frozen
import biv150_konly_prefill_reference as reference

PINS={'bench_biv150_konly_allreader_numeric.py':'054bf42f0915b36f7c8e168e78ed17cc57a2ae549752b8c6b4cbf046951fd464',
      'biv150_konly_allreader_numeric_contract.py':'5fea58c98fc505477eb7b541a9e15abff789bdb3c5d78fbcbd22f9b14f03f7e6',
      'biv150_konly_prefill_reference.py':'4ffcebd6fc666d839d01a0399b624d5d03e1b80f75a4e7917255cbf1ecb916c0',
      'biv150_konly_prefill_compile_contract.py':'4d3b256bb270963cd5848aafc255fdbd6d0a946d44df9570c6e3ae7ba94c4fd4'}


def plan(profile,case):
    if profile!='raw3d': raise ValueError('Only private raw3D max-q2 scope')
    p=frozen.contract.plan(profile,case)
    qlens=([1,2] if case==0 else [2,1])*32
    klens=([15,16,17,31,32,33,63,64] if case==0 else [17,2,33,16,65,32,2,3])*8
    p.update(qlens=qlens,klens=klens,total=sum(qlens),pages=2*sum((k+15)//16 for k in klens)+5,
             grid=reference.grid(profile,qlens,16),fixture='private64seq max-q2/token-indexed raw3D scope',raw_qlen2_private_scope=True)
    return p


def statuses(p):
    """Per query-token/segment:0 inactive,1 nonempty,2 causal-empty live tile.

    BM16/BQ2 and max-q2 put each sequence's live queries in one block. Every
    physically entered segment loops over >=1 tile because the last query sees
    the full sequence. A fully masked EARLY query row is processed by the pinned
    softmax helper: M=-inf→0, P=0, L=0, O=0. A zero-iteration M=-inf/L=1 case is
    excluded by this max-q2/full-block geometry, not treated as a finite fault.
    """
    if p['profile']!='raw3d' or p['config']!=dict(BLOCK_M=16,BLOCK_Q=2,TILE_SIZE=16,IS_3D=True) or p['segments']!=16 or max(p['qlens'])!=2:
        raise ValueError('Unknown raw segment/query geometry')
    result=[]
    for q,k in zip(p['qlens'],p['klens']):
        if q not in (1,2) or k<q: raise ValueError('Only positive max-q2 sequence scope')
        tiles=(k+255)//256
        for local in range(q):
            prefix=k-q+local+1;row=[]
            for seg in range(16):
                start=seg*tiles*16
                row.append(0 if start>=k else (1 if start<prefix else 2))
            result.append(row)
    if len(result)!=p['total']: raise ValueError('Token-indexed workspace rows changed')
    return result


def workspace_check(t,p,torch):
    kind=torch.tensor(statuses(p),device=t['sums'].device)
    nonempty=(kind==1)[:,None,:].expand(-1,16,-1)
    empty=(kind==2)[:,None,:].expand(-1,16,-1)
    inactive=(kind==0)[:,None,:].expand(-1,16,-1)
    good=True
    for name in ('scratch','maxima','sums'):
        value=t[name]
        good=good and bool(torch.isfinite(value[nonempty]).all()) and bool(torch.isnan(value[inactive]).all())
        # This is the helper-derived entered-but-causal-empty row, not a blanket
        # finite or -inf exemption for an unknown loop/mask configuration.
        good=good and bool((value[empty]==0).all())
        if name=='sums': good=good and bool((value[nonempty]>0).all())
    return good


def transforms():
    parent=frozen.parent
    for name,digest in PINS.items():
        if parent.factory.sha((parent.ROOT/'tools'/name).read_bytes())!=digest: raise ValueError('Frozen metadata/host input changed')
    old=inspect.getsource(reference.validate_binding)
    before='if is3d and max(query_lengths) > 1 or not is3d and max(query_lengths) <= 1:'
    after='if is3d and max(query_lengths) > 2 or not is3d and max(query_lengths) <= 1:'
    if old.count(before)!=1: raise ValueError('Unknown metadata max-q grammar')
    changed=old.replace(before,after,1)
    if changed.replace(after,before,1)!=old: raise ValueError('Metadata inverse failed')
    namespace=reference.__dict__.copy();exec(compile(changed,'<private raw max-q2 metadata>','exec'),namespace)
    desc=frozen.contract.descriptors;ns=desc.__dict__.copy();ns['validate_binding']=namespace['validate_binding']
    exec(compile(inspect.getsource(desc.check_descriptors),'<unchanged descriptor checks/private max-q2>','exec'),ns)
    descriptors=SimpleNamespace(**dict(desc.__dict__,check_descriptors=ns['check_descriptors']))
    private=SimpleNamespace(**dict(frozen.contract.__dict__,plan=plan,descriptors=descriptors))
    raw=inspect.getsource(frozen.run)
    anchor="    binding();args.out.mkdir(parents=True,exist_ok=False)\n"
    added=anchor+"    (args.out/'private-q2-host.py').write_text(_q2_source)\n    parent.save(args.out/'private-q2-host-proof.json',_q2_proof)\n"
    start="                        active=torch.arange(p['segments'],device='cuda')[None,:] < torch.tensor([(n+15)//16 for n in p['klens']],device='cuda')[:,None]\n"
    end="                            if name=='sums': workspace_ok=workspace_ok and bool((selected>0).all())\n"
    lo=raw.find(start);hi=raw.find(end)
    if lo<0 or hi<lo or raw.count(anchor)!=1: raise ValueError('Unknown allq1 workspace host span')
    span=raw[lo:hi+len(end)];replacement="                        workspace_ok=workspace_ok and workspace_check(live[role],p,torch)\n"
    host=raw.replace(span,replacement,1).replace(anchor,added,1)
    if host.replace(added,anchor,1).replace(replacement,span,1)!=raw: raise ValueError('Host exact inverse failed')
    proof=dict(parent_frontend='a4324fd975b34461486a8e80ef6a89867a12c039',source_pins=PINS,
        private_metadata_sha256=parent.factory.sha(changed),private_host_sha256=parent.factory.sha(host),
        metadata_only_maxq1_to2=True,metadata_exact_inverse=True,host_workspace_check_only_and_sidecar=True,host_exact_inverse=True,
        kernel_codec_numeric_gate_bytes_unchanged=True,actual_binding_verified=False,actual_I8_operand_hoist_verified=False,
        empty_row_source='59d9 softmax_step: m_j=-inf→0, P=0, L_new=0; original rawL1 times alpha0',
        raw_qlen2_verified=False,precision_gate=False,timing_attempted=False)
    ns=frozen.__dict__.copy();ns.update(contract=private,workspace_check=workspace_check,_q2_source=host,_q2_proof=proof)
    exec(compile(host,'<private max-q2 allreader host>','exec'),ns)
    return ns['run'],proof


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--numeric-diagnostic',action='store_true');parser.add_argument('--commit');parser.add_argument('--out',type=Path);parser.add_argument('--source-root',type=Path)
    args=parser.parse_args();args.profile='raw3d';run,proof=transforms()
    if args.run_gpu: result=run(args)
    else:
        if any((args.numeric_diagnostic,args.commit,args.out,args.source_root)): parser.error('Device flags require explicit --run-gpu')
        result=dict(cpu_only=True,proof=proof,cases=[dict(plan=plan('raw3d',i),empty_token_segments=sum(x==2 for row in statuses(plan('raw3d',i)) for x in row)) for i in (0,1)],
            actual_binding_verified=False,raw_qlen2_verified=False,Graph_verified=False,side_stream_verified=False,
            timing_attempted=False,precision_gate=False,performance_verified=False,service_verified=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
