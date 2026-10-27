#!/usr/bin/env python3
"""Reverify completed9b71 artifacts and deterministic archive; never load code/SO."""
import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

from biv150_unchanged_max_factory import build
from biv150_unchanged_max_ir import audit_artifacts

FROZEN='9b71be7362179cd35524b8698e886c1d5173cf41'
MANIFEST='4531539074567aec352ca28d658d30c23c9377d6a8041d6fd7ce556554727b14'
RESULT='9b9625dff32409687a38ea29a1a566d7902633eb5daa103f454f6036d4d16e9e'
PINS={'tools/bench_biv150_unchanged_max.py':'b8614253bba599e3e3f4350ab7295260dd425ad8c186e1139d4714fa2abda6c8',
      'tools/biv150_unchanged_max_ir.py':'1c0f03ccd5c24df6a25314d2d2b0f39f816ec90592ae3153a3ebca51e13f9fb5',
      'tools/biv150_unchanged_max_factory.py':'30b4917d1be43a084e524e83e6cfb89093c1cf81cd4d5f38dc5f013c838f8f8f',
      'vllm_fl/patches/biv150_gqa_decode.py':'2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def verify_files(files):
    manifest=json.loads(files['download-manifest.json'])
    names={'download-manifest.json'}
    for row in manifest['files']:
        name=row['relative'];p=PurePosixPath(name)
        need(not p.is_absolute()and'..'not in p.parts and name not in names,'Unsafe/duplicate member')
        raw=files[name];need(len(raw)==row['bytes']and sha(raw)==row['sha256'],'Raw mismatch: '+name);names.add(name)
    need(set(files)==names,'Unmanifested or missing member')
    need(len(names)-1==manifest['raw_files']and sum(len(files[n])for n in names if n!='download-manifest.json')==manifest['raw_bytes'],'Raw counts differ')
    return manifest


def old_failures(root):
    result=[];base=root/'docs/evidence/gqa-unchanged-max-20261027'
    for name,pin in [('v1-raw.tar.gz','18aa36f0973c3a678a0617b185df6fbc2de837373a1937b8cbe5eaaae30519b5'),
                     ('v2-raw.tar.gz','78ad9d2b3a759f137f7d6ba3e0c75f403b8d16c7740344e369a818f66e49cd76')]:
        data=(base/name).read_bytes();need(sha(data)==pin,'Old archive changed')
        with tarfile.open(fileobj=io.BytesIO(data))as archive:
            files={m.name:archive.extractfile(m).read()for m in archive.getmembers()if m.isfile()}
        manifest=verify_files(files);raw=json.loads(files['out/result.json'])
        need(not raw['records'][0]['passed']and not raw['records'][0]['timing_attempted']and raw.get('error'),'Old failure lost')
        result.append(dict(archive=name,sha256=pin,raw_files=manifest['raw_files'],raw_bytes=manifest['raw_bytes'],
            result_sha256=sha(files['out/result.json']),original_error=raw['error'],original_passed=False,original_timing_attempted=False))
    for row,name in zip(result,('v1-provenance.json','v2-fixture-failure-provenance.json')):
        row['unchanged_provenance']=dict(path=name,sha256=sha((base/name).read_bytes()))
    return result


def audit(result,files):
    r=result
    need(r['source_commit']==FROZEN and r['completed']is True and r['component_only']is True and r['gpu_executed']is True,'Wrong completion/source')
    need(all(r[k]is False for k in('official_score','precision_gate','model_evaluated'))and'error'not in r,'Unexpected model/score/error')
    need(sha(files['out/generated-gqa.py'])==r['proof']['generated_sha256']=='225c9cc1be483f7cee32eb8c61b87f97f238ea740ee1348c4f5a4023adf35313','Generated source changed')
    need([(c['batch'],c['kv'])for c in r['records']]==[(64,4096),(32,16384)],'Wrong requested matrix')
    rows=[]
    def artifacts(record):
        need(set(record['files'])=={'source','ttir','ttgir','llir','asm','cubin'},'Incomplete actual IR/ASM/cubin')
        for item in record['files'].values():
            raw=files['out/'+item['path']];need(sha(raw)==item['sha256']and len(raw)==item['bytes'],'Actual artifact changed')
        return {kind:files['out/'+record['files'][kind]['path']].decode()for kind in('ttgir','llir','asm')}
    for c in r['records']:
        need(c['passed']is True and c['timing_attempted']is True and c['segments']=={4096:2,16384:4}[c['kv']],'Incomplete case')
        need(c['packed_shape']==[c['batch']*c['kv']//16,2,16,2,128]and c['kv_stride']==[8192,256,128,1],'Packing changed')
        for label in('baseline','candidate'):
            resource=c[label];artifacts(resource)
            need(resource['phase']=='post_launch'and resource['registers_known']is True,'Unknown resource phase')
            need((resource['n_regs'],resource['n_spills'])==(93,0),'Actual counts changed')
            need(resource['metadata']==dict(shared=8192,num_warps=1,num_stages=1,use_sme=14,warp_size=64),'Launch config changed')
        need(all(c['resource_comparison'][k]=='known_equal'for k in('n_regs','n_spills','shared','num_warps','num_stages','use_sme','warp_size')),'Resource guard failed')
        mechanism=audit_artifacts(artifacts(c['candidate']),r['proof'])
        need(json.loads(json.dumps(mechanism))==c['mechanism'],'Actual IR re-audit differs')
        need([n['mode']for n in c['numeric_checks']]==['original','ragged','zero_qk','one_head_update','tiny_v'],'Mode coverage changed')
        reference=[]
        for n in c['numeric_checks']:
            need(n['finite']is True and n['producer_and_output_bits']==dict(scratch=True,maxima=True,sums=True,out=True),'Bitwise/finite guard failed')
            need([x['row']for x in n['reference']]==[0,1,c['batch']-1],'Reference sample rows changed')
            for x in n['reference']:
                before,after=x['baseline_relative_rmse'],x['candidate_relative_rmse']
                need(math.isfinite(before)and math.isfinite(after)and before>=0 and after>=0,'Invalid FP64 error')
                need(x['passed']is True and after<=max(.005,1.2*before),'FP64 gate failed')
                need(before==after,'Bitwise outputs but reference errors differ')
            reference.append(dict(mode=n['mode'],sampled_rows=n['reference'],all_producer_output_bits_equal=True))
        need(all(c[k]is True for k in('changed_graph_finite','changed_graph_exact','side_stream_exact','side_stream_finite')),'Graph/side guard failed')
        need(len(c['timings'])==3,'Timing rounds missing')
        rounds=[]
        for i,t in enumerate(c['timings']):
            order=['baseline_ms','candidate_ms']if i%2==0 else['candidate_ms','baseline_ms']
            need([k for k in t if k!='speedup']==order,'AB/BA/AB order changed')
            need(all(math.isfinite(t[k])and t[k]>0 for k in order),'Bad timing')
            ratio=t['baseline_ms']/t['candidate_ms'];need(ratio==t['speedup'],'Ratio differs')
            rounds.append(dict(iteration=i,order=order,**t))
        median=statistics.median(x['speedup']for x in rounds);need(median==c['median_speedup'],'Median differs')
        rows.append(dict(batch=c['batch'],kv=c['kv'],rounds=rounds,median_speedup=median,
            observed_ratio_change=median-1,reference_coverage=reference,mechanism_reaudited=mechanism,
            baseline_hash=c['baseline']['hash'],candidate_hash=c['candidate']['hash'],observed_registers=93,
            observed_spills=0,compiler_shared=8192,graph_side_exact_finite=True))
    p=r['primitive'];artifacts(p['resources'])
    need(json.loads(files['out/identity-primitive.json'])==p,'Primitive mirror differs')
    need(all(p[k]is True for k in('passed','real_exp_and_mul_present','alpha_one_exact','signed_zero_subnormal_normal_identity_exact')),'Identity primitive failed')
    return dict(source_commit=FROZEN,generated_sha256=r['proof']['generated_sha256'],component_only=True,
        source_factory_inverse=True,device_tensor_comparisons_reexecuted=False,recorded_device_guards_verified=True,
        independent_FP64_scope='Three sampled output rows per mode; not all-element independent exact arithmetic.',
        cases=rows,primitive=p,official_score=False,precision_gate=False,model_evaluated=False,
        service_evaluated=False,net_gain_observed=False,statistical_significance_claimed=False,
        conclusion='No measured benefit for this exact unchanged-max implementation; no additional whole-model run warranted. This does not reject all C09/conditional-rescale directions.')


def pack(files):
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=buffer,mode='wb',mtime=0,compresslevel=9)as zipped:
        with tarfile.open(fileobj=zipped,mode='w',format=tarfile.PAX_FORMAT)as archive:
            for name,raw in sorted(files.items()):
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.uid=info.gid=info.mtime=0;info.uname=info.gname='';archive.addfile(info,io.BytesIO(raw))
    return buffer.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    for name,pin in PINS.items():need(sha((root/name).read_bytes())==pin,'Frozen source changed: '+name)
    files={x.relative_to(a.snapshot).as_posix():x.read_bytes()for x in a.snapshot.rglob('*')if x.is_file()}
    need(sha(files['download-manifest.json'])==MANIFEST and sha(files['out/result.json'])==RESULT,'Wrong original snapshot')
    manifest=verify_files(files);need(manifest['raw_files']==38 and manifest['raw_bytes']==1815467,'Wrong raw inventory')
    need(manifest['original_owner']==dict(pid=623736,start_ticks=342729110,same_start=True,state='Z'),'Wrong owner')
    result=json.loads(files['out/result.json']);generated,proof=build((root/'vllm_fl/patches/biv150_gqa_decode.py').read_text())
    # Python3.14 omits empty AST fields by default; target3.12 emitted them.
    # Restore the exact target rendering, without altering source or raw proof.
    tree=ast.parse('tl.dot(probability.to(V.dtype), V)',mode='eval').body
    try:canonical=ast.dump(tree,include_attributes=False,show_empty=True)
    except TypeError:canonical=ast.dump(tree,include_attributes=False)
    pv_hash=sha(canonical.encode());need(pv_hash=='c389ad4bfeb0183ff788703501f8f93f21f9a5fdfd6bba6f66e5a6dc8ff85f6d','Unexpected PV AST')
    for span in proof['pv_call_spans']:span['call_ast_sha256']=pv_hash
    need(generated.encode()==files['out/generated-gqa.py']and proof==result['proof'],'Source factory reconstruction differs')
    summary=audit(result,files);summary.update(cpu_only=True,gpu_executed=False,original_manifest_sha256=MANIFEST,
        original_result_sha256=RESULT,raw_files=38,raw_bytes=1815467,opaque_pyc_files_preserved=2,pyc_executed=False,
        original_owner=manifest['original_owner'],old_failed_originals_verified=old_failures(root))
    faults=[]
    for name,mutate in [('bitwise',lambda r:r['records'][0]['numeric_checks'][0]['producer_and_output_bits'].update(out=False)),
        ('reference',lambda r:r['records'][0]['numeric_checks'][0]['reference'][0].update(candidate_relative_rmse=1.)),
        ('graph',lambda r:r['records'][0].update(changed_graph_exact=False)),
        ('resource',lambda r:r['records'][0]['candidate'].update(n_regs=None)),
        ('median',lambda r:r['records'][0].update(median_speedup=1.)),
        ('completion',lambda r:r.update(completed=False))]:
        bad=copy.deepcopy(result);mutate(bad)
        try:audit(bad,files)
        except(ValueError,AssertionError):faults.append(name)
        else:raise ValueError('Fault accepted: '+name)
    summary['rejected_faults']=faults;summary['archive_checker_sha256']=sha(Path(__file__).read_bytes())
    raw=pack(files);need(len(raw)<=2*1024*1024,'Archive too large')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'v3-raw.tar.gz').write_bytes(raw)
    (a.out/'download-manifest.json').write_bytes(files['download-manifest.json']);(a.out/'summary.json').write_bytes(encode(summary))
    (a.out/'archive-manifest.json').write_bytes(encode(dict(archive='v3-raw.tar.gz',bytes=len(raw),sha256=sha(raw),
        archive_members=39,original_manifest_sha256=MANIFEST,original_result_sha256=RESULT,summary_sha256=sha(encode(summary)),files=manifest['files'])))
    need('torch'not in sys.modules and'triton'not in sys.modules,'GPU import occurred')
    print(json.dumps(dict(passed=True,cpu_only=True,raw_files=38,archive_sha256=sha(raw),archive_bytes=len(raw))))


if __name__=='__main__':main()
