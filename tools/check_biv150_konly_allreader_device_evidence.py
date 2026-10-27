"""CPU-only full original-SHA/IR/codec/output/metric replay of three device feeds."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import biv150_konly_allreader_numeric_contract as c
from biv150_cpu_tensor_archive import read_tensor_archive
from check_biv150_konly_q32_eager_evidence import independent_metrics
from biv150_konly_decode_quant_reference import rn32_div

def sha(b): return hashlib.sha256(b).hexdigest()

def codec(bits,np):
    shape=bits.shape;bits=bits.reshape(-1,128);out=np.empty(bits.shape,dtype=np.int8);scales=np.empty(len(bits),np.float32)
    for start in range(0,len(bits),4096):
        block=bits[start:start+4096];exp=(block>>7)&255;assert not (exp==255).any()
        value=np.ldexp((block&127).astype(np.float64)+(exp!=0)*128,np.maximum(exp.astype(np.int32),1)-134)
        value=np.where((block&0x8000)!=0,-value,value);peak=np.max(abs(value),axis=1)
        scale=np.array([1. if p==0 else max(rn32_div(float(p),127.),2.**-126) for p in peak],dtype=np.float32)
        ratio=(value/scale.astype(np.float64)[:,None]).astype(np.float32);code=np.clip(np.rint(ratio),-127,127).astype(np.int8)
        mag=np.abs(ratio).astype(np.float64);near=np.abs(mag-np.floor(mag)-.5)<=2*np.spacing(np.abs(ratio)).astype(np.float64)
        for row,col in zip(*np.nonzero(near)): code[row,col]=max(-127,min(127,round(rn32_div(float(value[row,col]),float(scale[row])))))
        out[start:start+len(block)]=code;scales[start:start+len(block)]=scale
    return out.reshape(shape),scales.reshape(shape[:-1])

def run(root,profile):
    import numpy as np
    baseprofile='raw3d' if profile=='rawq2' else profile
    root=Path(root);manifest=json.loads((root/'download-manifest.json').read_text());names=set()
    for row in manifest['files']:
        name=row['relative'];assert name not in names and not name.startswith('/') and '..' not in Path(name).parts;names.add(name)
        raw=(root/name).read_bytes();assert (len(raw),sha(raw))==(row['bytes'],row['sha256'])
    assert len(names)==manifest['raw_files'] and sum(r['bytes'] for r in manifest['files'])==manifest['raw_bytes']
    assert manifest['read_only'] and not manifest['model_rerun'] and manifest['original_owner']['same_start'] and manifest['original_owner']['state']=='Z'
    summary=json.loads((root/'out/allreader-numeric-summary.json').read_text());assert summary['completed'] and summary['errors']==[] and summary['frontend_commit']==('7db70ef33ab3ac77c45a3dcef459889f593bd09f' if profile=='rawq2' else 'a4324fd975b34461486a8e80ef6a89867a12c039')
    assert not summary['eager_numeric_passed'] and not any(summary[k] for k in ('Graph_verified','side_stream_verified','timing_attempted','precision_gate','service_verified','capacity_reclaimed'))
    assert (summary['encoder_launches'],summary['reader_launches'],summary['reducer_launches'])==(4,8,8 if baseprofile=='raw3d' else 0)
    transition=summary['admission']['original_transition'];assert transition['state']=='completed' and transition['errors']==[] and transition['initialization_attempted']
    assert all(r['native_host_pid']==transition['native_host_pid'] and r['token_sha256']==transition['token_sha256'] for r in summary['admission']['actual_phase_checks'])
    old,original,_=c.original(baseprofile);ir=[]
    for case in (0,1):
        directory=root/'out'/('case-'+str(case));rows=json.loads((directory/'actual-records.json').read_text());saved=json.loads((directory/'actual-binding.json').read_text())
        for record in rows:
            label=record['label'];previous=next(r for r in original['records'] if r['label']==label)
            observed={e:(directory/record['files'][e]['path']).read_bytes() for e in ('source','ttir','ttgir','llir','asm','cubin')}
            prior={e:old['out/'+previous['files'][e]['path']] for e in observed}
            assert c.bind_actual(baseprofile,record,previous,observed,prior)==saved[label]
            ir.append(dict(case=case,role=label,hash=record['hash'],native=saved[label]['actual_native_matrix_counts']))
    results=[]
    for record in summary['records']:
        path=root/'out'/record['artifact']['path'];blob=path.read_bytes();assert sha(blob)==record['artifact']['sha256'] and len(blob)==record['artifact']['bytes']
        data=read_tensor_archive(blob);del blob;npout={k:v.array(np) for k,v in data['outputs'].items()};p=record['geometry']
        assert all(a.shape==(p['total'],16,128) and ((a&0x7f80)!=0x7f80).all() for a in npout.values())
        codes,scales=codec(data['original_CPU']['packed'].array(np)[:,0].copy(),np)
        assert codes.tobytes()==data['expected_codes'].array(np).tobytes() and scales.tobytes()==data['expected_scales'].array(np).tobytes()
        packed=np.full((p['pages'],16384),0xa5,np.uint8);packed[:,:4096]=codes.view(np.uint8).reshape(p['pages'],4096)
        packed[:,4096:4224]=scales.view(np.uint8).reshape(p['pages'],128)
        packed[:,8192:]=data['original_CPU']['packed'].array(np)[:,1].copy().view(np.uint8).reshape(p['pages'],8192)
        assert packed.tobytes()==data['expected_raw'].array(np).tobytes()==data['actual_raw'].array(np).tobytes()
        cpu=data['original_CPU'];digests={name:sha(cpu[key].array(np).tobytes()) for name,key in [('q','q'),('packed','packed'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('kd','k_scale'),('vd','v_scale')]};digests['encoded']=sha(packed.tobytes())
        assert digests==record['readonly_input_digests_before']==record['readonly_input_digests_after']
        for role,guard in data['guard_rows'].items():
            assert (guard['output'].values(np)==71).all()
            assert all((a.values(np)==43).all() for a in guard['workspace_suffix'].values())
        if profile=='rawq2':
            import bench_biv150_konly_raw_q2_numeric as q2
            kind=np.array(q2.statuses(p))[:,None,:];kind=np.broadcast_to(kind,(p['total'],16,16))
            for role,workspace in data['workspace'].items():
                for name,t in workspace.items():
                    a=t.values(np);assert np.isfinite(a[kind==1]).all() and np.isnan(a[kind==0]).all() and (a[kind==2]==0).all()
                    if name=='sums': assert (a[kind==1]>0).all()
            _,proof=q2.transforms();assert proof==json.loads((root/'out/private-q2-host-proof.json').read_text())
            assert sha((root/'out/private-q2-host.py').read_bytes())==proof['private_host_sha256']
        elif profile=='raw3d':
            active=(np.arange(16)[None,:]<np.array([(k+15)//16 for k in p['klens']])[:,None])[:,None,:]
            active=np.broadcast_to(active,(p['total'],16,16))
            for role,workspace in data['workspace'].items():
                for name,t in workspace.items():
                    a=t.values(np);assert np.isfinite(a[active]).all() and np.isnan(a[~active]).all()
                    if name=='sums': assert (a[active]>0).all()
        metrics={};rows=data['reference']['rows']
        for name,role,ref in [('baseline','baseline','original'),('quantized_operand','candidate','quantized'),('original_BF16','candidate','original')]:
            a=(npout[role][rows].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(-1).tolist()
            b=data['reference'][ref].values(np).reshape(-1).tolist();metrics[name]=independent_metrics(a,b);assert metrics[name]==record['numeric_checks'][name];del a,b;gc.collect()
        limit=max(.005,1.2*metrics['baseline']['relative_rmse']);passed=metrics['baseline']['pointwise_passed'] and metrics['quantized_operand']['pointwise_passed'] and metrics['original_BF16']['relative_rmse']<=limit
        assert passed==record['numeric_checks']['candidate_numeric_passed']==record['policy']['eager_numeric_passed']
        assert all(v is True for v in record['hard_checks'].values()) and limit==record['numeric_checks']['original_relative_rmse_limit']
        results.append(dict(case=record['case'],pattern=record['pattern'],full_output_elements=p['total']*16*128,full_finite=True,full_CPU_codec_recomputed=True,full_encoded_bytes=True,canary_arrays_and_readonly_digests=True,
            uniform_output_bitwise=npout['baseline'].tobytes()==npout['candidate'].tobytes() if record['pattern']==0 else None,metrics=metrics,eager_numeric_passed=passed,loaded_resources=record['actual_postlaunch_resources']))
        print(json.dumps(dict(progress=profile,case=record['case'],pattern=record['pattern'],replayed=True)),file=sys.stderr,flush=True)
        del data,npout,packed,codes,scales;gc.collect()
    return dict(profile=profile,raw_files=manifest['raw_files'],raw_bytes=manifest['raw_bytes'],all_original_SHA_verified=True,owner=manifest['original_owner'],actual_IR_replayed=ir,arrays=results,
        references='Saved independent FP64 reference metrics recomputed; global softmax reference not regenerated',eager_numeric_passed=False,precision_gate=False,timing_attempted=False,service_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=('multi2','raw3d','rawq2'),required=True);p.add_argument('--raw',required=True);a=p.parse_args();print(json.dumps(run(a.raw,a.profile),indent=2,sort_keys=True))
