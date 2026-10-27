#!/usr/bin/env python3
"""Reassemble93 original files; optional NumPy full-output/metric CPU replay, no Torch."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import bench_biv150_konly_q32_numeric as runner
from biv150_cpu_tensor_archive import read_tensor_archive

DIRECTORY=runner.parent.ROOT/'docs/evidence/konly-prefill-reader-20261027/q32-eager-device-v3'


def sha(data): return hashlib.sha256(data).hexdigest()


def original():
    provenance=json.loads((DIRECTORY/'provenance.json').read_text());parts={}
    assert provenance['runtime_commit']=='2d79d9546a8a240447fee05479c5f6caa00aa296'
    assert provenance['omitted_files']==[] and len(provenance['archives'])==29
    for index,row in enumerate(provenance['archives']):
        data=(DIRECTORY/row['file']).read_bytes()
        assert len(data)==row['bytes']<2*1024**2 and sha(data)==row['sha256']
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members=archive.getmembers();assert len(members)==row['members']
            for m in members:
                assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
                key=(index,m.name);assert key not in parts;parts[key]=archive.extractfile(m).read()
    files={};used=set()
    for row in provenance['files']:
        name=row['relative'];assert name not in files and not name.startswith('/') and '..' not in Path(name).parts
        blocks=[]
        for part in row['parts']:
            key=(part['archive'],part['member']);assert key not in used;used.add(key)
            data=parts[key];assert len(data)==part['bytes'] and sha(data)==part['sha256'];blocks.append(data)
        value=b''.join(blocks);assert len(value)==row['bytes'] and sha(value)==row['sha256'];files[name]=value
    assert used==set(parts) and len(files)==94
    manifest_bytes=files['download-manifest.json'];assert sha(manifest_bytes)==provenance['manifest_sha256']
    manifest=json.loads(manifest_bytes)
    assert manifest['original_owner']==dict(pid=628305,start_ticks=344402049,same_start=True,state='Z')
    assert manifest['raw_files']==93 and manifest['raw_bytes']==57073824
    assert set(files)=={r['relative'] for r in manifest['files']}|{'download-manifest.json'}
    for row in manifest['files']:
        value=files[row['relative']];assert (len(value),sha(value))==(row['bytes'],row['sha256'])
    return files,provenance


def independent_metrics(actual,expected):
    assert actual and len(actual)==len(expected) and all(math.isfinite(x) for x in actual+expected)
    errors=[a-b for a,b in zip(actual,expected)];rms=math.hypot(*errors)/math.sqrt(len(errors))
    reference_rms=math.hypot(*expected)/math.sqrt(len(expected));maximum=max(map(abs,errors))
    violations=sum(abs(d)>.005+.005*abs(r) for d,r in zip(errors,expected))
    return dict(finite=True,max_abs=maximum,rms=rms,relative_rmse=rms/reference_rms if reference_rms else (0. if maximum==0 else math.inf),
                pointwise_violations=violations,pointwise_passed=violations==0)


def replay_arrays(files,summary):
    import numpy as np
    results=[]
    for record in summary['records']:
        pattern=record['pattern'];blob=files['out/pattern-'+str(pattern)+'.pt.gz']
        assert sha(blob)==record['artifact']['sha256'] and len(blob)==record['artifact']['bytes']
        data=read_tensor_archive(blob);outputs=data['outputs']
        arrays={role:output.array(np) for role,output in outputs.items()}
        assert set(arrays)=={'baseline','unhoisted','hoisted'}
        assert all(a.shape==(2048,16,128) and a.dtype==np.dtype('<u2') for a in arrays.values())
        assert arrays['unhoisted'].tobytes()==arrays['hoisted'].tobytes()
        assert all(bool(((a&0x7f80)!=0x7f80).all()) for a in arrays.values())
        expected=data['expected_encoded_bytes'].array(np);actual=data['actual_encoded_bytes'].array(np)
        assert expected.tobytes()==actual.tobytes() and len(expected)==773*16384
        # Independently reconstruct byte placement from saved CPU code/scale/V.
        packed=np.full((773,16384),0xa5,dtype=np.uint8)
        packed[:,:4096]=data['expected_codes'].array(np).view(np.uint8).reshape(773,4096)
        packed[:,4096:4224]=data['expected_scales'].array(np).view(np.uint8).reshape(773,128)
        packed[:,8192:]=data['original_CPU']['packed'].array(np)[:,1].copy().view(np.uint8).reshape(773,8192)
        assert packed.tobytes()==expected.tobytes()
        rows=data['reference']['rows'];assert rows==[0,1,15,16,31,32,1024,2047]
        metrics={}
        for label,role,reference in [('baseline','baseline','original'),('quantized_operand','hoisted','quantized'),('original_BF16','hoisted','original')]:
            selected=(arrays[role][rows].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(-1).tolist()
            expected_values=data['reference'][reference].values(np).reshape(-1).tolist()
            metric=independent_metrics(selected,expected_values);assert metric==record['numeric_checks'][label];metrics[label]=metric
        limit=max(.005,1.2*metrics['baseline']['relative_rmse']);passed=metrics['baseline']['pointwise_passed'] and metrics['quantized_operand']['pointwise_passed'] and metrics['original_BF16']['relative_rmse']<=limit
        assert passed==record['numeric_checks']['candidate_numeric_passed']==record['policy']['eager_numeric_passed']
        assert limit==record['numeric_checks']['original_relative_rmse_limit'] and not record['numeric_checks']['thresholds_relaxed']
        results.append(dict(pattern=pattern,full_output_elements=4194304,full_output_bytes=8388608,
                    output_bits_equal=True,all_three_outputs_finite=True,full_encoded_bytes_equal=True,
                    independent_byte_layout_equal=True,metrics=metrics,eager_numeric_passed=passed))
    assert [r['eager_numeric_passed'] for r in results]==[True,False]
    return dict(patterns=results,independent_metric_replay=True,full_output_bitwise_replay=True,
                reference_scope='Metrics recomputed from archived independent FP64 references; global softmax reference not regenerated here',
                canary_readonly_scope='Recorded device guards; prefix/suffix canary buffers and live before/after fingerprints are not serialized')


def run(arrays=False,extract=None):
    files,provenance=original();summary=json.loads(files['out/numeric-summary.json'])
    assert summary['frontend_commit']==provenance['runtime_commit'] and summary['completed'] and summary['errors']==[]
    assert summary['actual_binding_verified'] and summary['actual_mechanism_verified']
    assert (summary['encoder_launches'],summary['reader_launches'])==(2,6)
    assert not summary['eager_numeric_passed'] and not summary['precision_gate'] and not summary['full_validation_passed']
    assert all(all(x is True for x in r['hard_checks'].values()) for r in summary['records'])
    assert not any(summary[k] for k in ('Graph_verified','side_stream_verified','performance_verified','service_verified','capacity_reclaimed','timing_attempted'))
    assert summary['timings'] is None and summary['admission']['state']=='completed' and summary['admission']['errors']==[]
    assert summary['admission']['container_identity']==dict(pid=628305,start_ticks=344402049)
    assert all(p['native_host_pid']==summary['admission']['native_host_pid'] for p in summary['admission']['phases'])
    old,os,_=runner.actual.old.original();bb,bs,_=runner.actual.original();IR={}
    for role in ('baseline','unhoisted','hoisted'):
        origin,previous,label=(bb,bs,'candidate') if role=='hoisted' else (old,os,'baseline' if role=='baseline' else 'candidate')
        record=json.loads(files['out/ir/'+role+'/record.json']);expected=next(r for r in previous['records'] if r['label']==label)
        resource=record['loaded_resources']
        assert resource['loaded_resource_fields_known'] and type(resource['n_regs']) is int and resource['n_regs']>0
        assert type(resource['n_spills']) is int and resource['n_spills']==0
        observed={ext:files['out/ir/'+role+'/actual.'+ext] for ext in ('source','ttir','ttgir','llir','asm','cubin')}
        reviewed={ext:origin['out/ir/'+label+'/actual.'+ext] for ext in observed}
        proof=runner.verify_actual_compilation(record,expected,role,observed,reviewed)
        saved=json.loads(files['out/ir/'+role+'/actual-compilation-binding.json'])
        assert proof==saved and not saved['backend_binary_equivalence_claimed']
        IR[role]=dict(hash=record['hash'],actual_cubin_exact=saved['actual_cubin_exact'],actual_cubin_sha256=saved['actual_cubin_sha256'],loaded_resources=resource)
        if role=='baseline': assert saved['actual_strict_audit']['source_IR_and_cubin_exact'] is True
    if extract:
        target=Path(extract);target.mkdir(parents=True,exist_ok=False)
        for name,data in files.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    result=dict(original_raw_files=93,original_raw_bytes=57073824,all_original_SHA_verified=True,
                original_owner=provenance['original_owner'],archives=29,maximum_archive_bytes=max(r['bytes'] for r in provenance['archives']),
                recorded_device_hard_checks_passed=True,actual_IR_replayed=IR,eager_numeric_passed=False,
                legacy_baseline_source_IR_and_cubin_exact_label_preserved_but_invalid=True,
                statement='Exact source/ABI/key IR plus actual ASM mechanism; binary equality is not claimed',
                precision_gate=False,Graph_verified=False,side_stream_verified=False,context16k_tested=False,
                performance_verified=False,service_verified=False,capacity_reclaimed=False)
    if arrays: result['arrays']=replay_arrays(files,summary)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arrays',action='store_true');parser.add_argument('--extract');args=parser.parse_args()
    print(json.dumps(run(args.arrays,args.extract),indent=2,sort_keys=True))
