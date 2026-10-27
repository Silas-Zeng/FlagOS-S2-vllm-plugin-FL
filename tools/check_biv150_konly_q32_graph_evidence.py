#!/usr/bin/env python3
"""Replay98 immutable Graph/side originals, optionally all CPU tensor modes."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import check_biv150_konly_q32_eager_evidence as eager
import bench_biv150_konly_q32_graph_side as frontend
from biv150_cpu_tensor_archive import read_tensor_archive

DIRECTORY=eager.runner.parent.ROOT/'docs/evidence/konly-prefill-reader-20261027/q32-graph-side-device-v1'


def original():
    provenance=json.loads((DIRECTORY/'provenance.json').read_text())
    assert provenance['runtime_commit']=='d2ceb3bd8301ba846792e3370803832dc64270ca' and not provenance['omitted_files']
    previous,unused=eager.original()
    assert eager.sha((eager.DIRECTORY/'provenance.json').read_bytes())==provenance['previous_provenance_sha256']
    parts={}
    for index,row in enumerate(provenance['archives']):
        data=(DIRECTORY/row['file']).read_bytes();assert len(data)==row['bytes']<2*1024**2 and eager.sha(data)==row['sha256']
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members=archive.getmembers();assert len(members)==row['members']
            for m in members:
                assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
                key=(index,m.name);assert key not in parts;parts[key]=archive.extractfile(m).read()
    files={};used=set();reused=0
    for row in provenance['files']:
        name=row['relative'];assert name not in files and not name.startswith('/') and '..' not in Path(name).parts
        blocks=[]
        for part in row['parts']:
            if part['origin']=='eager-v3':
                data=previous[part['relative']][part['offset']:part['offset']+part['bytes']];reused+=len(data)
            else:
                assert part['origin']=='new';key=(part['archive'],part['member']);assert key not in used
                used.add(key);data=parts[key]
            assert len(data)==part['bytes'] and eager.sha(data)==part['sha256'];blocks.append(data)
        data=b''.join(blocks);assert len(data)==row['bytes'] and eager.sha(data)==row['sha256'];files[name]=data
    assert used==set(parts) and reused==provenance['reused_eager_v3_bytes']
    assert len(files)==99 and eager.sha(files['download-manifest.json'])==provenance['manifest_sha256']
    manifest=json.loads(files['download-manifest.json'])
    assert manifest['original_owner']==dict(pid=628883,start_ticks=344599705,same_start=True,state='Z')
    assert manifest['raw_files']==98 and manifest['raw_bytes']==117797007
    assert manifest['read_only'] and manifest['model_rerun'] is False
    assert set(files)=={x['relative'] for x in manifest['files']}|{'download-manifest.json'}
    for row in manifest['files']:
        value=files[row['relative']];assert (len(value),eager.sha(value))==(row['bytes'],row['sha256'])
    return files,provenance


def arrays(files,summary,graph):
    import numpy as np
    metric=eager.replay_arrays(files,summary)
    saved=read_tensor_archive(files['out/graph-side-full-outputs.pt.gz']);assert saved['changed_controls']==graph['changed_controls']
    assert saved['postlaunch_resources']==graph['actual_postlaunch_resources']
    results=[]
    for role in ('baseline','unhoisted','hoisted'):
        values=saved['outputs'][role];assert set(values)=={'eager0','graph1','eager1','side1'}
        bits={mode:value.array(np) for mode,value in values.items()}
        assert all(a.shape==(2048,16,128) and a.dtype==np.dtype('<u2') for a in bits.values())
        assert all(bool(((a&0x7f80)!=0x7f80).all()) for a in bits.values())
        assert bits['graph1'].tobytes()==bits['eager1'].tobytes()==bits['side1'].tobytes()
        for pattern,mode in ((0,'eager0'),(1,'side1')):
            artifact=read_tensor_archive(files['out/pattern-'+str(pattern)+'.pt.gz'])
            assert bits[mode].tobytes()==artifact['outputs'][role].array(np).tobytes()
        results.append(dict(role=role,full_elements_per_mode=4194304,Graph_eager_side_full_bitwise=True,all_modes_finite=True))
    for mode in ('eager0','graph1','eager1','side1'):
        assert saved['outputs']['unhoisted'][mode].array(np).tobytes()==saved['outputs']['hoisted'][mode].array(np).tobytes()
    for pattern in (0,1):
        artifact=read_tensor_archive(files['out/pattern-'+str(pattern)+'.pt.gz']);cpu=artifact['original_CPU']
        for key in ('q','table','lengths'):
            assert eager.sha(cpu[key].array(np).tobytes())==graph['changed_controls'][str(pattern)][key]
        assert cpu['lengths'].array(np).tolist()==[6144-pattern]
    return dict(eager_artifact_replay=metric,full_mode_replay=results,all_modes_I8_hoisted_unhoisted_bitwise=True,
                actual_Q_table_length_digests_recomputed=True,reference_scope='Archived selected-row FP64 references; metrics independently recomputed')


def run(replay=False,extract=None):
    files,p=original();summary=json.loads(files['out/numeric-summary.json']);graph=json.loads(files['out/graph-side-summary.json'])
    assert summary['frontend_commit']==p['runtime_commit'] and summary['completed'] and graph['completed']
    assert summary['errors']==graph['errors']==[] and summary['Graph_verified'] and summary['side_stream_verified']
    assert summary['Graph_executed'] and summary['side_stream_executed'] and not summary['eager_numeric_passed']
    assert graph['Graph_verified'] and graph['side_stream_verified'] and graph['counts']==dict(capture=3,eager=6,graph_replay=3,side=3)
    assert not any(summary[k] for k in ('precision_gate','performance_verified','service_verified','capacity_reclaimed','timing_attempted'))
    assert summary['timings'] is None and all(all(v is True for v in r['hard_checks'].values()) for r in summary['records'])
    assert graph['coherent_roles']==['baseline','unhoisted','hoisted']
    assert all(graph['changed_controls']['0'][key]!=graph['changed_controls']['1'][key] for key in ('q','table','lengths'))
    admission=summary['admission'];assert admission['state']=='completed' and admission['errors']==[]
    assert admission['container_identity']==dict(pid=628883,start_ticks=344599705)
    assert len(graph['native_phase_checks'])==15 and all(x['native_host_pid']==admission['native_host_pid'] and x['token_sha256']==admission['token_sha256'] for x in graph['native_phase_checks'])
    resources=graph['actual_postlaunch_resources'];assert len(resources)==12
    assert all(x['observed_after_real_attention'] and x['same_object_as_warmup'] and not x['zero_spill_required'] for x in resources)
    old,oldsummary,_=eager.runner.actual.old.original();bb,bbsummary,_=eager.runner.actual.original();IR={}
    for role in ('baseline','unhoisted','hoisted'):
        origin,previous,label=(bb,bbsummary,'candidate') if role=='hoisted' else (old,oldsummary,'baseline' if role=='baseline' else 'candidate')
        record=json.loads(files['out/ir/'+role+'/record.json']);expected=next(r for r in previous['records'] if r['label']==label)
        observed={ext:files['out/ir/'+role+'/actual.'+ext] for ext in ('source','ttir','ttgir','llir','asm','cubin')}
        reviewed={ext:origin['out/ir/'+label+'/actual.'+ext] for ext in observed}
        proof=eager.runner.verify_actual_compilation(record,expected,role,observed,reviewed)
        assert proof==json.loads(files['out/ir/'+role+'/actual-compilation-binding.json'])
        post=[x for x in resources if x['role']==role];assert len(post)==4
        assert all(x['compiled_hash']==record['hash'] and x['metadata']['hash']==record['hash'] for x in post)
        IR[role]=dict(hash=record['hash'],n_regs=[x['n_regs'] for x in post],n_spills=[x['n_spills'] for x in post],shared=[x['metadata']['shared'] for x in post])
    host,reference,proof=frontend.transforms()
    assert files['out/private-host-runner.py.txt']==host.encode() and files['out/private-CPU-reference.py.txt']==reference.encode()
    assert json.loads(files['out/private-host-transform.json'])==proof
    if extract:
        target=Path(extract);target.mkdir(parents=True,exist_ok=False)
        for name,data in files.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    result=dict(original_raw_files=98,original_raw_bytes=117797007,all_original_SHA_verified=True,
        original_owner=p['original_owner'],new_archives=len(p['archives']),reused_eager_v3_bytes=p['reused_eager_v3_bytes'],
        maximum_archive_bytes=max(r['bytes'] for r in p['archives']),actual_IR_and_postlaunch_resources=IR,
        recorded_Graph_verified=True,recorded_side_stream_verified=True,eager_numeric_passed=False,
        precision_gate=False,performance_verified=False,service_verified=False,capacity_reclaimed=False,
        transport_recovery_only=p['transport'],legacy_baseline_binary_equality_label_invalid=True)
    if replay: result['arrays']=arrays(files,summary,graph)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arrays',action='store_true');parser.add_argument('--extract');args=parser.parse_args()
    print(json.dumps(run(args.arrays,args.extract),indent=2,sort_keys=True))
