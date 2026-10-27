#!/usr/bin/env python3
"""Preserve/analyze the actual ee579 two-arm profile on CPU; never a score."""
import argparse
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pstats
import sys
import tarfile

MANIFEST='c655e47d7b3c16357a7f42b9287c758d100e261ad7717a69ffd943699151b154'
DIAGNOSTIC='47b8020fdbdbb4818eb52f4e98e77c76b1ef4fd545e745c4b80704c14caf3280'
SOURCE='ee579b3401b1a34dd4316ab60299236e435b9c50'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def audit(report,root):
    need(report['source_commit']==SOURCE and report['diagnostic_only']is True,'Wrong source/scope')
    need(report['performance_score']is False and report['speedup_not_computed']is True
        and report['model_precision_gate']is False and report['full_component_acceptance']is False,'Diagnostic misrepresented')
    need(report['completed']is True and report['diagnostic_checks_passed']is True
        and report['parent_restored']is True and report['errors']==[],'Diagnostic incomplete')
    need(report['query']==2048 and report['kv']==4096 and report['calls_per_arm']==2
        and report['warm_calls_per_arm']==3 and report['round_sweep']is False,'Bounded input changed')
    need([a['arm']for a in report['arms']]==['baseline','candidate'],'Wrong arms')
    arms=[]
    for a in report['arms']:
        output=a['post_profile_output_exact']
        need(output['passed']is True and output['finite']is True and output['numeric_equal']is True,'Output differs')
        need(a['observed_resource_guard']['strict_allowed']is True,'Resource guard failed')
        need(a['live_descriptor_calls_by_block_m']=={'128':2,'256':2},'Live descriptor checks changed')
        need(len(a['host_call_ns'])==2 and all(type(v)is int and v>0 for v in a['host_call_ns']),'Invalid call durations')
        need(a['gc_before']['enabled']==a['gc_after']['enabled']and a['gc_before']['threshold']==a['gc_after']['threshold'],'GC policy changed')
        stats=pstats.Stats(str(root/'out'/(a['arm']+'.cprofile')),stream=io.StringIO()).stats
        groups={}
        for row in a['functions']:
            key=(row['filename'] or '~',row['first_line'] or 0,
                 row['function'].split('.')[-1] if row['filename'] else row['function'])
            groups.setdefault(key,[]).append(row)
        need(set(groups)==set(stats),'JSON/profile function keys differ')
        collisions=[];unique=0
        for key,members in groups.items():
            stat=stats[key];observed=(stat[1],stat[1]-stat[0],stat[2],stat[3])
            matches=[row for row in members if observed==(row['calls'],row['recursive'],row['self_seconds'],row['inclusive_seconds'])]
            need(matches,'JSON/profile metrics differ: '+str(key))
            if len(members)==1:unique+=1
            else:
                # cProfile's dump_stats key loses identity when generated
                # dataclass constructors share filename/line/name. Do not
                # pretend the binary independently verifies each collided row.
                collisions.append(dict(filename=key[0],line=key[1],name=key[2],
                    JSON_rows=len(members),binary_rows=1,JSON_self_seconds=sum(x['self_seconds']for x in members),
                    retained_binary_self_seconds=stat[2],individual_rows_independently_resolved=False))
        buckets={}
        for row in a['functions']:buckets[row['bucket']]=buckets.get(row['bucket'],0)+row['self_seconds']
        need(set(buckets)==set(a['exclusive_cProfile_bucket_seconds']),'Bucket keys differ')
        need(all(math.isclose(v,a['exclusive_cProfile_bucket_seconds'][k],rel_tol=1e-12,abs_tol=1e-15)
                 for k,v in buckets.items()),'Exclusive bucket sum differs')
        need(not any('inspect.py'==(Path(row['filename']).name if row['filename'] else '')
            and row['function'].split('.')[-1]in('bind','bind_partial','_bind')for row in a['functions']),'Unexpected signature binding')
        arms.append(dict(arm=a['arm'],host_call_ns=a['host_call_ns'],profiled_wall_ns=a['profiled_wall_ns'],
            profiled_event_ms=a['profiled_event_ms'],end_sync_ns=a['end_sync_ns'],
            exclusive_buckets_seconds=buckets,top_self_functions=a['functions'][:16],
            postlaunch_audit_span_ns=a['boundary_postlaunch_audit_span_ns'],audit_span_overlaps_boundary=True,
            gc_events=a['gc_events'],rusage_deltas=a['rusage_deltas'],live_descriptor_calls=a['live_descriptor_calls_by_block_m'],
            unique_binary_profile_rows_matched=unique,JSON_function_rows=len(a['functions']),
            binary_profile_rows=len(stats),profile_key_collisions=collisions))
    candidate=arms[1];own=('certificate','route-and-one-shot','boundary-and-audit-plumbing','host-binding-scope')
    own_total=sum(candidate['exclusive_buckets_seconds'][k]for k in own)
    return dict(cpu_only=True,new_GPU_execution=False,source_commit=SOURCE,diagnostic_only=True,
        performance_score=False,speedup_computed=False,old_v4_gap_cause_proven=False,
        no_new_runtime_optimization_justified=True,arms=arms,
        candidate_only_exclusive_total_seconds=own_total,
        candidate_only_instrumented_exclusive_seconds_per_call=own_total/2,
        candidate_audit_instrumented_seconds_per_call=sum(candidate['postlaunch_audit_span_ns'])/2e9,
        shared_SDK_self_seconds=[next(row['self_seconds']for row in a['functions']if row['function']=='unified_attention')for a in report['arms']],
        findings=[
            'Shared SDK/descriptor/JIT paths shift substantially between the two instrumented arms; their difference is not an algorithm speedup.',
            'Both arms retain two BM128 and two BM256 descriptor checks; the qualified cache did not remove live parent/Manager validation.',
            'No GC events, thread involuntary context switches or major faults were observed in these captured windows; this says nothing conclusive about older v4 rounds.',
            'End synchronize includes remaining GPU wait and cannot independently identify host scheduling or old discrete gaps.',
            'The kwargs normalizer path already bypasses inspect.Signature.bind; no new argument-binding cache is indicated.',
            'Postlaunch audit is evidence-only but remains part of this component lifecycle/charge. Do not disable it only in timed calls.',
            'Keep runtime code/guards unchanged; move toward controlled real-service/105 diagnostics rather than more proof-cache complexity.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    files={x.relative_to(a.snapshot).as_posix():x.read_bytes()for x in a.snapshot.rglob('*')if x.is_file()}
    need(sha(files['download-manifest.json'])==MANIFEST and sha(files['out/diagnostic.json'])==DIAGNOSTIC,'Wrong original snapshot')
    manifest=json.loads(files['download-manifest.json']);names={'download-manifest.json'}
    for row in manifest['files']:
        name=row['relative'];path=PurePosixPath(name)
        need(not path.is_absolute()and'..'not in path.parts and name not in names,'Unsafe/duplicate member')
        need(len(files[name])==row['bytes']and sha(files[name])==row['sha256'],'Raw mismatch: '+name);names.add(name)
    need(set(files)==names and len(names)==45 and manifest['raw_files']==44 and manifest['raw_bytes']==2708690,'Inventory differs')
    need(sum(len(v)for k,v in files.items()if k!='download-manifest.json')==2708690,'Byte count differs')
    need(manifest['original_owner']==dict(pid=627617,start_ticks=344253804,same_start=True,state='Z'),'Wrong owner')
    report=json.loads(files['out/diagnostic.json']);summary=audit(report,a.snapshot)
    faults=[]
    for name,mutate in [('score',lambda r:r.update(performance_score=True)),('cleanup',lambda r:r.update(parent_restored=False)),
        ('resource',lambda r:r['arms'][1]['observed_resource_guard'].update(strict_allowed=False)),
        ('profile-metric',lambda r:r['arms'][0]['functions'][0].update(self_seconds=1.))]:
        bad=copy.deepcopy(report);mutate(bad)
        try:audit(bad,a.snapshot)
        except ValueError:faults.append(name)
        else:raise ValueError('Fault accepted: '+name)
    summary.update(raw_files=44,raw_bytes=2708690,original_owner=manifest['original_owner'],
        original_manifest_sha256=MANIFEST,original_diagnostic_sha256=DIAGNOSTIC,rejected_faults=faults,
        opaque_cache_SO_cubin_preserved_not_loaded=True)
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=buffer,mode='wb',mtime=0,compresslevel=9)as z:
        with tarfile.open(fileobj=z,mode='w',format=tarfile.PAX_FORMAT)as archive:
            for name,raw in sorted(files.items()):
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.uid=info.gid=info.mtime=0;info.uname=info.gname='';archive.addfile(info,io.BytesIO(raw))
    raw=buffer.getvalue();need(len(raw)<=2*1024*1024,'Publisher archive too large')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'profile-v2-raw.tar.gz').write_bytes(raw)
    (a.out/'download-manifest.json').write_bytes(files['download-manifest.json']);(a.out/'summary.json').write_bytes(encode(summary))
    (a.out/'archive-manifest.json').write_bytes(encode(dict(archive='profile-v2-raw.tar.gz',bytes=len(raw),sha256=sha(raw),
        members=45,summary_sha256=sha(encode(summary)),original_manifest_sha256=MANIFEST,files=manifest['files'])))
    need('torch'not in sys.modules and'triton'not in sys.modules,'Framework imported')
    print(json.dumps(dict(passed=True,cpu_only=True,archive_sha256=sha(raw),archive_bytes=len(raw),summary_sha256=sha(encode(summary)))))


if __name__=='__main__':main()
