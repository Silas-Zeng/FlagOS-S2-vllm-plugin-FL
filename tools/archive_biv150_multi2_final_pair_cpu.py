"""Read-only CPU replay of final multi2 raw evidence and deterministic split archives."""
import argparse
import ast
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import audit_biv150_multi2_pair_v2 as audit
import summarize_biv150_multi2_evidence_cpu as component

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = '2c79015c8d58ab410261d1eb3333900b1836f6de89c2a8452a7e89cda970f480'
V1_FAILURE = '8cf451c91e7dbb04f6f2d9465335d86b89280ca891c4af9c07c2b47d548d0c4f'
V2_SUCCESS = '136e96135092d3c725b69c9496ad503caf5a81f4cec51ac76b5933c00f2017a5'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(args):
    snapshot=args.snapshot; manifest=audit.read(snapshot/'download-manifest.json')
    need(sha(Path(audit.__file__).read_bytes())==AUDITOR,'Frozen v2 auditor changed')
    files={}
    for record in manifest['files']:
        name=record['relative']; need(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe input path')
        raw=(snapshot/name).read_bytes()
        need(name not in files and len(raw)==record['bytes'] and sha(raw)==record['sha256'],'Original download SHA/length differs')
        files[name]=raw
    need(len(files)==manifest['raw_files']==133 and sum(map(len,files.values()))==manifest['raw_bytes']==11744840,'Incomplete raw download')
    present={str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file()}
    expected=set(files)|{'download-manifest.json'}
    extra_audits={'audits/completed-pair-independent-v1-failure.json':V1_FAILURE,
                  'audits/completed-pair-independent-v2.json':V2_SUCCESS}
    need(present==expected or present==expected|set(extra_audits),'Uninventoried input file')
    for name,expected_sha in extra_audits.items():
        if name in present:need(sha((snapshot/name).read_bytes())==expected_sha,'Extracted audit evidence differs')
    target=audit.read(args.v2_audit)
    need(sha(args.v2_audit.read_bytes())==V2_SUCCESS and target['audit_script_sha256']==AUDITOR,'Original target audit identity differs')
    for key in ('structural_integrity_passed','precision_gate','formal_gates_passed','eligible_for_agent_merge_review_on_frozen_pair'):
        need(target[key] is True,'Target v2 audit did not pass: '+key)
    need(target['automatic_merge_performed'] is False and target['model_rerun'] is False,'Unexpected target action')
    observation=target['post_audit_live_observation']
    need(observation['owner']==dict(pid=603821,same_start=True,state='Z') and
         all(observation[k] is True for k in ('port_9031_closed','gpu_no_running_processes','idle_for_next_component')),'Missing final root idle observation')
    # Replay the unchanged auditor against the downloaded raw files, not its saved summaries.
    local=json.loads(json.dumps(audit.audit(snapshot/'pair')))
    mirrored=copy.deepcopy(local); localroot=snapshot/'pair'
    for arm in ('baseline','candidate'):
        for record in mirrored[arm]['accuracy']['raw_files']:
            record['path']=str(Path(target['root'])/Path(record['path']).relative_to(localroot))
        for record in mirrored[arm]['raw_score_files'].values():
            record['path']=str(Path(target['root'])/Path(record['path']).relative_to(localroot))
    for key,value in mirrored.items():
        if key not in ('utc','root'):
            need(value==target[key],'Target/local raw audit mirror differs: '+key)
    # Only locally generated reference paths and audit timestamp/root differ; raw state/manifest remain exact.
    arms={}
    for label in ('baseline','candidate'):
        arm=mirrored[label]; rawarm=snapshot/'pair'/label; generated=[]
        for module in arm['manifest']['generated_out_modules']:
            raw=(rawarm/module['path']).read_bytes()
            need(sha(raw)==module['sha256'],'Generated output module SHA differs')
            calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Call)]
            need(calls.count('extern_kernels.biv150_small_linear_out')==module['out_call_sites'],'Generated out call-site count differs')
            generated.append(module)
        need(len(generated)==4 and sum(m['out_call_sites'] for m in generated)==11,'Missing actual generated output modules')
        warm=audit.read(rawarm/'warmups.json'); benchmarks={}
        for length in ('4096','16384'):
            score=arm['scores'][length]; recomputed=(score['total_input_tokens']+score['total_output_tokens'])/score['duration']
            need(math.isclose(score['total_token_throughput'],recomputed,rel_tol=1e-12),'Raw TPS arithmetic differs')
            need(len(warm[length])==4 and warm[length][3]['full_workload'] is True,'Missing four warmups')
            benchmarks[length]=dict(raw_score={key:score[key] for key in ('duration','total_token_throughput','mean_ttft_ms',
                'num_prompts','max_concurrency','completed','failed','total_input_tokens','total_output_tokens')},
                recomputed_tps=recomputed,short_warmups=3,full_workload_warmups=1,unique_scored_rounds=1,
                working_gates=arm['working_gates'][length])
        arms[label]=dict(commit=arm['manifest']['commit'],accuracy=arm['accuracy'],benchmarks=benchmarks,
            initial_and_post_semantics=arm['semantics'],actual_runtime=arm['runtime'],
            cleanup_complete=arm['state']['cleanup_complete'],cleanup_remaining=arm['state']['cleanup_remaining'],
            port_9031_closed=arm['state']['port_9031_closed'],production_defaults=arm['manifest']['production_defaults'],
            prefill_multi2_mode=arm['manifest']['prefill_multi2_mode'],loaded_source_fingerprint=arm['manifest']['expected_source_fingerprint'],
            native_library_sha256=arm['manifest']['gate_library_sha256'],initial_tuning_snapshot=arm['manifest']['tuning_snapshot'],
            generated_out_modules=generated)
    failed=ROOT/'docs/evidence/prefill-multi2-20261027/completed-pair-independent-v1-failure.json'
    need(sha(failed.read_bytes())==V1_FAILURE,'Original v1 failure changed')
    files['download-manifest.json']=(snapshot/'download-manifest.json').read_bytes()
    files['audits/completed-pair-independent-v1-failure.json']=failed.read_bytes()
    files['audits/completed-pair-independent-v2.json']=args.v2_audit.read_bytes()
    groups={name:{} for name in ('baseline','candidate','common')}
    for name,raw in files.items():
        label='baseline' if name.startswith('pair/baseline/') else 'candidate' if name.startswith('pair/candidate/') else 'common'
        groups[label][name]=raw
    archives={}; members=[]
    for label,values in groups.items():
        packed=component.archive(values)
        need(len(packed)<=2*1024*1024 and packed==component.archive(values),'Archive too large or nondeterministic')
        with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
            need({m.name for m in tar.getmembers()}==set(values),'Archive member set differs')
            for member in tar.getmembers():
                need(member.isfile() and tar.extractfile(member).read()==values[member.name],'Archive member bytes differ')
        filename=label+'-raw.tar.gz';archives[filename]=packed
        members.extend(dict(relative=name,bytes=len(raw),sha256=sha(raw),archive=filename) for name,raw in sorted(values.items()))
    need(len(members)==136,'Final raw/audit inventory incomplete')
    inventory=dict(raw_download_files=133,raw_download_bytes=11744840,archive_members=136,
        archives=[dict(file=name,bytes=len(raw),sha256=sha(raw)) for name,raw in archives.items()],members=members,
        download_manifest_sha256=sha((snapshot/'download-manifest.json').read_bytes()),
        v1_failure_sha256=V1_FAILURE,v2_success_sha256=V2_SUCCESS,frozen_v2_auditor_sha256=AUDITOR,
        deterministic_tar_metadata=True,gzip_mtime=0)
    summary=dict(cpu_only=True,model_rerun=False,gpu_work=False,source_model_frozen=audit.CANDIDATE,
        summary_script_sha256=sha(Path(__file__).read_bytes()),v2_auditor_sha256=AUDITOR,
        all_133_download_sha_reverified=True,local_raw_full_audit_matches_target=True,
        mirror_normalization='Only generated reference paths map local pair root to target root; utc/root are host metadata; raw state/manifest unmodified.',
        structural_integrity_passed=local['structural_integrity_passed'],precision_gate=local['precision_gate'],
        formal_gates_passed=local['formal_gates_passed'],eligible_for_agent_merge_review_on_frozen_pair=local['eligible_for_agent_merge_review_on_frozen_pair'],
        automatic_merge_performed=False,post_audit_root_observation=observation,arms=arms,
        comparison=local['comparison'],paired_question_changes=local['paired_question_changes'],
        interpretation=dict(single_pair=True,statistical_significance_claimed=False,stable_effect_claimed=False,
            twenty_percent_gain_from_this_change_claimed=False,baseline_correct=102,candidate_correct=101,
            baseline_precision_threshold_passed=True,candidate_precision_threshold_passed=True),
        caveats=['One fixed baseline/candidate pair; +0.373865%/+0.839909% are observations, not a stable or20% effect.',
                 'All105 are preserved; three per-question outcomes differ, with one net fewer candidate correct.',
                 'v1 failure is preserved; v2 changes only validated grid tuple/list representation, no original score/source/tolerance.',
                 'Root owns ready/merge and subsequent serial GPU scheduling; this tool does not access a target/GPU.'])
    args.out.mkdir(parents=False,exist_ok=False)
    for name,raw in archives.items():(args.out/name).write_bytes(raw)
    (args.out/'completed-pair-independent-v2.json').write_bytes(args.v2_audit.read_bytes())
    (args.out/'archive-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(archives=inventory['archives'],local_raw_full_audit_matches_target=True,
                         baseline_correct=102,candidate_correct=101,comparison=local['comparison'],automatic_merge_performed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--v2-audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
