"""One post-numeric ABBA successor; frozen efd/522c source bodies are reused."""
import ast
import json
from pathlib import Path
import sys
import biv150_konly_q32_s2_numeric_source as n
import biv150_konly_q32_s2_numeric_contract as numeric

ROOT=n.ROOT
BASE=n.BASE
PARENT='efd012eb7f31a70ad0a37bc5a05b9aa9fc8f0efe'
NUMERIC_REVIEW_SHA='fbf84f3552b2eb7350252c95d881e94eba38ec8775a42b85051ea9602274276d'
NUMERIC_MANIFEST_SHA='d59fc07263b6ef1605a55411e6d01628b5f81416ff09857f2cbcba6db98e636d'
NUMERIC_SUMMARY_SHA='a87961d8f3e7f7fba6474a6371f3ee2e9c8e5f3656821db5cdeacc2956c2add6'
PREFLIGHT_SHA='1fec73ce1b965b82b33a4d1f27a6f051fadc34fb8448002707f63ca3c2a49f16'
NUMERIC_RECEIPT_SHA='03b27d0db683318b31ae0214ecbca20c5bc2c1f736e7ae6e38d14a52b5fd36bf'
need=n.need
sha=n.sha
ROLES=numeric.ROLES
MODES=numeric.MODES
STAGES=numeric.STAGES
LENGTHS=numeric.LENGTHS
LOADED_COMPILER=numeric.LOADED_COMPILER
RESOURCES=numeric.RESOURCES
WRITER=numeric.WRITER
prepare_cache=numeric.prepare_cache
export_references=numeric.export_references
load_reference=numeric.load_reference
snapshot_module=numeric.snapshot_module
check_loaded=numeric.check_loaded
execution_sources=numeric.execution_sources


def timing_block():
    old=n.original('probe_biv150_konly_q64_screen.py')
    begin=old.index('        for mode in f.MODES:\n            with torch.cuda.stream(stream):\n                for role in f.ROLES:\n                    for _ in range(8):replay(role,mode)')
    end=old.index('        result.update(completed=True,performance_measured=True,',begin)
    return n.replace(old[begin:end],[("('q32','q64','q64','q32')","('s1','s2','s2','s1')"),
        ("Q32_over_Q64=average['q32']/average['q64']","S1_over_S2=average['s1']/average['s2']")])


def run_source():
    return n.replace(n.run_source(),[
        ('        # No timing or timing warmups in this numerical-only frontend.\n',timing_block()),
        ('completed=True,performance_measured=False,','completed=True,performance_measured=True,'),
        ('timing_scope=None,reader_ABBA_pending=True,full_service_verified=False',
         "timing_scope='CUDA Graph complete-reader or unchanged-writer+complete-reader; no Python helper/service cost',reader_ABBA_pending=False,full_service_verified=False"),
        ('Root_numeric_admission=True,Root_numeric_admission_receipt_sha256=a.admission_sha256,',
         'Root_numeric_admission=True,Root_timing_admission=True,Root_final_numeric_review_confirmed=True,Root_A4K_B4K_pair_closed_confirmed=True,timing_requested=True,Root_timing_receipt_sha256=a.admission_sha256,Root_final_numeric_review_sha256=a.numeric_review_sha256,')])


def plan():
    value=numeric.plan()
    value.update(schema='cjj-q32-S1-S2-single-ABBA-v1',parent=PARENT,
        numeric_evidence_review_sha256=NUMERIC_REVIEW_SHA,numeric_evidence_manifest_sha256=NUMERIC_MANIFEST_SHA,
        derived_timing_driver_sha256=sha(run_source()),Root_timing_admission=False,Root_final_numeric_review_confirmed=False,
        Root_A4K_B4K_pair_closed_confirmed=False,timing_requested=False,
        timing_pattern=1,timing_total_KV=6143,timing_prefix=4095,aligned6144_timing_measured=False,
        ABBA_order=['s1','s2','s2','s1'],ABBA_rounds=1,captured_steps=4,
        warm_graph_replays_per_arm_mode=8,warm_complete_steps_per_arm_mode=32,
        timed_graph_replays_per_block=16,complete_steps_per_event_block=64,
        blocks_per_mode=4,event_blocks_total=8,
        host_wall='16 graph replay submissions plus end-event synchronization, divided by64; validation/readback excluded',
        event_scope='64 complete device reader steps, or64 unchanged-writer+reader steps; no Python JIT/helper/service claim',
        before_timing='rerun all original8 numeric/Graph/side stages with original gates and full arrays',
        scheduling='Root may admit only after the current fresh A4K then B4K pair is terminal; no automatic queue')
    return value


def evidence(pre,review):
    need(pre['plan']==numeric.plan() and pre['static_review_sha256']==numeric.REVIEW_SHA,'same actual S1/S2 preflight')
    numeric.check_pair(pre['records'],pre['metadata'])
    need(review['passed'] is True and review['source_commit']==PARENT and
         review['manifest_sha256']==NUMERIC_MANIFEST_SHA and review['summary_sha256']==NUMERIC_SUMMARY_SHA,'exact completed efd numerical evidence')
    need((review['raw_files'],review['raw_bytes'])==(92,193272247),'same complete numerical originals')
    need(review['Graph_before_fresh_and_changed_counterexample'] is True and review['side_nondefault_stream'] is True,'reviewed Graph/side')
    need(len(review['stage_checks'])==8 and all(v['all_hard_data_checks'] is True for v in review['stage_checks']),'all numeric hard checks')
    need(all(review[k] is False for k in ('original_BF16_quality_passed','precision_gate','model_verified','service_verified','performance_measured')),'original BF16 diagnostic and qualification labels preserved')
    for role in ROLES:
        item=review['executable_binding'][role]
        need(item['kernel_hash']==numeric.HASHES[role] and item['cubin_sha256']==numeric.CUBINS[role],'same actual executable '+role)


def admission_template(commit):
    return dict(schema='cjj-124-N1-S1-S2-timing-root-admission-v1',Root_timing_approved=False,
        Root_final_numeric_review_confirmed=False,Root_current_A4K_B4K_pair_closed_no_continuation=False,
        frontend_commit=commit,numeric_source=PARENT,preflight_sha256=PREFLIGHT_SHA,
        final_numeric_review_sha256=NUMERIC_REVIEW_SHA,numeric_manifest_sha256=NUMERIC_MANIFEST_SHA,
        numeric_summary_sha256=NUMERIC_SUMMARY_SHA,prior_numeric_receipt_sha256=NUMERIC_RECEIPT_SHA,
        hashes=numeric.HASHES,cubins=numeric.CUBINS,query_rows=2048,num_seqs=1,stages=STAGES,
        timing_pattern=1,timing_total_KV=6143,ABBA_rounds=1,timing_modes=list(MODES),
        precision_qualified=False,service_qualified=False,model_qualified=False)


def admit(args,pre):
    need(args.timing is True,'explicit timing opt-in required')
    need(args.preflight_sha256==PREFLIGHT_SHA,'fixed actual two-root proof')
    body=args.numeric_review.read_bytes()
    need(args.numeric_review_sha256==NUMERIC_REVIEW_SHA and sha(body)==NUMERIC_REVIEW_SHA,'final independently replayed numerical review SHA')
    evidence(pre,json.loads(body))
    body=args.admission.read_bytes();need(sha(body)==args.admission_sha256,'independent Root timing receipt SHA')
    actual=json.loads(body)
    flags=('Root_timing_approved','Root_final_numeric_review_confirmed','Root_current_A4K_B4K_pair_closed_no_continuation')
    need(all(actual.get(k) is True for k in flags),'Root timing/final-review/closed-pair confirmation pending')
    need(all(actual.get(k) is False for k in ('precision_qualified','service_qualified','model_qualified')),'no model/precision/service grant')
    expected=dict(admission_template(args.commit),**{k:True for k in flags})
    need(actual==expected,'fixed timing-only source/ABBA scope')
    for paths in pre['runtime_paths'].values():
        need(sha(Path(paths['source']).read_bytes())==paths['source_sha256'],'original role source')
        need(sha(Path(paths['override']).read_bytes())==paths['override_sha256'],'original role override')


def run(args,pre):
    import errno,gzip,inspect,os,socket,subprocess,time
    d=n.dependencies();ns=dict(errno=errno,gzip=gzip,inspect=inspect,os=os,socket=socket,subprocess=subprocess,time=time,
        json=json,Path=Path,sys=sys,c=d.c,f=sys.modules[__name__],compiler=d.compiler,reviewed_cache=d.cache,kwargs=d.kwargs)
    exec(compile(run_source(),'<efd-plus-original522c-single-ABBA>','exec'),ns)
    return ns['run'](args,pre)
