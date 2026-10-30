"""One post-numeric ABBA successor; 618 numeric and the accepted 522c timing body are reused."""
import ast
import json
from pathlib import Path
import sys
import biv150_konly_q32_w8_numeric as adapter
_framework=adapter.framework()
n,numeric=_framework.source,_framework.contract

ROOT=n.ROOT
BASE=n.BASE
PARENT='618acfef826acfcf2eaff6293d3fd3252681e94d'
# Actual618 originals and Root's independent full-array review; no timing grant.
NUMERIC_REVIEW_SHA='b9f32e2ab2e81edd2b54129f42a5fd0efb4bbab8c4d183378c82a599f6fffacf'
NUMERIC_MANIFEST_SHA='3251b74a5f82c067edbc5c2ba314b3f2703877d91be75baa0bd068dcf8651bac'
NUMERIC_SUMMARY_SHA='46b213ed4cf55a087bfd38fa0cb4e7b2cdba74d19e446cfbc8dc5698b365e2ee'
NUMERIC_RECEIPT_SHA='f3d1f42fbfede9f416d97c80a52819762952ddf965a4a22ef9086e5e1ca85ad1'
NUMERIC_RAW_FILES=92
NUMERIC_RAW_BYTES=194116802
PREFLIGHT_SHA='9a7832afadd9dd0ace1104ccc28773e9273b854cc95662f884b58735cbe53db6'
need=n.need
sha=n.sha
ROLES=numeric.ROLES
MODES=numeric.MODES
STAGES=numeric.STAGES
WARPS=numeric.WARPS
full_output_metrics=numeric.full_output_metrics
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
    return n.replace(old[begin:end],[("('q32','q64','q64','q32')","('w16','w8','w8','w16')"),
        ("Q32_over_Q64=average['q32']/average['q64']","W16_over_W8=average['w16']/average['w8']")])


def run_source():
    return n.replace(n.run_source(),[
        ('        # No timing or timing warmups in this numerical-only frontend.\n',timing_block()),
        ('completed=True,performance_measured=False,','completed=True,performance_measured=True,'),
        ('timing_scope=None,reader_ABBA_pending=True,full_service_verified=False',
         "timing_scope='CUDA Graph complete-reader or unchanged-writer+complete-reader; no Python helper/service cost',reader_ABBA_pending=False,ABBA_pending=False,full_service_verified=False"),
        ('Root_numeric_admission=True,Root_numeric_admission_receipt_sha256=a.admission_sha256,',
         'Root_numeric_admission=True,Root_timing_admission=True,Root_final_numeric_review_confirmed=True,Root_queue_closed_confirmed=True,timing_requested=True,Root_timing_receipt_sha256=a.admission_sha256,Root_final_numeric_review_sha256=a.numeric_review_sha256,')])


def plan():
    value=numeric.plan()
    value.update(schema='cjj-q32-w16-w8-single-ABBA-pending-v1',parent=PARENT,
        numeric_evidence_review_sha256=NUMERIC_REVIEW_SHA,numeric_evidence_manifest_sha256=NUMERIC_MANIFEST_SHA,
        derived_timing_driver_sha256=sha(run_source()),Root_timing_admission=False,Root_final_numeric_review_confirmed=False,
        Root_queue_closed_confirmed=False,timing_requested=False,
        timing_pattern=1,timing_total_KV=6143,timing_prefix=4095,aligned6144_timing_measured=False,
        ABBA_order=['w16','w8','w8','w16'],ABBA_rounds=1,captured_steps=4,
        warm_graph_replays_per_arm_mode=8,warm_complete_steps_per_arm_mode=32,
        timed_graph_replays_per_block=16,complete_steps_per_event_block=64,
        blocks_per_mode=4,event_blocks_total=8,
        host_wall='16 graph replay submissions plus end-event synchronization, divided by64; validation/readback excluded',
        event_scope='64 complete device reader steps, or64 unchanged-writer+reader steps; no Python JIT/helper/service claim',
        before_timing='rerun all8 numeric/Graph/side stages, full4M w16-control hard gate, original FP64/BF16 gates and full arrays',
        ABBA_future='conditional only after an actual numeric evidence pin commit and a separate Root timing decision',
        scheduling='Actual numeric originals and Root independent array review must first be pinned in a later commit; then Root must admit a closed serial slot. No automatic queue',
        numeric_originals_and_Root_review_pending=not qualification_fields_known())
    return value


def qualification_fields_known():
    hashes=(NUMERIC_REVIEW_SHA,NUMERIC_MANIFEST_SHA,NUMERIC_SUMMARY_SHA,NUMERIC_RECEIPT_SHA)
    return (all(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value) for value in hashes)
        and type(NUMERIC_RAW_FILES) is int and NUMERIC_RAW_FILES>0 and type(NUMERIC_RAW_BYTES) is int and NUMERIC_RAW_BYTES>0)


def qualification_ready():
    need(qualification_fields_known(),'actual numeric originals/Root array review still pending; timing is not admitted')


def evidence(pre,review):
    qualification_ready()
    need(pre['plan']==numeric.plan() and pre['static_review_sha256']==numeric.REVIEW_SHA,'same actual accepted w16/S1 and w8/S1 preflight')
    numeric.check_pair(pre['records'],pre['metadata'])
    need(review['passed'] is True and review['source_commit']==PARENT and
         review['manifest_sha256']==NUMERIC_MANIFEST_SHA and review['summary_sha256']==NUMERIC_SUMMARY_SHA,'exact completed618 numerical evidence')
    need((review['raw_files'],review['raw_bytes'])==(NUMERIC_RAW_FILES,NUMERIC_RAW_BYTES),'same complete numerical originals')
    need(review['Graph_before_fresh_and_changed_counterexample'] is True and review['side_nondefault_stream'] is True,'reviewed Graph/side')
    need(len(review['stage_checks'])==8 and all(v['all_hard_data_checks'] is True for v in review['stage_checks']),'all numeric hard checks')
    need(type(review['original_BF16_quality_passed']) is bool and all(review[k] is False for k in ('precision_gate','model_verified','service_verified','performance_measured')),'actual BF16 diagnostic and qualification labels preserved')
    for stage in review['stage_checks']:
        metric=stage['full_output_reference']
        need(type(metric) is dict and type(metric.get('elements')) is int and metric['elements']==4194304
             and metric.get('finite') is True and metric.get('pointwise_passed') is True
             and type(metric.get('pointwise_violations')) is int and metric['pointwise_violations']==0,
             'independently reviewed complete w16-control comparisons for every stage')
    for role in ROLES:
        item=review['executable_binding'][role]
        need(item['kernel_hash']==numeric.HASHES[role] and item['cubin_sha256']==numeric.CUBINS[role],'same actual executable '+role)


def admission_template(commit):
    return dict(schema='cjj-126-N1-w16-w8-timing-root-admission-v1',Root_timing_approved=False,
        Root_final_numeric_review_confirmed=False,Root_current_queue_closed_no_continuation=False,
        frontend_commit=commit,numeric_source=PARENT,preflight_sha256=PREFLIGHT_SHA,
        final_numeric_review_sha256=NUMERIC_REVIEW_SHA,numeric_manifest_sha256=NUMERIC_MANIFEST_SHA,
        numeric_summary_sha256=NUMERIC_SUMMARY_SHA,prior_numeric_receipt_sha256=NUMERIC_RECEIPT_SHA,
        hashes=numeric.HASHES,cubins=numeric.CUBINS,query_rows=2048,num_seqs=1,stages=STAGES,warps=WARPS,full_output_reference_required=True,
        timing_pattern=1,timing_total_KV=6143,ABBA_rounds=1,timing_modes=list(MODES),
        precision_qualified=False,service_qualified=False,model_qualified=False)


def admit(args,pre):
    qualification_ready()
    need(args.timing is True,'explicit timing opt-in required')
    need(args.preflight_sha256==PREFLIGHT_SHA,'fixed actual two-root proof')
    body=args.numeric_review.read_bytes()
    need(args.numeric_review_sha256==NUMERIC_REVIEW_SHA and sha(body)==NUMERIC_REVIEW_SHA,'final independently replayed numerical review SHA')
    evidence(pre,json.loads(body))
    body=args.admission.read_bytes();need(sha(body)==args.admission_sha256,'independent Root timing receipt SHA')
    actual=json.loads(body)
    flags=('Root_timing_approved','Root_final_numeric_review_confirmed','Root_current_queue_closed_no_continuation')
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
    exec(compile(run_source(),'<618-plus-original522c-single-ABBA>','exec'),ns)
    return ns['run'](args,pre)
