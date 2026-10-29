"""Narrow reader timing suffix for the frozen d39 numeric driver."""
import json
from pathlib import Path
import biv150_konly_multi2_numeric_contract as f
c=f.c
NUMERIC_SOURCE='d39c76f4db8a37b89a4fafe43c714dbeff9506b2'
# Complete d39 diagnostic arrays independently replayed by Root; model gate false.
NUMERIC_MANIFEST='367689567f0d36f600dc2670a81c5281a040c7b0fc8af6406239fbb6818ebc17'
NUMERIC_SUMMARY='bfe95889b93e0928721475e1b5357f605b7bfb05c16bafbbc0119bb9d8fd4eb3'
NUMERIC_REVIEW='2792e7297e3fcd02aa9097bf5eded44703307b73e439f14a5c19ad2e0669a2e3'


def plan():
    return dict(f.plan(),schema='cjj-private-multi2-hoist-reader-timing-v1',
        numeric_source=NUMERIC_SOURCE,numeric_manifest_sha256=NUMERIC_MANIFEST,
        numeric_summary_sha256=NUMERIC_SUMMARY,numeric_review_sha256=NUMERIC_REVIEW,
        actual_numeric_admission_pinned=all(type(p) is str and len(p)==64 and all(x in '0123456789abcdef' for x in p) for p in (NUMERIC_MANIFEST,NUMERIC_SUMMARY,NUMERIC_REVIEW)),
        timing_pattern=1,timing_order=['unhoisted','hoisted','hoisted','unhoisted'],
        captured_steps=4,warm_graph_replays_per_role=8,timed_graph_replays_per_arm=16,
        steps_per_timed_arm=64,append_writer_included=False,explicit_numeric_diagnostic_required=True,
        BF16_proxy_failure_does_not_bypass_hard_gates=True,model_verified=False)


def numeric_admission(raw,review_path):
    pins=(NUMERIC_MANIFEST,NUMERIC_SUMMARY,NUMERIC_REVIEW)
    c.need(all(type(p) is str and len(p)==64 and all(x in '0123456789abcdef' for x in p) for p in pins),'actual complete numeric review pending')
    c.need(raw is not None and review_path is not None,'complete numeric raw/review required')
    raw=Path(raw);review_path=Path(review_path)
    for path,pin in ((raw/'download-manifest.json',NUMERIC_MANIFEST),(raw/'out/summary.json',NUMERIC_SUMMARY),(review_path,NUMERIC_REVIEW)):
        c.need(c.sha(path.read_bytes())==pin,'actual numeric admission bytes '+path.name)
    manifest=json.loads((raw/'download-manifest.json').read_text())
    c.need(manifest['raw_files']==len(manifest['files']) and manifest['raw_bytes']==sum(r['bytes'] for r in manifest['files']),'numeric archive counts')
    for row in manifest['files']:
        path=raw/row['relative'];c.need(path.resolve().is_relative_to(raw.resolve()),'numeric raw path escape');body=path.read_bytes()
        c.need(len(body)==row['bytes'] and c.sha(body)==row['sha256'],'complete numeric raw '+row['relative'])
    s=json.loads((raw/'out/summary.json').read_text());r=json.loads(review_path.read_text())
    c.need(s['commit']==r['source_commit']==NUMERIC_SOURCE and r['summary_sha256']==NUMERIC_SUMMARY and r['manifest_sha256']==NUMERIC_MANIFEST,'complete numeric identity')
    c.need(s['completed'] is True and s['errors']==[] and all(s[k] is True for k in ('numeric_verified','Graph_verified','side_stream_verified')),'complete original numeric/Graph/side gates')
    c.need(all(r[k] is True for k in ('review_completed','all_file_hashes_checked','all_available_full_outputs_bitwise','Graph_B_fresh_eager_match_verified','side_C_verified')),'independent full array and changed-input review')
    expected=[(0,'eager-A'),(0,'Graph-A'),(1,'Graph-B-first'),(1,'eager-B-after-Graph'),(2,'side-C')]
    c.need([(x['pattern'],x['label']) for x in s['stages']]==expected,'complete original stage order')
    c.need(all(x['mode']=='reader' and x['numeric_diagnostic'] is True and all(v is True for v in x['hard_gates'].values()) and x['requested_component_gates_passed'] is True for x in s['stages']),'diagnostic never bypasses hard gates')
    c.need(all(x['all_available_hard_data_checks'] is True for x in r['stage_checks']) and [(x['pattern'],x['label']) for x in r['stage_checks']]==expected,'independent five-stage hard data')
    c.need(all(s[k] is False and r[k] is False for k in ('precision_gate','service_verified','performance_measured')),'numeric evidence is not model or performance qualification')
    return dict(source=NUMERIC_SOURCE,manifest_sha256=NUMERIC_MANIFEST,summary_sha256=NUMERIC_SUMMARY,
        review_sha256=NUMERIC_REVIEW,hard_numeric_Graph_side_verified=True,
        original_BF16_quality_passed=s['original_BF16_quality_passed'],precision_gate=False,service_verified=False)


def timing_suffix():
    original=f.source.original('probe_biv150_konly_q64_screen.py')
    start=original.index('        for mode in f.MODES:\n            with torch.cuda.stream(stream):\n                for role in f.ROLES:\n                    for _ in range(8):replay(role,mode)')
    stop=original.index('        result.update(completed=True,performance_measured=True',start)
    block=original[start:stop];before=block
    edits=(("('q32','q64','q64','q32')","('unhoisted','hoisted','hoisted','unhoisted')"),
           ("Q32_over_Q64=average['q32']/average['q64']","unhoisted_over_hoisted=average['unhoisted']/average['hoisted']"))
    for a,b in edits:c.need(block.count(a)==1,'original timing role edit');block=block.replace(a,b)
    inverse=block
    for a,b in reversed(edits):inverse=inverse.replace(b,a)
    c.need(inverse==before,'original event/wall/guard/fingerprint timing suffix preserved')
    return block,dict(original_sha256=c.sha(before),adapted_sha256=c.sha(block),edits=edits)


def adapted_driver():
    original=f.source.adapted_driver()[0];text=original;edits=[]
    def edit(before,after):
        nonlocal text
        c.need(text.count(before)==1,'unique timing edit '+repr(before[:60]));at=text.index(before)
        text=text[:at]+after+text[at+len(before):];edits.append(dict(offset=at,before=before,after=after))
    edit('Private multi2 numeric screen; original math and no performance execution.','Private multi2 complete-reader ABBA; explicit diagnostic, model qualification false.')
    edit('import biv150_konly_multi2_numeric_contract as f','import biv150_konly_multi2_numeric_contract as f\nimport biv150_konly_multi2_timing_source as timing')
    edit("    a.out.mkdir(parents=True)","    c.need(a.numeric_diagnostic,'reader timing requires explicit numeric diagnostic')\n    a.out.mkdir(parents=True)")
    edit("(a.out/'actual-numeric-driver.py').write_text(f.source.adapted_driver()[0])","(a.out/'actual-timing-driver.py').write_text(timing.adapted_driver()[0])")
    edit('result=dict(f.plan(),commit=a.commit','result=dict(timing.plan(),commit=a.commit')
    edit('actual_preflight=pre,reader_launch_attempts=0','actual_preflight=pre,prior_numeric_admission=a.numeric_proof,reader_launch_attempts=0')
    edit('inputs={};references={};gold_outputs={};uploaded={}','inputs={};references={};gold_outputs={};uploaded={};phase_cpu={};phase_readonly={}')
    edit("            def stage(label,mode,use_graph=False):","            phase_cpu[pattern]=cpu;phase_readonly[pattern]=immutable\n            def stage(label,mode,use_graph=False):")
    restore="""        # C used the same buffers; restore the exact archived B inputs before timing.
        cpu=phase_cpu[1];owner.copy_(uploaded[1])
        for dest,source in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]:tensors[dest].copy_(cpu[source])
        torch.cuda.synchronize();immutable=phase_readonly[1]
        c.need(fingerprints()==immutable and byte_cpu(owner)==byte_cpu(uploaded[1]),'complete B inputs restored before reader timing')
"""
    block,_=timing_suffix()
    edit('        result.update(completed=True,numeric_verified=True,Graph_verified=True,side_stream_verified=True,performance_measured=False,precision_gate=False,service_verified=False,',restore+block+'        result.update(completed=True,numeric_verified=True,Graph_verified=True,side_stream_verified=True,performance_measured=True,precision_gate=False,service_verified=False,')
    edit("timing_scope='not run; future direct unhoisted/hoisted reader ABBA must include Q quantization/staging/QK/softmax/PV/epilogue'","timing_scope='reader only: complete Q quantization/loads/staging/QK/softmax/PV/output epilogue; no writer or service claim'")
    edit("p.add_argument('--references',type=Path);a=p.parse_args()","p.add_argument('--references',type=Path);p.add_argument('--numeric-raw',type=Path);p.add_argument('--numeric-review',type=Path);a=p.parse_args()")
    edit('        pre=f.preflight(a.raw,a.review,a.references)','        a.numeric_proof=timing.numeric_admission(a.numeric_raw,a.numeric_review)\n        pre=f.preflight(a.raw,a.review,a.references)')
    edit("'CPU preflight rejects device options');value=pre","'CPU preflight rejects device options');value=dict(pre,prior_numeric_admission=a.numeric_proof)")
    edit("a.out,a.raw,a.review,a.references)),'paths/options need explicit mode');value=f.plan()","a.out,a.raw,a.review,a.references,a.numeric_raw,a.numeric_review)),'paths/options need explicit mode');value=timing.plan()")
    inverse=text
    for e in reversed(edits):
        at=e['offset'];c.need(inverse[at:at+len(e['after'])]==e['after'],'timing inverse location');inverse=inverse[:at]+e['before']+inverse[at+len(e['after']):]
    c.need(inverse==original,'exact original d39 numeric driver recovered')
    return text,edits
