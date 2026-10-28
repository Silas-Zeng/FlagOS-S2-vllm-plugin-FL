#!/usr/bin/env python3
"""CPU-only adaptation of the existing saved Math105 and PR109 smoke auditors."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re

SMOKE = Path('/private/tmp/check_cjj_native109_service_smoke_actual_20261028.py')
SMOKE_SHA = 'de2af0073be25d399b54f70fcb902f4145596a33c50bbc1afe3ae8ab189e7c9f'
MATH = Path('/private/tmp/cjj-native-k8-service-math105-root-review-20261028.py')
MATH_SHA = '0cda2839bd6953fea844317ad1612293cfde4a1203833b89c792d66872c9b353'
HELPER = Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def replace_once(source, old, new):
    need(source.count(old) == 1, 'Bounded evaluator adaptation: ' + old)
    return source.replace(old, new)


def audit(a):
    need(sha(SMOKE) == SMOKE_SHA and sha(MATH) == MATH_SHA, 'Existing audit source dependency SHA')
    spec = importlib.util.spec_from_file_location('pr109_saved_smoke_stdlib', SMOKE)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)  # Definitions only: stdlib; no smoke execution.
    root, out = a.raw.resolve(), a.raw.resolve()/'out'
    d, m, state = read(root/'download-manifest.json'), read(out/'manifest.json'), read(out/'state.json')
    need(d['read_only'] is True and d['model_rerun'] is False, 'Read-only transport scope')
    need(d['original_owner'] == dict(pid=44037, start_ticks=355048326, same_start=True, state='Z'), 'Terminal original owner')
    need(d['raw_files'] == len(d['files']) == 41 and d['raw_bytes'] == sum(x['bytes'] for x in d['files']) == 11198363, 'Actual inventory size')
    paths = set()
    for e in d['files']:
        p = Path(e['relative'])
        need(not p.is_absolute() and '..' not in p.parts and str(p) not in paths, 'Original relative path')
        paths.add(str(p)); q = root/p
        need(q.stat().st_size == e['bytes'] and sha(q) == e['sha256'], 'Original bytes/SHA: ' + str(p))
    need({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == paths | {'download-manifest.json'}, 'Exact collected file set')
    owner = read(root/'exports/cjj-agent-native109-math105-v1-20261028-owner.json')
    need((owner['pid'],owner['start_ticks'],owner['source_commit']) == (44037,355048326,smoke.COMMIT), 'Owner source and start')
    need(m['source_commit'] == smoke.COMMIT and sha(out/'executed-controller.py') == m['controller_sha256'] == smoke.CONTROLLER, 'Actual executed controller')
    need(m['helper_sha256'] == sha(HELPER) == smoke.HELPER, 'Original helper94db')
    source = smoke.GitSource(a.repo)
    for relative, pin in m['source_hashes'].items():
        need(hashlib.sha256(source.read(relative)).hexdigest() == pin, 'Immutable tested source: ' + relative)
    compact_sources = {Path(k).name:v for k,v in m['source_hashes'].items() if k.startswith('vllm_fl/patches/biv150_konly_compact_') and (k.endswith('.py') or k.endswith('_sources.json'))}
    implementation = hashlib.sha256(smoke.canonical(compact_sources).encode()).hexdigest()
    expected = m['expected_compile_semantics']
    need(implementation == expected['konly_compact']['implementation_sha256'], 'Independent compact source digest')
    need(m['command'] == smoke.SERVICE and m['official_service_parameters_unchanged'], 'Fixed official service command')
    need(m['actual_resolution']['origin'] == m['source']+'/vllm_fl/__init__.py', 'Actual source package root')
    need(m['native_sha256'] == smoke.GATE and m['native_gqa_library_sha256'] == smoke.STOCK and m['native_decode_library']['sha256'] == smoke.COMPACT, 'Distinct gate/stock/compact libraries')
    need([e['sha256'] for e in m['tuning_snapshot']] == ['5764f125233211d8ed8369f0bcdc6a08a58a00c9a3470bfaa36be0af7665ca55','2c5aa448065453188f1ff1794d8e929bacb63dcad84a600e086ae05165821314'], 'Original v6 snapshot')
    log_bytes = (out/'server.log').read_bytes(); log = log_bytes.decode()
    c3 = smoke.rows(log, 'BIV150_COMPILE_SEMANTICS')
    need(len(c3) == 2 and all(x == expected for x in c3), 'Actual worker C3')
    need(not any(x in log for x in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK')), 'Existing C3/scope gates')
    library = dict(m['native_decode_library'], qualification=expected['konly_compact']['native_decode']['qualification'])
    installs = smoke.rows(log, 'BIV150_KONLY_NATIVE_DECODE_INSTALL')
    native_source = 'vllm_fl/patches/biv150_konly_compact_native.py'
    need(len(installs) == 1 and m['native_identity']['module_sha256'] == m['source_hashes'][native_source], 'Actual native source identity')
    for x in installs:
        need(x['library'] == library and x['operator'] == smoke.OPERATOR and x['schema'] == smoke.SCHEMA and x['module_sha256'] == m['source_hashes'][native_source] and x['module_file'] == m['source']+'/'+native_source and x['torch_file'] == m['native_identity']['torch']['file'], 'Actual loaded module/library/schema/import')
        need(x['installed'] and x['enabled'] and x['loaded_library_confirmed'] and not x['runtime_compile'] and x['format'] == smoke.FORMAT and x['physical_page_bytes'] == 12544, 'Actual compact native installation')
    stock = smoke.rows(log, 'BIV150_NATIVE_GQA_INSTALL')
    need(stock and all(x['installed'] and x['overlap_v'] and x['library_sha256'] == smoke.STOCK for x in stock), 'Preserved actual stock0c66 installation')
    final_runtime = m['actual_runtime_after_supplemental64']
    for key in ('actual_runtime','actual_runtime_after_math','actual_runtime_after_supplemental64'):
        runtime = m[key]
        need(runtime['actual_compile_semantics'] == c3 and runtime['native_decode']['installations'] == installs and runtime['native_decode']['actual_imports_verified'] and runtime['native_decode']['quality_covered'] is False, 'C3/runtime snapshot: ' + key)
        for marker, records in runtime['preserved_parent_installations'].items():
            observed = smoke.rows(log, marker)
            if marker == 'BIV150_NATIVE_W8A8_INSTALL':
                need(all(x['layers'] == ('gate_up_proj',) for x in observed), 'Checked Python tuple serialization')
                observed = [dict(x,layers=list(x['layers'])) for x in observed]
            need(observed == records and all(x['installed'] for x in records), 'Preserved parent installer: ' + marker)
    native_events = [dict(marker=marker,event=e) for marker in ('BIV150_KONLY_NATIVE_DECODE_SELECTED','BIV150_KONLY_NATIVE_DECODE_CAPTURED') for e in smoke.rows(log, marker)]
    need(native_events == final_runtime['native_decode']['events'], 'All native records mirror actual log')
    for e in native_events:
        x = e['event']
        need(x['library'] == library and x['format'] == smoke.FORMAT and x['operator'] == smoke.OPERATOR and x['physical_page_bytes'] == 12544 and x['native_launch_returned'] and x['host_or_capture_event_not_Graph_replay_count'], 'Native host/capture source and library boundary')

    # Reuse the complete old evaluator consistency logic. Only historical observed
    # correctness/truncation counts are removed; all protocol/numeric checks stay.
    old = MATH.read_text()
    block = old[old.index("base=m['source']"):old.index("coverage=parse('BIV150_NATIVE_GRAPH_COVERAGE')")]
    block = replace_once(block, ' assert correct==(103 if parallel==8 else 100)', ' # Observe the current candidate count; do not target a prior run.')
    block = replace_once(block, ' assert stopped==limit_rows and len(stopped)==(0 if parallel==8 else 2)', ' assert stopped==limit_rows')
    block = replace_once(block, " assert Counter(x['stop_reason'] for x in rows)==({'stop':105} if parallel==8 else {'stop':103,'max_tokens':2})", " assert sum(Counter(x['stop_reason'] for x in rows).values())==105")
    ns = dict(m=m,d=d,out=out,root=root,read=read,sha=sha,json=json,re=re,ast=ast,Counter=Counter,hashlib=hashlib)
    exec(compile(block,str(MATH)+'::<current counts observed>','exec'),ns)
    results, maps = ns['results'], ns['question_maps']
    response_ids = set()
    for phase, qmap in maps.items():
        for q, row in qmap.items():
            rid = row['model_output']['id']
            need(type(rid) is str and rid and rid not in response_ids, 'Distinct evaluator API response IDs')
            response_ids.add(rid)
    need(len(response_ids) == 210, 'Two independent 105-response runs')
    def prompt(messages):
        return [{k:v for k,v in row.items() if k not in ('id','perf_metrics')} for row in messages if row.get('source') != 'generate']
    for q, row in maps['official_parallel8'].items():
        need(prompt(row['messages']) == prompt(maps['supplemental_parallel64'][q]['messages']), 'Same complete input prompt: ' + q)
    official = read(out/'math500-level3/command.json')
    supplemental = read(out/'supplemental-native-math105-parallel64/math500-level3/command.json')
    helper_original = read(out/'supplemental-native-math105-parallel64/math500-level3/helper-original-command.json')
    need([i for i,(x,y) in enumerate(zip(official,supplemental,strict=True)) if x!=y] == [15,21], 'Only parallelism and output directory changed')
    helper_expected = list(supplemental); helper_expected[15] = '8'
    need(helper_original == helper_expected and m['official_Math105_parallel'] == 8 and m['supplemental_Math105_parallel64_requested'], 'Actual helper command retains official batch8')
    for phase, verdict in (('official_parallel8',state['accuracy']),('supplemental_parallel64',state['supplemental64_raw_verdict'])):
        need(verdict['score'] == results[phase]['reported_rounded_accuracy'] and verdict['metric_count'] == verdict['predictions'] == verdict['unique_questions'] == 105, 'Controller/evaluator report consistency')
    window = m['supplemental64_request_window']; data = (out/'supplemental-native-math105-parallel64-runtime-window.log').read_bytes()
    need(len(data) == window['bytes'] and hashlib.sha256(data).hexdigest() == window['sha256'] and data == log_bytes[window['log_byte_offset']:window['log_byte_offset']+len(data)], 'Exact supplemental request window')
    need(window['native_events'] == [] and window['Graph_replays_counted'] is False, 'No inferred native replay coverage')
    need(m['graph_coverage_bootstrap']['enabled'] is False and not smoke.rows(log,'BIV150_NATIVE_GRAPH_COVERAGE'), 'Graph observer genuinely disabled')
    need(m['compact_final_exports'] == [] and m['compact_final_export_found'] is False, 'Final stats absence retained')
    cleanup, graceful = read(out/'owned-cleanup.json'), read(out/'graceful-api-shutdown.json')
    need(cleanup['complete'] and cleanup['remaining_owned'] == cleanup['unobserved_same_session'] == cleanup['signal_errors'] == [] and cleanup['pid_reuse_never_signalled'] and not cleanup['group_signal_used'], 'Owned process cleanup')
    need(graceful['identity_verified'] and graceful['API_exited'] and graceful['returncode'] == 0, 'Actual graceful API shutdown')
    need(state['completed'] and state['cleanup_complete'] and state['errors'] == [] and state['phase'] == 'completed', 'Terminal state')
    need(state['native_quality_covered'] is False and state['native_model_precision_verified'] is False and not state['throughput_measured'], 'Qualification distinctions')
    need(state['model_precision_verified'] == results['official_parallel8']['metric_gate_095_passed'], 'Official precision gate matches complete grades')
    need('EngineDeadError' not in log, 'Unexpected engine failure')
    return dict(schema=1,audit_passed=True,local_CPU_only=True,raw_root=str(root),raw_files=41,raw_bytes=11198363,download_manifest_sha256=sha(root/'download-manifest.json'),source_commit=smoke.COMMIT,source_hashes_verified=len(m['source_hashes']),source_symlinks_followed=source.symlinks,
        dependencies=dict(existing_smoke_auditor_sha256=SMOKE_SHA,existing_Math105_auditor_sha256=MATH_SHA,helper_sha256=smoke.HELPER),
        identities=dict(controller_sha256=smoke.CONTROLLER,stock_native=smoke.STOCK,compact_native=smoke.COMPACT,w8a8_gate=smoke.GATE,compact_implementation=implementation,C3=expected['sha256'],C3_matching_records=len(c3),actual_loaded_SO_receipts_verified=True,SO_bodies_saved=False),
        official_service_command=smoke.SERVICE,official_evaluation_command=official,supplemental_evaluation_command=supplemental,only_supplemental_changes=['parallel8 ->64','output directory'],results=results,identical_question_set_and_input_prompts=True,separate_unique_API_response_ids=210,
        native_coverage=dict(bootstrap_enabled=False,host_capture_event_counts=dict(Counter(x['marker'] for x in native_events)),supplemental_runtime_window_native_host_events=0,Graph_replays_observed=False,actual_request_frequency_proven=False,native_quality_covered=False,native_model_precision_verified=False,final_stats_saved=False,final_overflow_and_errors_known=False),
        cleanup=dict(complete=True,original_owner=d['original_owner'],API_returncode=0,errors=[],EngineDeadError=False,post_GPU_idle_observation_added=False),
        official_Math105_precision_gate=results['official_parallel8']['metric_gate_095_passed'],supplemental_diagnostic_metric_gate=results['supplemental_parallel64']['metric_gate_095_passed'],native_request_quality_gate=False,throughput_verified=False,full_acceptance=False,
        limitations=['All105 saved per-question evaluator grades are recomputed against the aggregate report, not independently mathematically regraded.','Temperature1.0/top_p.95/max32768 and all original protocol parameters remain fixed; count differences are observed without causal attribution.','Supplemental64 is diagnostic and does not replace the official parallel8 result.','Max-token truncation and incorrect-question records are retained separately from failed_predictions; zero failed does not imply zero truncation.','No Graph bootstrap, final lifetime statistics, or saved replay proof: native host/capture events do not establish real request bucket coverage.','This is the samefc25/40f9 service result only; no prior component, model, Graph or performance qualification is transferred.','Library load identities are checked from saved actual source/load receipts; fresh remote SO bytes were not read by this audit.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path('/private/tmp/cjj-native109-math105-v1-raw-20261029'))
    p.add_argument('--repo',type=Path,default=Path('/Users/chenjunjie/Documents/flagos/repos/FlagOS-S2-vllm-plugin-FL'))
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args(); need(not a.out.exists(),'New output report path required')
    report = audit(a)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print(json.dumps(dict(audit_passed=True,report=str(a.out),report_sha256=sha(a.out),script_sha256=sha(__file__),results={k:{n:v[n] for n in ('correct','exact_accuracy','max_token_truncation_count','empty_final_text_count','request_error_count')} for k,v in report['results'].items()}),sort_keys=True))


if __name__ == '__main__':
    main()
