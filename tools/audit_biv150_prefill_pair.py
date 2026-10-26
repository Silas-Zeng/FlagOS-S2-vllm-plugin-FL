"""CPU-only audit of a completed pair and actual default w16 selection; never merge."""
import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(log, marker):
    result = []
    for line in log.splitlines():
        if marker + ' ' not in line:
            continue
        value = line.split(marker + ' ', 1)[1]
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        assert isinstance(parsed, dict), marker
        result.append(parsed)
    return result


def precision(folder):
    reports = list(folder.glob('math500-level3/*/reports/minicpm/math_500.json'))
    assert len(reports) == 1
    report = reports[0]
    review = report.parents[2] / 'reviews/minicpm/math_500_Level 3.jsonl'
    prediction = report.parents[2] / 'predictions/minicpm/math_500_Level 3.jsonl'
    reviews = [json.loads(x) for x in review.read_text().splitlines() if x]
    predictions = [json.loads(x) for x in prediction.read_text().splitlines() if x]
    scores = {x['sample_score']['sample_metadata']['question_id']:
              x['sample_score']['score']['value']['accuracy'] for x in reviews}
    assert len(reviews) == len(predictions) == len(scores) == 105
    assert set(scores) == {x['metadata']['question_id'] for x in predictions}
    assert all(x in (0, 1) for x in scores.values())
    assert all(x['sample_score']['score']['status'] == 'success' for x in reviews)
    assert all(not x['model_output'].get('error') and x['model_output'].get('choices')
               for x in predictions)
    metric = [x for x in read(report)['metrics'] if x['identity']['name'] == 'accuracy']
    correct = sum(scores.values())
    assert len(metric) == 1 and metric[0]['num'] == 105
    assert math.isclose(round(correct / 105, 4), metric[0]['score'], abs_tol=1e-9)
    return dict(correct=correct, count=105, fraction=correct / 105,
                passed=correct / 105 >= .95, question_scores=scores,
                raw_files=[dict(path=str(p), sha256=sha(p))
                           for p in [report, review, prediction]])


def arm(folder, commit):
    state, manifest = read(folder / 'state.json'), read(folder / 'manifest.json')
    assert state['complete'] and state['cleanup_complete']
    assert state['semantic_passed'] and state['post_score_semantic_passed']
    assert state['commit'] == manifest['commit'] == commit
    assert manifest['loaded_source_fingerprint_verified'] and manifest['production_defaults']
    assert manifest['native_mode'] == 'on'
    assert all(manifest[k] for k in ['run_accuracy', 'run_throughput',
                                    'full_workload_warmup', 'post_score_semantic_check'])
    assert manifest['stable_small_o'] == manifest['out_lowering'] == manifest['large_linear'] == '1'
    assert not any(name in manifest['env'] for name in [
        'VLLM_FL_BIV150_SMALL_OUT_LOWERING', 'VLLM_FL_BIV150_LARGE_LINEAR',
        'VLLM_FL_BIV150_PREFILL_Q32'])
    accuracy = precision(folder)
    assert accuracy['passed'] == state['precision_passed']
    gates = {}
    warmups = read(folder / 'warmups.json')
    for length, count in [('4096', 256), ('16384', 128)]:
        warm = warmups[length]
        assert len(warm) == 4 and warm[-1]['full_workload'] is True
        for row in warm[:3]:
            assert row['completed'] == row['num_prompts'] == 64 and row['failed'] == 0
            assert row['total_input_tokens'] == int(length) * 64
            assert row['total_output_tokens'] == 128 * 64
        assert warm[-1]['completed'] == warm[-1]['num_prompts'] == count
        assert warm[-1]['failed'] == 0
        assert warm[-1]['total_input_tokens'] == int(length) * count
        assert warm[-1]['total_output_tokens'] == 1024 * count
        score = state['scores'][length]
        assert all(score[k] == value for k, value in dict(
            num_prompts=count, completed=count, failed=0, max_concurrency=64,
            total_input_tokens=int(length) * count, total_output_tokens=1024 * count).items())
        assert math.isfinite(score['total_token_throughput']) and score['total_token_throughput'] > 0
        limit = state['working_gates'][length]['working_mean_ttft_limit_ms']
        ttft = score['mean_ttft_ms']
        gates[length] = dict(mean_ttft_ms=ttft, working_ttft_limit_ms=limit,
                             passed=math.isfinite(ttft) and 0 < ttft <= limit)
    cache = {}
    for length in ['4096', '16384']:
        selected = read(folder / f'score-{length}-selected-config.json')
        assert selected['changed'] == (selected['before'] != selected['after'])
        cache[length] = dict(selected_config_changed=selected['changed'], scope=selected['scope'],
                             new_metadata=read(folder / f'score-{length}-new-kernels.json'))
    return dict(state=state, manifest=manifest, accuracy=accuracy, ttft=gates, cache=cache,
                evidence_sha256={p.name: sha(p) for p in [folder / 'state.json',
                    folder / 'manifest.json', folder / 'server.log']})


def audit(root, baseline_commit, candidate_commit):
    control = read(root / 'state.json')
    assert control['phase'] == 'completed', 'Pair has not completed; do not infer a result from partial state'
    baseline = arm(root / 'baseline', baseline_commit)
    candidate = arm(root / 'candidate', candidate_commit)
    for key in ['command', 'tuning_snapshot', 'gate_library_sha256', 'driver_sha256']:
        assert baseline['manifest'][key] == candidate['manifest'][key], key
    assert baseline['manifest']['gate_library_sha256'] == '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
    baseline_log = (root / 'baseline/server.log').read_text()
    log = (root / 'candidate/server.log').read_text()
    assert not events(baseline_log, 'BIV150_PREFILL_Q32_SELECTED')
    installs = events(log, 'BIV150_PREFILL_Q32_INSTALL')
    expected = dict(installed=True, block_m=256, block_q=32, num_warps=16,
                    num_stages=1, tile_size=64, minimum_query_rows=1024, num_sequences=1)
    assert any(all(row.get(k) == v for k, v in expected.items()) for row in installs)
    selected = events(log, 'BIV150_PREFILL_Q32_SELECTED')
    assert selected, 'Candidate feature never selected in the actual model'
    assert 'BIV150_PREFILL_Q32_FALLBACK' not in log, 'Candidate permanently fell back'
    records = {x['candidate_hash']: x for x in events(log, 'BIV150_Q_FRAGMENT_SELECTED')
               if x.get('mode') == 'selected'}
    for row in selected:
        assert all(row.get(k) == v for k, v in expected.items()
                   if k not in ['installed', 'minimum_query_rows'])
        assert row['rows'] >= 1024
        compiled = records[row['hash']]
        meta = compiled['candidate_resources']['metadata']
        assert (meta['num_warps'], meta['num_stages'], meta['shared']) == (16, 1, 65536)
        assert compiled['mechanism']['mechanism_pass']
        assert compiled['mechanism']['llvm_q_memory_reads_in_loop'] == 0
    comparison = {}
    for length in ['4096', '16384']:
        a = baseline['state']['scores'][length]['total_token_throughput']
        b = candidate['state']['scores'][length]['total_token_throughput']
        change = b / a - 1
        assert math.isclose(control['comparison'][length]['observed_relative_change'], change,
                            rel_tol=1e-10, abs_tol=1e-12)
        comparison[length] = dict(baseline_tps=a, candidate_tps=b,
                                  observed_relative_change=change, higher=b > a)
    changes = [dict(question_id=k, baseline=v, candidate=candidate['accuracy']['question_scores'][k])
               for k, v in baseline['accuracy']['question_scores'].items()
               if v != candidate['accuracy']['question_scores'][k]]
    qualified = (all(x['accuracy']['passed'] and all(g['passed'] for g in x['ttft'].values())
                     for x in [baseline, candidate]) and all(x['higher'] for x in comparison.values()))
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root=str(root),
                cpu_only=True, baseline=baseline, candidate=candidate, q32_installations=installs,
                q32_selections=selected, comparison=comparison, paired_question_changes=changes,
                eligible_for_agent_merge_review_on_frozen_pair=qualified,
                caveats=['One observed pair is not a confidence interval or stable effect estimate.',
                         'Config-change interval includes client preflight; metadata is not a compile-cost measure.',
                         'Fresh team-main check and any necessary synchronization/revalidation remain external.',
                         'No merge or GPU work is performed by this audit.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.baseline_commit, args.candidate_commit)
    result['audit_script_sha256'] = sha(Path(__file__))
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(eligible=result['eligible_for_agent_merge_review_on_frozen_pair'],
                         comparison=result['comparison'], out=str(args.out))))
