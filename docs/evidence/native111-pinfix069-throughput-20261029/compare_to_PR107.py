#!/usr/bin/env python3
"""Derive a descriptive single-run comparison from two pinned CPU reviews."""
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert not a.out.exists()
def read(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)
baseline_sha = '57bb07087627709f2ccc27b73d4def3d78793c5a562b5fb0d7a89aa96bd0c218'
current_sha = '1a595917940d95cdac0c271f4ed0f0ac7c6acb0cfcec1c04aab0c683bf616aad'
b = read(HERE.parent/'native107-throughput-v1-20261028/review.json', baseline_sha)
c = read(HERE/'independent-review.json', current_sha)
assert b['source_commit'] == 'b5d1a241f45efd21517d04e25a7dc5d44ac1ca06'
assert c['source_commit'] == '069838d6fbb7591de4384aac8caef49324c43aef'
assert b['benchmark_function_sha256'] == c['benchmark_function_sha256'] == 'e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d'
assert b['warmup_plan'] == c['warmup_plan']
metrics = ('total_token_throughput', 'mean_ttft_ms', 'mean_tpot_ms')
rows = {}
for k, count in [('4096', 256), ('16384', 128)]:
    old, new = b['scores'][k], c['scores'][k]
    assert old['completed'] == new['completed'] == count
    assert old['failed'] == new['failed'] == 0
    assert old['max_concurrency'] == new['max_concurrency'] == 64
    assert old['total_input_tokens'] == new['total_input_tokens'] == int(k)*count
    assert old['total_output_tokens'] == new['total_output_tokens'] == 1024*count
    rows[k] = {m: dict(PR107=old[m], pinfix069=new[m], percent_change=(new[m]/old[m]-1)*100) for m in metrics}
report = dict(schema='native111-vs-accepted-PR107-single-score-v1', baseline_report_sha256=baseline_sha,
    candidate_report_sha256=current_sha, baseline_tested_source=b['source_commit'],
    candidate_tested_source=c['source_commit'], warmup_plan=c['warmup_plan'], comparisons=rows,
    current_Math105_run=False, precision_gate=False, merge_qualification=False,
    repeat_distribution_available=False, causal_attribution_proven=False,
    notes=['PR107 is the recorded accepted reference later merged as main a455; this is not a fresh main rerun.',
           '4K -0.3844% is recorded as no demonstrated gain; statistical equivalence is not established.',
           '16K -7.7973% is a negative single-score observation, not a confidence interval.',
           'PR107 encoded K8/8b86 and this compact K8/7892+2ad candidate differ in several components; both enable their respective Q32 paths.',
           'Current 4K score interval adds six index metadata files and changes selected tuning rows; current 16K interval does neither.',
           'a6d remains a failed service run and is not substituted for the accepted reference.'])
a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
print(hashlib.sha256(a.out.read_bytes()).hexdigest())
