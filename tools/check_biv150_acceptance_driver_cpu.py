"""Execute selected actual driver AST with fake processes; no service/GPU imports."""
import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).with_name('run_biv150_native_prefill_acceptance.py'))
parser.add_argument('--out', type=Path)
args = parser.parse_args()
SOURCE = args.source.resolve()
text = SOURCE.read_text()
tree = ast.parse(text)
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
remember = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'remember')
accuracy_block = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'all'")
initial_state = next(n for n in main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'state' for t in n.targets))
cleanup_guard = main.body[-1]
assert isinstance(cleanup_guard, ast.If) and 'cleanup_complete' in ast.unparse(cleanup_guard.test)
checks = []


def run(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(SOURCE), 'exec'), namespace)


def process(pid, start, sid=100, state='S'):
    return dict(pid=pid, start_ticks=start, sid=sid, state=state)


def ownership_case(name, live, owned, expected):
    class FakeProc:
        def iterdir(self):
            return [SimpleNamespace(name=str(pid)) for pid in live]
    namespace = dict(server=SimpleNamespace(pid=100), launcher=process(100, 10),
                     owned=copy.deepcopy(owned), identity=lambda pid: live.get(pid),
                     Path=lambda path: FakeProc())
    run([remember], namespace)
    namespace['remember']()
    assert namespace['owned'] == expected, (name, namespace['owned'], expected)
    checks.append(dict(name=name, passed=True))


ownership_case('live_launcher_adopts_only_same_session',
               {100: process(100, 10), 101: process(101, 11), 200: process(200, 20, 200)},
               {}, {100: process(100, 10), 101: process(101, 11)})
ownership_case('reused_launcher_pid_sid_cannot_anchor',
               {100: process(100, 99), 101: process(101, 100)}, {}, {})
ownership_case('dead_launcher_no_anchor_cannot_adopt_recycled_session',
               {101: process(101, 100)}, {}, {})
ownership_case('surviving_owned_child_anchors_original_session',
               {101: process(101, 11), 102: process(102, 12)},
               {101: process(101, 11)}, {101: process(101, 11), 102: process(102, 12)})
ownership_case('reused_owned_child_pid_cannot_anchor',
               {101: process(101, 111), 102: process(102, 112)},
               {101: process(101, 11)}, {101: process(101, 11)})
ownership_case('old_pid_identity_is_never_overwritten',
               {100: process(100, 10), 101: process(101, 111)},
               {101: process(101, 11)}, {100: process(100, 10), 101: process(101, 11)})


with tempfile.TemporaryDirectory(prefix='biv150-driver-cpu-') as folder:
    out = Path(folder)
    base = dict(metric_count=105, predictions=105, unique_questions=105,
                failed_predictions=[], score=99 / 105)
    for name, changes, mode, expected_error, expected_precision in (
        ('near_miss_allows_diagnostic_throughput_only', {}, 'all', False, False),
        ('below_near_miss_fails_closed', {'score': 98 / 105}, 'all', True, None),
        ('near_miss_with_failed_prediction_fails_closed', {'failed_predictions': [4]}, 'all', True, None),
        ('missing_prediction_fails_closed', {'predictions': 104}, 'all', True, None),
        ('duplicate_question_fails_closed', {'unique_questions': 104}, 'all', True, None),
        ('precision_pass_still_needs_paired_acceptance', {'score': 100 / 105}, 'all', False, True),
        ('throughput_only_keeps_precision_unknown', {}, 'throughput', False, None),
    ):
        verdict = dict(base, **changes)
        (out / 'accuracy-verdict.json').write_text(json.dumps(verdict))
        def accuracy(_):
            if verdict['score'] < .95 or verdict['failed_predictions'] or any(verdict[k] != 105 for k in ('metric_count', 'predictions', 'unique_questions')):
                raise RuntimeError('synthetic original accuracy gate failure')
        namespace = dict(args=SimpleNamespace(mode=mode, out=out, commit='cpu-fixture', candidate=True),
                         json=json, accuracy=accuracy)
        run([initial_state], namespace)
        namespace['save'] = lambda phase, **values: namespace['state'].update(phase=phase, **values)
        error = False
        try:
            run([accuracy_block], namespace)
        except RuntimeError:
            error = True
        assert error is expected_error, (name, error)
        assert namespace['state']['precision_passed'] is expected_precision, (name, namespace['state'])
        assert namespace['state']['acceptance_passed'] is False
        if name.startswith('near_miss_allows'):
            assert namespace['state']['diagnostic_only'] is True
        checks.append(dict(name=name, passed=True))

    calls = []
    def benchmark(_, name, length, output, count, seed):
        calls.append((length, output, count, seed))
        return dict(name=name)
    throughput = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'throughput')
    namespace = dict(json=json, benchmark=benchmark)
    run([throughput], namespace)
    namespace['throughput'](out)
    expected = []
    for length, count in ((4096, 256), (16384, 128)):
        expected.extend((length, 128, 64, 920000 + i) for i in range(3))
        expected.append((length, 1024, count, 0))
    assert calls == expected, calls
    checks.append(dict(name='fixed_three_warmups_then_one_score_per_length', passed=True))

for cleanup_complete in (False, True):
    raised = False
    try:
        run([cleanup_guard], dict(state=dict(cleanup_complete=cleanup_complete)))
    except RuntimeError:
        raised = True
    assert raised is (not cleanup_complete)
    checks.append(dict(name=f'cleanup_complete_{cleanup_complete}_controls_normal_exit', passed=True))

report = dict(source=str(SOURCE), source_sha256=hashlib.sha256(text.encode()).hexdigest(),
              validation='Selected actual source AST, synthetic processes/accuracy, CPU only',
              limits='No real process signals, HTTP, service, GPU, timing or concurrent PID-reuse race was exercised.',
              checks=checks, all_passed=all(x['passed'] for x in checks))
if args.out:
    args.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
