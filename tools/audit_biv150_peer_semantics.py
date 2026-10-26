"""Independently replay initial/post peer semantics from raw files, with no HTTP/GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REQUEST_HASHES = {
    'decode': '90e8284043b131ead2842c04cf83b5e3af3aa2f79358bee412769b76f61574c2',
    'prefill': 'ecf8e319aa48e3c66e8b40019df2dff483ace79dbfdd61de41d436bb3961a42b',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible(text):
    if '<think>' in text:
        assert '</think>' in text, 'Unfinished reasoning text'
        text = text.rsplit('</think>', 1)[1]
    return text.strip()


def phase(folder, fixtures):
    requests = read(folder / 'requests.json')
    for name, expected in REQUEST_HASHES.items():
        fixture = fixtures / (name + '-request.json')
        assert sha(fixture) == expected == requests[name]['sha256']
        assert read(fixture) == requests[name]['payload'], 'Fixture payload mismatch'
    events, content, done = [], [], 0
    for line in (folder / 'decode-sse.txt').read_text().splitlines():
        if not line.startswith('data: '):
            continue
        value = line[6:].strip()
        if value == '[DONE]':
            done += 1
            continue
        assert done == 0, 'SSE event after DONE'
        event = json.loads(value)
        assert not event.get('error'), 'Error in raw SSE response'
        events.append(event)
        for choice in event.get('choices', []):
            text = choice.get('delta', {}).get('content')
            if text:
                content.append(text)
    streamed = read(folder / 'decode-response.json')
    assert done == 1 and events == streamed['events'] and content == streamed['content']
    decode_text = visible(''.join(content))
    prefill = read(folder / 'prefill-response.json')
    assert not prefill.get('error')
    prefill_text = visible(prefill['choices'][0]['message'].get('content') or '')
    assert re.findall(r'(?<!\d)\d+(?!\d)', decode_text) == [str(x) for x in range(1, 129)]
    answers = re.findall(r'(?<!\d)\d+(?!\d)', prefill_text)
    assert answers and answers[-1] == '85'
    result = read(folder / 'result.json')
    assert all(result[key] is True for key in
               ['completed', 'all_correct', 'decode_correct', 'prefill_correct', 'http_overlap'])
    assert result['decode_visible'] == decode_text and result['prefill_visible'] == prefill_text
    assert result['decode_started'] == streamed['started']
    assert result['decode_finished'] == streamed['finished']
    times = [streamed['started'], streamed['first_content'], streamed['finished'],
             result['prefill_started'], result['prefill_finished']]
    assert all(type(x) in (int, float) and math.isfinite(x) for x in times)
    assert times[0] <= times[1] <= times[3] < times[2] and times[3] <= times[4]
    paths = [folder / name for name in ['requests.json', 'decode-sse.txt',
             'decode-response.json', 'prefill-response.json', 'result.json']]
    return dict(passed=True, raw_sha256={str(p): sha(p) for p in paths},
                transport_timing_overlap=True, result=result)


def audit(arm, fixtures):
    state = read(arm / 'state.json')
    initial = phase(arm / 'mixed-requests', fixtures)
    post = phase(arm / 'post-score-semantic/mixed-requests', fixtures)
    assert initial['result'] == state['diagnostic']
    assert post['result'] == state['post_score_diagnostic']
    assert all(state[key] is True for key in
               ['initial_semantic_passed', 'post_score_semantic_passed', 'semantic_passed'])
    return dict(passed=True, cpu_only=True, initial=initial, post=post,
                fixture_sha256={name: sha(fixtures / (name + '-request.json'))
                                for name in REQUEST_HASHES},
                limitation='Two peer requests only; not105-question precision or model-step overlap proof.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.arm, args.fixtures)
    result['audit_script_sha256'] = sha(Path(__file__))
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], out=str(args.out))))
