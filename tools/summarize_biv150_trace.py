"""Summarize actual Kineto events; cumulative GPU time is not wall-clock time."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def summarize(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        data = json.load(stream)
    groups = defaultdict(lambda: dict(count=0, cumulative_ms=0.0, example_shapes=None))
    for event in data.get('traceEvents', []):
        if event.get('ph') != 'X' or 'dur' not in event:
            continue
        category = event.get('cat', '')
        if category not in ('kernel', 'cpu_op', 'gpu_memcpy', 'gpu_memset'):
            continue
        row = groups[(category, event['name'])]
        row['count'] += 1
        row['cumulative_ms'] += event['dur'] / 1000
        shapes = event.get('args', {}).get('Input Dims')
        if shapes and row['example_shapes'] is None:
            row['example_shapes'] = shapes
    result = dict(trace=str(path), note='Cumulative event durations, not model wall-clock attribution')
    for category in ('kernel', 'cpu_op', 'gpu_memcpy', 'gpu_memset'):
        rows = [dict(name=name, **row) for (cat, name), row in groups.items() if cat == category]
        result[category] = sorted(rows, key=lambda row: row['cumulative_ms'], reverse=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value[:15] if isinstance(value, list) else value
                      for key, value in result.items()}, indent=2))
