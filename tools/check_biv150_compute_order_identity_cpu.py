"""Verify the package source identities used by the BI-V150 compute-order guard."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_pins():
    order = ROOT / 'vllm_fl/patches/biv150_compute_order.py'
    for node in ast.parse(order.read_bytes()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('EXPECTED_RUNNER', 'SOURCES'):
                yield name, ast.literal_eval(node.value)


def main():
    config = dict(source_pins())
    files = {'vllm_fl.worker.model_runner': config['EXPECTED_RUNNER']}
    files.update({name: digest for name, digest in config['SOURCES'].items()
                  if name.startswith('vllm_fl.')})
    for name, expected in files.items():
        path = ROOT / (name.replace('.', '/') + '.py')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f'{name}: {actual} != {expected}')
    print(json.dumps({'passed': True, 'validated_sources': sorted(files)}, indent=2))


if __name__ == '__main__':
    main()
