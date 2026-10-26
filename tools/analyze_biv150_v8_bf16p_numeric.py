"""Read saved V8 failure-coordinate JSON on CPU; never imports torch/Triton."""
import argparse
import json
from pathlib import Path

from biv150_v8_bf16p_validation import attribute_coordinate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    record = json.loads(args.trace.read_text())
    print(json.dumps(dict(source_commit=record['source_commit'], precision_gate=False,
                          explanatory_only=True,
                          coordinates=[attribute_coordinate(trace) for trace in record['coordinates']]), indent=2))
