#!/usr/bin/env python3
"""Reuse PR120's unchanged CPU guard checks against the merged PR118 baseline."""
import argparse
import json
import check_biv150_native_storage_main56_cpu as checks

BASE = '899ce77470b0b539aebc108fd122f668c2d5894d'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=checks.Path, default=checks.SO)
    parser.add_argument('--helper', type=checks.Path, default=checks.HELPER)
    args = parser.parse_args()
    # Use the current baseline for both the parent adapter and real mixed
    # forward/producer AST fixtures. All checks and numeric limits are unchanged.
    checks.MAIN = checks.MIXED = BASE
    report = checks.run(args.library, args.helper)
    # Relabel only the historical report names; the underlying checks ran
    # against BASE above, and their outcomes remain unchanged.
    report['actual899_mixed_forward'] = report.pop('actual698_mixed_forward')
    for row in report['actual899_mixed_forward']:
        row['actual899_forward_two_calls'] = row.pop('actual698_forward_two_calls')
        if row['implementation'] == 'main56':
            row['implementation'] = 'main899'
    print(json.dumps(report, sort_keys=True, indent=2))
