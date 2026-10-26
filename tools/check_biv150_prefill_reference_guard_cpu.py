"""Reject added oracle error while retaining identical accepted-baseline limits."""
import json
import torch
import biv150_prefill_probe_fixtures as fixture


def main():
    fixture.torch = torch
    reference = torch.tensor([2., 0.], dtype=torch.bfloat16)
    baseline = torch.tensor([2.015625, 0.], dtype=torch.bfloat16)
    same = fixture.reference_guard(baseline, baseline, reference)
    assert same['passed'] and same['existing_baseline_violations'] == 1
    assert not fixture.reference_guard(torch.tensor([2.03125,0.]), baseline, reference)['passed']
    assert not fixture.reference_guard(torch.tensor([2.015625,.01]), baseline, reference)['passed']
    assert not fixture.reference_guard(torch.tensor([float('nan'),0.]), baseline, reference)['passed']
    assert not fixture.reference_guard(torch.tensor([2.,.006]), reference, reference)['passed']
    print(json.dumps({'passed':True,'gpu_executed':False,'checks':[
        'identical one-ULP baseline exception retained', 'worsened existing error rejected',
        'new error at another element rejected', 'nonfinite rejected',
        'original tolerance enforced on accurate baseline']}))


if __name__ == '__main__':
    main()
