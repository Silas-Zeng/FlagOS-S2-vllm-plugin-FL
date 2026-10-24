"""Describe preserved failure outputs without changing any original pass flag."""
import argparse,hashlib,json
from pathlib import Path
import torch


def describe(actual,reference):
    assert actual.dtype==reference.dtype==torch.bfloat16
    assert actual.device.type==reference.device.type=='cpu'
    finite=torch.isfinite(actual)&torch.isfinite(reference)
    delta=(actual.float()-reference.float()).abs()
    strict=(delta>.005+.005*reference.float().abs())|~finite
    def ordered(x):
        bits=x.contiguous().view(torch.int16).to(torch.int32)&65535
        return torch.where((bits&32768)!=0,65535-bits,bits+32768)
    ulp=(ordered(actual)-ordered(reference)).abs()
    ulp=torch.where(actual==reference,0,ulp)
    bad_finite=strict&finite
    return dict(elements=actual.numel(),nonfinite=int((~finite).sum()),
        strict_violations=int(strict.sum()),strict_passed=not bool(strict.any()),
        max_abs=float(delta.max()),max_finite_ulp=int(ulp[finite].max()) if bool(finite.any()) else None,
        strict_finite_violations=int(bad_finite.sum()),
        strict_finite_max_ulp=int(ulp[bad_finite].max()) if bool(bad_finite.any()) else 0,
        strict_violations_single_ulp=int((bad_finite&(ulp==1)).sum()),
        strict_violations_above_one_ulp=int((bad_finite&(ulp>1)).sum()))


def main(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=json.loads((a.probe/'result.json').read_text())
    assert source['diagnostic_completed'] and source['baseline']=='out' and len(source['records'])==1134
    report=dict(cpu_only=True,gpu_used=False,preserved_fixture_patterns_only=True,
        saved_reference_origin='full CPU FP64 product rounded to BF16 in frozen probe',
        reference_recomputed_here=False,source_commit=source['commit'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_summary=source['summary'],original_flags_unchanged=True,model_accuracy_verified=False,records=[])
    for r in source['records']:
        if not r.get('fixture'):continue
        p=a.probe/r['fixture'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['fixture_sha256']
        d=torch.load(p,map_location='cpu',weights_only=True)
        report['records'].append(dict(layer=r['layer'],geometry=r['geometry'],m=r['m'],config=r['config'],
            pattern=d['pattern'],fixture=r['fixture'],fixture_sha256=r['fixture_sha256'],
            original_candidate_all_patterns=r['candidate_cpu_all_passed'],
            original_native_all_patterns=r['native_cpu_all_passed'],
            candidate=describe(d['candidate'],d['cpu']),native=describe(d['native'],d['cpu'])))
    assert len(report['records'])==701
    report['summary']={'fixtures':len(report['records'])}
    for key in ('candidate','native'):
        stats=[r[key] for r in report['records']]
        report['summary'][key]=dict(strict_failed_saved_patterns=sum(not s['strict_passed'] for s in stats),
            strict_violations=sum(s['strict_violations'] for s in stats),
            strict_single_ulp=sum(s['strict_violations_single_ulp'] for s in stats),
            strict_above_one_ulp=sum(s['strict_violations_above_one_ulp'] for s in stats),
            nonfinite=sum(s['nonfinite'] for s in stats),
            max_strict_finite_ulp=max(s['strict_finite_max_ulp'] for s in stats))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
