"""Independent CPU audit of main-Q32 versus default K128 official service pair."""
import argparse
import datetime
import json
import math
from pathlib import Path

from audit_biv150_prefill_pair import arm, events, read, sha


def selection(log, name, tile):
    expected=dict(installed=True,block_m=256,block_q=32,num_warps=16,
                  num_stages=1,tile_size=tile,minimum_query_rows=1024,num_sequences=1)
    if tile==128:expected['maxnreg']=128
    installs=events(log,name+'_INSTALL')
    assert any(all(x.get(k)==v for k,v in expected.items()) for x in installs)
    selected=events(log,name+'_SELECTED');assert selected,'Feature not selected in model'
    assert name+'_FALLBACK' not in log,'Feature permanently disabled'
    records={x['candidate_hash']:x for x in events(log,'BIV150_Q_FRAGMENT_SELECTED')
             if x.get('mode')=='selected'}
    for row in selected:
        assert all(row.get(k)==v for k,v in expected.items() if k not in ('installed','minimum_query_rows'))
        assert row['rows']>=1024
        compiled=records[row['hash']];meta=compiled['candidate_resources']['metadata']
        assert (meta['num_warps'],meta['num_stages'],meta['shared'])==(16,1,65536)
        assert compiled['mechanism']['mechanism_pass']
        assert compiled['mechanism']['llvm_q_memory_reads_in_loop']==0
        if tile==128:
            assert 0<row['n_regs']<=128 and row['n_spills']==0 and row['shared']==65536
            options={k[1]:v for k,v in compiled['options'][1]}
            assert options['maxnreg']==['int',128]
    return dict(installations=installs,selections=selected)


def audit(root,baseline_commit,candidate_commit):
    control=read(root/'state.json');assert control['phase']=='completed'
    baseline=arm(root/'baseline',baseline_commit);candidate=arm(root/'candidate',candidate_commit)
    for key in ('command','tuning_snapshot','gate_library_sha256','driver_sha256'):
        assert baseline['manifest'][key]==candidate['manifest'][key],key
    assert baseline['manifest']['gate_library_sha256']=='5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
    for data in (baseline,candidate):
        assert 'VLLM_FL_BIV150_PREFILL_K128' not in data['manifest']['env']
        data['working_throughput_gates']={}
        for length,(floor,ttft_limit) in {'4096':(4809.36*.99,4770.3209),
                                        '16384':(3364.03*.99,156938.85)}.items():
            assert math.isclose(data['ttft'][length]['working_ttft_limit_ms'],ttft_limit)
            value=data['state']['scores'][length]['total_token_throughput']
            data['working_throughput_gates'][length]=dict(floor=floor,observed=value,passed=value>=floor)
    logs={label:(root/label/'server.log').read_text() for label in ('baseline','candidate')}
    assert not events(logs['baseline'],'BIV150_PREFILL_K128_SELECTED')
    paths=dict(baseline=selection(logs['baseline'],'BIV150_PREFILL_Q32',64),
               candidate=selection(logs['candidate'],'BIV150_PREFILL_K128',128))
    for label,log in logs.items():
        assert 'BIV150_COMPUTE_ORDER_SCOPE_REJECTED' not in log
        assert events(log,'BIV150_COMPUTE_ORDER_FIRST_SELECTED'),label+' C3 not naturally selected'
    comparison={}
    for length in ('4096','16384'):
        a=baseline['state']['scores'][length]['total_token_throughput']
        b=candidate['state']['scores'][length]['total_token_throughput'];change=b/a-1
        assert math.isclose(control['comparison'][length]['observed_relative_change'],change,rel_tol=1e-10,abs_tol=1e-12)
        comparison[length]=dict(baseline_tps=a,candidate_tps=b,observed_relative_change=change,higher=b>a)
    qualified=all(x['accuracy']['passed'] and all(g['passed'] for g in x['ttft'].values())
                  and all(g['passed'] for g in x['working_throughput_gates'].values())
                  for x in (baseline,candidate)) and all(c['higher'] for c in comparison.values())
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
                cpu_only=True,baseline=baseline,candidate=candidate,actual_paths=paths,
                comparison=comparison,eligible_for_agent_merge_review_on_frozen_pair=qualified,
                caveats=['Single full-warm scored round per length; not a confidence interval.',
                         'Config-change interval includes client preflight, not a compile-cost measurement.',
                         'Fresh team main, source synchronization, process cleanup and any borderline noise assessment remain required.',
                         'No GPU work or merge performed by this audit.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-commit',required=True)
    p.add_argument('--candidate-commit',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.baseline_commit,a.candidate_commit)
    result['audit_script_sha256']=sha(Path(__file__))
    with a.out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(eligible=result['eligible_for_agent_merge_review_on_frozen_pair'],comparison=result['comparison'])))
