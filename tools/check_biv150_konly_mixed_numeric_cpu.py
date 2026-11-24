#!/usr/bin/env python3
"""Pure CPU metadata/numeric-entry policy checks, never a device or model proof."""
import json
from pathlib import Path
import subprocess
import sys

from bench_biv150_konly_mixed_numeric import CASES, control_metadata
from biv150_konly_decode_numeric_contract import reference_gates
from biv150_konly_mixed_reference import metadata

ROOT=Path(__file__).resolve().parents[1]


def run():
    records=[]
    for name,(base,kv) in CASES.items():
        for iteration in range(4):
            c=control_metadata(base,kv,iteration);rows=metadata(c['cu_q'],c['lengths'],sum(base)+3)
            assert c['query_capacity']==sum(base)+3 and c['cu_q'][-1]==sum(base)
            assert c['decode_sequences']==[r['sequence'] for r in rows if r['decode']]
            assert len(set(c['partition']['decode_tokens']))==len(c['decode_sequences'])
            assert c['partition']['padded_query_rows']==list(range(sum(base),sum(base)+3))
            assert all(r['q_len']<=r['kv_length'] for r in rows if r['q_len']>1)
            assert not set(c['partition']['decode_tokens']) & set(c['partition']['prefill_tokens'])
            if iteration==3:assert not c['decode_sequences']
            if iteration in (1,2):assert any(r['decode'] and r['q_start']!=r['sequence'] for r in rows)
            records.append(dict(case=name,iteration=iteration,decode_sequences=len(c['decode_sequences']),total_queries=sum(base)))
        if name!='tail':
            c=control_metadata(base,kv,1)
            assert {c['lengths'][s] for s in c['decode_sequences'][:8]}=={0,1,15,16,17,63,100,kv-1}
        else:
            c=control_metadata(base,kv,1);assert [c['lengths'][s] for s in c['decode_sequences']]==[0,15]
    # The unchanged relative-RMSE proxy must still reject a close implementation.
    gate=reference_gates([1.]*128,[1.009]*128,[1.]*128,[1.009]*128)
    assert gate['quantized_operand']['pointwise_passed'] and not gate['candidate_numeric_passed']
    assert gate['original_relative_rmse_limit']==.005 and gate['thresholds_relaxed'] is False
    path=ROOT/'tools/bench_biv150_konly_mixed_numeric.py'
    text=path.read_text()
    assert text.index("result['compile_passed']=True")<text.index('encoder=support.encode_initial[(')
    assert 'query_start' not in str(gate)  # Policy fixture contains no hardware/layout simulation.
    plan=json.loads(subprocess.check_output([sys.executable,str(path)],text=True))
    assert plan['cpu_only'] and plan['numeric_entry_ready']
    assert not any(plan[k] for k in ('torch_imported','triton_imported','numpy_imported','gpu_executed','gpu_numeric_verified','service_installed','timing_prepared'))
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','c33b3d9','--',
        'tools/biv150_konly_mixed_factory.py','tools/biv150_konly_mixed_reference.py','tools/kernels/biv150_konly_decode_support.py'],check=True)
    assert not {'torch','triton','numpy','vllm'}.intersection(sys.modules)
    return dict(passed=True,cpu_only=True,metadata_controls=records,default_cli_no_framework=True,
        original_proxy_failure_retained=True,source_compile_gate_precedes_private_numeric_launch=True,
        c33_mapping_and_codec_unchanged=True,gpu_numeric_verified=False,service_installed=False,gpu_executed=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
