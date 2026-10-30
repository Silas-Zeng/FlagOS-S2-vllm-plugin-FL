"""Single dense4K reader falsifier; no new library, clock, threshold or service."""
import ast
import json
import re
from types import FunctionType
import biv150_native127_v8_timing as short
import biv150_native127_v8_dense_fixture as fixture

c=short.c
PARENT='8393f2799dd926820d5db4cde7a553bdb757bd8b'
DENSE_ADMISSION_SHA='5824ff9fffa96fd0b8442f7b16f9deb5b60b1a87eb994c864a74ce27082bb8c9' # Root-reviewed source scope; future serial launch still required.
clock_source=short.clock_source
prepare_case=short.prepare_case
run_pair=short.run_pair


def plan():
    return dict(component='single dense4K producer+original reducer reader falsifier; no writer',
      parent=PARENT,development_base=c.DEVELOPMENT_BASE,actual_compile_source=c.SOURCE,
      libraries={role:dict(sha256=x[0],namespace=x[1]) for role,x in c.ROLES.items()},
      static_review_sha256=c.STATIC_REVIEW_SHA,prior_numeric_review_sha256=short.NUMERIC['Root_review_sha256'],
      fixture=fixture.layout(),original3eb_clock=True,unroll=8,warm_replays_per_arm=128,
      timed_replays_per_block=64,complete_steps_per_block=512,order=short.ORDER,event_blocks=4,
      validation='newB Q/table+poison Graph before freshB; full OML/final bits across arms, original FP64/readonly/guards',
      original_BF16_proxy='original unquantized BF16 Q/K/V diagnostic; unchanged .005 metrics',
      no_extra_A_or_side_numeric_matrix=True,no_second_pitch=True,
      new_Root_dense_admission_sha256=DENSE_ADMISSION_SHA,Root_admission=False,device_admission=False,
      numeric_verified=False,performance_verified=False,precision_gate=False,
      model_verified=False,service_verified=False,capacity_recovered=False)


def evidence(a):
    assert type(DENSE_ADMISSION_SHA) is str and re.fullmatch('[0-9a-f]{64}',DENSE_ADMISSION_SHA),'new dense-source Root receipt pending'
    assert a.dense_admission_sha256==DENSE_ADMISSION_SHA and c.sha(a.dense_admission)==DENSE_ADMISSION_SHA
    receipt=json.loads(a.dense_admission.read_text())
    assert receipt==dict(schema='root_native127_dense_reader_scope_v1',parent_source=PARENT,
      actual_static_review_sha256=c.STATIC_REVIEW_SHA,prior_numeric_review_sha256=short.NUMERIC['Root_review_sha256'],
      Root_read_source_and_checked_CPU_plan=True,approved_component_dense_falsifier_only=True,
      requires_explicit_serial_launch=True,scope=json.loads(json.dumps(fixture.layout())),order=list(short.ORDER),
      unroll=8,warm_replays_per_arm=128,timed_replays_per_block=64,
      precision_verified=False,service_verified=False,capacity_recovered=False)
    # Old small-fixture evidence is a prerequisite, never dense fixture permission.
    return dict(prior_small_numeric_and_short_scope=short.evidence(a),new_dense_scope_sha256=DENSE_ADMISSION_SHA)


def adapted_run(parent):
    original,parent_proof=short.adapted_run(parent)
    changed,proof=c.previous.edit(original,[
      ("all(f.input_bytes(cases[0])[k]!=f.input_bytes(cases[1])[k] for k in ('query','cache','lengths','table'))",
       "all(f.input_bytes(cases[0])[k]!=f.input_bytes(cases[1])[k] for k in ('query','table'))\n            assert all(f.input_bytes(cases[0])[k]==f.input_bytes(cases[1])[k] for k in ('cache','lengths'))",1),
      ('        digest=f.sha(raw);path=out/\'blobs\'/(digest+\'.bin\')',
       '        if len(raw)==f.DENSE_CACHE_BYTES:return f.cache_evidence(raw,out)\n        digest=f.sha(raw);path=out/\'blobs\'/(digest+\'.bin\')',1),
      ('bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())',
       'tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()',1),
      ("            record['full_CPU_oracles_authority']='prior Root-reviewed cd2f numeric originals; current B inputs/gold/snapshots retained'",
       "            record['dense_initial_CPU_and_recipe']=f.initial_metadata()\n            record['full_CPU_oracles_authority']='original3eb fullstep counter/codec/golden; last-KV replacement proved idempotent; dense cache reconstructed per chunk'",1)])
    proof['parent_8393_adapter']=parent_proof
    return changed,proof


def execute(a):
    # Retain the accepted runner's call frames and full pair comparisons/clock.
    import probe_biv150_native127_v8_timing as prior
    run=FunctionType(prior.execute.__code__,dict(vars(prior),t=__import__(__name__)))
    return run(a)
