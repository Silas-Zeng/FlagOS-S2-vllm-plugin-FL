"""CPU-only guards against unsafe edits to actual pipeline IR and source."""
import importlib.util
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gqa_pipeline_probe',root/'tools/bench_biv150_gqa_page_pipeline.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=(root/'vllm_fl/patches/biv150_gqa_decode.py').read_text()
_,ast_proof=module.pipeline_source(source)
for old,new in [('position += PAGE_SIZE','position += 2*PAGE_SIZE'),
                ('while position < full_upper:','while position <= full_upper:')]:
    changed=source.replace(old,new);assert changed!=source
    try:module.pipeline_source(changed)
    except AssertionError:pass
    else:raise AssertionError('Changed iteration set accepted')
raw=(root/'docs/evidence/gqa-page-pipeline-20261027/stage2-actual-b64.ttgir').read_text()
changed,proof=module.reuse_invariant_query(raw)
alloc=proof['q_descriptor'];old=proof['removed_duplicate_value'];outside=proof['existing_dominating_value']
needle=old+' = ttg.local_load '+alloc
lines=raw.splitlines(keepends=True)
at=next(i for i,x in enumerate(lines) if needle in x)
mismatch=lines.copy();mismatch[at]=mismatch[at].replace('kWidth = 2','kWidth = 4')
mutations={
    'unknown_q_write':raw.replace('    '+outside+' =','    ttg.local_store %Q_54, '+alloc+' : placeholder\n    '+outside+' =',1),
    'mismatched_type':''.join(mismatch),
    'mismatched_pointer':raw.replace(needle,old+' = ttg.local_load %different',1),
    'missing_for':raw.replace('= scf.for ','= scf.while ',1),
    'outside_duplicate_use':raw+'\n    %bad = use '+old+'\n',
    'extra_query_load':raw.replace('    '+outside+' =','    %extra = ttg.local_load '+alloc+' : invalid\n    '+outside+' =',1),
}
for name,value in mutations.items():
    assert value!=raw,name
    try:module.reuse_invariant_query(value)
    except ValueError:pass
    else:raise AssertionError('Unsafe actual IR accepted: '+name)
assert 'torch' not in sys.modules
print(json.dumps(dict(source_inverse=ast_proof,actual_ir_inverse=proof,
    changed_iteration_sets_rejected=2,unsafe_ir_mutations_rejected=list(mutations),
    torch_imported=False,gpu_executed=False)))
