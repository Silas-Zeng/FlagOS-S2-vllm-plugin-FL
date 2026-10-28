#!/usr/bin/env python3
"""Default CPU plan. Actual5cf6 four-case consumer reuses the original N48 driver."""
import ast
import hashlib
from pathlib import Path
import sys

from probe_biv150_native_k8_compact_v_overlap_consumer import replace_exact

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='tools/probe_biv150_native_k8_compact_n48_consumer.py'
ORIGINAL_SHA='5a9e6aaa51b12daae4e1767ca2ae01cbd290c4b7c25d6a610b0176692c8a5ea3'


def adapted_source():
    raw=(ROOT/ORIGINAL).read_bytes();assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA
    edits=[
      ('import biv150_native_k8_compact_n48_factory as source','import biv150_native_k8_compact_n48_shared6k_factory as source'),
      ('import biv150_native_k8_compact_n48_consumer_contract as admission','import biv150_native_k8_compact_n48_shared6k_admission as admission'),
      ('candidate=torch.ops.cjj_native_k8_compact_n48_s4_probe.produce','candidate=torch.ops.cjj_native_k8_compact_n48_shared6k_probe.produce'),
      ("for name in ('probe_biv150_native_k8_compact_n48_consumer.py',", "for name in ('probe_biv150_native_k8_compact_n48_shared6k_consumer.py','biv150_native_k8_compact_n48_shared6k_admission.py',\n          'biv150_native_k8_compact_n48_shared6k_factory.py','compile_biv150_native_k8_compact_n48_shared6k.py',\n          'probe_biv150_native_k8_compact_n48_consumer.py',"),
      ("        for k in ('candidate_compile_summary'", "        admission.record_executed_source(out,result)\n        for k in ('candidate_compile_summary'")]
    text,proof=replace_exact(raw.decode(),edits)
    restored=text
    for x,y in reversed(edits):assert restored.count(y)==1;restored=restored.replace(y,x)
    assert restored.encode()==raw
    ast.parse(text)
    proof.update(original_path=ORIGINAL,original_sha256=ORIGINAL_SHA,inverse_byte_exact=True,
      actual_executed_sha256=hashlib.sha256(text.encode()).hexdigest(),numeric_and_fixture_code_changed=False,
      four_cases_6phases_Graph_first_side_thresholds_bitwise_and_saved_buffers_unchanged=True)
    return text,proof


def frontend():
    text,proof=adapted_source()
    ns=dict(__name__='cjj_native113_existing_N48_consumer',__file__=str(ROOT/ORIGINAL))
    exec(compile(text,'<actual113-reused-N48-consumer>','exec'),ns)
    return ns,proof


if __name__=='__main__':raise SystemExit(frontend()[0]['main']())
