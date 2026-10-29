#!/usr/bin/env python3
"""CPU-only frozen component contract. No numeric/throughput entry before actual SO review."""
import json
from pathlib import Path
import sys
import biv150_native_k8_mixed_factory as factory

def plan():
    return {'source':factory.verify(),'phase':'compile-and-actual-object-review-first',
      'next_compile':['python','tools/compile_biv150_native_k8_mixed.py','--compile-only','--root-owned-exclusive','--expected-source-commit','<full-frozen-commit>','--out','<fresh-absolute-directory>'],
      'cases':[{'decode_rows':n,'prefill_rows':2048-n,'total_query_rows':2048,'context':ctx,
                'current_main_TL_segments':old_s,'candidate_Native_segments':s,'diagnostic_TL_segments':s,
                'bitwise_reference':{'library_sha256':'8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5',
                                     'physical_rows':n+1,'last_length':0,'valid_rows_compared':n,'segments':s}}
               for n,ctx,old_s,s in [(63,4096,4,2),(31,16384,8,4)]],
      'physical_page_bytes':16384,'Q_pitch_elements':[2048,2560],'output_pitch_elements':2048,
      'original_width16_scratch':{'partial_stride':[32768,2048,128,1],'scalar_stride':[256,16,1]},
      'numeric_rules':['Retain original INT64 QK/FP64 quantized-operand hard gates; record BF16 proxy failures unchanged.',
        'Candidate vs same-S accepted8b padded-one-zero valid rows: O/M/L/final bitwise, with separate untouched extra-row guards.',
        'Current TL uses different S; do not require cross-partition bitwise or relax its independent oracle tolerance.',
        'Require real valid prefill page tables; PR85 poisoned-prefill fixture cannot execute paired prefill.',
        'Check complete cache/readonly inputs, output/prefill disjointness and unused scratch rows/segments.',
        'Reference-only padding prep is labeled; all preparation genuinely required by candidate is included in timing.'],
      'full_step_timing_includes':['owned dispatch and metadata/view setup','all-token append/codec','decode producer','decode reducer',
                                   'same current-main prefill reader','actual stream dependencies and completion'],
      'pending':['fresh actual new SO/host-symbol/ELF binding and dual-library coexistence',
                 'numeric runner after fresh object admission; no GPU-numeric entry exists in this initial freeze',
                 'complete-step current-main A/C comparison; same-S TL diagnostic is not the performance baseline',
                 'actual service coverage, Math105, throughput and latest-main comparison'],
      'device_or_GPU_executed':False,'numeric_verified':False,'performance_verified':False,'service_verified':False}

if __name__=='__main__':
    if len(sys.argv)!=1:raise SystemExit('CPU plan only; compile-only has a separate explicit ownership-gated entry. Numeric entry is pending actual SO review.')
    print(json.dumps(plan(),sort_keys=True,indent=2))
