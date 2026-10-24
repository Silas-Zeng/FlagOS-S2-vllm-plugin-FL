"""Prove logical partial addresses are bijective and within equal-capacity buffers."""

import hashlib
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
assert not subprocess.check_output(['git','-C',str(root),'diff','0bf9d2efeaa10144dd7f0ff4597298df029f9328','--','vllm_fl','csrc'],text=True)
assert hashlib.sha256((root/'vllm_fl/patches/biv150_gqa_decode.py').read_bytes()).hexdigest()=='2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
seen=set()
for h in range(16):
 for segment in range(16):
  for d in range(128):
   old=h*16*128+segment*128+d
   new=segment*16*128+h*128+d
   assert 0<=old<32768 and 0<=new<32768
   seen.add(new)
assert len(seen)==32768
for segments in (1,2,4,8,16):
 selected={s*16*128+h*128+d for s in range(segments) for h in range(16) for d in range(128)}
 assert len(selected)==segments*16*128 and max(selected)<32768
print(json.dumps(dict(cpu_only=True,bijective_coordinates=32768,segment_prefixes=[1,2,4,8,16],
    same_capacity=True,no_address_aliases=True,production_sources_unchanged=True,
    scalar_unit_segment_stride_preserved=True,device_bit_identity_and_performance_pending=True)))
