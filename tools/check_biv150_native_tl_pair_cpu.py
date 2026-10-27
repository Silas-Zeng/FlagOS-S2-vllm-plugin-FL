#!/usr/bin/env python3
"""Bounded fixture/policy/source contracts; never imports a device framework."""
import ast
import json
import math
from pathlib import Path
import subprocess
import struct
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import biv150_native_tl_pair_contract as c
import bench_biv150_native_tl_pair as runner


def run():
    c.pins();coordinates=0
    for name,(batch,length,segments) in c.CASES.items():
        mapping=c.table(batch,length);flat=[p for row in mapping for p in row]
        assert len(flat)==len(set(flat))==batch*(length//16)
        assert min(flat)>=0 and max(flat)<c.ARENA_PAGES
        assert max(flat)>c.ARENA_PAGES-32,'fixture accidentally occupies only a small arena prefix'
        assert mapping==c.table(batch,length)
        for n in (0,1,2,15,16,17,31,32,33,63,64,65,length-1,length):
            slots=c.append_slots(mapping,[n]*batch)
            for row,slot in zip(mapping,slots):
                expected=-1 if n==0 else row[math.ceil(n/16)-1]*16+(n+15)%16
                assert slot==expected
                if n:
                    page,column=divmod(slot,16)
                    assert page==row[(n-1)//16] and column==(n-1)%16
                    for plane in (0,8192):
                        for head in (0,1):
                            byte=page*16384+plane+column*512+head*256
                            assert 0<=byte and byte+256<=c.ARENA_PAGES*16384
                coordinates+=1
        assert c.append_slots(mapping,[17]*batch)[0]//16==mapping[0][1]
        assert c.append_slots(mapping,[16]*batch)[0]//16==mapping[0][0]
    assert c.metrics([0.,1.],[0.,1.])['passed']
    m=c.metrics([1.000244140625],[1.]);assert m['rms']>0 and m['max_abs']>0
    for values in ([.006],[math.nan],[math.inf]):assert not c.metrics(values,[0.])['passed']
    assert not c.metrics([0.1],[1.])['passed']
    text=(c.ROOT/'tools/bench_biv150_native_tl_pair.py').read_text();tree=ast.parse(text)
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any(any(a.name in ('torch','triton','numpy') for a in n.names) for n in imports)
    append=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='append')
    call=append.body[0].value
    assert isinstance(call.args[7],ast.Constant) and call.args[7].value is False
    assert ast.unparse(call.args[6])=='ARENA_PAGES * 16'
    step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
    assert [ast.unparse(n.value.func) for n in step.body]==['append','produce','consume']
    assert '_launch_producer' not in text and 'helper.main_call(base,c,segments)' in text
    assert 'native_runtime.try_produce' in text and 'helper.reduce_call(base,cases[arm],segments,4)' in text
    assert text.index("assert check['passed']")<text.index('for arm in ORDER:')
    assert "arena.copy_(cold);append();torch.cuda.synchronize()" in text
    assert "assert fingerprints()==dense_readonly and digest(raw(arena))==expected_sha" in text
    assert c.ORDER==('native','triton','triton','native')
    assert not subprocess.check_output(['git','-C',str(c.ROOT),'diff',c.MAIN,'--','vllm_fl','csrc'])
    # Execute actual bits/raw with an explicit stdlib-only metadata recorder.
    # No real Torch import: preserve signed int64 bytes, including invalid slots.
    values=(-2**63,-1,0,548415,2**63-1)
    token=object()
    class Int64Host:
        def detach(self):return self
        def cpu(self):return self
        def contiguous(self):return self
        def element_size(self):return 8
        def view(self,dtype):assert dtype is token;return self
        def numpy(self):return NS(tobytes=lambda:struct.pack('<5q',*values))
    fake=NS(int8=object(),int16=object(),int32=object(),int64=token)
    with patch.dict(sys.modules,{'torch':fake}):
        assert runner.raw(Int64Host())==struct.pack('<5q',*values)
    assert not {'torch','triton','vllm','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,GPU_executed=False,JIT_executed=False,
        main=c.MAIN,arena_pages=c.ARENA_PAGES,active_pages={'4k':16384,'16k':32768},
        deterministic_scattered_no_overlap=True,post_append_coordinates_checked=coordinates,
        page16_to17_transition_verified=True,complete_step_order_verified=True,
        append_ENCODE_false=True,full_FP64_gate_before_timing=True,
        int64_signed_serializer_actual_function_checked=True,
        actual_device_numeric_and_ABBA_verified=False,model_extrapolation=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
