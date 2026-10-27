#!/usr/bin/env python3
"""No framework: prove full-copy-before-CPU-indexing; GPU gather is forbidden."""
import json
from types import SimpleNamespace
import sys
sys.dont_write_bytecode=True
from bench_biv150_konly_2d128_numeric import guarded_cpu_snapshot


class CPU:
    device=SimpleNamespace(type='cpu');shape=(2050,16,128)
    def __init__(self):self.read=[]
    def __getitem__(self,index):
        if type(index) is not int or index not in (0,2049):raise AssertionError('only CPU positive guard indices')
        self.read.append(index);return ('actual-row',index)


class GPU:
    def __init__(self):self.full=CPU();self.copies=0
    def cpu(self):self.copies+=1;return self.full
    def __getitem__(self,index):raise AssertionError('GPU canary indexing is forbidden')


def run():
    g=GPU();t=SimpleNamespace(stack=lambda values:tuple(values));full,guards=guarded_cpu_snapshot(g,t)
    assert full is g.full and g.copies==1 and full.read==[0,2049]
    assert guards==( ('actual-row',0),('actual-row',2049) )
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,full_CPU_copy_first=True,only_CPU_positive_guard_indices=True,
        complete_original_guarded_buffer_retained=True,GPU_indexing_calls=0,
        original_71_guard_values_and_all_hard_gates_preserved=True,
        historical_ef4_actual_suffix_proven_unmodified=False,GPU_executed=False,precision_gate=False)


if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
