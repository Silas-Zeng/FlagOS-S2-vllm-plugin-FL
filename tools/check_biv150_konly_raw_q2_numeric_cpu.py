#!/usr/bin/env python3
"""Private raw q2 metadata inverse, query/segment address and failure fixtures."""
from copy import deepcopy
import json
import sys

import bench_biv150_konly_raw_q2_numeric as q2
from check_biv150_konly_allreader_numeric_cpu import partition,reject
from check_biv150_konly_prefill_compile_cpu import EmptyOnlyTorch


def run():
    device,proof=q2.transforms();private=device.__globals__['contract'];results=[]
    for case in (0,1):
        p=q2.plan('raw3d',case);kind=q2.statuses(p);partition(p)
        assert p['total']==96 and len(p['qlens'])==64 and max(p['qlens'])==2
        tensor,owners=private.descriptors.allocate(EmptyOnlyTorch(),p);desc=private.descriptors.descriptors(tensor)
        private.descriptors.check_descriptors(p,desc)
        reject(lambda:q2.frozen.contract.descriptors.check_descriptors(p,desc))
        for role,field,value in [('k','dtype','bfloat16'),('kscale','storage_offset',1024),('scratch','stride',(16384,2048,128,1))]:
            mutant=deepcopy(desc);mutant[role][field]=value
            reject(lambda mutant=mutant:private.descriptors.check_descriptors(p,mutant))
        expected={}
        for row in q2.reference.rows('raw3d',p['qlens'],p['klens'],16):
            if row['head']==0: expected[row['token'],row['segment']]=1 if row['keys'] else 2
        for token,segments in enumerate(kind):
            for seg,value in enumerate(segments): assert value==expected.get((token,seg),0)
        coverage=q2.reference.coverage('raw3d',p['qlens'],p['klens'],16)
        empty=sum(value==2 for row in kind for value in row);assert empty>0
        results.append(dict(case=case,plan=p,coverage=coverage,causal_empty_token_segments=empty,
                            private_metadata_passed=True,frozen_maxq1_still_refuses=True))
    bad=deepcopy(p);bad['qlens'][0]=3;reject(lambda:q2.statuses(bad))
    reject(lambda:q2.plan('multi2',0));reject(lambda:q2.plan('raw3d',True))
    assert not any(name in sys.modules for name in ('torch','triton','numpy','vllm_fl'))
    return dict(cpu_only=True,proof=proof,cases=results,actual_binding_verified=False,raw_qlen2_verified=False,
        zero_iteration_empty_segment_not_permitted_by_this_geometry=True,
        Graph_verified=False,side_stream_verified=False,timing_attempted=False,precision_gate=False,service_verified=False)


if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
