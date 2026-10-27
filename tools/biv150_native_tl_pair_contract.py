"""Bounded native/Triton BF16 pair plan and CPU numeric/address contracts."""
import hashlib
import json
import math
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
MAIN='b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd'
LIBRARY_SHA='aace693bdb302d2685ec1b5e11e859c8a4a77d7f45c0f4dacddc6b5e49b015b8'
ARENA_PAGES=34276
CASES={'4k':(64,4096,2),'16k':(32,16384,4)}
ORDER=('native','triton','triton','native')


def pins():
    expected=json.loads((ROOT/'docs/evidence/native-tl-gqa-pair-20261028/source-pins.json').read_text())
    for path,value in expected.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    return expected


def table(batch,length):
    pages=(length+15)//16
    if batch*pages>ARENA_PAGES:raise ValueError('active physical pages exceed fixed arena')
    values=list(range(ARENA_PAGES));random.Random(20261028).shuffle(values)
    return [values[r*pages:(r+1)*pages] for r in range(batch)]


def append_slots(mapping,lengths):
    if len(mapping)!=len(lengths):raise ValueError('row count')
    result=[]
    for row,length in zip(mapping,lengths):
        if not 0<=length<=len(row)*16:raise ValueError('length outside mapped table')
        result.append(-1 if length==0 else row[(length-1)//16]*16+(length-1)%16)
    assert len([x for x in result if x>=0])==len(set(x for x in result if x>=0))
    return result


def metrics(actual,reference):
    if not actual or len(actual)!=len(reference):raise ValueError('full output extent')
    if not all(math.isfinite(x) for x in (*actual,*reference)):return dict(passed=False,finite=False)
    delta=[x-y for x,y in zip(actual,reference)]
    maximum=max(map(abs,delta));rms=math.hypot(*delta)/math.sqrt(len(delta))
    rr=math.hypot(*reference)/math.sqrt(len(reference))
    assert rms!=0 or maximum==0
    relative=rms/rr if rr else (0. if maximum==0 else math.inf)
    bad=sum(abs(a-b)>.005+.005*abs(b) for a,b in zip(actual,reference))
    return dict(passed=bad==0 and relative<=.005,finite=True,pointwise_violations=bad,
        relative_rmse=relative,max_abs=maximum,rms=rms,thresholds_relaxed=False)


def plan():
    return dict(main=MAIN,cases=CASES,arena_pages=ARENA_PAGES,page_bytes=16384,
        physical_bytes=ARENA_PAGES*16384,table_stride=8192,permutation_seed=20261028,
        arms=['native-b3-BF16','exact-b3-Triton-GQA'],ABBA_order=ORDER,
        complete_step=['same_BF16_append','producer','same_b3_reducer_warps4'],
        initial_pack_copy_separate=True,K8_in_pair=False,old_K8_timing_comparable=False,
        source_pins=pins(),default_CPU=True,GPU_executed=False,model_extrapolation=False)
