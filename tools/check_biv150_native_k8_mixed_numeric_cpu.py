#!/usr/bin/env python3
"""CPU-only consumer source/layout/review refusals; no NumPy/SDK/device compilation."""
import argparse,copy,json,struct,sys
from pathlib import Path
import biv150_native_k8_mixed_numeric_contract as c

def rejected(fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):return
    raise AssertionError('expected refusal')

def run(parent):
    source,proof=c.consumer_source(parent);assert proof['inverse_byte_exact']
    module=c.load(parent,'probe_biv150_native_k8_consumer_v2','cpu118_parent_consumer')
    records=[]
    # One phase per boundary case is sufficient for this metadata/source check;
    # the device entry retains original three phases and all numerical arrays.
    for n,s,pitch in c.CASES:
        size=32 if s==4 else 64
        f=c.fixture(module,size,n,s,pitch);case=f.fixture(size,s,pitch,0);data=f.input_bytes(case)
        lens=struct.unpack('<'+str(len(data['lengths'])//4)+'i',data['lengths'])[64:-64]
        table=struct.unpack('<'+str(len(data['table'])//4)+'i',data['table'])[64:-64]
        assert len(lens)==size and len(table)==size*8192 and all(x==0 for x in lens[n:])
        assert all(x==-1 for x in table[n*8192:])
        for row,length in enumerate(lens[:n]):
            assert all(0<=x<29 for x in table[row*8192:row*8192+(length+15)//16])
            assert all(x==-1 for x in table[row*8192+(length+15)//16:(row+1)*8192])
        # Physical width16 row/head/segment addressing stays in its real span.
        for row in range(n):
            for head in range(16):
                for seg in range(s):assert row*32768+head*2048+seg*128+127<(row+1)*32768
        records.append(dict(case=[n,s,pitch],reference_rows=size,zero_dummy_rows=size-n,real_table_pitch=8192,reference_dummy_table_all_invalid=True,source_only=True))
    r=dict(approved=True,required_new_rows_segments=[list(x) for x in c.PAIRS],resources={s:{'.vgpr_count':90,'.sgpr_count':70,'.group_segment_fixed_size':8192,'.vgpr_spill_count':0,'.sgpr_spill_count':0,'.private_segment_fixed_size':0,'.warp_size':64} for s in ('S2','S4')},**{x:True for x in c.FLAGS})
    # Labeled schema fixture, never written as an approval receipt or SO proof.
    c.review_fields(r);negative=[]
    for name,fn in [('unknown_regs',lambda x:x['resources']['S2'].__setitem__('.vgpr_count',None)),
                    ('bool_regs',lambda x:x['resources']['S2'].__setitem__('.vgpr_count',True)),
                    ('missing_spill',lambda x:x['resources']['S4'].pop('.sgpr_spill_count')),
                    ('unapproved',lambda x:x.__setitem__('approved',False)),
                    ('old_scope',lambda x:x.__setitem__('required_new_rows_segments',[[31,4],[63,2]]))]:
        v=copy.deepcopy(r);fn(v);rejected(lambda:c.review_fields(v));negative.append(name)
    for flag in c.FLAGS:
        v=copy.deepcopy(r);v[flag]=False;rejected(lambda:c.review_fields(v));negative.append(flag)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,parent_fixture=c.original_consumer.PARENT,consumer_source_inverse=proof,
        case_metadata=records,review_schema_fixture_not_actual_approval=True,review_refusals=negative,
        new_library_loaded=False,device_executed=False,numeric_verified=False,performance_verified=False,service_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-source',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.parent_source),sort_keys=True,indent=2))
