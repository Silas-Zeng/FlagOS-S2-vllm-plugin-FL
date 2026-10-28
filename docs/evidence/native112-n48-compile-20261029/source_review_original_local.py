#!/usr/bin/env python3
"""Fixed-source N48 CPU audit. No Torch, NumPy, SSH, GPU or candidate SO loading."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/private/tmp/flagos-cjj-compact-native-n48-s4-20261029')
SOURCE='83b6c8c9282445219782ee1377680778a2aa8d3a'
PARENT='4d1166792a5247b774d6b1977e60466574e6a34c'
OLDRAW=Path('/private/tmp/cjj-native110-compile-v1-raw-20261029')
OLDREVIEW=Path('/private/tmp/cjj-root-native110-actual-so-review-20261029.json')
OLDSCRIPT=Path('/private/tmp/flagos-cjj-compact-native-v-overlap-so-review-20261029/docs/evidence/native-k8-compact-v-overlap-actual-so-20261029/original_root_replayed_review.py')
CPUREVIEW=Path('/private/tmp/cjj-n48-independent-existing-cpu-review-resumed-v1-20261029.json')
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists(), 'Never overwrite an existing report'
    assert git('rev-parse','HEAD').decode().strip()==SOURCE
    assert git('rev-parse','HEAD^').decode().strip()==PARENT
    assert not git('status','--porcelain')
    assert not git('diff',PARENT,SOURCE,'--','vllm_fl','csrc')
    sys.path.insert(0,str(ROOT/'tools'))
    import biv150_native_k8_compact_n48_factory as factory
    proof=factory.verify()
    # Independently undo only the declared names and two host admission edits.
    identities={}
    for role,oldfile in factory.FILES.items():
        current=factory.path(role).read_bytes()
        assert current==git('show',SOURCE+':'+str(factory.path(role).relative_to(ROOT)))
        original=git('show',PARENT+':tools/kernels/'+oldfile)
        text=current.decode()
        for old,new in reversed(factory.RENAMES):text=text.replace(new,old)
        if role=='qk.cpp':
            text=text.replace('TORCH_CHECK(segments==4,"Private N48/S4 only");','TORCH_CHECK(segments==2 || segments==4,"S2/S4 only");')
            text=text.replace('const std::int64_t rows=48;','const std::int64_t rows=segments==2?64:32;')
        assert text.encode()==original
        assert (OLDRAW/'out'/oldfile).read_bytes()==original
        identities[role]={'candidate_sha256':sha(current),'actual7892_source_sha256':sha(original)}
    gpu=factory.path('qk.cu').read_text();cpp=factory.path('qk.cpp').read_text()
    early=gpu.index('if(lower>=length)return;')
    assert early<gpu.index('const auto* base=query+')<gpu.index('const auto* row=block_table+')
    assert early<gpu.index('acc_out[a0+n*16+col]')
    assert 'const unsigned lane=__ivcorex_lane_id(),seq=blockIdx.x,h=blockIdx.y,segment=blockIdx.z' in gpu
    assert 'dim3(batch,2,4),64,0,stream' in gpu
    # Full N48/S4 lane-to-output enumeration: unique stores, padded rows untouched.
    partial=set();scalar=set()
    for seq in range(48):
        for segment in range(4):
            for h in range(2):
                for lane in range(64):
                    g,col=divmod(lane,16)
                    for head in (h*8+g,h*8+g+4):
                        for n in range(8):
                            address=seq*32768+head*2048+segment*128+n*16+col
                            assert address not in partial;partial.add(address)
                        if col==0:
                            address=seq*256+head*16+segment
                            assert address not in scalar;scalar.add(address)
    assert len(partial)==48*16*4*128 and max(partial)<48*32768
    assert len(scalar)==48*16*4 and max(scalar)<48*256
    assert all((x%32768)%2048<512 for x in partial)
    assert all((x%256)%16<4 for x in scalar)
    # Producer/reducer use the same active-segment arithmetic, including length 0.
    lengths=[0,1,15,16,17,31,32,63,64,65,193,16896,16899,17408,17411]
    segment_checks=[]
    for length in lengths:
        steps=(length+63)//64;active=[s for s in range(4) if s*steps*16<length]
        reducer_active=(length+max(steps,1)*16-1)//(max(steps,1)*16)
        assert active==list(range(reducer_active))
        segment_checks.append({'length':length,'active_segments':active})
    reducer=ROOT/'vllm_fl/patches/biv150_gqa_decode.py'
    assert reducer.read_bytes()==git('show',PARENT+':vllm_fl/patches/biv150_gqa_decode.py')
    assert 'acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)' in reducer.read_text()
    existing=json.loads(CPUREVIEW.read_text());assert existing['passed'] and len(existing['six_small_fixtures'])==6
    for item in existing['six_small_fixtures']:assert item['all48_rows_full128D_reference_mapping']
    assert [(x['valid_rows'],x['pages_reserved_for_three_appends']) for x in existing['fullstep_budgets']]==[(41,44649),(42,44394)]
    assert sha(OLDREVIEW.read_bytes())=='f5a40726b7ae1c1db122e710c1921815db887a1f06112901c842f976a1c9d0dc'
    assert sha(OLDSCRIPT.read_bytes())=='02fc4aff949091969a058bd1ffed64465b9a5baa73b1e7d57d6cf9178bcaa271'
    prior=json.loads(OLDREVIEW.read_text())
    oldlib=OLDRAW/'out/build/cjj_native_k8_compact_v_overlap_probe.so'
    assert sha(oldlib.read_bytes())==prior['SO_sha256']=='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
    assert not {'torch','triton','numpy'}&set(sys.modules)
    result=dict(schema=1,source=SOURCE,parent=PARENT,source_inverse_and_CPU_address_checks_passed=True,
      CPU_only=True,framework_imported=False,worktree_unchanged=True,production_unchanged=True,
      source_identities=identities,source_factory_proof=proof,source_review_scope='N48/S4 only; Q pitches2048/2560',
      parent_actual_SO_sha256=sha(oldlib.read_bytes()),parent_actual_SO_review_sha256=sha(OLDREVIEW.read_bytes()),
      reused_CPU_fixture_report_sha256=sha(CPUREVIEW.read_bytes()),existing_fixture_count=6,
      partial_distinct_store_positions=len(partial),scalar_distinct_store_positions=len(scalar),
      largest_partial_element_offset=max(partial),largest_scalar_element_offset=max(scalar),
      inactive_segments_and_rows_remain_poison=True,producer_does_not_zero_padding=True,
      reducer_zero_length_semantics='Masked scratch loads, active=0, final output selected +0.0; source proof only',
      reducer_sha256=sha(reducer.read_bytes()),segment_arithmetic=segment_checks,
      no_N32_batch_limit_in_GPU_source=True,
      stale_inherited_comment='qk.cu lines226-227 still describe first-draft S2/future R32; nonexecuting history retained for exact inverse',
      new_candidate_actual_SO_sha256=None,new_candidate_actual_SO_or_IR_audited=False,
      numeric_verified=False,Graph_verified=False,side_stream_verified=False,fullstep_verified=False,
      model_verified=False,performance_verified=False,precision_gate=False,service_enabled=False,
      manual_Root_approval=False,
      limitations=['Host contract checker is a CPU model, not actual TORCH_CHECK execution',
        'No real device, SO loading, encoder, writer, 42-layer capacity or service execution',
        'The two N32 control calls preserve global48-row Q/data identity; extra16 control rows have distinct padding semantics',
        'New actual library/IR/resources/ABI/numerics remain unknown; parent7892 is a reference only'],
      script_sha256=sha(Path(__file__).read_bytes()))
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(a.out),'sha256':sha(a.out.read_bytes()),'source_checks_passed':True,'candidate_device_qualification':False}))
if __name__=='__main__':main()
