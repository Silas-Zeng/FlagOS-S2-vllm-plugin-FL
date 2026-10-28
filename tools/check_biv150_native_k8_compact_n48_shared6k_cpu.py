#!/usr/bin/env python3
"""Complete shared-address/fragment/tail model; no framework/device imports."""
import argparse
import json
from pathlib import Path
import re
import struct
import sys

import biv150_native_k8_compact_n48_shared6k_factory as f

HEADER_SHA='36a0b4f3e5d518780f4549f7a42c9938c982b493c26f0696daa22091c635e17d'
IMPLEMENTATION_SHA='ddb5cd4a1d53aa2cacad501a1b2d138715c045bbf837a162d6c20a144ca45094'


def ki(slot,dim):
    panel,within=divmod(dim,64);quartet,byte=divmod(within,4);bank,subslot=divmod(slot,4)
    return panel*1024+bank*256+(quartet%4)*64+((quartet//4)^bank)*16+subslot*4+byte


def vi(slot,dim):
    panel,c=divmod(dim,16);r=slot//2
    return panel*256+(r//4)*128+((r*32+c*2+(slot&1)+panel*64)&127)


def qk_word(lane,k,component):
    # Actual SDK FragmentCol<...,matrix_b,int8>: Kloop4/Nloop1.
    em=(2*k+component)%4
    return (k//2)*256+(lane//4%4)*64+(lane^(em*4))


def pv_word(lane,n,component):
    # Actual SDK FragmentRow<...,matrix_b,bf16>: Nloop8/Kloop1.
    return n*128+(lane^((n&1)*32))+component*64


def frame(base,valid):
    stage=bytearray([0xa5])*(base+4096)
    for slot in range(16):
        for dim in range(128):stage[ki(slot,dim)]=((slot*73+dim*29+11)&255) if slot<valid else 0
    for slot in range(16):
        for dim in range(128):
            # BF16 special bit patterns are payloads here, never float-converted.
            specials=(0,0x8000,0x7f80,0xff80,0x7fc1,0x7f81,0xffff,0x3f80)
            bits=specials[(slot*5+dim*3)%len(specials)] if dim%5==0 else (slot*1217+dim*43+0x1234)&65535
            struct.pack_into('<H',stage,base+2*vi(slot,dim),bits)
    before=bytes(stage)
    # Actual clear_v_tail: unique uint32 writer, preserve odd-cut low half.
    if valid<16:
        for lane in range(64):
            for i in range(lane,1024,64):
                slot=(i//128)*2;dim=i%128;address=base+4*(vi(slot,dim)//2)
                word=struct.unpack_from('<I',stage,address)[0]
                if slot>=valid:word=0
                elif slot+1>=valid:word&=65535
                struct.pack_into('<I',stage,address,word)
    assert stage[:2048]==before[:2048]
    for slot in range(16):
        for dim in range(128):
            address=base+2*vi(slot,dim)
            assert stage[address:address+2]==(before[address:address+2] if slot<valid else b'\0\0')
    return bytes(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--sdk-header',type=Path);p.add_argument('--sdk-implementation',type=Path);p.add_argument('--old-compile-out',type=Path);a=p.parse_args()
    proof=f.verify();assert struct.calcsize('I')==4
    kbytes=[ki(slot,dim) for slot in range(16) for dim in range(128)]
    vhalves=[vi(slot,dim) for slot in range(16) for dim in range(128)]
    assert sorted(kbytes)==list(range(2048)) and sorted(vhalves)==list(range(2048))
    stores=[ki(w//32,(w%32)*4)//4 for lane in range(64) for w in range(lane,512,64)]
    assert sorted(stores)==list(range(512))
    kr=[qk_word(l,k,c) for l in range(64) for k in range(4) for c in range(2)]
    vr=[pv_word(l,n,c) for l in range(64) for n in range(8) for c in range(2)]
    assert sorted(kr)==list(range(512)) and sorted(vr)==list(range(1024))
    fragments=0;tails=[]
    for valid in range(17):
        old,new=frame(4096,valid),frame(2048,valid)
        assert old[:2048]==new[:2048] and old[4096:]==new[2048:]
        for lane in range(64):
            for n in range(8):
                for c in range(2):
                    offset=4*pv_word(lane,n,c)
                    assert old[4096+offset:4100+offset]==new[2048+offset:2052+offset];fragments+=1
        for offset in kr:assert old[4*offset:4*offset+4]==new[4*offset:4*offset+4]
        tails.append(dict(valid_slots=valid,old_image_sha256=f.sha(old),new_image_sha256=f.sha(new),K_unchanged=True,V_fragment_words_identical=True))
    assert max(2048+4*w+3 for w in vr)==6143 and max(4*w+3 for w in kr)==2047
    assert all((4096+i*1024)%1024==(2048+i*1024)%1024==0 for i in range(4))
    negative=[];cuda=f.path('qk.cu').read_text()
    for label,old,new in (
      ('tail not relocated','index=512+v_half_index','index=1024+v_half_index'),
      ('PV not relocated','stage+512,n,0','stage+1024,n,0'),
      ('global scale accidentally shifted','*3136+1024+col*2','*3136+512+col*2'),
      ('dead BF16 K helper called','issue_k_i8(stage,key,page','clear_k_tail(stage,valid,lane);issue_k_i8(stage,key,page')):
        assert cuda.count(old)==1
        try:f.inverse(cuda.replace(old,new),'qk.cu')
        except AssertionError:negative.append(label)
        else:raise AssertionError('undeclared source mutation accepted')
    sdk_checked=False
    if a.sdk_header or a.sdk_implementation:
        assert a.sdk_header and a.sdk_implementation
        header=a.sdk_header.read_bytes();impl=a.sdk_implementation.read_bytes()
        assert f.sha(header)==HEADER_SHA and f.sha(impl)==IMPLEMENTATION_SHA
        assert b'return laneId ^ (EmIdx * 0x20);' in header
        assert b'SLBTCUOffset = SLBTCUIndex * TCUEmStride' in impl
        assert b'const void*' in impl;sdk_checked=True
    actual=None
    if a.old_compile_out:
        from biv150_native_k8_compact_n48_consumer_contract import contracts
        binding=contracts()[0].compile_binding(a.old_compile_out/'compile-summary.json')
        j=json.loads((a.old_compile_out/'compile-summary.json').read_text())
        assert j['completed'] is True and j['compile_passed'] is True and j['errors']==[]
        assert j['library']['sha256']=='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
        ll=(a.old_compile_out/'biv150_native_k8_compact_n48_qk.ll').read_bytes()
        assert ll.count(b'global [2048 x i32] undef, comdat, align 16')==2
        assert b'clear_k_tail' not in ll and b'k_half_index' not in ll
        assert all(('i32 '+str(value)+'), <4 x i32>').encode() in ll for value in (4096,5120,6144,7168))
        so_ir=(a.old_compile_out/'actual-so-IR.stdout.txt').read_bytes()
        assert so_ir.count(b'global [2048 x i32] undef, comdat, align 16')==2
        assert b'clear_k_tail' not in so_ir and b'k_half_index' not in so_ir
        actual=dict(actual_SO_binding=binding,standalone_LLVM_sha256=f.sha(ll),actual_SO_IR_sha256=f.sha(so_ir),observed_stage_bytes=8192,
          observed_stage_IR_alignment=16,old_SME_destination_offsets=[4096,5120,6144,7168],
          dead_BF16_helpers_absent_in_old_LLVM=True,new_actual_artifact_qualification=False)
    parent_frontend=f.subprocess.check_output(['git','-C',str(f.ROOT),'show',f.PARENT+':tools/compile_biv150_native_k8_compact_n48.py']).decode()
    frontend=(f.ROOT/'tools/compile_biv150_native_k8_compact_n48_shared6k.py').read_text()
    for old,new in (
      ('import biv150_native_k8_compact_n48_factory as factory','import biv150_native_k8_compact_n48_shared6k_factory as factory'),
      ("name='cjj_native_k8_compact_n48_s4_probe'","name='cjj_native_k8_compact_n48_shared6k_probe'"),
      ("requested='deferred-V-wait compile-only; no device qualification'","requested='shared-stage6144-source compile-only; hardware alignment/resources/numerics pending'")):
        assert frontend.count(new)==1;frontend=frontend.replace(new,old)
    assert frontend==parent_frontend
    assert not {'torch','triton','numpy'}&set(sys.modules)
    report=dict(passed=True,source=proof,K_store_and_TCU_read_words=512,V_TCU_read_words=1024,
      K_byte_range=[0,2047],old_V_byte_range=[4096,8191],new_V_byte_range=[2048,6143],
      all_valid_counts=list(range(17)),PV_word_bit_comparisons=fragments,tails=tails,
      SDK_header_sha256=HEADER_SHA,SDK_implementation_sha256=IMPLEMENTATION_SHA,SDK_files_checked=sdk_checked,
      SDK_TCU_alignment_bytes=4,source_stage_alignment_bytes=16,
      compile_frontend_inverse_byte_exact=True,
      all_observed_SME_tile_offsets_keep_1024_alignment=True,SME_hardware_alignment_contract_complete=False,
      V_offset_translation_bytes=-2048,preserves_SME_destination_low_bits=10,actual_destination_base_not_measured=True,
      source_mutations_rejected=negative,dead_BF16_K_helper_would_overlap_new_V_if_called=True,
      old_actual_compile_context=actual,device_executed=False,compile_verified=False,
      actual_shared_bytes=None,occupancy_verified=False,numeric_verified=False,performance_verified=False)
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.out:assert not a.out.exists();a.out.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
