#!/usr/bin/env python3
"""Bounded K-pair address/image proof, no frameworks and no device simulation."""
import json
import random
import struct
import sys
import biv150_native_k8_pairload_factory as f
from biv150_native_k8_qk_cpu import key_byte,key_address


def main():
    proof=f.verify();candidate=f.build();cu='biv150_native_k8_pairload_qk.cu';text=candidate[cu]
    negatives=[]
    for name,old,new in (
      ('odd_first_word','word=2*pair,slot=word/32','word=2*pair+1,slot=word/32'),
      ('wrong_pair_step','pair+=64','pair+=32'),
      ('unguarded_read','if(slot<valid){','if(true){'),
      ('assumed_contiguous_shared','stage[k_i8_index(slot,dim+4)/4]','stage[k_i8_index(slot,dim)/4+1]'),
      ('new_8byte_ABI','alignof(WordPair)==4','alignof(WordPair)==8'),
      ('PV_change','acc[n][2]=acc[n][2]*rho0;','acc[n][2]=0.0f;'),
      ('Q_codec_change','cjj_k8_pairload_code_rne(values0[i],input_scale0)','0'),
      ('wait_change','__ivcorex_sl_waitcnt(8ULL)','__ivcorex_sl_waitcnt(0ULL)')):
        assert old in text
        bad=dict(candidate);bad[cu]=text.replace(old,new,1)
        try:f.inverse(bad)
        except ValueError:negatives.append(name)
        else:raise AssertionError('Accepted source mutant '+name)
    records=[];permutation_words=0;readonly_cases=0;unaligned8_seen=False
    rng=random.Random(0xC88102028)
    for layer in (4,256):
        raw=bytes(rng.getrandbits(8) for _ in range(layer+3*16384));before=f.sha(raw)
        for page in (0,1,2):
            for head in (0,1):
                for valid in range(17):
                    guard=64;old=bytearray([0xD3])*(guard+8192+guard);new=bytearray(old);expected=bytearray(old)
                    old_words=[];new_words=[];old_reads=[];new_reads=[]
                    for lane in range(64):
                        for word in range(lane,512,64):
                            slot,dim=divmod(word,32);dim*=4
                            address=key_address(page,slot,head,dim,3,layer)
                            value=raw[address:address+4] if slot<valid else bytes(4)
                            if slot<valid:old_reads.append(address)
                            target=guard+key_byte(dim,slot);old[target:target+4]=value;old_words.append(word)
                        for pair in range(lane,256,64):
                            word=2*pair;slot=word//32;dim=(word%32)*4
                            assert word%2==0 and dim<=120
                            address=key_address(page,slot,head,dim,3,layer)
                            assert address%4==0;unaligned8_seen |= address%8==4
                            value=raw[address:address+8] if slot<valid else bytes(8)
                            if slot<valid:new_reads.append(address)
                            a,b=guard+key_byte(dim,slot),guard+key_byte(dim+4,slot)
                            assert a!=b and b-a==64 # checked, never assumed adjacent
                            new[a:a+4]=value[:4];new[b:b+4]=value[4:];new_words.extend((word,word+1))
                    # Independent bytewise logical NHD->TCU map, no pair/word enumeration.
                    for slot in range(16):
                        for dim in range(128):
                            expected[guard+key_byte(dim,slot)]=raw[key_address(page,slot,head,dim,3,layer)] if slot<valid else 0
                    assert sorted(old_words)==sorted(new_words)==list(range(512))
                    assert len(old_reads)==valid*32 and len(new_reads)==valid*16
                    assert sorted(old_reads)==sorted(a+d for a in new_reads for d in (0,4))
                    assert old==new==expected
                    assert new[:guard]==new[-guard:]==bytes([0xD3])*guard
                    assert new[guard+2048:guard+8192]==bytes([0xD3])*6144
                    assert before==f.sha(raw);readonly_cases+=1;permutation_words+=512
                    records.append(dict(layer_byte_offset=layer,page=page,head=head,valid=valid,
                      original_word_reads=len(old_reads),pair_copies=len(new_reads),source_bytes=len(new_reads)*8,
                      shared_words_written=512,shared_image_sha256=f.sha(new[guard:guard+2048])))
    assert unaligned8_seen and readonly_cases==204
    assert not {'torch','numpy','triton','vllm'}.intersection(sys.modules)
    print(json.dumps(dict(passed=True,CPU_only=True,source=proof,records=records,
      cases=readonly_cases,all_512word_permutations_checked=permutation_words,
      original_4byte_not_8byte_alignment_preserved=True,readonly_inputs_and_whole_stage_guards=True,
      source_mutants_rejected=negatives,
      full_CTA='64 lanes *4 source copies *8B =2048B; original64*8*4B =2048B',
      coalescing_change='old2 slots x32 lanes x4B=256B/wave issue; new4 slots x16 lanes x8B=512B/wave issue; each row128B',
      actual_64bit_load_lowering_verified=False,actual_register_or_performance_gain_verified=False,
      GPU_numeric_or_model_verified=False,previous22e_qualification_transferred=False),sort_keys=True,indent=2))


if __name__=='__main__':main()
