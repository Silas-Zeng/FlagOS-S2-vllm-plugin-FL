#!/usr/bin/env python3
"""Stdlib exhaustive byte mapping and source/format rejection; no GPU claims."""
import copy
import json
import struct
import subprocess
import sys

import biv150_native_k8_compact_factory as f
import probe_biv150_native_k8_compact as frontend


def run():
    proof = f.verify()
    rejected = []
    def reject(name, call):
        try:
            call()
        except (AssertionError,KeyError,TypeError):
            rejected.append(name)
        else:
            raise AssertionError('accepted mutant: '+name)
    source = f.path('qk.cu').read_text()
    for name,old,new in (
        ('wrong_K_page','page)*12544+head','page)*16384+head'),
        ('wrong_V_page','page)*6272+head','page)*8192+head'),
        ('double_scale_offset','page)*3136+1024','page)*3136+2048'),
        ('shared_V_moved','dst+4096,desc','dst+4352,desc'),
        ('wait_changed','__ivcorex_sl_waitcnt(8ULL)','__ivcorex_sl_waitcnt(7ULL)'),
        ('BF16P_changed','__float2bfloat16_rn(p0)','__float2bfloat16_rn(p0*.99f)'),
    ):
        assert old in source
        reject(name,lambda bad=source.replace(old,new):f.inverse(bad,'qk.cu'))
    host = f.path('qk.cpp').read_text()
    for name,old,new in (
        ('host_unapproved_N40','segments==2?64:32','segments==2?64:40'),
        ('host_double_V_offset','kb+4352','kb+8704'),
        ('host_old_span','kb+pages*12544','kb+pages*16384'),
    ):
        assert old in host
        reject(name,lambda bad=host.replace(old,new):f.inverse(bad,'qk.cpp'))
    receipt = dict(format=f.FORMAT,epoch=17,purpose='Root-owned synthetic component fixture',
                   storage_id=1234,layer_base=2**40+256,pages=2,layer_bytes=2*f.PAGE,layout_checked=True)
    actual = f.roles(2,receipt['layer_base'],receipt['storage_id'])
    assert f.admit_fixture(receipt,actual,epoch=17)
    for name,field,value in (
        ('old_format','format','cjj-konly-rn32-rne-v2'),
        ('absent_epoch','epoch',None),('stale_epoch','epoch',16),('bool_epoch','epoch',True),
        ('nominal_BF16_not_payload_proof','layout_checked',False),('stock_purpose','purpose','stock'),
        ('old_16KiB_extent','layer_bytes',32768),
    ):
        bad = dict(receipt);bad[field] = value
        reject(name,lambda bad=bad:f.admit_fixture(bad,actual,epoch=17))
    for name,role,field,value in (
        ('K_BF16_alias','key','dtype','bfloat16'),
        ('K_old_stride','key','strides',(16384,256,128,1)),
        ('bool_stride_not_int','key','strides',(12544,256,128,True)),
        ('scale_preoffset','scale','pointer',receipt['layer_base']+4096),
        ('V_fullbase','value','pointer',receipt['layer_base']),
        ('V_old_offset','value','pointer',receipt['layer_base']+8192),
        ('V_double_offset','value','pointer',receipt['layer_base']+8704),
        ('V_old_stride','value','strides',(8192,256,128,1)),
        ('foreign_storage','value','storage_id',4321),
    ):
        bad = copy.deepcopy(actual);bad[role][field] = value
        reject(name,lambda bad=bad:f.admit_fixture(receipt,bad,epoch=17))
    # Independent CPU logical payload -> two physical formats -> all bytes
    # read back. Padding stays sentinel. No synthetic target IR is involved.
    compact = bytearray([0xa5])*(2*f.PAGE)
    canonical = bytearray([0xa5])*32768
    counts = dict(K=0,scale=0,V=0)
    for page in range(2):
        for slot in range(16):
            for head in range(2):
                s = struct.pack('<f',(1+page+slot/32+head/4)/127)
                old_s = page*16384+4096+(slot*2+head)*4
                new_s = f.addresses(page,slot,head,0)[1]
                canonical[old_s:old_s+4] = compact[new_s:new_s+4] = s
                counts['scale'] += 4
                for dim in range(128):
                    k = struct.pack('b',(page*67+slot*11+head*19+dim)%255-127)
                    v = struct.pack('<H',0x3f00+((dim+slot*3+head*7+page*13)%128))
                    old_k = page*16384+slot*256+head*128+dim
                    old_v = page*16384+8192+(slot*256+head*128+dim)*2
                    new_k,_,new_v = f.addresses(page,slot,head,dim)
                    canonical[old_k:old_k+1] = compact[new_k:new_k+1] = k
                    canonical[old_v:old_v+2] = compact[new_v:new_v+2] = v
                    counts['K'] += 1;counts['V'] += 2
    # Read both layouts independently after construction, not only at writes.
    for page in range(2):
        for slot in range(16):
            for head in range(2):
                for dim in range(128):
                    k,s,v = f.addresses(page,slot,head,dim)
                    old = page*16384
                    assert compact[k] == canonical[old+slot*256+head*128+dim]
                    assert compact[s:s+4] == canonical[old+4096+(slot*2+head)*4:old+4100+(slot*2+head)*4]
                    old_v = old+8192+(slot*256+head*128+dim)*2
                    assert compact[v:v+2] == canonical[old_v:old_v+2]
        assert compact[page*f.PAGE+4224:page*f.PAGE+4352] == bytes([0xa5])*128
    default = subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_compact.py')],capture_output=True,text=True,check=True)
    assert json.loads(default.stdout) == frontend.plan()
    no_root = subprocess.run([sys.executable,'-B',str(f.ROOT/'tools/probe_biv150_native_k8_compact.py'),
        '--compile-only','--out','/private/tmp/not-created-cjj-compact-cpu'],capture_output=True,text=True)
    assert no_root.returncode != 0 and 'Root exclusive' in no_root.stderr
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,source_proof=proof,rejected=rejected,byte_remap_counts=counts,
        complete_logical_payload_bits_equal=True,compact_payload_sha256=f.sha(compact),canonical_payload_sha256=f.sha(canonical),
        default_plan_equal=True,no_Root_compile_rejected=True,actual_target_or_runtime_verified=False)


if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True,indent=2))
