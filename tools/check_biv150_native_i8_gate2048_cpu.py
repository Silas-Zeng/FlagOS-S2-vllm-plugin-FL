#!/usr/bin/env python3
"""Small stdlib-only panel/fragment ownership check; no hardware simulation."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE='899ce77470b0b539aebc108fd122f668c2d5894d'


def a_byte(row,col):
    i,c=divmod(col,16)
    return i*256+((row//4*64+c*4+(row&3)+64*i)&255)


def b_byte(row,col):
    r=row//4
    return col//4*256+(r&3)*64+((r//4)^(col//4))*16+(col&3)*4+(row&3)


def check():
    a={a_byte(r,k):(r,k) for r in range(16) for k in range(64)}
    b={b_byte(k,n):(k,n) for k in range(64) for n in range(16)}
    assert set(a)==set(b)==set(range(1024))
    # Independent translations of the installed SDK load_matrix_sync_tcu and
    # getRow/ColEMOffset definitions, checked against semantic coordinates.
    for lane in range(64):
        g,c=divmod(lane,16)
        for half in range(2):
            for component in range(2):
                em=half*2+component
                aword=half*128+component*64+((g+em)%4)*16+c
                bword=(lane//4%4)*64+(lane^(em*4))
                for byte in range(4):
                    assert a[aword*4+byte]==(4*g+byte,32*half+16*component+c)
                    assert b[bword*4+byte]==(32*half+16*component+4*g+byte,c)
    # Actual cooperative writer coordinates: no partial shared-word races.
    writes=[]
    for tid in range(256):
        for word in range(tid,1024,256):
            panel,within=divmod(word,256)
            ar,ak=(within//64)*4,within%64
            bk,bn=(within//16)*4,within%16
            writes.extend((panel*256+a_byte(ar,ak)//4,
                1024+panel*256+b_byte(bk,bn)//4,2048+panel*256+b_byte(bk,bn)//4))
    assert len(writes)==len(set(writes))==3072 and set(writes)==set(range(3072))
    outputs=[(warp*16+lane//16+4*j,n*16+lane%16)
        for warp in range(4) for lane in range(64) for n in range(4) for j in range(4)]
    assert len(outputs)==len(set(outputs))==4096
    assert set(outputs)=={(m,n) for m in range(64) for n in range(64)}
    assert 2048%64==6144%64==0 and 2048*128*128 < 2**31
    # No production path, quantizer or baseline epilogue change is permitted.
    for path in ('csrc','vllm_fl'):
        assert not subprocess.check_output(['git','-C',str(ROOT),'diff',BASE,'--',path])
    for name in ('check_biv150_native_i8_gate2048_cpu.py','compile_biv150_native_i8_gate2048.py'):
        ast.parse((ROOT/'tools'/name).read_text())
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(passed=True,panel_bytes=1024,unique_shared_word_writers=3072,
        unique_CTA_outputs=4096,CTA=[96,32,1],threads=256,shared_source_bytes=12288,
        production_unchanged=True,unchanged_quantizer=True,
        device_compile=False,numeric_verified=False,performance_verified=False,
        note='CPU mapping proof only; actual device ABI/resources/arithmetic remain unqualified')


if __name__=='__main__':
    print(json.dumps(check(),sort_keys=True,indent=2))
