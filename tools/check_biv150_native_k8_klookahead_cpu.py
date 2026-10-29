#!/usr/bin/env python3
"""Bounded address/stage ownership oracle. No compiler, Torch or device work."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import biv150_native_k8_klookahead_factory as f
import compile_biv150_native_k8_klookahead as frontend
import compile_biv150_native_k8_v_overlap as parent_frontend


def kindex(slot, dim):
    panel, within = divmod(dim, 64)
    quartet, byte = divmod(within, 4)
    bank, subslot = divmod(slot, 4)
    return panel*1024+bank*256+(quartet%4)*64+((quartet//4)^bank)*16+subslot*4+byte


def trace(full_pages, tail):
    """Logical stage oracle; not an execution/synchronization simulator."""
    events = []
    if full_pages:
        events += [('Kdirect',full_pages[0]),('barrier',None)]
    for i, page in enumerate(full_pages):
        events += [('Vissue',page),('QK',page),('P',page),('Vwait',page),('barrier',None)]
        if i+1 < len(full_pages):
            events += [('NextPageId',full_pages[i+1]),('Kread',full_pages[i+1])]
        events += [('PV',page)]
        if i+1 < len(full_pages):
            events += [('Kpublish',full_pages[i+1])]
        events += [('barrier',None)]
    if tail is not None:
        events += [('Kdirect',tail),('barrier',None),('Vissue',tail),('QK',tail),
                   ('P',tail),('Vwait',tail),('Vclear',tail),('barrier',None),
                   ('PV',tail),('barrier',None)]
    return events


def audit(events, expected):
    k = None; k_ready = False; pending = None; v = None; v_ready = False; v_completed = False
    qk = None; pv = None; next_id = None; reads = []; consumed = []
    for op, page in events:
        if op == 'Kdirect':
            assert pending is None and v is None
            k = page; k_ready = False; reads.append(page)
        elif op == 'barrier':
            k_ready = k is not None
            if v_ready and pv == v:
                v = None; v_ready = False; v_completed = False
            elif v_completed:
                v_ready = True
        elif op == 'Vissue':
            assert v is None and k_ready and k == page
            v = page; v_ready = False; v_completed = False; qk = pv = None
        elif op == 'QK':
            assert k_ready and k == page and v == page
            qk = page
        elif op == 'P':
            assert qk == page
        elif op == 'Vwait':
            assert v == page and qk == page and pending is None
            v_completed = True
        elif op == 'Vclear':
            assert v_completed and v == page
        elif op == 'NextPageId':
            assert v_ready and qk == v and pv is None and pending is None and next_id is None
            assert page in expected and page != v
            next_id = page
        elif op == 'Kread':
            assert v_ready and qk == v and pv is None and pending is None
            assert page in expected and page != v and next_id == page
            pending = page; reads.append(page)
        elif op == 'PV':
            assert v_ready and v == page and k_ready and k == page and qk == page
            pv = page; consumed.append(page)
        elif op == 'Kpublish':
            assert pending == page and pv == v and qk == v
            k = page; k_ready = False; pending = None; next_id = None
        else:
            raise AssertionError('Unknown stage transition')
    assert pending is None and v is None and next_id is None
    assert reads == expected and consumed == expected


def reject(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Unsafe mutant accepted')


def check():
    proof = f.verify()
    assert frontend.CUDA_FLAGS == parent_frontend.CUDA_FLAGS
    source = f.path('qk.cu').read_text()
    # Exact inverse protects *all* arithmetic, original tail and typed host ABI.
    for role in f.FILES:
        assert f.inverse(f.path(role).read_text(),role) == f.original(role)
    body = source.split('__device__ __forceinline__ int consume_page(',1)[1]
    body = body.split('\n}\n}',1)[0]
    anchors = ['nvcuda::wmma::mma_sync_tcu(scores,q[k],b,scores);',
        'A<PVShape> probability;probability[0]=packed;probability[1]=packed;',
        '__ivcorex_sl_waitcnt(8ULL);','if(valid<16)clear_v_tail(stage,valid,lane);',
        '  __syncthreads();','next_page=next_row[next_position/16];',
        'read_next_k(next,key,next_page,head,lane);',
        'BRow<PVShape> b;', 'nvcuda::wmma::mma_sync_tcu(acc[n],probability,b,acc[n]);',
        'if(has_next)publish_next_k(stage,next,lane);',
        '__syncthreads(); // Publish dependent K stores;']
    positions = [body.index(x) for x in anchors]
    assert positions == sorted(positions)
    assert body.count('__ivcorex_sl_waitcnt(8ULL);') == 1
    assert 'row[next_position/16]' not in f.NEW_LOOP
    assert 'template<bool OverlapV,bool KReady=false>' in source
    old_tail = f.original('qk.cu').split(' if(full_upper<upper){',1)[1].split('\n }',1)[0]
    new_tail = source.split(' if(full_upper<upper){',1)[1].split('\n }',1)[0]
    for old, new in reversed(f.RENAMES):
        new_tail = new_tail.replace(new, old)
    assert new_tail == old_tail
    # Full words cover the exact original K image; every source is aligned and
    # within a valid full page. Check the int32 max page with I64 byte arithmetic.
    address_checks = 0
    for page in (0,1,2**31-1):
        for head in (0,1):
            destination = set(); sources = set()
            for lane in range(64):
                for i in range(8):
                    word = lane+i*64; slot, dim = word//32,(word%32)*4
                    address = page*16384+slot*256+head*128+dim
                    target = kindex(slot,dim)
                    assert address%4 == target%4 == 0
                    assert page*16384 <= address < address+4 <= page*16384+4096
                    assert target not in destination
                    destination.add(target);sources.update(range(address,address+4))
                    address_checks += 1
            assert destination == set(range(0,2048,4)) and len(sources) == 2048
            assert not set(range(2048)) & set(range(4096,8192))
    # Independent segment/page sets, including no-full-page, exactly-one, last
    # full before tail, and boundaries crossing the division into S2/S4.
    cases = 0; empty = 0; max_read_pages = 0
    for length in (*range(66),127,128,129,193,257,4096,16384):
        for segments in (2,4):
            page_span = (length+segments*16-1)//(segments*16)
            for segment in range(segments):
                lower = segment*page_span*16
                upper = min(lower+page_span*16,length)
                expected = list(range(lower//16,(upper+15)//16)) if lower < length else []
                full_upper = lower+(max(upper-lower,0)//16)*16
                full = list(range(lower//16,full_upper//16)) if lower < length else []
                tail = full_upper//16 if lower < length and full_upper < upper else None
                assert full+([] if tail is None else [tail]) == expected
                audit(trace(full,tail),expected)
                max_read_pages = max(max_read_pages,len(expected))
                cases += 1;empty += not expected
    faults = []
    good = trace([0,1],None)
    for name, mutate in (
        ('page_id_before_V_wait',lambda e:e.insert(e.index(('Vwait',0)),e.pop(e.index(('NextPageId',1))))),
        ('page_id_before_V_ready_barrier',lambda e:e.insert(e.index(('Vwait',0))+1,e.pop(e.index(('NextPageId',1))))),
        ('read_before_V_wait',lambda e:e.insert(e.index(('Vwait',0)),e.pop(e.index(('Kread',1))))),
        ('read_before_V_ready_barrier',lambda e:e.insert(e.index(('Vwait',0))+1,e.pop(e.index(('Kread',1))))),
        ('publish_before_PV',lambda e:e.insert(e.index(('PV',0)),e.pop(e.index(('Kpublish',1))))),
        ('missing_publish_barrier',lambda e:e.pop(e.index(('Kpublish',1))+1)),
        ('next_segment_read',lambda e:e.__setitem__(e.index(('Kread',1)),('Kread',2))),
        ('unknown_transition',lambda e:e.insert(0,('unknown',0))),
    ):
        mutant = copy.deepcopy(good);mutate(mutant)
        reject(lambda:audit(mutant,[0,1]));faults.append(name)
    for name, old, new in (
        ('wrong_K_page_stride','static_cast<std::int64_t>(page)*16384','static_cast<std::int64_t>(page)*8192'),
        ('tail_KReady_true','bool KReady=false','bool KReady=true'),
        ('prefetch_tail','next_position<full_upper','next_position<=full_upper'),
        ('lost_V_wait','__ivcorex_sl_waitcnt(8ULL);',''),
        ('QK_math','qscale0*kscale*scale','qscale0*scale'),
        ('P_rounding','__float2bfloat16_rn(p0)','__float2bfloat16_rz(p0)'),
    ):
        assert old in source
        reject(lambda:f.inverse(source.replace(old,new),'qk.cu'));faults.append(name)
    # No hidden installed-package dependency or accidental opt-in in default.
    default = subprocess.run([sys.executable,'-B','-S',str(f.ROOT/'tools/compile_biv150_native_k8_klookahead.py')],
                             capture_output=True,text=True,check=True)
    assert json.loads(default.stdout) == frontend.cpu_plan()
    denied = subprocess.run([sys.executable,'-B','-S',str(f.ROOT/'tools/compile_biv150_native_k8_klookahead.py'),
                             '--compile-only','--out','/tmp/cjj-klookahead-never-created'],capture_output=True,text=True)
    assert denied.returncode == 2 and 'Root must confirm' in denied.stderr
    assert not Path('/tmp/cjj-klookahead-never-created').exists()
    assert not subprocess.check_output(['git','-C',str(f.ROOT),'diff',f.BASE,'--','vllm_fl','csrc'])
    for path in (f.ROOT/'tools').glob('*klookahead*.py'):
        ast.parse(path.read_text())
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(source=proof,source_order_only=True,source_tail_byte_exact=True,
        address_word_checks=address_checks,segment_cases=cases,empty_segments=empty,
        max_page_reads_one_segment=max_read_pages,each_logical_page_loaded_once=True,
        stage_ownership_model_only=True,rejected=faults,default_stdlib_only=True,
        Root_ack_required=True,production_diff_empty=True,
        actual_CoreX_wait_semantics_verified=False,actual_instruction_order_verified=False,
        compiled=False,kernel_launches=0,numeric_verified=False,performance_verified=False)


if __name__ == '__main__':
    print(json.dumps(check(),sort_keys=True,indent=2))
