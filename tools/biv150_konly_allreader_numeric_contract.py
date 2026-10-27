"""CPU contracts for actual M2/raw3D numeric feeds; no framework imports."""
import difflib
import gzip
import io
import json
from pathlib import Path
import re
import tarfile

import bench_biv150_konly_prefill_compile as parent
import bench_biv150_konly_q32_numeric as frozen
import biv150_konly_prefill_compile_contract as descriptors
from biv150_konly_prefill_reference import grid

DIRECTORY=parent.ROOT/'docs/evidence/konly-prefill-reader-20261027/allreader-numeric-v1'
PINS={
 'multi2':('c92b9581b56e555aeef4ee9ca12e80dbf8e225575d9691ae3acf90babb1afb79','2d1702bf2a05fe88fcce43f85e13efd469df4bfe91d538722f958b7fd56c11d4',40,3043094,637816,345220722),
 'raw3d':('a31c89a7ac563eb90cd525507fcbf3b48acff540b34dcaebef1e2a8d697e84ef','d7d871575b3c9b24e6ace9439f57cea4800cb3e28bcb9ac6a2fd59dc09ba38b7',55,3155673,637948,345239761)}
HARD={'codec','finite','baseline_reference','output_canaries','readonly_inputs','actual_IR','workspace_guards'}


def original(profile):
    if profile not in PINS: raise ValueError('Only M2/raw3D actual profiles')
    digest,mdigest,count,size,pid,start=PINS[profile]
    packed=(DIRECTORY/(profile+'-compile-original.tar.gz')).read_bytes()
    if parent.factory.sha(packed)!=digest: raise ValueError('Original compile archive changed')
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        rows=archive.getmembers()
        if len(rows)!=count+1 or len({m.name for m in rows})!=len(rows) or any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in rows):
            raise ValueError('Unsafe/unknown original members')
        files={m.name:archive.extractfile(m).read() for m in rows}
    if parent.factory.sha(files['download-manifest.json'])!=mdigest: raise ValueError('Original manifest changed')
    manifest=json.loads(files['download-manifest.json'])
    if (manifest['raw_files'],manifest['raw_bytes'],manifest['read_only'],manifest['model_rerun'])!=(count,size,True,False): raise ValueError('Original transport scope changed')
    if manifest['original_owner']!=dict(pid=pid,start_ticks=start,same_start=True,state='Z'): raise ValueError('Original owner changed')
    if set(files)!={r['relative'] for r in manifest['files']}|{'download-manifest.json'}: raise ValueError('Original membership changed')
    for row in manifest['files']:
        value=files[row['relative']]
        if (len(value),parent.factory.sha(value))!=(row['bytes'],row['sha256']): raise ValueError('Original SHA changed')
    summary=json.loads(files['out/compile-summary.json'])
    if (summary['completed'],summary['errors'],summary['kernel_launches'],summary['plan']['profile'])!=(True,[],0,profile): raise ValueError('Not actual successful compile-only profile')
    source=parent.source_bundle(profile)
    if files['out/sources/baseline.py'].decode()!=source[1] or files['out/sources/candidate.py'].decode()!=source[2]: raise ValueError('Reader source changed')
    roots={}
    for role in ('baseline','candidate'):
        record=next(r for r in summary['records'] if r['label']==role)
        roots[role]=str(Path(re.search(r'#loc = loc\("([^\"]+)"',files['out/'+record['files']['ttir']['path']].decode())[1]).parent)
    if len(set(roots.values()))!=1: raise ValueError('Unknown role source roots')
    return files,summary,roots['candidate']


def plan(profile,case=0):
    if profile not in PINS or type(case) is not int or case not in (0,1): raise ValueError('Explicit profile/case required')
    p=descriptors.plan(profile)
    if profile=='multi2':
        qlens=[2047,1] if case==0 else [2046,2]
        klens=[4096,33] if case==0 else [4095,17]
    else:
        qlens=[1]*64
        klens=([15,16,17,31,32,33,63,64] if case==0 else [1,2,3,16,17,32,33,65])*8
    p.update(qlens=qlens,klens=klens,total=sum(qlens),sequences=len(qlens),pages=2*sum((n+15)//16 for n in klens)+5,
             grid=grid(profile,qlens,p['segments']),table_shape=(len(qlens),8192),
             fixture='private numeric geometry; not a formal batchTokens change',formal_batch_tokens_changed=False)
    # Allocation estimates of the old compile-only geometry are not this case.
    for name in ('empty_allocation_bytes','total_empty_allocation_bytes','packed_allocation_bytes','baseline_allocation_bytes','q_and_out_allocation_bytes','initialized_tensor_elements','metadata_values_initialized','kernel_launches'):
        p.pop(name,None)
    return p


def matrix_counts(assembly):
    return dict(I8_QK=assembly.count('ml_matrix_mad_i32_i8_'),BF16=assembly.count('ml_matrix_mad_f32_bf16_'))


def bind_actual(profile,record,expected,observed,reviewed):
    changed=frozen.differences(record,expected)
    if set(changed)-{'asm','cubin'}: raise ValueError('Actual source/typed ABI/options/required IR changed: '+json.dumps(changed,sort_keys=True))
    for ext in ('source','ttir','ttgir','llir','asm','cubin'):
        if parent.factory.sha(observed[ext])!=record['files'][ext]['sha256'] or parent.factory.sha(reviewed[ext])!=expected['files'][ext]['sha256']:
            raise ValueError('Actual/reviewed original SHA mismatch')
    if len(observed['cubin'])<64 or observed['cubin'][:6]!=b'\x7fELF\x02\x01': raise ValueError('Unknown actual binary')
    now,old=observed['asm'].decode(),reviewed['asm'].decode()
    instructions=lambda text:[s.strip() for s in text.splitlines() if s.strip().startswith(('ml_','sl_'))]
    current,prior=instructions(now),instructions(old)
    if [s.split()[0] for s in current]!=[s.split()[0] for s in prior]: raise ValueError('Actual ASM opcode sequence changed')
    controls=lambda ops:[s for s in ops if s.startswith(('sl_cbr','sl_jump','sl_barrier','sl_wait'))]
    if controls(current)!=controls(prior): raise ValueError('Actual ASM control/synchronization changed')
    role=record['label'];counts=matrix_counts(now)
    wanted=({'I8_QK':16,'BF16':32} if profile=='multi2' else {'I8_QK':4,'BF16':8}) if role=='candidate' else (
        {'I8_QK':0,'BF16':64 if profile=='multi2' else 16} if role=='baseline' else {'I8_QK':0,'BF16':0})
    if counts!=wanted: raise ValueError('Actual native I8-QK/BF16-PV/reducer counts changed')
    delta='\n'.join(difflib.unified_diff(old.splitlines(),now.splitlines(),fromfile='reviewed.asm',tofile='actual.asm',n=2))
    if len(delta.encode())>65536: raise ValueError('Actual ASM diff outside bounded contract')
    return dict(source_typed_ABI_options_required_IR_exact=True,actual_native_matrix_counts=counts,
                ASM_opcode_and_control_exact=True,ASM_text_exact=now==old,ASM_diff=delta,
                actual_cubin_sha256=parent.factory.sha(observed['cubin']),actual_cubin_exact=observed['cubin']==reviewed['cubin'],
                binary_equivalence_claimed=False,actual_I8_operand_hoist_verified=False)


def policy(hard,checks,diagnostic):
    if set(hard)!=HARD or any(v is not True for v in hard.values()): raise ValueError('Implementation/codec/guard/IR failure cannot be bypassed')
    if not checks['baseline_hard_passed'] or not checks['quantized_operand']['finite'] or not checks['quantized_operand']['pointwise_passed']:
        raise ValueError('Baseline/quantized implementation failure cannot be bypassed')
    if checks['candidate_numeric_passed'] is not True and not diagnostic: raise ValueError('Original BF16 proxy failed; explicit diagnostic required')
    return dict(implementation_passed=True,eager_numeric_passed=checks['candidate_numeric_passed'],precision_gate=False,
                Graph_verified=False,side_stream_verified=False,timing_attempted=False,writer_ms=None,fullcost_ms=None)


def references(cpu,codes,scales,qlens,klens,uniform,torch,np):
    """Independent global FP64 softmax; probability/online rounding not modeled."""
    rows=[];original=[];quantized=[];start=0
    for seq,(qlen,length) in enumerate(zip(qlens,klens)):
        local=list(range(qlen)) if uniform or qlen<=64 else sorted({0,1,15,16,31,32,qlen//2,qlen-1})
        ids=cpu['table'][seq,:(length+15)//16].long()
        value=cpu['packed'][ids,1].reshape(-1,2,128)[:length].double()
        selected=cpu['q'][[start+x for x in local]]
        if uniform:
            answer=value.cumsum(0)[torch.tensor([length-qlen+x for x in local])]
            answer=answer/torch.tensor([length-qlen+x+1 for x in local],dtype=torch.float64)[:,None,None]
            answer=answer[:, :,None,:].expand(-1,2,8,128).reshape(len(local),16,128)
            old=new=answer
        else:
            key=cpu['packed'][ids,0].reshape(-1,2,128)[:length].double()
            kc=codes[ids].reshape(-1,2,128)[:length].long();ks=scales[ids].reshape(-1,2)[:length].double()
            qc,qs,_=frozen.numeric.quantize_cpu(selected,torch,np)
            q=selected.double().reshape(len(local),2,8,128).permute(1,2,0,3)
            score=torch.einsum('hgrd,thd->hgrt',q,key)*128**-.5
            integer=np.einsum('hgrd,thd->hgrt',qc.long().reshape(len(local),2,8,128).permute(1,2,0,3).numpy(),kc.numpy(),dtype=np.int64,optimize=True)
            if int(np.max(np.abs(integer)))>128*127**2: raise ValueError('Independent INT64 QK bound failed')
            qscore=torch.from_numpy(integer).double()*qs.double().reshape(len(local),2,8).permute(1,2,0)[:,:,:,None]*ks.T[:,None,None,:]*128**-.5
            mask=torch.arange(length)[None,:]<(length-qlen+torch.tensor(local)[:,None]+1)
            def consume(s):
                probability=s.masked_fill(~mask[None,None],-float('inf')).softmax(-1)
                return torch.einsum('hgrt,thd->rhgd',probability,value).reshape(len(local),16,128)
            old,new=consume(score),consume(qscore)
        rows.extend(start+x for x in local);original.append(old);quantized.append(new);start+=qlen
    return dict(rows=rows,original=torch.cat(original),quantized=torch.cat(quantized),uniform_full_rows=uniform,
                probability_rounding_modeled=False,online_softmax_rounding_modeled=False)
