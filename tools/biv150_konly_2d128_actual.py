"""CPU-only actual BM128 source/ABI/native-path audit; admits one frozen compile."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import bench_biv150_konly_2d128_compile as frontend
import biv150_konly_2d128_compile_contract as contract
from biv150_konly_2d128_llvm_observation import llvm_observation

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'docs/evidence/konly-prefill-2d128-20261027/compile-v1'
COMMIT='78ebd23bc0ae1517293dbbd07b4aef895dff7755'
MANIFEST_SHA='a07113f3c79cda2ef68b044996ce525a5587b60775604242bc01ec83abba49d4'
CANDIDATE_HASH='aacfe384e0d1562b51a8f9eb27e174a9bb66caf4a06e660ff4cbc94fb4230fcd'
BASELINE_HASH='780806697389702032978aa79c9e12ec47a54b6b84153e20b3578e57c010b479'
IR_SHA=dict(ttgir='8ad3eb5cbe623f62f40efa974d88f456bcea2801803538498159106a0551b533',
            llir='4ed3c0e41d0b12c601354de7823530a822b80a68078c6fc5ae5410b115d139db',
            asm='d3abd5eaa89fde26c1966ac45da369f4a3483b530e201fa30f15c5e0482f7c17')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def canonical(value):return json.loads(json.dumps(value))


def encode(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def packed(files):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w',format=tarfile.PAX_FORMAT) as tar:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
            info.uid=info.gid=0;info.uname=info.gname='';tar.addfile(info,io.BytesIO(data))
    out=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0,compresslevel=9) as gz:gz.write(raw.getvalue())
    need(len(out.getvalue())<=2*1024*1024,'2MiB publisher bound');return out.getvalue()


def files_from(path):
    if path.is_dir():
        need(not any(p.is_symlink() for p in path.rglob('*')),'raw symlink')
        files={p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*') if p.is_file()}
    else:
        files={}
        with tarfile.open(path,'r:gz') as tar:
            for member in tar.getmembers():
                need(member.isfile() and member.name not in files,'regular unique archive member')
                files[member.name]=tar.extractfile(member).read()
    need(all(not n.startswith('/') and '..' not in Path(n).parts for n in files),'safe membership')
    return files


def mechanism(ttgir,llir,assembly,*,known=True):
    if known:need(all(sha(s)==IR_SHA[k] for k,s in [('ttgir',ttgir),('llir',llir),('asm',assembly)]),'unknown actual BM128 IR')
    lines=ttgir.splitlines()
    def one(text):
        rows=[i+1 for i,line in enumerate(lines) if text in line]
        need(len(rows)==1,'unknown TTGIR grammar '+text);return rows[0]
    qload=one('%Q_157 = tt.load');cast=one('%4 = arith.fptosi');allocation=one('%5 = ttg.local_alloc %4')
    loop=one('%acc:3 = scf.for');read=one('%10 = ttg.local_load %5');qk=one('%score_integer = tt.dot');pv=one('%acc_300 = tt.dot')
    need(qload<cast<allocation<loop<read<qk<pv,'Q encode/allocation outside, operand read inside KV')
    for text in ('%cst_16 = arith.constant dense<4096>','%cst_17 = arith.constant dense<1024>',
                 '%cst_18 = arith.constant dense<2>','%cst_19 = arith.constant dense<1.000000e+00>',
                 '%physical_block_idx_224 = arith.extsi %physical_block_idx_221',
                 '%offset_260 = arith.muli %physical_block_idx_224, %cst_16',
                 '%offset_261 = arith.addi %offset_260, %cst_17',
                 '%offset_262 = arith.extsi %v_offset_232',
                 '%offset_263 = arith.muli %offset_262, %cst_18',
                 '%offset_264 = arith.addi %offset_261, %offset_263',
                 '%offset_265 = arith.addi %offset_264, %offset',
                 '%offset = tt.splat %v_offset : i64',
                 '%v_offset = arith.extsi %kv_head_idx : i32 to i64',
                 '%key_code_scale_266 = tt.addptr %key_code_scale, %offset_265',
                 '%key_code_scale_267 = tt.load %key_code_scale_266, %tile_mask_211, %cst_19'):
        one(text)
    need('-> tensor<128x64xi32, #mma>' in lines[qk-1] and '128x128xi8' in lines[qk-1],'native signed I8/I32 QK')
    need('128x64xbf16' in lines[pv-1] and '64x128xbf16' in lines[pv-1],'BF16 probability/V PV')
    llvm=llvm_observation(llir)
    need(len(llvm['Q_shared_reads_inside_KV'])==16 and not llvm['all_Q_operand_reads_outside_KV'],'actual known unhoisted Q operand')
    asm=assembly.splitlines();body=next(i+1 for i,s in enumerate(asm) if s.startswith('BB0_37:'))
    header=next(i+1 for i,s in enumerate(asm) if s.startswith('BB0_38:'))
    need('Header=BB0_38' in asm[body-1] and 'Inner Loop Header' in asm[header-1]
         and any('sl_jump BB0_38' in s for s in asm[:body])
         and any('sl_jump BB0_37' in s for s in asm[header:]),'actual ASM KV control')
    byte=[i+1 for i,s in enumerate(asm) if 'ml_slb_load_b8 ' in s]
    i8=[i+1 for i,s in enumerate(asm) if 'ml_matrix_mad_i32_i8_m16n16k32' in s]
    bf16=[i+1 for i,s in enumerate(asm) if 'ml_matrix_mad_f32_bf16_m16n16k16' in s]
    need(len(byte)==32 and len(i8)==16 and len(bf16)==32
         and all(body<i<header for i in byte+i8+bf16),'actual ASM native paths/Q byte reads')
    return dict(known_actual_BM128_mechanism_reviewed=True,actual_I8_operand_hoist_verified=False,
                Q_BF16_global_read_and_RN32_RNE_quantization_outside_KV=True,
                Q_shared_operand_reads_remain_inside_KV=True,
                TTGIR=dict(Q_global=qload,Q_code_cast=cast,Q_code_store=allocation,KV_loop=loop,Q_operand_read=read,I8_QK=qk,BF16_PV=pv),
                LLVM=llvm,ASM=dict(KV_body=body,KV_header=header,Q_byte_read_lines=byte,I8_QK_lines=i8,BF16_PV_lines=bf16),
                scale_address='full F32 page0 + physical_page_i64*4096 + 1024 + slot_i64*2 + kv_head_i64',
                scale_mask='tile_mask, other=1.0',V_address='already-offset BF16 V view + physical_page*8192 + slot*256 + kv_head*128 + dim',
                native_I8_QK=True,BF16_PV=True,loaded_resources_verified=False,numerical_verified=False,
                precision_gate=False,performance_verified=False,service_verified=False)


def audit(files):
    need(sha(files['download-manifest.json'])==MANIFEST_SHA,'immutable collector SHA')
    m=json.loads(files['download-manifest.json']);need((m['raw_files'],m['raw_bytes'])==(38,3077682),'raw size/count')
    need(m['original_owner']==dict(pid=636649,start_ticks=345102304,same_start=True,state='Z'),'collector owner')
    need(set(files)=={r['relative'] for r in m['files']}|{'download-manifest.json'},'all original files')
    for row in m['files']:
        data=files[row['relative']];need((len(data),sha(data))==(row['bytes'],row['sha256']),'raw original SHA')
    need(sum(len(files[row['relative']]) for row in m['files'])==3077682,'raw bytes')
    summary=json.loads(files['out/compile-summary.json'])
    need(summary['completed'] is True and summary['compile_passed'] is True and summary['errors']==[]
         and summary['commit']==COMMIT and summary['kernel_launches']==0 and summary['JIT_warmups_attempted']==2,'compile-only terminal')
    need(summary['plan']==canonical(contract.plan()),'same component plan')
    source,baseline,candidate,proof=frontend.source_bundle()
    need(summary['source_proof']==proof and files['out/sources/candidate.py']==candidate.encode()
         and files['out/sources/baseline.py']==baseline.encode(),'actual generated source/AST inverse')
    need(files['out/sources/sdk.py.txt']==source['sdk.py'].encode()
         and files['out/sources/helpers.py.txt']==source['helpers.py'].encode(),'actual SDK/helper')
    contract.check_descriptors(contract.plan(),summary['tensor_descriptors'])
    need(summary['actual_I8_operand_hoist_verified'] is False and summary['mechanism_status']=='unknown_not_audited_for_BM128'
         and summary['loaded_resources_verified'] is False,'preserve historical unknown/false fields')
    rows=[]
    for r,label,kernel,shared in zip(summary['records'],('baseline','candidate'),(BASELINE_HASH,CANDIDATE_HASH),(65536,40960)):
        standalone=json.loads(files['out/ir/'+label+'/record.json'])
        need(all(r[k]==v for k,v in standalone.items()),'original record/summary mirror')
        need(r['label']==label and r['hash']==kernel and r['grid']==[129,2],'actual record binding')
        need(r['resources']['n_regs'] is None and r['resources']['n_spills'] is None
             and r['resources']['shared']==shared and r['phase']=='pre_launch_compile_only'
             and r['kernel_launches']==0 and r['loaded_resources_verified'] is False,'prelaunch only')
        need(r['actual_options']['num_warps']==8 and r['actual_options']['num_stages']==1
             and r['actual_options']['warp_size']==64 and r['actual_options']['ir_override'] is None,'no changed options/override')
        signature=r['typed_cache_pointer_proof']['complete_signature']
        need(signature==r['signature'] and r['arguments']['KV_QUANT_MODE']==0
             and r['arguments']['BLOCK_M']==128 and r['arguments']['BLOCK_Q']==16
             and r['arguments']['TILE_SIZE']==64,'actual private mode0/BM128 constants')
        need(signature['key_cache_ptr']==('*i8' if label=='candidate' else '*bf16')
             and signature['value_cache_ptr']=='*bf16','actual K/V typed signature')
        if label=='candidate':need(signature['key_code_scale_ptr']=='*fp32','actual F32 scale typed signature')
        for kind,row in r['files'].items():
            b=files['out/'+row['path']];need((len(b),sha(b))==(row['bytes'],row['sha256']) and len(b)>0,'actual artifact SHA')
        need({'source','ttir','ttgir','llir','asm','cubin'}<=set(r['files']),'complete actual artifacts')
        for ext in ('ttir','ttgir'):
            text=files['out/ir/'+label+'/actual.'+ext].decode()
            need('%key_cache_ptr: !tt.ptr<'+('i8' if label=='candidate' else 'bf16')+'>' in text
                 and '%value_cache_ptr: !tt.ptr<bf16>' in text,'actual typed IR K/V')
            if label=='candidate':need('%key_code_scale_ptr: !tt.ptr<f32>' in text,'actual typed IR scale')
        rows.append(dict(label=label,hash=kernel,shared=shared,n_regs=None,n_spills=None,phase='prelaunch'))
    facts=mechanism(*(files['out/ir/candidate/actual.'+e].decode() for e in ('ttgir','llir','asm')))
    base_asm=files['out/ir/baseline/actual.asm'].decode()
    need(base_asm.count('ml_matrix_mad_f32_bf16_m16n16k16')==64
         and 'ml_matrix_mad_i32_i8_' not in base_asm,'same-geometry BF16 control')
    return dict(raw_files=38,raw_bytes=3077682,download_manifest_sha256=MANIFEST_SHA,
                tested_commit=COMMIT,kernel_launches=0,actual_records=rows,mechanism=facts,
                baseline_is_current_main_policy=False,device_numeric_verified=False,precision_gate=False,
                performance_measured=False,service_installed=False)


def original():
    files=files_from(DIRECTORY/'raw-v1.tar.gz');summary=audit(files)
    return files,summary,json.loads(files['out/compile-summary.json'])
