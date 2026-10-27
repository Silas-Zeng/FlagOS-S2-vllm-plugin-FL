"""Exact-owner steady4K byte-stage contract. CPU only; v1 proof stays failed."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/evidence/v8-bias-decode-20261027/byte-stage-v2'
SOURCE='a6980ddf1984422b93438d53429bddf36434ccde'
PROOF_SHA='d27751146b16823166a2ed8f080c83255a7434b739fcc6c618ab58374cac93f2'
OLD_ROOT=Path('/workspace/experiments/minicpm-s2/code/cjj-v8-bias-compile4k-v1-20261027')
OLD_OUT=Path('/workspace/experiments/minicpm-s2/cjj-v8-bias-compile4k-v1-20261027')
REJECTIONS=['Shared staging changed: ml_slb_store_b16','Shared staging changed: sl_barrier',
 'Shared layout changed; manual review required','Shared store immediate/mode or barrier signature changed']
KINDS={'ttir','ttgir','llir','asm','cubin','source'}
CONTRACT='steady4k_owner622613_byte_stage_v2'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def source_pins(source_root):
    pins=json.loads((DATA/'a698-source-pins.json').read_text())
    for path,wanted in pins.items():assert sha((source_root/path).read_bytes())==wanted, ('a698 source changed',path)
    return pins


def frozen_namespace(source_root):
    """Execute pinned stdlib definitions only; do not import/pollute old tools."""
    source_pins(source_root)
    factory=dict(__file__=str(source_root/'tools/biv150_v8_decode_factory.py'),__name__='pinned_a698_factory')
    exec(compile((source_root/'tools/biv150_v8_decode_factory.py').read_bytes(),'pinned_factory','exec'),factory)
    ns=dict(ROOT=source_root,sha=sha,ast=ast,copy=copy,re=re,Path=Path)
    for filename,names in [('biv150_v8_bias_probe.py',{'PINS','CASES','OPTION_KEYS'}),
                           ('bench_biv150_v8_bias.py',{'ROLES','MAIN_HASHES'})]:
        tree=ast.parse((source_root/'tools'/filename).read_bytes())
        body=[node for node in tree.body if isinstance(node,ast.FunctionDef) or
              isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)]
        exec(compile(ast.Module(body=body,type_ignores=[]),'pinned_a698_definitions','exec'),ns)
    return ns,factory


def archived_files():
    item=json.loads((DATA/'archive.json').read_text());path=DATA/item['path'];raw=path.read_bytes()
    assert len(raw)==item['bytes']<=2*1024*1024 and sha(raw)==item['sha256']
    with tarfile.open(path) as archive:
        members=archive.getmembers();assert all(x.isfile() for x in members)
        files={x.name:archive.extractfile(x).read() for x in members}
        assert len(files)==len(members)
    download=json.loads(files['download-manifest.json'])
    assert download['original_owner']==dict(pid=622613,start_ticks=342383793,same_start=True,state='Z')
    assert download['raw_files']==67 and download['raw_bytes']==2141528
    assert len(download['files'])==67 and sum(x['bytes'] for x in download['files'])==2141528
    assert set(files)=={x['relative'] for x in download['files']}|{'download-manifest.json'}
    for entry in download['files']:
        raw=files[entry['relative']];assert len(raw)==entry['bytes'] and sha(raw)==entry['sha256']
    return files,download


class LLVM:
    """Narrow grammar for this exact IR, including shared addresses and SSA."""
    def __init__(self,text):
        self.lines=text.splitlines()
        self.defs={m[1]:m[2].split(', !dbg')[0] for line in self.lines if (m:=re.match(r'\s*(%[\w.]+) = (.*)',line))}
        self.cache={}
    def integer(self,name,lane):
        if re.fullmatch(r'-?\d+',name):return int(name)
        key=(name,lane)
        if key in self.cache:return self.cache[key]
        body=self.defs[name]
        if '@llvm.nvvm.read.ptx.sreg.tid.x()' in body:value=lane
        elif m:=re.fullmatch(r'(and|or|xor|shl|lshr|add|mul)(?: \w+)* i\d+ ([%\w.]+|-?\d+), ([%\w.]+|-?\d+)',body):
            a,b=self.integer(m[2],lane),self.integer(m[3],lane)
            value={'and':lambda:a&b,'or':lambda:a|b,'xor':lambda:a^b,'shl':lambda:a<<b,
                   'lshr':lambda:a>>b,'add':lambda:a+b,'mul':lambda:a*b}[m[1]]()
        elif m:=re.fullmatch(r'(?:zext|sext)(?: \w+)* i\d+ (%[\w.]+) to i\d+',body):value=self.integer(m[1],lane)
        elif m:=re.fullmatch(r'icmp eq i\d+ (%[\w.]+), (-?\d+)',body):value=self.integer(m[1],lane)==int(m[2])
        elif m:=re.fullmatch(r'select i1 (%[\w.]+), i\d+ (-?\d+), i\d+ (-?\d+)',body):value=int(m[2] if self.integer(m[1],lane) else m[3])
        elif body.startswith('getelementptr'):
            assert 'ptr addrspace(3)' in body and re.match(r'getelementptr(?: \w+)* i8,',body), body
            if '@global_smem' in body:
                constants=re.findall(r'i32 (-?\d+)',body)
                base=int(constants[0]) if 'getelementptr (i8,' in body else 0
            else:
                parent=re.search(r'ptr addrspace\(3\) (%[\w.]+)',body)[1];base=self.integer(parent,lane)
            offset=re.search(r', i32 ([%\w.]+|-?\d+)$',body)[1]
            value=base+self.integer(offset,lane)
        else:raise AssertionError(('Unknown shared-address SSA',name,body))
        self.cache[key]=value;return value


def byte_stage_ssa(old_text,new_text):
    old,new=LLVM(old_text),LLVM(new_text)
    def packed(ir):
        return [m[1] for line in ir.lines if (m:=re.match(r'\s*(%[\w.]+) = .*@llvm.bi.load.kop.v4i8\(',line))]
    old_packed,new_packed=packed(old),packed(new)
    assert len(old_packed)==len(new_packed)==16
    def stores(ir,kind):
        result=[]
        for i,line in enumerate(ir.lines,1):
            m=re.match(r'\s*store <1 x '+kind+r'> (%[\w.]+), ptr addrspace\(3\) (%[\w.]+)',line)
            if m:result.append((i,m[1],m[2]))
        assert len(result)==64;return result
    old_stores,new_stores=stores(old,'bfloat'),stores(new,'i8')
    matrices=lambda ir:[re.findall(r'<4 x bfloat> (%[\w.]+)',line)[1] for line in ir.lines
        if '@llvm.bi.matrix.mad.f32x4.bf16x4(' in line and not line.startswith('declare')]
    om,nm=matrices(old),matrices(new);assert len(om)==len(nm)==32
    barriers=[i for i,line in enumerate(new.lines,1) if 'call void @llvm.nvvm.barrier.cta.sync.aligned.all' in line]
    assert barriers==[162,435,549,1339,1564], 'Unknown synchronization schema'
    evidence=[]
    for path in (0,1):
        source_ids=[old_packed[path*8:path*8+8],new_packed[path*8:path*8+8]]
        memories=[]
        def raw_symbol(ir,name,lane,ids):
            body=ir.defs[name]
            if m:=re.match(r'sitofp i8 (%[\w.]+) to bfloat',body):return raw_symbol(ir,m[1],lane,ids)
            if m:=re.match(r'insertelement <1 x bfloat> poison, bfloat (%[\w.]+), i64 0',body):return raw_symbol(ir,m[1],lane,ids)
            if m:=re.match(r'extractelement <4 x i8> (%[\w.]+), i64 (\d+)',body):vector,index=m[1],int(m[2])
            else:
                m=re.match(r'shufflevector <4 x i8> (%[\w.]+), <4 x i8> poison, <1 x i32> (.*)',body);assert m,body
                vector=m[1];index=0 if m[2]=='zeroinitializer' else int(re.fullmatch(r'<i32 (\d+)>',m[2])[1])
            if vector not in ids:
                m=re.fullmatch(r'phi <4 x i8> \[ (%[\w.]+), %[\w.]+ \], \[ zeroinitializer, %[\w.]+ \]',ir.defs[vector])
                assert path==1 and m, ('Tail must merge exact zero',ir.defs[vector]);vector=m[1]
            assert vector in ids and 0<=index<4
            return (lane//32+2*ids.index(vector),(lane*4&124)+index)
        for which,(ir,store_list) in enumerate(((old,old_stores),(new,new_stores))):
            memory={}
            for lane in range(64):
                for _,value,ptr in store_list[path*32:path*32+32]:
                    address=ir.integer(ptr,lane);symbol=raw_symbol(ir,value,lane,source_ids[which])
                    assert address not in memory, 'Shared-byte writer collision'
                    memory[address]=symbol
            assert len(memory)==len(set(memory.values()))==2048
            assert set(memory)==(set(range(0,4096,2)) if which==0 else set(range(4096,6144)))
            memories.append(memory)
        assert all(memories[0][2*(address-4096)]==symbol for address,symbol in memories[1].items())
        read_ids=[];read_lines=[]
        def scalar(name,lane):
            # Exact reviewed bias chain, starting at each PV BF16 component.
            for pattern in (r'fptrunc float (%[\w.]+) to bfloat',
                            r'fadd float (%[\w.]+), 0xC160001000000000',
                            r'bitcast i32 (%[\w.]+) to float',
                            r'or disjoint i32 (%[\w.]+), 1258291200',
                            r'zext i8 (%[\w.]+) to i32',r'xor i8 (%[\w.]+), -128'):
                m=re.fullmatch(pattern,new.defs[name]);assert m,('Unknown V bias SSA',name,new.defs[name]);name=m[1]
            m=re.fullmatch(r'load i8, ptr addrspace\(3\) (%[\w.]+), align [12]',new.defs[name]);assert m,new.defs[name]
            read_ids.append(name);read_lines.append(next(i for i,line in enumerate(new.lines,1) if re.match(r'\s*'+re.escape(name)+r' = ',line)))
            return memories[1][new.integer(m[1],lane)]
        def vector(ir,name,lane,which):
            body=ir.defs[name]
            if m:=re.fullmatch(r'insertelement <4 x bfloat> (poison|%[\w.]+), bfloat (%[\w.]+), i64 ([0-3])',body):
                result=[None]*4 if m[1]=='poison' else vector(ir,m[1],lane,which)
                result[int(m[3])]=scalar(m[2],lane);return result
            if m:=re.fullmatch(r'load <2 x bfloat>, ptr addrspace\(3\) (%[\w.]+), align 4',body):
                assert which==0;address=ir.integer(m[1],lane);return [memories[0][address],memories[0][address+2]]
            m=re.fullmatch(r'shufflevector <2 x bfloat> (%[\w.]+), <2 x bfloat> (%[\w.]+), <4 x i32> <i32 0, i32 1, i32 2, i32 3>',body)
            assert m,('Unknown PV fragment',name,body);return vector(ir,m[1],lane,which)+vector(ir,m[2],lane,which)
        consumed=[]
        for lane in range(64):
            for index in range(path*16+8,path*16+16):
                a,b=vector(old,om[index],lane,0),vector(new,nm[index],lane,1)
                assert a==b and None not in b, 'PV coordinate mapping differs'
                consumed.extend(b)
        assert len(consumed)==len(set(consumed))==2048
        assert len(set(read_ids))==32
        pre,post=barriers[1+2*path:3+2*path]
        stage_stores=new_stores[path*32:path*32+32]
        assert pre<min(x[0] for x in stage_stores)<=max(x[0] for x in stage_stores)<post<min(read_lines)
        evidence.append(dict(path=path,unique_bytes=2048,range=[4096,6143],pv_coordinate_matches=2048,
            byte_reads=32,pre_store_barrier=pre,post_store_barrier=post))
    # All BF16 Q/K shared stores stay in their separate lower bank.
    for line in new.lines:
        m=re.match(r'\s*store <([48]) x bfloat> %\w+, ptr addrspace\(3\) (%[\w.]+)',line)
        if m:
            for lane in range(64):assert 0<=new.integer(m[2],lane) and new.integer(m[2],lane)+2*int(m[1])<=4096
    return dict(paths=evidence,barrier_lines=barriers,shared_banks_disjoint=True)


def ttgir_chain(text):
    assert '#shared2 = #ttg.swizzled_shared<{vec = 64, perPhase = 1, maxPhase = 1, order = [1, 0], useTcu = true}>' in text
    definitions={m[1]:m[2] for line in text.splitlines() if (m:=re.match(r'\s*(%[\w.]+) = (.*)',line))}
    def use(operation,value,needle):
        hits=[name for name,body in definitions.items() if body.startswith(operation+' '+value+' ') and needle in body]
        assert len(hits)==1,(operation,value);return hits[0]
    loads=[name for name,body in definitions.items() if body.startswith('tt.load ') and 'tensor<16x128x!tt.ptr<i8>' in body]
    assert len(loads)==2
    assert 'dense<128>' in definitions['%cst'] and 'dense<1258291200>' in definitions['%cst_0']
    assert 'dense<0x4B000080>' in definitions['%cst_1']
    assert 'dense<0>' in definitions['%cst_8'] and 'xi8' in definitions['%cst_8']
    paths=[]
    for index,load in enumerate(loads):
        if index==1:assert ', %V_141, %cst_8 ' in definitions[load], 'Tail other must be zero'
        alloc=use('ttg.local_alloc',load,'memdesc<16x128xi8, #shared2')
        local=use('ttg.local_load',alloc,'tensor<16x128xi8, #ttg.dot_op')
        cast=use('tt.bitcast',local,'tensor<16x128xi8')
        wide=use('arith.extui',cast,'to tensor<16x128xi32')
        xor=use('arith.xori',wide+',','%cst ')
        bits=use('arith.ori',xor+',','%cst_0 ')
        floating=use('tt.bitcast',bits,'tensor<16x128xf32')
        bias=use('arith.subf',floating+',','%cst_1 ')
        bf16=use('arith.truncf',bias,'to tensor<16x128xbf16')
        pv=[name for name,body in definitions.items() if body.startswith('tt.dot ') and ', '+bf16+',' in body]
        assert len(pv)==1
        paths.append(dict(raw_load=load,byte_alloc=alloc,byte_load=local,bitcast=floating,bias=bias,bf16=bf16,pv=pv[0]))
    return paths


def audit(proof_raw,reader,source_root=ROOT):
    assert sha(proof_raw)==PROOF_SHA,'Only the exact owner622613 proof is supported'
    proof=json.loads(proof_raw);ns,factory=frozen_namespace(source_root)
    assert proof['source_commit']==SOURCE and proof['completed'] is False
    assert proof['errors']==["AssertionError('Unknown/unchanged decode mechanism: stop before further cases or device execution')"]
    assert proof['compile_only'] is True and proof['kernel_launches']==0 and proof['mechanism_passed'] is False
    assert proof['requested_cases']==['steady4k'] and len(proof['cases'])==1
    assert proof['pins']==ns['pinned_sources']()
    generated,source_proof=factory['make_vdecode']();assert source_proof==proof['source_proof']
    for variant,raw in [('old',factory['FIXTURE'].read_bytes()),('A',generated)]:
        entry=proof['sources'][variant]
        assert Path(entry['path'])==OLD_OUT/'sources'/('frozen_v8.py' if variant=='old' else 'bias_A.py')
        assert reader(entry['path'])==raw and sha(raw)==entry['sha256']
    row=copy.deepcopy(proof['cases'][0]);assert row['case']=='steady4k' and row['segments']==2
    plan,_=ns['descriptor_functions']();assert row['descriptors']==plan(64,4096,2)
    assert row['separate_cache_allocations'] is True and set(row['kernels'])==ns['ROLES']
    original_row=copy.deepcopy(row);actual={};file_count=0
    # Inspect the unchanged v1 with temporary path rebinding, never changing its JSON.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='v8-byte-stage-actual-ir-') as folder:
        for role,kernel in row['kernels'].items():
            ns['known_resources'](kernel);assert set(kernel['files'])==KINDS
            actual[role]={}
            for kind,item in kernel['files'].items():
                assert Path(item['path'])==OLD_OUT/'steady4k'/role/('actual.'+kind)
                raw=reader(item['path']);assert len(raw)==item['bytes'] and sha(raw)==item['sha256']
                file_count+=1
                path=Path(folder)/(role+'.'+kind);path.write_bytes(raw);item['path']=str(path)
                if kind in ('ttgir','llir','asm'):actual[role][kind]=raw.decode()
        replay=ns['inspect'](row)
    assert file_count==60 and replay==original_row['mechanism']
    assert replay['reasons']==REJECTIONS and replay['mechanism_passed'] is False
    old,new=actual['gqa_old'],actual['gqa_A']
    graph=ttgir_chain(new['ttgir']);mapping=byte_stage_ssa(old['llir'],new['llir'])
    ops=replay['evidence']['A_ops']
    for name in ('shared','shared1'):
        pattern=r'^#'+name+r' = (.*)$'
        assert re.search(pattern,old['ttgir'],re.M)[1]==re.search(pattern,new['ttgir'],re.M)[1], 'Q/K shared layout changed'
    for name in ('ml_slb_store_b32x2','ml_slb_store_b32x4','ml_slb_shuffle_rtn_b32'):
        assert ops[name]==replay['evidence']['old_ops'][name], 'Unreviewed Q/K/P shared operation change'
    assert {name:ops.get(name,0) for name in ('ml_slb_store_b8','ml_slb_load_b8','ml_xor_b8','ml_cvt_bf16_f32',
            'ml_slb_store_b16','ml_cvt_i16_i8','ml_cvt_bf16_i16','sl_barrier')}==dict(
        ml_slb_store_b8=64,ml_slb_load_b8=64,ml_xor_b8=64,ml_cvt_bf16_f32=68,
        ml_slb_store_b16=0,ml_cvt_i16_i8=0,ml_cvt_bf16_i16=0,sl_barrier=5)
    assert len(re.findall(r'\bml_add_i32[^\n]*0x4b000000',new['asm']))==64
    assert len(re.findall(r'\bml_add_f32[^\n]*0xcb000080',new['asm']))==64
    for byte in range(256):
        assert ((0x4b000000|(byte^128))==(0x4b000000+(byte^128)))
        assert factory['fp32_bias_byte'](byte)==(byte if byte<128 else byte-256)
    assert row['kernels']['gqa_main']['hash']==ns['MAIN_HASHES']['steady4k']
    assert [(row['kernels'][x]['n_regs'],row['kernels'][x]['metadata']['shared']) for x in ('gqa_old','gqa_A')]==[(143,4096),(124,6144)]
    return proof,dict(contract=CONTRACT,diagnostic_mechanism_passed=True,v1_mechanism_passed=False,
        original_proof_sha256=PROOF_SHA,original_completed=False,original_errors=proof['errors'],
        original_source_commit=SOURCE,allowed_cases=['steady4k'],compiled_files_verified=file_count,
        ttgir_chain=graph,llvm_byte_stage=mapping,resources=replay['evidence']['resources'],
        numerical_equivalence_tested=False,precision_gate=False,official_score=False)


def archive_audit():
    files,_=archived_files()
    reader=lambda path:files['out/'+str(Path(path).relative_to(OLD_OUT))]
    return audit(files['out/compile-proof.json'],reader)


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def live_roots():
    assert ROOT.resolve()!=OLD_ROOT.resolve(), 'Validator and original source roots must be distinct'
    assert git(OLD_ROOT,'rev-parse','HEAD')==SOURCE, 'Old source root must remain a698'
    assert not git(OLD_ROOT,'status','--porcelain') and not git(ROOT,'status','--porcelain'), 'Dirty source root'
    source_pins(OLD_ROOT)
    version=git(ROOT,'rev-parse','HEAD')
    validator_files={name:sha((ROOT/'tools'/name).read_bytes()) for name in
        ('biv150_v8_bias_byte_stage.py','bench_biv150_v8_bias_byte_stage.py')}
    for name,wanted in validator_files.items():
        committed=subprocess.check_output(['git','-C',str(ROOT),'show',version+':tools/'+name])
        assert sha(committed)==wanted, 'Validator bytes differ from recorded commit'
    return dict(validator_root=str(ROOT.resolve()),validator_commit=version,validator_files=validator_files,
        original_source_root=str(OLD_ROOT),original_source_commit=SOURCE,
        original_output_root=str(OLD_OUT),original_source_pins=source_pins(OLD_ROOT))


def make_sidecar(path):
    binding=live_roots()
    proof,report=audit((OLD_OUT/'compile-proof.json').read_bytes(),lambda p:Path(p).read_bytes(),OLD_ROOT)
    sidecar=dict(binding=binding,report=report,original_proof_path=str(OLD_OUT/'compile-proof.json'),
        original_proof_sha256=PROOF_SHA,allowed_cases=['steady4k'],precision_gate=False,official_score=False)
    assert not path.exists();path.write_text(json.dumps(sidecar,indent=2)+'\n');return sidecar


def sidecar_header(sidecar,binding):
    assert sidecar['binding']==binding, 'Validator/original-root binding changed'
    assert sidecar['allowed_cases']==['steady4k'] and sidecar['original_proof_sha256']==PROOF_SHA
    assert sidecar['original_proof_path']==str(OLD_OUT/'compile-proof.json')
    assert sidecar['precision_gate'] is False and sidecar['official_score'] is False


def validate_sidecar(path):
    sidecar=json.loads(path.read_text());sidecar_header(sidecar,live_roots())
    proof,report=audit((OLD_OUT/'compile-proof.json').read_bytes(),lambda p:Path(p).read_bytes(),OLD_ROOT)
    assert report==sidecar['report']
    return proof,sidecar
