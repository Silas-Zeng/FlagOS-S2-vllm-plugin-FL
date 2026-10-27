"""One V-issue ordering experiment; stdlib inverse/byte-layout contract only."""
import ast
import hashlib
from pathlib import Path
import subprocess
import biv150_native_k8_source_factory as parent

ROOT=Path(__file__).resolve().parents[1]
PARENT='3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
MAIN='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
NAMESPACE='cjj_native_k8_vfirst_probe'
RENAMES=(('corex_native_k8_gqa_producer_draft','cjj_native_k8_vfirst_gqa_producer'),
         ('launch_corex_native_k8_gqa_s2_draft','launch_cjj_native_k8_vfirst_s2'),
         ('launch_corex_native_k8_gqa_s4_draft','launch_cjj_native_k8_vfirst_s4'),
         ('cjj_launch_native_k8_codec','cjj_launch_native_k8_vfirst_codec'),
         ('cjj_native_k8_codec_probe','cjj_native_k8_vfirst_codec_probe'),
         ('cjj_native_k8_qk_probe',NAMESPACE))
OLD=''' issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();
 static_assert(!OverlapV,"Only original overlap=false is admitted");
 issue_v(dst,descriptor(value,page,head));
 __ivcorex_sl_waitcnt(8ULL);__syncthreads();'''
NEW=''' issue_v(dst,descriptor(value,page,head));
 issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();
 static_assert(!OverlapV,"Only original overlap=false is admitted");
 __ivcorex_sl_waitcnt(8ULL);__syncthreads();'''
FILES={'qk.cu':'biv150_native_k8_qk.cu','qk.cpp':'biv150_native_k8_qk.cpp','codec.cu':'biv150_native_k8_codec.cu'}

def sha(b):return hashlib.sha256(b).hexdigest()
def blob(path):return subprocess.check_output(['git','-C',str(ROOT),'show',PARENT+':'+path])
def path(role):return ROOT/'tools/kernels'/('biv150_native_k8_vfirst_'+role)
def generate(role):
    text=blob('tools/kernels/'+FILES[role]).decode()
    if role=='qk.cu':
        assert text.count(OLD)==1;text=text.replace(OLD,NEW)
    for old,new in RENAMES:text=text.replace(old,new)
    return text
def inverse(text,role):
    for old,new in reversed(RENAMES):text=text.replace(new,old)
    if role=='qk.cu':
        assert text.count(NEW)==1;text=text.replace(NEW,OLD)
    assert text.encode()==blob('tools/kernels/'+FILES[role]),'nonordering edit in '+role
    return text
def verify():
    base=parent.verify();proof={}
    for role,original in FILES.items():
        text=path(role).read_text();assert text==generate(role);restored=inverse(text,role)
        if role=='qk.cu':assert parent.inverse(restored).encode()==parent.SOURCE.read_bytes()
        proof[role]=dict(parent_sha256=sha(restored.encode()),variant_sha256=sha(text.encode()),inverse_byte_exact=True)
    # Enumerate all 512 aligned word stores, not a byte-load rewrite.
    k=[]
    for slot in range(16):
        for dim in range(0,128,4):
            panel,within=divmod(dim,64);quartet=within//4;bank,subslot=divmod(slot,4)
            offset=panel*1024+bank*256+(quartet%4)*64+((quartet//4)^bank)*16+subslot*4
            k.extend(range(offset,offset+4))
    assert len(k)==len(set(k))==2048 and set(k)==set(range(2048))
    v=set(range(4096,8192));assert not set(k)&v
    # Source operands/calls after wait, including tail clear and all math, are literal.
    return dict(parent_commit=PARENT,main=MAIN,parent_inverse=base,files=proof,
        K_shared_bytes=[0,2048],V_shared_bytes=[4096,8192],shared_bytes=8192,
        K_unique_aligned_uint32_words=512,source_issue_order=['V_SME','K_uint32_global_shared','barrier','V_wait8','barrier'],
        OverlapV=False,wait_tail_codec_QK_PV_math_unchanged=True,namespace=NAMESPACE,
        actual_schedule_verified=False,numeric_verified=False,performance_verified=False,
        service_installed=False,capacity_reclaimed=False,precision_gate=False)

def host_source(kind):
    """Narrow exact-inverse adapters of unchanged parent test/build bodies."""
    file={'compile':'compile_biv150_native_k8_qk.py','numeric':'probe_biv150_native_k8_consumer_v2.py','fullstep':'probe_biv150_native_k8_fullstep.py'}[kind]
    original=blob('tools/'+file).decode();tree=ast.parse(original)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==('compile_only' if kind=='compile' else 'run'))
    text=ast.get_source_segment(original,fn);changes=[]
    def edit(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(kind,old);text=text.replace(old,new);changes.append((old,new,count))
    if kind=='compile':edit("name='cjj_native_k8_qk_probe'",'name='+repr(NAMESPACE))
    elif kind=='numeric':
        edit('f.sha(args.library.read_bytes())==f.LIBRARY_SHA','f.sha(args.library.read_bytes())==args.variant_sha256',2)
        edit('torch.ops.cjj_native_k8_qk_probe.produce','torch.ops.'+NAMESPACE+'.produce')
        edit('result=dict(f.plan(),codec_precondition=receipt,','result=dict(variant_plan(args),codec_precondition=receipt,')
    else:
        edit('f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA','f.sha(a.k8_library.read_bytes())==a.variant_sha256')
        edit('torch.ops.cjj_native_k8_qk_probe.produce','torch.ops.'+NAMESPACE+'.produce')
        edit('result=dict(f.plan(),actual_consumer=pre,','result=dict(variant_plan(a),actual_consumer=pre,')
    restored=text
    for old,new,count in reversed(changes):assert restored.count(new)==count;restored=restored.replace(new,old)
    assert restored==ast.get_source_segment(original,fn)
    return text,dict(parent_file=file,parent_run_sha256=sha(restored.encode()),adapter_run_sha256=sha(text.encode()),
                     substitutions=[dict(old=x,new=y,count=count) for x,y,count in changes],inverse_byte_exact=True,
                     math_fixture_guards_Graph_side_timing_body_unchanged=True)
