"""Finite adaptation of the reviewed efd/522 N1 screen. No SDK imports here."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE='efd012eb7f31a70ad0a37bc5a05b9aa9fc8f0efe'
PARENT='e8b62f0c91d3ecca54749f55c07174aa9a4c18ef'
PINS={
 'biv150_konly_q32_s2_numeric_source.py':'942b4004b24071566d424c6c608c4252f2bdae43133e94184bbb5811e323f109',
 'biv150_konly_q32_s2_numeric_contract.py':'c1cc3777e8c10a4cbd058cf97db35e43c7934f0597df9ab9cb6c0119ec8f5ce7',
}

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(data):return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()

def original(name):
    data=subprocess.check_output(['git','-C',str(ROOT),'show',TEMPLATE+':tools/'+name])
    need(sha(data)==PINS[name],'immutable efd screen '+name)
    return data.decode()

def change(text,edits):
    old=text
    for before,after in edits:
        need(text.count(before)==1 and after not in text,'unique numeric seam '+before[:95])
        text=text.replace(before,after,1)
    inverse=text
    for before,after in reversed(edits):
        need(inverse.count(after)==1,'unique numeric inverse')
        inverse=inverse.replace(after,before,1)
    need(inverse==old,'complete numeric adapter inverse')
    return text

def roles(text):
    # Private role names only, including literals inside the source adapter's
    # seam strings. No numeric expression or memory operation is rewritten.
    need("'w16'" not in text and "'w8'" not in text,'fresh private role labels')
    new=text.replace("'s1'","'w16'").replace("'s2'","'w8'")
    need(new.replace("'w16'","'s1'").replace("'w8'","'s2'")==text,'role-label inverse')
    return new

def source_text():
    text=roles(original('biv150_konly_q32_s2_numeric_source.py'))
    return change(text,[
      ("PARENT='2e0929793d9e744a964c719e27d253927bcb52f3'","PARENT='"+PARENT+"'"),
      ('dict(num_warps=16,num_stages=f.STAGES[role],ir_override=str(override))',
       'dict(num_warps=f.WARPS[role],num_stages=1,ir_override=str(override))'),
      ("output_s1=guarded['w16'],output_s2=guarded['w8']","output_w16=guarded['w16'],output_w8=guarded['w8']"),
      ('run_edits()+[(timing,','run_edits()+run_extra_edits()+[(timing,'),
      ("actual_roles='main899 N1 hoisted S1 and preheader N1 hoisted S2'",
       "actual_roles='accepted main899 N1 w16/S1 and exact-preheader N1 w8/S1'"),
      ("legacy_policy_bitwise_key_means='actual N1 hoisted S1 versus actual N1 hoisted S2'",
       "cross_layout_bitwise_required=False,cross_layout_full_output_bitwise=byte_cpu(outputs['w16'])==byte_cpu(outputs['w8'])"),
      ("    # Numeric/codec, archive, Graph-before-fresh and nondefault-stream bodies",
       "    # Numeric/codec, archive, Graph-before-fresh and nondefault-stream bodies; layout equality is diagnostic."),
      ("    ]\n\n\ndef run_source():",
       "      (\"                    hoisted_unhoisted_bitwise=byte_cpu(outputs['w16'])==byte_cpu(outputs['w8']),\\n\",\n       \"                    # Different warp layouts: bitwise is recorded separately, not a hard gate.\\n\"),\n    ]\n\n\ndef run_source():"),
    ])

def contract_text():
    text=roles(original('biv150_konly_q32_s2_numeric_contract.py'))
    return change(text,[
      ("STAGES={'w16':1,'w8':2}","STAGES={'w16':1,'w8':1}\nWARPS={'w16':16,'w8':8}"),
      ("'w8':dict(n_regs=124,n_spills=0,shared=41216,ELF_sgpr_spills=93)","'w8':dict(n_regs=237,n_spills=0,shared=33792,ELF_sgpr_spills=478)"),
      ('2d8938635879f904f3d844b27cb8919ec3d6d4e71cf9a5b990362059065d90e1','773d51c9821807457d71b7eb2a1a3701c1e3fc78410ce8f0d7b7ca39a338c51c'),
      ('dab726c48cabe17bd795e2b35f094c5fefcb40db0e802f655bff51470b8715ce','03d373323bef024b3dec81d76de9c19913d9a03d0c05f471e01178523110d1a8'),
      ('9f45fd740d0eca78f00e80f43b890a1a624301f12aee84664c3707a41e9326db','dfdc2c2af60b93153170ebbcc9477e8bf200fe243a03069f0e23aff4f7467aec'),
      ("'w8':('a7583d8afb11bf0614adbeaeeb05349716fc3f7793f827b5932da1d5f6c0316c',27,2227237,219924,364718401,s.PARENT)",
       "'w8':('4b6af02abb6694ba736ce53e12e03b93f3a1aac26173b62c4eee6430347ba1f1',36,4325044,289827,366265885,s.PARENT)"),
      ('c35d0001e07b2b0e59879eee790d32e9745aeecff98d5dca0823c45991eeac32','ba0b4d0b7dda3db06f25bf6a9baeb39f3cbb09aa756fc269f7c82dfdd7bdd26d'),
      ('80f1b3388822eb50c7db1d953430d726d15b0269dd927a02638c5c2328cc58a0','5409b37ec09961b1459fbab689fd7407ee610341250eeb48857b3761e8a8c18d'),
      ("schema='cjj-q32-S1-S2-preheader-numeric-v1'","schema='cjj-q32-w16-w8-preheader-numeric-v1'"),
      ('roles=list(ROLES),stages=STAGES,hashes=HASHES,','roles=list(ROLES),stages=STAGES,warps=WARPS,cross_layout_bitwise_required=False,full_output_pointwise_hard_gate=True,full_output_reference="accepted w16/S1, not full random FP64",full_output_elements=4194304,hashes=HASHES,'),
      ("ABBA_future='same complete S1/S2 reader and unchanged-writer+reader; no raw-S2 or N2 control'",
       "ABBA_future='not permitted by this frontend; any later timing must compare accepted hoisted w16/S1'"),
      ('[16,STAGES[role],64] and (m[\'num_warps\'],m[\'num_stages\'],m[\'warp_size\'])==(16,STAGES[role],64)',
       '[WARPS[role],1,64] and (m[\'num_warps\'],m[\'num_stages\'],m[\'warp_size\'])==(WARPS[role],1,64)'),
      ("{'hash','num_stages','shared','ir_override'}","{'hash','num_warps','ir_override'}"),
      ("review['source']==s.PARENT and review['new_compiled']['cubin_sha256']==CUBINS['w8']",
       "review['source_commit']==s.PARENT and review['new_final_cubin_sha256']==CUBINS['w8']"),
      ("review['Q_operand_hoist_TTGIR_LLVM_compiler_ASM_static_verified'] is True and review['S2_bank_publish_reuse_acquire_static_verified'] is True",
       "review['actual_TTGIR_matches_fixed_override_modulo_SSA'] is True and review['static_Q_publish_read_reuse_acquire_verified_in_LLVM_and_compiler_ASM'] is True"),
      ("summary['records']['preheader-q32-stage2'];r=item['preheader-stage2']",
       "summary['records']['preheader-q32-w8-stage1'];r=item['preheader-w8-stage1']"),
      ("folder=out/'preheader-q32-stage2/preheader-stage2';source=remote/'sources/candidate_q32_s2.py';module='cjj_private_q32_s2_scout'",
       "folder=out/'preheader-q32-w8-stage1/preheader-w8-stage1';source=Path(summary['reviewed_w8']['source']);module='cjj_private_q32_w8_scout'"),
      ("out/'q-local-preheader.ttgir'","out/'q-local-preheader-w8.ttgir'"),
      ("schema='cjj-124-N1-S1-S2-numeric-root-admission-v1'","schema='cjj-126-N1-w16-w8-numeric-root-admission-v1'"),
      ('hashes=HASHES,cubins=CUBINS,query_rows=2048,num_seqs=1,stages=STAGES,','hashes=HASHES,cubins=CUBINS,query_rows=2048,num_seqs=1,stages=STAGES,warps=WARPS,'),
      ("    ns={'h':types.SimpleNamespace(need=need)};exec(compile(ast.Module(body=[fn],type_ignores=[]),'unchanged-e4dd-policy','exec'),ns)",
       "    policy_source=layout_policy(fn)\n    ns={'h':types.SimpleNamespace(need=need)};exec(compile(policy_source,'e4dd-policy-with-full-output-reference','exec'),ns)"),
      ('import biv150_konly_q32_s2_preheader_source as candidate','import biv150_konly_q32_w8_preheader_source as candidate'),
      ("after=s.replace(before,[(\"==(16,1,64),'loaded geometry'\",\"==(16,STAGES[role],64),'loaded geometry'\")])",
       "after=s.replace(before,[(\"==(16,1,64),'loaded geometry'\",\"==(WARPS[role],1,64),'loaded geometry'\"),('threads=1024','threads=kernel.metadata.num_warps*kernel.metadata.warp_size')])"),
      ('RESOURCES=RESOURCES,STAGES=STAGES,execution_snapshot=','RESOURCES=RESOURCES,WARPS=WARPS,execution_snapshot='),
      ("schema='cjj-N1-S1-S2-prepared-actual-cache-v1'","schema='cjj-N1-w16-w8-prepared-actual-cache-v1'"),
    ])

def layout_policy(fn):
    before=ast.dump(fn);required=fn.body[0].value
    need(isinstance(required,ast.Set) and {n.value for n in required.elts}==
         {'finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference'},'exact original policy set')
    index=next(i for i,n in enumerate(required.elts)if n.value=='hoisted_unhoisted_bitwise')
    item=required.elts[index];required.elts[index]=ast.Constant(value='full_output_reference');result=ast.unparse(fn)
    required.elts[index]=item;need(ast.dump(fn)==before,'policy single-key inverse')
    return result

def full_output_metrics(chunks,np,expected_size):
    """Original pointwise rule; bounded float64 chunks, w16 reference values."""
    need(type(expected_size) is int and expected_size>0,'full output expected geometry')
    count=violations=0;error2=reference2=maximum=0.;finite=True
    for actual,reference in chunks:
        a=np.asarray(actual,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
        need(a.ndim==r.ndim==1 and a.size==r.size and 0<a.size<=262144,'equal nonempty bounded flat output chunks')
        count+=int(a.size)
        good=bool(np.isfinite(a).all() and np.isfinite(r).all());finite=finite and good
        if not good:continue
        error=a-r;absolute=np.abs(error)
        violations+=int(np.count_nonzero(absolute>.005+.005*np.abs(r)))
        maximum=max(maximum,float(absolute.max()))
        error2+=float(np.sum(error*error,dtype=np.float64));reference2+=float(np.sum(r*r,dtype=np.float64))
    need(count==expected_size,'complete output comparison geometry')
    if not finite:return dict(elements=count,finite=False,pointwise_passed=False,pointwise_violations=None,relative_rmse=None,rms=None,max_abs=None)
    rms=math.sqrt(error2/count);rrms=math.sqrt(reference2/count)
    return dict(elements=count,finite=True,pointwise_passed=violations==0,pointwise_violations=violations,
        relative_rmse=rms/rrms if rrms else (0. if maximum==0 else math.inf),rms=rms,max_abs=maximum)

def run_extra_edits():
    header='                hard=dict(finite=True,codec=True,output_canaries=True,read_only_inputs=True,actual_IR=True,\n'
    before=header
    after="""                c.need(all(tuple(outputs[r].shape)==(2048,16,128) for r in f.ROLES),'full output comparison shape')
                flat={r:outputs[r].reshape(-1) for r in f.ROLES}
                full=f.full_output_metrics(((flat['w8'][i:i+262144].double().numpy(),flat['w16'][i:i+262144].double().numpy()) for i in range(0,4194304,262144)),np,4194304)
                record['full_output_comparison']=dict(full,actual_role='w8',reference_role='accepted w16/S1',reference_is_full_random_FP64=False,arithmetic='CPU float64 bounded chunks',pointwise_atol=.005,pointwise_rtol=.005)
"""+header+"                    full_output_reference=full['finite'] and full['pointwise_passed'],\n"
    return [(before,after)]

_FRAMEWORK=None
def framework():
    global _FRAMEWORK
    if _FRAMEWORK is not None:return _FRAMEWORK
    modules=[]
    for name,text in (('biv150_konly_q32_s2_numeric_source',source_text()),('biv150_konly_q32_s2_numeric_contract',contract_text())):
        need(name not in sys.modules,'private numeric namespace already occupied')
        m=types.ModuleType(name);m.__file__=str(ROOT/'tools'/(name+'.py'))
        m.layout_policy=layout_policy;m.run_extra_edits=run_extra_edits;m.full_output_metrics=full_output_metrics;sys.modules[name]=m
        exec(compile(text,'<efd-w16-w8/'+name+'>','exec'),vars(m));modules.append(m)
    _FRAMEWORK=types.SimpleNamespace(source=modules[0],contract=modules[1])
    return _FRAMEWORK
