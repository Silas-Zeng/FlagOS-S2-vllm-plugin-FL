"""Conservative full-decoder compile gates. No primitive success is a decoder pass."""
import ast
from collections import Counter
import re


def audit(artifacts,source):
    tree=ast.parse(source)
    dots=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.dot']
    qk=[n for n in dots if any(k.arg=='out_dtype' and ast.unparse(k.value)=='tl.int32' for k in n.keywords)]
    pv=[n for n in dots if ast.unparse(n)=='tl.dot(probability.to(V.dtype), V)']
    assert len(qk)==len(pv)==2
    tt,ll,asm=[artifacts.get(k,'') for k in ('ttgir','llir','asm')]
    dot_lines=[line for line in tt.splitlines() if '= tt.dot ' in line]
    i8_shapes=[line for line in dot_lines if re.search(r'tensor<8x128xi8,.*\* tensor<128x16xi8,.*-> tensor<8x16xi32,',line)]
    pv_shapes=[line for line in dot_lines if re.search(r'tensor<8x16xbf16,.*\* tensor<16x128xbf16,.*-> tensor<8x128xf32,',line)]
    locations={int(k):int(v) for k,v in re.findall(r'!(\d+) = !DILocation\(line: (\d+)',ll)}
    def site(line):
        match=re.search(r'!dbg !(\d+)',line)
        return locations.get(int(match[1])) if match else None
    qk_sites={n.lineno for n in qk};pv_sites={line for n in pv for line in range(n.lineno,n.end_lineno+1)}
    native_i8=[line for line in ll.splitlines() if 'call <4 x i32> @llvm.bi.matrix.mad.i32x4.i8x8(' in line]
    groups={s:[line for line in native_i8 if site(line)==s] for s in qk_sites}
    c_chain=True
    for group in groups.values():
        previous='zeroinitializer'
        c_chain=c_chain and len(group)==4
        for line in group:
            lhs=re.search(r'(%[\w.$-]+) = ',line);c=re.search(r', <4 x i32> (zeroinitializer|%[\w.$-]+)\)',line)
            c_chain=c_chain and bool(lhs and c and c[1]==previous)
            previous=lhs[1] if lhs else None
    native_pv=[line for line in ll.splitlines() if 'call <4 x float> @llvm.bi.matrix.mad.f32x4.bf16x4(' in line]
    defs=dict(re.findall(r'^\s*(%[\w.$-]+) = (.*)$',ll,re.M))
    users={}
    for value,expression in defs.items():
        for operand in re.findall(r'%[\w.$-]+',expression): users.setdefault(operand,[]).append((value,expression))
    all_c=[];pv_c_chain=True;pv_post_add_absent=True
    for line in native_pv:
        result=re.search(r'(%[\w.$-]+) = ',line)
        extracts=users.get(result[1],[]) if result else []
        pv_post_add_absent=pv_post_add_absent and len(extracts)==4
        for value,expression in extracts:
            pv_post_add_absent=pv_post_add_absent and expression.startswith('extractelement <4 x float> ') and bool(users.get(value)) and all(e.startswith('phi float ') for _,e in users[value])
        argument=re.search(r', <4 x float> (%[\w.$-]+)\)',line)
        if argument is None: pv_c_chain=False;continue
        value=argument[1];components={}
        for _ in range(4):
            packed=re.match(r'insertelement <4 x float> (poison|%[\w.$-]+), float (%[\w.$-]+), i\d+ ([0-3])',defs.get(value,''))
            if packed is None: pv_c_chain=False;break
            components[int(packed[3])]=packed[2];value=packed[1]
        pv_c_chain=pv_c_chain and value=='poison' and set(components)==set(range(4))
        for component in components.values():
            pv_c_chain=pv_c_chain and defs.get(component,'').startswith('fmul float ')
            all_c.append(component)
    casts=[dict(llir_line=i,source_line=site(line)) for i,line in enumerate(ll.splitlines(),1)
           if re.search(r'sitofp i32 .* to float',line) and site(line) in
           {n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n)=='score_integer.to(tl.float32)'}]
    opcodes=Counter(re.findall(r'^\s*(ml_\w+)\s',asm,re.M))
    flags=dict(required_artifacts=all(isinstance(artifacts.get(k),str) and artifacts[k] for k in ('ttgir','llir','asm')),
               full_tail_logical_i8_shapes=len(dot_lines)==4 and len(i8_shapes)==2,
               original_bf16_pv_shapes=len(pv_shapes)==2,native_k32_encoding='instrShape = [16, 16, 32]' in tt,
               eight_native_i8_calls=len(native_i8)==8,complete_i8_c_chains=bool(c_chain),
               original_pv_calls=len(native_pv)==16 and all(site(line) in pv_sites for line in native_pv),
               all_64_scaled_native_pv_c_scalars=bool(pv_c_chain) and len(all_c)==len(set(all_c))==64,
               pv_post_add_absent=bool(pv_post_add_absent),
               eight_i8_machine_mmas=opcodes['ml_matrix_mad_i32_i8_m16n16k32']==8,
               sixteen_bf16_pv_machine_mmas=opcodes['ml_matrix_mad_f32_bf16_m16n16k16']==16,
               score_casts_at_exact_ast_sites=len(casts)>0)
    return dict(passed=all(flags.values()),flags=flags,score_casts_per_lane=casts,
                score_cast_elements_per_lane_recorded=len(casts),isa_opcodes=dict(opcodes),
                primitive_geometry_not_reused=True,native_component_independence_assumed=False,
                precise_div_native_support_inferred=False,device_numeric_verified=False)
