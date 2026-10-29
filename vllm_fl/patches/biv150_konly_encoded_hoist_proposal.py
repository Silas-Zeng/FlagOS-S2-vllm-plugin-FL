"""Default-off CPU proposal for ONE structural Q32 I8 operand move.

No framework import/compiler hook/device work. A proposal is not a native-code,
resource, numerical, Graph or service certificate. Unknown IR must stay on the
correct private unhoisted encoded reader, never a stock BF16 cache reader.
"""
import hashlib
import re
from copy import deepcopy

SCHEMA='cjj-konly-service-q32-hoist-v1'
MULTI2_SCHEMA='cjj-konly-private-multi2-hoist-v1'
FORMAT='cjj-konly-rn32-rne-v2'
SSA=r'%[-.A-Za-z0-9_]+(?:#[0-9]+)?'
PINS=dict(sdk='54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f',
          helpers='59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1',
          jit='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775',
          codec='d73af8368193e1cab4ba9e74132d056d5b61d0c0950d600d494d75c3151f4cf7')


class Unsupported(ValueError):pass


def need(ok,message):
    if not ok:raise Unsupported(message)


def sha(text):return hashlib.sha256(text.encode() if isinstance(text,str) else text).hexdigest()


def clean(line):
    # SSA/brace strings in source debug locations and quoted op names do not
    # participate in dataflow or region nesting. Keep token columns unchanged.
    return re.sub(r'"(?:\\.|[^"\\])*"',lambda m:' '*len(m[0]),line).split('//',1)[0]


def regions(lines):
    """Brace-balanced region ancestry; inline type/attribute braces close inline.

    The admitted parent is a straight-line structured block. cf/branch/parallel
    regions are rejected; nested while/reduce/if regions remain distinct.
    """
    stack=[];parents=[];after=[];counter=0
    for line in lines:
        text=clean(line);prefix=re.match(r'\s*(}*)',text)[1]
        for _ in prefix:
            need(bool(stack),'region underflow');stack.pop()
        parents.append(tuple(stack));position=text.find(prefix) if prefix else 0
        start=position+len(prefix) if prefix else 0
        for token in re.finditer('[{}]',text[start:]):
            if token[0]=='{':counter+=1;stack.append(counter)
            else:need(bool(stack),'region underflow');stack.pop()
        after.append(tuple(stack))
    need(not stack,'unclosed region')
    return parents,after


def one(lines,predicate,reason):
    found=[i for i,s in enumerate(lines) if predicate(s)]
    need(len(found)==1,reason);return found[0]


def definition(line):
    m=re.match(r'\s*('+SSA+r')\s*=',clean(line));need(m is not None,'single-result SSA definition required');return m[1]


def blocks(lines,parents):
    current={};result=[]
    for i,line in enumerate(lines):
        m=re.match(r'\s*\^([-A-Za-z0-9_]+)',clean(line))
        if m:current[parents[i]]=(m[1],i)
        result.append(current.get(parents[i],('implicit',parents[i])))
    return result


def dominates(lines,parents,block,parent,definition_line,use_line):
    rows=[i for i,p in enumerate(parents) if p==parent]
    nodes=list(dict.fromkeys(block[i] for i in rows));entry=nodes[0]
    labels={b[0]:b for b in nodes if b[0]!='implicit'}
    succ={b:set() for b in nodes}
    for i in rows:
        line=clean(lines[i])
        if not re.search(r'\bcf\.',line):continue
        need(re.search(r'\bcf\.(?:br|cond_br)\b',line) is not None,'unknown parent CFG terminator')
        targets=re.findall(r'\^([-A-Za-z0-9_]+)',line)
        need(targets and all(t in labels for t in targets),'unknown parent CFG target')
        succ[block[i]].update(labels[t] for t in targets)
    pred={b:{p for p in nodes if b in succ[p]} for b in nodes}
    need(all(pred[b] for b in nodes if b!=entry),'unreachable parent block')
    dom={b:({b} if b==entry else set(nodes)) for b in nodes}
    while True:
        new={b:({b} if b==entry else {b}|set.intersection(*(dom[p] for p in pred[b]))) for b in nodes}
        if new==dom:break
        dom=new
    return block[definition_line] in dom[block[use_line]]


def uses(lines,value):
    pattern=re.compile(re.escape(value)+r'(?![-.A-Za-z0-9_#])')
    return [i for i,line in enumerate(lines) if pattern.search(clean(line))]


def grammar(text,function_name,*,hoisted=False):
    lines=text.splitlines(keepends=True);parents,ends=regions(lines);block=blocks(lines,parents)
    need(not re.search(r'\b(?:scf\.(?:parallel|execute_region)|ttg\.(?:async_copy|memdesc_subslice))',text),
         'unknown branch/parallel/async/alias grammar')
    funcs=[s for s in lines if re.search(r'\btt.func\b',s)]
    need(len(funcs)==1 and '@'+function_name+'(' in funcs[0],'unique actual kernel symbol')
    parameters=re.findall('('+SSA+r'): !tt.ptr<([A-Za-z0-9]+)>',funcs[0])
    need([t for _,t in parameters[:5]]==['bf16','bf16','i8','f32','bf16'],
         'private output/Q/K/scale/V pointer parameter order and types')
    need('"ttg.num-warps" = 16 : i32' in text and '"ttg.threads-per-warp" = 64 : i32' in text
         and 'ttg.target = "cuda:71"' in text,'actual target/warp module attributes')
    loop=one(lines,lambda s:re.search(r'= scf.for\b',clean(s)) is not None,'one KV structured loop required')
    parent=parents[loop];body=ends[loop]
    need(len(body)==len(parent)+1 and body[:-1]==parent,'KV loop directly owned body')
    end=next((i for i in range(loop+1,len(lines)) if len(ends[i])==len(parent) and ends[i]==parent),None)
    need(end is not None,'KV loop end')
    qglobal=one(lines,lambda s:' = tt.load ' in s and 'tensor<256x128x!tt.ptr<bf16>' in s,'unique BF16 Q tile load')
    cast=one(lines,lambda s:' = arith.fptosi ' in s and 'to tensor<256x128xi8,' in s,'unique I8 Q result')
    alloc=one(lines,lambda s:' = ttg.local_alloc ' in s and '(tensor<256x128xi8,' in s,'unique initialized QCode allocation')
    match=re.match(r'\s*('+SSA+r') = ttg.local_alloc ('+SSA+r') :',lines[alloc]);need(match is not None,'Q allocation SSA')
    memory,code=match.groups()
    need(definition(lines[cast])==code and qglobal<cast<alloc<loop
         and all(parents[i]==parent and block[i]==block[loop] for i in (qglobal,cast,alloc)),
         'Q read/code/allocation do not dominate KV in same block')
    # Pointer-only dataflow: metadata offsets may themselves load other pointer
    # arguments, but the tt.addptr base chain must terminate at actual Q arg1.
    pointer=re.search(r'= tt.load ('+SSA+r')',lines[qglobal]);need(pointer is not None,'Q global pointer')
    def origin(value,before,seen):
        need(value not in seen,'Q pointer cycle');seen=seen|{value}
        if value in dict(parameters):return value
        matches=[i for i in range(before) if re.match(r'\s*'+re.escape(value)+r' = ',clean(lines[i]))]
        need(matches,'Q pointer SSA definition');i=matches[-1]
        need(parents[i]==parent and dominates(lines,parents,block,parent,i,qglobal),'Q pointer definition dominates load')
        op=re.match(r'\s*'+re.escape(value)+r' = ([A-Za-z0-9_.]+) ('+SSA+r')',clean(lines[i]))
        need(op is not None and op[1] in ('tt.addptr','tt.splat','tt.broadcast','tt.expand_dims','ttg.convert_layout'),
             'unknown Q pointer origin operation')
        return origin(op[2],i,seen)
    need(origin(pointer[1],qglobal,set())==parameters[1][0],'Q tile does not originate from actual query pointer')
    need(not any(re.search(r'\bcf\.',clean(lines[i])) for i in range(qglobal,loop)),
         'branch between Q initializer and KV preheader')
    divs=[i for i,s in enumerate(lines) if ' = tt.precise_divf ' in s]
    need(len(divs)==2 and all(qglobal<i<cast and parents[i]==parent and block[i]==block[loop] for i in divs),'two RN32 codec divisions outside KV')
    # Fresh initialized memdesc + these only two uses excludes stores, aliasing,
    # deallocation, escapes and other reads; backend bank reuse is a later gate.
    load=one(lines,lambda s:re.search(r'= ttg.local_load '+re.escape(memory)+r' :',s) is not None,'one QCode operand read')
    need(uses(lines,memory)==sorted([alloc,load]),'QCode memory aliases/escapes/is modified')
    operand=definition(lines[load])
    layout=re.search(r'!ttg.memdesc<256x128xi8, (#[A-Za-z0-9_]+), #smem> -> tensor<256x128xi8, #ttg.dot_op<\{opIdx = 0, parent = (#[A-Za-z0-9_]+), kWidth = 4, useSme = 0\}>>',lines[load])
    need(layout is not None,'signed I8 Q operand shape/layout')
    need(all(any(s.startswith(name+' = ') for s in lines) for name in layout.groups()),'layout aliases defined')
    if hoisted:need(alloc<load<loop and parents[load]==parent,'Q operand not in same parent preheader')
    else:need(loop<load<end and parents[load]==body,'Q operand must be directly inside KV')
    dots=[i for i,s in enumerate(lines) if ' = tt.dot ' in s]
    need(len(dots)==2 and all(loop<i<end and parents[i]==body for i in dots)
         and block[dots[0]]==block[dots[1]],'two direct QK/PV dots in KV')
    if not hoisted:need(block[dots[0]]==block[load],'Q operand/dot share one loop block')
    qk=one(lines,lambda s:re.search(r'= tt.dot '+re.escape(operand)+r', ',s) is not None,'Q directly feeds QK')
    need(qk==dots[0] and 'tensor<128x64xi8,' in lines[qk] and '-> tensor<256x64xi32,' in lines[qk],'I8/I32 QK geometry')
    c=re.search(r', ('+SSA+r'), inputPrecision',lines[qk]);need(c is not None,'QK C operand')
    zero=one(lines,lambda s:s.lstrip().startswith(c[1]+' = arith.constant dense<0> : tensor<256x64xi32,'),'QK zero I32 C')
    need(zero<loop and parents[zero]==parent and dominates(lines,parents,block,parent,zero,loop),'zero C dominates loop')
    pv=dots[1];need('tensor<256x64xbf16,' in lines[pv] and 'tensor<64x128xbf16,' in lines[pv]
        and '-> tensor<256x128xf32,' in lines[pv],'unchanged BF16 PV geometry')
    need(uses(lines,operand)==sorted([load,qk]),'Q operand has extra uses')
    indent=re.match(r' *',lines[loop])[0];loadindent=re.match(r' *',lines[load])[0]
    need(loadindent==(indent if hoisted else indent+'  '),'unknown structured indentation')
    return dict(q_global=qglobal+1,q_code_cast=cast+1,q_allocation=alloc+1,Q_operand_load=load+1,
        KV_loop=loop+1,KV_end=end+1,QK=qk+1,PV=pv+1,memory=memory,operand=operand,
        actual_Q_argument_pointer_origin=True,same_parent_dominance=True,fresh_initialized_immutable_QCode=True,zero_trip_local_read_safe=True,
        no_Q_memory_alias_or_escape=True,operand_only_direct_QK_use=True,layout_aliases=list(layout.groups()))


def validate_meta(text,meta,*,_multi2=False):
    need(type(_multi2) is bool,'Boolean multi2 scope')
    schema,profile,sequences = (MULTI2_SCHEMA,'multi2',2) if _multi2 else (SCHEMA,'q32',1)
    keys={'schema','enabled','profile','format','target','launch_options','constants','query_rows','grid',
          'kernel_hash','function_name','source_digest','ttgir_sha256','pins'}
    need(type(meta) is dict and set(meta)==keys,'complete exact actual compilation metadata schema')
    need(meta['schema']==schema and meta['profile']==profile and meta['format']==FORMAT,
         'private multi2 format/schema' if _multi2 else 'private Q32 format/schema')
    need(meta['enabled'] is True,'explicit proposal enable flag')
    need(meta['target']==dict(backend='corex',arch=71,warp_size=64),'actual compiler target')
    need(meta['launch_options']==dict(num_warps=16,num_stages=1),'Q32 w16/s1 options')
    expected=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,num_seqs=sequences,num_query_heads=16,num_queries_per_kv=8,
                  KV_QUANT_MODE=0,IS_3D=False)
    need(type(meta['constants']) is dict and set(meta['constants'])==set(expected)
         and all(type(meta['constants'][k]) is type(v) and meta['constants'][k]==v for k,v in expected.items()),
         'two-sequence private Q32 selected constants' if _multi2 else 'single-sequence Q32 selected constants')
    q=meta['query_rows'];need(type(q) is int and 1024<=q<2**31,'CPU-visible Q32 query scope')
    need(list(meta['grid'])==[q//32+sequences,2] and all(type(v) is int for v in meta['grid']),
         'complete two-sequence parent grid' if _multi2 else 'complete single-sequence parent grid')
    for key in ('kernel_hash','source_digest','ttgir_sha256'):
        need(type(meta[key]) is str and re.fullmatch('[0-9a-f]{64}',meta[key]),'actual digest '+key)
    need(meta['ttgir_sha256']==sha(text),'actual metadata/input TTGIR binding')
    need(type(meta['function_name']) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',meta['function_name']),'actual function symbol')
    need(meta['pins']==PINS,'SDK/helper/JIT/codec pins')


def propose(actual_ttgir,actual_meta,*,_multi2=False):
    """Return (None, disabled-proof) by default; otherwise a fresh CPU proposal.

    Actual metadata must come from the caller's authenticated compiled object /
    selected launch. This function does not manufacture that binding. Catch
    Unsupported only BEFORE launch to retain the same correct private unhoisted
    call; postlaunch errors must propagate, not become a BF16 cache fallback.
    """
    need(type(_multi2) is bool,'Boolean multi2 scope')
    need(type(actual_meta) is dict,'metadata dictionary')
    enabled=actual_meta.get('enabled',False);need(type(enabled) is bool,'Boolean default-off flag')
    if not enabled:return None,dict(enabled=False,override_generated=False,actual_I8_hoist_verified=False)
    need(type(actual_ttgir) is str and actual_ttgir.endswith('\n'),'complete TTGIR text')
    validate_meta(actual_ttgir,actual_meta,_multi2=_multi2)
    before=grammar(actual_ttgir,actual_meta['function_name']);lines=actual_ttgir.splitlines(keepends=True)
    old=lines.pop(before['Q_operand_load']-1);indent=re.match(r' *',lines[before['KV_loop']-1])[0]
    lines.insert(before['KV_loop']-1,indent+old.lstrip(' '));override=''.join(lines)
    after=grammar(override,actual_meta['function_name'],hoisted=True)
    restored=lines.copy();del restored[after['Q_operand_load']-1];restored.insert(before['Q_operand_load']-1,old)
    need(''.join(restored)==actual_ttgir,'one-operation exact byte inverse')
    return override,dict(schema=MULTI2_SCHEMA if _multi2 else SCHEMA,enabled=True,override_generated=True,moved_operations=1,
        input_sha256=sha(actual_ttgir),override_sha256=sha(override),actual_metadata=deepcopy(actual_meta),before=before,after=after,
        exact_byte_inverse=True,all_other_source_math_masks_stores_barriers_unchanged=True,
        old_overlay_or_hash_used_as_admission_predicate=False,caller_actual_metadata_attestation_required=True,
        fresh_override_compile_required=True,actual_override_compiled=False,actual_I8_hoist_verified=False,
        native_Q_operand_dominance_alias_synchronization_verified=False,loaded_resources_verified=False,
        runtime_binding_qualified=False,numerical_verified=False,Graph_verified=False,precision_gate=False,
        performance_verified=False,service_verified=False)
