#!/usr/bin/env python3
"""Stdlib replay of frozen cache-role ABI schemas and integer address recipes."""
import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from biv150_konly_view_broker import FORMAT, ROLES, View, element_addresses, layer_pages, need, parameters, validate_binding

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'docs/evidence/konly-view-broker-20261027/frozen-abi-sources-v1.tar.gz'
ARCHIVE_SHA='8a68f9211c9c09fc17fdc124aef12e50fd2aba0c8a8f8b8d28054f78ad56230b'
MAIN='dac1b8fb48e7983e48283032efeee9b592c5fbe5'


def constants(source):
    result={}
    def value(node):
        if isinstance(node,ast.Name) and node.id in result:
            return result[node.id]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
            a,b=value(node.left),value(node.right)
            if type(a) is str and type(b) is str: return a+b
            raise ValueError('Only source string concatenation is interpreted')
        return ast.literal_eval(node)
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try: result[n.targets[0].id]=value(n.value)
            except (ValueError,TypeError): pass
    return result


def schema(source,name):
    f=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return [a.arg for a in f.args.args]


def sources():
    data=ARCHIVE.read_bytes();need(hashlib.sha256(data).hexdigest()==ARCHIVE_SHA,'frozen source archive')
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
        members=t.getmembers();need(len(members)==8 and all(m.isfile() for m in members),'source inventory')
        f={m.name:t.extractfile(m).read() for m in members}
    manifest=json.loads(f.pop('manifest.json'));need(set(f)==set(manifest),'source membership')
    for name,b in f.items():
        need(len(b)==manifest[name]['bytes'] and hashlib.sha256(b).hexdigest()==manifest[name]['sha256'],'source bytes')
        ast.parse(b)
    c79,c80=constants(f['79-factory.py']),constants(f['80-factory.py'])
    need(c79['SOURCE_SHA']==manifest['main-gqa.py']['sha256'],'actual main GQA unchanged from79')
    decode=f['main-gqa.py'].decode().replace(c79['ABI_OLD'],c79['ABI_NEW'],1)
    prefill=f['80-sdk.py'].decode().replace(c80['ABI_OLD'],c80['ABI_NEW'],1)
    args={'decode79':schema(decode,'_gqa_decode_attention'),
          'prefill80':schema(prefill,'kernel_unified_attention'),
          'writer83':schema(f['83-writer.py'],'encode_append')}
    # These are parameter schemas, not compiled device pointer signatures.
    need('key_scale_ptr + page * 4096 + 1024' in c79['SCALE_FULL'],'79 scale full-base formula')
    need('pages.to(tl.int64) * 4096 + 1024' in c80['SCALE_HELPER'],'80 scale full-base formula')
    need('value_full + page * 8192 + 4096' in f['83-writer.py'].decode(),'83 full BF16 base formula')
    for filename,target in [('79-bench.py','packed_value'),('80-binding.py','packed_bf16_v')]:
        found=[n.value for n in ast.walk(ast.parse(f[filename])) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id==target for t in n.targets)]
        need(len(found)==1 and isinstance(found[0],ast.Call)
             and ast.unparse(found[0].func)=='torch.as_strided'
             and ast.literal_eval(found[0].args[2])==(8192,256,128,1)
             and ast.literal_eval(found[0].args[3])==4096,'actual already-offset V recipe '+filename)
    return args,manifest


def make_layer(pages,byte_offset):
    need(byte_offset%2==0,'test BF16 offset')
    base=0x100000000
    return View('bfloat16',(pages,2,16,2,128),(8192,4096,256,128,1),1,base,
                byte_offset+pages*16384+256,byte_offset//2,base+byte_offset,'cuda:0')


def serialize(params):
    return {k:asdict(v) if type(v) is View else v for k,v in params.items()}


def run():
    args,manifest=sources();cases=[];address_checks=0
    for pages,offset in ((1,0),(1,4),(3,256),(3,16384),(17,65536)):
        layer=make_layer(pages,offset)
        plans={role:parameters(layer,role,format_id=FORMAT) for role in ROLES}
        for role,p in plans.items():
            need(set(p).issubset(args[role]),'cache kwarg absent from frozen raw ABI')
            need(validate_binding(layer,role,p,format_id=FORMAT),'exact plan')
            cases.append(dict(pages=pages,layer_byte_offset=offset,role=role,cache_kwargs=serialize(p)))
        for page in sorted({0,pages-1}):
            for slot in range(16):
                for head in range(2):
                    for dim in range(128):
                        expected=(layer.pointer+page*16384+slot*256+head*128+dim,
                                  layer.pointer+page*16384+4096+slot*8+head*4,
                                  layer.pointer+page*16384+8192+slot*512+head*256+dim*2)
                        for role,p in plans.items():
                            if role=='writer83':
                                actual=(p['code_full'].pointer+page*16384+slot*256+head*128+dim,
                                        p['scale_full'].pointer+4*(page*4096+1024+slot*2+head),
                                        p['value_full'].pointer+2*(page*8192+4096+slot*256+head*128+dim))
                            else:
                                actual=(p['key_cache_ptr'].pointer+page*16384+slot*256+head*128+dim,
                                        p['key_scale_ptr' if role=='decode79' else 'key_code_scale_ptr'].pointer+4*(page*4096+1024+slot*2+head),
                                        p['value_cache_ptr'].pointer+2*(page*8192+slot*256+head*128+dim))
                            need(actual==expected,'role byte address mismatch');address_checks+=1
                            need(layer.pointer <= min(actual) and max(actual)+1 < layer.pointer+pages*16384,'last-byte boundary')
            for role in ROLES:
                need(element_addresses(layer,role,page,15,1,127)[2]+2 == layer.pointer+(page+1)*16384,'last V word')
    large=make_layer(2**31-1,2**36)
    for role in ROLES:
        need(element_addresses(large,role,large.shape[0]-1,15,1,127)[2]+2
             == large.pointer+large.shape[0]*16384,'legal large I64 byte offset')
    layer=make_layer(3,16384)
    source=View('bfloat16',(17,2,128),(2560,128,1),2,0x200000000,17*2560*2+8192,2048,
                0x200000000+4096,'cuda:0')
    value=replace(source,storage_offset=2304,pointer=source.storage_base+4608)
    parameters(layer,'writer83',format_id=FORMAT,readonly_sources=(source,value))
    faults=[]
    def reject(name,fn):
        try: fn()
        except (ValueError,TypeError): faults.append(name)
        else: raise AssertionError('accepted '+name)
    for name,changed in [('HND',dict(strides=(8192,4096,128,2048,1))),
                         ('page-padding',dict(strides=(16384,4096,256,128,1))),
                         ('odd-BF16-offset',dict(storage_offset=layer.storage_offset+1,pointer=layer.pointer+2)),
                         ('wrong-pointer',dict(pointer=layer.pointer+4)),('short-storage',dict(storage_bytes=16384)),
                         ('CPU-device',dict(device='cpu')),('wrong-dtype',dict(dtype='int8')),
                         ('unknown-storage',dict(storage_id=0)),('bool-dimension',dict(shape=(True,2,16,2,128))),
                         ('zero-pages',dict(shape=(0,2,16,2,128)))]:
        reject(name,lambda changed=changed:parameters(replace(layer,**changed),'writer83',format_id=FORMAT))
    reject('unknown-role',lambda:parameters(layer,'generic-bf16',format_id=FORMAT))
    reject('missing-format',lambda:parameters(layer,'writer83',format_id=None))
    reject('live-cache-alias',lambda:parameters(layer,'writer83',format_id=FORMAT,
           readonly_sources=(replace(source,storage_id=layer.storage_id,storage_base=layer.storage_base,
                                     pointer=layer.storage_base+4096),)))
    reject('same-address-other-storage',lambda:parameters(layer,'writer83',format_id=FORMAT,
           readonly_sources=(replace(source,storage_base=layer.storage_base,pointer=layer.storage_base+4096),)))
    for role in ROLES:
        expected=parameters(layer,role,format_id=FORMAT)
        scale_name={'decode79':'key_scale_ptr','prefill80':'key_code_scale_ptr','writer83':'scale_full'}[role]
        scale=expected[scale_name]
        bad=dict(expected,**{scale_name:replace(scale,shape=(32,),storage_offset=scale.storage_offset+1024,pointer=scale.pointer+4096)})
        reject(role+'-preoffset-scale',lambda role=role,bad=bad:validate_binding(layer,role,bad,format_id=FORMAT))
        vname='value_full' if role=='writer83' else 'value_cache_ptr';view=expected[vname]
        delta=4096 if role=='writer83' else -4096
        bad=dict(expected,**{vname:replace(view,shape=(1,),strides=(1,),storage_offset=view.storage_offset+delta,pointer=view.pointer+delta*2)})
        reject(role+'-wrong-V-base',lambda role=role,bad=bad:validate_binding(layer,role,bad,format_id=FORMAT))
        reject(role+'-extra-field',lambda role=role,expected=expected:validate_binding(layer,role,dict(expected,USE_PER_TOKEN_HEAD_SCALES=False),format_id=FORMAT))
        name='code_full' if role=='writer83' else 'key_cache_ptr'
        foreign=dict(expected,**{name:replace(expected[name],storage_id=99)})
        reject(role+'-foreign-storage',lambda role=role,foreign=foreign:validate_binding(layer,role,foreign,format_id=FORMAT))
    p=parameters(layer,'prefill80',format_id=FORMAT)
    reject('prefill-I8-stride-in-BF16-units',lambda:validate_binding(layer,'prefill80',dict(p,stride_k_cache_0=8192),format_id=FORMAT))
    reject('prefill-V-stride-in-byte-units',lambda:validate_binding(layer,'prefill80',dict(p,stride_v_cache_0=16384),format_id=FORMAT))
    reject('builtin-mode2',lambda:validate_binding(layer,'prefill80',dict(p,KV_QUANT_MODE=2),format_id=FORMAT))
    need(not {'torch','triton','numpy'}.intersection(sys.modules),'framework imported')
    return dict(cpu_only=True,passed=True,main=MAIN,dependencies=[79,80,82,83],source_archive_sha256=ARCHIVE_SHA,
                frozen_sources=manifest,raw_parameter_schemas=args,examples=cases,address_checks=address_checks,
                large_lastpage_i64=True,shared_readonly_qkv_allowed=True,rejected=faults,
                encoded_payload_verified=False,encoded_registered=False,activation_allowed=False,
                actual_device_views_created=False,compiled_typed_ABI_verified=False,service_installed=False,
                all_reader_coverage_complete=False,freshness_graph_reallocation_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);args=p.parse_args()
    text=json.dumps(run(),sort_keys=True,indent=2)+'\n'
    if args.out:
        need(not args.out.exists(),'fresh output');args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    else: print(text,end='')
