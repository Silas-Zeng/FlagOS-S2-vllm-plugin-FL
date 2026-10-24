"""Read generated Python ASTs without importing/executing cached code.

Counts are static wrapper evidence, not runtime launch counts or memory peaks.
Native opaque bodies and attention graph boundaries are explicitly out of scope.
"""

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


def function_name(node):
    return ast.unparse(node.func)


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    result=dict(static_generated_wrapper_audit=True,cache_presence_not_runtime_hit=True,
                native_opaque_bodies_not_inspected=True,device_time_not_measured=True,modules=[])
    for entry in manifest:
        path=root/entry['relative'];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
        tree=ast.parse(data.decode());wrappers=[];kernels=[]
        for function in ast.walk(tree):
            if not isinstance(function,ast.FunctionDef) or function.name!='call':continue
            calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
            names=Counter(function_name(n) for n in calls)
            operations={'clone','copy','copy_','_to_copy','contiguous','convert_element_type'}
            conversions={name:count for name,count in names.items()
                if name.startswith(('torch.ops.aten.','torch.ops.prims.','aten.','prims.'))
                and any(piece in operations for piece in name.split('.'))}
            fused_copy_names={name:count for name,count in names.items()
                if name.endswith('.run') and any(t in name for t in ('_to_copy','clone','contiguous'))}
            allocations={};aliases=[];gate=[]
            for node in ast.walk(function):
                if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
                target=ast.unparse(node.targets[0]);value=node.value
                if isinstance(value,ast.Name):aliases.append(dict(target=target,source=value.id,kind='python_assignment'))
                if not isinstance(value,ast.Call):continue
                name=function_name(value)
                if name=='empty_strided_cuda':
                    allocations[target]=dict(shape=ast.unparse(value.args[0]),strides=ast.unparse(value.args[1]),dtype=ast.unparse(value.args[2]))
                elif name=='reinterpret_tensor':
                    aliases.append(dict(target=target,source=ast.unparse(value.args[0]),kind='view',
                        shape=ast.unparse(value.args[1]),strides=ast.unparse(value.args[2])))
                if 'biv150_fused_gate' in name:
                    argument=ast.unparse(value.args[0]);gate.append(dict(name=name,input=argument,
                        input_allocation=allocations.get(argument),output=target))
            wrappers.append(dict(lineno=function.lineno,calls=dict(names),explicit_copy_or_convert_calls=conversions,
                kernel_names_mention_copy_not_standalone_copy_evidence=fused_copy_names,
                allocations=allocations,aliases=aliases,fused_gate_calls=gate,
                triton_run_statements=sum(count for name,count in names.items() if name.endswith('.run')),
                external_mm_statements=names.get('extern_kernels.mm',0)))
        for node in ast.walk(tree):
            if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Call):continue
            if function_name(node.value)!='async_compile.triton':continue
            source=next((arg.value for arg in node.value.args if isinstance(arg,ast.Constant)
                and isinstance(arg.value,str) and '@triton.jit' in arg.value),None)
            if not source:continue
            inner=ast.parse(source);stores=[];casts=Counter();loaded=set()
            for n in ast.walk(inner):
                if not isinstance(n,ast.Call):continue
                name=function_name(n)
                if name=='tl.store':stores.append(ast.unparse(n.args[0]))
                if name=='tl.load':
                    pointer=n.args[0]
                    while isinstance(pointer,ast.BinOp):pointer=pointer.left
                    loaded.add(ast.unparse(pointer))
                if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and n.args:
                    casts[ast.unparse(n.args[0])]+=1
            kernels.append(dict(name=ast.unparse(node.targets[0]),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                pointer_loads=sorted(loaded),store_addresses=stores,casts_in_kernel=dict(casts),
                casts_not_extra_wrapper_launches=True))
        result['modules'].append(dict(relative=entry['relative'],sha256=entry['sha256'],wrappers=wrappers,kernels=kernels))
    result['summary']=dict(modules=len(result['modules']),wrappers=sum(len(m['wrappers']) for m in result['modules']),
        fused_gate_wrappers=sum(bool(w['fused_gate_calls']) for m in result['modules'] for w in m['wrappers']),
        explicit_copy_convert_calls=sum(sum(w['explicit_copy_or_convert_calls'].values()) for m in result['modules'] for w in m['wrappers']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=audit(args.export)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as h:json.dump(result,h,indent=2)
    print(json.dumps(result['summary']))
