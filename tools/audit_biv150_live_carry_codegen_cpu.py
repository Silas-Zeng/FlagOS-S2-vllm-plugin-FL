"""CPU AST audit of actual loaded original/replacement Runner and FP32 carry."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def paths(node,path=()):
    yield node,path
    for field,value in ast.iter_fields(node):
        if isinstance(value,ast.AST):yield from paths(value,path+((field,None),))
        elif isinstance(value,list):
            for index,item in enumerate(value):
                if isinstance(item,ast.AST):yield from paths(item,path+((field,index),))


def replace(root,path,value):
    parent=root
    for field,index in path[:-1]:
        parent=getattr(parent,field)
        if index is not None:parent=parent[index]
    field,index=path[-1]
    if index is None:setattr(parent,field,copy.deepcopy(value))
    else:getattr(parent,field)[index]=copy.deepcopy(value)


def fp32_add_trees(source):
    tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    assigned={}
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            assigned[node.targets[0].id]=node.value
    def expand(node,depth=0):
        if depth>20:return None
        if isinstance(node,ast.Name) and node.id in assigned:return expand(assigned[node.id],depth+1)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='to':
            inner=expand(node.func.value,depth+1)
            if not inner or len(node.args)!=1:return None
            dtype=ast.unparse(node.args[0])
            if dtype!='tl.float32':return None
            return dict(inputs=inner['inputs'],dtype='f32',expression=inner['expression'])
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.load':
            pointers={n.id for n in ast.walk(node.args[0]) if isinstance(n,ast.Name) and n.id.startswith('in_ptr')}
            if len(pointers)!=1:return None
            return dict(inputs=pointers,dtype='source',expression=next(iter(pointers)))
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
            a=expand(node.left,depth+1);b=expand(node.right,depth+1)
            if a and b and a['dtype']==b['dtype']=='f32':
                return dict(inputs=a['inputs']|b['inputs'],dtype='f32',expression=f"({a['expression']}+{b['expression']})")
        return None
    variance_sums=set()
    for node in ast.walk(fn):
        if (isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult)
            and ast.dump(node.left,include_attributes=False)==ast.dump(node.right,include_attributes=False)):
            value=expand(node.left)
            if value and len(value['inputs'])==3 and value['dtype']=='f32':variance_sums.add(value['expression'])
    results=[]
    for node in ast.walk(fn):
        if not isinstance(node,ast.BinOp) or not isinstance(node.op,ast.Add):continue
        expanded=expand(node)
        if expanded and len(expanded['inputs'])==3 and expanded['expression'] in variance_sums:
            row=dict(line=node.lineno,input_pointers=sorted(expanded['inputs']),
                expression=expanded['expression'],all_add_operands_fp32=True,no_intermediate_bf16_cast=True,
                variance_square_uses_unrounded_fp32_sum=True)
            if row not in results:results.append(row)
    return results


def audit(source,generated,event,transform):
    raw=Path(source).read_text();replacement=Path(generated).read_text()
    assert hashlib.sha256(raw.encode()).hexdigest()==event['source_sha256']
    assert hashlib.sha256(replacement.encode()).hexdigest()==event['generated_sha256']
    change=transform(raw);assert change and change['function_source']==replacement
    original_module=ast.parse(raw)
    cls=next(n for n in original_module.body if isinstance(n,ast.ClassDef) and n.name=='Runner')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='call')
    rewritten=ast.parse(replacement).body[0];restored=copy.deepcopy(rewritten)
    all_paths=list(paths(original))
    for pair in change['pairs']:
        for line in (pair['norm_line'],pair['gate_line']):
            matches=[(node,path) for node,path in all_paths if isinstance(node,ast.stmt) and node.lineno==line]
            assert len(matches)==1
            replace(restored,matches[0][1],matches[0][0])
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False), 'unrelated Runner AST changed'
    carry=[]
    for node in original_module.body:
        if not (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
            and ast.unparse(node.value.func)=='async_compile.triton'):continue
        code=ast.literal_eval(node.value.args[1]);trees=fp32_add_trees(code)
        if trees:carry.append(dict(kernel=ast.literal_eval(node.value.args[0]),
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),fp32_three_input_trees=trees))
    assert carry,'no preserved down+(O+residual) FP32 tree found in actual generated module'
    return dict(source=str(source),generated=str(generated),pairs=change['pairs'],
        only_two_call_statements_changed=True,all_other_runner_ast_unchanged=True,
        original_cached_source_unchanged=True,fp32_carry_kernels=carry,
        runtime_frequency_or_cost_measured=False)


def main(args):
    manifest=json.loads((args.run/'manifest.json').read_text());root=Path(manifest['source'])
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==manifest['git_commit']
    spec=importlib.util.spec_from_file_location('runtime_codegen',root/'vllm_fl/patches/biv150_live_carry_codegen.py')
    own=importlib.util.module_from_spec(spec);spec.loader.exec_module(own)
    marker='BIV150_LIVE_CARRY_CODEGEN_PAIR '
    events=[json.loads(line.split(marker,1)[1]) for line in (args.run/'server.log').read_text().splitlines() if marker in line]
    assert events;rows=[];seen=set()
    for event in events:
        identity=(event['source_file'],event['generated_file'])
        if identity in seen:continue
        seen.add(identity);rows.append(audit(*identity,event,own.transform))
    result=dict(cpu_only=True,imports_torch_or_executes_cached_code=False,runtime_commit=manifest['git_commit'],
        accuracy_or_performance_not_credited=True,passed=True,records=rows)
    with args.out.open('x') as file:json.dump(result,file,indent=2)
    print(json.dumps(dict(cpu_only=True,passed=True,modules=len(rows),fp32_carry_preserved=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
