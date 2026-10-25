"""CPU-only generated-body/launch-choice audit; no runtime-frequency attribution."""
import argparse
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path


def dotted(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return dotted(node.value)+'.'+node.attr
    return '?'


def inspect_root(root):
    kernels=[];calls=collections.Counter();devices={}
    for path in root.rglob('*.py'):
        try:tree=ast.parse(path.read_text())
        except (SyntaxError,UnicodeError):continue
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call):continue
            name=dotted(node.func)
            if name.startswith(('extern_kernels.','torch.ops.biv150_')):calls[name]+=1
            if name!='async_compile.triton' or len(node.args)<2:continue
            source=node.args[1]
            if not isinstance(source,ast.Constant) or not isinstance(source.value,str):continue
            try:inner=ast.parse(source.value)
            except SyntaxError:continue
            for item in ast.walk(inner):
                if isinstance(item,ast.Call) and dotted(item.func)=='DeviceProperties':
                    values={v.arg:ast.literal_eval(v.value) for v in item.keywords}
                    devices[json.dumps(values,sort_keys=True)]=values
            for fn in inner.body:
                if not isinstance(fn,ast.FunctionDef):continue
                normalized=copy.deepcopy(fn);normalized.name='KERNEL';normalized.decorator_list=[]
                body=ast.dump(normalized,include_attributes=False)
                kernels.append({'file':str(path.relative_to(root)),'name':fn.name,
                                'math_ast_sha256':hashlib.sha256(body.encode()).hexdigest()})
    configs={str(p.relative_to(root)):json.loads(p.read_text()) for p in root.rglob('*.best_config')}
    return {'root':str(root),'kernel_bodies':kernels,'generated_external_call_sites':dict(calls),
            'recorded_device_properties':list(devices.values()),'saved_configs':configs,
            'not_runtime_counts':True}


def main(args):
    reports=[inspect_root(p.resolve()) for p in args.root]
    reference={v['math_ast_sha256'] for v in reports[0]['kernel_bodies']}
    for item in reports[1:]:
        values={v['math_ast_sha256'] for v in item['kernel_bodies']}
        item['normalized_math_body_comparison']={'common':len(reference&values),'old_only':len(reference-values),'new_only':len(values-reference)}
    result={'reports':reports,'limitation':'Generated artifacts are not runtime launch counts or times; matching math AST does not prove identical launch metadata, machine code, tensor layouts, or performance.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'root':v['root'],'bodies':len(v['kernel_bodies']),'saved_configs':len(v['saved_configs']),'comparison':v.get('normalized_math_body_comparison')} for v in reports]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
