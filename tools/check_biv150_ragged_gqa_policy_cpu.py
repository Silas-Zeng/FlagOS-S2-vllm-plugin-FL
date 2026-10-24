"""Execute old/new chooser ASTs and verify bounded changes and exact source pins."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

BASE='bed003eca3affcd46907a95017331990b200c1fb'

def chooser(raw,target):
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments')
    scope=dict(TARGET_PROGRAMS=target)
    exec(ast.unparse(node),scope)
    return scope['_pick_segments']

def main(args):
    root=Path(__file__).resolve().parents[1]
    path='vllm_fl/patches/biv150_gqa_decode.py'
    old=subprocess.check_output(['git','-C',str(root),'show',BASE+':'+path],text=True)
    new=(root/path).read_text()
    assert hashlib.sha256(old.encode()).hexdigest()=='2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
    before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert before.keys()==after.keys()
    assert all(before[k]==after[k] for k in before if k!='_pick_segments')
    count=changed=0
    for target in (128,256,384,512):
        original,candidate=chooser(old,target),chooser(new,target)
        for batch in range(257):
            for heads in (1,2,4,8):
                for builder in (1,2,4,8,16,32,64):
                    got=candidate(batch,heads,builder)
                    selected=target==256 and heads==2 and builder==16 and batch in (29,30,60,61,62,63)
                    expected=(4 if batch in (29,30) else 2) if selected else original(batch,heads,builder)
                    assert got==expected and 1<=got<=builder,(batch,heads,builder,target,got,expected)
                    count+=1;changed+=got!=original(batch,heads,builder)
    q=(root/'vllm_fl/patches/biv150_q_fragment_service.py').read_text()
    order=(root/'vllm_fl/patches/biv150_compute_order.py').read_text()
    pins={}
    for tree,name in [(ast.parse(q),'EXPECTED_GQA'),(ast.parse(order),'SOURCES')]:
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        pins[name]=ast.literal_eval(node.value)
    gqa_hash=hashlib.sha256(new.encode()).hexdigest();q_hash=hashlib.sha256(q.encode()).hexdigest()
    assert pins['EXPECTED_GQA']==pins['SOURCES']['vllm_fl.patches.biv150_gqa_decode']==gqa_hash
    assert pins['SOURCES']['vllm_fl.patches.biv150_q_fragment_service']==q_hash
    observer=ast.parse((root/'tools/biv150_gqa_coverage.py').read_text())
    observer_pins=ast.literal_eval(next(n.value for n in observer.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets)))
    assert observer_pins['vllm_fl.patches.biv150_gqa_decode']==gqa_hash
    result=dict(cpu_only=True,base=BASE,chooser_cases=count,changed_cases=changed,
                unchanged_kernel_and_wrapper_asts=True,scratch_bounds_preserved=True,
                unselected_and_nondefault_targets_exact=True,gqa_sha256=gqa_hash,q_fragment_sha256=q_hash,source_pins_exact=True)
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);main(p.parse_args())
