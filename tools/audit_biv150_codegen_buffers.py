"""CPU-only visible-storage liveness in the three MiniCPM generated runners.

Parse AST, never import or execute generated CUDA code. Inputs/weights, hidden
native workspaces, allocator reservations/fragmentation, binaries and graph
pools are excluded. Per-module peaks are not model/device peak measurements.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import operator
from pathlib import Path
import re

BYTES = {'torch.bfloat16':2, 'torch.float16':2, 'torch.float32':4,
         'torch.int8':1, 'torch.uint8':1, 'torch.int32':4, 'torch.int64':8}


def name(node):
    if isinstance(node, ast.Name):return node.id
    if isinstance(node, ast.Attribute):return name(node.value)+'.'+node.attr
    raise ValueError('Unsupported callable: '+ast.dump(node))


def scalar(node, env):
    if isinstance(node, ast.Constant) and type(node.value) is int:return node.value
    if isinstance(node, ast.Name) and type(env.get(node.id)) is int:return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):return tuple(scalar(x,env) for x in node.elts)
    if isinstance(node, ast.BinOp):
        op={ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
            ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod}.get(type(node.op))
        if op:return op(scalar(node.left,env),scalar(node.right,env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):return -scalar(node.operand,env)
    raise ValueError('Unsupported shape expression: '+ast.unparse(node))


def extent(shape, strides, offset=0):
    assert len(shape)==len(strides) and offset>=0
    assert all(v>=0 for v in (*shape,*strides))
    return 0 if 0 in shape else offset+1+sum((n-1)*s for n,s in zip(shape,strides))


@dataclass
class View:
    root:int
    shape:tuple
    strides:tuple
    dtype:str
    offset:int=0


class Audit:
    def __init__(self, rows):
        self.rows=rows;self.env={};self.roots={};self.events=[];self.peak=0;self.peak_live=[]
        self.linear_geometries=[];self.native=False

    def live(self, extra=()):
        ids={v.root for v in self.env.values() if isinstance(v,View)}|set(extra)
        return sorted(i for i in ids if not self.roots[i]['external'])

    def record(self, line, event, extra=()):
        ids=self.live(extra);total=sum(self.roots[i]['bytes'] for i in ids)
        self.events.append(dict(line=line,event=event,visible_live_bytes=total))
        if total>self.peak:self.peak=total;self.peak_live=[self.roots[i] for i in ids]

    def allocate(self, shape, strides, dtype, line, external=False):
        rid=len(self.roots);size=extent(shape,strides)*BYTES[dtype]
        self.roots[rid]=dict(root=rid,shape=shape,strides=strides,dtype=dtype,bytes=size,
                             external=external,allocation_line=line)
        if not external:self.record(line,'allocate',extra=(rid,))
        return View(rid,shape,strides,dtype)

    def tensor(self, node):
        if isinstance(node,ast.Name):
            value=self.env[node.id];assert isinstance(value,View);return value
        assert isinstance(node,ast.Call) and name(node.func)=='reinterpret_tensor',ast.unparse(node)
        base=self.tensor(node.args[0]);shape=scalar(node.args[1],self.env)
        strides=scalar(node.args[2],self.env);offset=base.offset+scalar(node.args[3],self.env)
        assert extent(shape,strides,offset)*BYTES[base.dtype]<=self.roots[base.root]['bytes'],'View exceeds root'
        return View(base.root,shape,strides,base.dtype,offset)

    def assignment(self, value, line):
        if isinstance(value,ast.Name):return self.env[value.id]
        if isinstance(value,ast.Call):
            target=name(value.func)
            if target in ('empty_strided_cuda','rand_strided'):
                shape=scalar(value.args[0],self.env);strides=scalar(value.args[1],self.env)
                dtype=name(value.args[2]) if target=='empty_strided_cuda' else name(next(k.value for k in value.keywords if k.arg=='dtype'))
                return self.allocate(shape,strides,dtype,line,external=target=='rand_strided')
            if target=='reinterpret_tensor':return self.tensor(value)
            if target=='torch.ops.biv150_stable_small_o.linear.default':
                x,w=map(self.tensor,value.args);assert len(x.shape)==len(w.shape)==2 and x.shape[1]==w.shape[1]
                self.linear_geometries.append((w.shape[0],w.shape[1]))
                return self.allocate((x.shape[0],w.shape[0]),(w.shape[0],1),x.dtype,line)
            if target=='torch.ops.biv150_native.gate_silu_auto.default':
                x,w,scale=map(self.tensor,value.args[:3]);assert x.shape[1]==w.shape[0] and w.shape[1]==scale.shape[0] and w.shape[1]%2==0
                self.native=True;n=w.shape[1]//2
                return self.allocate((x.shape[0],n),(n,1),x.dtype,line)
            if target=='get_raw_stream':return None
            raise ValueError('Unknown allocating expression: '+ast.unparse(value))
        return scalar(value,self.env)

    def statements(self, nodes):
        for node in nodes:
            if isinstance(node,ast.With):self.statements(node.body)
            elif isinstance(node,ast.Assign):
                assert len(node.targets)==1
                target=node.targets[0]
                if isinstance(target,ast.Tuple):
                    assert isinstance(node.value,ast.Name) and node.value.id=='args'
                    assert all(v.id in self.env for v in target.elts)
                    continue
                assert isinstance(target,ast.Name)
                if re.fullmatch(r's\d+',target.id):
                    assert isinstance(node.value,ast.Name) and type(self.env[node.value.id]) is int
                    self.env[target.id]=self.rows
                else:self.env[target.id]=self.assignment(node.value,node.lineno)
                self.record(node.lineno,'assign '+target.id)
            elif isinstance(node,ast.Delete):
                for v in node.targets:assert isinstance(v,ast.Name);del self.env[v.id]
                self.record(node.lineno,'delete')
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                call=node.value;target=name(call.func)
                if target=='extern_kernels.biv150_small_linear_out':
                    x,w=map(self.tensor,call.args);out=self.tensor(next(k.value for k in call.keywords if k.arg=='out'))
                    assert len(x.shape)==len(w.shape)==2 and x.shape[1]==w.shape[1]
                    assert out.shape==(x.shape[0],w.shape[0]) and out.root not in (x.root,w.root)
                    self.linear_geometries.append((w.shape[0],w.shape[1]))
                else:assert target in ('args.clear','torch.cuda.set_device') or target.endswith('.run'),target
            elif isinstance(node,ast.Return):
                assert isinstance(node.value,ast.Tuple)
                for v in node.value.elts:self.tensor(v)
                self.record(node.lineno,'return')
            else:raise ValueError('Unknown runner statement: '+ast.unparse(node))


def inspect_module(path, rows):
    text=path.read_text();tree=ast.parse(text)
    runners=[f for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Runner'
             for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='call']
    if not runners:return None
    examples=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='benchmark_compiled_module']
    assert len(runners)==len(examples)==1
    audit=Audit(rows)
    for node in examples[0].body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and re.fullmatch(r'arg\d+_\d+',node.targets[0].id):
            audit.env[node.targets[0].id]=audit.assignment(node.value,node.lineno)
    audit.statements(runners[0].body)
    geometry=sorted(set(audit.linear_geometries))
    kind='native-layer' if audit.native else 'edge-layer' if (12288,2048) in geometry else 'embedding'
    return dict(file=str(path),sha256=hashlib.sha256(text.encode()).hexdigest(),kind=kind,rows=rows,
                visible_created_peak_bytes=audit.peak,peak_live_roots=audit.peak_live,
                created_allocation_bytes=sum(r['bytes'] for r in audit.roots.values() if not r['external']),
                linear_geometries=geometry,events=audit.events)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--rows',type=int,action='append');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    reports=[]
    for root in args.root:
        for rows in args.rows or [32,64,2048]:
            assert 0<rows<=2048
            for path in sorted(root.rglob('*.py')):
                report=inspect_module(path,rows)
                if report:reports.append(report)
    result=dict(cpu_only=True,gpu_executed=False,reports=reports,
                limitations=__doc__,input_metadata='Static rand_strided fixtures parsed from generated benchmark_compiled_module; never executed')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('kind','rows','visible_created_peak_bytes','created_allocation_bytes')} for r in reports]))
