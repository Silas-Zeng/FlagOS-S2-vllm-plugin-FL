"""CPU default. Exact steady4K sidecar binding; explicit owner GPU mode only."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

import biv150_v8_bias_byte_stage as contract


def adapted_run_tree(source_root):
    tree=ast.parse((source_root/'tools/bench_biv150_v8_bias.py').read_bytes())
    original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_gpu')
    changed=copy.deepcopy(original);count=0
    for node in ast.walk(changed):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='validate_proof':
            node.func.id='validated_byte_stage_original_proof';count+=1
    assert count==1
    restored=copy.deepcopy(changed)
    for node in ast.walk(restored):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='validated_byte_stage_original_proof':node.func.id='validate_proof'
    assert ast.dump(restored)==ast.dump(original), 'Numeric/bitwise/timing body changed'
    return ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))


def all_files_proven(original,module):
    class AllFilesProven(original):
        """Add six-file checks before the unchanged v1 pre-launch guard."""
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.six_checked=set()
        def __getitem__(self,grid):
            guarded=super().__getitem__(grid)
            def call(*args,**kwargs):
                abi=module.abi(args,kwargs,grid)
                signature=(json.dumps(abi,sort_keys=True),tuple(v.data_ptr() for v in [*args,*kwargs.values()] if hasattr(v,'data_ptr')))
                if signature not in self.six_checked:
                    compiled=self.kernel.warmup(*args,grid=grid,**kwargs);compiled._init_handles()
                    expected=next((x for x in self.allowed if x['hash']==str(compiled.hash) and x['abi']==abi),None)
                    assert expected is not None, 'Unproven byte-stage runtime specialization'
                    assert set(expected['files'])==contract.KINDS
                    for kind,item in expected['files'].items():
                        value=compiled.asm[kind];raw=value.encode() if isinstance(value,str) else value
                        assert len(raw)==item['bytes'] and contract.sha(raw)==item['sha256'], ('Actual six-file IR differs',self.role,kind)
                result=guarded(*args,**kwargs)
                self.six_checked.add(signature)
                return result
            return call
    return AllFilesProven


def run(args):
    assert args.numeric_diagnostic and args.case==['steady4k']
    assert not args.out.exists(), 'New output path required; existing artifacts are immutable'
    proof,sidecar=contract.validate_sidecar(args.sidecar)
    # Import baseline/helpers from a698, never from this validator checkout.
    for name in ('biv150_v8_decode_factory','biv150_v8_bias_probe','biv150_benchmark_utils',
                 'biv150_v8_bf16p_reference','biv150_v8_bf16p_validation'):
        assert name not in sys.modules, ('Helper already imported from another root',name)
    sys.dont_write_bytecode=True  # Preserve the original source-root/out caches too.
    sys.path.insert(0,str(contract.OLD_ROOT/'tools'))
    path=contract.OLD_ROOT/'tools/bench_biv150_v8_bias.py'
    spec=importlib.util.spec_from_file_location('a698_original_private_consumer',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    assert old.ROOT.resolve()==contract.OLD_ROOT.resolve()
    def validate_original(path,chosen):
        assert path==contract.OLD_OUT/'compile-proof.json' and chosen==['steady4k']
        original,_=contract.validate_sidecar(args.sidecar)
        assert original==proof and original['completed'] is False and original['mechanism_passed'] is False
        return original  # Never synthesize a successful v1 proof.
    old.validated_byte_stage_original_proof=validate_original
    old.Proven=all_files_proven(old.Proven,old)
    exec(compile(adapted_run_tree(contract.OLD_ROOT),'a698_run_with_explicit_v2_contract','exec'),old.__dict__)
    args.ir_proof=contract.OLD_OUT/'compile-proof.json'
    try:old.run_gpu(args)
    finally:
        if args.out.exists():
            binding=dict(sidecar=sidecar,sidecar_sha256=contract.sha(args.sidecar.read_bytes()),
                original_v1_completed=False,original_v1_mechanism_passed=False,precision_gate=False,official_score=False)
            (args.out/'byte-stage-runtime-binding.json').write_text(json.dumps(binding,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--make-sidecar',action='store_true');mode.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--source-root',type=Path,default=contract.OLD_ROOT)
    parser.add_argument('--sidecar',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--numeric-diagnostic',action='store_true')
    parser.add_argument('--case',action='append',choices=['steady4k'])
    args=parser.parse_args();args.case=args.case or ['steady4k']
    if args.case!=['steady4k'] or args.source_root!=contract.OLD_ROOT:parser.error('Only the exact original a698 root and steady4k are licensed')
    if args.make_sidecar or args.run_gpu:
        if args.out is None:parser.error('--out required')
        for root in (contract.ROOT.resolve(),contract.OLD_ROOT.resolve()):
            if args.out.resolve()==root or root in args.out.resolve().parents:parser.error('--out must be outside both source checkouts')
    if args.run_gpu and (not args.numeric_diagnostic or args.sidecar is None):parser.error('--run-gpu requires --numeric-diagnostic and --sidecar')
    if args.make_sidecar:
        print(json.dumps(contract.make_sidecar(args.out),indent=2))
    elif args.run_gpu:run(args)
    else:
        _,report=contract.archive_audit();compile(adapted_run_tree(contract.ROOT),'cpu_run_inverse','exec')
        print(json.dumps(dict(report=report,original_source_root_live_verified=False,device_work_executed=False,
            numeric_bitwise_and_full_cost_body_unchanged=True,sidecar_required_before_device=True),indent=2))
