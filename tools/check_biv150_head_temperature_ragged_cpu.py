"""Standard-library CPU checks; fake launches prove protocol only, never numerics."""
import argparse
import ast
import copy
import contextlib
import io
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS
import weakref

BASE = 'a455a3de364051fbea80226870c239de3a48e716'
FLAG = 'VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED'
ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / 'vllm_fl/patches/biv150_lm_head_temperature.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def kernel_specialization_proof(kernel, original):
    """False specialization is the original AST; True has explicit load masks."""
    before=ast.parse(original)
    current=ast.parse(kernel.read_bytes())
    class Choose(ast.NodeTransformer):
        def __init__(self, ragged):self.ragged=ragged;self.count=0
        def visit_If(self,node):
            if isinstance(node.test,ast.Name) and node.test.id=='RAGGED_M':
                self.count+=1
                return [self.visit(n) for n in (node.body if self.ragged else node.orelse)]
            return self.generic_visit(node)
    projections={}
    for enabled in (False,True):
        tree=copy.deepcopy(current);chooser=Choose(enabled);tree=chooser.visit(tree)
        assert chooser.count==4
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        assert fn.args.args[-1].arg=='RAGGED_M' and len(fn.args.defaults)==1
        assert isinstance(fn.args.defaults[0],ast.Constant) and fn.args.defaults[0].value is False
        fn.args.args.pop();fn.args.defaults=[]
        tree.body[0]=copy.deepcopy(before.body[0])  # descriptive module docstring only
        projections[enabled]=tree
    assert ast.dump(projections[False],include_attributes=False)==ast.dump(before,include_attributes=False)
    assignments=[n for n in ast.walk(projections[True]) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ram' for t in n.targets)]
    assert len(assignments)==1 and isinstance(assignments[0].value,ast.Name) and assignments[0].value.id=='rm'
    loads=[n for n in ast.walk(projections[True]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='A']
    assert len(loads)==3
    row=ast.parse('rm[:, None] < M',mode='eval').body
    both=ast.parse('(rm[:, None] < M) & (rk[None, :] < k_remaining)',mode='eval').body
    class OnlyAllowedRaggedEdits(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='ram' for t in node.targets):
                node.value=ast.Name(id='rm',ctx=ast.Load())
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and node.func.attr=='load' and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='A':
                masks=[k for k in node.keywords if k.arg=='mask']
                if masks:
                    assert len(masks)==1
                    masks[0].value=ast.BinOp(left=copy.deepcopy(row),op=ast.BitAnd(),right=masks[0].value)
                else:
                    assert not node.keywords
                    node.keywords=[ast.keyword(arg='mask',value=copy.deepcopy(row)),ast.keyword(arg='other',value=ast.Constant(value=0))]
            return self.generic_visit(node)
    expected_true=OnlyAllowedRaggedEdits().visit(copy.deepcopy(before))
    assert ast.dump(projections[True],include_attributes=False)==ast.dump(expected_true,include_attributes=False)
    masks=[]
    for node in loads:
        kw={k.arg:k.value for k in node.keywords};assert set(kw)=={'mask','other'}
        mask=ast.dump(kw['mask'],include_attributes=False);masks.append(mask)
        assert mask in (ast.dump(row,include_attributes=False),ast.dump(both,include_attributes=False))
        other=kw['other'];assert (isinstance(other,ast.Constant) and other.value==0) or (isinstance(other,ast.Name) and other.id=='_0')
    assert masks.count(ast.dump(row,include_attributes=False))==2 and masks.count(ast.dump(both,include_attributes=False))==1
    assert b'tl.zeros((1, 1), dtype=output_ptr.dtype.element_ty)' in kernel.read_bytes()
    return dict(aligned_false_specialization_AST_matches_original=True,
      whole_kernel_bytes_unchanged=False,explicit_ragged_input_loads=len(loads),
      ragged_ram_is_true_rm=True,ragged_A_loads_row_masked_with_zero=True,
      ragged_true_only_address_and_load_masks_changed=True,dot_weight_load_loop_and_rounding_AST_unchanged=True,
      non_even_K_tail_combines_original_K_mask=True,compiled_IR_or_numerics_verified=False,
      original_kernel_sha256=hashlib.sha256(original).hexdigest(),new_kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest())


def check():
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    m = load(HEAD, '_ragged_head_cpu')
    source = HEAD.read_text()
    before = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':vllm_fl/patches/biv150_lm_head_temperature.py']).decode()
    inverse = source.replace("_RAGGED_FLAG = 'VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED'\n", '')
    start = inverse.index('# Only explicit experimental rows;')
    end = inverse.index('_PINS = {', start)
    inverse = inverse[:start] + inverse[end:]
    inverse = inverse.replace('(runner.input_batch.num_reqs not in _CONFIGS\n                and _config_for_rows(runner.input_batch.num_reqs) is None)', 'runner.input_batch.num_reqs not in _CONFIGS')
    inverse = inverse.replace('(rows in _CONFIGS or _config_for_rows(rows) is not None)\n                and runner.input_batch.num_reqs == rows', 'rows in _CONFIGS and runner.input_batch.num_reqs == rows')
    inverse = inverse.replace("config = _CONFIGS[rows] if rows in _CONFIGS else _config_for_rows(rows)\n        if config is None:\n            raise ValueError('head row scope changed before launch')", 'config = _CONFIGS[rows]')
    inverse = inverse.replace(', RAGGED_M=rows not in _CONFIGS', '')
    inverse = inverse.replace('# 保留原W descriptor、K循环和温度舍入；ragged输入mask由独立constexpr控制。', '# 原受测自然W descriptor，完整原runtime K循环与配置；仅store前新epilogue。')
    assert inverse == before.replace('\r\n', '\n')
    kernel = ROOT / 'vllm_fl/patches/biv150_lm_head_temperature_kernel.py'
    original_kernel=subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':vllm_fl/patches/biv150_lm_head_temperature_kernel.py'])
    kernel_proof=kernel_specialization_proof(kernel,original_kernel)
    configs = {r: dict(c) for r, c in m._CONFIGS.items()}
    routes = []
    for flag in (None, '0', '1', 'true', '2'):
        if flag is None: os.environ.pop(FLAG, None)
        else: os.environ[FLAG] = flag
        accepted = []
        for rows in range(0, 131):
            c = m._config_for_rows(rows)
            expected = rows in (32, 64) or (flag == '1' and rows in (31, 41, 42, 43))
            assert (c is not None) == expected
            if c is not None:
                base = rows if rows in (32, 64) else (32 if rows == 31 else 64)
                assert c is m._CONFIGS[base]
                accepted.append(rows)
        for bad in ('31', None, True, 31.0, 41.0, 42.0, 43.0):
            assert m._config_for_rows(bad) is None
        routes.append(dict(flag=flag, accepted=accepted))
    assert m._CONFIGS == configs
    # Ragged rows load only true rm<M; aligned False preserves the old hints.
    # This is source arithmetic, not proof of fresh generated device behavior.
    address = []
    for rows, base in [(31, 32), (32, 32), (41, 64), (42, 64), (43, 64), (64, 64)]:
        bm, bn = configs[base]['BLOCK_M'], configs[base]['BLOCK_N']
        assert configs[base]['grid'] == (130560 // bn, 1)
        ragged=rows not in (32,64)
        reads = {r for r in range(bm) if r<rows} if ragged else {r%rows for r in range(bm)}
        assert reads == set(range(rows))
        stores = [r for r in range(bm) if r < rows]
        assert stores == list(range(rows))
        assert (130560 % bn) == (2048 % configs[base]['BLOCK_K']) == 0
        assert max(reads) * 2048 + 2047 == rows * 2048 - 1
        address.append(dict(rows=rows,tile_rows=bm,RAGGED_M=ragged,zero_filled_inactive_read_rows=bm-rows if ragged else 0,
          repeated_read_rows=0,active_stores=len(stores),temperature_masked_rows=bm-rows))
    # Real complete module, no AST function substitution. Only imported runtime
    # dependencies are explicit opaque stand-ins; no SDK admission is claimed.
    saved = dict(sys.modules)
    class Tensor:
        def __init__(self, shape, dtype='bf16', device=None):
            self.shape = tuple(shape); self.ndim = len(shape); self.dtype = dtype
            self.device = device or NS(type='cuda'); self.requires_grad = False
        def is_contiguous(self): return True
    class Kernel:
        calls = []
        failure = None
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))
                if self.failure is not None: raise self.failure
                return NS(metadata=NS(hash='CPU_FAKE_NOT_DEVICE', shared=None), n_regs=None, n_spills=None)
            return launch
    class Model:
        def __init__(self): self.lm_head = NS(weight=Tensor((130560, 2048)))
        def compute_logits(self, hidden): return ('raw', hidden)
    class Sampler:
        def __call__(self, **kwargs): return ('original_sampler', kwargs)
    class Noop: pass
    torch = ModuleType('torch'); torch.bfloat16 = 'bf16'; torch.float32 = 'fp32'
    capture = [False]
    torch.cuda = NS(is_current_stream_capturing=lambda: capture[0])
    torch.empty = lambda shape, dtype, device: Tensor(shape, dtype, device)
    platform = ModuleType('vllm.platforms'); platform.current_platform = NS(vendor_name='iluvatar')
    processors = ModuleType('vllm.v1.sample.logits_processor.builtin')
    processors.MinTokensLogitsProcessor = type('MinTokensLogitsProcessor', (), {})
    processors.LogitBiasLogitsProcessor = type('LogitBiasLogitsProcessor', (), {})
    km = ModuleType('vllm_fl.patches.biv150_lm_head_temperature_kernel'); km.head_temperature_kernel = Kernel()
    sys.modules.update({'torch': torch, 'vllm.platforms': platform, processors.__name__: processors, km.__name__: km})
    protocol = []
    def fixture(rows):
        sm = NS(all_random=True, all_greedy=False, no_penalties=True, max_num_logprobs=None,
                logprob_token_ids=None, allowed_token_ids_mask=None, bad_words_token_ids=None,
                spec_token_ids=None, top_k=None, top_p=Tensor((rows,), 'fp32'),
                temperature=Tensor((rows,), 'fp32'), thinking_budget_state_holder=None,
                logitsprocs=NS(non_argmax_invariant=[]))
        model, sampler = Model(), Sampler()
        runner = NS(input_batch=NS(num_reqs=rows, sampling_metadata=sm, req_ids=['r'+str(i) for i in range(rows)]),
                    num_prompt_logprobs=None, speculative_config=None, broadcast_pp_output=False,
                    use_aux_hidden_state_outputs=False, is_pooling_model=False, lora_config=None,
                    routed_experts_initialized=False, requests={ 'r'+str(i):NS(sampling_params=NS(structured_outputs=None)) for i in range(rows)},
                    model=model, sampler=sampler, get_model=lambda: model)
        rt = m._Runtime.__new__(m._Runtime)
        rt.model, rt.weight, rt.sampler = weakref.ref(model), weakref.ref(model.lm_head.weight), weakref.ref(sampler)
        rt.selected, rt.sampled, rt.proxy = set(), set(), sampler
        setattr(runner, m._STATE, rt)
        return runner, NS(has_structured_output_requests=False), Tensor((rows, 2048)), rt
    try:
        os.environ[FLAG] = '1'
        os.environ[m._FLAG] = '1'
        for rows in (31, 32, 41, 42, 43, 64):
            r, so, hidden, rt = fixture(rows)
            before_calls = len(km.head_temperature_kernel.calls)
            out, state = m.compute_logits(r, hidden, so, None)
            assert state is not None and out.shape == (rows, 130560)
            grid, args, kw = km.head_temperature_kernel.calls[-1]
            assert len(km.head_temperature_kernel.calls) == before_calls+1
            assert args[0] is hidden and args[1] is r.model.lm_head.weight and args[-1] is r.input_batch.sampling_metadata.temperature
            assert args[4:7] == (rows, 130560, 2048)
            base = rows if rows in (32,64) else (32 if rows==31 else 64)
            assert grid == configs[base]['grid'] and all(kw[k] == v for k,v in configs[base].items() if k != 'grid')
            assert kw['RAGGED_M'] is (rows not in (32,64))
            assert m.prepare_sampling(r, out, state, so, None) == (out, state)
            assert m.prepare_sampling(r, out, state, so, object())[1] is None
            # Same-shape fresh metadata object must invalidate the old ticket.
            old = r.input_batch.sampling_metadata
            r.input_batch.sampling_metadata = NS(**vars(old))
            assert m.prepare_sampling(r, out, state, so, None)[1] is None
            r.input_batch.sampling_metadata = old
            if rows in (31,41,42,43):
                os.environ[FLAG] = '0'
                assert m.prepare_sampling(r, out, state, so, None)[1] is None
                os.environ[FLAG] = '1'
            protocol.append(dict(rows=rows, original_argument_identity=True, grid=grid, stale_and_grammar_fallback=True))
        refusals = []
        for name, mutate in [
            ('greedy',lambda r: (setattr(r.input_batch.sampling_metadata,'all_random',False),setattr(r.input_batch.sampling_metadata,'all_greedy',True))),
            ('penalties',lambda r:setattr(r.input_batch.sampling_metadata,'no_penalties',False)),
            ('top_k',lambda r:setattr(r.input_batch.sampling_metadata,'top_k',Tensor((41,)))),
            ('request_count',lambda r:setattr(r.input_batch,'num_reqs',40)),
            ('prompt_logprobs',lambda r:setattr(r,'num_prompt_logprobs',True)),
            ('temperature_shape',lambda r:setattr(r.input_batch.sampling_metadata,'temperature',Tensor((40,),'fp32'))),
        ]:
            r,so,h,rt=fixture(41);mutate(r);n=len(km.head_temperature_kernel.calls)
            out,state=m.compute_logits(r,h,so,None);assert state is None and out[0]=='raw' and len(km.head_temperature_kernel.calls)==n;refusals.append(name)
        r,so,h,rt=fixture(41);capture[0]=True
        assert m.compute_logits(r,h,so,None)[1] is None;capture[0]=False;refusals.append('capture')
        for env in ({FLAG:'0'},{m._FLAG:'0'}):
            os.environ.update(env);assert m.compute_logits(r,h,so,None)[1] is None;os.environ[FLAG]=os.environ[m._FLAG]='1'
        # An actual launch failure must propagate, with no original retry.
        km.head_temperature_kernel.failure=RuntimeError('CPU fake launch error')
        try: m.compute_logits(r,h,so,None)
        except RuntimeError as e: assert str(e)=='CPU fake launch error'
        else: raise AssertionError('launch failure swallowed')
        km.head_temperature_kernel.failure=None
        try: rt.compute(Tensor((40,2048)),r.input_batch.sampling_metadata)
        except ValueError as e: assert str(e)=='head row scope changed before launch'
        else: raise AssertionError('unsupported compute launched')
    finally:
        for k in set(sys.modules)-set(saved):sys.modules.pop(k,None)
        sys.modules.update(saved)
        os.environ.pop(FLAG,None);os.environ.pop(m._FLAG,None)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    # Same source-only importlib entry used by independent service helpers:
    # no package parent, no already-loaded torch, explicit ABI description.
    program = r'''
import builtins, importlib.util, json, sys
from types import SimpleNamespace as NS
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch','triton','vllm','vllm_fl'):
        raise AssertionError('unexpected framework import: '+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
spec=importlib.util.spec_from_file_location('_standalone_semantics',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
c=NS(model_config=NS(hf_config=NS(**m._GEOMETRY)))
e={k:'0' for k in m._FEATURES};e['VLLM_FL_BIV150_HEAD_TEMPERATURE']='1'
abi={'torch':'2.10.0','cxx11_abi':False}
def run(env):return m.build_fingerprint(c,'iluvatar',environ=env,abi=abi)['sha256']
f='VLLM_FL_BIV150_HEAD_TEMPERATURE_RAGGED'
a=run(e);b=run(dict(e,**{f:'0'}));d=run(dict(e,**{f:'1'}))
assert a==b and a!=d
head_off=dict(e,VLLM_FL_BIV150_HEAD_TEMPERATURE='0')
assert run(head_off)==run(dict(head_off,**{f:'1'}))
assert 'torch' not in sys.modules and 'triton' not in sys.modules
print(json.dumps({'unset_equals_zero':True,'one_changes_hash':True,'head_off_masks_ragged':True,'no_framework_import':True}))
'''
    output = subprocess.check_output([sys.executable,'-I','-B','-W','error','-c',program,str(ROOT/'vllm_fl/compile_semantics.py')],text=True)
    fingerprint = json.loads(output)
    legacy = ROOT / 'docs/evidence/head-temperature-ragged-20261029/legacy'
    expected_labels = {'normal_real_head','independent_fp64','changed','zero','extreme','per_request_generators','side_stream','changed_graph_0','changed_graph_1','after_timing'}
    historical = []
    for label in ('r32','r64'):
        receipt = json.loads((legacy/(label+'-archive-receipt.json')).read_text())
        result_file = legacy/(label+'-result.json')
        expected = next(x for x in receipt['files'] if x['path'].endswith('/result.json'))
        assert hashlib.sha256(result_file.read_bytes()).hexdigest() == expected['sha256']
        result = json.loads(result_file.read_text())
        assert {c['label'] for c in result['cases']} == expected_labels and len(result['cases']) == 10
        assert all(all(v is True for v in c.values() if type(v) is bool) for c in result['cases'])
        historical.append(dict(rows=result['rows'], original_ten_gates_passed=True,sha256=expected['sha256']))
    assert (legacy/'r32-candidate.py').read_bytes()==original_kernel
    for label in ('r32','r64'):
        receipt=json.loads((legacy/(label+'-archive-receipt.json')).read_text())
        row=next(x for x in receipt['files'] if x['path'].endswith('/source/candidate.py'))
        assert row['sha256']==hashlib.sha256(original_kernel).hexdigest()
    assert not subprocess.check_output(['git','-C',str(ROOT),'diff','b6066b0899df81605ca1a19867459954e88d8699','--',str(legacy)])
    return dict(historical_original_only=historical,gpu_executed=False,launcher_source_inverse_to_main=True,
                kernel_specialization=kernel_proof,kernel_bytes_unchanged=False,
                standalone_fingerprint=fingerprint,
                main=BASE,routes=routes,address_enumeration=address,protocol=protocol,refusals=refusals,
                launch_errors_propagated=True,scope_change_prelaunch_rejected=True,
                actual_SDK_admission_tested=False,numerical_or_RNG_qualification=False,
                compiler_hint_validity='unsafe ragged row hints removed; fresh masked IR/device behavior still pending',
                source_sha256=hashlib.sha256(HEAD.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        result=check()
    data=json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+'\n'
    if args.out:
        with args.out.open('x') as f:f.write(data)
    print(data,end='')


if __name__=='__main__':main()
