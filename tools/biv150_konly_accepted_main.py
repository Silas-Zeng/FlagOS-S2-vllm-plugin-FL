"""Actual DAC1 SDK→M2/Q32 materializer. Import/CPU contracts never import Torch."""
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import bench_biv150_konly_prefill_compile as parent

MAIN='dac1b8fb48e7983e48283032efeee9b592c5fbe5'
PINS={'biv150_q_fragment_service.py':'38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82',
      'biv150_prefill_q32.py':'3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b',
      'biv150_prefill_multi2.py':'fc3cdd066aacc6dfdc5029bb8aae9589f868b6800534511d2198f1f0d88da342',
      'biv150_prefill_normalized.py':'957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1'}


def source_contract():
    for name,digest in PINS.items():
        raw=subprocess.check_output(['git','-C',str(parent.ROOT),'show',MAIN+':vllm_fl/patches/'+name])
        if parent.factory.sha(raw)!=digest: raise ValueError('DAC1 attention source changed: '+name)
        previous=subprocess.check_output(['git','-C',str(parent.ROOT),'show',parent.factory.MAIN+':vllm_fl/patches/'+name])
        if raw!=previous: raise ValueError('The declared four-file attention equivalence no longer holds')
    return dict(accepted_main_commit=MAIN,attention_source_pins=PINS,
                accepted_attention_bytes_equal_to56f=True,actual_parent_selected=False,
                actual_BF16_Q_hoist_verified=False,actual_control_hash=None,
                service_performance_baseline_is_not_raw_d6=True)


def root_contract(root):
    root=Path(root).resolve()
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=MAIN:
        raise ValueError('Separate accepted-main checkout must be exact DAC1')
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip():
        raise ValueError('Accepted-main checkout must be clean')
    for name,digest in PINS.items():
        if parent.factory.sha((root/'vllm_fl/patches'/name).read_bytes())!=digest:
            raise ValueError('Actual main attention module changed: '+name)
    return dict(root=str(root),commit=MAIN,attention_source_pins=PINS)


def require_selected(manager,compiled):
    if compiled is None or str(compiled.hash) not in manager.optimized:
        raise ValueError('Actual main Q32 Manager did not select its proved Q-hoisted kernel')
    proof=manager.optimized[str(compiled.hash)]
    if proof.get('mode')!='selected' or not proof.get('mechanism',{}).get('mechanism_pass'):
        raise ValueError('Actual main optimized entry lacks the BF16 lowered proof')
    return proof


class Materializer:
    def __init__(self,root,torch,fixture,out):
        self.root=Path(root).resolve();self.torch,self.fixture,self.out=torch,fixture,out
        self.root_identity=root_contract(self.root);self.source=source_contract()
        sys.path.insert(0,str(self.root))
        plugin=importlib.import_module('vllm_fl')
        if Path(plugin.__file__).resolve()!=self.root/'vllm_fl/__init__.py': raise ValueError('Wrong actual main plugin import')
        for name in ('VLLM_FL_BIV150_Q_FRAGMENT','VLLM_FL_BIV150_PREFILL_Q32','VLLM_FL_BIV150_PREFILL_MULTI2'):
            if os.environ.get(name,'1')!='1': raise ValueError('Accepted main policy disabled: '+name)
        function,base,states=fixture.install_baseline()
        q32=importlib.import_module('vllm_fl.patches.biv150_prefill_q32')
        m2=importlib.import_module('vllm_fl.patches.biv150_prefill_multi2')
        self.fragment=importlib.import_module('vllm_fl.patches.biv150_q_fragment_service')
        installed=dict(states,q32=q32.install(),multi2=m2.install())
        parent.save(self.out/'main-installations.json',installed)
        if any(v.get('installed') is not True for v in installed.values()): raise ValueError('Actual main attention installation failed')
        for name,digest in PINS.items():
            module=importlib.import_module('vllm_fl.patches.'+name[:-3])
            if Path(module.__file__).resolve()!=self.root/'vllm_fl/patches'/name or parent.factory.sha(Path(module.__file__).read_bytes())!=digest:
                raise ValueError('Main import path/source is not the checked DAC1 module')
        self.manager=q32._MANAGER;self.base,self.function=base,function;self.chain=base.holder[base.kernel_key]
        if not isinstance(self.chain,m2._Policy) or not isinstance(self.chain.base,q32._Policy) or self.chain.base.manager is not self.manager:
            raise ValueError('Actual accepted M2→Q32 parent chain missing')
        expected=parent.factory.canonical(parent.factory.inputs(),'q32')
        actual_source=inspect.getsource(self.manager.raw.fn)
        (self.out/'main-normalized-raw.py.txt').write_text(actual_source)
        (self.out/'expected-normalized-raw.py.txt').write_text(expected)
        if actual_source!=expected: raise ValueError('Actual main normalized raw source differs from audited source')
        self.installed=installed;self.compiled=None;self.kwargs=None;self.grid=None;self.calls=0
    def intact(self):
        return (self.base.holder[self.base.kernel_key] is self.chain and not self.chain.disabled
                and not self.chain.base.disabled and self.chain.base.manager is self.manager
                and self.manager.active and self.manager.async_mode.get() is None
                and not self.manager.raw.pre_run_hooks and not self.manager.candidate.pre_run_hooks)
    def call(self,t,out,length):
        if not self.intact(): raise ValueError('Accepted main policy changed')
        if t['baseline_k'].dtype!=self.torch.bfloat16 or t['baseline_v'].dtype!=self.torch.bfloat16:
            raise ValueError('Encoded K must never enter the BF16 SDK control')
        manager=self.manager;original=manager.wrapper;owner=self;before=self.calls;self.compiled=None
        class Tap:
            def __getitem__(self,grid):
                def invoke(*args,**kwargs):
                    if args or tuple(grid)!=(65,2) or not owner.fragment._eligible(kwargs,256,num_warps=16,num_stages=1,tile_size=64):
                        raise ValueError('Actual SDK parent did not select the canonical Q32 boundary')
                    if kwargs['key_cache_ptr'].data_ptr()!=t['baseline_k'].data_ptr() or kwargs['value_cache_ptr'].data_ptr()!=t['baseline_v'].data_ptr() or kwargs['query_ptr'].data_ptr()!=t['q'].data_ptr() or kwargs['output_ptr'].data_ptr()!=out.data_ptr():
                        raise ValueError('Main control Tensor binding changed')
                    compiled=original[grid](**kwargs)
                    owner.compiled,owner.kwargs,owner.grid=compiled,dict(kwargs),tuple(grid);owner.calls+=1
                    require_selected(manager,compiled)
                    return compiled
                return invoke
        manager.wrapper=Tap()
        try:
            # High-level SDK arguments only. Never feed canonical B256 raw kwargs
            # through the SDK's original B128 entry point.
            self.function(q=t['q'],k=t['baseline_k'],v=t['baseline_v'],out=out,
                cu_seqlens_q=t['cu_q'],max_seqlen_q=2048,seqused_k=t['lengths'],max_seqlen_k=length,
                softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=t['table'],
                softcap=0,q_descale=None,k_descale=t['k_descale'],v_descale=t['v_descale'])
        finally:
            manager.wrapper=original
            parent.save(self.out/'main-materializer-attempt.json',dict(actual_call_count=self.calls,
                actual_grid=self.grid,observed_hash=str(self.compiled.hash) if self.compiled is not None else None,
                manager_snapshot=parent.freeze(manager.snapshot()),parent_chain_intact=self.intact()))
        if self.compiled is None or self.calls!=before+1 or not self.intact(): raise ValueError('No unique actual main materialization')
        proof=require_selected(manager,self.compiled)
        override=(manager.artifact_dir/proof['override_filename']).read_bytes()
        if parent.factory.sha(override)!=proof['override_sha256']: raise ValueError('Actual main override changed')
        (self.out/'actual-main-q-hoist-override.ttgir').write_bytes(override)
        lines=self.fragment._q_debug_lines(manager.candidate.fn)
        lines.update(self.fragment._q_debug_lines(manager.attention.__file__))
        actual=self.fragment.lowered_proof(self.compiled.asm['ttgir'],self.compiled.asm['llir'],lines,256)
        if not actual.get('mechanism_pass'): raise ValueError('Actual BF16 main lowering lacks Q-hoist')
        return dict(actual_parent_selected=True,actual_BF16_Q_hoist_verified=True,actual_control_hash=str(self.compiled.hash),
                    manager_optimized_entry=parent.freeze(proof),actual_lowered_proof=actual,
                    cpu_source_contract=self.source,accepted_main_root=self.root_identity,installations=self.installed,
                    canonical_B256_never_fed_to_SDK_raw_B128=True)
    def warmed_call(self):
        if not self.intact(): raise ValueError('Accepted main parent chain lost')
        compiled=self.manager.candidate[self.grid](**self.kwargs)
        if str(compiled.hash)!=str(self.compiled.hash): raise ValueError('Actual main materialized specialization changed')
        require_selected(self.manager,compiled);return compiled
