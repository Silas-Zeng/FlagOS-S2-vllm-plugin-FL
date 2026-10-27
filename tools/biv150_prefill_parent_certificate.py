"""Installation-only proof of the actual parent-selected-Q32 descriptor gate.

No framework import. A cache may reuse only the duplicate private descriptor
check: the audited parent policy and original Manager still check every launch.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import CodeType

PINS={
    'biv150_prefill_q32.py':'3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b',
    'biv150_prefill_multi2.py':'fc3cdd066aacc6dfdc5029bb8aae9589f868b6800534511d2198f1f0d88da342',
    'biv150_q_fragment_service.py':'38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82'}


class ParentCertificate:
    def __init__(self,root,chain,manager):
        root=Path(root);trees={}
        for name,pin in PINS.items():
            raw=(root/'vllm_fl/patches'/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Unaudited parent source: '+name)
            trees[name]=ast.parse(raw)
        self.chain,self.policy,self.manager=chain,chain.base,manager
        if self.policy.manager is not manager:raise ValueError('Parent manager identity differs')
        self.m2_type,self.q32_type=type(chain),type(self.policy)
        self.m2_method,self.q32_method=self.m2_type.__getitem__,self.q32_type.__getitem__
        self.fragment=self.q32_method.__globals__['qfragment']
        if self.m2_method.__globals__['qfragment']is not self.fragment:
            raise ValueError('Parent descriptor module identity differs')
        self.predicate=self.fragment._eligible
        for name,actual in [('biv150_prefill_multi2.py',self.m2_method),('biv150_prefill_q32.py',self.q32_method)]:
            cls=next(n for n in trees[name].body if isinstance(n,ast.ClassDef)and n.name=='_Policy')
            ns=dict(actual.__globals__)
            exec(compile(ast.Module(body=[copy.deepcopy(cls)],type_ignores=[]),actual.__code__.co_filename,
                         'exec',flags=0,dont_inherit=True),ns)
            if actual.__code__!=ns['_Policy'].__getitem__.__code__:
                raise ValueError('Loaded parent policy code differs from pinned source')
        fn=next(n for n in trees['biv150_q_fragment_service.py'].body if isinstance(n,ast.FunctionDef)and n.name=='_eligible')
        ns={};exec(compile(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[]),self.predicate.__code__.co_filename,
                         'exec',flags=0,dont_inherit=True),ns)
        if self.predicate.__code__!=ns['_eligible'].__code__:
            raise ValueError('Loaded parent descriptor predicate differs from pinned source')
        self.codes=(self.m2_method.__code__,self.q32_method.__code__,self.predicate.__code__)
        invokes=[c for c in self.q32_method.__code__.co_consts if isinstance(c,CodeType)and c.co_name=='invoke']
        if len(invokes)!=1:raise ValueError('Unique audited parent-selected invocation required')
        self.selected_invoke_code=invokes[0]
        self.proof=dict(source_pins=PINS.copy(),loaded_policy_and_predicate_code_equal=True,
            parent_checks_mutable_tensor_descriptors_each_call=True,
            private_duplicate_constexpr_qualification_only=True,
            actual_parent_invoke_code_checked_at_boundary=True,
            arbitrary_direct_launchers_certified=False,source_or_hash_reads_in_hot_path=False)
        if not self.intact():raise ValueError('Parent qualification changed during setup')

    def intact(self):
        try:
            defaults,options=self.predicate.__defaults__,self.predicate.__kwdefaults__
            if (type(defaults)is not tuple or len(defaults)!=1 or type(defaults[0])is not int
                    or defaults[0]!=128 or type(options)is not dict or len(options)!=3):return False
            warps,stages,tile=options.get('num_warps'),options.get('num_stages'),options.get('tile_size')
            return (self.chain.base is self.policy and self.policy.manager is self.manager
                and type(self.chain)is self.m2_type and type(self.policy)is self.q32_type
                and self.m2_type.__getitem__ is self.m2_method and self.q32_type.__getitem__ is self.q32_method
                and self.m2_method.__code__ is self.codes[0]and self.q32_method.__code__ is self.codes[1]
                and self.m2_method.__defaults__ is None and self.m2_method.__kwdefaults__ is None
                and self.q32_method.__defaults__ is None and self.q32_method.__kwdefaults__ is None
                and self.fragment._eligible is self.predicate and self.predicate.__code__ is self.codes[2]
                and type(warps)is int and warps==8 and type(stages)is int and stages==1
                and type(tile)is int and tile==64
                and self.m2_method.__globals__.get('qfragment')is self.fragment
                and self.q32_method.__globals__.get('qfragment')is self.fragment)
        except (AttributeError,KeyError,TypeError):return False
