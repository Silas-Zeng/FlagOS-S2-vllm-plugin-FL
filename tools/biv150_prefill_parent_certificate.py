"""Installation-only proof of the actual parent-selected-Q32 descriptor gate.

No framework import. A cache may reuse only the duplicate private descriptor
check: the audited parent policy and original Manager still check every launch.
"""
import hashlib
from pathlib import Path
import sys
from types import CodeType

PINS={
    'biv150_prefill_q32.py':'3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b',
    'biv150_prefill_multi2.py':'fc3cdd066aacc6dfdc5029bb8aae9589f868b6800534511d2198f1f0d88da342',
    'biv150_q_fragment_service.py':'38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82'}


def child_code(code,name):
    found=[c for c in code.co_consts if isinstance(c,CodeType)and c.co_name==name]
    if len(found)!=1:raise ValueError('Unique compiled code required: '+name)
    return found[0]


def expected_code(raw,filename,predicate=False):
    # Match SourceFileLoader compilation, including the COMPLETE module symbol
    # table/import-call optimization and source future flags. Do not exec it.
    module=compile(raw,filename,'exec',flags=0,dont_inherit=True,optimize=sys.flags.optimize)
    return (child_code(module,'_eligible')if predicate
            else child_code(child_code(module,'_Policy'),'__getitem__'))


def code_differences(actual,expected,path='code'):
    fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize',
        'co_flags','co_code','co_names','co_varnames','co_filename','co_name','co_qualname',
        'co_firstlineno','co_linetable','co_exceptiontable','co_freevars','co_cellvars')
    changed=[path+'.'+key for key in fields if getattr(actual,key)!=getattr(expected,key)]
    if len(actual.co_consts)!=len(expected.co_consts):changed.append(path+'.co_consts.length')
    for i,(before,after)in enumerate(zip(actual.co_consts,expected.co_consts)):
        nested=path+'.co_consts['+str(i)+']'
        if isinstance(before,CodeType)and isinstance(after,CodeType):
            changed.extend(code_differences(before,after,nested+'('+before.co_name+')'))
        elif type(before)is not type(after)or before!=after:changed.append(nested)
    return changed


class ParentCertificate:
    def __init__(self,root,chain,manager):
        root=Path(root).resolve();sources={}
        for name,pin in PINS.items():
            raw=(root/'vllm_fl/patches'/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Unaudited parent source: '+name)
            sources[name]=raw
        self.chain,self.policy,self.manager=chain,chain.base,manager
        if self.policy.manager is not manager:raise ValueError('Parent manager identity differs')
        self.m2_type,self.q32_type=type(chain),type(self.policy)
        self.m2_method,self.q32_method=self.m2_type.__getitem__,self.q32_type.__getitem__
        self.fragment=self.q32_method.__globals__['qfragment']
        if self.m2_method.__globals__['qfragment']is not self.fragment:
            raise ValueError('Parent descriptor module identity differs')
        self.predicate=self.fragment._eligible
        for name,actual in [('biv150_prefill_multi2.py',self.m2_method),('biv150_prefill_q32.py',self.q32_method)]:
            self._verify_loaded(actual,name,root,sources[name])
        self._verify_loaded(self.predicate,'biv150_q_fragment_service.py',root,
            sources['biv150_q_fragment_service.py'],predicate=True)
        self.codes=(self.m2_method.__code__,self.q32_method.__code__,self.predicate.__code__)
        invokes=[c for c in self.q32_method.__code__.co_consts if isinstance(c,CodeType)and c.co_name=='invoke']
        if len(invokes)!=1:raise ValueError('Unique audited parent-selected invocation required')
        self.selected_invoke_code=invokes[0]
        self.proof=dict(source_pins=PINS.copy(),loaded_policy_and_predicate_code_equal=True,
            whole_module_compile_context_preserved=True,module_not_executed_for_reference=True,
            compile_optimize=sys.flags.optimize,compile_dont_inherit=True,
            loaded_function_and_globals_paths_equal_pinned_root=True,all_CodeType_fields_strictly_compared=True,
            parent_checks_mutable_tensor_descriptors_each_call=True,
            private_duplicate_constexpr_qualification_only=True,
            actual_parent_invoke_code_checked_at_boundary=True,
            arbitrary_direct_launchers_certified=False,source_or_hash_reads_in_hot_path=False)
        if not self.intact():raise ValueError('Parent qualification changed during setup')

    @staticmethod
    def _verify_loaded(actual,name,root,raw,predicate=False):
        path=(root/'vllm_fl/patches'/name).resolve()
        globals_file=actual.__globals__.get('__file__')
        if (Path(actual.__code__.co_filename).resolve()!=path or not isinstance(globals_file,str)
                or Path(globals_file).resolve()!=path):
            raise ValueError('Loaded parent path differs: '+name+' code='+actual.__code__.co_filename
                +' globals='+str(globals_file)+' expected='+str(path))
        expected=expected_code(raw,actual.__code__.co_filename,predicate)
        differences=code_differences(actual.__code__,expected)
        if actual.__code__!=expected or differences:
            raise ValueError('Loaded parent code differs: '+name+' fields='
                +repr(differences))

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
