"""Small fixed-profile adapter over frozen80 metadata helpers, stdlib only."""
from copy import deepcopy

import biv150_konly_prefill_compile_contract as original
import biv150_konly_prefill_factory as factory
import biv150_konly_q32_numeric_contract as old_numeric
from biv150_konly_prefill_reference import grid

PROFILE='normalized2d128'
MAIN='dac1b8fb48e7983e48283032efeee9b592c5fbe5'


def plan():
    # Exact existing80 Q32 numerical geometry:2048 fresh queries after4096 prefix.
    p=deepcopy(old_numeric.plan())
    is3d,bm,bq,tile,warps,stages,_=factory.PROFILES[PROFILE]
    if (is3d,bm,bq,tile,warps,stages)!=(False,128,16,64,8,1):
        raise ValueError('Frozen BM128/Q16 profile changed')
    p.update(profile=PROFILE,grid=grid(PROFILE,p['qlens'],p['segments']),
             config=dict(BLOCK_M=bm,BLOCK_Q=bq,TILE_SIZE=tile,IS_3D=is3d),
             requested_options=dict(num_warps=warps,num_stages=stages),
             table_shape=(p['sequences'],8192),metadata_values_initialized=False,
             forced_component_profile=True,current_main_policy_selected=False,
             kernel_launches=0,initialized_tensor_elements=0,compile_only=True)
    return p


allocate=original.allocate
descriptors=original.descriptors
kwargs=original.kwargs


def check_descriptors(p,desc):
    result=original.check_descriptors(p,desc)
    # This new two-role compile adapter keeps output separate from BOTH caches.
    # The frozen80 helper remains unchanged.
    if desc['out']['storage'] in {desc[n]['storage'] for n in ('q','k','baseline_k')}:
        raise ValueError('Compile control/candidate output aliases a reader allocation')
    return dict(result,both_cache_controls_nonalias=True)


class WarmupOnly:
    """Do not expose the JIT launch subscript or call through this adapter."""
    def __init__(self,kernel):
        self.kernel=kernel;self.warmups=0;self.forbidden_launch_attempts=0
    def warmup(self,**values):
        self.warmups+=1
        return self.kernel.warmup(**values)
    def __getitem__(self,key):
        self.forbidden_launch_attempts+=1
        raise RuntimeError('Compile-only boundary forbids kernel launch')
    def __call__(self,*args,**kwargs):
        self.forbidden_launch_attempts+=1
        raise RuntimeError('Compile-only boundary forbids direct kernel call')


def typed_cache_proof(signature,names,candidate):
    if not isinstance(signature,dict): raise ValueError('Missing actual signature')
    normalized={}
    for key,value in signature.items():
        if type(key) is tuple and len(key)==1: key=key[0]
        if type(key) is int and 0 <= key < len(names): key=names[key]
        if type(key) is not str or key not in names or key in normalized:
            raise ValueError('Unknown/duplicate actual signature key')
        normalized[key]=str(value)
    required={'key_cache_ptr':'*i8' if candidate else '*bf16','value_cache_ptr':'*bf16'}
    if candidate: required['key_code_scale_ptr']='*fp32'
    if any(normalized.get(k)!=v for k,v in required.items()):
        raise ValueError('Actual cache-pointer types differ: '+repr(normalized))
    return dict(required=required,actual={k:normalized[k] for k in required},
                complete_signature=normalized,actual_signature_checked=True,
                compiled_only=True,numerical_verified=False)
