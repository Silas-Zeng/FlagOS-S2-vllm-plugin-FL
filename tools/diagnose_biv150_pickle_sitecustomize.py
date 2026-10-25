"""DIAGNOSTIC ONLY: copy as sitecustomize.py in an isolated startup driver.

Observe the offending pybind object without changing pickle results or cache
settings. Never use this instrumentation for accuracy/performance acceptance.
"""
import os
if os.environ.get('BIV150_PICKLE_DIAGNOSTIC')=='1':
    import gc
    import json
    from torch._inductor.codecache import FxGraphCachePickler
    previous=getattr(FxGraphCachePickler,'reducer_override',None)
    def observe(self,obj):
        kind=type(obj).__module__+'.'+type(obj).__name__
        if 'pybind11' in kind:
            refs=[]
            for ref in gc.get_referrers(obj):
                if isinstance(ref,dict):
                    refs.append(dict(kind='dict',keys=[str(k)[:180] for k,v in list(ref.items()) if v is obj][:15]))
                elif getattr(ref,'__self__',None) is obj:
                    refs.append(dict(kind=type(ref).__name__,name=getattr(ref,'__name__',None),
                        module=getattr(ref,'__module__',None),qualname=getattr(ref,'__qualname__',None)))
            print('BIV150_PICKLE_DIAG '+json.dumps(dict(kind=kind,referrers=refs)),flush=True)
        return previous(self,obj) if previous else NotImplemented
    FxGraphCachePickler.reducer_override=observe
