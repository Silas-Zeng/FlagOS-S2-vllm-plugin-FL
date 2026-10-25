"""DIAGNOSTIC ONLY: copy as sitecustomize.py in an isolated startup driver.

Observe the offending pybind object without changing pickle results or cache
settings. Never use this instrumentation for accuracy/performance acceptance.
"""
import os
if os.environ.get('BIV150_PICKLE_DIAGNOSTIC')=='1':
    import gc
    import json
    import pickle
    from torch._inductor.codecache import FxGraphCachePickler
    previous=getattr(FxGraphCachePickler,'reducer_override',None)
    def observe(self,obj):
        kind=type(obj).__module__+'.'+type(obj).__name__
        if 'pybind11_detail_function_record' in kind:
            refs=[]
            for ref in gc.get_referrers(obj):
                if isinstance(ref,dict):
                    refs.append(dict(kind='dict',keys=[str(k)[:180] for k,v in list(ref.items()) if v is obj][:15]))
                elif getattr(ref,'__self__',None) is obj:
                    owners=[]
                    for owner in gc.get_referrers(ref):
                        if isinstance(owner,dict):
                            owners.append(dict(keys=[str(k)[:100] for k,v in list(owner.items()) if v is ref][:10],
                                operator={k:str(owner[k])[:250] for k in ('_name','_qualified_op_name','_schema','_opname','_overloadname','__name__') if k in owner}))
                    refs.append(dict(kind=type(ref).__name__,name=getattr(ref,'__name__',None),
                        module=getattr(ref,'__module__',None),qualname=getattr(ref,'__qualname__',None),owners=owners[:30]))
            print('BIV150_PICKLE_DIAG '+json.dumps(dict(kind=kind,referrers=refs)),flush=True)
        return previous(self,obj) if previous else NotImplemented
    FxGraphCachePickler.reducer_override=observe
    original_dumps=FxGraphCachePickler.dumps
    def dump_with_field_diagnostic(self,obj):
        try:return original_dumps(self,obj)
        except RuntimeError:
            budget=[120]
            def probe(value):
                # Match existing reducers, but use a fresh memo/stream. This is
                # only after the actual serialization has already failed.
                pickler=FxGraphCachePickler(None)
                pickler.dispatch_table=self.dispatch_table.copy()
                pickle.Pickler.dump(pickler,value)
            def visit(value,path,depth):
                if depth>5 or budget[0]<=0:return
                if isinstance(value,dict):items=list(value.items())
                elif isinstance(value,(tuple,list)):items=list(enumerate(value))
                elif hasattr(value,'__dict__') and not isinstance(value,(__import__('torch').Tensor,type)):
                    items=list(vars(value).items())
                else:return
                for key,child in items:
                    if budget[0]<=0:return
                    budget[0]-=1
                    try:probe(child)
                    except Exception as error:
                        target=path+'.'+str(key)
                        print('BIV150_PICKLE_FIELD '+json.dumps(dict(path=target,kind=type(child).__module__+'.'+type(child).__name__,error=repr(error)[:250])),flush=True)
                        visit(child,target,depth+1)
            try:visit(obj,'details',0)
            except Exception as diagnostic_error:
                print('BIV150_PICKLE_FIELD_DIAGNOSTIC_ERROR '+repr(diagnostic_error),flush=True)
            raise
    FxGraphCachePickler.dumps=dump_with_field_diagnostic
