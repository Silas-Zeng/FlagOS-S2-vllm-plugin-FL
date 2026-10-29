"""Private N32/S4 host scope; original N64 modules and all numeric code unchanged."""
import ast
import importlib
import json
from pathlib import Path
import types

HOST='83cd884a0631feca5f6c0044e8924e53e01f1bac'
REVIEW='195f763667b37fe144700bfeac0282540ac23c70048dfe06b87acbcbbe9d575a'
CASES=((32,4,2048),(32,4,2560))
PINS={
    'biv150_native117_pair_contract':'1934659c9fafc676ad412c9d4301c1985a2cbcedee3319f6960626b7bd1357f0',
    'probe_biv150_native117_n64_consumer':'e7a8f0fc6559c1fa89e0450bc9bab6d7120bd0ed1c976f951c975d33722b03a3',
    'probe_biv150_native117_n64_fullstep':'73b146debac64bc93e96742b951ac898e331dcef29cd31f1c0a635f439721e52',
}

def private_module(original,name,**overrides):
    """Same function code, new globals. Never monkeypatch an imported N64 module."""
    result=types.ModuleType(name);result.__dict__.update(vars(original))
    result.__dict__.update(overrides);result.__dict__['__name__']=name
    for key,value in vars(original).items():
        if isinstance(value,types.FunctionType) and value.__module__==original.__name__:
            copy=types.FunctionType(value.__code__,result.__dict__,value.__name__,value.__defaults__,value.__closure__)
            copy.__kwdefaults__=value.__kwdefaults__;result.__dict__[key]=copy
    return result

def programs():
    original={name:importlib.import_module(name) for name in PINS}
    base=original['biv150_native117_pair_contract']
    root=Path(__file__).resolve().parent
    for name,digest in PINS.items():
        path=root/(name+'.py');assert base.sha(path)==digest
        assert Path(original[name].__file__).resolve()==path
    proof=dict(parent_host_commit=HOST,host_module_pins=PINS,
        scope_source_sha256=base.sha(Path(__file__)),
        consumer_cases=CASES,fullstep_case={'16k':(32,16384,4,2560)},
        separate_static_receipt_sha256=REVIEW,N64_qualification_inherited=False,
        original_N64_modules_mutated=False,numeric_or_timing_function_body_rewritten=False,
        scope_function_edits={})
    scope=private_module(base,'cjj117_n32_private_contract',CASES=CASES,REVIEW_SHA=REVIEW)
    def plan():
        value=base.plan()
        value.update(consumer_cases=CASES,fullstep_case={'16k':(32,16384,4,2560)},
            active_scope='N32/S4 only; consumer both Q pitches before 16K fullstep',
            static_receipt_sha256=REVIEW,scope_adaptation=proof)
        return json.loads(json.dumps(value))
    scope.plan=plan
    consumer=private_module(original['probe_biv150_native117_n64_consumer'],
        'cjj117_n32_private_consumer',c=scope,
        __doc__='Default CPU; Root-only original N32/S4 consumer pair, no N64 rerun.')
    full=private_module(original['probe_biv150_native117_n64_fullstep'],
        'cjj117_n32_private_fullstep',c=scope,arguments=consumer.arguments,
        __doc__='Default CPU; original width16 N32/S4 16K fullstep after its own consumer.')
    text=(root/'probe_biv150_native117_n64_fullstep.py').read_text()
    changes={
        'fixture':[("'4k'","'16k'",2)],
        'execute':[("assert a.case in (None,'4k')","assert a.case in (None,'16k')",1),
            ("f.CASES['4k']==(64,4096,2,2048)","f.CASES['16k']==(32,16384,4,2560)",1),
            ("a.case=['4k']","a.case=['16k']",1)],
        'main':[("choices=('4k',)","choices=('16k',)",1)],
    }
    for name,edits in changes.items():
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        source=ast.get_source_segment(text,node);changed,record=base.edit(source,edits)
        record['parent']=HOST
        proof['scope_function_edits'][name]=record
        exec(compile(changed,'<117-N32-S4-host-scope-'+name+'>','exec'),full.__dict__)
    assert base.CASES==((64,2,2048),(64,2,2560)) and base.REVIEW_SHA!=REVIEW
    assert scope.admission.__globals__ is scope.__dict__
    assert scope.require_pair.__globals__ is scope.__dict__
    assert consumer.execute.__globals__['c'] is scope and full.execute.__globals__['c'] is scope
    return scope,consumer,full
