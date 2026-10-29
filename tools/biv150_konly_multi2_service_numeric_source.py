"""Narrow ce905 numeric adaptation for the actual 9b production cache pair."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import biv150_konly_multi2_service_compile_source as production

c = production.frozen()  # Private helper instance, never mutate a frozen module/file.
c.plan, c.proposal, c.driver = production.plan, production.proposal, production.driver
COMPILE = '9b4e032dc6390f4f19c07a7c996ed898ba908d9e'
PRODUCTION_ROOT = Path('/workspace/experiments/minicpm-s2/code/cjj-native123-service-compile-v1-20261030')
BINDING = 'bindings/205019/f37cd742bd051d3601dc34b7163bd90afc5bdbadd8e0b49164a7abec41d8947e'
PINS = {
    'biv150_konly_multi2_numeric_source.py':'7de0d9d8048e25e5d31339354ca0ace91cdc54341594341631f8fbcee96002a1',
    'biv150_konly_multi2_numeric_contract.py':'5453a2409df82f8e867b709a72e4e5305ac5d0c0fcbf9d148109790ac5d91757',
}


def original(name):
    body=subprocess.check_output(['git','-C',str(c.ROOT),'show',production.FROZEN+':tools/'+name])
    c.need(c.sha(body)==PINS[name],'frozen ce905 numeric source '+name)
    return body.decode()


def transform(parent, pairs):
    text=parent;edits=[]
    for before,after in pairs:
        c.need(before!=after and text.count(before)==1,'unique service numeric edit '+repr(before[:70]))
        at=text.index(before);text=text[:at]+after+text[at+len(before):]
        edits.append(dict(offset=at,before=before,after=after))
    inverse=text
    for e in reversed(edits):
        at=e['offset'];c.need(inverse[at:at+len(e['after'])]==e['after'],'numeric inverse span')
        inverse=inverse[:at]+e['before']+inverse[at+len(e['after']):]
    c.need(inverse==parent,'complete ce905 inverse')
    return text,edits


source_text,source_edits=transform(original('biv150_konly_multi2_numeric_source.py'),[
    ('import subprocess\nimport biv150_konly_multi2_hoist_source as c','import subprocess\nfrom biv150_konly_multi2_service_numeric_source import c')])
source=c.module_from_source('_cjj_service_numeric_original_source',source_text)
parent_cache_source=source.cache_source


def cache_source():
    return transform(parent_cache_source(),[
        ('import biv150_konly_multi2_hoist_source as c','from biv150_konly_multi2_service_numeric_source import c'),
        ('def warmup(fn, device, group, call):','def warmup(fn, device, group, call, *, expected_memory_cache=None):'),
        ("    c.need(not memory_cache,'reader already compiled in runtime cache')",
         "    if expected_memory_cache is None:\n        c.need(not memory_cache,'first reader already compiled in runtime cache')\n"
         "    else:\n        c.need(type(expected_memory_cache) is dict and len(expected_memory_cache)==1,'exact prior unhoisted map')\n"
         "        c.need(memory_cache.keys()==expected_memory_cache.keys() and all(memory_cache[k] is v for k,v in expected_memory_cache.items()),'prior reviewed key/object map changed')\n"
         "    before=dict(memory_cache)"),
        ("    validate_group(manager,group)\n    return kernel",
         "    validate_group(manager,group)\n"
         "    c.need(len(memory_cache)==len(before)+1 and all(memory_cache.get(k) is v for k,v in before.items()),'prior keys/objects not retained or wrong cache growth')\n"
         "    added=[k for k in memory_cache if k not in before]\n"
         "    c.need(len(added)==1 and memory_cache[added[0]] is kernel,'exactly one reviewed new object required')\n"
         "    return kernel"),
    ])[0]
source.cache_source=cache_source


def contract_source():
    return transform(original('biv150_konly_multi2_numeric_contract.py'),[
        ('import biv150_konly_multi2_hoist_source as c\nimport biv150_konly_multi2_numeric_source as source',
         'from biv150_konly_multi2_service_numeric_source import c, source'),
        ("MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'", "MAIN='d5d9c56f51d731e03b78a7087ffdb1a64de93c11'"),
        ("COMPILE_COMMIT='f667e2c7150662460d60c6c870444f53e52eff6d'",'COMPILE_COMMIT='+repr(COMPILE)),
        ("MANIFEST='8b6af5b1aa715d449b95ae4826793afb8d6627de4d990eed3cba4cb2765b2391'","MANIFEST='621205ab3678cf4a597ce8c2bb363e52bf9c23276f5b55b0e6e059bbb24aafb2'"),
        ("SUMMARY='048f02b18fa27b1f96f1705408e431ed1fe47512e1831c5b5893cf6c73bedd9d'","SUMMARY='12df8d1cbf08f58ae53562a9ba0abec9cde5389917ee129a642365f0cdfb625b'"),
        ("REVIEW='8737facf61e20be7afcb265e7df113059a5eebc18c4cb536014079668b9e1729'","REVIEW='f7d2ff77125dbe832d9cff98965858d638707c79fd706c00fceaff8c9152692a'"),
        ("schema='cjj-private-multi2-hoist-numeric-v1'","schema='cjj-production-pair-multi2-hoist-numeric-v1'"),
        ("old_compile_Root_admission_granted=False,device_admission_requires_new_source_review=True,",
         "old_compile_Root_admission_granted=False,device_admission_requires_new_source_review=True, fixed_production_worktree="+repr(str(Path("/workspace/experiments/minicpm-s2/code/cjj-native123-service-compile-v1-20261030")))+", frontend_production_base=MAIN, same_production_JIT_for_both_arms=True, new_service_prepare_executed=False,"),
        ("==44 and manifest['raw_bytes']==3818887","==52 and manifest['raw_bytes']==4208045"),
        ("owner['pid']==179282 and owner['start_ticks']==362567924","owner['pid']==205019 and owner['start_ticks']==364277503"),
        ("    c.need((raw/'out/sources/candidate_multi2.py').read_text()==c.candidate_source(),'original actual GPU source still byte-identical')",
         "    from biv150_konly_multi2_service_numeric_source import production, PRODUCTION_ROOT, BINDING\n"
         "    c.need(Path(owner['command'][1]).parent.parent==PRODUCTION_ROOT,'exact original production worktree')\n"
         "    for name,digest in production.SOURCES.items():\n"
         "        c.need(c.sha((raw/'out/sources'/Path(name).name).read_bytes())==digest,'original production source '+name)\n"
         "    c.need(review['cache_complete_seven_file_groups_verified'] is True and review['production_binding']['non_none_binding'] is True,'actual prepare/cache review')"),
        ("raw/'out/multi2'/role/'record.json'","raw/'out'/BINDING/role/'record.json'"),
        ("runtime[role]=dict(source=str(output/'sources/candidate_multi2.py'),override=str(output/'multi2/override.ttgir') if role=='hoisted' else None)",
         "runtime[role]=dict(source=str(PRODUCTION_ROOT/'vllm_fl/patches/biv150_konly_encoded_multi2.py'),override=str(output/BINDING/'override.ttgir') if role=='hoisted' else None)"),
    ])


f=c.module_from_source('_cjj_service_numeric_contract',contract_source()[0])


def production_fn():
    """Resolve the same normal import/absolute source that produced the two caches."""
    git=lambda *args:subprocess.check_output(['git','-C',str(PRODUCTION_ROOT),*args])
    c.need(git('rev-parse','HEAD').decode().strip()==COMPILE and not git('status','--porcelain'),'original 9b production worktree changed')
    for name,digest in production.SOURCES.items():
        c.need(c.sha((PRODUCTION_ROOT/name).read_bytes())==digest,'original production source '+name)
    module=importlib.import_module('vllm_fl.patches.biv150_konly_encoded_multi2')
    expected=PRODUCTION_ROOT/'vllm_fl/patches/biv150_konly_encoded_multi2.py'
    c.need(Path(module.__file__).resolve()==expected.resolve(),'actual normal production import path')
    fn=getattr(module,c.NAME)
    c.need(c.sha(fn.src)==c.FUNCTION_SHA and not fn.device_caches and not fn.pre_run_hooks,'fresh original production JIT')
    return fn


def driver_source():
    parent=source.adapted_driver()[0]
    return transform(parent,[
        ('import biv150_konly_multi2_hoist_source as c\nimport biv150_konly_multi2_numeric_contract as f\nfrom biv150_konly_multi2_hoist_source import kwargs\ncompiler=c.driver()',
         'import biv150_konly_multi2_service_numeric_source as bridge\nfrom biv150_konly_multi2_service_numeric_source import c, f\nkwargs=c.kwargs\ncompiler=bridge.production.driver()'),
        ('Private multi2 numeric screen; original math and no performance execution.',
         'Actual production multi2 cache-pair numeric screen; unchanged ce905 math, no service claim.'),
        ("(a.out/'actual-numeric-driver.py').write_text(f.source.adapted_driver()[0])","(a.out/'actual-numeric-driver.py').write_text(bridge.driver_source()[0])"),
        ('        import torch\n','        sys.path.insert(0,str(bridge.PRODUCTION_ROOT))\n        import torch\n'),
        ("'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'):", "'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST','VLLM_FL_BIV150_KONLY_SERVICE_MULTI2_HOIST'):"),
        ('        for role in f.ROLES:\n            bq=32;',
         '        fn=bridge.production_fn();prior_cache=None\n        for role in f.ROLES:\n            bq=32;'),
        ("            c.need(source.read_text()==c.candidate_source(),'original compile source '+role)",
         "            c.need(source==bridge.PRODUCTION_ROOT/'vllm_fl/patches/biv150_konly_encoded_multi2.py' and c.sha(source.read_bytes())==bridge.production.SOURCES['vllm_fl/patches/biv150_konly_encoded_multi2.py'],'original production source '+role)"),
        ("a.raw/'out/multi2/override.ttgir'","a.raw/'out'/bridge.BINDING/'override.ttgir'"),
        ("            module=c.load_file('cjj_private_multi2_compile_candidate',source)\n            fn=getattr(module,c.NAME);functions[role]=fn", "            functions[role]=fn"),
        ('lambda:fn.warmup(grid=grid,**calls[role],**launch_options[role]));handles[role]=kernel',
         'lambda:fn.warmup(grid=grid,**calls[role],**launch_options[role]),expected_memory_cache=prior_cache);handles[role]=kernel'),
        ("            result['records'][role]=dict(snapshot=snapshot,loaded=resources,actual_source_path=str(source),actual_source_sha256=c.sha(source.read_bytes()))\n            save()",
         "            result['records'][role]=dict(snapshot=snapshot,loaded=resources,actual_source_path=str(source),actual_source_sha256=c.sha(source.read_bytes()))\n"
         "            if role=='unhoisted':\n"
         "                prior_cache=dict(fn.device_caches[torch.cuda.current_device()][0])\n"
         "                c.need(len(prior_cache)==1 and next(iter(prior_cache.values())) is kernel,'first reviewed actual cache object')\n"
         "            save()\n"
         "        c.need(functions['unhoisted'] is functions['hoisted'] and len(fn.device_caches[torch.cuda.current_device()][0])==2,'one production JIT, exactly two reviewed objects')"),
    ])


def plan():
    production.plan()
    return dict(original=f.plan(),frontend_production_base=production.BASE,production_compile_commit=COMPILE,
        fixed_production_worktree=str(PRODUCTION_ROOT),driver_sha256=c.sha(driver_source()[0]),
        cache_loader_sha256=c.sha(cache_source()),driver_edits=len(driver_source()[1]),
        original_math_fixture_policy_unchanged=True,same_JIT_for_both_arms=True,
        cache_transition='empty -> exact verified unhoisted mapping -> same mapping plus exactly one verified hoisted object',
        new_backend_compilation_prohibited=True,new_service_prepare_executed=False,
        numerical_verified=False,Graph_verified=False,service_verified=False,GPU_executed=False)
