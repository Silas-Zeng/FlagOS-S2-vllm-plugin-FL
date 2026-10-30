"""Readonly admission of the failed-smoke actual SME1 group; no new qualification."""
import base64,json,types
from pathlib import Path
import biv150_dynamic_hidden_gather_reviewed_cache as prior
c=prior.c
SOURCE='8b7ea7a876f01e059f35fa7e2f266ba248af03ef'
REVIEW='5f3f7f734d5ea01e7b82c084b79b4b3e1309cf90c1ce1968df5d27962f23694a'
HASH='b0fa4a52f87904075643190bf02047cdd59bd288aff4a1c1cd62a6416ed0976f'
CUBIN='4a9f93b84bd32f725d101ffb15f081f6b7345af149ba64abcc4e117369ef928e'
REPORTED_MODULE_SHA='4105509e003b73dd782a52edfb86254c751201e42485c09b757cfb1fa7ad9c59'
LAUNCHER='512414b6fbfae2135b62153619ef1f5919bd2fd07018058cd45035de1d83e361'
DIRECTORY=base64.b32encode(bytes.fromhex(HASH)).decode().rstrip('=')
HANDLER='H7POXA2W3X5GQEOXOQQOM2VCB5S6HLHVZLJF3SAXLAJPDJFQOIOA/__triton_launcher.cpython-312-x86_64-linux-gnu.so'
NAME='gather_hidden_bits'
ROOT='/workspace/experiments/minicpm-s2/code/cjj-native122-service-smoke-v1-20261030'
PINS={
 'state.json':'bac1437c7d8330b7ea9bf6165156bb0523d5d34bb26195c0e68190c284ff83c7',
 'gather-smoke-snapshots/smoke-before.json':'cb5147de1285938445e4ab199035bd7fefb7762252fb46d91286f4bee550f555',
 'gather-smoke-snapshots/smoke-after.json':'cd5f1c5b0914c0ef2ec677640d08a99af9face841658518cb7950683a46b3ab8',
 'gather-smoke-snapshots/index.json':'f0d9c3f58216a62cad49390782673ab3556e7dba3f9b2febd6485de71f8834e9',
 'root-source-bridge.json':'451d7a5858141049e61317fb2c5e3c75c6a92df432e37cb4168a370c2f2bc27c',
 'triton-cache/'+DIRECTORY+'/gather_hidden_bits.json':'335b11ba830c40e6ab53650aad16d54b17b28ca5c5413252a74d5318990df590',
 'triton-cache/'+DIRECTORY+'/__grp__gather_hidden_bits.json':'7bc1e07e00bf2ccdfe1311705bd17a170775e5c81c41ede458579f8044863cb7',
 'triton-cache/'+HANDLER:LAUNCHER,
 'triton-cache/BWNUEQYDUA2UCX5OVOXXXGXFGKQMTEK6QO3ONMXK6GVLAZSEISPQ/__triton_launcher.cpython-312-x86_64-linux-gnu.so':REPORTED_MODULE_SHA}

def read(root,relative,digest):
 p=root/relative;c.need(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'contained regular original '+relative)
 b=p.read_bytes();c.need(c.sha(b)==digest,'actual service object SHA '+relative);return b

def proof(original,review):
 c.need(c.sha(review.read_bytes())==REVIEW,'Root-replayed71-check failed-smoke review')
 r=json.loads(review.read_bytes());c.need(r['evidence_review_passed'] is True and r['check_count']==71 and r['source_commit']==SOURCE and r['actual_new_numeric_verified'] is False and r['service_verified'] is False,'review boundary; new numeric remains unknown')
 original=original.resolve();data={k:read(original,k,h) for k,h in PINS.items()}
 state=json.loads(data['state.json']);c.need(state['completed'] is False and state['smoke_passed'] is False and state['cleanup_complete'] is True,'preserve original failed service')
 ack=json.loads(data['gather-smoke-snapshots/smoke-after.json']);before=json.loads(data['gather-smoke-snapshots/smoke-before.json'])
 c.need((ack['pid'],ack['start_ticks'])==(215193,364667701)==(before['pid'],before['start_ticks']) and ack['compiled_overflow']==0 and len(ack['compiled'])==1,'same actual worker and one object')
 a=ack['compiled'][0];c.need(a['hash']==HASH and a['launcher_sha256']==REPORTED_MODULE_SHA and a['files']['cubin']['sha256']==CUBIN and a['options']['use_sme']==1,'actual SME1 object, no inheritance from5')
 for ext,item in a['files'].items():
  b=read(original,'triton-cache/'+DIRECTORY+'/'+NAME+'.'+ext,item['sha256']);c.need(len(b)==item['bytes'],'actual child length')
 return dict(source_commit=SOURCE,original_failed_state=state,review_sha256=REVIEW,actual_ACK=ack),a

def prepare(original,destination,review):
 summary,record=proof(original,review);destination=destination.resolve()
 c.need(not destination.exists() and not destination.is_relative_to(original.resolve()),'new private cache outside readonly original')
 destination.mkdir();target=destination/DIRECTORY;target.mkdir();prefix='triton-cache/'+DIRECTORY+'/'
 oldindex=read(original,prefix+'__grp__'+NAME+'.json',PINS[prefix+'__grp__'+NAME+'.json'])
 children=json.loads(oldindex)['child_paths'];c.need(set(children)=={NAME+'.'+x for x in prior.EXTENSIONS},'exact7 children')
 paths={};digests={}
 for filename,oldpath in children.items():
  c.need(Path(oldpath).name==filename and Path(oldpath).parent.name==DIRECTORY,'original group child identity')
  ext=filename.rsplit('.',1)[1];digest=PINS[prefix+filename] if ext=='json' else record['files'][ext]['sha256']
  b=read(original,prefix+filename,digest);p=target/filename;p.write_bytes(b);paths[filename]=str(p);digests[filename]=digest
 index=target/('__grp__'+NAME+'.json');index.write_text(json.dumps({'child_paths':paths},sort_keys=True)+'\n')
 p=destination/HANDLER;p.parent.mkdir();p.write_bytes(read(original,'triton-cache/'+HANDLER,LAUNCHER))
 return dict(hash=HASH,name=NAME,cache_root=str(destination),paths=paths,sha256=digests,host=[dict(path=str(p),sha256=LAUNCHER)],original_source_root=ROOT,original_failed_state_sha256=PINS['state.json'],actual_evidence_review_sha256=REVIEW,original_index_sha256=c.sha(oldindex),relocated_index_sha256=c.sha(index.read_bytes()))

# Original barriers/standard cache manager/force+override checks are unchanged.
# The original function bytecode is reused, with only its expected HASH replaced.
validate_group=prior.validate_group
prohibit_candidate_compile=prior.prohibit_candidate_compile

def warmup(fn,device,group,call):
 namespace=dict(vars(prior),HASH=HASH)
 return types.FunctionType(prior.warmup.__code__,namespace)(fn,device,group,call)
