"""Private actual-afc/805a cache: unchanged JIT forms the ABI; no new shader compile."""
import base64,inspect,json,sys
from pathlib import Path
from contextlib import contextmanager
import compile_biv150_dynamic_hidden_gather as c
SUMMARY='29abcf5e6d5618f781da311304fe675e94bfb9e160eba0645a7282831da2bf5b'
REVIEW='39efd3769db7dfdc324ee37a05f726bb944a3b2420a8dbea0d615c2e599caa1f'
HASH='afcbb980637dd36ccf51c124652c41efc8de8f99f2bff21f7904beb586a51b58'
CUBIN='805a6fa835b60270ddbcd0c059afad9d44d553cfe80b1600e17c3cdbb5da0cce'
CACHE_SOURCE='faf84021196925228e20fdab44b3372c5ac40a57f653c212c0655e8fb97757d0'
EXTENSIONS=('source','ttir','ttgir','llir','asm','cubin','json')
SOURCE_ROOT='/workspace/experiments/minicpm-s2/code/cjj-dynamic-hidden-gather-helper-numeric-v1-20261029'
SOURCE='04fa5737c5ee95974c917235e9da237aac985dd5'

def proof(original,review):
 rb=review.read_bytes();c.need(c.sha(rb)==REVIEW,'separate Root-read805a static comparison')
 r=json.loads(rb);c.need(r['static_comparison_complete'] is True and r['summary_sha256']==SUMMARY and r['actual_hash']==HASH and r['actual_cubin_sha256']==CUBIN and r['typed_ABI_exact'] is True and r['all_loadable_sections_and_symbol_tables_exact'] is True,'actual static scope')
 b=(original/'helper-summary.json').read_bytes();c.need(c.sha(b)==SUMMARY,'original failed-before-launch helper summary')
 s=json.loads(b);c.need(s['source_commit']==SOURCE and s['completed'] is False and s['production_warmups_completed']==1 and s['numeric_candidate_launches_profile_observed']==0,'originalv1 unknown specialization refusal retained')
 a=s['records'][0]['actual_compiled'];c.need(a['hash']==HASH and a['files']['cubin']['sha256']==CUBIN and a['resources']==r['actual_resources'] and a['options']['use_sme']==5,'actual naturalSME5 binding')
 return s,a

def prepare(original,destination,review):
 """Only index child paths relocate: every original child byte is immutable."""
 summary,record=proof(original,review);members={x['path']:x for x in summary['artifacts']}
 def read(relative):
  p=original/relative;c.need(not p.is_symlink() and p.resolve().is_relative_to(original.resolve()),'reviewed cache path')
  row=members[relative];b=p.read_bytes();c.need(len(b)==row['bytes'] and c.sha(b)==row['sha256'],'reviewed artifact bytes '+relative);return b
 destination=destination.resolve();destination.mkdir() # fresh own cache only
 directory=base64.b32encode(bytes.fromhex(HASH)).decode().rstrip('=');name=record['options']['name'];prefix='triton-cache/'+directory+'/'
 index='__grp__'+name+'.json';old_index=read(prefix+index);children=json.loads(old_index)['child_paths']
 c.need(set(children)=={name+'.'+ext for ext in EXTENSIONS},'all7 reviewed cache children')
 target=destination/directory;target.mkdir();paths={};digests={}
 for filename,oldpath in children.items():
  c.need(Path(oldpath).name==filename and Path(oldpath).parent.name==directory,'original cache child role')
  b=read(prefix+filename);ext=filename.rsplit('.',1)[1]
  if ext in record['files']:c.need(c.sha(b)==record['files'][ext]['sha256'],'actual record/cache artifact')
  if ext=='ttir':c.need((SOURCE_ROOT+'/vllm_fl/patches/biv150_dynamic_hidden_gather_kernel.py').encode() in b,'actual original debug-source root')
  if ext=='json':c.need(json.loads(b)['hash']==HASH and json.loads(b)['use_sme']==5,'actual metadata natural5')
  p=target/filename;p.write_bytes(b);paths[filename]=str(p);digests[filename]=c.sha(b)
 (target/index).write_text(json.dumps({'child_paths':paths},sort_keys=True)+'\n')
 host=[]
 for relative in members:
  if relative.startswith('triton-cache/') and relative.endswith('.so'):
   b=read(relative);p=destination/Path(relative).relative_to('triton-cache');p.parent.mkdir(exist_ok=True);p.write_bytes(b);host.append(dict(path=str(p),sha256=c.sha(b)))
 return dict(hash=HASH,name=name,cache_root=str(destination),paths=paths,sha256=digests,host=host,
  original_summary_sha256=SUMMARY,original_source_root=SOURCE_ROOT,original_index_sha256=c.sha(old_index),
  relocated_index_sha256=c.sha((target/index).read_bytes()),actual_static_review_sha256=REVIEW)

def validate_group(manager,group):
 actual=manager.get_group(group['name']+'.json');c.need(actual==group['paths'],'reviewed group incomplete/missing/redirected')
 for name,path in actual.items():c.need(c.sha(Path(path).read_bytes())==group['sha256'][name],'reviewed child changed '+name)
 return actual

@contextmanager
def prohibit_backend_compile(backend_class):
 original=backend_class.__dict__['add_stages']
 def forbidden(*args,**kwargs):raise ValueError('actual805a cache miss: backend compilation prohibited')
 backend_class.add_stages=forbidden
 try:yield
 finally:backend_class.add_stages=original

@contextmanager
def prohibit_candidate_compile(fn):
 """After initial licensed load, any candidate memory-cache miss must stop.

 Other framework/control kernels are outside this per-JITFunction barrier.
 No change to run/_do_compile, natural SME selection, helper or launch options.
 """
 original=fn.compile
 def forbidden(*args,**kwargs):raise ValueError('actual805a candidate runtime-cache miss: compile prohibited')
 fn.compile=forbidden
 try:yield
 finally:fn.compile=original

def warmup(fn,device,group,call):
 # Same reviewed116 pattern: normal binder initializes fn.compile lazily;
 # unchanged _do_compile creates typed ASTSource and actual ordinary cache key.
 memory,_,_,backend,_=fn.device_caches[device]
 c.need(not memory,'initial reviewed load requires fresh candidate memory cache')
 compiler=sys.modules[fn.compile.__module__];base=Path('/usr/local/lib/python3.12/site-packages/triton')
 c.need(Path(inspect.getfile(compiler)).resolve()==(base/'spec/iluvatar/triton/compiler/compiler.py').resolve(),'actual compiler')
 c.need(Path(inspect.getfile(type(backend))).resolve()==(base/'backends/iluvatar/compiler.py').resolve(),'actual backend')
 cm=sys.modules[compiler.get_cache_manager.__module__];p=Path(inspect.getfile(cm)).resolve()
 c.need(p==(base/'runtime/cache.py').resolve() and c.sha(p.read_bytes())==CACHE_SOURCE,'actual cache implementation')
 c.need(sys.modules[type(fn).__module__]._async_compile.active_mode.get() is None,'async compilation prohibited')
 knobs=compiler.knobs;c.need(knobs.compilation.always_compile is False and knobs.compilation.override is False,'compiler force/override prohibited')
 c.need(knobs.cache.manager_class in (None,cm.FileCacheManager) and Path(knobs.cache.dir).resolve()==Path(group['cache_root']),'private standard cache manager/root')
 manager=compiler.get_cache_manager(HASH);expected=validate_group(manager,group)
 with prohibit_backend_compile(type(backend)):kernel=call()
 c.need(kernel.hash==HASH and kernel.metadata_group==expected,'loaded actual805a group identity')
 validate_group(manager,group)
 return kernel
