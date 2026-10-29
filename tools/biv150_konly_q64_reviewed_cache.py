"""Private reviewed cache only; unchanged JIT creates the typed ASTSource/ABI."""
import base64
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys
import biv150_konly_q64_compile_source as c

CACHE_SOURCE = 'faf84021196925228e20fdab44b3372c5ac40a57f653c212c0655e8fb97757d0'
EXTENSIONS = ('source','ttir','ttgir','llir','asm','cubin','json')


def prepare(raw, destination, records, manifest_sha):
    """All seven original children stay byte-exact; only the group paths move."""
    data=(raw/'download-manifest.json').read_bytes()
    c.need(c.sha(data)==manifest_sha,'reviewed cache manifest')
    members={row['relative']:row for row in json.loads(data)['files']}
    def original(path):
        c.need(path.resolve().is_relative_to(raw.resolve()),'cache member escapes raw')
        row=members[path.relative_to(raw).as_posix()];body=path.read_bytes()
        c.need(len(body)==row['bytes'] and c.sha(body)==row['sha256'],'reviewed cache member changed')
        return body
    destination=destination.resolve();destination.mkdir()  # never overwrite a prior run/cache
    result={}
    for role,record in records.items():
        directory=base64.b32encode(bytes.fromhex(record['hash'])).decode().rstrip('=')
        source=raw/'out/triton-cache'/directory;target=destination/directory;target.mkdir()
        name=record['function_name'];index='__grp__'+name+'.json'
        old_index=original(source/index);children=json.loads(old_index)['child_paths']
        c.need(set(children)=={name+'.'+ext for ext in EXTENSIONS},'complete reviewed seven-file group')
        relocated={};digests={}
        for filename,oldpath in children.items():
            c.need(Path(oldpath).name==filename and Path(oldpath).parent.name==directory,'original cache group path')
            body=original(source/filename);ext=filename.rsplit('.',1)[1]
            if ext in record['artifacts']:c.need(c.sha(body)==record['artifacts'][ext],'reviewed record/cache artifact mismatch')
            if ext=='json':c.need(json.loads(body)['hash']==record['hash'],'reviewed metadata hash')
            out=target/filename;out.write_bytes(body);relocated[filename]=str(out);digests[filename]=c.sha(body)
        (target/index).write_text(json.dumps({'child_paths':relocated},sort_keys=True)+'\n')
        result[role]=dict(hash=record['hash'],name=name,cache_root=str(destination),paths=relocated,sha256=digests,
            original_index_sha256=c.sha(old_index),relocated_index_sha256=c.sha((target/index).read_bytes()))
    return result


def validate_group(manager, group):
    actual=manager.get_group(group['name']+'.json')
    c.need(actual==group['paths'],'private cache group missing, incomplete or redirected')
    for name,path in actual.items():
        c.need(c.sha(Path(path).read_bytes())==group['sha256'][name],'private reviewed cache bytes changed')
    return actual


def validate_knobs(knobs, cache_module, group):
    c.need(knobs.compilation.always_compile is False,'always_compile prohibited for reviewed cache')
    c.need(knobs.compilation.override is False,'global IR override prohibited; original pinned ir_override remains required')
    c.need(knobs.cache.manager_class in (None,cache_module.FileCacheManager),'custom/remote cache manager prohibited')
    c.need(Path(knobs.cache.dir).resolve()==Path(group['cache_root']),'actual cache root differs')


@contextmanager
def prohibit_backend_compile(backend_class):
    original=backend_class.__dict__['add_stages']
    def forbidden(*args,**kwargs):
        raise ValueError('reviewed reader cache miss: new backend compilation prohibited')
    backend_class.add_stages=forbidden
    try:yield
    finally:backend_class.add_stages=original


def warmup(fn, device, group, call):
    # Actual JIT creates its ordinary binder/backend before _do_compile creates
    # ASTSource. No hand-built source, serialized ABI reconstruction or cubin swap.
    memory_cache,_,_,backend,_=fn.device_caches[device]
    c.need(not memory_cache,'reader already compiled in runtime cache')
    compiler=sys.modules[fn.compile.__module__]
    base=Path('/usr/local/lib/python3.12/site-packages/triton')
    c.need(Path(inspect.getfile(compiler)).resolve()==(base/'spec/iluvatar/triton/compiler/compiler.py').resolve(),'actual compiler module')
    c.need(Path(inspect.getfile(type(backend))).resolve()==(base/'backends/iluvatar/compiler.py').resolve(),'actual backend class')
    cache_module=sys.modules[compiler.get_cache_manager.__module__]
    cache_path=Path(inspect.getfile(cache_module)).resolve()
    c.need(cache_path==(base/'runtime/cache.py').resolve() and c.sha(cache_path.read_bytes())==CACHE_SOURCE,'actual cache implementation')
    c.need(sys.modules[type(fn).__module__]._async_compile.active_mode.get() is None,'async reader compilation prohibited')
    validate_knobs(compiler.knobs,cache_module,group)
    manager=compiler.get_cache_manager(group['hash']);expected=validate_group(manager,group)
    # Pinned compiler returns on hit at 301–312; add_stages is only at 324,
    # before context/IR creation. A changed key cannot silently compile anew.
    with prohibit_backend_compile(type(backend)):
        kernel=call()
    c.need(kernel.hash==group['hash'] and kernel.metadata_group==expected,'loaded original cache/metadata binding')
    validate_group(manager,group)
    return kernel
