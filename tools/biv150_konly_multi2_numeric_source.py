"""Narrow, reversible reader-only adaptation of the executed PR116 component."""
import ast
import subprocess
import biv150_konly_multi2_hoist_source as c

DEPENDENCY = '522c0b0617a8d2d862f60ebb03f05f6e08ccfcfe'
PINS = {
    'probe_biv150_konly_q64_screen.py':'c186982bc964a1eef66c9d5edd228b6d13978b4a129a850bb38bdc03c293efea',
    'biv150_konly_q64_screen_contract.py':'912e485d189f611c571a4439e6356c72fd699c80a88eb8f8db92bea23a67ac31',
    'biv150_konly_q64_reviewed_cache.py':'9d38d2a69946e7ba9c8670714e732c9ba1cb5d04834e40b595bda04195812a1f',
}


def original(name):
    text=subprocess.check_output(['git','-C',str(c.ROOT),'show',DEPENDENCY+':tools/'+name]).decode()
    c.need(c.sha(text)==PINS[name],'executed original numeric helper '+name)
    return text


def inherited_contract():
    # Keep original loader/source checks and resource validation functions;
    # the calling contract supplies only this experiment's exact resource map.
    text=original('biv150_konly_q64_screen_contract.py');tree=ast.parse(text)
    names={'EXECUTION_SOURCES','LOADED_COMPILER','REFERENCES','REFERENCE_COMMIT','FIXTURE'}
    functions={'execution_sources','execution_snapshot','check_loaded'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in functions)]
    return ast.Module(body=nodes,type_ignores=[])


def cache_source():
    return c.transform(original('biv150_konly_q64_reviewed_cache.py'),(
        ('import biv150_konly_q64_compile_source as c','import biv150_konly_multi2_hoist_source as c'),))


def adapted_driver():
    parent=original('probe_biv150_konly_q64_screen.py');text=parent;edits=[]
    def edit(before,after):
        nonlocal text
        c.need(before!=after and text.count(before)==1,'unique numeric edit '+repr(before[:70]))
        text=text.replace(before,after,1);edits.append((before,after))
    def section(start,stop,after):
        a=text.index(start);b=text.index(stop,a);edit(text[a:b],after)
    edit('One isolated Q2048 screen; default CPU plan, explicit Root-only GPU execution.',
         'Private multi2 numeric screen; original math and no performance execution.')
    section('import biv150_konly_q64_compile_source as c','\n\ndef run(',
        'import biv150_konly_multi2_hoist_source as c\nimport biv150_konly_multi2_numeric_contract as f\nfrom biv150_konly_multi2_hoist_source import kwargs\ncompiler=c.driver()\nreviewed_cache=f.reviewed_cache\n')
    edit("compiler.git('diff',c.BASE,'--','vllm_fl','csrc')","compiler.git('diff',f.MAIN,'--','vllm_fl','csrc')")
    edit('    a.out.mkdir(parents=True)', "    a.out.mkdir(parents=True)\n    (a.out/'actual-numeric-driver.py').write_text(f.source.adapted_driver()[0])\n    (a.out/'actual-cache-loader.py').write_text(f.source.cache_source())")
    edit("(a.out/'reference-with-63-64.py')","(a.out/'original-reference-with-sequence-coordinates.py')")
    section("        backing={name:","        def interval(t):",'')
    edit("dict(tensors,cache=owner,writer_key=backing['key'],writer_value=backing['value'],slots=slots)","dict(tensors,cache=owner)")
    edit("guarded['q32'])!=interval(guarded['q64']","guarded['unhoisted'])!=interval(guarded['hoisted']")
    edit('output_q32=guarded[\'q32\'],output_q64=guarded[\'q64\']',"output_unhoisted=guarded['unhoisted'],output_hoisted=guarded['hoisted']")
    edit('stream=torch.cuda.Stream();graphs={}', 'stream=torch.cuda.Stream();side=torch.cuda.Stream();graphs={}')
    edit("result['streams']=dict(actual=stream.cuda_stream,default=torch.cuda.default_stream().cuda_stream)",
         "c.need(side.cuda_stream not in (stream.cuda_stream,torch.cuda.default_stream().cuda_stream),'distinct side C stream')\n        result['streams']=dict(capture=stream.cuda_stream,side_C=side.cuda_stream,default=torch.cuda.default_stream().cuda_stream)")
    edit("'cjj_q64_screen_snapshot'","'cjj_private_multi2_numeric_snapshot'")
    section('            bq=int(role[1:]);', '            pins=adapter.Adapter.installed_pins',
        "            bq=32;paths=pre['runtime_paths'][role];source=Path(paths['source']);override=paths['override']\n"
        "            c.need(source.read_text()==c.candidate_source(),'original compile source '+role)\n"
        "            if override is not None:c.need(Path(override).read_bytes()==(a.raw/'out/multi2/override.ttgir').read_bytes(),'original override path/content')\n"
        "            module=c.load_file('cjj_private_multi2_compile_candidate',source)\n"
        "            fn=getattr(module,c.NAME);functions[role]=fn\n")
    edit('dict(profile=role,sequences=1,segments=1,config=config)',"dict(profile='multi2',sequences=2,segments=1,config=config)")
    edit('dict(num_warps=16,num_stages=1,ir_override=str(override))',"dict(num_warps=16,num_stages=1,**({'ir_override':override} if override is not None else {}))")
    section("        writer_module=c.load_file(","        def byte_cpu(t):",'')
    edit(",writer_key=backing['key'],writer_value=backing['value'],slots=slots",'')
    section('            if mode==\'append-reader\':','        def replay(', '            reader(role)\n')
    edit('inputs={};references={};gold_outputs={};uploaded={};append_poison={}', 'inputs={};references={};gold_outputs={};uploaded={}')
    section('        for pattern in (0,1):',"            owner.copy_(uploaded[pattern])",
        "        for pattern,(qlens,klens) in enumerate(f.PHASES):\n"
        "            cpu=fixture.cpu_inputs(qlens,klens,20261025,pattern)\n"
        "            c.need(cpu['packed'].shape[0]==773 and cpu['q'].shape==(2048,16,128),'fixed complete fixture geometry')\n"
        "            encoded,codes,scales,codec=reference.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)\n"
        "            refs=f.references(reference,fixture,cpu,codes,scales,qlens,klens,pattern,torch,np)\n"
        "            uploaded[pattern]=torch.from_numpy(np.frombuffer(encoded,dtype=np.uint8).copy())\n")
    section('            for name in backing:',"            c.need(byte_cpu(owner)==encoded",'            torch.cuda.synchronize()\n')
    edit(",writer_key=source_cpu['key'],writer_value=source_cpu['value'],slots=slot_cpu",'')
    edit(',append_poison_bytes=append_poison[pattern]','')
    edit(',writer_sources=source_cpu,writer_slots=slot_cpu',',qlens=qlens,klens=klens')
    edit("actual_roles='hoisted Q32 and hoisted Q64'","actual_roles='actual multi2 unhoisted and hoisted'")
    # Two occurrences are individually anchored to preserve the exact inverse.
    edit('owner.copy_(append_poison[pattern] if mode==\'append-reader\' else uploaded[pattern])','owner.copy_(uploaded[pattern])')
    edit('byte_cpu(append_poison[pattern] if mode==\'append-reader\' else uploaded[pattern])','byte_cpu(uploaded[pattern])')
    edit('                    with torch.cuda.stream(stream):\n                        if use_graph:',
         "                    execution_stream=side if label=='side-C' else stream\n                    record['actual_stream']=execution_stream.cuda_stream\n                    with torch.cuda.stream(execution_stream):\n                        if use_graph:")
    edit('                    stream.synchronize();after=snapshot(guarded[role]);','                    execution_stream.synchronize();after=snapshot(guarded[role]);')
    edit("byte_cpu(outputs['q32'])==byte_cpu(outputs['q64'])","byte_cpu(outputs['unhoisted'])==byte_cpu(outputs['hoisted'])")
    edit("'actual hoisted Q32 versus actual hoisted Q64'","'actual multi2 unhoisted versus actual multi2 hoisted'")
    edit('            else:\n                # Both actual-handle graphs consume B', '            elif pattern==1:\n                # Both actual-handle graphs consume B')
    edit('                result.update(numeric_verified=True,Graph_verified=True,side_stream_verified=True)',"                result['Graph_B_stage_completed']=True")
    section("            c.need(all(byte_cpu(gold_outputs[pattern,'reader']",'        save();return result',
        "            else:\n"
        "                gold_outputs[2,'reader']=stage('side-C','reader')\n"
        "                c.need(all(byte_cpu(gold_outputs[2,'reader'][r])!=byte_cpu(gold_outputs[1,'reader'][r]) for r in f.ROLES),'changed-input side C counterexample absent')\n"
        "        result.update(completed=True,numeric_verified=True,Graph_verified=True,side_stream_verified=True,performance_measured=False,precision_gate=False,service_verified=False,\n"
        "            original_BF16_quality_passed=all(r['original_BF16_quality_passed'] for r in result['stages']),input_archives=inputs,\n"
        "            timing_scope='not run; future direct unhoisted/hoisted reader ABBA must include Q quantization/staging/QK/softmax/PV/epilogue')\n")
    # Deletions make a reverse find ambiguous; verify each ordered before/after
    # against the pinned parent, and return explicit positions for exact undo.
    current=parent;positioned=[]
    for before,after in edits:
        at=current.index(before);positioned.append(dict(offset=at,before=before,after=after));current=current[:at]+after+current[at+len(before):]
    c.need(current==text,'ordered numeric transformation')
    for e in reversed(positioned):
        at=e['offset'];c.need(current[at:at+len(e['after'])]==e['after'],'numeric inverse location')
        current=current[:at]+e['before']+current[at+len(e['after']):]
    c.need(current==parent,'exact inverse including removed writer/timing')
    return text,positioned
