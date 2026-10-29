"""Narrow CPU proof of timing suffix/diagnostic boundary; never starts transport."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import biv150_konly_multi2_timing_source as t
c=t.c;f=t.f

def main(args):
    checks={}
    def ok(name,value):c.need(value,name);checks[name]=True
    def bad(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError):checks[name]=True;return
        raise AssertionError(name+' accepted')
    original_popen=subprocess.Popen
    def local_git(argv,*args,**kwargs):
        c.need(isinstance(argv,list) and argv[:2]==['git','-C'] and argv[3] in ('show','diff','rev-parse'),'CPU local Git only')
        return original_popen(argv,*args,**kwargs)
    with patch.object(subprocess,'Popen',local_git),tempfile.TemporaryDirectory(prefix='cjj-multi2-timing-cpu-') as temp:
        code,edits=t.adapted_driver();ast.parse(code);restored=code
        for i,e in enumerate(reversed(edits)):
            at=e['offset'];ok('inverse_'+str(i),restored[at:at+len(e['after'])]==e['after']);restored=restored[:at]+e['before']+restored[at+len(e['after']):]
        original=f.source.adapted_driver()[0];ok('exact_d39_numeric_driver_inverse',restored==original)
        suffix,proof=t.timing_suffix();back=suffix
        for a,b in reversed(proof['edits']):back=back.replace(b,a)
        ok('original116_timing_suffix_exact',c.sha(back)==proof['original_sha256'])
        ok('same_capture4_warm8_timed16_divisor64',code.count('for _ in range(4):step(role,mode)')==1 and suffix.count('for _ in range(8):replay(role,mode)')==1 and suffix.count('for _ in range(16):replay(role,mode)')==1 and 'begin.elapsed_time(end)/64' in suffix and 'wall*1000/64' in suffix)
        ok('direct_same_input_ABBA',"enumerate(('unhoisted','hoisted','hoisted','unhoisted'))" in suffix)
        getstage=lambda source:next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='stage')
        ok('whole_numeric_stage_AST_unchanged',ast.dump(getstage(code))==ast.dump(getstage(original)))
        for name in ('reader','step','replay','fingerprints','snapshot'):
            a=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in ast.walk(ast.parse(original)) if isinstance(n,ast.FunctionDef) and n.name==name)
            ok('same_function_'+name,ast.dump(a)==ast.dump(b))
        ok('Graph_B_before_fresh_B_before_side_C_before_restore_before_timing',code.index("stage('Graph-B-first'")<code.index("stage('eager-B-after-Graph'")<code.index("stage('side-C'")<code.index('cpu=phase_cpu[1]')<code.index('for _ in range(8):replay'))
        ok('restore_all_six_readonly_and_whole_cache',"tensors[dest].copy_(cpu[source])\n        torch.cuda.synchronize();immutable=phase_readonly[1]" in code and "fingerprints()==immutable and byte_cpu(owner)==byte_cpu(uploaded[1])" in code)
        ok('no_writer',all(v not in code for v in ('writer.warmup','writer[','append_poison','writer_module=')))
        ok('compiled_source_and_loader_not_changed',not subprocess.check_output(['git','-C',str(c.ROOT),'diff',t.NUMERIC_SOURCE,'--','tools/biv150_konly_multi2_hoist_source.py','tools/biv150_konly_multi2_numeric_source.py','tools/biv150_konly_multi2_numeric_contract.py']))
        ok('production_main899_unchanged',not subprocess.check_output(['git','-C',str(c.ROOT),'diff',f.MAIN,'--','vllm_fl','csrc']))
        with patch.object(t,'NUMERIC_REVIEW',None),patch.object(Path,'read_bytes',side_effect=AssertionError('pending must not read a receipt')):
            bad('pending_numeric_admission_rejects_before_reads',lambda:t.numeric_admission(None,None))
        for name,value in (('NUMERIC_MANIFEST',True),('NUMERIC_SUMMARY','x'*64),('NUMERIC_REVIEW','0'*63)):
            with patch.object(t,name,value):bad('reject_invalid_pin_'+name,lambda:t.numeric_admission(None,None))
        ok('numeric_admission_before_original_preflight_and_run',code.index('a.numeric_proof=timing.numeric_admission')<code.index('pre=f.preflight')<code.index('if a.gpu:value=run'))
        ok('explicit_diagnostic_before_output_or_framework',code.index("c.need(a.numeric_diagnostic,'reader timing requires explicit numeric diagnostic')")<code.index('a.out.mkdir')<code.index('import numpy as np'))
        folder=Path(temp)
        for name,pin in f.REFERENCES.items():
            b=subprocess.check_output(['git','-C',str(c.ROOT),'show',f.REFERENCE_COMMIT+':tools/'+name]);ok('original_reference_'+name,c.sha(b)==pin);(folder/name).write_bytes(b)
        _,policy,_=f.load_reference(folder);hard={n:True for n in ('finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference')}
        for name in hard:bad('diagnostic_rejects_hard_'+name,lambda name=name:policy(dict(hard,**{name:False}),False,True))
        bad('quality_failure_requires_diagnostic',lambda:policy(hard,False,False))
        admitted=policy(hard,False,True);ok('original_quality_failure_remains_false',admitted['original_BF16_quality_passed'] is False and admitted['precision_gate'] is False and admitted['service_verified'] is False)
        ok('original_quality_and_model_flags_preserved',"original_BF16_quality_passed=all(r['original_BF16_quality_passed'] for r in result['stages'])" in code and 'performance_measured=True,precision_gate=False,service_verified=False' in code)
        ok('actual_pin_format_does_not_grant_device_or_model',t.plan()['actual_numeric_admission_pinned'] is True and all(t.plan()[k] is False for k in ('GPU_executed','performance_measured','precision_gate','service_verified')))
        actual=None
        if args.numeric_raw is not None or args.numeric_review is not None:
            c.need(args.numeric_raw is not None and args.numeric_review is not None,'both actual numeric inputs required')
            actual=t.numeric_admission(args.numeric_raw,args.numeric_review)
            ok('actual_complete_diagnostic_admission',actual['hard_numeric_Graph_side_verified'] is True and actual['original_BF16_quality_passed'] is False and actual['precision_gate'] is False)
        ok('no_framework_import',not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules))
    return dict(completed=True,errors=[],check_count=len(checks),checks=checks,plan=t.plan(),source_edits=len(edits),driver_sha256=c.sha(code),timing_suffix=proof,CPU_only=True,GPU_executed=False,actual_numeric_admission_verified=actual is not None,actual_numeric_admission=actual,source_root=str(c.ROOT))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--numeric-raw',type=Path);p.add_argument('--numeric-review',type=Path)
    print(json.dumps(main(p.parse_args()),sort_keys=True,indent=2))
