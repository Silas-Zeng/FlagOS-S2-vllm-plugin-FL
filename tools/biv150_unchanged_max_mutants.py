"""One-change negative replays of frozen real IR; no compiler/GPU execution."""
import hashlib
import re

from biv150_unchanged_max_ir import (audit_asm, audit_llir, audit_ttgir,
                                    pv_source_sites, source_location)


def run_mutants(artifacts, proof):
    results = []

    def change(text, old, new):
        assert old in text, 'Mutation anchor missing: '+old
        assert old != new
        return text.replace(old,new,1)

    def check(stage, label, text):
        assert text != artifacts[stage], 'Mutation did not change bytes'
        auditor = dict(ttgir=audit_ttgir, llir=lambda t:audit_llir(t,proof), asm=lambda t:audit_asm(t,proof))[stage]
        try:
            auditor(text)
        except ValueError as error:
            results.append(dict(stage=stage, label=label, rejected=True, reason=str(error),
                                mutant_sha256=hashlib.sha256(text.encode()).hexdigest()))
        else:
            raise AssertionError('Mutant was accepted: '+stage+'/'+label)

    tt, ll, asm = (artifacts[k] for k in ('ttgir','llir','asm'))
    check('ttgir','missing-gate',change(tt,'%12:2 = scf.if','%12:2 = scf.unknown'))
    check('ttgir','wrong-full-result-selector',change(tt,'%12#1, inputPrecision','%12#0, inputPrecision'))
    check('ttgir','wrong-tail-result-selector',change(tt,'%running_sum#1, inputPrecision','%running_sum#0, inputPrecision'))
    check('ttgir','wrong-tail-old-anchor-selector',change(tt,
          '%rescale = arith.subf %accumulator#0, %running_max',
          '%rescale = arith.subf %accumulator#1, %running_max'))
    check('ttgir','fma-missing',change(tt,' = math.fma ',' = arith.addf '))
    check('ttgir','p-newmax-changed',change(tt,'%new_max_125 = arith.maxnumf ','%new_max_125 = arith.minnumf '))
    check('ttgir','skip-changes-c',change(tt,'scf.yield %running_sum, %accumulator_99 :',
          'scf.yield %running_sum, %accumulator_141 :'))
    check('ttgir','sum-and-pv-use-different-p',change(tt,'"tt.reduce"(%probability_128)',
          '"tt.reduce"(%probability_127)'))
    alpha_line = next(line for line in tt.splitlines() if '%rescale_138 = math.exp2' in line)
    hoisted = change(tt,alpha_line+'\n','')
    gate_line = next(line for line in tt.splitlines() if '%12:2 = scf.if' in line)
    check('ttgir','alpha-outside-branch',change(hoisted,gate_line,alpha_line+'\n'+gate_line))

    check('llir','missing-gate',change(ll,'br i1 %570, label %571, label %610','br label %571'))
    check('llir','wrong-join',change(ll,'br label %613, !dbg !76','br label %610, !dbg !76'))
    check('llir','fma-missing',change(ll,'@llvm.fma.f32(float %282, float %574, float %539)',
          '@not_a_fma(float %282, float %574, float %539)'))
    check('llir','wrong-c-phi-old-state',change(ll,'phi float [ %250, %610 ], [ %578, %571 ]',
          'phi float [ %251, %610 ], [ %578, %571 ]'))
    check('llir','wrong-c-alpha-component',change(ll,'fmul float %250, %574','fmul float %250, %575'))
    check('llir','skip-denominator-changed',change(ll,'fadd float %282, %539','fadd float %283, %539'))
    # Move the actual alpha instruction to the parent block; its source site/count remains intact.
    alpha_line = next(line for line in ll.splitlines() if '%574 = ' in line)
    hoisted = change(ll,alpha_line+'\n','')
    gate_line = next(line for line in ll.splitlines() if 'br i1 %570, label %571, label %610' in line)
    check('llir','alpha-outside-branch',change(hoisted,gate_line,alpha_line+'\n'+gate_line))
    locations, pv_sites = source_location(ll), pv_source_sites(proof)
    pv_line = next(line for line in ll.splitlines() if 'call ' in line and
                   '@llvm.bi.matrix.mad.f32x4.bf16x4' in line and
                   locations[int(re.search(r'!dbg !(\d+)',line)[1])] in pv_sites)
    c_arg = re.search(r', (<4 x float> '+r'%[\w.$-]+'+r')\)',pv_line)[1]
    check('llir','pv-zero-c',change(ll,pv_line,pv_line.replace(c_arg,'<4 x float> zeroinitializer')))
    pv_result = re.search(r'(%[\w.$-]+) = ',pv_line)[1]
    extracted_line = next(line for line in ll.splitlines() if
                          'extractelement <4 x float> '+pv_result+', i64 0' in line)
    extracted = re.search(r'(%[\w.$-]+) = ',extracted_line)[1]
    added = '  %postadd77 = fadd float '+extracted+', 0.000000e+00'
    post = change(ll,extracted_line,extracted_line+'\n'+added)
    old_incoming = '[ '+extracted+', %613 ]'
    check('llir','pv-post-add',change(post,old_incoming,'[ %postadd77, %613 ]'))
    for label, sites in [('qk-debug-site',proof['source_lines']['qk']),
                         ('outside-pv-call-site',[s['end']+1 for s in proof['pv_call_spans']])]:
        changed = ll
        for span,site in zip(proof['pv_call_spans'],sites):
            for line in range(span['start'],span['end']+1):
                changed = changed.replace('!DILocation(line: '+str(line)+',','!DILocation(line: '+str(site)+',')
        check('llir',label,changed)

    check('asm','missing-gate',change(asm,'sl_cbr_tmskaz BB0_3','sl_nop BB0_3'))
    check('asm','wrong-backward-join',change(asm,'sl_jump BB0_3','sl_jump BB0_8'))
    check('asm','wrong-tail-fallthrough',change(asm,'BB0_73:', '\tsl_jump BB0_74\nBB0_73:'))
    check('asm','fma-missing',change(asm,'ml_fma_f32','ml_add_f32'))
    check('asm','duplicate-c-scale',change(asm,'ml_mul_f32 v2, v2, v47','ml_mul_f32 v0, v0, v47'))
    check('asm','wrong-c-alpha-component',change(asm,'ml_mul_f32 v0, v0, v47','ml_mul_f32 v0, v0, v76'))
    check('asm','missing-mask-restore',change(asm,'sl_or_b64 {tmsk, wcr}, tmsk, s[12:13]',
          'sl_or_b64 {tmsk, wcr}, tmsk, s[14:15]'))
    check('asm','wrong-mask-complement',change(asm,'sl_xor_savetmsk_b64 {s[14:15], wcr}, s[12:13], tmsk',
          'sl_xor_savetmsk_b64 {s[14:15], wcr}, s[10:11], tmsk'))
    check('asm','unbound-loc0-gate',change(asm,'.loc\t1 97 11','.loc\t1 0 11'))
    check('asm','skip-changes-c',change(asm,'ml_add_f32 v65, v65, v79',
          'ml_mul_f32 v0, v0, v47\n\tml_add_f32 v65, v65, v79'))
    check('asm','nested-correction-branch',change(asm,'ml_sub_f32 v47, v47, v70',
          'sl_cbr_tmskaz BB0_3\n\tml_sub_f32 v47, v47, v70'))
    alpha_line = '\tml_exp_f32 v47, v47\n'
    hoisted = change(asm,alpha_line,'')
    check('asm','alpha-outside-branch',change(hoisted,'BB0_4:',
          '\t.loc\t1 98 35\n'+alpha_line+'BB0_4:'))
    check('asm','pv-c-register-changed',change(asm,
          'v[0:3], v[80:81], v[76:77], v[0:3]', 'v[0:3], v[80:81], v[76:77], v[4:7]'))
    for label,sites in [('qk-debug-site',proof['source_lines']['qk']),
                        ('outside-pv-call-site',[s['end']+1 for s in proof['pv_call_spans']])]:
        changed = asm
        for span,site in zip(proof['pv_call_spans'],sites):
            for line in range(span['start'],span['end']+1):
                changed = changed.replace('.loc\t1 '+str(line)+' ','.loc\t1 '+str(site)+' ')
        check('asm',label,changed)
    return results
