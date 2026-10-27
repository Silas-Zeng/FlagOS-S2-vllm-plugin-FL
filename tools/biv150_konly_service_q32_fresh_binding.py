"""Strict CPU binding of fresh service Q32 compiler artifacts; no old hash licence."""
import json
from pathlib import Path
import biv150_konly_service_q32_hoist as h
import biv150_konly_service_q32_cache as cache
import biv150_konly_service_q32_native as native

ROLES=('unhoisted','proposed-hoist')
EXTS=('ttir','ttgir','llir','asm','cubin')

def same(first,second):return json.dumps(first,sort_keys=True,separators=(',',':'))==json.dumps(second,sort_keys=True,separators=(',',':'))


def load_compile(folder):
    folder=Path(folder);files,meta,override,proof,source=cache.original()
    summary=json.loads((folder/'fresh-compile-summary.json').read_text())
    h.need(summary['completed'] is True and summary['compile_passed'] is True and summary['errors']==[], 'fresh compile did not pass')
    h.need(summary['kernel_launches']==0 and summary['encoder_launches']==0 and summary['JIT_warmups_attempted']==2,'zero-launch compiler protocol changed')
    h.need(summary['reconstructed_unhoisted_matches_runtime_cache'] is True and summary['source_provenance']==proof['provenance'],'service reconstruction provenance changed')
    h.need((folder/'fresh-service-override.ttgir').read_text()==override and json.loads((folder/'proposal.json').read_text())==proof,'fresh proposal changed')
    records={};actual={}
    for role in ROLES:
        directory=folder/role;r=json.loads((directory/'record.json').read_text());records[role]=r
        h.need(set(EXTS)<=set(r['files']),'actual compiler artifacts incomplete')
        actual[role]={ext:(directory/('actual.'+ext)).read_bytes() for ext in r['files']}
        for ext,value in actual[role].items():h.need(r['files'][ext]==dict(bytes=len(value),sha256=h.sha(value)),'actual artifact hash '+role+'/'+ext)
        text=(directory/'actual.jit_python_source.py').read_text()
        h.need(h.sha(text)==r['source_digest']==meta['source_digest'],'live JIT source digest changed')
        h.need(r['phase']=='prelaunch_compile_only' and r['loaded_resources_verified'] is False,'prelaunch resources mislabelled')
        h.need(same(r,summary['records'][ROLES.index(role)]),'summary record mirror changed')
    before,after=(records[r] for r in ROLES)
    h.need(before['hash']==meta['kernel_hash'],'unhoisted service-selected hash changed')
    for ext in ('ttgir','llir'):h.need(actual['unhoisted'][ext]==files['kernel_cjj_konly_q32.'+ext],'unhoisted original service IR changed')
    h.need(all(same(before[k],after[k]) for k in ('signature','constants','attrs','source_digest')),'source/typed ABI changed beyond override')
    h.need(actual['unhoisted']['ttir']==actual['proposed-hoist']['ttir'],'source TTIR changed beyond override')
    h.grammar(actual['unhoisted']['ttgir'].decode(),meta['function_name'])
    h.grammar(actual['proposed-hoist']['ttgir'].decode(),meta['function_name'],hoisted=True)
    audit=native.audit(override,*(actual['proposed-hoist'][e].decode() for e in ('ttgir','llir','asm')))
    baseline=native.llvm_observation(actual['unhoisted']['llir'].decode())
    h.need(len(baseline['Q_shared_reads_inside_KV'])==16,'unhoisted Q operand placement changed')
    return dict(summary=summary,metadata=meta,records=records,actual=actual,proposal=proof,override=override,
        override_path=str((folder/'fresh-service-override.ttgir').resolve()),source=source,
        audit=audit,baseline_LLVM=baseline)


def verify_recompile(record,expected,observed,reviewed):
    for field in ('hash','signature','constants','attrs','source_digest','metadata'):
        h.need(same(record[field],expected[field]),'fresh numeric recompile '+field+' differs')
    for ext in ('ttir','ttgir','llir'):h.need(observed[ext]==reviewed[ext],'fresh numeric '+ext+' differs')
    def ops(text):return [s.strip() for s in text.decode().splitlines() if s.strip().startswith(('ml_','sl_'))]
    first,second=ops(observed['asm']),ops(reviewed['asm'])
    h.need([s.split()[0] for s in first]==[s.split()[0] for s in second],'native instruction sequence changed')
    control=lambda ss:[s for s in ss if s.startswith(('sl_cbr','sl_jump','sl_barrier','sl_wait'))]
    h.need(control(first)==control(second),'native control/synchronization changed')
    h.need(len(observed['cubin'])>=64 and observed['cubin'][:6]==b'\x7fELF\x02\x01','actual ELF target cubin absent')
    return dict(source_typed_ABI_TTIR_TTGIR_LLVM_exact=True,ASM_text_exact=observed['asm']==reviewed['asm'],
        actual_cubin_exact=observed['cubin']==reviewed['cubin'],cubin_equality_required=False,
        backend_binary_equivalence_claimed=False,numerical_validation_required=True)


def loaded(kernel):
    kernel._init_handles();md=kernel.metadata
    fields=dict(n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),shared=getattr(md,'shared',None),
        num_warps=getattr(md,'num_warps',None),num_stages=getattr(md,'num_stages',None),warp_size=getattr(md,'warp_size',None))
    h.need(type(fields['n_regs']) is int and fields['n_regs']>0 and type(fields['n_spills']) is int and fields['n_spills']==0,'loaded resource fields unknown/spilled')
    h.need(type(fields['shared']) is int and 0<=fields['shared']<=65536 and (fields['num_warps'],fields['num_stages'],fields['warp_size'])==(16,1,64),'loaded geometry/shared bound changed')
    return dict(fields,phase='handles_initialized_pre_reader',zero_spill_observed=True,loaded_resources_verified=True)
