#!/usr/bin/env python3
"""Historical grammar regression and protocol faults only; stdlib/no device."""
import copy,json,re,sys,tarfile
from pathlib import Path
import biv150_konly_service_q32_native as native
import biv150_konly_service_q32_fresh_binding as binding
import bench_biv150_konly_service_q32_bitwise as bench


def check(fixture):
    with tarfile.open(fixture) as t:files={e:t.extractfile(e).read().decode() for e in ('ttgir','llir','asm')}
    audit=native.audit(files['ttgir'],files['ttgir'],files['llir'],files['asm'])
    faults=[]
    def rejects(name,fn):
        try:fn()
        except (ValueError,binding.h.Unsupported):faults.append(name);return
        raise AssertionError('accepted '+name)
    rejects('TTGIR operation change',lambda:native.audit(files['ttgir'],files['ttgir'].replace('arith.muli','arith.addi',1),files['llir'],files['asm']))
    for role in ('Q_publish_barrier','K_reuse_barrier','K_acquire_barrier'):
        lines=files['asm'].splitlines();lines[audit['ASM'][role]-1]='\tsl_nop'
        rejects('ASM missing '+role,lambda lines=lines:native.assembly_observation('\n'.join(lines)))
    qreg=audit['ASM']['Q_registers'][0];lines=files['asm'].splitlines();line=audit['ASM']['I8_QK_lines'][0]-1
    lines.insert(line,f'\tml_mov_b32 v{qreg}, 0x0')
    rejects('ASM Q overwritten in KV',lambda:native.assembly_observation('\n'.join(lines)))
    lines=files['asm'].splitlines();line=audit['ASM']['Q_byte_read_lines'][0]-1;lines[line]=lines[line].replace('ml_slb_load_b8','ml_slb_load_b16')
    rejects('ASM Q byte origin changed',lambda:native.assembly_observation('\n'.join(lines)))
    lines=files['asm'].splitlines();line=audit['ASM']['K_read_lines'][0]-1;lines[line]=lines[line].replace('ml_slb_load_b32x1','ml_slb_load_b16')
    rejects('ASM K origin changed',lambda:native.assembly_observation('\n'.join(lines)))
    sync=audit['shared_memory_synchronization'];lines=files['llir'].splitlines();lines[sync['Q_publish_barrier']-1]='  ; removed publish'
    rejects('LLVM missing publish',lambda:native.audit(files['ttgir'],files['ttgir'],'\n'.join(lines),files['asm']))
    good={k:True for k in ('finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference')}
    for name in good:
        bad=dict(good,**{name:False});rejects('hard '+name,lambda bad=bad:bench.policy(bad,False,True))
    rejects('original quality requires explicit diagnostic',lambda:bench.policy(good,False,False))
    rejects('integer hard flag',lambda:bench.policy(dict(good,finite=1),True,False))
    assert bench.policy(good,False,True)['precision_gate'] is False
    expected=dict(hash='fresh',signature={'K':'*i8','scale':'*fp32','V':'*bf16'},constants={'mode':0},attrs={},source_digest='source',metadata={'warps':16})
    raw={e:b'fixture' for e in binding.EXTS};raw['asm']=b'sl_barrier\nsl_jump BB0_1\n';raw['cubin']=b'\x7fELF\x02\x01'+bytes(64)
    binding.verify_recompile(expected,expected,raw,raw)
    for field in expected:
        bad=copy.deepcopy(expected);bad[field]='changed';rejects('binding '+field,lambda bad=bad:binding.verify_recompile(bad,expected,raw,raw))
    bad=copy.deepcopy(expected);bad['constants']['mode']=False
    rejects('typed bool/int constexpr',lambda:binding.verify_recompile(bad,expected,raw,raw))
    for ext in ('ttir','ttgir','llir','asm','cubin'):
        bad=dict(raw);bad[ext]=b'changed';rejects('recompile '+ext,lambda bad=bad:binding.verify_recompile(expected,expected,bad,raw))
    class Kernel:
        n_regs=120;n_spills=0
        class metadata:shared=57344;num_warps=16;num_stages=1;warp_size=64
        def _init_handles(self):self.initialized=True
    assert binding.loaded(Kernel())['zero_spill_observed'] is True
    for field,value in [('n_regs',None),('n_spills',None),('n_spills',1),('n_regs',True)]:
        kernel=Kernel();setattr(kernel,field,value);rejects('resource '+field+repr(value),lambda kernel=kernel:binding.loaded(kernel))
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(CPU_only=True,historical_fixture_only=True,actual_new_service_native_verified=False,
        discovered_fixture_Q_registers=audit['ASM']['Q_registers'],discovered_fixture_KV_header=audit['ASM']['KV_header'],
        fixture_LLVM_Q_reads=len(audit['LLVM']['Q_shared_read_lines']),fixture_ASM_Q_byte_reads=len(audit['ASM']['Q_byte_read_lines']),
        fixture_ASM_K_reads=len(audit['ASM']['K_read_lines']),rejected_faults=faults,fault_count=len(faults),
        all_protocol_tests_passed=True,GPU_executed=False,JIT_executed=False,precision_gate=False,service_verified=False)


def main():
    fixture=Path(__file__).resolve().parents[1]/'docs/evidence/konly-service-q32-hoist-20261027/historical-native-regression.tar.gz'
    print(json.dumps(check(fixture),sort_keys=True,indent=2))
if __name__=='__main__':main()
