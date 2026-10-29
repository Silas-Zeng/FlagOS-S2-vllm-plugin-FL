#!/usr/bin/env python3
"""Preparation only: valid original-producer fixtures and exact helper seam contract."""
import array,hashlib,importlib.util,json,sys
from pathlib import Path
import run_biv150_dynamic_hidden_gather_bits as bits
ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'vllm_fl/patches/biv150_dynamic_hidden_gather.py'
sha=lambda b:hashlib.sha256(b).hexdigest()
def helper():
 spec=importlib.util.spec_from_file_location('_real_gather_helper_plan',HELPER);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def producer_case(m,n,mode):
 """The real positive-length producers admit unique ends, not direct-test repeats."""
 bits.need((m,n) in bits.CASES and mode in ('ordinary','Plan'),'two fixed representative shapes')
 scheduled=([m-n+1]+[1]*(n-1)) if mode=='ordinary' else ([1,m-n+1]+[1]*(n-2))
 permutation=tuple(range(n)) if mode=='ordinary' else tuple(i for i in range(n) if scheduled[i]==1)+tuple(i for i in range(n) if scheduled[i]!=1)
 prefix=[0];values=[None]*n
 for i in permutation:prefix.append(prefix[-1]+scheduled[i]);values[i]=prefix[-1]-1
 h=helper();p,v=h.cpu_indices(tuple(scheduled),m,permutation)
 bits.need(tuple(prefix)==p and tuple(values)==v and len(set(values))==n and max(values)==m-1,'independent original producer')
 phases=[]
 for phase in (0,1):
  f=bits.fixture(m,n,phase);f['logical_indices']=values
  f['indices']=(array.array('i',[-987654321])*bits.IGUARD).tobytes()+array.array('i',values).tobytes()+(array.array('i',[-987654321])*bits.IGUARD).tobytes()
  payload=f['hidden'][2*bits.GUARD:-2*bits.GUARD]
  f['expected']=b''.join(payload[i*4096:(i+1)*4096] for i in values)
  poison=array.array('H',(x^65535 for x in bits.words(f['expected']))).tobytes()
  f['output_poison']=(array.array('H',[0xd34b])*bits.GUARD).tobytes()+poison+(array.array('H',[0xd34b])*bits.GUARD).tobytes()
  good=f['output_poison'][:256]+f['expected']+f['output_poison'][-256:]
  bits.need(all(bits.checks(f,f['hidden'],f['indices'],good).values()),'reuse existing complete byte checks')
  phases.append(dict(phase='AB'[phase],byte_sha256={k:sha(v) for k,v in f.items() if isinstance(v,bytes)},full_uint16_words=n*2048))
 return dict(M=m,N=n,source=mode,scheduled=scheduled,permutation=list(permutation),prefix=prefix,logits_indices=values,phases=phases,
  repeated_indices=False,original_cpu_source_proven_on_device=False)

def plan():
 cases=[producer_case(m,n,mode) for m,n in bits.CASES for mode in ('ordinary','Plan')]
 bits.need(not {'torch','triton','numpy'}.intersection(sys.modules),'stdlib preparation never imports device runtime')
 return dict(stage='CPU-prepared actual-frame device frontend; main899 synchronized; SDK constructor, real ingress and cost all untested',default_production_feature='OFF',parent=bits.PARENT,worker_parent='22ba8b7cdc3be4e9619f4fa3f4e91c9cca6cfdfc',main_dependency='899ce77470b0b539aebc108fd122f668c2d5894d',
  helper_sha256=sha(HELPER.read_bytes()),cases=cases,
  invariants=['do not replace helper._cold, _codes, _Receipt, _SEAL or state scope/caller guards','prepare and try_gather must be called by exact unwrapped execute_model code','original _prepare_inputs and _get_cumsum_and_arange remain exact','real CpuGpuBuffer pinnedCPU/NumPy alias and original GPU index-producing operations','component-built exact Plan is explicit and cannot prove make_plan/C3; never preinstall a fake helper cold state'],
  ingress_proposal='isolated exact ModelRunnerFL execution frame with synthetic scheduler/model fixture; any non-production surrounding stubs must be enumerated, never presented as actual service qualification',
  timing_design_pending=['candidate complete receipt+new-output allocation+host validation+launch versus torch.index_select on identical hidden/indices/current stream','time the entire original controlled execute_model frame symmetrically, including common preparation/H2D/event waits; do not subtract noisy measurements','ordinary worker calls cannot benchmark helper directly: immediate caller guard must remain intact','profiling only proves ingress during numeric checks; timed full execute_model fixtures run with profiling OFF and include common input prep/H2D/event waits','one cold input per mode then warmed ABBA only after actual entry, full bits and artifacts pass; no84-shape repeat'],
  fallback_required=['feature OFF','spec metadata','Graph mode non-NONE','actual capture when safely isolated','unsupported dtype/shape or metadata changed after prepare','one-use receipt reuse must raise'],
  artifact_admission=['actual production JIT function key/typed ABI/final cubin and handler exported','reuse old reviewed cache only if real production load proves same identity; otherwise stop for fresh actual review','keep old8d compile and direct22ba bit proof immutable'],
  actual_kernel_launches=0,actual_warmups=0,actual_entry_verified=False,actual_cold_verified=False,Graph_verified=False,performance_verified=False,service_verified=False)
if __name__=='__main__':print(json.dumps(plan(),sort_keys=True,indent=2))
