import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
def metrics(a,b):
 a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 delta=a-b;n=len(delta);rms=math.hypot(*delta)/math.sqrt(n);den=math.hypot(*b)/math.sqrt(n);v=int(np.count_nonzero(abs(delta)>.005+.005*abs(b)))
 return dict(finite=True,pointwise_passed=v==0,pointwise_violations=v,relative_rmse=rms/den if den else (0. if not np.any(delta) else math.inf),max_abs=float(abs(delta).max()),rms=rms)
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();r=a.raw_root;m=json.loads((r/'download-manifest.json').read_text());assert m['original_owner']['same_start'] and m['original_owner']['state']=='Z' and not m['model_rerun'] and m['raw_files']==296
 for f in m['files']:
  b=(r/f['relative']).read_bytes();assert len(b)==f['bytes'] and sha(b)==f['sha256']
 j=json.loads((r/'out/result.json').read_text());assert j['completed'] and not j['errors'] and j['implementation_numeric_verified'] and not j['passed'] and not j['compile_only'];assert j['source_commit']=='dc5fd2e5b86a3d57aceaad26e8556d169ec7462a'
 rows=[];witnesses=0
 for c in j['records']:
  assert c['physical_page_bytes']==12544 and c['side'] and len(c['graph'])==3 and c['implementation_hard_gates'] and all(c['hard_checks'].values()) and not c['original_bf16_proxy_passed']
  for n in c['numeric']:
   f=r/'out'/n['witness']['path'];assert len(f.read_bytes())==n['witness']['bytes'] and sha(f.read_bytes())==n['witness']['sha256'];witnesses+=1
   with np.load(f,allow_pickle=False) as z:
    base=(z['baseline_out'].view(np.uint16).astype(np.uint32)<<16).view(np.float32);candidate=(z['compact_out'].view(np.uint16).astype(np.uint32)<<16).view(np.float32);assert base.shape==candidate.shape==(c['batch'],16,128)
    assert z['table'].shape==(c['batch'],8192) and len(np.unique(z['slots'][z['slots']>=0]))==np.count_nonzero(z['slots']>=0)
    for idx,check in enumerate(n['checks']):
     row=check['row'];assert row==int(z['selected_rows'][idx]);values={k:metrics(x,y) for k,x,y in [('baseline',base[row],z['original_reference'][idx]),('quantized_operand',candidate[row],z['quantized_reference'][idx]),('original_bf16',candidate[row],z['original_reference'][idx])]}
     for k,vals in values.items():
      for name,v in vals.items():
       expected=check[k][name]
       if type(v) in (bool,int):assert v==expected,(c['case'],row,k,name)
       else:assert math.isclose(v,expected,rel_tol=2e-12,abs_tol=1e-15),(c['case'],row,k,name,v,expected)
     limit=max(.005,1.2*values['baseline']['relative_rmse']);passed=values['baseline']['pointwise_passed'] and values['quantized_operand']['pointwise_passed'] and values['original_bf16']['relative_rmse']<=limit
     assert values['baseline']['pointwise_passed'] and values['quantized_operand']['pointwise_passed'];assert passed==check['candidate_numeric_passed'];assert check['thresholds_relaxed'] is False
     rows.append(dict(case=c['case'],iteration=n['iteration'],row=row,length=int(z['lengths'][row]),**values,original_bf16_proxy_passed=passed))
 assert witnesses==6 and len(rows)==288
 report=dict(passed=True,CPU_only=True,source_commit=j['source_commit'],raw_files=m['raw_files'],raw_bytes=m['raw_bytes'],all_original_SHA_verified=True,full_final_FP64_saved_reference_arithmetic_replayed=True,saved_FP64_cold_gold_regenerated=False,partial_FP64_reference_verified=False,graph_side_cache_guard_evidence='frozen runtime assertions and recorded checks; separate replay snapshots/cache bodies not saved',full_final_rows=len(rows),witnesses=witnesses,quantized_operand_hard_gates_passed=True,original_bf16_proxy_passed=False,thresholds_changed=False,precision_gate=False,service_verified=False,capacity_verified=False,performance_measured=False,records=rows)
 a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(passed=True,out=str(a.out),sha256=sha(a.out.read_bytes()),rows=len(rows),original_proxy_failed_rows=sum(not x['original_bf16_proxy_passed'] for x in rows))))
if __name__=='__main__':main()
