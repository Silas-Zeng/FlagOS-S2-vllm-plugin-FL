from pathlib import Path
import json,hashlib,math,numpy as np
r=Path('/private/tmp/cjj-native-gqa-batch95-numeric-v2-raw-20261028');o=r/'out';m=json.loads((r/'download-manifest.json').read_text());d=json.loads((o/'summary.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
for row in m['files']:
 b=(r/row['relative']).read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256']
assert d['completed'] and d['passed'] and not d['errors'] and not d['service_activation']
assert d['library_sha256']=='06c5228e1feb0b26cb71dfcf703d60693bd65926491fd7fa5f916d970f879a89'
def bf16(a):return (a.astype(np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float64)
checks=[]
for x in d['records']:
 n,control=x['rows'],x['control']
 with np.load(o/(str(n)+'-'+str(control)+'.npz'),allow_pickle=False) as z:
  q=bf16(z['q_bits']);cache=bf16(z['cache_bits']);table=z['table'];lengths=z['lengths'];ref=np.zeros((n,16,128),dtype=np.float64)
  for row,ln in enumerate(lengths):
   if not ln:continue
   t=np.arange(ln);ids=table[row,t//16];assert np.all(ids>=0) and np.all(ids<cache.shape[0])
   k=cache[ids,0,t%16];v=cache[ids,1,t%16];qq=q[row].reshape(2,8,128)
   s=np.einsum('hgd,thd->hgt',qq,k)*128**-.5;p=np.exp(s-s.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
   ref[row]=np.einsum('hgt,thd->hgd',p,v).reshape(16,128)
  assert np.allclose(ref,z['reference'],rtol=1e-12,atol=1e-12)
  for field in ('out','scratch','maxima','sums'):assert np.array_equal(z['native_'+field+'_bits'],z['triton_'+field+'_bits'])
  for arm in ('triton','native'):
   got=bf16(z[arm+'_out_bits']);assert np.array_equal(got,z[arm+'_out']);delta=got-ref
   bad=int(np.count_nonzero(np.abs(delta)>.005+.005*np.abs(ref)));rel=float(np.linalg.norm(delta)/np.linalg.norm(ref));assert bad==0 and rel<=.005
   reported=next(y for y in x['numeric'] if y['arm']==arm);assert math.isclose(rel,reported['relative_rmse'],rel_tol=1e-12)
   checks.append(dict(rows=n,control=control,arm=arm,pointwise_violations=bad,relative_RMSE=rel))
result=dict(CPU_only=True,source='9d4f4e6d33b3365ac3628c2d09d71472ac809b70',all_original_files_verified=len(m['files']),all_full_FP64_references_independently_regenerated=True,full_element_metrics=checks,all_saved_partial_and_output_native_Triton_bits_equal=True,original_run_guards=True,outer_guard_snapshots_independently_replayed=False,readonly_snapshots_independently_replayed=False,library_SHA=d['library_sha256'],driver_identity_claimed=False,Root_explicit_serial_admission_required=True,service_activation=False,Graph_side_performance_or_model_qualified=False)
p=Path('/private/tmp/cjj-root-batch95-numeric-review-20261028.json');p.write_text(json.dumps(result,indent=2)+'\n');print('reviewSHA',sha(p.read_bytes()),'metrics',len(checks))
