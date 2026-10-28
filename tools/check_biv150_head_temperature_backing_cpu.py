"""Local CPU source inverse and archived selector evidence; never imports GPU frameworks."""
import argparse,ast,hashlib,json,sqlite3,subprocess
from pathlib import Path
from check_biv150_head_temperature_ragged_component_cpu import functions

ROOT=Path(__file__).resolve().parents[1]
BASE='00213bdd9afa75d7ea7ebdd851b2d4ed0061ce66'
TABLE='linear_kernel_91df7045e2de13a4c456e8d13079eb7c-0f015d35c97293819201f206510edb40'


def main():
    p=argparse.ArgumentParser();p.add_argument('--linear-source',type=Path,required=True);p.add_argument('--libentry-source',type=Path,required=True);p.add_argument('--tuning-db',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    sha=lambda b:hashlib.sha256(b).hexdigest()
    def old(name):return subprocess.check_output(['git','show',BASE+':tools/'+name],cwd=ROOT).decode()
    name='biv150_head_temperature_ragged_worker.py'
    before=ast.parse(old(name));after=ast.parse((ROOT/'tools'/name).read_text())
    old_worker=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    worker=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    old_try=next(n for n in old_worker.body if isinstance(n,ast.Try));new_try=next(n for n in worker.body if isinstance(n,ast.Try))
    replacements={n.name:n for n in old_try.body if isinstance(n,ast.FunctionDef) and n.name in ('buffer','guard_check')}
    new_try.body=[replacements.get(n.name,n) if isinstance(n,ast.FunctionDef) else n for n in new_try.body]
    telemetry=[n for n in new_try.body if isinstance(n,ast.Assign) and ast.unparse(n).startswith("result['baseline_prelaunch_admission'] =")]
    assert len(telemetry)==1;new_try.body.remove(telemetry[0])
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),'worker changed beyond physical allocation/guards and receipt'
    name='biv150_head_temperature_ragged_factory.py';b=functions(old(name));c=functions((ROOT/'tools'/name).read_text())
    for key in ('configure','actual_linear_module','launch','PreScaledSampler'):
        assert ast.dump(b[key],include_attributes=False)==ast.dump(c[key],include_attributes=False),key
    unchanged=[]
    for name in ('biv150_head_temperature_ragged_timing.py','compile_biv150_head_temperature_ragged.py'):
        assert old(name).encode()==(ROOT/'tools'/name).read_bytes();unchanged.append(name)
    assert not subprocess.check_output(['git','diff',BASE,'--','vllm_fl','csrc'],cwd=ROOT)
    linear=a.linear_source.read_bytes();entry=a.libentry_source.read_bytes()
    assert sha(linear)=='d431a4fa144ed67dc3deab46dd5beace3612858075f85c39e602447ba3bf768a'
    assert sha(entry)=='414e8a3f74e2522e40659e7fccd84e301f4517220837a87af0311f050bbf3025'
    assert 'ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)' in linear.decode()
    assert 'a = tl.load(A)' in linear.decode()
    conn=sqlite3.connect(a.tuning_db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    cols=[r[1] for r in conn.execute('PRAGMA table_info("'+TABLE+'")')]
    records=[]
    for m,expected in [(32,(32,64,64,8,1,1)),(64,(64,256,32,16,1,2))]:
        rows=list(conn.execute('SELECT * FROM "'+TABLE+'" WHERE key_0=? AND key_1=130560 AND key_2=2048',(m,)))
        assert len(rows)==1;d=dict(zip(cols,rows[0]));keys=['BLOCK_M','BLOCK_N','BLOCK_K','num_warps','num_ctas','num_stages']
        assert tuple(d[k] for k in keys)==expected
        assert all(d['key_'+str(i)]=='torch.bfloat16' for i in range(3,7))
        records.append({k:d[k] for k in ['key_0','key_1','key_2',*keys]})
    conn.close()
    result=dict(gpu_executed=False,base=BASE,production_and_GPU_sources_unchanged=True,
        worker_inverse_except_buffer_guard_and_receipt=True,original_factory_launch_and_sampler_unchanged=True,
        byte_identical_tools=unchanged,source_sha256={'linear':sha(linear),'libentry':sha(entry)},
        tuning_db_sha256=sha(a.tuning_db.read_bytes()),table=TABLE,archived_config_rows=records,
        limitation='Archived selector evidence plus source/opaque CPU gates; actual next-process cache/prelaunch and numerical qualification remain pending. No raw arrays or SDK source uploaded.')
    text=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as f:f.write(text)
    print(text,end='')


if __name__=='__main__':main()
