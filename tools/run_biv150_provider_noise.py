"""Same-source, same-process A/A screen; no optimization or official score claim."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sqlite3
import subprocess
from types import SimpleNamespace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tuning_config_fingerprint(root):
    result={}
    for p in sorted(root.glob('*.db')):
        connection=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            names=[x[0] for x in connection.execute("SELECT name FROM sqlite_master WHERE type='table'") if 'benchmark' not in x[0].lower() and not x[0].startswith('sqlite_')]
            tables={}
            for name in names:
                quoted='"'+name.replace('"','""')+'"'
                rows=connection.execute('SELECT * FROM '+quoted).fetchall()
                tables[name]=sorted((json.dumps(row,default=str,sort_keys=True) for row in rows))
            result[p.name]=hashlib.sha256(json.dumps(tables,sort_keys=True).encode()).hexdigest()
        finally:connection.close()
    return result


def summarize(rows):
    values=[x['score']['total_token_throughput'] for x in rows]
    last_dirty=max((i for i,x in enumerate(rows) if x['new_metadata'] or x.get('changed_metadata') or x.get('tuning_db_changed')),default=-1)
    clean=[x['score']['total_token_throughput'] for x in rows[last_dirty+1:]]
    def stats(v):
        return dict(n=len(v),values=v,mean=statistics.mean(v),median=statistics.median(v),
                    range_percent=100*(max(v)-min(v))/statistics.mean(v),
                    cv_percent=100*statistics.stdev(v)/statistics.mean(v) if len(v)>1 else None)
    return dict(all_runs=stats(values),no_new_metadata_runs=stats(clean) if clean else None,valid_mature_suffix=len(clean)>=2,
                limitation='Observed finite-sample spread, not a confidence interval; within-process only. New-metadata counts do not bound all compile/tuning cost. No cross-workload transfer.')


def derive(text,source_sha):
    assert hashlib.sha256(text.encode()).hexdigest()==source_sha
    anchor='for length,count in ((4096,256),(16384,128)):'
    assert text.count(anchor)==1
    changed=text.replace(anchor,'for length,count in [(4096,256)]:')
    marker='sha(Path(__file__).with_name("biv150_peer_mixed_diagnostic.py"))'
    assert changed.count(marker)==1
    changed=changed.replace(marker,'sha(source / "tools/biv150_peer_mixed_diagnostic.py")')
    # Replace only the diagnostic workload loop, preserving serving, semantics,
    # source checks and owned cleanup. Production package files are untouched.
    a,b=ast.parse(text),ast.parse(changed)
    def finals(t):return [ast.dump(x,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Try) and n.finalbody for x in n.finalbody]
    assert finals(a)==finals(b),'Owned cleanup changed'
    return changed


def main(a):
    seed=a.seed.resolve();m=json.loads((seed/'manifest.json').read_text());s=json.loads((seed/'state.json').read_text())
    assert s['complete'] and s['cleanup_complete'] and s['precision_passed'] and s['semantic_passed'] and s['post_score_semantic_passed']
    assert m['native_mode']=='on' and m['stable_small_o']==m['out_lowering']==m['large_linear']=='1'
    source=Path(m['source']).resolve();commit=m['commit'];assert commit==a.commit
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    base=source/'tools/run_biv150_small_out_acceptance.py';assert sha(base)==m['driver_sha256']
    code=derive(base.read_text(),m['driver_sha256']);generated=a.out.parent/(a.out.name+'-owned-harness.py');assert not generated.exists();generated.write_text(code)
    spec=importlib.util.spec_from_file_location('_provider_noise_owned_harness',generated);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    snapshot=a.out.parent/(a.out.name+'-fixed-tuning');snapshot.mkdir(exist_ok=False)
    dbs=[]
    for p in sorted((seed/'flaggems-cache/config_cache').glob('*.db')):
        assert not p.is_symlink();shutil.copy2(p,snapshot/p.name);assert sha(p)==sha(snapshot/p.name);dbs.append(dict(name=p.name,sha256=sha(p)))
    assert dbs;(snapshot/'manifest.json').write_text(json.dumps(dict(databases=dbs),indent=2)+'\n')
    def throughput(out,*,full_workload_warmup=False):
        assert full_workload_warmup
        warm=[]
        for i in range(3):warm.append(h.benchmark(out,f'warm-4096-{i}',4096,128,64,920000+i))
        warm.append(dict(h.benchmark(out,'full-warm-4096',4096,1024,256,0),full_workload=True))
        (out/'warmups.json').write_text(json.dumps({'4096':warm},indent=2)+'\n')
        rows=[]
        for i in range(a.repeats):
            before=h.cache_inventory(out/'triton-cache')
            db_before=tuning_config_fingerprint(out/'flaggems-cache/config_cache')
            (out/f'aa-{i}-gpu-before.txt').write_text(subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True))
            score=h.benchmark(out,f'aa-4096-{i}',4096,1024,256,0)
            after=h.cache_inventory(out/'triton-cache')
            assert score['completed']==score['num_prompts']==256 and score['failed']==0 and score['max_concurrency']==64 and score['total_input_tokens']==1048576 and score['total_output_tokens']==262144
            db_after=tuning_config_fingerprint(out/'flaggems-cache/config_cache')
            (out/f'aa-{i}-gpu-after.txt').write_text(subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True))
            rows.append(dict(score=score,new_metadata=[after[k] for k in sorted(after.keys()-before.keys())],changed_metadata=[after[k] for k in sorted(after.keys() & before.keys()) if after[k]['sha256']!=before[k]['sha256']],tuning_db_before=db_before,tuning_db_after=db_after,tuning_db_changed=db_before!=db_after))
            (out/'noise-scores.json').write_text(json.dumps(rows,indent=2)+'\n')
            (out/'noise-summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')
        return {'4096':rows[-1]['score']}
    h.throughput=throughput
    args=SimpleNamespace(source=source,commit=commit,out=a.out,gate_library=Path(m['env']['BIV150_NATIVE_W8A8_LIBRARY']),native_mode='on',stable_small_o='1',out_lowering='1',large_linear='1',provider_probe=True,post_score_semantic_check=True,diagnostic_after_semantic_failure=False,run_accuracy=False,run_throughput=True,full_workload_warmup=True,tuning_snapshot=snapshot,fixtures=a.fixtures)
    h.main(args)
    plan=dict(seed=str(seed),seed_manifest_sha256=sha(seed/'manifest.json'),source=str(source),commit=commit,qualified_precision_reused=True,accuracy_rerun=False,tuning_database_fixed=True,triton_cache_fresh=True,only_mature_suffix_used_for_noise=True,repeats=a.repeats,workload=4096,serving_command_unchanged=True,within_process_only=True,production_modified=False,driver_sha256=sha(__file__))
    (a.out/'noise-protocol.json').write_text(json.dumps(plan,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--repeats',type=int,choices=[2,3],default=3);main(p.parse_args())
