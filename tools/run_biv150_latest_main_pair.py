"""Serial exact-main/default-candidate acceptance; records evidence, never merges."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),state=fields[0])
    except FileNotFoundError:return None


def command(a,label):
    candidate=label=='candidate'
    baseline_out=getattr(a,'baseline_out_lowering','0')
    baseline_large=getattr(a,'baseline_large_linear','0')
    result=['/usr/local/bin/python',str(a.tool),'--source',str(a.candidate_source if candidate else a.baseline_source),
            '--commit',a.candidate_commit if candidate else a.baseline_commit,'--out',str(a.out/label),
            '--out-lowering','1' if candidate else baseline_out,'--large-linear','1' if candidate else baseline_large,
            '--native-mode','on','--stable-small-o','1','--gate-library',str(a.library),
            '--tuning-snapshot',str(a.tuning_snapshot),'--fixtures',str(a.fixtures),
            '--provider-probe','--post-score-semantic-check','--run-accuracy','--run-throughput','--full-workload-warmup']
    if candidate or getattr(a,'baseline_production_defaults',False):result.append('--production-defaults')
    multi2=getattr(a,'candidate_prefill_multi2' if candidate else 'baseline_prefill_multi2','default')
    if multi2!='default':result.extend(['--prefill-multi2',multi2])
    return result


def idle():
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'Port occupied'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)


def main(a):
    a.out.mkdir(parents=False,exist_ok=False)
    state=dict(owner=identity(os.getpid()),baseline_commit=a.baseline_commit,
               candidate_commit=a.candidate_commit,tool_sha256=hashlib.sha256(a.tool.read_bytes()).hexdigest(),
               controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results={})
    def save(phase,**extra):
        state.update(phase=phase,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        temporary=a.out/'state.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(a.out/'state.json')
    try:
        for label in ['baseline','candidate']:
            idle()
            assert hashlib.sha256(a.tool.read_bytes()).hexdigest()==state['tool_sha256']
            cmd=command(a,label);source=a.candidate_source if label=='candidate' else a.baseline_source
            revision=a.candidate_commit if label=='candidate' else a.baseline_commit
            assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==revision
            assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
            save('starting_'+label,command=cmd)
            with (a.out/(label+'-model.log')).open('x') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(label,child=identity(child.pid));code=child.wait();save(label+'_finished',returncode=code)
            assert code==0,(label,code)
            result=json.loads((a.out/label/'state.json').read_text())
            manifest=json.loads((a.out/label/'manifest.json').read_text())
            assert result['complete'] and result['cleanup_complete'] and result['semantic_passed'] and result['post_score_semantic_passed']
            assert manifest['loaded_source_fingerprint_verified']
            if label=='candidate' or getattr(a,'baseline_production_defaults',False):
                assert manifest['production_defaults']
                assert not any(k in manifest['env'] for k in ['VLLM_FL_BIV150_SMALL_OUT_LOWERING','VLLM_FL_BIV150_LARGE_LINEAR'])
            state['results'][label]=dict(state=result,manifest=manifest)
            save(label+'_validated')
        baseline=state['results']['baseline'];candidate=state['results']['candidate']
        assert baseline['manifest']['command']==candidate['manifest']['command']
        assert baseline['manifest']['tuning_snapshot']==candidate['manifest']['tuning_snapshot']
        assert baseline['manifest']['gate_library_sha256']==candidate['manifest']['gate_library_sha256']
        comparison={length:dict(baseline_tps=baseline['state']['scores'][length]['total_token_throughput'],
                    candidate_tps=candidate['state']['scores'][length]['total_token_throughput'],
                    observed_relative_change=candidate['state']['scores'][length]['total_token_throughput']/baseline['state']['scores'][length]['total_token_throughput']-1)
                    for length in ['4096','16384']}
        idle();save('completed',comparison=comparison,comparison_is_single_pair=True,automatic_merge_performed=False)
    except BaseException as error:
        save('failed',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['tool','out','baseline-source','candidate-source','library','tuning-snapshot','fixtures']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--baseline-commit',required=True);p.add_argument('--candidate-commit',required=True)
    p.add_argument('--baseline-out-lowering',choices=['0','1'],default='0')
    p.add_argument('--baseline-large-linear',choices=['0','1'],default='0')
    p.add_argument('--baseline-production-defaults',action='store_true')
    p.add_argument('--baseline-prefill-multi2',choices=['default','0','1'],default='default')
    p.add_argument('--candidate-prefill-multi2',choices=['default','0','1'],default='default')
    main(p.parse_args())
