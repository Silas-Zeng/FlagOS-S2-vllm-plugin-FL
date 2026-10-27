"""Check explicit feature controls without changing official workloads; CPU only."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS


def main():
    root=Path(__file__).resolve().parents[1];path=root/'tools/run_biv150_small_out_acceptance.py'
    spec=importlib.util.spec_from_file_location('_multi2_acceptance',path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    default=driver.prefill_multi2_settings(root,'default')
    assert default['expected_installed'] and default['env']=={}
    assert driver.prefill_multi2_settings(root,'0')['env']=={'VLLM_FL_BIV150_PREFILL_MULTI2':'0'}
    assert not driver.prefill_multi2_settings(root,'0')['expected_installed']
    verdict=dict(metric_count=105,predictions=105,unique_questions=105,failed_predictions=0,score=99/105)
    near=driver.precision_state(verdict)
    assert near['accuracy'] is verdict and not near['precision_gate'] and near['near_miss_diagnostic']
    assert driver.precision_state(dict(verdict,score=100/105))['precision_gate']
    for bad in (dict(verdict,score=98/105),dict(verdict,predictions=104),
                dict(verdict,failed_predictions=1),dict(verdict,score=float('nan'))):
        try:driver.precision_state(bad)
        except RuntimeError:pass
        else:raise AssertionError('Hard accuracy failure continued to scoring')
    driver.require_runtime_scope('BIV150_COMPUTE_ORDER_FIRST_SELECTED {}')
    for marker in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK'):
        try:driver.require_runtime_scope(marker)
        except RuntimeError:pass
        else:raise AssertionError('Hard runtime binding failure continued to scoring')
    with tempfile.TemporaryDirectory() as folder:
        assert not driver.prefill_multi2_settings(folder,'default')['available']
        try:driver.prefill_multi2_settings(folder,'1')
        except ValueError:pass
        else:raise AssertionError('Feature-on claimed for an unavailable source')
    baseline=subprocess.check_output(['git','-C',str(root),'show','241220723a09061c182764f8f738e5ec01190367:tools/run_biv150_small_out_acceptance.py'],text=True)
    old,new=ast.parse(baseline),ast.parse(path.read_text())
    def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(function(old,'throughput'))==ast.dump(function(new,'throughput'))
    def serving(tree):
        return next(n.value for n in ast.walk(function(tree,'main')) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets))
    assert ast.dump(serving(old))==ast.dump(serving(new))
    pair_path=root/'tools/run_biv150_latest_main_pair.py'
    pair_spec=importlib.util.spec_from_file_location('_multi2_pair',pair_path)
    pair=importlib.util.module_from_spec(pair_spec);pair_spec.loader.exec_module(pair)
    args=NS(tool=path,candidate_source=root,baseline_source=root,candidate_commit='candidate',
        baseline_commit='baseline',out=Path('/tmp/not-executed'),baseline_out_lowering='1',
        baseline_large_linear='1',baseline_production_defaults=True,library=Path('/not-loaded.so'),
        tuning_snapshot=Path('/not-read'),fixtures=Path('/not-read'),baseline_prefill_multi2='0')
    control,candidate=pair.command(args,'baseline'),pair.command(args,'candidate')
    assert control[-2:]==['--prefill-multi2','0'] and '--prefill-multi2' not in candidate
    for flag in ('--run-accuracy','--run-throughput','--full-workload-warmup','--post-score-semantic-check','--production-defaults'):
        assert flag in control and flag in candidate
    result=dict(passed=True,gpu_executed=False,precision_gate=False,
        explicit_off_reaches_child_env=True,default_leaves_feature_env_absent=True,
        unavailable_source_on_rejected=True,three_short_fullwarm_and_single_score_loop_unchanged=True,
        official_serving_command_unchanged=True,matched_off_default_on_commands_prepared=True,
        near99of105_retains_failed_precision_for_diagnostic_timing=True,
        incomplete_nonfinite_and_below_near_miss_rejected=True,runtime_scope_faults_stop_scoring=True)
    (root/'docs/evidence/prefill-multi2-20261027/service-acceptance-cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
