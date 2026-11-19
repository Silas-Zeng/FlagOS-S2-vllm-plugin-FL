"""Owned actual-service validation of optional compiler-owned BF16 output buffers."""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace



def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), state=fields[0])
    except FileNotFoundError:
        return None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_inventory(root):
    result = {}
    if root.exists():
        for path in root.rglob("*.json"):
            if path.name.startswith("__grp__"):
                continue
            result[str(path.relative_to(root))] = dict(
                path=str(path.relative_to(root)), sha256=sha(path),
                mtime_ns=path.stat().st_mtime_ns)
    return result


def expected_source_fingerprint(source, env):
    import torch
    path = source / 'vllm_fl/compile_semantics.py'
    spec = importlib.util.spec_from_file_location('_acceptance_expected_semantics', path)
    semantics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantics)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**semantics._GEOMETRY)))
    return semantics.build_fingerprint(config, 'iluvatar', package_root=source/'vllm_fl', environ=env,
        abi={'torch': str(torch.__version__), 'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI)})


def prefill_multi2_settings(source, mode):
    """Feature selection only; the serving and benchmark arguments are fixed."""
    if mode not in ('default','0','1'):
        raise ValueError('Unknown multi2 mode')
    available=(Path(source)/'vllm_fl/patches/biv150_prefill_multi2.py').is_file()
    if mode=='1' and not available:
        raise ValueError('Source does not provide the requested multi2 implementation')
    return dict(available=available,expected_installed=available and mode!='0',
                env={} if mode=='default' else {'VLLM_FL_BIV150_PREFILL_MULTI2':mode})


def precision_state(verdict):
    """Preserve the original full105 score; near misses get diagnostic timing."""
    complete=(verdict['metric_count']==verdict['predictions']==verdict['unique_questions']==105
              and not verdict['failed_predictions'])
    score=verdict['score']
    if not complete or not math.isfinite(score) or not .94<=score<=1:
        raise RuntimeError('Incomplete, nonfinite or below-near-miss full105 verdict')
    gate=score>=.95
    return dict(accuracy=verdict,precision_passed=gate,precision_gate=gate,
                near_miss_diagnostic=not gate,official_accuracy_evaluated=True)


def require_runtime_scope(logs):
    for marker in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK'):
        if marker in logs:
            raise RuntimeError('Hard runtime source/scope/prelaunch failure: '+marker)


def throughput(out, *, full_workload_warmup=False):
    from run_biv150_provider_noise import tuning_config_fingerprint
    scores, warmups = {}, {}
    for length, count in ((4096, 256), (16384, 128)):
        warmups[str(length)] = []
        for index in range(3):
            value = benchmark(out, f"warm-{length}-{index}", length, 128, 64, 920000 + index)
            warmups[str(length)].append(value)
            (out / "warmups.json").write_text(json.dumps(warmups, indent=2) + "\n")
        if full_workload_warmup:
            # Short 64-request warmups do not visit the late mixed/tail shapes
            # of the official multi-wave workload. Keep this unscored run in
            # the evidence rather than treating cache-cold stalls as a gain.
            value = benchmark(out, f"full-warm-{length}", length, 1024, count, 0)
            warmups[str(length)].append(dict(value, full_workload=True))
            (out / "warmups.json").write_text(json.dumps(warmups, indent=2) + "\n")
        before = cache_inventory(out / "triton-cache")
        config_before = tuning_config_fingerprint(out / 'flaggems-cache/config_cache')
        scores[str(length)] = benchmark(out, f"score-{length}", length, 1024, count, 0)
        config_after = tuning_config_fingerprint(out / 'flaggems-cache/config_cache')
        (out/f'score-{length}-selected-config.json').write_text(json.dumps(dict(
            before=config_before, after=config_after, changed=config_before!=config_after,
            scope='Complete benchmark command including client preflight; not isolated timed scoring cost.'),indent=2)+'\n')
        after = cache_inventory(out / "triton-cache")
        additions = [after[key] for key in sorted(after.keys() - before.keys())]
        (out / f"score-{length}-new-kernels.json").write_text(json.dumps(additions, indent=2) + "\n")
        (out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def main(args):
    global benchmark
    assert args.out_lowering=="0" or args.stable_small_o=="1"
    assert args.large_linear=='0' or (args.out_lowering==args.stable_small_o=='1' and args.provider_probe)
    source = args.source.resolve()
    sys.path.insert(0, str(source / "tools"))
    from run_biv150_gate_team_acceptance import get_json, accuracy, benchmark, performance_gates
    from biv150_peer_mixed_diagnostic import run as mixed_diagnostic
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == args.commit
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", 9031)) != 0, "Port 9031 is occupied"
    idle = subprocess.check_output(["/usr/local/corex/bin/ixsmi"], text=True)
    assert "No running processes found" in idle, "GPU is busy"
    assert sha(args.gate_library) == "5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "idle.txt").write_text(idle)
    snapshot = args.tuning_snapshot.resolve()
    records = json.loads((snapshot / "manifest.json").read_text())["databases"]
    target = args.out / "flaggems-cache/config_cache"
    target.mkdir(parents=True)
    for record in records:
        name = record["name"]
        assert Path(name).name == name and name.endswith(".db")
        assert sha(snapshot / name) == record["sha256"]
        shutil.copy2(snapshot / name, target / name)
        assert sha(target / name) == record["sha256"]
    state = dict(complete=False, cleanup_complete=False, acceptance_passed=False,
                 precision_passed=None, precision_gate=False, diagnostic_only=True, phase="starting",
                 commit=args.commit, native_mode=args.native_mode)

    def save(phase, **values):
        state.update(phase=phase, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        temporary = args.out / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.out / "state.json")

    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("VLLM_FL_BIV150_", "BIV150_NATIVE_", "BIV150_PROFILE_")) or key in ("FLAGTUNE", "USE_C_EXTENSION"):
            env.pop(key)
    env.update(VLLM_PLUGINS="fl", PYTHONPATH=str(source), PYTHONUNBUFFERED="1",
               VLLM_DISABLE_COMPILE_CACHE="1", TORCHINDUCTOR_CACHE_DIR=str(args.out / "inductor-cache"),
               TRITON_CACHE_DIR=str(args.out / "triton-cache"),
               FLAGGEMS_CACHE_DIR=str(args.out / "flaggems-cache"),
               BIV150_NATIVE_W8A8_LIBRARY=str(args.gate_library.resolve()),
               VLLM_FL_BIV150_NATIVE_W8A8="1" if args.native_mode == "on" else "0",
               VLLM_FL_BIV150_STABLE_SMALL_O=args.stable_small_o,
               VLLM_FL_BIV150_SMALL_OUT_LOWERING=args.out_lowering,
               VLLM_FL_BIV150_LARGE_LINEAR=args.large_linear,
               VLLM_FL_BIV150_SMALL_OUT_RECORD_SELECTION=args.out_lowering)
    production_defaults = bool(getattr(args, 'production_defaults', False))
    if production_defaults:
        assert args.out_lowering == args.large_linear == args.stable_small_o == '1'
        env.pop('VLLM_FL_BIV150_SMALL_OUT_LOWERING')
        env.pop('VLLM_FL_BIV150_LARGE_LINEAR')
    multi2_mode=getattr(args,'prefill_multi2','default')
    multi2=prefill_multi2_settings(source,multi2_mode)
    env.update(multi2['env'])
    command = ["/usr/local/bin/vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
               "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
               "--served-model-name", "minicpm", "--gpu-memory-utilization", "0.85", "--max-model-len", "131072"]
    manifest = dict(command=command, source=str(source), commit=args.commit,
                    prefill_multi2_mode=multi2_mode,prefill_multi2_available=multi2['available'],
                    native_mode=args.native_mode, out_lowering=args.out_lowering, large_linear=args.large_linear,production_defaults=production_defaults,provider_probe=args.provider_probe, post_score_semantic_check=args.post_score_semantic_check, stable_small_o=args.stable_small_o, run_accuracy=args.run_accuracy, run_throughput=args.run_throughput, full_workload_warmup=args.full_workload_warmup, driver_sha256=sha(__file__),
                    diagnostic_sha256=sha(source / "tools/biv150_peer_mixed_diagnostic.py"),
                    gate_library_sha256=sha(args.gate_library), tuning_snapshot=records,
                    env={k: v for k, v in env.items() if k.startswith(("VLLM_", "BIV150_", "TRITON_", "TORCHINDUCTOR_", "FLAGGEMS_"))})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = args.out / "server.log"
    log = log_path.open("x")
    server = None
    owned = {}
    launcher = None

    def remember():
        if server is None or launcher is None:
            return
        anchors = [launcher, *owned.values()]
        if not any((current := identity(member["pid"])) and
                   current["start_ticks"] == member["start_ticks"] and
                   current["sid"] == launcher["sid"] for member in anchors):
            return
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                member = identity(int(path.name))
                if member and member["sid"] == server.pid:
                    owned.setdefault(member["pid"], member)

    try:
        server = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launcher = identity(server.pid)
        assert launcher and launcher["sid"] == server.pid
        owned[server.pid] = launcher
        save("waiting_service", owner=launcher, driver=identity(os.getpid()))
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            remember()
            if server.poll() is not None:
                raise RuntimeError("Owned service exited; inspect server.log")
            try:
                get_json("http://127.0.0.1:9031/v1/models")
                break
            except Exception:
                time.sleep(3)
        else:
            raise TimeoutError("Service startup timeout")
        logs = log_path.read_text()
        actual_fingerprints = [json.loads(line.split('BIV150_COMPILE_SEMANTICS ', 1)[1])
                               for line in logs.splitlines() if 'BIV150_COMPILE_SEMANTICS ' in line]
        expected_fingerprint = expected_source_fingerprint(source, env)
        assert actual_fingerprints and all(x == expected_fingerprint for x in actual_fingerprints), {
            'expected_source': expected_fingerprint, 'loaded': actual_fingerprints}
        manifest['loaded_source_fingerprint_verified'] = True
        manifest['expected_source_fingerprint'] = expected_fingerprint
        installations = {}
        for marker in ("BIV150_NATIVE_W8A8_INSTALL",
                       "BIV150_Q_FRAGMENT_INSTALL", "BIV150_PREFILL_NORMALIZATION_INSTALL",
                       "BIV150_GQA_DECODE_INSTALL", "BIV150_MIXED_DECODE_INSTALLED", "BIV150_STABLE_SMALL_O_INSTALL"):
            values = [ast.literal_eval(line.split(marker + " ", 1)[1]) for line in logs.splitlines() if marker + " " in line]
            assert values, f"Missing installation provenance: {marker}"
            installations[marker] = values[-1]
            expected = ((args.native_mode == "on") if marker == "BIV150_NATIVE_W8A8_INSTALL"
                        else (args.stable_small_o == "1") if marker == "BIV150_STABLE_SMALL_O_INSTALL" else True)
            assert values[-1].get("installed") is expected, values[-1]
        out_installs=[ast.literal_eval(line.split('BIV150_SMALL_OUT_LOWERING_INSTALL ',1)[1])
                      for line in logs.splitlines() if 'BIV150_SMALL_OUT_LOWERING_INSTALL ' in line]
        assert bool(out_installs) is (args.out_lowering=='1')
        if out_installs:
            assert out_installs[-1]['installed']
            assert out_installs[-1]['large_linear']==(args.large_linear=='1')
            installations['BIV150_SMALL_OUT_LOWERING_INSTALL']=out_installs[-1]
        manifest["installations"] = installations
        if (source/'vllm_fl/patches/biv150_prefill_q32.py').is_file():
            q32_installs=[ast.literal_eval(line.split('BIV150_PREFILL_Q32_INSTALL ',1)[1])
                         for line in logs.splitlines() if 'BIV150_PREFILL_Q32_INSTALL ' in line]
            assert q32_installs and q32_installs[-1].get('installed') is True,q32_installs
            manifest['installations']['BIV150_PREFILL_Q32_INSTALL']=q32_installs[-1]
        multi2_installs=[ast.literal_eval(line.split('BIV150_PREFILL_MULTI2_INSTALL ',1)[1])
                        for line in logs.splitlines() if 'BIV150_PREFILL_MULTI2_INSTALL ' in line]
        if multi2['available']:
            assert multi2_installs and multi2_installs[-1].get('installed') is multi2['expected_installed'],multi2_installs
            manifest['installations']['BIV150_PREFILL_MULTI2_INSTALL']=multi2_installs[-1]
        else:
            assert not multi2_installs
        manifest["kv_cache_lines"] = [line for line in logs.splitlines() if "GPU KV cache size" in line]
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save("ready")
        save("mixed_diagnostic")
        summary = mixed_diagnostic(args.out, args.fixtures)
        selected_marker='BIV150_SMALL_OUT_SELECTED ' if args.out_lowering=='1' else 'BIV150_STABLE_SMALL_O_SELECTED '
        selected=[ast.literal_eval(line.split(selected_marker,1)[1])
                  for line in log_path.read_text().splitlines() if selected_marker in line]
        assert bool(selected) is (args.stable_small_o=='1')
        if args.stable_small_o=='1':
            expected={(2048,2048),(12288,2048),(2560,2048),(2048,6144)}
            actual={(x['output_columns'],x['input_columns']) for x in selected if x['rows']==2}
            assert actual==expected,{'expected_M2':sorted(expected),'actual_M2':sorted(actual)}
        manifest["selected_small_o"] = selected
        (args.out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        passed = summary["all_correct"] and summary["http_overlap"]
        save("mixed_completed", diagnostic=summary, semantic_passed=passed)
        assert passed or args.diagnostic_after_semantic_failure,'Mixed semantic gate failed before formal accuracy/scoring'
        require_runtime_scope(log_path.read_text())
        if args.provider_probe:
            # A separate post-ready diagnostic, excluded from official scoring.
            # Exact input token count is chosen to avoid startup profile2048 and
            # decode capture sizes<=512, making new runtime selection observable.
            import urllib.request
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained('/workspace/MiniCPM5-2B')
            ids=tokenizer.encode('A short sentence for projection diagnostics. '*1000,add_special_tokens=False)[:1739]
            assert len(ids)==1739
            offset=len(log_path.read_text())
            payload=dict(model='minicpm',prompt=ids,max_tokens=1,temperature=0,ignore_eos=True)
            request=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as response:probe=json.load(response)
            assert probe['usage']['prompt_tokens']==1739 and probe['usage']['completion_tokens']==1,probe
            marker='BIV150_SMALL_OUT_PROVIDER_SELECTED '
            events=[ast.literal_eval(line.split(marker,1)[1]) for line in log_path.read_text()[offset:].splitlines() if marker in line]
            if args.out_lowering=='1':
                expected_provider='provided_linear' if args.large_linear=='1' else 'mm_out'
                geometries={(e['output_columns'],e['input_columns']) for e in events if e['rows']==1739 and e['provider']==expected_provider}
                assert geometries=={(2048,2048),(2560,2048),(12288,2048),(2048,6144)},events
            else:
                assert not events,events
            proof=dict(request=payload,response=probe,post_ready_events=events,large_linear=args.large_linear,official_benchmark=False)
            (args.out/'post-ready-provider-probe.json').write_text(json.dumps(proof,indent=2)+'\n')
            manifest['post_ready_provider_probe']=proof
            (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if args.run_accuracy:
            save("accuracy")
            try:
                accuracy(args.out)
            except RuntimeError:
                path=args.out/"accuracy-verdict.json"
                if not path.exists():raise
                verdict=json.loads(path.read_text())
                if not (verdict["metric_count"]==verdict["predictions"]==verdict["unique_questions"]==105
                        and not verdict["failed_predictions"] and verdict["score"]>=.94):raise
            verdict=json.loads((args.out/"accuracy-verdict.json").read_text())
            state.update(precision_state(verdict))
        require_runtime_scope(log_path.read_text())
        if args.run_throughput:
            assert passed or args.diagnostic_after_semantic_failure, "Mixed semantic failure requires explicit diagnostic scoring"
            save("throughput")
            scores=throughput(args.out, full_workload_warmup=args.full_workload_warmup)
            for length,count in ((4096,256),(16384,128)):
                value=scores[str(length)]
                expected=dict(completed=count,failed=0,num_prompts=count,max_concurrency=64,total_input_tokens=length*count,total_output_tokens=1024*count)
                assert all(value.get(key)==val for key,val in expected.items()), expected
            state.update(scores=scores,working_gates=performance_gates(scores),matched_validated_control_required=True)
        if args.post_score_semantic_check:
            assert args.run_throughput and args.full_workload_warmup
            save('post_score_semantics');after=args.out/'post-score-semantic';after.mkdir()
            post=mixed_diagnostic(after,args.fixtures)
            save('post_score_completed',post_score_diagnostic=post,initial_semantic_passed=passed,
                 post_score_semantic_passed=bool(post['all_correct'] and post['http_overlap']))
            passed=bool(passed and post['all_correct'] and post['http_overlap'])
        generated=[]
        if args.out_lowering=='1':
            for p in (args.out/'inductor-cache').rglob('*.py'):
                text=p.read_text(errors='replace')
                if 'biv150_small_linear_out' not in text:continue
                calls=[ast.unparse(node.func) for node in ast.walk(ast.parse(text)) if isinstance(node,ast.Call)]
                count=sum(name=='extern_kernels.biv150_small_linear_out' for name in calls)
                if count:generated.append(dict(path=str(p.relative_to(args.out)),sha256=sha(p),out_call_sites=count))
            assert generated,'No actual compiler-owned out calls in model'
        manifest['generated_out_modules']=generated
        final_logs=log_path.read_text()
        if multi2['available']:
            assert 'BIV150_COMPUTE_ORDER_SCOPE_REJECTED' not in final_logs,'Strict C3 identity/scope rejected'
            assert 'BIV150_COMPUTE_ORDER_FIRST_SELECTED ' in final_logs,'No actual compute-order selection'
            manifest['compute_order_preserved']=True
        multi2_selected=[ast.literal_eval(line.split('BIV150_PREFILL_MULTI2_SELECTED ',1)[1])
                         for line in final_logs.splitlines() if 'BIV150_PREFILL_MULTI2_SELECTED ' in line]
        assert 'BIV150_PREFILL_MULTI2_FALLBACK' not in final_logs,'Multi2 preparation fell back; no candidate acceptance'
        if multi2['expected_installed'] and args.run_throughput:
            assert multi2_selected,'No actual two-sequence service selection was observed'
        if not multi2['expected_installed']:
            assert not multi2_selected
        for event in multi2_selected:
            assert event['num_sequences']==2 and event['rows']>=1024
            assert tuple(event['grid'])==(event['rows']//32+2,2)
            assert (event['block_m'],event['block_q'],event['num_warps'],event['tile_size'],event['num_stages'])==(256,32,16,64,1)
        manifest['prefill_multi2_selected']=multi2_selected
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        state.update(diagnostic_scores_only=not (passed and state.get('precision_passed') is True),
                     precision_gate=state.get('precision_passed') is True)
        save("workloads_completed", work_completed=True, diagnostic=summary,
             semantic_passed=passed, official_throughput_claim=False,
             accuracy_skipped_due_mixed_failure=False)
    except BaseException as error:
        save("failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        remember()
        if server is not None:
            for member in list(owned.values()):
                current = identity(member["pid"])
                if current and current["start_ticks"] == member["start_ticks"] and current["sid"] == server.pid and current["state"] != "Z":
                    try:
                        os.kill(member["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                remember()
                if not any((identity(pid) or {}).get("state") not in (None, "Z") for pid in owned):
                    break
                time.sleep(1)
            for member in owned.values():
                current = identity(member["pid"])
                if current and current["start_ticks"] == member["start_ticks"] and current["sid"] == server.pid and current["state"] != "Z":
                    try:
                        os.kill(member["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            deadline = time.monotonic() + 10
            while True:
                living = [current for member in owned.values()
                          if (current := identity(member["pid"])) and
                          current["start_ticks"] == member["start_ticks"] and
                          current["state"] != "Z"]
                if not living or time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            with socket.socket() as sock:
                port_closed = sock.connect_ex(("127.0.0.1", 9031)) != 0
            cleanup_ok = not living and port_closed
            state.update(owned_service_members=list(owned.values()), cleanup_complete=cleanup_ok,
                         cleanup_remaining=living, port_9031_closed=port_closed,
                         complete=bool(state.get("work_completed") and cleanup_ok))
            if state["complete"]:
                save("completed")
            elif state.get("work_completed"):
                save("failed", cleanup_error="Owned process or port still active; no further GPU task allowed")
            else:
                save(state["phase"])
        log.close()
    if not state.get("cleanup_complete"):
        raise RuntimeError("Owned service cleanup did not complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-lowering",choices=("0","1"),required=True)
    parser.add_argument('--prefill-multi2',choices=('default','0','1'),default='default')
    parser.add_argument('--large-linear',choices=('0','1'),default='0')
    parser.add_argument('--provider-probe',action='store_true')
    parser.add_argument('--production-defaults',action='store_true')
    parser.add_argument("--post-score-semantic-check",action="store_true")
    parser.add_argument("--diagnostic-after-semantic-failure",action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gate-library", type=Path, required=True)
    parser.add_argument("--native-mode", choices=("on", "off"), required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--stable-small-o", choices=("0","1"), required=True)
    parser.add_argument("--run-accuracy", action="store_true")
    parser.add_argument("--run-throughput", action="store_true")
    parser.add_argument("--full-workload-warmup", action="store_true")
    parser.add_argument("--tuning-snapshot", type=Path, required=True)
    main(parser.parse_args())
