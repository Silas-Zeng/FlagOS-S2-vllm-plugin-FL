#!/usr/bin/env python3
"""Recheck and deterministically archive the completed 140cecc component only.

Uses the downloaded original records, not a device, Torch, Triton or a driver.
The numerical verdicts are saved harness observations; CPU double reference
coverage is reported explicitly and is not promoted to independent exact math.
"""
import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import tarfile

from biv150_resource_records import compare_resource_records, resource_evidence

FROZEN = "140cecc07ba199d199eeb620debe3082195ffacb"
BASELINE = "241220723a09061c182764f8f738e5ec01190367"
FETCHED_MAIN = "56f1838192abc940b3fb9accef490022ac125000"
MANIFEST_SHA = "563a8efb2f1e019d97f42dc7b646987a500f2cada2c0776822a35b7450c12e10"
RESULT_SHA = "56dbfd2c33672e4bd177e52b57ffc5928ca37f7daabf31ff834c6deb13ab9bb8"
SOURCE_PINS = {
    "vllm_fl/patches/biv150_q_fragment_service.py": "38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82",
    "tools/biv150_resource_records.py": "25ff60821e73ce198b32a8819bd95768654ce4fbd7eb06ef9bd5c9220aea3fc8",
    "tools/biv150_prefill_reverse.py": "7f06589453c2d59310394bc47406d91c6b7a0d28b4781a9073047bcb78daba0f",
    "tools/bench_biv150_prefill_reverse.py": "061bb4f274776c65f1e6e0e1796bc132e9047650c7469dfcda05653f8afd692f",
    "tools/biv150_prefill_reverse_dynamic.py": "aee135df39d322c246a21fb2c779f10b63485f70f1b6afb377b02a11e17bc2de",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def exact(check):
    need(all(check.get(k) is True for k in ("finite", "passed", "numeric_equal")), "Exact comparison failed")
    need(all(check.get(k) == 0 for k in ("max_abs_diff", "mismatched_elements", "outside_tolerance", "atol", "rtol")), "Exact comparison has error/tolerance")


def reference(check):
    need(check["passed"] is True and check["finite"] is True, "Reference guard failed")
    for key in ("existing_baseline_violations", "baseline_exceptions_over_one_bf16_ulp", "candidate_oracle_violations", "new_or_worsened_violations"):
        need(check[key] == 0, "Unexpected reference exception: " + key)


def snapshot(root):
    manifest_raw = (root / "download-manifest.json").read_bytes()
    need(sha(manifest_raw) == MANIFEST_SHA, "Original download manifest changed")
    manifest = json.loads(manifest_raw)
    need(manifest["raw_files"] == 26 and manifest["raw_bytes"] == 1633508, "Raw inventory changed")
    need(manifest["read_only"] is True and manifest["model_rerun"] is False, "Wrong snapshot provenance")
    need(manifest["original_owner"] == dict(pid=622007, start_ticks=341609077, same_start=True, state="Z"), "Wrong final owner")
    files = {"download-manifest.json": manifest_raw}
    for row in manifest["files"]:
        name = row["relative"]
        path = PurePosixPath(name)
        need(not path.is_absolute() and ".." not in path.parts and name not in files, "Unsafe/duplicate path")
        data = (root / name).read_bytes()
        need(len(data) == row["bytes"] and sha(data) == row["sha256"], "Raw SHA/size mismatch: " + name)
        files[name] = data
    need(len(files) == 27 and sum(map(len, files.values())) - len(manifest_raw) == 1633508, "Incomplete inventory")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    need(actual == set(files), "Unmanifested/missing original files")
    need(sha(files["out/result.json"]) == RESULT_SHA, "Wrong result")
    return manifest, files


def lowered_checker(source):
    # Reuse exactly the real Manager's pure-text proof; no package import/JIT.
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lowered_proof"]
    need(len(functions) == 1, "Manager proof definition changed")
    scope = {"re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<actual-manager-proof>", "exec"), scope)
    return scope["lowered_proof"]


def audit(result, files, prove):
    r = result
    for key in ("completed", "component_only", "production_unchanged", "actual_parent_policy", "dynamic_metadata_requested", "parent_policy_restored"):
        need(r[key] is True, "Missing component gate: " + key)
    for key in ("official_score", "precision_gate", "service_installed", "full_matrix", "refined_route_numerical_matrix_complete", "complete_resource_equivalence_claimed"):
        need(r[key] is False, "Unexpected acceptance/scope: " + key)
    need(r["source_commit"] == FROZEN and r["production_baseline"] == BASELINE and r["errors"] == [], "Source/errors changed")
    need(r["resource_mode"] == "strict", "Wrong resource mode")
    transform = r["source_transform"]
    baseline_source = files["out/baseline-normalized-kernel.py"]
    candidate_source = files["out/candidate-reversed-kernel.py"]
    need(sha(baseline_source) == transform["normalized_source_sha256"] == "fa6ddc420a95102d116714fcbea41bc81aa9c09379d68518af65755c015b8744", "Wrong normalized source")
    need(sha(candidate_source) == transform["reversed_source_sha256"] == "572c3a765b8206d0debd7617999c0c1e26ade387abf72493b1bf0b53f42408a3", "Wrong conditional source")
    old = "    q_block_global_idx = tl.program_id(0)\n"
    new = "    " + transform["source_edit"] + "\n"
    need(candidate_source.decode().count(new) == 1 and candidate_source.decode().replace(new, old) == baseline_source.decode(), "Exact byte inverse failed")
    need(transform["context_query_ratio"] == 4 and transform["metadata_cse_device_verified"] is False, "Wrong threshold/CSE claim")
    probe_source = files["out/route-probe.py"]
    need(sha(probe_source) == transform["route_probe_source_sha256"], "Route probe source changed")
    prefix = ast.parse("def route():\n" + new).body[0].body
    probe = [n for n in ast.parse(probe_source).body if isinstance(n, ast.FunctionDef)]
    need(len(probe) == 1 and [ast.dump(n) for n in probe[0].body[:len(prefix)]] == [ast.dump(n) for n in prefix], "Probe/candidate route-prefix AST differs")
    managers, proofs = {}, {}
    for label in ("baseline", "candidate"):
        manager = r["terminal_manager_proofs"][label]
        need(manager["selected_compiles"] == 1 and manager["selected_calls"] > 0 and manager["private_jit_cache"] is True, "Missing actual private selection")
        need(all(manager[k] == 0 for k in ("fallback_calls", "fallback_compiles", "original_3d_calls", "omitted_records")) and manager["record_errors"] == [], "Manager fallback/error")
        need(len(manager["compile_records"]) == 1, "Unexpected compile records")
        record = manager["compile_records"][0]
        need(record["mode"] == "selected", "Unselected binary")
        name = "out/" + label + "-" + record["candidate_hash"]
        proof = prove(files[name + ".ttgir"].decode(), files[name + ".llir"].decode(), set(transform["q_debug_lines_" + label]), block_m=256)
        need(proof == record["mechanism"] and proof["mechanism_pass"], "Actual saved Q-hoist proof differs")
        need(proof["llvm_q_memory_reads_in_loop"] == 0 and proof["llvm_q_memory_reads_total"] == 32, "Q reads changed")
        override = "out/compiler-ir/" + label + "/" + record["override_filename"]
        need(sha(files[override]) == record["override_sha256"], "Override IR changed")
        managers[label], proofs[label] = record, proof
    need(all(managers["baseline"][k] == managers["candidate"][k] for k in ("abi", "options", "target")), "Typed ABI/options/target changed")
    rows = []
    need([c["name"] for c in r["cases"]] == ["chunk4k", "chunk16k"], "Unexpected timed cases")
    for case, kv in zip(r["cases"], (4096, 16384)):
        need(case["completed"] is True and case["qlens"] == [2048] and case["klens"] == [kv] and case["grid"] == [65, 2], "Timed case scope changed")
        desc = case["launch_descriptors"]
        for key, value in dict(BLOCK_M=256, BLOCK_Q=32, TILE_SIZE=64, num_warps=16, num_stages=1, num_seqs=1, KV_QUANT_MODE=0, IS_3D=False).items():
            need(desc[key] == value, "Launch descriptor changed: " + key)
        need(len(case["checks"]) == 2 and [x["pattern"] for x in case["checks"]] == [0, 1], "Input patterns incomplete")
        samples = []
        for check in case["checks"]:
            exact(check["direct"]); reference(check["reference_guard"]); exact(check["side_stream"])
            need(check["output_canaries_unchanged"] is True and check["inputs_unchanged"] is True, "Canary/input changed")
            need([g["route"] for g in check["graph_checks"]] == ["baseline", "candidate"], "Graph arms incomplete")
            for g in check["graph_checks"]:
                exact(g)
                need(g["changed_input"] is bool(check["pattern"]), "Missing changed-input replay")
            for arm in ("baseline_cpu", "candidate_cpu"):
                cpu = check[arm]
                need(cpu["passed"] is True and cpu["finite"] is True and cpu["outside_tolerance"] == 0 and cpu["atol"] == cpu["rtol"] == 0.005, "CPU reference failed")
                need(cpu["elements"] == (4194304 if check["pattern"] == 0 else 16384), "CPU reference coverage changed")
            samples.append(dict(pattern=check["pattern"], independent_cpu_reference_elements=check["baseline_cpu"]["elements"], direct_elements=check["direct"]["elements"], reference_exceptions=0))
        resources = {}
        for label in ("baseline", "candidate"):
            saved = case[label + "_resources"]
            need(saved["hash"] == managers[label]["candidate_hash"], "Wrong binary resources")
            evidence = resource_evidence(saved, managers[label], already_launched=True)
            need(evidence == saved["post_launch_resource_evidence"] and evidence["counts_status"] == "observed_post_launch", "Resource phase mismatch")
            need(evidence["manager_prelaunch_counts"] == dict(n_regs=None, n_spills=None) and evidence["n_regs"] == 113 and evidence["n_spills"] == 0, "Unknown/changed actual counts")
            for kind, record in saved["files"].items():
                data = files["out/" + record["path"]]
                need(len(data) == record["bytes"] and sha(data) == record["sha256"] == saved["asm_sha256"][kind], "Saved IR/ASM/cubin mismatch")
            need(set(saved["files"]) == {"source", "ttir", "ttgir", "llir", "asm", "cubin"}, "Compiled artifacts incomplete")
            resources[label] = copy.deepcopy(saved)
        comparison = compare_resource_records(resources["baseline"], resources["candidate"])
        need(comparison == case["resource_comparison"] and comparison["strict_allowed"] is True, "Strict resources failed")
        need(case["same_registers_and_spills"] is True and case["resource_changed"] is False and case["resource_mode"] == "strict", "Wrong resource attribution")
        need(len(case["rounds"]) == 3, "Timing rounds incomplete")
        speeds = []
        for i, row in enumerate(case["rounds"]):
            need(row["iteration"] == i and row["order"] == (["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"]), "AB/BA/AB changed")
            need(all(math.isfinite(row[k]) and row[k] > 0 for k in ("baseline_ms", "candidate_ms")), "Invalid timer")
            speed = row["baseline_ms"] / row["candidate_ms"]
            need(speed == row["speedup"], "Timing ratio differs")
            speeds.append(speed)
        median = statistics.median(speeds)
        need(median == case["median_speedup"], "Median differs")
        rows.append(dict(name=case["name"], query_rows=2048, kv_rows=kv, grid=case["grid"], rounds=case["rounds"], median_speedup=median, throughput_ratio_change=median - 1, reference_coverage=samples, numerical_guards_passed=True, resources={label: value["post_launch_resource_evidence"] for label, value in resources.items()}, resource_comparison=comparison))
    dynamic = r["dynamic_metadata"]
    need(all(dynamic[k] is True for k in ("completed", "fixed_pointer_addresses", "captured_once_per_route", "route_probe_exact_generated_predicate")), "Dynamic capture/probe incomplete")
    need(dynamic["fixed_grid"] == [65, 2] and dynamic["metadata_cse_device_verified"] is False and len(dynamic["cases"]) == 13, "Dynamic scope/CSE changed")
    states = []
    for i, state in enumerate(dynamic["cases"]):
        q, kv, start = state["actual_query_rows"], state["kv_rows"], state["q_start"]
        order = "reverse" if start == 0 and q >= 1024 and 0 <= kv - q <= 4 * q else "forward"
        need(state["prefix_rows"] == kv - q and state["expected_order"] == order, "Expected metadata decision differs")
        exact(state["direct"]); reference(state["reference_guard"])
        need([g["route"] for g in state["graphs"]] == ["baseline", "candidate"], "Dynamic graph arms incomplete")
        for g in state["graphs"]:
            exact(g); need(g["changed_metadata"] is (i > 0), "Missing changed-metadata replay")
        for key in ("device_route_ids_exact", "both_heads_same_permutation", "inactive_output_canaries_unchanged", "input_bytes_unchanged"):
            need(state[key] is True, "Dynamic guard failed: " + key)
        states.append({k: state[k] for k in ("name", "q_start", "actual_query_rows", "kv_rows", "prefix_rows", "expected_order")})
    need(states[0]["name"] == "short_prefix" and states[-1]["name"] == "return_to_short", "Missing state return")
    probe = dynamic["route_probe_resources"]
    for kind, record in probe["files"].items():
        data = files["out/" + record["path"]]
        need(sha(data) == record["sha256"] == probe["asm_sha256"][kind] and len(data) == record["bytes"], "Probe artifact mismatch")
    need(set(probe["files"]) == {"source", "ttir", "ttgir", "llir", "asm", "cubin"}, "Probe artifact set incomplete")
    return dict(source_commit=FROZEN, measured_parent=BASELINE, fetched_team_main=FETCHED_MAIN, component_only=True, official_score=False, precision_gate=False, service_installed=False, numerical_observations_rechecked=True, independently_reran_device_math=False, refined_route_numerical_matrix_complete=False, full_matrix=False, metadata_cse_device_verified=False, complete_resource_equivalence_claimed=False, cases=rows, dynamic_metadata=dict(cases=states, completed=True, fixed_grid=[65, 2], fixed_pointer_addresses=True, captured_once_per_route=True, exact_device_route_map_reported=True, candidate_probe_prefix_ast_equal=True), q_hoist_recomputed_from_saved_actual_ir=proofs, actual_parent_typed_abi_and_compiler_options_equal=True, exact_source_byte_inverse=True, next_condition="Keep short-context positive direction open; the measured long-context identity route still costs time. A host decision with original Q32 fallback is a source-feasibility study only.")


def rejection_tests(result, files, prove):
    mutations = [
        ("direct_error", lambda r: r["cases"][0]["checks"][0]["direct"].update(max_abs_diff=1)),
        ("oracle_error", lambda r: r["cases"][0]["checks"][0]["reference_guard"].update(candidate_oracle_violations=1)),
        ("changed_input_missing", lambda r: r["cases"][0]["checks"][1]["graph_checks"][1].update(changed_input=False)),
        ("unknown_postlaunch", lambda r: r["cases"][0]["baseline_resources"].update(n_regs=None)),
        ("wrong_median", lambda r: r["cases"][1].update(median_speedup=1.0)),
        ("wrong_route", lambda r: r["dynamic_metadata"]["cases"][1].update(expected_order="reverse")),
        ("probe_error", lambda r: r["dynamic_metadata"]["cases"][0].update(device_route_ids_exact=False)),
        ("changed_metadata_missing", lambda r: r["dynamic_metadata"]["cases"][-1]["graphs"][1].update(changed_metadata=False)),
    ]
    rejected = []
    for name, mutate in mutations:
        bad = copy.deepcopy(result); mutate(bad)
        try:
            audit(bad, files, prove)
        except (ValueError, KeyError):
            rejected.append(name)
        else:
            raise ValueError("Fault was accepted: " + name)
    return rejected


def archive_bytes(files):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, raw in sorted(files.items()):
                info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o644
                info.mtime = info.uid = info.gid = 0; info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for name, pin in SOURCE_PINS.items():
        need(sha((repo / name).read_bytes()) == pin, "Frozen source changed: " + name)
    manifest, files = snapshot(args.snapshot)
    prove = lowered_checker((repo / "vllm_fl/patches/biv150_q_fragment_service.py").read_text())
    result = json.loads(files["out/result.json"])
    summary = audit(result, files, prove)
    summary.update(original_manifest_sha256=MANIFEST_SHA, original_result_sha256=RESULT_SHA,
                   original_owner=manifest["original_owner"], raw_files=26, raw_bytes=1633508,
                   cpu_only=True, gpu_executed=False, frozen_source_pins=SOURCE_PINS)
    if args.self_test:
        summary["rejected_faults"] = rejection_tests(result, files, prove)
    packed = archive_bytes(files)
    need(len(packed) <= 2 * 1024 * 1024, "Archive exceeds publisher limit")
    args.out.mkdir(parents=True, exist_ok=True)
    archive_name = "causal-skew-140-raw.tar.gz"
    (args.out / archive_name).write_bytes(packed)
    (args.out / "download-manifest.json").write_bytes(files["download-manifest.json"])
    (args.out / "summary.json").write_bytes(encoded(summary))
    (args.out / "archive-manifest.json").write_bytes(encoded(dict(
        archive=archive_name, bytes=len(packed), sha256=sha(packed), deterministic=True,
        raw_files=26, raw_bytes=1633508, archive_members=27,
        original_manifest_sha256=MANIFEST_SHA, original_result_sha256=RESULT_SHA,
        summary_sha256=sha(encoded(summary)), files=manifest["files"])))
    need("torch" not in sys.modules and "triton" not in sys.modules, "Unexpected GPU package import")
    print(json.dumps(dict(passed=True, raw_files=26, dynamic_states=13, timed_cases=2,
        archive_bytes=len(packed), archive_sha256=sha(packed), cpu_only=True)))


if __name__ == "__main__":
    main()
