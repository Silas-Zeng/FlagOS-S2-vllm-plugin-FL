#!/usr/bin/env python3
"""Offline CPU scheduler accounting for completed BI-V150 diagnostic windows.

No Torch, SDK, network, service, or GPU access. Workload identity comes only
from the driver's validation record, never the epoch or request-ID spelling.
Graph-profile validation and CPU accounting are deliberately separate results.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram(values):
    return dict(sorted(collections.Counter(values).items()))


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "histogram": {}}
    n = len(values)
    return dict(count=n, minimum=values[0], maximum=values[-1],
                mean=sum(values) / n, p50=values[(n - 1) // 2],
                p90=values[int((n - 1) * .9)], p99=values[int((n - 1) * .99)],
                histogram=histogram(values))


def overlap(ranges, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in ranges)


def merge(ranges, start, end):
    merged = []
    for a, b in sorted([*ranges, (start, end)]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def scope_counter_audit(window, previous_window=None):
    """Validate cumulative counters against an independent window checkpoint."""
    errors = []
    epoch, pid = window.get("epoch"), window.get("pid")
    cumulative = window.get("scope_diagnostics", {}).get("observation_counts", {}).get("worker.execute_model")
    current_scope = window.get("scope_diagnostics", {})
    if (window.get("diagnostic_only") is not True
            or window.get("finished_monotonic_ns", 0) <= window.get("started_monotonic_ns", 0)
            or current_scope.get("errors") != [] or current_scope.get("error_total") != 0
            or current_scope.get("callback_disabled") is not False or current_scope.get("enabled") is not False):
        errors.append("cumulative scope checkpoint is not a completed healthy diagnostic window")
    previous_count = None
    if epoch == 1:
        previous_count = 0
    elif isinstance(epoch, int) and epoch > 1 and previous_window is not None:
        ps = previous_window.get("scope_diagnostics", {})
        if (previous_window.get("epoch") != epoch - 1 or previous_window.get("pid") != pid
                or previous_window.get("diagnostic_only") is not True
                or previous_window.get("finished_monotonic_ns", 0) <= previous_window.get("started_monotonic_ns", 0)
                or previous_window.get("finished_monotonic_ns", 0) > window.get("started_monotonic_ns", 0)
                or ps.get("errors") != [] or ps.get("error_total") != 0
                or ps.get("callback_disabled") is not False or ps.get("enabled") is not False):
            errors.append("preceding cumulative scope checkpoint has invalid identity, time, or callback state")
        else:
            previous_count = ps.get("observation_counts", {}).get("worker.execute_model")
    else:
        errors.append("missing preceding cumulative scope checkpoint; window boundary cannot be verified")
    steps = window.get("scheduled_steps", [])
    if type(cumulative) is not int or type(previous_count) is not int or previous_count < 0:
        errors.append("missing or invalid cumulative worker observation count")
        delta = None
    else:
        delta = cumulative - previous_count
        if delta != len(steps):
            errors.append("worker cumulative observation count delta does not match recorded steps")
        if any(step.get("sequence") != previous_count + i or step.get("step") != i
               or step.get("pid") != pid or step.get("source") != "worker.execute_model"
               or step.get("event") != "worker.execute_model.begin"
               for i, step in enumerate(steps, 1)):
            errors.append("worker observation sequence has a missing, duplicate, or out-of-window entry")
    return dict(previous_cumulative_count=previous_count, cumulative_count=cumulative,
                window_count_delta=delta, recorded_steps=len(steps),
                checkpoint_epoch=epoch - 1 if type(epoch) is int else None,
                passed=not errors, errors=errors)


def analyze(window, benchmark, length, block_size, capacity, previous_window=None):
    errors, warnings = [], []
    scope = window.get("scope_diagnostics", {})
    for condition, error in [
        (window.get("diagnostic_only") is True, "window is not diagnostic_only"),
        (window.get("finished_monotonic_ns", 0) > window.get("started_monotonic_ns", 0), "window has no valid end"),
        (scope.get("errors") == [] and scope.get("error_total") == 0 and scope.get("callback_disabled") is False,
         "scope reports errors, disabled callback, or incomplete status"),
    ]:
        if not condition:
            errors.append(error)
    steps = window["scheduled_steps"]
    if not steps:
        errors.append("no scheduled steps")
    counter_audit = scope_counter_audit(window, previous_window)
    errors.extend(counter_audit["errors"])
    records, rows, request_rows = {}, [], []
    boundary_events = []
    qlens, qlen_semantic = collections.Counter(), collections.Counter()
    totals, phases = collections.Counter(), collections.Counter()
    previous_timestamp = window.get("started_monotonic_ns", 0)
    for ordinal, step in enumerate(steps, 1):
        if step.get("step") != ordinal or step.get("pid") != window.get("pid"):
            errors.append(f"step {ordinal}: noncontiguous step or wrong PID")
        if not window["started_monotonic_ns"] <= step["monotonic_ns"] <= window["finished_monotonic_ns"]:
            errors.append(f"step {ordinal}: timestamp outside window")
        if step["monotonic_ns"] < previous_timestamp:
            errors.append(f"step {ordinal}: non-monotonic timestamp")
        previous_timestamp = step["monotonic_ns"]
        scheduled = step["num_scheduled_tokens"]
        if (sum(scheduled.values()) != step["total_num_scheduled_tokens"]
                or step.get("scheduled_token_sum") != sum(scheduled.values())
                or step["scheduled_request_count"] != len(scheduled)):
            errors.append(f"step {ordinal}: scheduled token/request counts disagree")
        new, cached = step["new_requests"], step["cached_requests"]
        if len(new) != step["new_request_count"] or len(cached) != step["cached_request_count"]:
            errors.append(f"step {ordinal}: new/cached counts disagree")
        info = {item["req_id"]: item for item in new + cached}
        if len(info) != len(new) + len(cached) or set(info) != set(scheduled):
            errors.append(f"step {ordinal}: scheduled IDs and CPU metadata disagree")
        for item in new:
            rid, prompt = item["req_id"], item["prompt_token_count"]
            if type(prompt) is not int or prompt != length:
                errors.append(f"step {ordinal}: request {rid} prompt does not match fixed {length}-token protocol")
            if rid in records and records[rid]["prompt"] not in (None, prompt):
                errors.append(f"step {ordinal}: prompt size changed for {rid}")
            rec = records.setdefault(rid, dict(prompt=prompt, ranges=[], last_end=None,
                                              last_computed=None, last_output=None, counts=collections.Counter()))
            rec["prompt"] = prompt
            rec["counts"]["new_metadata_emissions"] += 1
        counts = collections.Counter()
        logical_slots = 0
        for rid, qlen in scheduled.items():
            if not isinstance(qlen, int) or qlen <= 0 or rid not in info:
                errors.append(f"step {ordinal}: invalid qlen or missing request metadata")
                continue
            item = info[rid]
            rec = records.setdefault(rid, dict(prompt=None, ranges=[], last_end=None,
                                              last_computed=None, last_output=None, counts=collections.Counter()))
            start, prompt = item["num_computed_tokens"], rec["prompt"]
            if not isinstance(start, int) or start < 0:
                errors.append(f"step {ordinal}: invalid computed token count")
                continue
            end = start + qlen
            semantic = ("unknown" if prompt is None else "decode" if start >= prompt
                        else "prefill" if end <= prompt else "prompt_and_generated_history")
            counts[semantic + "_requests"] += 1
            qclass = "1" if qlen == 1 else "2_to_8" if qlen <= 8 else "9_plus"
            counts["qlen_" + qclass + "_requests"] += 1
            qlens[qlen] += 1
            qlen_semantic[(semantic, qclass)] += 1
            if qlen == 1 and semantic == "decode":
                counts["single_token_decode_requests"] += 1
            repeated = overlap(rec["ranges"], start, end)
            rec["counts"]["repeated_scheduled_positions"] += repeated
            rec["counts"]["overlap_events"] += int(repeated > 0)
            rec["counts"]["scheduled_tokens"] += qlen
            rec["counts"]["scheduled_request_steps"] += 1
            rec["counts"]["resumed_metadata_events"] += int(item.get("resumed") is True)
            if prompt is not None:
                prompt_tokens = max(0, min(end, prompt) - start)
                history_tokens = qlen - prompt_tokens
                repeated_prompt = overlap(rec["ranges"], start, min(end, prompt)) if start < prompt else 0
                repeated_history = overlap(rec["ranges"], max(start, prompt), end) if end > prompt else 0
                rec["counts"]["scheduled_prompt_tokens"] += prompt_tokens
                rec["counts"]["scheduled_generated_history_tokens"] += history_tokens
                rec["counts"]["repeated_prompt_positions"] += repeated_prompt
                rec["counts"]["repeated_generated_history_positions"] += repeated_history
                counts["scheduled_prompt_tokens"] += prompt_tokens
                counts["scheduled_generated_history_tokens"] += history_tokens
                counts["repeated_prompt_positions"] += repeated_prompt
                counts["repeated_generated_history_positions"] += repeated_history
                if semantic == "prompt_and_generated_history":
                    rec["counts"]["prompt_boundary_crossings"] += 1
                    boundary_events.append(dict(step=ordinal, request_id=rid, start=start, end=end,
                                                prompt_boundary=prompt, scheduled_tokens=qlen,
                                                prompt_tokens=prompt_tokens, generated_history_tokens=history_tokens,
                                                repeated_prompt_positions=repeated_prompt,
                                                repeated_generated_history_positions=repeated_history,
                                                new_prompt_positions=prompt_tokens - repeated_prompt,
                                                new_generated_history_positions=history_tokens - repeated_history,
                                                cached_num_output_tokens=item.get("num_output_tokens"),
                                                resumed_metadata=item.get("resumed")))
            else:
                rec["counts"]["unknown_phase_tokens"] += qlen
            if rec["last_end"] is not None:
                delta = start - rec["last_end"]
                rec["counts"]["backward_restart_events"] += int(delta < 0)
                rec["counts"]["backward_restart_distance"] += max(0, -delta)
                rec["counts"]["forward_gap_events"] += int(delta > 0)
                rec["counts"]["forward_gap_distance"] += max(0, delta)
            nout = item.get("num_output_tokens")
            if nout is not None:
                if rec["last_output"] is not None and nout < rec["last_output"]:
                    errors.append(f"step {ordinal}: output counter regressed for {rid}")
                rec["last_output"] = nout
            rec["last_computed"], rec["last_end"] = start, end
            rec["ranges"] = merge(rec["ranges"], start, end)
            logical_slots += ((end + block_size - 1) // block_size) * block_size
        phase = ("empty" if not scheduled else "unknown" if counts["unknown_requests"]
                 else "mixed" if counts["prompt_and_generated_history_requests"]
                 or (counts["decode_requests"] and counts["prefill_requests"])
                 else "decode_only" if counts["decode_requests"] else "prefill_only")
        phases[phase] += 1
        for field in ("finished_request_count", "preempted_request_count"):
            totals[field] += step[field]
        totals["preemption_steps"] += int(step["preempted_request_count"] > 0)
        totals["speculative_tokens"] += sum(step["speculative_token_counts"].values())
        totals["scheduled_tokens"] += step["total_num_scheduled_tokens"]
        rows.append(dict(step=ordinal, monotonic_ns=step["monotonic_ns"], phase=phase,
                         scheduled_requests=len(scheduled), scheduled_tokens=step["total_num_scheduled_tokens"],
                         **{key: counts[key] for key in ("decode_requests", "single_token_decode_requests", "prefill_requests",
                                                        "unknown_requests", "prompt_and_generated_history_requests", "qlen_1_requests",
                                                        "qlen_2_to_8_requests", "qlen_9_plus_requests", "scheduled_prompt_tokens",
                                                        "scheduled_generated_history_tokens", "repeated_prompt_positions",
                                                        "repeated_generated_history_positions")},
                         finished_count=step["finished_request_count"], preempted_count=step["preempted_request_count"],
                         scheduled_requests_logical_paged_extent=logical_slots,
                         logical_extent_div_capacity=logical_slots / capacity if capacity else None))
    for rid, rec in records.items():
        counts = rec["counts"]
        unique = sum(b - a for a, b in rec["ranges"])
        unique_prompt = sum(max(0, min(b, rec["prompt"]) - a) for a, b in rec["ranges"]) if rec["prompt"] is not None else None
        request_rows.append(dict(request_id=rid, prompt_tokens=rec["prompt"], **counts,
                                 unique_scheduled_positions=unique, unique_prompt_positions=unique_prompt,
                                 last_cached_output_counter=rec["last_output"], last_scheduled_end=rec["last_end"],
                                 interval_union=rec["ranges"]))
        totals.update(counts)
    # totals scheduled_tokens was also counted by request above: preserve both independently.
    step_tokens = sum(row["scheduled_tokens"] for row in rows)
    totals["scheduled_tokens"] -= step_tokens
    if totals["scheduled_tokens"] != step_tokens:
        errors.append("request and step token totals disagree")
    expected_count = {4096: 256, 16384: 128}[length]
    expected = dict(completed=expected_count, num_prompts=expected_count, failed=0,
                    max_concurrency=64, total_input_tokens=length * expected_count,
                    total_output_tokens=1024 * expected_count)
    for key, value in expected.items():
        if benchmark.get(key) != value:
            errors.append(f"benchmark {key}: expected {value}, got {benchmark.get(key)}")
    unique_tokens = sum(row["unique_scheduled_positions"] for row in request_rows)
    unique_prompt = sum(row["unique_prompt_positions"] or 0 for row in request_rows)
    expected_scheduled = expected["total_input_tokens"] + expected["total_output_tokens"] - expected_count
    closure = dict(unique_requests=len(records), observed_prompt_tokens=sum(rec["prompt"] or 0 for rec in records.values()),
                   unique_prompt_positions=unique_prompt, unique_scheduled_positions=unique_tokens,
                   scheduled_tokens=step_tokens, repeated_scheduled_positions=totals["repeated_scheduled_positions"],
                   expected_unique_positions_input_plus_output_minus_requests=expected_scheduled,
                   scheduled_minus_expected_minus_repeated=step_tokens - expected_scheduled - totals["repeated_scheduled_positions"],
                   aggregate_finished_count=totals["finished_request_count"],
                   all_requests_contiguous_from_zero=all(rec["ranges"] == [(0, length + 1023)] for rec in records.values()),
                   all_last_cached_output_counters_1023=all(rec["last_output"] == 1023 for rec in records.values()))
    if (len(records) != expected_count or unique_prompt != expected["total_input_tokens"]
            or unique_tokens != expected_scheduled or not closure["all_requests_contiguous_from_zero"]
            or totals["finished_request_count"] != expected_count or totals["speculative_tokens"]):
        errors.append("request/token closure failed for fixed non-speculative protocol")
    if not closure["all_last_cached_output_counters_1023"]:
        warnings.append("not all last cached output counters are 1023; inspect request rows")
    result = dict(cpu_accounting_passed=not errors, errors=errors, warnings=warnings,
                  pid=window["pid"], epoch=window["epoch"], step_count=len(steps),
                  scope_counter_audit=counter_audit, prompt_boundary_crossing_events=boundary_events,
                  phases=dict(phases), totals=dict(totals), closure=closure,
                  request_qlen_histogram=dict(sorted(qlens.items())),
                  request_qlen_semantics=[dict(semantic=a, qlen_class=b, request_steps=n)
                                          for (a, b), n in sorted(qlen_semantic.items())],
                  distributions={key: distribution([row[key] for row in rows]) for key in
                                 ("decode_requests", "single_token_decode_requests", "scheduled_requests", "scheduled_tokens",
                                  "scheduled_requests_logical_paged_extent")},
                  active_step_distributions={key: distribution([row[key] for row in rows if row["scheduled_tokens"]])
                                             for key in ("decode_requests", "scheduled_requests", "scheduled_tokens")},
                  kv_capacity_tokens_supplied=capacity, kv_block_size=block_size,
                  benchmark=benchmark,
                  limitations=[
                      "CPU schedule observations, not measured GPU work or production throughput.",
                      "Decode means computed >= known prompt length; new/cached membership is never a phase classifier.",
                      "Prompt-boundary crossings are known mixed intervals, split into prompt/history and repeated/new positions; they are not unknown or single-token decode.",
                      "Scope observation counters are cumulative across windows; epoch > 1 requires the preceding mapped window checkpoint.",
                      "Request qlen bins count request-step occurrences; step distributions are not time weighted.",
                      "Metadata dictionary/list order is not input_batch or compute-order; routing/optimization hit counts are unknown.",
                      "Only finished/preempted COUNTS were captured; their request IDs, causes and per-request attribution are unknown.",
                      "Repeated interval positions verify scheduled recomputation within this window; not its GPU cost or preemption cause.",
                      "Backward restart distance is lost position extent, not necessarily later recomputed token count.",
                      "Logical paged extent covers scheduled requests only; it is not physical KV usage, resident inventory, fragmentation or cache pressure.",
                      "The final sampled output token is not fed back, so unique input positions equal input + output - requests."])
    return result, rows, request_rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def load_mapped_windows(run):
    """Only driver-mapped, fully written windows; never infer identity by epoch."""
    mapped, checkpoints, seen_paths = {}, {}, set()
    for length in (4096, 16384):
        validation_path = run / f"diagnostic-{length}-profile-validation.json"
        if not validation_path.is_file():
            continue
        validation = read(validation_path)
        if len(validation["windows"]) != 1:
            raise ValueError("This TP1/DP1 analyzer requires exactly one mapped worker window per workload")
        entry = validation["windows"][0]
        path = run / "observations" / Path(entry["path"]).name
        if path in seen_paths:
            raise ValueError("A worker window is mapped to two workloads")
        seen_paths.add(path)
        window = read(path)
        identity = (window["pid"], window["epoch"])
        if (window["pid"] != entry["pid"] or window["epoch"] != entry["epoch"]
                or len(window["scheduled_steps"]) != entry["scheduled_steps"] or identity in checkpoints):
            raise ValueError("Validation/window identity or count mismatch")
        mapped[length] = (validation_path, validation, path, window)
        checkpoints[identity] = window
    return mapped, checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(4096, 16384), action="append")
    parser.add_argument("--kv-capacity-tokens", type=int)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    if args.block_size <= 0 or (args.kv_capacity_tokens is not None and args.kv_capacity_tokens <= 0):
        parser.error("capacity and block size must be positive")
    run = args.run_dir.resolve()
    diagnostics_path = run / "diagnostic_results.json"
    diagnostics = read(diagnostics_path)
    mapped, checkpoints = load_mapped_windows(run)
    lengths = args.length or list(mapped)
    if not lengths or len(set(lengths)) != len(lengths):
        parser.error("need unique completed validation windows")
    report = dict(diagnostic_only=True, official_score=False, source_run=str(run),
                  diagnostics_sha256=sha(diagnostics_path), workloads={})
    outputs = []
    for length in lengths:
        if length not in mapped:
            raise ValueError(f"No completed mapped window for input length {length}")
        validation_path, validation, path, window = mapped[length]
        # Resolve only within the supplied mirror/run. Never fall through to a live remote path.
        previous_window = checkpoints.get((window["pid"], window["epoch"] - 1))
        result, rows, request_rows = analyze(window, diagnostics["results"][str(length)], length,
                                            args.block_size, args.kv_capacity_tokens, previous_window)
        # A preceding checkpoint must itself be a complete observed window.
        for epoch in range(1, window["epoch"]):
            preceding = checkpoints.get((window["pid"], epoch))
            if preceding is None:
                result["errors"].append(f"missing mapped cumulative checkpoint epoch {epoch}")
            else:
                audit = scope_counter_audit(preceding, checkpoints.get((window["pid"], epoch - 1)))
                result["errors"].extend(f"checkpoint epoch {epoch}: {error}" for error in audit["errors"])
        result["cpu_accounting_passed"] = not result["errors"]
        result.update(profile_validation_passed=validation["passed"], profile_validation_errors=validation["errors"],
                      validation_path=str(validation_path), validation_sha256=sha(validation_path),
                      window_path=str(path), window_sha256=sha(path))
        if previous_window is not None:
            previous_validation_path, _, previous_path, _ = next(
                value for value in mapped.values() if value[3] is previous_window)
            result["previous_window_checkpoint"] = dict(
                pid=previous_window["pid"], epoch=previous_window["epoch"],
                window_path=str(previous_path), window_sha256=sha(previous_path),
                validation_path=str(previous_validation_path), validation_sha256=sha(previous_validation_path),
                cumulative_count=previous_window["scope_diagnostics"]["observation_counts"]["worker.execute_model"])
        report["workloads"][str(length)] = result
        outputs.append((length, rows, request_rows))
    report["cpu_accounting_passed"] = all(x["cpu_accounting_passed"] for x in report["workloads"].values())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["Diagnostic CPU scheduling accounting; not official scores."]
    for length, rows, request_rows in outputs:
        write_csv(args.out / f"steps-{length}.csv", rows)
        write_csv(args.out / f"requests-{length}.csv", request_rows)
        r = report["workloads"][str(length)]
        if r["prompt_boundary_crossing_events"]:
            write_csv(args.out / f"prompt-boundary-crossings-{length}.csv", r["prompt_boundary_crossing_events"])
        lines += [f"\n{length}: CPU accounting={r['cpu_accounting_passed']}; graph profile validation={r['profile_validation_passed']}",
                  f"Cumulative worker counter: {r['scope_counter_audit']}",
                  f"Phases: {r['phases']}", f"Token closure: {r['closure']}",
                  f"qlen classification: {r['request_qlen_semantics']}",
                  f"Preemptions={r['totals']['preempted_request_count']}; resumes={r['totals'].get('resumed_metadata_events', 0)}; repeated scheduled positions={r['totals'].get('repeated_scheduled_positions', 0)}",
                  f"Scheduled prompt/history tokens={r['totals'].get('scheduled_prompt_tokens', 0)}/{r['totals'].get('scheduled_generated_history_tokens', 0)}; repeated prompt/history positions={r['totals'].get('repeated_prompt_positions', 0)}/{r['totals'].get('repeated_generated_history_positions', 0)}; prompt-boundary events={len(r['prompt_boundary_crossing_events'])}",
                  f"Decode lanes/step: {r['distributions']['decode_requests']}",
                  f"Logical paged extent max={r['distributions']['scheduled_requests_logical_paged_extent']['maximum']}; supplied capacity={args.kv_capacity_tokens}",
                  *r["errors"], *r["warnings"]]
    lines += ["\nLimitations:", *next(iter(report["workloads"].values()))["limitations"]]
    (args.out / "report.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(out=str(args.out.resolve()), cpu_accounting_passed=report["cpu_accounting_passed"],
                          workloads=lengths)))
    return 0 if report["cpu_accounting_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
