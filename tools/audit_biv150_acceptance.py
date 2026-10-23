"""Read-only audit of saved service evidence; never launches or scores a workload."""

import argparse
import json
import math
from pathlib import Path

BASELINES = {4096: 4809.36, 16384: 3364.03}
# Stricter working TTFT reference used throughout this operator experiment.
TTFT_REFERENCE = {4096: 4723.09, 16384: 155385.0}


def audit(folder, require_mixed=False):
    missing, failures = [], []

    def read(name):
        path = folder / name
        if not path.is_file():
            missing.append(name)
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            failures.append(f"{name}: invalid JSON: {error}")
            return None

    precision = read("accuracy-verdict.json")
    scores = read("scores.json")
    warmups = read("warmups.json")
    manifest = read("manifest.json")
    if precision is not None and not (
        isinstance(precision.get("score"), (int, float))
        and math.isfinite(precision["score"])
        and precision["score"] >= 0.95
        and all(
            precision.get(k) == 105
            for k in ("metric_count", "predictions", "unique_questions")
        )
        and precision.get("failed_predictions") == []
    ):
        failures.append("precision gate failed or question/error accounting incomplete")
    rows = {}
    for length, count in [(4096, 256), (16384, 128)]:
        key = str(length)
        score = scores.get(key) if isinstance(scores, dict) else None
        if score is None:
            missing.append(f"scores.json:{key}")
            continue
        expected = dict(
            completed=count,
            failed=0,
            num_prompts=count,
            max_concurrency=64,
            total_input_tokens=length * count,
            total_output_tokens=1024 * count,
        )
        for name, value in expected.items():
            if score.get(name) != value:
                failures.append(f"{key}: {name}={score.get(name)!r}, expected {value}")
        value, ttft = score.get("total_token_throughput"), score.get("mean_ttft_ms")
        duration = score.get("duration")
        numeric = all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0
            for x in (value, ttft, duration)
        )
        if not numeric:
            failures.append(f"{key}: missing/invalid throughput, TTFT or duration")
            continue
        calculated = count * (length + 1024) / duration
        if not math.isclose(value, calculated, rel_tol=1e-6):
            failures.append(f"{key}: throughput disagrees with exact tokens/duration")
        throughput_floor = value >= 0.99 * BASELINES[length]
        ttft_pass = ttft <= 1.01 * TTFT_REFERENCE[length]
        if not throughput_floor:
            failures.append(f"{key}: throughput below baseline minus 1%")
        if not ttft_pass:
            failures.append(f"{key}: TTFT exceeds working reference plus 1%")
        history = warmups.get(key) if isinstance(warmups, dict) else None
        if not (
            isinstance(history, list)
            and 2 <= len(history) <= 4
            and all(
                isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                for x in history
            )
            and abs(history[-1] / history[-2] - 1) < 0.03
        ):
            failures.append(f"{key}: stabilized warmup not proven")
        rows[key] = dict(
            total_tokens_per_second=value,
            improvement_percent=(value / BASELINES[length] - 1) * 100,
            target_20_percent_met=value >= 1.2 * BASELINES[length],
            mean_ttft_ms=ttft,
            working_ttft_limit_ms=1.01 * TTFT_REFERENCE[length],
            ttft_pass=ttft_pass,
        )
    if manifest is not None:
        if manifest.get("git_status") != "":
            failures.append("service source worktree was not clean")
        if not manifest.get("git_commit") or not manifest.get("source_hashes"):
            failures.append("source identity missing")
    selection_count = None
    if require_mixed:
        path = folder / "server.log"
        if not path.is_file():
            missing.append("server.log")
        else:
            selection_count = path.read_text(errors="replace").count(
                "BIV150_MIXED_ATTENTION_SELECTED"
            )
            if selection_count == 0:
                failures.append("mixed attention workload selection not observed")
    status = (
        "incomplete"
        if missing
        else ("failed" if failures else "passes_saved_evidence_checks")
    )
    return dict(
        status=status,
        missing=missing,
        failures=failures,
        cases=rows,
        mixed_selected_shape_count=selection_count,
        target_20_percent_proven_by_saved_metrics=(
            not missing
            and not failures
            and len(rows) == 2
            and all(r["target_20_percent_met"] for r in rows.values())
        ),
        scope="Saved summary audit only. Raw request/report identity, exact commands, source hashes and process termination still require separate review.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--require-mixed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.require_mixed), indent=2))
