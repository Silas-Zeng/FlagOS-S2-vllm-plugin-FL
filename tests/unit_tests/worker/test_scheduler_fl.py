# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from vllm_fl.worker.scheduler_fl import AsyncSchedulerFL, _AsyncScheduler
from vllm_fl.worker.worker import _use_async_scheduler_fl


def _scheduler(monkeypatch, *, phase_aware: bool, cadence: int = 2):
    scheduler = object.__new__(AsyncSchedulerFL)
    scheduler._phase_aware = phase_aware
    scheduler._phase_cadence = cadence
    scheduler._phase_step = 0
    scheduler._sched_probe_path = None
    scheduler.max_num_scheduled_tokens = 10
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.requests = {}
    monkeypatch.setattr(
        _AsyncScheduler,
        "schedule",
        lambda self, throttle_prefills=False: SimpleNamespace(
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        ),
    )
    return scheduler


def test_phase_aware_schedule_cadence(monkeypatch):
    scheduler = _scheduler(monkeypatch, phase_aware=True, cadence=2)
    calls = []
    monkeypatch.setattr(
        _AsyncScheduler,
        "schedule",
        lambda self, throttle_prefills=False: calls.append(throttle_prefills)
        or SimpleNamespace(
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        ),
    )

    for _ in range(5):
        AsyncSchedulerFL.schedule(scheduler)

    assert calls == [False, True, False, True, False]


def test_phase_aware_schedule_disabled_keeps_existing_throttle(monkeypatch):
    scheduler = _scheduler(monkeypatch, phase_aware=False)
    calls = []
    monkeypatch.setattr(
        _AsyncScheduler,
        "schedule",
        lambda self, throttle_prefills=False: calls.append(throttle_prefills)
        or SimpleNamespace(
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        ),
    )

    AsyncSchedulerFL.schedule(scheduler)
    AsyncSchedulerFL.schedule(scheduler, throttle_prefills=True)

    assert calls == [False, True]


def test_phase_step_caps_budget_for_pure_decode(monkeypatch):
    scheduler = _scheduler(monkeypatch, phase_aware=True, cadence=2)
    scheduler.max_num_scheduled_tokens = 10
    scheduler.running = [
        SimpleNamespace(is_prefill_chunk=False),
        SimpleNamespace(is_prefill_chunk=False),
    ]
    scheduler.waiting = [object()]
    budgets = []
    monkeypatch.setattr(
        _AsyncScheduler,
        "schedule",
        lambda self, throttle_prefills=False: budgets.append(
            self.max_num_scheduled_tokens
        )
        or SimpleNamespace(
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        ),
    )

    AsyncSchedulerFL.schedule(scheduler)
    AsyncSchedulerFL.schedule(scheduler)

    assert budgets == [10, 2]
    assert scheduler.max_num_scheduled_tokens == 10


def test_schedule_probe_separates_prefill_and_decode(
    monkeypatch,
    tmp_path,
):
    scheduler = _scheduler(monkeypatch, phase_aware=False)
    probe_path = tmp_path / "schedule.jsonl"
    scheduler._sched_probe_path = str(probe_path)
    scheduler.running = [object(), object()]
    scheduler.requests = {
        "prefill": SimpleNamespace(
            num_computed_tokens=4,
            num_prompt_tokens=4,
        ),
        "decode": SimpleNamespace(
            num_computed_tokens=5,
            num_prompt_tokens=4,
        ),
    }
    monkeypatch.setattr(
        _AsyncScheduler,
        "schedule",
        lambda self, throttle_prefills=False: SimpleNamespace(
            num_scheduled_tokens={"prefill": 4, "decode": 1},
            total_num_scheduled_tokens=5,
        ),
    )

    AsyncSchedulerFL.schedule(scheduler)

    record = json.loads(probe_path.read_text())
    assert record["scheduled_prefill_tokens"] == 4
    assert record["scheduled_decode_tokens"] == 1
    assert record["total_tokens"] == 5
    assert record["running_before"] == 2


def test_phase_aware_env_selects_plugin_scheduler(monkeypatch):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        num_speculative_tokens=0,
    )

    monkeypatch.delenv("VLLM_FL_PHASE_AWARE_SCHED", raising=False)
    monkeypatch.delenv("VLLM_FL_SCHED_PROBE", raising=False)
    assert not _use_async_scheduler_fl(config)

    monkeypatch.setenv("VLLM_FL_PHASE_AWARE_SCHED", "1")
    assert _use_async_scheduler_fl(config)
