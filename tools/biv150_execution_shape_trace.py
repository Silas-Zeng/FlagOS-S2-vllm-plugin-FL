"""Diagnostic-only CPU shape recorder; never installed by production workers.

Wrap a model-runner class in a separate diagnostic process to record actual
scheduled shapes and graph padding. Original arguments and results pass through.
No GPU tensor reads, synchronization, scheduler limits or graph modes change.
"""

import atexit
import hashlib
import inspect
import json
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from pathlib import Path


class Recorder:
    def __init__(self, runner_class, output, flush_every=128):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.counts = Counter()
        self.total = 0
        self.active = ContextVar("biv150_active_shape_runner", default=None)
        self.runner_class = runner_class
        self.execute = runner_class.execute_model
        self.determine = runner_class._determine_batch_execution_and_padding
        self.signature = inspect.signature(self.determine)
        required = {"num_tokens", "num_reqs", "max_num_scheduled_tokens"}
        if not required.issubset(self.signature.parameters):
            raise ValueError("Unsupported model runner signature")
        self.source_sha256 = hashlib.sha256(
            inspect.getsource(self.determine).encode()
        ).hexdigest()
        self.config = None

    def flush(self):
        rows = [
            dict(
                zip(
                    (
                        "scheduled_tokens",
                        "requests",
                        "max_tokens_per_request",
                        "graph_mode",
                        "padded_tokens",
                        "padded_requests",
                    ),
                    key,
                ),
                calls=count,
            )
            for key, count in sorted(self.counts.items(), key=lambda x: str(x[0]))
        ]
        payload = dict(
            diagnostic_only=True,
            helper_source_sha256=self.source_sha256,
            config=self.config,
            calls_observed=self.total,
            counts=rows,
            note="Periodic snapshots may omit the final unflushed calls after forced process termination.",
        )
        temp = self.output.with_suffix(self.output.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2))
        temp.replace(self.output)

    def install(self):
        recorder = self

        @wraps(self.execute)
        def execute(runner, *args, **kwargs):
            token = recorder.active.set(id(runner))
            try:
                return recorder.execute(runner, *args, **kwargs)
            finally:
                recorder.active.reset(token)

        @wraps(self.determine)
        def determine(runner, *args, **kwargs):
            result = recorder.determine(runner, *args, **kwargs)
            if recorder.active.get() == id(runner):
                bound = recorder.signature.bind(runner, *args, **kwargs).arguments
                mode, descriptor = result[:2]
                key = (
                    int(bound["num_tokens"]),
                    int(bound["num_reqs"]),
                    int(bound["max_num_scheduled_tokens"]),
                    str(mode),
                    int(descriptor.num_tokens),
                    int(descriptor.num_reqs)
                    if getattr(descriptor, "num_reqs", None) is not None
                    else None,
                )
                recorder.counts[key] += 1
                recorder.total += 1
                if recorder.config is None:
                    config = getattr(runner, "scheduler_config", None)
                    recorder.config = dict(
                        max_num_batched_tokens=getattr(
                            config, "max_num_batched_tokens", None
                        ),
                        max_num_seqs=getattr(config, "max_num_seqs", None),
                    )
                if recorder.total == 1 or recorder.total % recorder.flush_every == 0:
                    recorder.flush()
            return result

        self.runner_class.execute_model = execute
        self.runner_class._determine_batch_execution_and_padding = determine
        atexit.register(self.flush)
        return self

    def uninstall(self):
        self.runner_class.execute_model = self.execute
        self.runner_class._determine_batch_execution_and_padding = self.determine
        atexit.unregister(self.flush)
        self.flush()


def self_test():
    import tempfile
    from types import SimpleNamespace

    result = ("FULL", SimpleNamespace(num_tokens=40, num_reqs=40), False, None, None)

    class Runner:
        scheduler_config = SimpleNamespace(
            max_num_batched_tokens=2048, max_num_seqs=256
        )

        def _determine_batch_execution_and_padding(
            self, num_tokens, num_reqs, max_num_scheduled_tokens
        ):
            assert (num_tokens, num_reqs, max_num_scheduled_tokens) == (33, 33, 1)
            return result

        def execute_model(self, fail=False):
            if fail:
                raise RuntimeError("intentional")
            return self._determine_batch_execution_and_padding(
                33, 33, max_num_scheduled_tokens=1
            )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        original = Runner.execute_model
        recorder = Recorder(Runner, path, 1).install()
        runner = Runner()
        assert runner._determine_batch_execution_and_padding(33, 33, 1) is result
        assert recorder.total == 0, "Ignore dummy calls outside execute_model"
        assert runner.execute_model() is result, "Preserve exact result object"
        assert recorder.total == 1
        try:
            runner.execute_model(fail=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Original exception must propagate")
        assert recorder.active.get() is None
        recorder.uninstall()
        assert Runner.execute_model is original
        data = json.loads(path.read_text())
        assert data["counts"][0]["scheduled_tokens"] == 33
        assert data["counts"][0]["padded_tokens"] == 40
        assert data["config"]["max_num_batched_tokens"] == 2048
        print(
            "CPU recorder self-test passed; synthetic config values are not service measurements."
        )


if __name__ == "__main__":
    self_test()
