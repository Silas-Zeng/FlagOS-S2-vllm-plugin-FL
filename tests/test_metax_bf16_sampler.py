# SPDX-License-Identifier: Apache-2.0
"""Compare fused BF16 sampling to the installed FP32 MetaX sampler."""
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.metax.impl.bf16_top_p_sample import (
    _top_p_grid_kernel, _convert_scale_kernel, sample_bf16_top_p,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.top_p_sample import (
    _top_p_sample_kernel, top_p_sample,
)
from vllm_fl.dispatch.backends.vendor.metax.patches.bf16_sampler import (
    _eligible, _processors_are_noops, _forward, _original_forward,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.logits_processor.builtin import (
    MinPLogitsProcessor, LogitBiasLogitsProcessor, MinTokensLogitsProcessor,
)


def inputs(kind, batch=8, vocab=130560):
    torch.manual_seed(1234)
    x = torch.randn(batch, vocab, device="cuda", dtype=torch.bfloat16)
    if kind in ("top2", "mixed"):
        x[:, 10:12] = torch.tensor([16., 15.7], device="cuda")
        if kind == "mixed":
            x[:batch // 2, 10] = 25
    elif kind == "flat":
        x.zero_()
    elif kind == "negative":
        x.sub_(20)
    elif kind == "ties":
        x.round_()
    t = torch.linspace(.7, 1.3, batch, device="cuda")
    p = torch.linspace(.5, 1., batch, device="cuda")
    return x, t, p


@pytest.mark.parametrize("layout", ["scalar", "vector", "expanded"])
def test_fused_conversion_matches_vendor(layout):
    x, t, _ = inputs("normal", vocab=32771)
    if layout == "scalar":
        t = t[:1]
    elif layout == "expanded":
        t = t[:1].expand(x.shape[0])
    out = torch.empty_like(x, dtype=torch.float32)
    _convert_scale_kernel[((x.shape[1] + 2047) // 2048, x.shape[0])](
        x, t, out, x.shape[1], 0 if t.numel() == 1 else t.stride(0),
        BLOCK=2048, num_warps=4)
    torch.testing.assert_close(out, x.float().div_(t[:, None]), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["normal", "top2", "mixed", "flat", "negative", "ties"])
@pytest.mark.parametrize("vocab", [32771, 130560])
def test_support_and_rng_match(kind, vocab):
    x, t, p = inputs(kind, vocab=vocab)
    batch = x.shape[0]
    scaled = x.float().div_(t[:, None])
    ref = torch.empty_like(scaled)
    actual = torch.empty_like(scaled)
    rd = torch.empty(batch, device="cuda", dtype=torch.int32)
    d = torch.empty_like(rd)
    rm = torch.empty(batch, device="cuda")
    m = torch.empty_like(rm)
    tiles = (vocab + 32767) // 32768
    _top_p_sample_kernel[(batch,)](
        scaled, p, ref, rd, rm, vocab, 1, vocab, 32768, tiles, 20, num_warps=8)
    _top_p_grid_kernel[(batch,)](
        scaled, t, p, actual, d, m, vocab, vocab, 1, 1,
        32768, tiles, 20, num_warps=8)
    torch.testing.assert_close(rd, d, atol=0, rtol=0)
    torch.testing.assert_close(rm, m, atol=0, rtol=0)
    live = rd < 0
    torch.testing.assert_close(ref[live], actual[live], atol=0, rtol=0)
    state = torch.cuda.get_rng_state()
    expected = top_p_sample(scaled, p).to(torch.int32)
    state_after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    result = sample_bf16_top_p(x, t, p)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert torch.equal(torch.cuda.get_rng_state(), state_after)
    assert result.dtype == torch.int32


def test_graph_replay_changes_rng_and_observes_inputs():
    x, t, p = inputs("flat")
    for _ in range(3):
        sample_bf16_top_p(x, t, p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = sample_bf16_top_p(x, t, p)
    graph.replay()
    before = result.clone()
    graph.replay()
    assert torch.any(before != result)
    x[:, 123] = 50
    graph.replay()
    assert torch.all(result == 123)


def metadata():
    return SimpleNamespace(
        all_random=True, all_greedy=False, temperature=object(), top_p=object(),
        top_k=None, generators={}, max_num_logprobs=None, logprob_token_ids=None,
        no_penalties=True, bad_words_token_ids={}, allowed_token_ids_mask=None,
        spec_token_ids=None, thinking_budget_state_holder=None,
        logitsprocs=SimpleNamespace(all=[]),
    )


@pytest.mark.parametrize("field,value", [
    ("all_random", False), ("all_greedy", True), ("temperature", None),
    ("top_p", None), ("top_k", object()), ("generators", {0: object()}),
    ("max_num_logprobs", 0), ("logprob_token_ids", {0: [1]}),
    ("no_penalties", False), ("bad_words_token_ids", {0: [[1]]}),
    ("allowed_token_ids_mask", object()), ("spec_token_ids", [[1]]),
    ("thinking_budget_state_holder", SimpleNamespace(has_tracked_requests=lambda: True)),
    ("logitsprocs", SimpleNamespace(all=[object()])),
])
def test_features_fall_back(field, value):
    x = torch.empty(8, 32768, device="cuda", dtype=torch.bfloat16)
    sampler = SimpleNamespace(use_fp64_gumbel=False, logprobs_mode="raw_logprobs")
    meta = metadata()
    assert _eligible(sampler, x, meta, False, None)
    setattr(meta, field, value)
    assert not _eligible(sampler, x, meta, False, None)


def test_builtin_noop_processors():
    processors = []
    for cls, field, empty, active in [
        (MinPLogitsProcessor, "min_p_count", 0, 1),
        (LogitBiasLogitsProcessor, "biases", {}, {0: {1: 1.}}),
        (MinTokensLogitsProcessor, "min_toks", {}, {0: object()}),
    ]:
        item = object.__new__(cls)
        setattr(item, field, empty)
        processors.append(item)
        assert _processors_are_noops(SimpleNamespace(all=processors))
        setattr(item, field, active)
        assert not _processors_are_noops(SimpleNamespace(all=processors))
        setattr(item, field, empty)


@pytest.mark.parametrize("unit_temperature", [False, True])
def test_complete_sampler_forward_matches_existing_path(unit_temperature):
    x, t, p = inputs("mixed", batch=64)
    if unit_temperature:
        t.fill_(1)
    meta = metadata()
    meta.temperature, meta.top_p = t, p
    meta.spec_token_ids = [[] for _ in range(64)]
    meta.output_token_ids = [[] for _ in range(64)]
    meta.logitsprocs = SimpleNamespace(all=[], argmax_invariant=[], non_argmax_invariant=[])
    sampler = SimpleNamespace(
        use_fp64_gumbel=False, logprobs_mode="raw_logprobs",
        apply_temperature=Sampler.apply_temperature,
        apply_penalties=Sampler.apply_penalties,
        topk_topp_sampler=lambda logits, generators, k, p: (top_p_sample(logits, p), None),
    )
    sampler.apply_logits_processors = lambda *args: Sampler.apply_logits_processors(sampler, *args)
    sampler.sample = lambda *args: Sampler.sample(sampler, *args)
    assert _eligible(sampler, x, meta, False, None)
    state = torch.cuda.get_rng_state()
    expected = _original_forward(sampler, x, meta)
    torch.cuda.set_rng_state(state)
    result = _forward(sampler, x, meta)
    torch.testing.assert_close(result.sampled_token_ids, expected.sampled_token_ids, atol=0, rtol=0)
    assert result.logprobs_tensors is None


def test_diagnostic_wrapper_accepts_standard_keyword(monkeypatch, tmp_path):
    from vllm_fl.dispatch.backends.vendor.metax.patches import bf16_sampler as module
    x, _, _ = inputs("flat")
    sampler = SimpleNamespace(use_fp64_gumbel=False, logprobs_mode="raw_logprobs")
    meta = metadata()
    sentinel = object()
    diagnostic = tmp_path / "sampler.jsonl"
    monkeypatch.setenv("VLLM_FL_METAX_FUSED_SAMPLER", "1")
    monkeypatch.setenv("VLLM_FL_METAX_SAMPLER_DIAGNOSTIC", str(diagnostic))
    monkeypatch.setattr(Sampler, "_metax_fused_bf16_installed", False, raising=False)
    monkeypatch.setattr(Sampler, "forward", Sampler.forward)
    monkeypatch.setattr(module, "_forward", lambda *args, **kwargs: sentinel)
    assert module.install()
    for _ in range(2):
        assert Sampler.forward(sampler, x, sampling_metadata=meta) is sentinel
    assert len(diagnostic.read_text().splitlines()) == 1


def test_warmup_preserves_rng_and_runs_once():
    from vllm_fl.dispatch.backends.vendor.metax.patches.bf16_sampler import warmup
    logits, _, _ = inputs("normal")
    sampler = SimpleNamespace(_metax_fused_bf16_installed=True)
    state = torch.cuda.get_rng_state()
    cpu_state = torch.get_rng_state()
    assert warmup(sampler, logits)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert not warmup(sampler, logits)
