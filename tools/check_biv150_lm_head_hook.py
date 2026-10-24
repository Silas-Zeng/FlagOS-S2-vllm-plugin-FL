"""Check the actual LM-head hook on real SDK classes with isolated test weights.

Minimal modules exercise after-loading/apply, not a complete model constructor.
Only the temporary test config is mocked; native math and graph execution are real.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main(args):
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    from vllm_fl.patches import biv150_lm_head_int8 as hook

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=42,
                num_attention_heads=16,
                num_key_value_heads=2,
                vocab_size=130560,
                tie_word_embeddings=False,
            )
        )
    )
    torch.manual_seed(20261024)
    with (
        patch("vllm.config.get_current_vllm_config", return_value=config),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        assert hook.install()["installed"]
        assert hook.install()["already"]
        method = UnquantizedEmbeddingMethod()
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.weight = torch.nn.Parameter(
            torch.randn((128, 2048), device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        before = embedding.weight
        method.process_weights_after_loading(embedding)
        assert embedding.weight is before and not hasattr(embedding, "_biv150_head_q")
        indices = torch.tensor([0, 7, 127], device="cuda")
        torch.testing.assert_close(
            method.embedding(embedding, indices), before[indices], atol=0, rtol=0
        )
        head = ParallelLMHead.__new__(ParallelLMHead)
        torch.nn.Module.__init__(head)
        head.prefix = "lm_head"
        head.bias = None
        head.weight = torch.nn.Parameter(
            torch.randn((130560, 2048), device="cuda", dtype=torch.bfloat16)
            / (2048**0.5),
            requires_grad=False,
        )
        cols = torch.linspace(0, 130559, 32, device="cuda").long()
        original_columns = head.weight[cols].detach().cpu().double()
        method.process_weights_after_loading(head)
        assert head.weight.untyped_storage().nbytes() == 2
        assert head.weight.stride() == (0, 0)
        assert tuple(head._biv150_head_q.shape) == (2048, 130560)
        q = head._biv150_head_q[:, cols].cpu().to(torch.int64)
        sw = head._biv150_head_scales[cols].cpu().float()
        result = dict(
            minimal_sdk_modules=True,
            real_model_validated=False,
            embedding_unchanged=True,
            compressed_storage_bytes=head.weight.untyped_storage().nbytes(),
            records=[],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for m in (0, 1, 8, 16, 24, 32, 40, 64):
            # Pitched rows also cover the op's contiguous input materialization.
            x = torch.randn((m, 2055), device="cuda", dtype=torch.bfloat16)[:, :2048]
            row = dict(m=m, stride=list(x.stride()))
            try:
                output = method.apply(head, x)
                assert tuple(output.shape) == (m, 130560)
                if m:
                    xc = x.cpu().float()
                    sa = xc.abs().amax(1).clamp_min(1e-6) / 127
                    raw = xc / sa[:, None]
                    qa = (raw.abs() + 0.5).floor().copysign(raw).clamp(-127, 127)
                    expected = (
                        (qa.to(torch.int64) @ q).float() * (sa[:, None] * sw[None, :])
                    ).bfloat16()
                    actual = output[:, cols].cpu()
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.008)
                    dense = xc.double() @ original_columns.t()
                    row["dense_relative_rmse"] = (
                        (actual.double() - dense).square().mean().sqrt()
                        / dense.square().mean().sqrt().clamp_min(1e-12)
                    ).item()
                if m in (8, 32):
                    compiled = torch.compile(
                        lambda value: method.apply(head, value),
                        backend="eager",
                        fullgraph=True,
                    )
                    torch.testing.assert_close(compiled(x), output, atol=0, rtol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        replayed = method.apply(head, x)
                    x.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert not replayed.count_nonzero().item()
                    x.normal_()
                    graph.replay()
                    torch.cuda.synchronize()
                    fresh = method.apply(head, x)
                    torch.testing.assert_close(replayed, fresh, atol=0, rtol=0)
                    row["fullgraph_and_changed_input"] = True
                row["passed"] = True
            except Exception as error:
                row["error"] = repr(error)
            result["records"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
