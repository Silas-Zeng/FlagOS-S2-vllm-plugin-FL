"""Check actual adapted scope and compiler option gates without Torch/GPU imports."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from bench_biv150_prefill_bm64_qhoist import adapted_manager


class Tensor:
    def __init__(self, shape, strides, address, dtype="torch.bfloat16"):
        self.shape, self.strides, self.address, self.dtype = shape, strides, address, dtype
        self.ndim = len(shape)
        self.device = DEVICE

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.address)


DEVICE = SimpleNamespace(type="cuda", index=0)


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_qfragment_cpu", root / "vllm_fl/patches/biv150_q_fragment_service.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    reports = []
    for rows, warps, tile, stages in ((256, 16, 64, 1), (256, 16, 128, 1),
                                      (128, 8, 64, 2), (256, 8, 64, 2),
                                      (512, 16, 64, 1)):
        module, source_sha = adapted_manager(original, rows, warps, tile, stages)
        values = dict(IS_3D=False, HEAD_SIZE=128, HEAD_SIZE_PADDED=128, num_query_heads=16,
                      num_queries_per_kv=8, BLOCK_M=rows, BLOCK_Q=rows // 8, BLOCK_SIZE=16,
                      TILE_SIZE=tile, NUM_SEGMENTS_PER_SEQ=1, KV_QUANT_MODE=0,
                      USE_CAUSAL=True, SLIDING_WINDOW=0, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1,
                      num_warps=warps, num_stages=stages, num_seqs=1,
                      query_ptr=Tensor((2017, 16, 128), (2048, 128, 1), 1),
                      output_ptr=Tensor((2017, 16, 128), (2048, 128, 1), 2),
                      key_cache_ptr=Tensor((1024, 16, 2, 128), (8192, 256, 128, 1), 3),
                      value_cache_ptr=Tensor((1024, 16, 2, 128), (8192, 256, 128, 1), 3),
                      query_start_len_ptr=Tensor((2,), (1,), 4, "torch.int32"),
                      seq_lens_ptr=Tensor((1,), (1,), 5, "torch.int32"),
                      block_tables_ptr=Tensor((1, 8192), (8192, 1), 6, "torch.int32"))
        assert module._eligible(values, rows)
        assert not module._eligible(dict(values, num_stages=3-stages), rows)
        assert not module._eligible(dict(values, USE_SINKS=True), rows)
        assert not module._eligible(dict(values, output_ptr=values["query_ptr"]), rows)

        # Reach the real Manager's preparation branch only for matching options.
        # Deliberately stop before any IR transformation or device work.
        calls = []
        def stop(raw, block_m):
            calls.append(block_m)
            raise RuntimeError("CPU probe stops before IR transform")
        module.transform_ir = stop
        module._abi = lambda src: {}
        module._resources = lambda compiled: {}
        normal = SimpleNamespace(hash="normal", asm={"ttgir": "fixture"})
        manager = object.__new__(module.Manager)
        manager.async_mode = SimpleNamespace(get=lambda: None)
        manager.candidate = object()
        manager.block_m, manager.fallback_compiles = rows, 0
        manager.original_compile = lambda *args, **kwargs: normal
        manager._record = lambda *args: None
        src = SimpleNamespace(fn=manager.candidate)
        assert manager._compile(src, target=None, options={"num_warps":warps,"num_stages":stages}) is normal
        assert calls == [rows]
        calls.clear()
        assert manager._compile(src, target=None, options={"num_warps":warps,"num_stages":3-stages}) is normal
        assert not calls
        reports.append(dict(rows=rows, warps=warps, tile=tile, stages=stages, adapted_source_sha256=source_sha))
    print(json.dumps(dict(passed=True, gpu_executed=False, configurations=reports,
                         limitation="Descriptor/compiler gate mocks only; no IR correctness, hardware capability, resources or speed claim."), indent=2))


if __name__ == "__main__":
    main()
