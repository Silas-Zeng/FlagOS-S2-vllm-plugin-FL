"""Verify Q32 compatibility pins and exactly reconstruct the audited base sources."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    order=root/'vllm_fl/patches/biv150_compute_order.py'
    constants={}
    for node in ast.parse(order.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('EXPECTED_RUNNER','SOURCES'):
                constants[node.targets[0].id]=ast.literal_eval(node.value)
    def digest(raw):return hashlib.sha256(raw).hexdigest()
    checked=[]
    for name,expected in constants['SOURCES'].items():
        if not name.startswith('vllm_fl.'):continue
        file=root/(name.replace('.','/')+'.py');assert digest(file.read_bytes())==expected,name
        checked.append(name)
    runner=(root/'vllm_fl/worker/model_runner.py').read_text()
    assert digest(runner.encode())==constants['EXPECTED_RUNNER']
    multi2='''    from vllm_fl.patches.biv150_prefill_multi2 import install as _install_prefill_multi2

    print("BIV150_PREFILL_MULTI2_INSTALL", _install_prefill_multi2(), flush=True)

'''
    assert runner.count(multi2)==1
    runner=runner.replace(multi2,'')
    assert digest(runner.encode())=='268804cb09d84baa8497d020adaf04329ead8bf29951609d8bf602bf92b04613'
    extra='''    from vllm_fl.patches.biv150_prefill_q32 import install as _install_prefill_q32

    print("BIV150_PREFILL_Q32_INSTALL", _install_prefill_q32(), flush=True)

'''
    assert runner.count(extra)==1
    assert digest(runner.replace(extra,'').encode())=='cbbbabcea66663b033f2209febc55e75a9e14ac72d1693173d9d4aad1be7fac6'
    source=(root/'vllm_fl/patches/biv150_q_fragment_service.py').read_text()
    assert digest(source.encode())=='38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82'
    # Reverse the already merged w16 generalization and stride audit as well,
    # retaining the original pre-Q32 exact-inverse proof below.
    shipping_inverse={
        'def _eligible(values, block_m=128, *, num_warps=8, num_stages=1, tile_size=64):':'def _eligible(values, block_m=128):',
        '"TILE_SIZE": tile_size,':'"TILE_SIZE": 64,',
        '"num_warps": num_warps,':'"num_warps": 8,',
        '"num_stages": num_stages,':'"num_stages": 1,',
        '_eligible(kwargs, m.block_m, num_warps=m.num_warps,\n                                  num_stages=m.num_stages, tile_size=m.tile_size)':'_eligible(kwargs, m.block_m)',
        'raw, candidate,\n                 block_m=128, *, num_warps=8, num_stages=1, tile_size=64):':'raw, candidate, block_m=128):',
        '        self.num_warps, self.num_stages, self.tile_size = num_warps, num_stages, tile_size\n':'',
        '(options.get("num_warps"), options.get("num_stages"))\n                    != (self.num_warps, self.num_stages)':'(options.get("num_warps"), options.get("num_stages")) != (8, 1)',
        '957720a31e63fd994e096e1bf5cfce9197541506cb06d66cb71ba5614bd323a1':'24a421fa25e0ce7e21b54deb760a5bc893d3899a86ced4849bd3f3bfab50ed42',
        'fn.__code__.co_name != "kernel_biv150_prefill_stride"':'fn.__code__.co_name != "kernel_biv150_prefill_normalized"',
    }
    for old,new in shipping_inverse.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    inverse={
        'def transform_ir(raw, block_m=128):':'def transform_ir(raw):',
        'def _eligible(values, block_m=128):':'def _eligible(values):',
        'def lowered_proof(ttgir, llir, q_source_lines, block_m=128):':'def lowered_proof(ttgir, llir, q_source_lines):',
        'rf"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<{block_m}x128xbf16"':'r"(%[A-Za-z0-9_]+) = ttg.local_alloc .*tensor<128x128xbf16"',
        '"BLOCK_M": block_m,':'"BLOCK_M": 128,',
        '"BLOCK_Q": block_m // 8,':'"BLOCK_Q": 16,',
        '_eligible(kwargs, m.block_m)':'_eligible(kwargs)',
        'raw, candidate, block_m=128):':'raw, candidate):',
        '        self.block_m = block_m\n':'',
        'transform_ir(normal.asm["ttgir"].encode(), self.block_m)':'transform_ir(normal.asm["ttgir"].encode())',
        '                self.block_m,\n':'',
    }
    for old,new in inverse.items():
        expected_count=2 if old.startswith('rf"') else 1
        assert source.count(old)==expected_count,old
        source=source.replace(old,new)
    assert digest(source.encode())=='3f90de854adeb7d3584b19bad1f317cb527f16f14bec15d40f55ffa3e13b4d8a'
    semantics=(root/'vllm_fl/compile_semantics.py').read_text()
    added='''    features["VLLM_FL_BIV150_PREFILL_MULTI2"] = (
        features["VLLM_FL_BIV150_PREFILL_Q32"]
        and env.get("VLLM_FL_BIV150_PREFILL_MULTI2", "1") == "1")
'''
    assert semantics.count(added)==1
    assert digest(semantics.replace(added,'').encode())=='e497e92132056d65977f7ea32b5709692c58fd4bacaaf54b5a008a6a3e5dd2b0'
    print(json.dumps(dict(passed=True,local_pins=checked,runner_inverse_exact=True,qfragment_inverse_exact=True,
                         multi2_installer_inverse_exact=True,compile_semantics_inverse_exact=True,
                         limits='CPU source identity only; live closure/cold scope and service numerical/performance validation still required.'),indent=2))


if __name__=='__main__':main()
