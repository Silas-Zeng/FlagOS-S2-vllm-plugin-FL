"""CPU source factory only; no Torch/Triton import, JIT, runner or serving hook."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
MAIN = '56f1838192abc940b3fb9accef490022ac125000'
V8_COMMIT = '4a1f7ddd38e4bcade30e702faba2e9a7cdb5441d'
V8_SHA = 'a7fc539dccbd2e2910538665abf42d0dd6204939d39f1ebeb7a6cac1f992c3ca'
MAIN_GQA_SHA = '2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
FIXTURE = ROOT/'tools/fixtures/biv150_v8_bf16p_4a.py.txt'
BIAS_EXPRESSION = '(((V.to(tl.uint8, bitcast=True).to(tl.uint32) ^ 128) | 0x4B000000).to(tl.float32, bitcast=True) - 8388736.0)'
INSERT = ('        # Exact signed-byte FP32 bias reconstruction; no V scale here.\n'
          '        V = '+BIAS_EXPRESSION+'\n').encode()
PV = 'tl.dot((probability * value_scale[None, :]).to(tl.bfloat16), V.to(tl.bfloat16))'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def producer(raw):
    return next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)
                and n.name == 'gqa_decode_v8_bf16p')


def is_assign(node, name):
    return isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(
        node.targets[0],ast.Name) and node.targets[0].id == name


def page_paths(function):
    return (next(n.body for n in function.body if isinstance(n,ast.While)),
            next(n.body for n in function.body if isinstance(n,ast.If) and
                 ast.unparse(n.test)=='full_upper < upper'))


def validate_emitted(raw, reference=None):
    original = FIXTURE.read_bytes() if reference is None else reference
    assert sha(original) == V8_SHA, 'Only frozen4a/a7fc is supported'
    assert raw.count(INSERT) == 2 and raw.replace(INSERT,b'') == original, 'Non-V byte inverse failed'
    expected = ast.parse('V = '+BIAS_EXPRESSION).body[0]
    tree = ast.parse(raw)
    removed = 0
    for body in page_paths(producer(raw)):
        index = next(i for i,n in enumerate(body) if ast.dump(n)==ast.dump(expected))
        assert index > 0 and is_assign(body[index-1],'running_sum')
        assert is_assign(body[index+1],'accumulator')
        del body[index]
        removed += 1
    # Independently remove exactly the two typed decode assignments from the
    # complete module AST, including untouched encoder/writer helpers.
    class RemoveDecode(ast.NodeTransformer):
        count = 0
        def visit_Assign(self,node):
            if ast.dump(node)==ast.dump(expected):
                self.count += 1
                return None
            return self.generic_visit(node)
    restore = RemoveDecode()
    restored = restore.visit(tree)
    assert removed == restore.count == 2
    assert ast.dump(restored)==ast.dump(ast.parse(original)), 'Non-V module AST changed'
    wanted = ast.parse(PV,mode='eval').body
    dots = [n for n in ast.walk(producer(raw)) if isinstance(n,ast.Call) and ast.dump(n)==ast.dump(wanted)]
    assert len(dots)==2, 'Original BF16(P*scale) / V.to(BF16) PV must remain exact'
    assert ast.dump(producer(raw).args)==ast.dump(producer(original).args), 'ABI changed'
    compile(raw,'generated_v8_bias.py','exec')  # syntax only, never execute imports
    return dict(exact_byte_inverse=True,exact_module_ast_inverse=True,
        unchanged_abi=True,unchanged_pv_sites=2,decode_sites=2,
        original_source_sha256=sha(original),generated_source_sha256=sha(raw),
        full_consumer_connected=False,actual_ir_available=False,target_support_validated=False,
        precision_gate=False,official_score=False,physical_capacity_reclaimed=False)


def make_vdecode(reference=None, *, variant='bias', table=None):
    if variant != 'bias':
        raise NotImplementedError('Only A/bias exists; B/LUT needs a separate immutable-table ABI contract')
    if table is not None:
        raise ValueError('A has no table parameter/reader ABI; all table arguments are rejected')
    original = FIXTURE.read_bytes() if reference is None else reference
    assert sha(original)==V8_SHA
    function = producer(original)
    lines = original.splitlines(keepends=True)
    sites = []
    for body in page_paths(function):
        node = next(n for n in body if is_assign(n,'accumulator'))
        sites.append(node.lineno-1)
    assert len(sites)==2
    for index in sorted(sites,reverse=True):
        lines.insert(index,INSERT)
    generated = b''.join(lines)
    return generated,validate_emitted(generated,original)


def fp32_bias_byte(raw_byte):
    """Stdlib CPU model of the emitted unsigned-bitcast/bias operation."""
    assert type(raw_byte) is int and 0 <= raw_byte <= 255
    bits = 0x4B000000 | (raw_byte ^ 128)
    wide = struct.unpack('<f',struct.pack('<I',bits))[0]
    return struct.unpack('<f',struct.pack('<f',wide-8388736.0))[0]


def code_address(page, slot, head, dimension):
    """CPU address contract only; does not add device bounds checks."""
    assert all(type(value) is int for value in (page,slot,head,dimension))
    assert 0 <= page < 2**31 and 0 <= slot < 16 and 0 <= head < 2 and 0 <= dimension < 128
    offset = page*16384+8192+slot*256+head*128+dimension
    assert 0 <= offset < 2**63
    return offset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit',type=Path,help='Write generated source once; no imports/JIT/device execution')
    args = parser.parse_args()
    source,proof = make_vdecode()
    if args.emit is not None:
        with args.emit.open('xb') as output:
            output.write(source)
    print(json.dumps(dict(main=MAIN,v8_source_commit=V8_COMMIT,factory='A/bias',
        table_factory_implemented=False,proof=proof,emitted=str(args.emit) if args.emit else None),indent=2))
