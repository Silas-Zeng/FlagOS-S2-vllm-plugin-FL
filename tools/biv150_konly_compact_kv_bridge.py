"""Private compact-cache source/host adapters. Import is stdlib-only; no installer.

The BF16 carrier's K plane overlaps V and is never a BF16 key cache. Activation
requires the complete encoded writer/reader stack and a fresh qualified epoch.
"""
import ast
import copy
from dataclasses import dataclass, fields
import hashlib

BASE = 'dac1b8fb48e7983e48283032efeee9b592c5fbe5'
CONTRACT88 = '65db624d39f554f2330fef578930c4048a109f06'
PARENT89 = '5b228e2c6e6cf0c4365cf0595c340f83b827e470'
FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'
PAGE = 12544
KERNEL_PINS = {
    'writer': '16afc46585f950f4b9c06c831b0e0dbf5515ee941a476a9ed300c492ebf947f9',
    'decode': '149f89b7869b017affce1651b894ae019a8c5c193d8ddf5352e73adac55fbe59',
    'normalized2d128': '205b6a902b906b6a80af8db6246fc9a3fea5e73f6125a16dc809042dc1527740',
    'q32': 'ac0ee4caaeed8dfd211a57cd4d3d130d8437a2a0129c1c915e1b19ed4fc66b4b',
    'multi2': '7046cf538bf06278982041d04b129c65c74bdb65531e528366fc62626c6adb19',
    'raw3d': '190975053927d8f326e47ab0769369b5535524ff4f32e25156f5d16ab717c39c',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class View:
    dtype: str
    shape: tuple
    strides: tuple
    byte_offset: int


def views(pages):
    need(type(pages) is int and 0 < pages <= 2**31 - 1, 'physical page count')
    return dict(
        carrier=View('bfloat16', (pages, 2, 16, 2, 128), (6272, 2176, 256, 128, 1), 0),
        key=View('int8', (pages, 16, 2, 128), (12544, 256, 128, 1), 0),
        scale=View('float32', (pages * 3136,), (1,), 0),
        value=View('bfloat16', (pages, 16, 2, 128), (6272, 256, 128, 1), 4352),
        code_full=View('int8', (pages * 12544,), (1,), 0),
        value_full=View('bfloat16', (pages * 6272,), (1,), 0))


def role_parameters(pages, role, *, format_id):
    need(type(format_id) is str and format_id == FORMAT, 'explicit compact format')
    v = views(pages)
    if role == 'writer83':
        return dict(code_full=v['code_full'], scale_full=v['scale'],
                    value_full=v['value_full'], cache_pages=pages)
    if role == 'decode79':
        return dict(key_cache_ptr=v['key'], key_scale_ptr=v['scale'], value_cache_ptr=v['value'],
                    k_stride_block=12544, k_stride_slot=256, k_stride_head=128,
                    v_stride_block=6272, v_stride_slot=256, v_stride_head=128)
    need(role == 'prefill80', 'unknown compact role')
    result = dict(key_cache_ptr=v['key'], key_code_scale_ptr=v['scale'], value_cache_ptr=v['value'],
                  k_scale_cache_ptr=v['key'], v_scale_cache_ptr=v['value'], KV_QUANT_MODE=0,
                  stride_ks_blk=0, stride_ks_slot=0, stride_ks_head=0,
                  stride_vs_blk=0, stride_vs_slot=0, stride_vs_head=0)
    for prefix, view in (('k', v['key']), ('v', v['value'])):
        result.update({f'stride_{prefix}_cache_{i}': stride for i, stride in enumerate(view.strides)})
    return result


def reshape_compact(raw, spec, shape, order, blocks, packing, *, torch, spec_class):
    """Actual Tensor host operation when explicitly supplied the target runtime.

    Called in place of _reshape_attention_kv_cache only for the exact private
    class. The unchanged allocator must have produced one independent int8 raw
    tensor of exactly blocks*12544 bytes. No allocation occurs in this helper.
    """
    need(type(spec) is spec_class and spec.format_id == FORMAT, 'exact owned compact spec')
    need(spec.page_size_bytes == PAGE and spec.block_size == 16, 'compact page')
    need(packing is None and tuple(order) == (0, 1, 2, 3, 4), 'only independent NHD pages')
    need(tuple(shape) == (blocks, 2, 16, 2, 128), 'backend semantic shape')
    v = views(blocks)['carrier']
    need(raw.dtype == torch.int8 and raw.ndim == 1 and raw.is_contiguous(), 'raw int8 allocation')
    need(raw.numel() == blocks * PAGE and raw.storage_offset() == 0, 'exact allocation extent')
    need(raw.untyped_storage().nbytes() == blocks * PAGE, 'no retained 16KiB backing')
    need(str(raw.device) == 'cuda:0' and raw.data_ptr() % 256 == 0, 'single aligned device allocation')
    return torch.as_strided(raw.view(torch.bfloat16), size=v.shape, stride=v.strides)


SPEC_SOURCE = '''@dataclass(frozen=True, kw_only=True)
class CompactFullAttentionSpec(FullAttentionSpec):
    format_id: str = FORMAT

    def __post_init__(self):
        super().__post_init__()
        if self.format_id != FORMAT:
            raise ValueError("Unknown compact format")
        if (self.block_size, self.num_kv_heads, self.head_size, self.head_size_v, self.dtype) != (16, 2, 128, 128, torch.bfloat16):
            raise ValueError("Only BF16 MiniCPM page16/head128 compact storage")
        if self.kv_quant_mode != KVQuantMode.NONE or self.page_size_padded is not None or self.sliding_window is not None or self.attention_chunk_size is not None or self.non_causal:
            raise ValueError("No stock quantization, padding, hybrid or noncausal compact scope")

    @property
    def real_page_size_bytes(self):
        return 12544

    @classmethod
    def merge(cls, specs):
        if not specs or any(type(spec) is not cls for spec in specs):
            raise ValueError("Compact and stock formats must never merge")
        return super().merge(specs)

    def is_uniform_with_collection(self, specs):
        return bool(specs) and all(type(spec) is type(self) for spec in specs.values())
'''


def spec_module_source():
    """Importable class source for a future private package, not a local class.

    No decorator/registration side effects. All participating processes must
    load the same module path before exchanging this type via scheduler RPC.
    """
    return ('from __future__ import annotations\nfrom dataclasses import dataclass\nimport torch\n'
            'from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode\n'
            'FORMAT = ' + repr(FORMAT) + '\n\n' + SPEC_SOURCE)


def convert_specs(stock, *, full_class, compact_class):
    """Post-hook for ModelRunnerFL.get_kv_cache_spec; validate all before return."""
    need(type(stock) is dict and len(stock) == 42, 'all 42 model layers required')
    need(all(type(spec) is full_class for spec in stock.values()), 'no hybrid/shared/unknown spec')
    result = {name: compact_class(**{f.name: getattr(spec, f.name) for f in fields(full_class)})
              for name, spec in stock.items()}
    need(all(spec.page_size_bytes == PAGE for spec in result.values()), 'complete compact specification')
    return result


def _dump(node):
    return ast.dump(node, include_attributes=False)


def manager_source(source):
    """Add only the private exact type to the two existing zero-notification guards.

    Returns a subclass body, not a registry mutation. Its module must export the
    spec/manager classes before scheduler-worker RPC serialization.
    """
    need(hashlib.sha256(source.encode()).hexdigest() ==
         '22e3677acf0e769461bdbf278df031d6e362c97aa1ba7b862c071b76d47d8914', 'pinned manager source')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SingleTypeKVCacheManager')
    names = ('allocate_external_computed_blocks', 'allocate_new_blocks')
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    need(tuple(n.name for n in methods) == names, 'fixed manager methods')
    old = ast.parse('type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec)', mode='eval').body
    class Add(ast.NodeTransformer):
        count = 0
        def visit_Compare(self, node):
            if _dump(node) == _dump(old):
                node.comparators[0].elts.append(ast.Name(id='CompactFullAttentionSpec', ctx=ast.Load()))
                self.count += 1
            return self.generic_visit(node)
    visitor = Add()
    for method in methods:
        visitor.visit(method)
    need(visitor.count == 2, 'both original zero guards required')
    inverse = copy.deepcopy(methods)
    for method in inverse:
        found = 0
        for node in ast.walk(method):
            if isinstance(node, ast.Compare) and isinstance(node.comparators[0], ast.Tuple):
                seq = node.comparators[0].elts
                if seq and isinstance(seq[-1], ast.Name) and seq[-1].id == 'CompactFullAttentionSpec':
                    seq.pop(); found += 1
        need(found == 1, 'exact manager inverse')
    original = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    need(all(_dump(a) == _dump(b) for a, b in zip(inverse, original)), 'manager algorithm changed')
    child = ast.ClassDef(name='CompactFullAttentionManager', bases=[ast.Name(id='FullAttentionManager', ctx=ast.Load())],
                         keywords=[], body=methods, decorator_list=[])
    return ast.unparse(ast.fix_missing_locations(child)) + '\n'


def c3_source(source):
    """Private _cold copy; predicate must certify the exact owned compact class.

    Not installed here. Original scope, source, device, dispatcher, builder and
    order checks remain; only the exact FullAttentionSpec rejection is replaced.
    """
    need(hashlib.sha256(source.encode()).hexdigest() ==
         'caff59fc7c277c85db78671c89984ab0354fbc9ac4fdb5312ce36a7714863ead', 'pinned parent C3 source')
    old = ast.parse('type(spec) is not spec_module.FullAttentionSpec', mode='eval').body
    new = ast.parse('not compact_spec_is_owned(spec)', mode='eval').body
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_cold')
    class Rewrite(ast.NodeTransformer):
        def __init__(self, before, after): self.before, self.after, self.count = before, after, 0
        def visit(self, node):
            if _dump(node) == _dump(self.before):
                self.count += 1
                return copy.deepcopy(self.after)
            return super().visit(node)
    forward = Rewrite(old, new)
    result = forward.visit(copy.deepcopy(fn)); need(forward.count == 1, 'exact C3 type guard')
    inverse = Rewrite(new, old)
    restored = inverse.visit(copy.deepcopy(result))
    need(inverse.count == 1 and _dump(restored) == _dump(fn), 'C3 non-type behavior changed')
    return ast.unparse(ast.fix_missing_locations(result)) + '\n'


def kernel_source(source, role):
    """Rewrite only named physical-address literals; no codec/dot/mask changes."""
    need(role in KERNEL_PINS and hashlib.sha256(source.encode()).hexdigest() == KERNEL_PINS[role],
         'pinned parent kernel role/source')
    if role == 'writer':
        changes = [('page * 16384', {16384: 12544}, 1),
                   ('page * 4096', {4096: 3136}, 1),
                   ('value_full + page * 8192 + 4096', {8192: 6272, 4096: 2176}, 1)]
    elif role == 'decode':
        changes = [('page * 4096', {4096: 3136}, 2)]
    else:
        need(role in ('normalized2d128', 'q32', 'multi2', 'raw3d'), 'known active parent reader')
        changes = [('pages.to(tl.int64) * 4096', {4096: 3136}, 1)]
    tree = ast.parse(source)
    offsets = []; cursor = 0
    for line in source.splitlines(keepends=True): offsets.append(cursor); cursor += len(line)
    edits = {}
    for expression, literals, expected in changes:
        target = _dump(ast.parse(expression, mode='eval').body)
        matches = [node for node in ast.walk(tree) if _dump(node) == target]
        need(len(matches) == expected, 'address grammar count: ' + expression)
        for node in matches:
            for leaf in ast.walk(node):
                if isinstance(leaf, ast.Constant) and type(leaf.value) is int and leaf.value in literals:
                    start = offsets[leaf.lineno - 1] + leaf.col_offset
                    end = offsets[leaf.end_lineno - 1] + leaf.end_col_offset
                    need(source[start:end] == str(leaf.value), 'decimal address literal')
                    replacement = str(literals[leaf.value])
                    need((start, end) not in edits, 'overlapping source edits')
                    edits[(start, end)] = (source[start:end], replacement)
    output = source
    # Track the shifted output positions so the reverse proves byte identity.
    shift = 0; inverse = []
    for (start, end), (before, after) in sorted(edits.items()):
        inverse.append((start + shift, start + shift + len(after), before))
        shift += len(after) - len(before)
    for (start, end), (_, after) in sorted(edits.items(), reverse=True): output = output[:start] + after + output[end:]
    restored = output
    for start, end, before in reversed(inverse): restored = restored[:start] + before + restored[end:]
    need(restored == source, 'kernel source byte inverse')
    ast.parse(output)
    return output, dict(literal_edits=len(edits), inverse_byte_exact=True,
                        codec_dot_softmax_masks_unchanged=True, actual_IR_verified=False)


def parent_host_source(source, role):
    """Retain PR89's ownership/alias/capture guards, changing only format geometry.

    The resulting broker has PR89's real View/parameters API; the compact runtime
    has its same strong-owner registry. Service manifest/digest binding and the
    private spec lifecycle must still be integrated before either is activated.
    """
    if role == 'views':
        pin = '018c993858803df250ec1fa089778dbed5627e2c1720c3ccd12aad4e40344ede'
        changes = [
            ("'cjj-konly-rn32-rne-v2'", repr(FORMAT), 1),
            ('layer.strides == (8192,4096,256,128,1)', 'layer.strides == (6272,2176,256,128,1)', 1),
            ('(16384,256,128,1)', '(12544,256,128,1)', 1),
            ("value = _view(layer,'bfloat16',(pages,16,2,128),(8192,256,128,1),8192)",
             "value = _view(layer,'bfloat16',(pages,16,2,128),(6272,256,128,1),4352)", 1),
            ('pages*16384', 'pages*12544', 2), ('pages*4096', 'pages*3136', 1),
            ('pages*8192', 'pages*6272', 1), ('k_stride_block=16384', 'k_stride_block=12544', 1),
            ('v_stride_block=8192', 'v_stride_block=6272', 1), ('extra_v=4096', 'extra_v=2176', 1),
            ('page*16384', 'page*12544', 1), ('page*4096', 'page*3136', 1), ('page*8192', 'page*6272', 1)]
    else:
        need(role == 'runtime', 'known parent host role')
        pin = '406d26bd3bd5b6b1ba3278972eb327103af9147760e84c00954df5000c03773c'
        changes = [("'cjj-konly-rn32-rne-v2'", repr(FORMAT), 1),
                   ('seen.pointer + 8192', 'seen.pointer + 4352', 1),
                   ('k.strides == v.strides == (8192, 256, 128, 1)',
                    'k.strides == v.strides == (6272, 256, 128, 1)', 1)]
    need(hashlib.sha256(source.encode()).hexdigest() == pin, 'pinned parent host source')
    output = source
    for old, new, count in changes:
        need(output.count(old) == count, 'parent host address grammar: ' + old)
        output = output.replace(old, new)
    restored = output
    for old, new, count in reversed(changes):
        need(restored.count(new) == count, 'parent host exact inverse')
        restored = restored.replace(new, old)
    need(restored == source, 'ownership/alias/epoch/dispatch guard changed')
    ast.parse(output)
    return output, dict(inverse_byte_exact=True, guard_code_unchanged=True,
                        complete_service_binding=False)
