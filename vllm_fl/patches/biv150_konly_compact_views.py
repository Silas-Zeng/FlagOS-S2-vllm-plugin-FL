"""Stateless integer-address recipes, never Tensor creation or encoded activation."""
from dataclasses import dataclass

FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'
WIDTH = {'bfloat16': 2, 'int8': 1, 'float32': 4}
ROLES = ('decode79', 'prefill80', 'writer83')


def need(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class View:
    """Explicit metadata observation/recipe. Pointer is independently checked."""
    dtype: str
    shape: tuple
    strides: tuple
    storage_id: int
    storage_base: int
    storage_bytes: int
    storage_offset: int  # Elements of this view's own dtype, not bytes.
    pointer: int
    device: str

    def validate(self):
        need(type(self.dtype) is str and self.dtype in WIDTH, 'typed dtype')
        need(type(self.shape) is tuple and type(self.strides) is tuple and len(self.shape) == len(self.strides), 'rank')
        need(all(type(n) is int and 0 <= n < 2**63 for n in self.shape), 'shape integers')
        need(all(type(n) is int and 0 < n < 2**63 for n in self.strides), 'positive stride integers')
        need(type(self.storage_id) is int and self.storage_id > 0, 'known storage identity')
        need(all(type(n) is int and 0 <= n < 2**63 for n in
                 (self.storage_base,self.storage_bytes,self.storage_offset,self.pointer)), 'storage observations')
        size = WIDTH[self.dtype]
        need(self.storage_base % size == 0 and self.pointer % size == 0, 'typed alignment')
        need(self.pointer == self.storage_base+self.storage_offset*size, 'pointer/offset identity')
        extent = 0 if 0 in self.shape else 1+sum((n-1)*s for n,s in zip(self.shape,self.strides))
        need((self.storage_offset+extent)*size <= self.storage_bytes, 'typed storage boundary')
        need(self.storage_base+self.storage_bytes < 2**63, 'signed-I64 storage range')
        need(type(self.device) is str and self.device == 'cuda:0', 'explicit single component device')


def layer_pages(layer):
    need(type(layer) is View, 'explicit layer observation')
    layer.validate()
    need(layer.dtype == 'bfloat16' and len(layer.shape) == 5
         and layer.shape[1:] == (2,16,2,128), 'NHD layer shape/dtype')
    need(layer.strides == (6272,2176,256,128,1), 'actual NHD layer strides')
    pages = layer.shape[0]
    need(0 < pages <= 2**31-1, 'physical page count')
    need(layer.pointer % 4 == 0 and (layer.storage_offset*2) % 4 == 0, 'FP32 full-layer base alignment')
    need(layer.storage_offset*2+pages*12544 <= layer.storage_bytes, 'complete layer byte span')
    return pages


def _view(layer, dtype, shape, strides, relative_bytes):
    byte_offset = layer.storage_offset*2+relative_bytes
    width = WIDTH[dtype]
    need(byte_offset % width == 0, 'view reinterpretation offset')
    view = View(dtype,shape,strides,layer.storage_id,layer.storage_base,layer.storage_bytes,
                byte_offset//width,layer.pointer+relative_bytes,layer.device)
    view.validate()
    return view


def reject_cache_alias(layer, readonly_sources):
    """K/V may share read-only QKV backing; writable-cache backing is excluded."""
    need(type(readonly_sources) is tuple, 'explicit source observations')
    for source in readonly_sources:
        need(type(source) is View, 'known readonly source')
        source.validate()
        need(source.dtype == 'bfloat16' and len(source.shape) == 3 and source.shape[1:] == (2,128), 'BF16 K/V source')
        need(source.strides[1:] == (128,1) and source.strides[0] >= 256, 'K/V source strides')
        need(source.device == layer.device, 'source device')
        overlap = max(source.storage_base,layer.storage_base) < min(
            source.storage_base+source.storage_bytes,layer.storage_base+layer.storage_bytes)
        need(source.storage_id != layer.storage_id and not overlap, 'readonly source aliases live cache backing')


def parameters(layer, role, *, format_id, readonly_sources=()):
    """Return only frozen cache-role kwargs, not a callable or an activation proof.

    format_id is an explicit requested interpretation. It does not register or
    prove the contents of layer storage. No default format/role/fallback exists.
    Other query/output/slot/scalar arguments remain the future caller's job.
    """
    need(type(role) is str and role in ROLES, 'unknown frozen kernel role')
    need(type(format_id) is str and format_id == FORMAT, 'explicit private format required')
    pages = layer_pages(layer)
    reject_cache_alias(layer,readonly_sources)
    key = _view(layer,'int8',(pages,16,2,128),(12544,256,128,1),0)
    scale = _view(layer,'float32',(pages*3136,),(1,),0)
    value = _view(layer,'bfloat16',(pages,16,2,128),(6272,256,128,1),4352)
    if role == 'decode79':
        return dict(key_cache_ptr=key,key_scale_ptr=scale,value_cache_ptr=value,
                    k_stride_block=12544,k_stride_slot=256,k_stride_head=128,
                    v_stride_block=6272,v_stride_slot=256,v_stride_head=128)
    if role == 'prefill80':
        result = dict(key_cache_ptr=key,key_code_scale_ptr=scale,value_cache_ptr=value,
                      k_scale_cache_ptr=key,v_scale_cache_ptr=value,KV_QUANT_MODE=0,
                      stride_ks_blk=0,stride_ks_slot=0,stride_ks_head=0,
                      stride_vs_blk=0,stride_vs_slot=0,stride_vs_head=0)
        for prefix,view in (('k',key),('v',value)):
            result.update({f'stride_{prefix}_cache_{i}':stride for i,stride in enumerate(view.strides)})
        return result
    return dict(code_full=_view(layer,'int8',(pages*12544,),(1,),0),
                scale_full=scale,value_full=_view(layer,'bfloat16',(pages*6272,),(1,),0),
                cache_pages=pages)


def validate_binding(layer, role, supplied, *, format_id, readonly_sources=()):
    expected = parameters(layer,role,format_id=format_id,readonly_sources=readonly_sources)
    need(type(supplied) is dict and set(supplied) == set(expected), 'complete exact cache-role parameter set')
    for name,value in expected.items():
        actual = supplied[name]
        need(type(actual) is type(value), 'cache role type '+name)
        if type(value) is View:
            actual.validate()
        need(actual == value, 'cache role pointer/stride/offset '+name)
    return True


def element_addresses(layer, role, page, slot, head, dim):
    """CPU-only byte-address projection; all frozen formulas use local page IDs."""
    pages=layer_pages(layer)
    need(all(type(x) is int for x in (page,slot,head,dim)), 'integer address indices')
    need(0 <= page < pages and 0 <= slot < 16 and 0 <= head < 2 and 0 <= dim < 128, 'address bounds')
    p=parameters(layer,role,format_id=FORMAT)
    if role == 'writer83':
        k,s,v=p['code_full'],p['scale_full'],p['value_full'];extra_v=2176
    else:
        k,v=p['key_cache_ptr'],p['value_cache_ptr'];s=p['key_scale_ptr' if role == 'decode79' else 'key_code_scale_ptr'];extra_v=0
    return (k.pointer+page*12544+slot*256+head*128+dim,
            s.pointer+4*(page*3136+1024+slot*2+head),
            v.pointer+2*(page*6272+extra_v+slot*256+head*128+dim))
