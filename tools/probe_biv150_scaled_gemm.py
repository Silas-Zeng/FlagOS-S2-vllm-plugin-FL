"""Capability probe for fused row/column-scaled GEMM, run without a live service."""
import ctypes as C
import json
import socket

import torch
import torch.nn.functional as F

from bench_biv150_fastpaths import graph_time


class Layout(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Desc(C.Structure):
    _fields_ = [("data", C.c_uint64 * 32)]


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t),
                ("state", C.c_int), ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]


def checked(code, operation):
    if code:
        raise RuntimeError(f"{operation}: ixblas status {code}")


class ScaledLt:
    def __init__(self, aq, bq, sa, sb, out, compute, column):
        self.lib = lib = C.CDLL("/usr/local/corex/lib64/libcublasLt.so")
        signatures = {
            "ixblasLtCreate": [C.c_void_p],
            "ixblasLtDestroy": [C.c_void_p],
            "ixblasLtMatrixLayoutInit_internal": [C.c_void_p, C.c_size_t, C.c_int, C.c_uint64, C.c_uint64, C.c_int64],
            "ixblasLtMatrixLayoutSetAttribute": [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t],
            "ixblasLtMatmulDescInit_internal": [C.c_void_p, C.c_size_t, C.c_int, C.c_int],
            "ixblasLtMatmulDescSetAttribute": [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t],
            "ixblasLtMatmulPreferenceInit_internal": [C.c_void_p, C.c_size_t],
            "ixblasLtMatmulPreferenceSetAttribute": [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t],
            "ixblasLtMatmulAlgoGetHeuristic": [C.c_void_p] * 7 + [C.c_int, C.c_void_p, C.c_void_p],
            "ixblasLtMatmul": [C.c_void_p] * 14 + [C.c_size_t, C.c_void_p],
        }
        for name, args in signatures.items():
            getattr(lib, name).argtypes = args
            getattr(lib, name).restype = C.c_int
        self.handle = C.c_void_p()
        checked(lib.ixblasLtCreate(C.byref(self.handle)), "create")
        self.desc, self.a, self.b, self.d, self.pref = Desc(), Layout(), Layout(), Layout(), Layout()
        checked(lib.ixblasLtMatmulDescInit_internal(C.byref(self.desc), C.sizeof(self.desc), compute, 0), "desc init")
        m, k = aq.shape
        n = bq.shape[1]
        self.aq, self.bq, self.sa, self.sb, self.out = aq, bq, sa, sb, out
        if column:
            specs = [(self.a, 3, n, k, n), (self.b, 3, k, m, k), (self.d, 14, n, m, n)]
            self.pa, self.pb, scale_a, scale_b = bq.data_ptr(), aq.data_ptr(), sb.data_ptr(), sa.data_ptr()
        else:
            specs = [(self.a, 3, m, k, k), (self.b, 3, k, n, n), (self.d, 14, m, n, n)]
            self.pa, self.pb, scale_a, scale_b = aq.data_ptr(), bq.data_ptr(), sa.data_ptr(), sb.data_ptr()
        for layout, dtype, rows, cols, ld in specs:
            checked(lib.ixblasLtMatrixLayoutInit_internal(C.byref(layout), C.sizeof(layout), dtype, rows, cols, ld), "layout init")
            order = C.c_int(0 if column else 1)
            checked(lib.ixblasLtMatrixLayoutSetAttribute(C.byref(layout), 1, C.byref(order), C.sizeof(order)), "layout order")
        for attr, value in [(17, C.c_void_p(scale_a)), (18, C.c_void_p(scale_b)),
                            (31, C.c_int(3)), (32, C.c_int(3))]:
            checked(lib.ixblasLtMatmulDescSetAttribute(C.byref(self.desc), attr, C.byref(value), C.sizeof(value)), f"desc attr {attr}")
        checked(lib.ixblasLtMatmulPreferenceInit_internal(C.byref(self.pref), C.sizeof(self.pref)), "pref init")
        size = C.c_size_t(32 << 20)
        checked(lib.ixblasLtMatmulPreferenceSetAttribute(C.byref(self.pref), 1, C.byref(size), C.sizeof(size)), "workspace preference")
        choices, count = (Heuristic * 8)(), C.c_int()
        checked(lib.ixblasLtMatmulAlgoGetHeuristic(self.handle, C.byref(self.desc), C.byref(self.a), C.byref(self.b),
                C.byref(self.d), C.byref(self.d), C.byref(self.pref), 8, choices, C.byref(count)), "heuristic")
        good = [choices[i] for i in range(count.value) if choices[i].state == 0]
        if not good:
            raise RuntimeError(f"No valid heuristic (returned {count.value})")
        self.algo = good[0].algo
        self.workspace = torch.empty(max(1, good[0].workspaceSize), dtype=torch.uint8, device="cuda")
        self.alpha, self.beta = C.c_float(1), C.c_float(0)

    def __call__(self):
        checked(self.lib.ixblasLtMatmul(self.handle, C.byref(self.desc), C.byref(self.alpha),
                self.pa, C.byref(self.a), self.pb, C.byref(self.b), C.byref(self.beta),
                self.out.data_ptr(), C.byref(self.d), self.out.data_ptr(), C.byref(self.d),
                C.byref(self.algo), self.workspace.data_ptr(), self.workspace.numel(),
                torch.cuda.current_stream().cuda_stream), "matmul")
        return self.out

    def close(self):
        if self.handle:
            self.lib.ixblasLtDestroy(self.handle)
            self.handle = None


def main():
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", 9031)) != 0, "Do not probe alongside a service"
    torch.manual_seed(20261023)
    for m, n, k in [(32, 2560, 2048), (64, 12288, 2048), (2048, 12288, 2048)]:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        sa = x.float().abs().amax(1).clamp_min(1.e-6) / 127
        sb = w.float().abs().amax(1).clamp_min(1.e-6) / 127
        aq = torch.round(x.float()/sa[:, None]).clamp(-127,127).to(torch.int8)
        bq = torch.round(w.float()/sb[:, None]).clamp(-127,127).to(torch.int8).t().contiguous()
        ref = (torch._int_mm(aq, bq).float() * (sa[:, None]*sb[None, :])).to(torch.bfloat16)
        bf16_ms = graph_time(lambda: F.linear(x, w), repeats=9, unroll=4)[0]
        for compute in [10, 0]:
            for column in [False, True]:
                op = None
                try:
                    out = torch.empty_like(ref)
                    op = ScaledLt(aq, bq, sa, sb, out, compute, column)
                    got = op()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(got, ref, atol=0.01, rtol=0.01)
                    ms = graph_time(op, repeats=9, unroll=4)[0]
                    print(json.dumps(dict(m=m,n=n,k=k,arm="Lt-int8",compute=compute,column=column,
                        gemm_epilogue_ms=ms,bf16_ms=bf16_ms,max_abs_diff=(got.float()-ref.float()).abs().max().item())),flush=True)
                except Exception as e:
                    print(json.dumps(dict(m=m,n=n,k=k,arm="Lt-int8",compute=compute,column=column,error=str(e)[:800])),flush=True)
                finally:
                    if op:
                        op.close()
        for dtype, maximum, name in [(torch.int8,127,"int8"), (torch.float8_e4m3fn,448,"fp8")]:
            try:
                a_scale = (x.float().abs().amax(1,keepdim=True).clamp_min(1.e-6)/maximum).contiguous()
                b_scale = (w.float().abs().amax(1).clamp_min(1.e-6)/maximum)[None, :].contiguous()
                a = (x.float()/a_scale).clamp(-maximum,maximum).to(dtype)
                b = (w.float()/b_scale.t()).clamp(-maximum,maximum).to(dtype).t()
                fn=lambda: torch._scaled_mm(a,b,scale_a=a_scale,scale_b=b_scale,out_dtype=torch.bfloat16)
                got=fn()
                torch.cuda.synchronize()
                reference=((a.float()*a_scale)@(b.float()*b_scale)).to(torch.bfloat16)
                torch.testing.assert_close(got,reference,atol=0.02,rtol=0.02)
                ms=graph_time(fn,repeats=9,unroll=4)[0]
                print(json.dumps(dict(m=m,n=n,k=k,arm="aten-scaled-"+name,gemm_epilogue_ms=ms,bf16_ms=bf16_ms)),flush=True)
            except Exception as e:
                print(json.dumps(dict(m=m,n=n,k=k,arm="aten-scaled-"+name,error=str(e)[:800])),flush=True)
        # Also compare vendor int_mm's accepted B layouts before assuming one packing.
        for b,name in [(bq,"row"),(bq.t().contiguous().t(),"column")]:
            try:
                got=torch._int_mm(aq,b)
                torch.testing.assert_close(got,torch._int_mm(aq,bq),atol=0,rtol=0)
                ms=graph_time(lambda: torch._int_mm(aq,b),repeats=9,unroll=4)[0]
                print(json.dumps(dict(m=m,n=n,k=k,arm="int-mm-"+name,gemm_ms=ms)),flush=True)
            except Exception as e:
                print(json.dumps(dict(m=m,n=n,k=k,arm="int-mm-"+name,error=str(e)[:800])),flush=True)


if __name__ == "__main__":
    main()
