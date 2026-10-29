Title: Draft: isolate NativeK8 packed K lookahead after V completion (CPU only)

The accepted Native8b86 producer reads each page's K only after the preceding page finishes. This private variant reads the next full page's eight packed uint32 words per thread after the current V wait8/ready barrier, then publishes them to the original K shared image after the current PV and before the original page-end barrier. It preserves Q/K codec, QK, softmax, BF16 P/PV, tail/empty behavior, host ABI and 16KiB allocation. The independent namespace does not install a service hook.

Based on a455 / the exact a1eb source that generated SO8b86. This is separate from PR114 Kpair, closed PR113 shared6K, Q64, and the earlier negative Triton staged loop. Source inverses and stdlib address/stage ownership checks pass; actual CoreX counter behavior, final SO scheduling/resources, numeric, performance, model and service are all pending. Added global loads must remain after the original V completion boundary in final machine code; otherwise this version stops before numerical execution. Source ordering and a few numerical cases cannot replace that proof.

Validation: `python3 -B -S tools/check_biv150_native_k8_klookahead_cpu.py`; default `compile_biv150_native_k8_klookahead.py` emits a CPU plan. Only explicit Root opt-in can compile/register the isolated library and export the final SO's IR/SASS; no attention, codec or reducer is launched. All manual review template fields remain false. No new numeric or timing framework is included.

This file is a prepared Draft body only. No PR/upload occurred in this offline CPU turn. No performance claim or promotion request.
