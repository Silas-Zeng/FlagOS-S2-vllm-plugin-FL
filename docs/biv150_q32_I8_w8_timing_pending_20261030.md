# Conditional w16/S1 versus w8/S1 reader ABBA (not admitted)

Independent child of `618acfef826acfcf2eaff6293d3fd3252681e94d`. Production, GPU source, codec, thresholds and the frozen numeric frontend stay unchanged. No timing/device job was executed or queued, and this child has not been published.

This copies the narrow accepted PR124 timing adapter (`887217375d113d62829e19b5794fce6f70458a88`, original source SHA `8a79ed27f9471b6c6293ab76f4c9e2566adb3fd762a19b66ca90db048f877828`) onto the current numeric framework. The timing body comes from the same pinned522c source. Its complete byte inverse differs only in role and ratio labels: `w16,w8,w8,w16` and `W16_over_W8`. No new reference, writer or numerical framework is introduced.

Control is the accepted hoisted w16/S1 cubin `7eadbdbf854be149083065cb30f0f8c1bdc593aea6d2b9baf454502adb6fbb43`; candidate is the reviewed fixed-preheader w8/S1 cubin `03d373323bef024b3dec81d76de9c19913d9a03d0c05f471e01178523110d1a8`. Both original source/debug/override/cache paths remain unchanged, with cache-miss compilation prohibited. The prepared proof remains `9a7832afadd9dd0ace1104ccc28773e9273b854cc95662f884b58735cbe53db6`; static review remains `5409b37ec09961b1459fbab689fd7407ee610341250eeb48857b3761e8a8c18d`.

Every run first repeats all8 numerical/Graph/side stages from618, including all4194304-element w8-versus-w16 hard comparisons, original quantized FP64 gates, BF16 diagnostic reporting, guarded output, complete cache and readonly checks. Changed-input Graph B still precedes fresh eager B. Cross-layout bitwise stays diagnostic. This does not substitute w16 control for the original FP64 oracle. Precision/model/service qualification stays false even if explicit numeric-diagnostic is used.

Timing is limited to final random pattern1, N1/Q2048/total KV6143/prefix4095, the same773 physical16KiB pages and original descriptors. KV6144 is numerical-only. For each of reader and append-reader there is one warmed ABBA:8 warm Graph replays per arm/mode, then16 Graph replays per event block, with4 complete steps per replay. Four blocks per mode give8 event blocks total. Event and host-wall values are divided by64; host wall includes replay submissions and end-event synchronization, excludes validation/readback. Append-reader includes the unchanged writer plus complete reader device work. Neither metric is a service or complete Python-helper cost. Warm fixture cache state cannot be extrapolated to service HBM behavior.

Each timed block retains the original full same-arm output-bit comparison against that run's validated B output, whole-cache equality, canaries and readonly check. Thus the earlier full4M cross-arm hard gate remains applicable after each exact same-arm replay. The original timing artifact retains post-timing hashes/checks, not independent per-block output arrays; the complete pre-timing stage arrays remain archived.

Actual future numeric review/manifest/summary/receipt SHA fields and raw file/byte counts are explicitly `None`. `qualification_ready()` rejects them before explicit CLI preflight/GPU work and before timing admission. There is no `None == None` path to approval. A later reviewed commit must pin the actual618 numeric originals and Root's independent array review; failed numeric data cannot qualify. The future review must verify all8 stages, Graph-before-fresh, side stream, all per-stage `full_output_reference` hard checks, both actual executable hashes/cubins and unchanged qualification labels. BF16 proxy pass/fail is recorded from actual data rather than predicted here. A separate exact Root timing receipt and fresh serial-slot admission are still required afterward. No approved receipt or runnable launcher accompanies this commit.

Default CPU reproduction (stdlib only):

```bash
python3 -B -S tools/probe_biv150_konly_q32_w8_timing.py
python3 -B -S tools/check_biv150_konly_q32_w8_timing_cpu.py
```

The18 checks cover the full original timing-body inverse, per64 accounting, all existing numerical/full-output/Graph/side guards, forwarding both actual geometry contracts, and actual CLI refusal while numeric evidence is pending. `--cpu-preflight` and `--gpu` both stop before framework/device work in this freeze. No extra Git helper dependency is introduced beyond the618 numeric runtime roots: current child, efd,522c,e4dd,844a. PR124's negative S2 result and all earlier raw archives remain unchanged; no performance claim follows from this conditional source.
