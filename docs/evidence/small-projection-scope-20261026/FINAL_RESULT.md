# Closed: gate-only projection scope fails precision and warm mixed semantics

Runtime47e1405a50f41d5a558877e5c0f0036c3dde6d88; initial and post-workload-warmup mixed request both fail, despite actual M2 gate-only selection and restoration of9 original generated MM sites. Full105 evaluates99/105=.9429 (105 unique, no failed_predictions). This configuration must not merge.

One scored round after3 short+1full warmup per length:4K5311.820237520383 total tok/s, TTFT3705.551362952974ms;16K4059.258246280803, TTFT123705.37485520617ms. All256/128 requests succeed with zero failures and exact official token counts.4K still adds26 kernel metadata files (17MM/7index/2normalization), so it is not a pure steady-state performance rejection.16K adds0 metadata. These are diagnostic scores, not a new accepted full baseline. The precision/semantic failures independently justify rejecting the tested gate-only configuration. No claim is made that every other scope is impossible.

Full driver/workloads and owned cleanup complete. Preserve historical PR28 code and original accepted100/105 result. PR53 has been reopened to reassess its102/105 passing implementation under better JIT controls. PR58 retains all small-row protections and tests compiler-owned output buffers instead; its first real component already passed36 cases/576 checks, but it is not yet a model gain.

Archive782496bytes SHA256aa8fc7025b8d799978e251a7c7ea29cac3b351ad8e1406b9a15b241512a7059e contains105 predictions/report, scores, warmups, both mixed request results/responses, generated code, server log and exact manifests/state. Remote originals:R/small-projection-gate-only-v1-20261026.
