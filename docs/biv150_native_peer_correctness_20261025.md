# Personal BI-V150 native-path correctness diagnostic

This is evidence and tooling only, related to team PR48; it does not change a production default or attribute a kernel root cause. Team main0915f01 was rechecked. Both fresh services used exactly that clean source, the official serving command, the same immutable SQLite tuning snapshot copied into private caches, and the personal instance's gate library SHA2565cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe. The native flag is the explicit diagnostic variable. The teammate instance was not modified or used for these runs.

## Results

| Fresh service | Count1–128 |4652-input-token long question|HTTP overlap|Cleanup|
|---|---|---|---|---|
| Native on |Pass|Fail:1024 output tokens, IDs8,220 then1022 copies of ID0; finish_reason=length|Yes|Complete|
| Native off |Pass|Pass:64 output tokens, visible answer85|Yes|Complete|
| Native on, long request alone |Not sent|Pass:61 output tokens, visible answer85|No, intentionally serial|Complete|

The first service's exact native-on source/library therefore reproduces the peer failure. The previously observed101/105 official accuracy does not cover this case and cannot erase it. The off arm is a two-request isolation result, not a new105-question accuracy or performance acceptance.

Both requests start from the peer's raw recorded JSON, including temperature0, max_tokens1024 and return_token_ids. The count request streams; the long request is sent after its first content. Payloads are stored unchanged in tools/fixtures/native-peer-mixed-20261025/, with SHA25690e8284043b131ead2842c04cf83b5e3af3aa2f79358bee412769b76f61574c2 and ecf8e319aa48e3c66e8b40019df2dff483ace79dbfdd61de41d436bb3961a42b. Origin: Silas-Zeng/FlagOS-Minicpm-S2, experiment20261025T094755Z-main526-mixed-token-diagnostic, mixed-requests/{decode,prefill}-request.json. Raw evidence and hashes are under docs/evidence/native-peer-correctness-20261025/.

## Limits and next discriminator

HTTP overlap does not prove that every model step is mixed. Disabling native changes computation, weight storage, allocator layout and timing, so the result does not prove that a native CUDA kernel is wrong. The third fresh native-on service running only the identical long request passed. This establishes a concurrency-dependent failure in this fixture but does not identify whether the native kernel, metadata, timing or layout causes it. The fixture has only two live requests; compute_order.make_plan requires at least three, so that reorder plan is excluded for this fixture. Earlier independent M8 BF16 operator instability is also not established as this model failure's cause.

The runners preserve owned-PID/start-tick cleanup, refuse a busy GPU/port, pin the library hash and record source/driver/helper/snapshot identities and installer markers. Fixture/transport CPU checks and syntax pass. No benchmark dimensions, model weights, dataset or serving CLI are modified. These tools must not be used to claim throughput or to accept an unsafe baseline.

Verbose raw HTTP responses are losslessly archived in raw-http-responses.tar.gz; member SHA256 values remain in inventory.json and the archive digest is in raw-http-archive.json.
