# Actual FL GraphWrapper short-service coverage

Frozen63836c7 includes team main1cbac531; actual stock GQA0c66, private NativeK8 original22e, W8A85c and initial tuning v6. No Math or throughput was run in this diagnostic. The observer is disabled for all scored throughput.

32 raw files /3317913B are completely archived. Root independently replayed the CPU check: report SHA256 fe2a61fb46a8a0eaa848e0f16b40a060dcefd6d5673f69af4d4fdda7b4766299. All raw/source/controller/worker/library hashes passed. Both native capture certificates have42 distinct cache epochs and payloads corresponding to actual native capture log entries. N32/S4 is capture-only here;14 returned real-request replays belong to N64/S2 (13×64 valid rows and1×63 valid rows +1 padding). They prove a lower bound of895 real decode rows. All64 engine request IDs uniquely associate with64 HTTP response IDs; the3 serial smoke requests and64 burst requests all completed with HTTP200.

There was no observer_error marker. No Finalizer worker-summary or exit-summary was written, so final error_count/overflow and complete recording integrity are unknown, not zero. Positive linked replay records are useful evidence; they do not prove all replay frequency, N32 real-request coverage, or Math quality.

The API owner initiated shutdown after all67 successful HTTP responses (server.log543); force-kill/EngineSIGTERM follow at547/548, and EngineDeadError first appears at562. The error is confined to owned shutdown, not the request phase. Owner9250/start353247596 is terminal; completed/smoke_passed/cleanup_complete=true, errors=[].

The diagnostic bundle is preserved in diagnostic-bootstrap-v2; it wraps the actual FL GraphWrapper and pins its source. The older a975 Math observer listened to the upstream wrapper and provided no usable request coverage; those original records are retained separately.

Archive SHA256: `f404a487b14f4bd33ac25519d10f99729b132d15fc9520371b8a65d336d2c03a`.
