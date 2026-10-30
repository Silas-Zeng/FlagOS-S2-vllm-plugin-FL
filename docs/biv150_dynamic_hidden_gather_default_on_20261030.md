# PR122 default-enable candidate

Parent: `5b109cb146aa3968ce9b718e6110dc4419db0224`, with main899 retained.
This is a local candidate, not a deployment or merge approval. The parent's
4K/16K measurements used explicit `VLLM_FL_BIV150_DYNAMIC_HIDDEN_GATHER=1`.
Official model quality and a real worker with this variable absent remain
separate; no qualification is inferred from the default change.

Only three production literals change: the defaults in `prepare`, `try_gather`,
and `compile_semantics.build_fingerprint`, from `'0'` to `'1'`. Exact inverse
checks recover all five parent production files and then the original a101
component bytes. The existing runner CRLF, compute-order pin, kernel source,
tiling/ABI, new-output allocation, one-use receipt, Graph-NONE/non-spec scope,
caller/event/storage/stream checks, fallback, and postlaunch error handling are
unchanged. Earlier default-OFF descriptions describe the parent; this document
records the proposed new policy.

For supported inputs, absence of the variable now makes the same choice as the
tested parent's explicit `1`. Explicit `0` still returns to the original gather;
other strings do not silently enable it. The C3 feature's default agrees with
both runtime gates. Its source digest changes honestly because two Python files
change; no old C3 digest or service receipt is retained as the new identity.

The controller still clears gather to explicit `0` for OFF/ABI probes and sets
explicit `1` for the candidate service. Removing the variable is no longer an
OFF operation. Its only source change checks the explicit zero in `off_env`;
the existing ten inverses still recover the full637d controller. CLI, requests,
Math105, benchmark, warmup, scoring, libraries, tuning and cleanup are unchanged.
The new bridge records the exact5b default-policy parent. Both Root admission
flags and model/service flags remain false when generating a receipt.

The existing service CPU checker covers the exact production/controller
inverses, actual prepare branches up to a labelled fake pre-cold sentinel,
explicit-zero early fallback, default/explicit-one C3 equality, distinct OFF C3,
and the original postlaunch observation/address/IR/full-ELF rejection cases.
It does not import CUDA/Torch or qualify a new worker. Use the same saved SME1
component files as5b and the unchanged actual-run preflight stopped at first
`idle`; CPU-only temporary approval flags are not Root approval.

## Actual artifact boundary

The original `actual-2018-ordinary/actual.source` is15961 bytes, SHA256
`5e5d73eec44d15caa62bf0294ba906511e9e537b5c369c4a508910cbaa1838fe`.
It is location-bearing `module { tt.func public @gather_hidden_bits ... }` IR,
not the Python helper module: it contains no `prepare`, `try_gather`, helper
environment defaults or `compile_semantics.py`. The three new literals do not
occur in it. This observation does not grant a future compiled object identity:
all original complete source/TTIR/TTGIR/LLVM/ASM/ELF checks, including exact
debug-directory relocation only, remain unchanged and mandatory. No binary,
path, ABI or resource mismatch is forgiven by this candidate.

## Remaining default-worker smoke

The current controller always ends with explicit `1`; running it unchanged does
not exercise a worker whose environment lacks the key. A later, separately
reviewed plain-smoke-only transport can remove this one key after service/observer
environment construction and immediately before worker launch. The computed C3
remains identical to explicit ON, but the raw absent key must be recorded.
The current snapshot requires `os.environ.get(module.ENV)=='1'`, so removal
alone would make observation reject. That future tool-only seam must recognize
the exact newly bound default, report raw absence plus effective ON, still reject
explicit0, and retain actual postlaunch counts/bindings and every compiled-byte
gate. No such smoke or observer change is included here. No formal CLI or extra
Math run is added.
