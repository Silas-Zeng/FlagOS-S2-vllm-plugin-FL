# PR118: retain the real v1 build failure; fix only the identifier probe

Parent:2dd472a547c0dd4e9befc867debef873a736f2ee, a normal fast-forward from
ff107, including e6c5/18cff and the independently reviewed two-reducer evidence
fix. The frozen ff107/e6c5/18cff worktrees and all CUDA/codec/host math, actual
kernel flags, SO binding and numeric gates remain unchanged. No new SO or device
qualification exists. Root owns publication and actual compile scheduling.

The original run123212/start360369848 is terminalZ. Its complete17files/53379B
remain unmodified at `/private/tmp/cjj-native118-mixed-compile-v1-raw-20261029`.
The replay in this evidence folder verifies every SHA, completed=false and the
exact error `RuntimeError('Symbol preprocessor failed: device')`. There is no SO.
The compiler was actual CoreX clang-22, SHA
8428e871501b05213d5b7ec0ecf796f4d04de24e5bf75c2539dc812f52ab3101.
The host -E probe succeeded; the device command included -E/-P and
--cuda-device-only but returned1/empty stdout, with an unknown-type diagnostic
for CJJ_MIXED_SYMBOL_PROBE_BEGIN and `compiling for ivcore11`. This proves that
this invocation performed semantic processing of the bare tokens; it does not
establish a general rule about every CoreX -E mode. We do not guess a cc1/API fix.
The stored Torch state is null/null, since this source failed before its Torch
import; it is not relabeled as an observed CUDA-initialized=false result.

The minimal new route preserves the host -E check. Its device source is a valid
empty C++ TU: two-level token concatenation and seven #if/#error conditions
require the precise seven macro mappings. Unique IDs also reject a permutation.
It uses the already successful CoreX --cuda-device-only -S -emit-llvm mode from
the actual8b build. A positive run must exit0, produce a nonempty LLVM module
with target triple bi-iluvatar-ilurt, and then a control removes only the first
rename macro. That control must exit exactly1 and report the dedicated
CJJ_MIXED_DEVICE_RENAME_ASSERT_1 error. Driver failures, wrong targets and
unexpected successful controls are refused before Torch import. Its nonzero
command is explicitly recorded as expected_exit_code=1, with the exact marker;
it is a deliberate negative control, not a successful compiler invocation.
All stdout/stderr/LLVM/source/commands remain in the manifest. The original v1
failure is not converted into this negative control or into a successful build.

CPU checks reproduce the bare-token syntax failure, accept all seven correct
mappings in the valid TU, and separately reject each missing/wrong mapping
(14controls), wrong target, unexpected success and unrelated failure. These use
only the existing local host C++ compiler, and do not claim CoreX target success.
No Torch/Triton/NumPy, JIT, CUDA runtime or GPU is used. Actual CoreX verification,
SO/math/resources/numerics/service/performance remain pending.

```sh
python3 -B -W error tools/check_biv150_native_k8_mixed_build_cpu.py
python3 -B tools/compile_biv150_native_k8_mixed.py
```

Root-only after its current device job has finished:

```sh
python3 tools/compile_biv150_native_k8_mixed.py \
  --compile-only --root-owned-exclusive \
  --expected-source-commit FULL_NEW_FROZEN_HEAD \
  --out /absolute/fresh-compile-out
```

No extra GPU probe or numeric queue is introduced. Later actual SO IR/SASS,
7symbol final ELF/dladdr, compiler/options/source hashes and independently known
resource checks are all still compulsory. This fix grants none of those results.
