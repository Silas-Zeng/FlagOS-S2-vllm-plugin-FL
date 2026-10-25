# Query tile + verified Q-fragment lifetime experiment

Team main0915f01. First run01fe3bb uses same packed/shuffled KV, causal bottom-right CPU reference, both uniform and changing random inputs. Both baseline and candidate prove Q load outside the loop in TTGIR and LLVM; poisoned changed-input Graph replay matches eager. This is a real kernel/layout experiment, not a vendor implementation switch.

BM64/Q8 with tile64,warps8,stages1 is negative:4K median baseline/candidate1.7899/3.4249ms (0.52260x),16K8.3848/16.4812ms (0.50875x). Both loaded kernels retain114 registers and32768 shared bytes; reducing Q rows did not reduce these resources. Output correctness passes. No actual service test is justified for this configuration.

Two bounded follow-ups preserve Q-hoisting: BM128/warps4 to test execution distribution, BM256/warps8 to amortize paged-address work. These have not yet run. No production change or end-to-end gain is claimed.
