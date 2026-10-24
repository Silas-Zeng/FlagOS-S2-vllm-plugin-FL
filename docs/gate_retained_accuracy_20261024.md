# Partial acceptance update

Frozen runtime source33d3d85b7458e2d14bb783adbeb792b404b6cce7 on the personal
BI-V150 instance completed Math500 Level3:103/105 (reported0.981),105 unique
questions, zero failed predictions. Source and model parameters were unchanged
throughout the run. This is one stochastic full evaluation, not a repeatability
claim or proof of improvement over101/105.

At16:05 the owned service/engine remained live and began4K warmup. Both formal
throughputs and TTFT acceptance remain pending. Startup capacity500432 KV tokens
is below the compressed residual parent's547632; the throughput impact has not
been measured. No merge is authorized by accuracy alone.

The running source remains frozen33d3d85; this subsequent evidence-only commit
is not a new model implementation or an update to the remote checkout.

At16:15 the first formal4K result is5401.515648831977 total tok/s (+12.31257%
vs4809.36), TTFT4054.8019629131886ms,256 complete,0 failed, input1048576,
output262144. Three warmup TPOT values143.93611/108.49393/108.51283ms; the last
two are stable. The parent's5404.54270 differs by about−0.056%, a separate-time
single-round observation consistent with preserving4K performance, not a
controlled repeatability claim.16K is warming and its formal result is pending.
