# Native paged-prefill capability result

Frozen probe0b62441 on the installed ixformer, without package upgrades. Direct HND views of the actual packed NHD-stride layout work; no gather is performed. Same shuffled KV and output buffer, bottom-right causal reference, poisoned nondefault-stream Graph, and changing input tests.

4K chunk: baseline1.795723ms/native0.815317ms,2.202484x.16K chunk:8.340790/3.469190ms,2.404247x. Both cases pass their component numerical and Graph gates. Eager component intervals include dispatch/internal costs, and no service benefit is claimed.

Ragged Q[1,17,33],KV[65,130,257]: uniform pattern passes. Random pattern has2 native-vs-baseline tolerance violations and1 native-vs-CPU-reference violation across104448 elements, max_abs0.0078125; baseline-vs-CPU passes. Coordinates were not retained, so the cause is not classified from aggregate error. The probe stops at this failure; fresh-prefill case did not run.

The competition explicitly prohibits switching existing operators without operator optimization. This direct-library substitution is therefore retained as a performance reference, not a competition optimization. It does establish that prefill has substantial headroom and motivates original kernel/compiler work in PR46. This capability direction is closed with raw results preserved; no production hook or merge.
