# Isolated Model 1 Colab benchmark

**incomplete** — 19/40 epochs.

Cause: epoch 5 verified; awaiting recreated runtime. GPU allocation 5.943 h; calendar 5.928 h. Epoch-5 recovery verified: True.

First epoch: 945.8129258370002 s; subsequent median: 943.5263101844998 s. Uninterrupted training projection: 37743.339023032495 s; actual full-run turnaround: None s.

Final-epoch validation: 0.878880407124682; epochs 31–40 mean: None; decrease against Modal: None percentage points (flag >1). Modal late mean: 90.92%.

Colab disables decoded-frame caching; Modal seed 0 enabled it. Both use sequential decoding, full precision, the same sealed cohort and unchanged 40-epoch cosine schedule. Hardware may differ between allocations.

The Modal value of a full seed is approximately US$4.15. Colab cash spend is zero; cash savings after Modal credits are unknown. Partial work has no full-seed saving assigned.

Setup/recovery timings, hardware/software, checkpoint overhead, memory peaks, interruptions, repeated work and manual-effort completeness are in evidence.json and sessions/. Missing measurements remain unknown. No test or new-park holdout evaluation; no parallel reliability or monthly capacity conclusion.
