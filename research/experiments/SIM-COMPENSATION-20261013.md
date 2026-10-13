# SIM-COMPENSATION-20261013 — prospective legal executor compensation

Status: completed, frozen overall gate failed. User authorized a bounded laptop
simulator experiment after the offline EXEC-TRANSFER feasibility check. No rig,
physical device, collector, paid cloud, model training or production change.

The prototype legally subdivides longer legs, retimes using the prior approximate
ceiling-clock model and fits added coordinates along original segments while
fixing semantic corners/holds/endpoints. It retains the 100 ms interval minimum,
300–1200 ms total, ten-segment maximum and unchanged compilation/control safety.
Five of twenty cases have explicitly unchanged constrained fallbacks.

Before requests, the exact candidate/payloads, four development and sixteen
prospective held-out paths, order and full-effect criteria were frozen.
160 measurements plus 2 setup smokes completed once, with no retries or retuning.
Held-out macro normalized trajectory RMS 0.019443→0.016609 (14.58% reduction,
below the 25% requirement); full XY/waypoint/hold/release passes 0→2/16, below 16/16.
Only the 463 and 739 ms straight paths pass. Better lift timing can coexist with worse
intermediate XY; no general compensator or physical transfer is established.

Frozen scorer limitations are retained and audited separately: missing stationary
callback boundaries, an intersection-vs-midpoint hold selector discrepancy, and
interpolation over sparse gaps. Primary/coalesced union has zero effect. A
post-freeze gap-aware sensitivity gives 13.82% RMS reduction and the same 2/16 pass
count. Missing sentinels are failures, not measured 1000 ms delays.

The dedicated simulator was restored to its initial Shutdown state; owned
WDA/logger/build processes stopped and localhost port 8199 was free. All source and
artifacts are isolated in the research worktree; original dirty WDA is untouched.

[Full report, methods and measured table](../evidence/SIM-COMPENSATION-20261013/REPORT.md),
[frozen plan](../evidence/SIM-COMPENSATION-20261013/PLAN-FROZEN.md),
[exact cases/payloads](../evidence/SIM-COMPENSATION-20261013/FROZEN.json),
[results](../evidence/SIM-COMPENSATION-20261013/results.json),
[verification/sensitivities](../evidence/SIM-COMPENSATION-20261013/verification.json),
[cleanup](../evidence/SIM-COMPENSATION-20261013/cleanup.json).

Development reference:
[EXEC-TRANSFER-20261013](EXEC-TRANSFER-20261013.md), especially its
[offline compensation check](../evidence/EXEC-TRANSFER-20261013/compensation-followup/REPORT.md).
No STATUS/JOURNAL or prior frozen evidence was edited by this branch.
