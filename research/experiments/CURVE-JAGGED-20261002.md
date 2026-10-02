# CURVE-JAGGED-20261002 — Jagged organic visual probe

After reviewing the curved recordings, the operator explicitly requested a jagged but organic gesture with sharper bends. One isolated XR1 recording on operator-confirmed Inbound executed two versions, 600 ms and 300 ms, plus start/middle/end timing controls. No collection jobs, training, or CURVE-EXEC main/confirmation stages were started.

Six anchors make a zigzag with four direction changes: (0.27,0.79), (0.48,0.55), (0.35,0.45), (0.67,0.39), (0.53,0.27), (0.91,0.33). Five cubic sections join with matching position and time derivatives. Short tangents leave tight, smooth turns. This is explicitly a piecewise visual probe; a single nine-number cubic cannot express this shape, and the model/schema contracts remain unchanged.

Each section's coefficient hull is within the original central rectangle and clear of expanded controls. Every quantized segment is checked. Segment counts are independently chosen from 4/8/16 per section to keep the combined interpolation/quantization bound within 0.005. The 600 ms command has 48 movements, minimum spacing 7 ms and maximum section bound 0.004531. The 300 ms command has 56 movements, minimum spacing 3 ms and maximum section bound 0.004740. Dense spacing is diagnostic, not certified execution fidelity.

Shared endpoints are emitted once, followed by all moves in one pointer-down/up W3C request per gesture. Synthetic verification checked exact duration sums, scaled positions, positive explicit durations, exactly one down/up, frozen reconstruction and derivative continuity at every join.

The [manifest](../evidence/CURVE-JAGGED-20261002/manifest.json) was frozen before device execution. Both gestures and all three controls completed successfully, with no recording or schedule error. Native video has 1,771 source frames. A lossless fast-start remux preserves every original PTS exactly. Human viewing is primary for this request; no orange extractor was run. Viewer jump points use approximate WDA epoch minus recorder startedAt. The interactive viewer provides 600/300 ms buttons, speed controls, native frame stepping and optional looping: http://127.0.0.1:8767/jagged/?v=1.

Admission outcome and isolated artifact provenance are recorded in the [summary](../evidence/CURVE-JAGGED-20261002/summary.json). No claim of observed timing accuracy or formal executor fidelity follows from command success or operator visual approval. Curve recovery and generation by the models remain the next tranche.

Post-recording admission failed with a gameplay-contamination flag. The attempt is preserved without replacement and is available for human viewing; no fidelity pass is claimed. The existing heuristic flag does not by itself identify the gameplay state. Appium session cleanup logged an expired session after offline verification; healthy WDA was not restarted.
