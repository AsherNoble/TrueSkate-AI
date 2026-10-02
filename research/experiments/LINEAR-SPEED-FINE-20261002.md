# LINEAR-SPEED-FINE-20261002 — 50–20 ms linear drag sweep

The operator requested a finer repeat after reporting a visible trail and board
movement at 50 ms, no visible trail but board movement at 20 ms, and a minuscule
tap/flicker without board movement at 10 ms in the earlier sweep.

Frozen workload: 50, 45, 40, 35, 30, 25, 20 ms on XR1/Inbound, then reversed.
Fourteen drags in two one-minute recordings, each with seventeen requests
(seven labelled resets, seven drags, start/middle/end controls). Preparation
begins three seconds before slots 6/12/18/24/36/42/48 s. Stop target is 59 s.
The full protected-control-safe path remains (0.27, 0.78) → (0.91, 0.30), with
placement before down, one linear movement and up. Only duration varies.

[Manifest](../evidence/LINEAR-SPEED-FINE-20261002/manifest.json), SHA256
`b5f1ecfbb5dd420896fe768df2dd892162521bbead03a2b34622a38cd05c58ea`. The original broad manifest is byte-compatible.
Foreground, gameplay, settling, 100 ms lateness, recorder-idle, root tunnel,
instrumented WDA, native decode and per-recording calibration checks remain.
No collector or service restart is authorized; failed attempts are preserved.
Viewing is the primary assessment. Gameplay response is separate from visible
feedback; commands below about 33 ms are shorter than one 30 fps native frame.

## Operator review policy

The operator explicitly requested that menu/editor detection use human vision
for this tiny experiment. Automated gameplay image guards and scans are skipped
for the reverse recording. The first recording's rig admission scan was stopped
after recording completion; its partial diagnostic output is retained.
Timing, foreground, recorder-idle, settling and native frame-count checks remain.
Neither recording is admitted for training; visual assessment belongs to the
operator. No automated gameplay gate is used to reject or replace these clips.

## Completed pair

All fourteen requested drags completed without replacement. Each recording has
seventeen successful WDA requests and no execution error. Full native decode
counts are 1,769 / 1,766. Start/end calibration spans 55.99 / 56.13 s; held-out
middle residuals are 11.8 / 6.4 ms, within the existing two-frame threshold.

[Summary and hashes](../evidence/LINEAR-SPEED-FINE-20261002/summary.json).
Raw files are preserved locally at `tmp/linear-fine-recordings/` and on the rig
at `/Users/training-server/trueskate-ai-runtime/tmp/LINEAR-SPEED-FINE-20261002/`.
The stopped first rig scan is recorded separately; it did not invalidate any
completed drag. The first local diagnostic was already run before the policy
change; it remains historical output. No image scan was run for repeat 2.

[Native-frame viewer](http://127.0.0.1:8767/linear-fine/) contains paired
duration buttons, slow playback, exact source-frame stepping and separate
feedback/gameplay fields. Human review and the transition range are pending
operator assessment; no automatic trail or gameplay conclusion is claimed.
Collection stays off and the recorder is idle.
