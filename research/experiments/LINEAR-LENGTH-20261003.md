# LINEAR-LENGTH-20261003 — blinded duration × length experiment

## Operator intent

After the fine sweep, the operator reports trace visibility in 13/14 clips and
board movement in all 14, including the trace-free 20 ms R2. They requested
anonymized human review across five lengths, three repetitions and nine exact
durations: 50/45/40/35/30/25/20/15/10 ms. Total: 135 diagnostics.

## Frozen workload

[Manifest](../evidence/LINEAR-LENGTH-20261003/manifest.json), SHA256
`1ca788d7b15229c40d9ac3a978961b1e150d8b2494566d5f5c16da794201dec0`. XR1/Inbound; common start (0.27, 0.78) and direction toward
(0.91, 0.30). Endpoints use 20/40/60/80/100% of that diagonal's length.
Each gesture has placement before down, one linear movement and up; no hold,
intermediate waypoints or y offset. Full paths clear expanded protected controls.

Each repetition randomizes all 45 cells. Fifteen one-minute segments contain
nine diagnostics each, with resets three seconds before slots 5/10/15/20/25/
35/40/45/50 s. Start/middle/end controls at 1/30/57 s; stop target 59 s.
Twenty-one WDA requests per recording; timing revision, recorder-idle, tunnel,
foreground, settling and overrun checks remain unchanged. No collection jobs,
WDA restart, automatic replacement or training admission. Human gameplay review
is authoritative; no automated menu/editor image guard or scan is used.

## Blinding and assessment

Review order is shuffled independently of execution. Public clips have opaque
identifiers, numbered frames and equal 2.2 s extraction windows (0.7 s before,
1.5 s after calibrated onset). The public payload contains no condition labels,
repetition, recording number, command IDs, absolute source PTS or full recordings.
The private condition/source mapping remains outside the web root. Observable
path length in the footage itself is inherent; its numeric label is hidden.

The blinded auditor uses decoded native frames, slow playback and ←/→ stepping.
At operator direction, there are exactly two labels: `trace_visible` with values
`flicker`, `hold`, `trace`; and boolean `board_moved`, shown as a toggle defaulting to True for unreviewed clips, preserving saved False. A comments
box is separate. M saves and advances; B toggles board movement. There are no
automatic classifications or extra uncertainty/onset labels. Browser labels are
saved locally and exported against opaque tokens. Assessment and any transition
analysis remain blinded until human labels exist.

## Completed workload

All 135 diagnostics completed without replacement, in fifteen recordings with
315 successful WDA requests and no execution error. Every length/duration cell
has exactly three completed repetitions. All fifteen source decodes and timing
checks pass; total native frame count is 26,502 and the largest absolute held-out
middle residual is 25.2 ms. Recorder is idle; the bounded process has exited.

[Summary and source hashes](../evidence/LINEAR-LENGTH-20261003/summary.json).
Raw files are in `tmp/linear-length-recordings/` locally and
`/Users/training-server/trueskate-ai-runtime/tmp/LINEAR-LENGTH-20261003/recordings/`
on the rig. Private source joins remain beside the local raw files, outside the
web root. The anonymous viewer contains 135 clips and 8,864 native frames:
[Open blind review](http://127.0.0.1:8767/blind-length/). Safari displays 135
numbered clips, the comments field and the Board moved switch initially on.
Offline checks cover balance, frozen-payload tampering, label leaks, native
frame stepping, default True and preservation of saved False (26 tests pass).
Human assessments are pending; no automatic visibility or board labels exist.

### Review UI update

The operator requested easier controls while preserving existing assessments.
Trace defaults to `trace` only for unreviewed clips; three adjacent, exclusive
buttons replace the dropdown. Saved labels, boolean board values and comments
retain the same schema and browser storage key. Clip IDs, order, bundle and frame
assets are unchanged. A compact review panel keeps Save & next prominent, hides
the full grid behind an expandable section, and resumes at the first unreviewed
clip. Synthetic restoration checks cover existing Hold/False/comments without
rewriting storage; Safari confirms Trace selected and Board moved on by default.

## Completed blinded operator assessment

All 135 exported labels match the frozen bundle and private key; no missing,
unknown or duplicate labels. The export is preserved byte-for-byte at
[operator-assessments.json](../evidence/LINEAR-LENGTH-20261003/operator-review/operator-assessments.json).
[Full tables and comments](../evidence/LINEAR-LENGTH-20261003/operator-review/results.md),
[unblinded CSV](../evidence/LINEAR-LENGTH-20261003/operator-review/unblinded-assessments.csv),
and [machine-readable results](../evidence/LINEAR-LENGTH-20261003/operator-review/results.json).

| Duration | Trace | Flicker | Board moved |
|---:|---:|---:|---:|
| 50/45/40/35/30 ms | 75/75 | 0/75 | 75/75 |
| 25 ms | 12/15 | 3/15 | 15/15 |
| 20 ms | 7/15 | 8/15 | 15/15 |
| 15 ms | 7/15 | 8/15 | 15/15 |
| 10 ms | 0/15 | 15/15 | 0/15 |

No Hold labels were used. At all five lengths and all three repeats, 10 ms has
no board movement and every ≥15 ms clip has movement: the observed response
transition lies between 10 and 15 ms. Thirty ms is the lowest tested duration
with trace in every length/repeat. Visible trace is mixed at 15–25 ms; 19 flicker
clips still move the board. Rendering and gameplay response are distinct.

Intermediate-duration length patterns are not consistently monotonic: at 15 ms
the shortest length has 0/3 traces and the three longest lengths each 2/3; at
20 ms the longest has 0/3 while most shorter lengths have 2/3. Three repeats per
cell do not establish a universal threshold or rule out subtler length effects.
Input path collapse is not established by these 30 fps viewing labels.

Both comments about remarkably short traces unblind to the shortest (20%)
gesture at 45 ms, consistent with the operator's suggested length explanation.
These conclusions use the saved human labels; the UI defaults were Trace and
Board moved True after its update. No automatic classifier was used.
