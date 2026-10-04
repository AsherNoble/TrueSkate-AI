# LINEAR-LENGTH-20261003 — Blinded linear duration × length audit

The operator reviewed 135 XR1/Inbound clips: nine durations from 10 through
50 ms in 5 ms increments, five lengths (20/40/60/80/100% of a common diagonal)
and three repetitions per condition. Each command used one linear movement
between pointer-down and pointer-up. Execution and review orders were shuffled
independently. Fifteen one-minute recordings used resets before samples,
start/end timing calibration and an independent middle control. All timing
checks passed; the largest absolute middle residual was 25.2 ms.

The supplied export matches the preserved operator assessments byte-for-byte
(SHA-256 `21541d9ed957505903c9e8d2f5c9424835679362177bf93e74f5bfd4fc3fcd36`).
All 135 IDs match the private key and blinded bundle, with three labels per cell.

| Duration (ms) | Trace | Flicker | Board moved |
|---:|---:|---:|---:|
| 30–50, in 5 ms steps | 75/75 | 0/75 | 75/75 |
| 25 | 12/15 | 3/15 | 15/15 |
| 20 | 7/15 | 8/15 | 15/15 |
| 15 | 7/15 | 8/15 | 15/15 |
| 10 | 0/15 | 15/15 | 0/15 |

The observed board-response transition lies between 10 and 15 ms in this
session. Nineteen Flicker-labelled gestures still moved the board, so trace
visibility is not a reliable proxy for board response. These 30 fps video
assessments do not establish input-path collapse. Length effects were not
consistently monotonic, and three repeats per condition do not establish a
universal threshold. Both comments about unusually short traces corresponded
to the shortest length at 45 ms.

Ratings were explicitly saved by the operator. The viewer preselected Trace
and Board moved after its UI update. The historical export and result schema
are preserved unchanged; the operator subsequently clarified that discrete
stationary-looking marks were not reliably distinguishable from flicker, so
no additional visual category is interpreted here.

## Evidence and provenance

- [Frozen manifest](../evidence/LINEAR-LENGTH-20261003/manifest.json)
- [Private condition mapping](../evidence/LINEAR-LENGTH-20261003/private-key.json)
- [Recording summary and hashes](../evidence/LINEAR-LENGTH-20261003/summary.json)
- [Original operator export](../evidence/LINEAR-LENGTH-20261003/operator-review/operator-assessments.json)
- [Full tables and comments](../evidence/LINEAR-LENGTH-20261003/operator-review/results.md)
- [Machine-readable results](../evidence/LINEAR-LENGTH-20261003/operator-review/results.json)
- [Unblinded assessments](../evidence/LINEAR-LENGTH-20261003/operator-review/unblinded-assessments.csv)

These evidence files are copied unchanged from preserved source commit
`f56d1964ddd4d37a615672477d5207cb53a9ad17` in `tmp/linear-speed-worktree`.
Raw recordings remain in laptop `tmp/linear-length-recordings/` and on the rig
at `/Users/training-server/trueskate-ai-runtime/tmp/LINEAR-LENGTH-20261003/recordings/`.
Git does not back up those ignored recordings. This journal update does not
modify or resume any collection workload.
