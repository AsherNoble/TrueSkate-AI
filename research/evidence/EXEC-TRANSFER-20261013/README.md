# Executor-transfer research evidence

This bundle preserves the interrupted Claude research and its laptop-only
continuation. Current conclusions are in the
[experiment record](../../experiments/EXEC-TRANSFER-20261013.md).
Historical drafts under `prior-reports/` are preserved verbatim; their numerical
confidence forecasts and unverified hypotheses are not current findings.

| Evidence | Purpose |
|---|---|
| [provenance.json](provenance.json) | Original sessions, scratch locations, source/export hashes |
| [claim review](REVIEW.md), [bundle verifier](verify_bundle.py) | Independent wording/method review and preserved-output integrity checks |
| [review/review.json](review/review.json), [score.json](review/score.json) | Exact operator judgements and independently reproduced 83/16/1 score |
| [input verification](review/input-verification.json) | All 100 clip and metadata hashes match the frozen manifest |
| [trail report](trail-model/REPORT.md), [methods](trail-model/METHODS.md) | All-clip trail sampling, fade, cap geometry, manual interference audit and clip-67 confound |
| [label report](label-tolerance/REPORT.md), [literature](label-tolerance/continuation-20261013/LITERATURE.md) | Primary sources and 15 CPU fits separating command recovery from fresh replay |
| [label verification](label-tolerance/continuation-20261013/verification.json) | Analytical inverse checks, stochastic oracle and exact non-identifiability examples |
| [XCTest report](xctest-transfer/REPORT.md), [preservation](xctest-transfer/preservation.json) | Measured simulator schedules, source-backed easing, short-segment boundaries and callback-logging controls |
| [compensation feasibility](compensation-followup/REPORT.md), [reference](compensation-followup/reference.py) | Retrospective duration-model holdouts and a mathematical inverse easing reference; no adjusted execution |
| [decoder follow-up](decoder-followup/REPORT.md), [validation](decoder-followup/validation.json) | Minimal coordinate refit, legal free knots, independent synthetic cases, compiled-clock checks and endpoint semantics |
| [simulator/video comparison](sim-video-followup/REPORT.md), [verification](sim-video-followup/verification.json) | Frozen recording split and forward predictions; sparse, sampling-sensitive, inconclusive physical transfer |
| [oracle script](oracle/oracle_decoder.py), [results](oracle/results.json) | Existing decoder on perfect targets plus hypothetical timing shifts |
| [human support script](oracle/human_support.py), [results](oracle/human-support.json) | Structural duration gap, finite-window support, geometric timing sensitivity |
| [legacy tracker output](lag-baseline/summary.json), [cap sensitivity](lag-baseline/cap-sensitivity.json) | Exploratory direction estimates and their sensitivity; not validated timing truth |
| `lag-baseline/vlag-all.json.gz` | Full preserved per-frame legacy tracker outputs |
| `prior-reports/` | Original human-coverage report, executor comparison and transfer draft |
| [Claude collaboration status](CLAUDE-COLLABORATION.md) | External-session proposal was blocked by automatic approval review, then cancelled by the operator |

The corpus movies and metadata were outside Git at:

`/private/tmp/claude-503/-Users-ashernoble-Projects-Robotics---hardware-TrueSkate-AI/e0f1c48e-973a-4ff3-a26a-f5f2350d9a43/scratchpad/audit6-mirror/`.

That temporary scratchpad disappeared during final verification. Its movies,
raw profiles, large trail caches, full disassembly and simulator builds are no
longer available at their original paths. The selected reports, raw simulator
requests, schedules and numeric results were copied into this bundle beforehand.
A copy of the frozen manifest is kept in the parent repository's ignored
`tmp/executor-research-preserved-20261013/`, outside this disposable worktree;
its external location and hashes are recorded
in provenance. Git does not back up that external manifest or the original
movies/caches. [Artifact availability](artifact-availability.json) records the
final checks. Historical source paths and hashes are provenance, not a claim
that their temporary files remain present.

## Reproduction

Use the repository's existing `.venv`. The oracle scripts accept absolute
paths so that the frozen implementation can be selected explicitly:

```sh
"/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/.venv/bin/python" \
  /absolute/path/to/oracle/oracle_decoder.py \
  --source /absolute/path/to/frozen/curved-checkout \
  --mirror /absolute/path/to/audit6-mirror \
  --out /absolute/path/to/temporary/output

"/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/.venv/bin/python" \
  /absolute/path/to/oracle/human_support.py \
  --strokes /absolute/path/to/strokes-v3.json \
  --out /absolute/path/to/temporary/human-support.json
```

The label study's `METHODS.md` gives its exact CPU command, seeds and metric.
Trail methods document the original external profile-cache location; the
large derived `.npz` caches were retained there rather than committed and are
now unavailable. The retained scripts support regeneration after recovering
the source footage. The legacy aggregate
script is an exact historical copy with its output redirected; it retains its
original scratchpad layout requirements.

The decoder follow-up report gives exact reproduction commands and source
hashes. The simulator/video comparison separates `study.py extract` from
`study.py fit`; preserve the frozen plan and complete visual review between
stages. Its `verify.py` independently checks hashes, command equality, recording
separation and aggregation. Large XCTest raw request logs/schedules and all
original CSV/diagnostic patches are deterministic gzip files in this bundle;
the latter preserve original CRLF and patch whitespace. Decompress to an isolated
temporary directory before using the original scripts, which retain their
external scratch layout. Full framework disassembly, WDA checkout and simulator
build products were external; original paths/hashes are in `preservation.json`,
and current availability is reported separately.

Human support results describe trail-derived moving spans. Their duration and
isolated-window calculations are conditional on the original extraction's event
partition, and do not independently measure complete touch-contact durations.

The original full-source simulator/video verification passed while media and
profiles were present (`sim-video-followup/verification.json`). After their
disappearance, `verify.py --preserved-only` independently reproduces command
equality, recording separation and primary arithmetic from the saved manifest
and simulator schedules. It writes `verification-preserved.json`, explicitly
marking the unavailable source/media checks incomplete. Extraction, visual
review and raw-profile regeneration require recovering the external corpus.
