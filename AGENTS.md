# Repository instructions

Canonical instructions for every agent. Claude Code and Codex read this file
directly; `.github/copilot-instructions.md` forwards here. Do not add a
`CLAUDE.md`: Claude Code loads `AGENTS.md` only when a project has none.

## Current direction

Behavioural cloning is authoritative. Read `research/SYSTEM_MAP.md` first, then
the short `research/JOURNAL.md`. Near-term target: a rough curved Model 1 MVP
and the first end-to-end kickflip in Workshop. Linear Model 1 is the current
trained model; hold, heatmap and temporal variants remain runnable experiments.
Model 2 is unfinished. PPO/CMA-ES implementations are retired in tagged history.

## How research runs here

Explore is the default. Certify only when the user says a result is going into
the arXiv preprint or another public claim.

- **Explore:** fast, rough, disposable. No blinding, frozen protocols, digests
  or hash manifests, research attempt ceilings, no-replacement rules, amendment
  records or red-team reviews. Automated data checks and rig safety caps stay.
  Record what ran, the result and the decision in a few journal lines.
- **Certify:** the existing rigour (frozen protocol, blinded human judgement,
  untouched test data, pass rule stated in advance, red-team review, retained
  evidence). If you think something needs it, ask. Never escalate on your own.
- **North star:** the intended trick lands when a predicted gesture or policy
  runs on the phone. Model 1 recovery % is a proxy. Every proposal names the
  link in the SYSTEM_MAP chain it unblocks and how that moves toward the north
  star. Near-term target: a rough curved Model 1 MVP, even a bad one, because
  the end-to-end loop is blocked on it.
- **Thin slice:** one trick (kickflip), one park (Workshop), end to end, before
  widening. Don't polish a component past what the next link needs.
- **Precision from outcomes:** before chasing timing or position accuracy, find
  how much error the outcome tolerates (e.g. add 0/50/100/200 ms jitter or
  position noise to a known-good kickflip and see where it stops landing).
  Labels are accurate enough when their error is smaller than the smallest
  gesture change that alters the outcome. That threshold is also the robotic
  hand's spec. "Within one frame" is not a default target.
- **Proposals:** anything bigger than a quick check (offline, under about an
  hour, no rig time or spend) needs a short proposal the user approves or edits
  first: the question; the link it unblocks; a time/cost box; "if X then A,
  else B".
- **Finish with a recommendation.** Never stop at "inconclusive, needs more
  data"; say what to do next.
- **The user architects; agents execute.** Update `research/SYSTEM_MAP.md` in
  the same commit as any finding that changes the latency chain, a model's
  inputs or outputs, the blocker or the open decision. Explain plainly, without
  ML-lab jargon. A decision the user can't follow from that page isn't ready.

Kept in both modes: select checkpoints on validation and never re-score the
test split; honest park/device provenance; the rig invariants below; check
JOURNAL, GESTURES and the archive before re-testing an execution idea; tests for
the rig control path, data contracts and training loaders (one-off analysis
scripts need none).

## Writing it down

- `research/SYSTEM_MAP.md`: the one page to read first; at most ~150 lines.
- `research/JOURNAL.md`: one entry per result (**Ran**, **Result**,
  **Decision**), at most six lines. Keep 30 entries; drop the oldest freely
  (main's history keeps them). Preserve existing experiment IDs.
- Experiment records only when a journal entry can't hold the result. Open with
  a five-line summary (question, result, decision, next step, link). Aim for
  150 lines or fewer; no separate `REPORT.md`.
- Git holds code, tests, records, decisions, short summaries and the review
  exports a decision rests on. No committed file over 500 KB or JSON over 2,000
  lines. Images only as small JPGs, and only when a human judgement rests on them.
- Bulk output, and anything a record cites, lives in the evidence folder:
  `/Users/ashernoble/TrueSkate-evidence/<ID>/` on the laptop,
  `/Users/training-server/trueskate-evidence/<ID>/` on the rig. Each folder's
  `INDEX.md` maps old paths to new ones.
- `tmp/` is scratch that may be cleared at any time. Remove your own `tmp/`
  output when your branch merges.
- The repo is public. Never commit UDIDs, IP or email addresses, Apple Team IDs
  or personal device names. Use `.env` key names and the `training-server`
  hostname.

## Navigation and conventions

- `README.md`: architecture; `docs/WORKFLOWS.md`: runnable commands.
- `GESTURES.md`: authoritative gesture, bounds, timing and schema contracts.
- `DEPLOYMENT.md`: rig operation, compatibility launchers and rollback.
- `.venv` in the main checkout is the only virtual environment. Run
  `python -m pytest` from your worktree's root with that `.venv`'s Python;
  scripts also need `PYTHONPATH=src` to import the worktree's code.
- Temporary output goes in `tmp/` (scratch; see "Writing it down");
  corpus/checkpoint artifacts stay out of Git.
- Use absolute filesystem paths in tools. Do not rely on tilde expansion.
- Commit messages: `type: message`, one coherent commit at a time.
- Prefer terse factual updates: what changed, evidence, next step.
- Preserve dirty work, including another session's. Treat the rig checkout as
  deployment state, not an untracked development line.
- Once a feature branch is merged into the default branch, delete it: local
  branch, remote branch and worktree. This applies here and to the WDA fork.
  Preserve/archive refs (`preserve/*`, `archive/*` tags) are exempt.

## Sessions and worktrees

- The main checkout belongs to the user. Agents only read it: no edits,
  commits, checkouts or pulls there.
- One worktree per session, at `.claude/worktrees/<name>` on its own branch
  from `origin/main`. Claude Code sessions start with `claude -w <name>`, which
  creates the worktree and blocks edits to the main checkout. A session that
  starts in the main checkout moves to a new worktree before changing anything
  (Claude Code: `EnterWorktree`; Codex: `git worktree add
  .claude/worktrees/<name> -b <branch> origin/main`). Existing worktrees
  elsewhere, such as `tmp/worktrees/`, stay where they are.
- `.githooks/` refuses commits and merge commits on `main`. Enable it once per
  clone with `git config core.hooksPath .githooks`. Never bypass it with
  `--no-verify`.
- **Handoffs.** A session can be cut off by usage limits without warning; the
  user then continues the task in the other tool (Claude Code ↔ Codex), which
  reads the previous session's transcript.
  - Commit small WIP commits on your branch after each working step, so git
    always holds the current state. They are squashed at merge.
  - A session continuing another's work uses that session's worktree
    (`claude -w <name>` reopens `.claude/worktrees/<name>`; start Codex in
    that directory). Begin with `git status` and `git log` there, and trust git
    over the transcript or any handoff note where they disagree.

## Commit hygiene

- Each PR lands on `main` as one commit. Split a PR only when it holds
  independent changes. WIP and other working commits stay on the branch and
  never reach `main`.
- Tag a branch only when a research record or evidence folder cites one of
  its commits. Then, before rebasing or squashing it, push an annotated tag
  at the branch tip, `archive/squash-YYYYMMDD/<branch>`, so the cited SHAs
  stay reachable. Squashing is not a retirement, so it needs no
  `research/ARCHIVE.md` entry.
- Squashed message: `type: message`, a short context paragraph, then one
  bullet per change. WIP commits never appear in it or in the PR
  description. If you tagged the branch, add the tag and its full tip SHA.
- Squash locally, force-push the one-commit branch, then use GitHub "Squash
  and merge"; the `main` ruleset requires a PR and allows no bypass. Merge
  stacked PRs in order, rebasing each next branch onto the new `main` first.

## Rig invariants

- Collection is OFF by operator intent. Do not infer authorization to resume
  an open-ended run from maintenance, device power-on, or stale heartbeat data.
  Use explicitly bounded isolated smoke tests when needed. Collector launchd
  jobs and collection watchdogs are disabled; the autonomous fixer is retired.
- XRs normally attach to `training-server`; use
  `tailscale ssh training-server@training-server`. Do not assume local USB.
- UDIDs come from `.env`. XR1 WDA/Appium: 8100/4723; XR2: 8103/4726.
- Keep XCTest segments at one minute: stop returns the full recording as base64.
- Root `com.trueskate.remotexpc-tunnel` must be running. Otherwise recording
  attachments accumulate and wedge the recorder. Recovery uses the documented
  attachment cleanup tool after confirming the daemon; never hammer recorder start.
- Do not restart a healthy WDA service to test a hypothesis. A running WDA
  may survive an expired signing profile that prevents rebuilding it.
- Respect recorder start-failure caps, foreground guards, per-segment
  calibration, and decoded frame-count checks. Screenshot menu checks are
  optional where every gesture path avoids the protected controls
  (`DEPLOYMENT.md`). Never weaken admission to pass a smoke test.
- Training loaders exclude contamination markers. `.menu` is always excluded;
  stricter loaders also enforce their existing calibration and trace contracts.
- Preserve normalised coordinates and curved/overlapping/spin gestures. iPhone
  11 Display Zoom yields 375×812; XR devices expect 414×896. No y offset.
- Notifications belong to the fleet incident/recovery mechanism. No periodic
  failure reminders or agent-generated messages without explicit authorization.
- Collection smoke output must use an isolated directory and honest park provenance.
- Human-review viewer servers on the rig are temporary. Once the reviewer has
  returned the exported report's path, shut that viewer's server down. Do not
  leave old viewer servers running.
- Open pages for the user (viewers, dashboards, links) in Google Chrome, the
  user's preferred browser: `open -a "Google Chrome" <url>`.
- No paid cloud training or new research tranche without explicit authorization.

## Research hygiene

- Durable findings update `research/SYSTEM_MAP.md`, reference docs or
  individual experiment records. Historical events do not become current
  operating instructions automatically.
- `research/ARCHIVE.md` is byte-for-byte append-only. Append corrections that
  reference an existing ID; never edit/reorder/delete published entries.
- Retiring a whole system, research direction or batch of tools gets an ARCHIVE
  entry: full SHA, tag, original paths, recovery command and external artifact
  locations. Verify GitHub preservation before deleting source. Deleting single
  files from `main` needs only a clear commit message: `main` is never rewritten.
  Journal trims need no tag or ARCHIVE entry.
- Run `python scripts/ops/check_archive.py --base origin/main` and
  `python scripts/ops/check_public_safety.py --base origin/main` before merging.
  The second blocks files over 500 KB, JSON over 2,000 lines, UDIDs, Tailscale
  addresses and personal email addresses. Changing the checkers/workflow is a
  deliberate policy change requiring review.
- Git tags do not back up ignored datasets, local logs, `.env`, or cloud volumes.
- Research holdouts are not refactor tests. Use synthetic fixtures and existing
  offline tests; preserve model/checkpoint and dataset schema compatibility.
