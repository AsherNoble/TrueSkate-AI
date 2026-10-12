# Offline compensation follow-up plan

Frozen before fitting or evaluating new compensation models. This is a
retrospective analysis of existing experiments: broad reported effects and
selected example traces were already known. The holdouts below are command
holdouts, not a claim of a new blind experiment.

1. Use the recovered Recap touch cubic exactly as extracted; fit no easing
   parameters. Verify its monotonicity and construct its inverse numerically.
2. Development commands: single_100, single_400, single_800, single_1200;
   sub_300_n2, sub_300_n3; boundary_d17_n2. Examine source scheduling code.
3. Freeze simple duration candidates before scoring: identity; nearest total
   60 Hz tick; sum of rounded segment ticks; sum of ceil segment ticks; and
   linear duration error vs segment count fitted only to development commands.
   Evaluate all other main probes and all 100 round-6 commands separately.
   Boundary holdouts exclude boundary_d17_n2. Do not use holdout results to
   claim a corrected universal law.
4. For duration reachability, enumerate existing integer-duration single-move
   and repeated-segment observations. Report empirical collisions and the
   limits of what finite tested grids establish. Pointer-up duration and
   waypoint arrival are distinct measurements.
5. Build an offline inverse-easing reference API. It may return the command
   time for a desired progress value; it must not imply that a time transform
   alone can make an entire single-segment trajectory linear at fixed duration.
6. Read the existing eight human stroke traces. Compare their durations and
   required segments with the current contract, and with the narrow observed
   short-segment envelope. No physical-phone claims follow from simulator data.
7. No new simulator experiments, no production edits, no rig/phone access.
   The smallest next phone validation is a proposal only.

Analysis output: reproducible Python, results JSON, source/input hashes,
compact tables, and REPORT.md. Preserve command IDs and avoid claiming that
the reused round-6 qualification corpus is an untouched research holdout.
