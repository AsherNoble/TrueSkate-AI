# data-findings: 3 seeds × 40 epochs, Modal vs Colab (2026-10-13)

Status: final v2. **M** = measured from our records; **E** = estimate.
Pricing inputs marked (web) come from the `web` agent (`web-findings.md`), read 2026-10-13.
Scenario arithmetic: `scenarios.py` in this agent's scratchpad (pure arithmetic on the inputs below).

## 0. Headline (3 seeds × 40 epochs, identical recipe)

| Scenario | GPU-h / seed | GPU-h / 3 seeds | Wall clock, 3 seeds | $ / seed | $ / 3 seeds | Operator min |
|---|---:|---:|---|---:|---:|---|
| **1. Modal L4 as run** (M) | 3.16 | 9.47 | ~3.2–3.6 h (parallel) | 4.15 (E bill 4.25–4.50) | **12.46** (E bill 12.8–13.5); $0 cash if Starter $30/mo credit is unused | ~0 (approve; agent launches) |
| **2. Colab free T4, no cache** (M inputs, extrapolated) | 11.8 | ~35.5 | **≥ 5 UTC days** (9 sessions at 2/day); 9+ days at 1/day or with lockouts | 0 | **0** | ~0 per run; one-time ~6–7 min + OAuth client |
| 3. Colab free T4, zstd cache (E) | 4.1–7.3 | 12–22 | 2–3 UTC days (3–6 sessions) | 0 | 0 | as 2; plus build + validate the cache |
| **4a. Colab PAYG L4, cache on** (E) | 3.7 | 10.2–11.2 | ~11–12 h serial, one day | 0.64–1.79 of CU | **1.91–5.37 of CU**; minimum buy $9.99/100 CU, expires in 90 days | ~0; one-time purchase |
| 4b. Colab Pro T4 High-RAM, cache on (E) | 4.1–6.8 | 12–20 | 12–21 h serial (one ~4–7 h session per seed) | 0.53–1.13 of CU | 1.58–3.38 of CU, plus Pro $9.99/mo | ~0 |
| 4c. Colab paid T4 standard, no cache (E) | 11.0 | 33 | ~33 h serial (≥ 3 sessions under the 12 h cap) | 1.31–2.02 | 3.93–6.07 | ~0 |
| 4d. Colab A100-40, 3 seeds in one VM (E; needs harness change) | 1.1–1.5 | 3.3–4.4 | ~3.3–4.4 h | 0.6–2.2 | 1.77–6.55 | ~0 |
| **5a. Modal L4, 32 GiB, 1 core** (E) | 3.16 | 9.47 | ~3.2–3.4 h (parallel) | 3.48 | **10.44** | ~0 |
| 5b. Modal T4, 24 GiB, 1 core, cache on (E) | 3.5–6.2 | 10.5–18.6 | 3.5–6.3 h (parallel) | 2.91–5.15 | 8.72–15.46 | ~0 |

**Per-run cash, headline:** Modal L4 as run ≈ $12.5 (≈ $10.4 at 32 GiB), unless free Starter credit
covers it. Colab PAYG L4 serial ≈ $2–5 of compute units, but units come in $9.99 packs that expire
after 90 days, so a once-a-quarter job really pays ~$10. Free Colab is $0 but takes ~5 days and
~3.7× the GPU-hours, because the 17 GiB frame cache does not fit in 12.7 GB of RAM.

## 1. Measured inputs

### Modal (M1-EXPAND-20260929, seed 0 history + record)

| Item | Value | Source |
|---|---|---|
| Machine | `train_remote`: gpu=L4 (pinned via `MODAL_TRAIN_GPU` + `--required-gpu L4`), `memory=65536` (64 GiB requested), **no `cpu=`** → Modal default 0.125 physical core, timeout 24 h, no region pin | `scripts/model1/train_basic_linear_modal.py:189`; `src/trueskate_ai/model1/scaling.py:46-60` |
| Seed-0 epochs | epoch 1 = 918 s (decode + fill cache); epochs 2–40 median 270 s (mean 271); sum 11,481 s = 3.19 h | `inputs/reference/modal_seed0.json` (M) |
| Three seeds | 3.12–3.19 h each (epoch-seconds sums); 9.47 h total; run in parallel (~3.2 h wall clock) | M1-EXPAND record (M) |
| Cost formula | 9.47 h × **$1.3166/h** = $12.46; per seed $4.155 | `research/evidence/M1-EXPAND-20260929/summarize_training.py` (commit `6ab7b49f`) |
| $1.3166/h split | L4 $0.7992 + memory 64 GiB × $0.007992 = $0.5115 + CPU 0.125 core × $0.04716 = $0.0059 | `scaling.py` rates (retrieved 2026-09-04; web agent confirms unchanged 2026-10-13) |
| Per-seed split | GPU **$2.52** (61%), memory **$1.61** (39%), CPU **$0.02** (<1%) | arithmetic |
| ×3 split | GPU $7.57, memory $4.84, CPU $0.06 = $12.47 | arithmetic |

**So the $4.15/$12.46 includes GPU + 64 GiB memory + the 0.125-core CPU floor.** It is a computed
estimate (epoch seconds × list rates), not an invoice. It leaves out:
- cold start / image load, pre-epoch manifest and content verification, final volume commit and the
  60 s idle window (not recorded; E: +2–10 min per seed ≈ +$0.04–0.22 per seed);
- CPU above the 0.125-core floor. Modal bills max(request, used) (web). The trainer keeps a Python
  main process busy all run (DataLoader `num_workers=0`, decode in-process), so billed CPU is probably
  0.5–1 physical core (E: +$0.07–0.15 per seed);
- the separate validation autopsy (~$0.3) recorded in M1-EXPAND.
- Realistic bill (E): **$4.25–4.50 per seed, $12.8–13.5 for three.** M1-EXPAND itself reported
  "≈ $12.8 against the $12.50 approval" including the autopsy.
- Volume storage: ~364 GiB after the 2026-10-01 deletion, under the free 1 TiB → $0
  (`research/history/MODAL_STORAGE_20261001.md`).
- Plan / free-credit balance: **not established in the repo.**

### Colab free T4 (M1-COLAB-20261006, stopped after epoch 19)

| Item | Value (M) |
|---|---|
| Runtime | Tesla T4 15 GB, 2 vCPU, torch threads 1, ~12.7 GB RAM; cache off; peak process RAM 2.03 GB; peak VRAM 3.38 GB allocated / 5.41 GB reserved |
| Epoch time | epoch 1 = 946 s; epochs 2–19 median 943.5 s (mean 942, range 925–953) |
| Checkpoint upload + verify | 11.4–12.0 s (session 1), 14.7–17.8 s (session 2); total 282 s for 19 epochs |
| In-session cycle per epoch | 957 s (session 1), **963.5 s** (session 2), from checkpoint-verified timestamps |
| Session setup (VM boot → first epoch start) | 20.2 min (session 1; includes a one-off preflight and ~7 min before step 2 started); **13.6 min** (session 2, resume) |
| Setup pieces | inputs pull 15–17 s; install 62–72 s (pull included); staging + content verification 510–539 s; `setup_and_preflight` 635 s / 520 s |
| Release after last epoch | 3–6 min (ledger closes conservatively at observed deletion) |
| Session 1 | VM boot 16:30:56 → released 18:13:53 AEDT, planned stop after epoch 5: **1.716 h** |
| Session 2 | VM boot 18:13:00 → cut by Colab ~22:21–22:22 (**~4.15 h**), exactly **14 epochs** (6–19), ~10 min of epoch 20 lost; ledger release 22:26:37 (4.227 h) |
| Total | 5.94 GPU-h (allocation ledger) for 19 epochs; calendar 5.93 h; $0 |
| Uninterrupted projection | 946 + 39 × 943.5 s = 37,743 s = **10.48 h** of epochs (the brief's 10.46 h); with setup, checkpoints and release ≈ **11.0 GPU-h per seed** |
| Preflight | Modal's seed-0 checkpoint scored **identically** on Colab T4: 91.25% (1,793/1,965), difference 0.0 pp |

**Does decode scale with vCPUs?** Not with the frozen trainer: `DataLoader` has `num_workers=0`, so
decode is single-process. The Colab T4 uncached epoch (943 s) ≈ the Modal L4 decode epoch (918 s)
on a very different host, consistent with a one-core decode bottleneck. More vCPUs help only after a
code change (DataLoader workers; E, unmeasured). The lever with no code change is **RAM for the
cache** (~16.8–17 GiB decoded train+val at 64×144; the launcher's budget check counts all 22,324
shard samples = 18.4 GiB).

### Operator effort (coordinator update + `ops/operator_effort.log`)

- **Per Colab run: ~0 active operator minutes** when an agent drives it (the operator only prompted
  the agent; the agent clicked every step over CDP: `ops/colab_ui.cjs`).
- **One-time Colab setup:** Screen Sharing sign-in ~5 min; `drive.file` consent 15 s; enabling the
  Drive API < 1 min; OAuth client creation in Google Cloud: unknown. (The ~7 min `drive.mount`
  consent path is obsolete.) ≈ 6–7 min + OAuth client time.
- **Not operator minutes but real costs:** agent wall-clock attention across every session
  (start, watch, release, ledger), and calendar waits for the 2-attempts-per-UTC-day cap.
- **Recurring asks:** if the operator's rclone OAuth client is in Google's "Testing" status (likely for
  a personal project; its status is not recorded), refresh tokens expire after 7 days (V via web:
  developers.google.com/identity/protocols/oauth2). An occasional
  job therefore needs a `drive.file` re-consent each time (measured at 15 s; E ≤ 1 min), unless the
  app is published. Also possible (E): re-sign-in if the rig Chrome's Google session expires, and
  captcha / "still there?" prompts (none seen in 2 sessions).
- **Modal (E):** one approval + one agent launch command for 3 parallel seeds; ~0–1 operator min.

## 2. Quality comparability (epochs 1–19, validation recovery, M)

| Epoch | Colab T4 | Modal L4 | Δ pp |
|---:|---:|---:|---:|
| 1 | 43.51 | 46.56 | −3.05 |
| 2 | 64.38 | 65.14 | −0.76 |
| 3 | 73.79 | 75.83 | −2.04 |
| 4 | 74.10 | 74.15 | −0.05 |
| 5 | 77.76 | 78.83 | −1.07 |
| 6 | 80.10 | 78.68 | +1.42 |
| 7 | 82.65 | 79.59 | +3.05 |
| 8 | 82.39 | 80.20 | +2.19 |
| 9 | 81.98 | 82.70 | −0.71 |
| 10 | 80.61 | 82.14 | −1.53 |
| 11 | 84.07 | 85.09 | −1.02 |
| 12 | 83.05 | 86.16 | −3.10 |
| 13 | 86.62 | 85.09 | +1.53 |
| 14 | 87.12 | 86.67 | +0.46 |
| 15 | 86.72 | 86.62 | +0.10 |
| 16 | 87.43 | 86.72 | +0.71 |
| 17 | 86.41 | 87.48 | −1.07 |
| 18 | 88.70 | 85.39 | +3.31 |
| 19 | 87.89 | 87.68 | +0.20 |

- Mean Δ: epochs 1–19 **−0.07 pp**; epochs 10–19 −0.04 pp; epochs 15–19 +0.65 pp.
- Components (mean Δ over 1–19): start −1.11 pp, end +0.51 pp, duration −0.09 pp.
- Per-epoch Δ (−3.1 to +3.3 pp) is the size of Modal's own epoch-to-epoch swings (e.g. 87.48 → 85.39
  → 87.68 at epochs 17–19). No systematic gap; evaluation is hardware-consistent (preflight 0.0 pp).
- **Not established:** the 40-epoch outcome (last-10 mean vs 90.92%). Decode with or without the cache
  gives the same frames, so the cache should not change the recipe; GPU type (T4 vs L4) changes cuDNN
  algorithms, so runs are not bit-comparable, as already seen between Modal seeds.

## 3. Scenarios (per seed and for 3 seeds)

Rates: Modal list rates (web, verified 2026-10-13). Colab CU price US$0.0999/CU (web; 100 CU = $9.99
PAYG or Pro $9.99/mo). Colab CU/h (web, third-party): T4 1.19, L4 1.71, A100-40GB 5.40 (2026-03);
older 2024 rates T4 1.84, L4 4.82, A100 11.77, shown as the high end.

### S1. Modal L4 as run (M)
- 3.157 GPU-h per seed (3.12–3.19); **9.47 GPU-h** for 3; parallel → **~3.2–3.4 h wall clock**.
  Upper bound from git: the training preregistration commit is 07:45 AEST on 2026-10-01 and seed 0's
  payload `created_at` is 11:21 AEST, so launch → seed-0 result took ≤ 3.6 h (M); the result was
  recorded at 11:51.
- **$4.15 per seed, $12.46 for 3** (GPU $7.57 + memory $4.84 + CPU $0.06). Realistic bill E $12.8–13.5.
- If the account is on Modal Starter with its $30/month credit unused (web), a 3-seed job costs
  **$0 cash** (two jobs a month fit). Plan not established.

### S2. Colab free T4, cache off (M inputs; session cap is one observation)
- Model: 13.6 min setup, 963.5 s per epoch cycle, cut at 4.15 h, ~5 min release.
- 14 epochs per session (matches session 2 exactly). Per seed 14 + 14 + 12 → **3 sessions,
  11.8 GPU-h**. 3 seeds: **9 sessions, ~35.5 GPU-h** (chained or separate, same count).
- Uninterrupted it would be 11.0 GPU-h per seed; the 4.15 h cut costs ~0.8 GPU-h per seed
  (two lost partial epochs + one extra setup).
- Calendar: 2 attempts per UTC day → per seed 2 UTC days; **3 seeds ≥ 5 UTC days** (9 sessions),
  assuming every attempt gets a T4 and Colab's undocumented usage limits do not lock the account
  out. At 1 session per day: 9 days. Free-tier GPU lockouts after heavy use are unpublished;
  anecdotes run from 12–24 h to more than 10 days (web, T), and this job needs ~35 GPU-h in ~5 days.
- The 4.15 h cut is one observation of an unpublished, dynamic limit (FAQ: "at most 12 hours,
  depending on availability and your usage patterns"). A 12 h session would make it 4 sessions /
  2 days; a shorter one, more.
- The self-imposed 12 GPU-h / 72 h per-seed budget leaves only ~0.2 h margin per seed.
- $0. ~0 operator min per run + one-time setup.

### S3. Colab free T4 + zstd compressed cache (E)
- Cached epoch 300–550 s (earlier research, unmeasured) + ~20 s checkpoint; first epoch of every
  session rebuilds the cache (~996 s). Cache ~5.7 GiB fits 12.7 GB RAM.
- Fast end (300 s): 4.06 h per seed uninterrupted → 1 session per seed (only just under a 4.15 h cut);
  3 seeds chained **3 sessions, ~12.2 GPU-h, 2 UTC days**.
- Slow end (550 s): 6.8 h per seed → 2 sessions, 7.3 GPU-h per seed; 3 seeds chained
  **6 sessions, ~22 GPU-h, 3 UTC days**.
- Needs a harness/trainer change (compressed cache) that is not built or validated.

### S4. Colab paid (E; CU/h third-party, CU price and shapes from web)
Shapes (web, T: varlog.info 2024-10, believed unchanged): T4 standard 2 vCPU / 13 GB (cache does
not fit); T4 High-RAM 8 vCPU / 51 GB (fits; needs Pro or Pro+, not PAYG alone); L4 one shape only,
12 vCPU / 53 GB (fits; any positive CU balance, so PAYG works); A100-40 12 vCPU / 84 GB (fits three
private caches). CU/h low = McCormick 2026-03, high = 2024 figures (A100 ~15 in a June 2026 review).

| Shape | h per seed | h for 3 (serial) | CU/h used | $ per seed | $ for 3 |
|---|---:|---:|---|---:|---:|
| T4 standard, cache off | 11.0 | 33 (32.4 chained) | 1.19–1.84 | 1.31–2.02 | 3.93–6.07 |
| T4 High-RAM (Pro), cache on, 300–550 s | 4.1–6.8 | 12.1–20.3 (11.2–19.4 chained) | 1.3–1.67 | 0.53–1.13 | 1.58–3.38 |
| **L4 (PAYG), cache on, ~270 s + 20 s ckpt** | 3.7 | 11.2 (10.2 chained) | 1.71–4.82 | 0.64–1.79 | **1.91–5.37** |
| A100-40, 3 seeds in one VM, 250–350 s | — | 3.3–4.4 wall (one GPU) | 5.40–15 | — | 1.77–6.55 |

- **Epoch-time assumption (E):** Colab L4 cached epochs ≈ Modal's measured 270 s. The frozen trainer
  decodes in one process, so Colab's 12 vCPU neither helps nor hurts epoch 1 (≈ 946 s, as on T4);
  later epochs depend on GPU and the in-process cache path. If Colab's slower host CPU pushes cached
  epochs to ~320 s, L4 costs rise ~9% (to ≤ $5.84 for 3 at 4.82 CU/h).
- **Serial:** Pro/PAYG are effectively one high-RAM session at a time (Colab PM, colabtools #4121,
  via web). L4 chained (one setup, cache reused across seeds) is 10.2 h: under the FAQ's "at most
  12 hours" but too close; three separate ~3.7 h sessions back-to-back (~11–12 h, one day) is safer.
  Without Pro+ background execution a live client connection must be held the whole time (web).
- **A100 3-in-1 (E):** 84 GB holds three 18.4 GiB caches; each seed runs as its own process on one
  GPU. Needs a harness change and an unmeasured GPU-contention check. GPU-h shown are VM hours.
  Note: A100 and L4 can use TF32 for convolutions by default (cuDNN), T4 cannot; the T4-vs-L4 curve
  match through epoch 19 suggests this does not matter, but it is not tested on A100.
- **Purchase granularity:** PAYG sells 100 CU for $9.99, expiring after 90 days (web). A 3-seed L4
  job uses ~19–54 CU, so one pack covers 2–5 jobs **only if they fall within 90 days**; for one job
  per quarter the effective cash cost is **$9.99 per job**. Pro ($9.99/mo, 100 CU) has the same
  90-day expiry, and adds High-RAM T4 for subscribed months.

### S5. Modal alternatives (E)
Rates per hour: L4 $0.7992, T4 $0.5904, memory $0.007992/GiB, CPU $0.04716/physical core.

| Configuration | $/h | h per seed | $ per seed | $ for 3 |
|---|---:|---:|---:|---:|
| As run: L4, 64 GiB, 0.125 core | 1.317 | 3.16 | 4.16 | 12.47 |
| L4, 32 GiB, 1 core | 1.102 | 3.16 | 3.48 | 10.44 |
| L4, 24 GiB, 1 core (tight for an 18.4 GiB cache) | 1.038 | 3.16 | 3.28 | 9.83 |
| T4, 24 GiB, 1 core, cached 300 s | 0.829 | 3.50 | 2.91 | 8.72 |
| T4, 24 GiB, 1 core, cached 550 s | 0.829 | 6.21 | 5.15 | 15.46 |
| T4, 8 GiB, 1 core, no cache | 0.702 | 10.48 | 7.35 | 22.06 |

- The only safe saving is the memory request: 64 GiB is ~3× what the cache needs, and memory is 39%
  of the bill. Requesting 32 GiB saves ~$2 per 3-seed job (~16%) with no change in time.
- T4 is cheaper only if its cached epoch is near 300 s (unmeasured); otherwise it costs more.

## 4. What dominates the difference

- **The frame cache, not the GPU or CPU count, sets the GPU-hours.** With the 17 GiB decoded-frame
  cache, later epochs take ~270 s on L4 (M); without it every epoch re-decodes in one process and takes
  ~943 s on T4 (M), the same as Modal's own decode epoch (918 s). Free Colab's 12.7 GB of RAM cannot
  hold the cache, so it needs ~3.7× Modal's GPU-hours (35.5 vs 9.47 for three seeds). More vCPUs do
  nothing for the frozen trainer (`num_workers=0`).
- **Free Colab's calendar cost comes from its session rules.** The measured 4.15 h cut (14 epochs per
  session) and the 2-attempts-per-UTC-day rule turn ~35 GPU-h into 9 sessions over ≥ 5 days;
  seeds cannot run in parallel. Modal runs all three in parallel in ~3.3 h.
- **Modal's bill is 61% GPU and 39% memory over-request.** It asks for 64 GiB when ~19 GiB is used;
  asking for 32 GiB cuts a 3-seed job from ~$12.5 to ~$10.4 (E) at the same speed. CPU is < 1%
  (rising to a few percent if billed at actual use).
- **Cash is decided by credits and pack sizes more than by hourly rates.** Colab L4 compute is 3–8×
  cheaper per hour than Modal L4 as configured ($0.17–0.48/h vs $1.32/h), so a cached Colab L4 job
  burns only $2–5 of units. But units come in $9.99 packs that expire in 90 days, so an occasional
  job effectively costs ~$10. If the Modal workspace is on Starter with its $30/month credit otherwise
  unused (plan not established), a $12.5 Modal job costs $0 cash, and free Colab saves nothing while
  adding ~5 days and per-session agent babysitting.
- **Quality is not a differentiator** on the evidence so far: the Colab T4 validation curve matches
  Modal's through epoch 19 (mean Δ −0.07 pp), and the reference checkpoint scores identically on both.
  The 40-epoch endpoint on Colab is unmeasured.
- **Operator minutes are ~0 per run on both** when an agent drives them. Colab adds one-time setup
  (~6–7 min plus OAuth client creation) and possibly a 15 s re-consent per job (7-day token expiry);
  the larger Colab cost is agent attention and calendar time, not operator time.
