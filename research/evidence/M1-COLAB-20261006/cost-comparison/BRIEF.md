# Colab vs Modal for TrueSkate-AI Model 1 training: shared brief (2026-10-13, AEDT)

Two research agents share this folder: `web` (internet research on Colab vs Modal) and `data`
(our measured data → time and cost for identical runs). Write your findings here as you go:
`web-findings.md` and `data-findings.md`. Message your peer with SendMessage for anything you
need from them; also read their findings file before finalising.

## The decision

The operator trains occasionally, not constantly. They want to know whether Colab's cost
effectiveness makes it worth using in place of Modal, even if seeds cannot run in parallel.
The standard job: **3 seeds × 40 epochs** of Model 1 with an identical recipe.

## The workload

- Model 1: ~31k parameters, ~105 GFLOP per batch of 8; 32 frames at 64×144; 16 base
  channels; temporal mixer (depthwise 3-D conv). Full precision. AdamW 1e-3, cosine, batch 8.
- Data: experiment `model1_expand_20260930`, 44 sealed shards of video clips (1.16 GB, 22,324
  samples): train 18,394, validation 1,965 (test not scored).
- Without a decoded-frame cache the job is **CPU-decode-bound**, not GPU-bound. The raw cache
  needs ~16.8–17 GiB RAM.
- Pinned environment: Python 3.11, torch 2.12.0 (+cu130), opencv-python-headless 4.13.0.92.

## Modal reference (M1-EXPAND-20260929, seed 0)

- L4. 3.12–3.19 h per seed; epoch 1 918 s (fills the cache); later epochs ~265 s.
  Machine RAM 64 GiB (cache on).
- About **US$4.15 per seed, US$12.46 for three**; seeds ran in parallel.
- Last-10-epoch validation mean 90.92%; best 91.25% (epoch 39).
- The Modal credit balance and plan are not established.

## Colab free-tier benchmark (M1-COLAB-20261006), stopped by the operator after epoch 19

- Free T4 (15 GB), 2 vCPU, ~12.7 GB RAM, so the cache was **off**.
- Epochs 1–19: ~925–953 s each (median ~941 s). Validation tracked Modal: epoch 1 43.5% vs
  46.6%; epoch 2 64.4% vs 65.1%.
- Per-session setup ~520–640 s: input pull 15–18 s, install 62–72 s, staging 510–540 s, plus
  preflight on the first session. Checkpoint upload per epoch 12–17 s.
- Colab VMs boot before they are assigned. Allocation is charged from VM boot.
- **Session-1** (planned stop after epoch 5): 1.716 h. **Session-2** was cut by Colab after
  4.15 h ("disconnected due to inactivity or reaching its maximum duration"; no prompt seen),
  mid-epoch 20. Total 5.94 GPU-h for 19 epochs.
- Uninterrupted projection: 10.46 h per seed.
- Self-imposed rules: one account (multiple accounts break Colab's terms); no keep-alives;
  at most 2 connection attempts per UTC day; 12 GPU-h within 72 h per seed.
- Operator effort: a temporary Chrome on the rig, sign-in, consent prompts and step clicks
  (timeline in `ops/operator_effort.log`).
- Security: Colab's `drive.mount` asks for all-Drive access, which the operator refused. Storage
  used rclone with the `drive.file` scope pinned to one folder. The operator wants a small
  blast radius for any credential.

## Earlier efficiency research (estimates, not measured)

- A zstd-3 compressed frame cache is about 5.7 GiB and fits free Colab RAM. It adds about +50 s
  to epoch 1 and +40 s to later epochs. Estimated cached T4 epoch: 300–550 s, so 4–6.5 h per seed.
- Other levers: GPU-side uint8→float, background checkpoint upload, and one staging pass instead
  of 3–4 (saves minutes per session).
- The official Colab CLI (googlecolab/google-colab-cli 0.7.x) can create, run and stop
  sessions headlessly. It asks for the `cloud-platform` and `drive.file` scopes, and whether a
  long run survives is unproven.
- Kaggle offers ~30 GPU-h per week. Colab pay-as-you-go T4 was reported at ~$0.12/h; verify this.

## Rules for both agents

- Research and analysis only. Never launch cloud jobs or use the `modal`, `rclone`, `gcloud`
  or `colab` CLIs. Never read credential files (`tmp/colab-benchmark-20261006/rclone/`,
  `.env`, `~/.modal.toml`). Never print secrets or private Drive/notebook URLs or IDs.
- Do not edit tracked repo files or commit. Write only inside this folder.
- Cite every price with its source URL and the date you read it. Mark estimates as estimates.
