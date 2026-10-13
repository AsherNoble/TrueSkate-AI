# web findings: Colab vs Modal (read 2026-10-13)

Legend: **V** = verified on an official source; **T** = third-party measurement or report;
**E** = estimate or inference. All prices were read on 2026-10-13 unless marked otherwise.

## 1. Pricing

### Modal (V, https://modal.com/pricing)
| Item | Per second | Per hour |
|---|---|---|
| T4 | $0.000164 | $0.590 |
| L4 | $0.000222 | $0.799 |
| A10 | $0.000306 | $1.10 |
| L40S | $0.000542 | $1.95 |
| A100 40GB | $0.000583 | $2.10 |
| A100 80GB | $0.000694 | $2.50 |
| RTX PRO 6000 | $0.000842 | $3.03 |
| H100 | $0.001097 | $3.95 |
| CPU (per physical core = 2 vCPU) | $0.0000131 | $0.0472 |
| Memory (per GiB) | $0.00000222 | $0.00799 |

- GPU, CPU and memory are billed separately and add up. Billing uses max(requested, used).
  The minimum is 0.125 core and 128 MiB.
- Multipliers: region selection 1.15–1.75x; non-preemptible 3x.
- Plans: Starter $0/mo with $30/mo credits (GPU concurrency 10, 100 containers); Team $250/mo
  with $100/mo credits (GPU concurrency 50).
- Volumes cost $0.09/GiB-month, with 1 TiB/month free. Egress costs $0.04/GiB, with 1 TiB/month
  free on Starter.
- Billing covers app load and execution. Containers idle for 60 s after the last input by default.
- Sanity check (E): one L4 seed at 3.15 h costs $0.80 × 3.15 = $2.52, plus 64 GiB × $0.00799 ×
  3.15 = $1.61, so about $4.13 before CPU. This matches the brief's ~$4.15/seed.

### Colab
- Units and prices (V, `colab.research.google.com/signup` page source, served localised to AUD
  incl. GST):
  - Pay As You Go: 100 CU for A$15.13, 500 CU for A$75.62.
  - Pro: A$15.13/mo, 100 CU/mo.
  - Pro+: A$75.62/mo, 600 CU/mo ("An additional 500 compute units per month, totaling 600").
- Expiry (V, same page): "compute units purchased from Pay As You Go or granted via Colab Pro or
  Colab Pro+ are not refundable and not transferable, and expire after 90 days."
- USD prices (T, several third-party pages agree): $9.99 per 100 CU, so $0.0999/CU; $49.99 per
  500 CU; Pro $9.99/mo; Pro+ $49.99/mo.
- Compute units per hour (T, Chris McCormick, measured 2026-03-10 in S. California,
  http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/). Google publishes no rate
  table, and the FAQ says rates vary.

  | Runtime | CU/h | ≈ US$/h |
  |---|---|---|
  | T4 | 1.19 | $0.12 |
  | L4 | 1.71 | $0.17 |
  | A100 40GB | 5.40 | $0.54 |
  | A100 80GB | 7.52 | $0.75 |
  | G4 (RTX PRO 6000) | 8.71 | $0.87 |
  | H100 | not measured | — |

  The April 2024 rates were higher: T4 1.84, L4 4.82, A100 11.77.
- Machine shapes (V, official Colab CLI docs): CPU, T4 and A100 offer standard and high-RAM
  (`shape=hm`, Pro/Pro+ only). L4, TPU v5e1 and TPU v6e1 offer one shape only. The CLI lists GPUs
  T4, L4, G4, H100 and A100, and TPUs v5e1 and v6e1.

- Shapes, RAM and vCPU (T, varlog.info, measured 2024-10-01 on Colab Pro,
  https://varlog.info/colab-credit-consumption-comparison-2024/). The CU/h column is stale; the
  shapes are probably unchanged.

  | Shape | vCPU | RAM | CU/h (2024-10) | 17 GiB cache fits? |
  |---|---|---|---|---|
  | CPU | 2 | 13 GB | 0.07 | no |
  | CPU High-RAM | 8 | 51 GB | 0.14 | yes |
  | T4 standard (free uses this) | 2 | 13 GB | 1.58 | no (zstd 5.7 GiB cache only) |
  | T4 High-RAM | 8 | 51 GB | 1.67 | yes |
  | L4 (one shape only) | 12 | 53 GB | 3.00 | yes, but 3 private caches do not fit |
  | A100 40GB | 12 | 84 GB | 10.59 | yes, 3 private caches fit |

  - A GitHub report shows the A100 at "System RAM 83.5 GB".
  - Our own free T4 measured 2 vCPU and ~12.7 GB (M).
  - H100 and G4 shapes were not found.
- Who gets High-RAM:
  - V (signup FAQ): "Subscribing to Colab Pro or Pro+ gives you access to higher RAM runtimes."
  - V (CLI docs): `--high-mem` "Requires Colab Pro or Pro+ entitlement".
  - E: PAYG alone may NOT unlock T4 High-RAM. L4 and A100 are premium GPUs available to any
    positive CU balance. L4 comes only in its 53 GB shape, so PAYG plus L4 still fits the cache.
- CU/h is not an official figure. It drifts (T4 1.19–1.84, L4 1.71–4.82, A100 5.40–15 across
  2024–2026 reports), and the live rate shows in `colab usage` or the resources panel.
- Google AI plans now bundle Colab CU allotments, and on higher plans background execution
  (V, FAQ). This applies only if the operator already pays for one.

### Colab Enterprise (brief)
- It bills per hour as a GCP product: the GCE machine plus a GPU plus a management fee. T4 is ~$0.42/h
  GPU plus a $0.07 fee; L4 is ~$0.67/h plus $0.11 (T, from cloud.google.com/colab/pricing search
  snippets; the page is JS-rendered and I could not read it directly).
- Idle shutdown defaults to 180 min.
- It needs a GCP project with billing and IAM, which is a bigger credential and setup surface.
  Its price is in Modal's range, not consumer Colab's. Not attractive here (E).

## 2. Limits and reliability

### Colab (V, official FAQ https://research.google.com/colaboratory/faq.html, read 2026-10-13)
- Free tier: "notebooks can run for at most 12 hours, depending on availability and your usage
  patterns."
- Paid: "In general, notebooks can run for at most 12 hours ... You can expect to experience
  backend termination if you exhaust your available compute units."
- Pro+: "continuous code execution for up to 24 hours if you have sufficient compute units. Idle
  timeouts only apply if code execution terminates."
- Signup FAQ: "Users without a paid subscription should not rely on execution to continue in the
  background; execution will be interrupted when user interaction ceases, and the VM will be deleted
  soon after that." It also says "Colab Pro users will see extended execution times ... based on
  availability."
- Idle timeout values and usage limits are deliberately unpublished ("Colab does not publish these
  limits"). The ~90 min idle figure is folklore (T).
- Our free T4 was cut at 4.15 h with no prompt (M). That is consistent with the dynamic free limits.
- Lockout after heavy free use: no official number. The FAQ's "Wait a day" refers to Drive quotas,
  not GPU limits. Anecdotes range from 12–24 h to more than 10 days (T: colabtools #1964, #3045,
  #2100).
- Concurrency is unpublished. A Google Colab PM said on colabtools #4121 (2023-11-16): "If you want
  to run more than one concurrent high ram session today you can upgrade to Pro+." Pro and PAYG are
  therefore effectively one high-RAM GPU session at a time. Pro+ allows more than one, but the number
  is unpublished and it was refused in 2021–22 reports. For planning, treat seeds as serial, or run
  3 seed processes in one VM (E).
- Billing: CUs burn while the runtime is assigned, idle or not (V, CLI skill doc: "idle VMs burn
  compute units"). Allocation starts at VM boot (M, brief).
- Preemption: there is no formal concept. Runtimes "may be terminated" and resources are "not
  guaranteed".

### Modal (V, modal.com/docs, read 2026-10-13)
- Function timeout: default 300 s, configurable from 1 s to 24 h. Each retry attempt gets a fresh
  timeout. Our ~3.2 h seeds fit easily.
- Preemption: "All Modal Functions are subject to preemption by default ... Preemptions are rare."
  When it happens, the input is gracefully terminated and restarted. Likelihood grows with run
  length. `nonpreemptible=True` is NOT supported for GPU functions; it is CPU-only and costs 3x CPU
  and memory. Checkpointing is advised. Our trainer already resumes from per-epoch checkpoints (M).
- Retries: `retries=n` uses a fixed 1 s delay, or `modal.Retries` for backoff. Container crashes
  are rescheduled.
- Parallelism: Starter allows 10 concurrent GPUs and 100 containers; Team allows 50 GPUs. Three
  parallel seeds are trivial.
- Resources: the default request is 0.125 core and 128 MiB. The CPU soft limit is request + 16
  cores, so decode can burst. Billing is max(request, used). The memory limit is hard (OOM-kill)
  when set.
- Billing covers cold start or app load plus a 60 s idle window by default.

## 3. Automation and operator effort

### Colab
- Official CLI: googlecolab/google-colab-cli (Apache-2.0).
  - Repo created 2026-04-17. Latest tag v0.7.4 (2026-09-25). 1.5k stars, 25 open issues.
  - Linux and macOS only.
  - `colab new/run/exec/upload/download/stop/usage`, `--gpu T4|L4|G4|H100|A100`, `--high-mem`.
  - `colab run` provisions, runs and tears down in one command.
- Liveness (V, CLI docs, 2026-09-25): "VM liveness is automatically maintained by the Colab backend
  based on kernel activity. The CLI runs no keep-alive daemon."
  - A Google maintainer said on issue #160 (2026-10-01): "Colab mostly depends on checking for live
    connections to the VM ... If you shut down the SSH connection, the VM will likely be shut down
    soon after ... keep the SSH connection alive for the duration".
  - So a headless run needs a local process (the rig) holding the connection for the whole run.
    "Fire and forget" is not supported without Pro+ background execution.
- Known long-run problems (V, GitHub issues):
  - #106, #147, #155 and #140: sessions pruned or unreachable after ~1 h runtime-proxy token expiry.
    Fixed in 0.7.4 (#149) on 2026-09-25, only 2.5 weeks ago.
  - #115: a Pro+ user's G4 sessions terminated after ~1.5 h (closed, no explanation).
  - #116 (OPEN, reproduced on 0.7.4 on 2026-10-03): refusing any requested scope on Google's
    granular consent screen (e.g. `cloud-platform`) breaks login. Workarounds then force a browser
    re-login every hour, which "rules out unattended and agent use".
  - #113 (OPEN): `colab drivemount` times out on headless hosts.
  - #82 (OPEN): a long exec timeout can spin a local CPU core.
  - Long multi-hour training survival through the CLI is unproven in public reports (E).
- Terms (V):
  - Additional ToS (last modified 2026-08-13): you may not "access Colab other than as authorized by
    Google". The official CLI is Google's own, hence authorized (E, but a strong inference).
  - FAQ, free runtimes without a positive CU balance: "remote control such as SSH shells, remote
    desktops" and "bypassing the notebook UI to interact primarily via a web UI" are disallowed and
    "may be terminated at any time without warning".
  - So `colab ssh` or `colab console` on the free tier is risky, and `colab exec` on free is a grey
    area. With a positive CU balance these restrictions are lifted ("You can remove these types of
    restrictions by purchasing one of our paid plans ... and maintaining a positive compute unit
    balance").
  - Our current agent-driven Chrome over CDP uses the notebook UI. It is not named as disallowed,
    but "Colab prioritizes users who are actively programming in a notebook" (E: grey).
  - Multiple accounts to work around limits is disallowed on all runtimes (V).
- Runtime version pinning is UI-only ("Runtime Version" dropdown), with past versions kept for
  1 year (V, runtime-version FAQ). I found no CLI flag for it (E).

### Modal
- Python SDK and CLI (`modal run`/`deploy`, `.spawn()`/`.map()`) run fully headless from the rig.
  The token can be minted with an expiry (`modal token new --expires-in 7d`, V).
- Volumes hold the dataset and checkpoints (1 TiB/month free; ~364 GiB used per data agent).
  The workflow already exists in-repo (M).
- Operator effort: one approval, then one command for 3 parallel seeds (E, data agent).

## 4. Security and credential scope

| | Colab browser + our rclone | Colab official CLI | Modal |
|---|---|---|---|
| Google sign-in | full Google account session in a rig Chrome profile | OAuth token at `~/.config/colab-cli/token.json` | none |
| Scopes | `drive.file` (rclone) plus Colab web session; `drive.mount` would request access to "any files in your Google Drive" (V, FAQ), which the operator refused | `openid`, `userinfo.*`, **`cloud-platform`**, `colaboratory`, `drive.file` (V, `src/colab_cli/auth.py`). ADC path needs `cloud-platform` too. Narrower auth requested in #71 (OPEN). | workspace API token (`ak-`/`as-`) with full access to the workspace: run code (so it can read workspace Secrets and Volumes) and spend. Service users and RBAC need Team/Enterprise (V); Starter has a personal token only. Optional token expiry (V). Spend limit and budget controls exist (V, docs/guide/budgets; plan eligibility unstated). |
| Blast radius if leaked (E) | `drive.file` = files the rclone app created or opened. Testing-status OAuth app tokens expire after 7 days (V, developers.google.com/identity/protocols/oauth2), so re-consent is due every 7 days unless the app is published. The browser profile itself is a full Google session (largest risk). | `cloud-platform` = all GCP resources the account can reach (small if the account has no GCP projects or billing; large otherwise), plus Colab CU spend | Modal compute spend up to the spend limit, plus our Modal volumes and secrets. No Google data. |

## 5. Reproducibility (Python 3.11, torch 2.12.0+cu130)
- Colab:
  - The default image moves under us. backend-info (V, github.com/googlecolab/backend-info,
    commit 2026-10-03) shows Python 3.13.15 and torch 2.11.0+cu130. "Update Colab to CUDA 13.0"
    landed 2026-09-30, and cuda-compat-13-0 is 580.126.20.
  - Runtime 2025.07 was the last with Python 3.11 (V, runtime-version FAQ). 2026.07 has Python
    3.12.13 and torch 2.11.0.
  - We pin by installing Python 3.11 and torch 2.12.0+cu130 ourselves each session (62–72 s, M). The
    evidence shows Python 3.11.17, torch 2.12.0+cu130, CUDA 13.0, cuda_available true on the free
    T4 (M).
  - Risk: the driver and image change without notice. Before 2026-09-30 the image was CUDA 12.x
    (E: cu130 wheels then depended on host driver support).
- Modal:
  - The image is ours (`python_version="3.11"`, pinned wheels), cached per layer.
  - Host driver 580.95.05 with CUDA driver API 13.0 (V, modal.com/docs/guide/cuda). "Older versions
    in the 12.* and 13.* series are guaranteed to be compatible" with the host driver.
  - The workspace-level Image Builder Version changes "every few months", but only when we opt in
    (V, docs/guide/images).
  - The driver can still change under us (E), but CUDA forward-compatibility within 13.x is
    promised.

## 6. Other options (brief)
- Kaggle (T/E, not verified on kaggle.com, whose docs page is JS-rendered):
  - ~30 GPU-h/week (floating), 12 h sessions, P100 or 2×T4.
  - ~29–30 GB RAM, so the 17 GiB cache would fit (E).
  - `kaggle kernels push` runs committed notebooks headless in the background under an official
    API token scoped to Kaggle only.
  - $0. Could beat free Colab on every axis for occasional runs. Worth one bounded check if cost
    ever matters.
- Lightning, RunPod and others are out of scope for this question.

## 7. Verdict (web agent; E unless marked)
- Modal has more pros for us.
- Colab's one decisive pro is cash price per GPU-hour. Paid L4 runs ~$0.17–0.48/h against Modal L4
  plus memory at ~$1.04–1.32/h, so a cached 3-seed L4 job is ~$1.9–5.4 on Colab against
  $10.4–12.5 on Modal (data agent S4/S5). Free Colab is $0.
- Everything else favours Modal:
  - parallel seeds and ~3.3 h wall clock, against ≥11 h serial on Colab;
  - headless and fire-and-forget, against Colab needing a live client connection all run;
  - documented limits (24 h timeout, rare preemption with auto-restart), against Colab's
    unpublished limits and a 4.15 h cut seen;
  - a pinned image with a stated CUDA 13.0 contract, against a Colab image that changed CUDA 6 days
    before our run;
  - a Modal-only credential, against Colab needing a Google account session or a `cloud-platform`
    token.
- Modal Starter's $30/month credit (V) covers ~2 runs per month at as-run settings, or ~3 with a
  32 GiB memory request. Inside that, Modal costs $0 cash and Colab saves nothing.
- Without Modal credits, Colab saves ~$6–10 per run. At occasional volume (≤1–2 runs/month) that is
  under ~$10–20/month, and the 100 CU / 90-day pack granularity eats part of it.
- Flips toward Colab paid L4 when all of these hold:
  - (a) Modal usage regularly exceeds credits, by roughly ≥4–5 runs/month or other Modal spend;
  - (b) a bounded smoke test proves a ≥3.7 h CLI-held L4 session survives;
  - (c) the operator accepts `cloud-platform` scope, or uses a dedicated Google account with no GCP
    projects.
- Flips toward Colab free (or Kaggle) only if cash must be $0, multi-day calendar time is acceptable,
  and the zstd cache is built.
