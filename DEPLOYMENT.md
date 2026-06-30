# Rig Deployment & Operations

Single source of truth for how TrueSkate-AI is deployed and operated on the
physical-device rig. Source files, scripts, operators, and AI assistants should
defer to this document for host identity, device roster, the two operating
modes, launchd jobs, and remote access rather than deriving these facts inline
or assuming runtime state.

Scope boundary: this document covers **on-device work only** — data collection
and CMA-ES evaluation, both of which run at 1× real time on the iPhones. Model
*training* (Model 1 trace extractor, Model 2 sequence policy) runs **off-rig**
on a GPU/MPS host (Modal first); it is intentionally out of scope here and has
no dependency on the rig.

Accuracy convention: claims about files and CLI flags are verifiable against the
repository. Claims about what is *loaded and running* on the rig are not — this
document never asserts a launchd job is active. Use the verification commands in
§5 to query the rig directly.

---

## 1. Host

| Property | Value |
|---|---|
| Machine | Intel MacBook (`x86_64`; Homebrew under `/usr/local`) |
| Login user | `training-server` |
| Repo path | `/Users/training-server/trueskate-ai` |
| Remote access | Tailscale SSH from an operator Mac (see §6) |

The host is a LaunchAgent-driven box: WebDriverAgent/`xcodebuild` and USB device
access require a logged-in GUI session, so the rig runs as per-user agents (not
system daemons) with **auto-login enabled** so it returns to a working state
after a reboot or power blip. The sole exception is the root remotexpc tunnel
daemon (§4.2), which must run as a system `LaunchDaemon`.

---

## 2. Device roster

Authoritative port and geometry map: `DEVICES` in
`src/trueskate_ai/rl/device_worker.py`. Do not duplicate port numbers elsewhere;
cite that list. The operational facts:

| Device | Role | WDA port | Logical size | Display Zoom |
|---|---|---|---|---|
| `iPhone_XR` | collection | 8100 | 414 × 896 | **OFF** |
| `iPhone_XS` | collection | 8102 | 375 × 812 | native |
| `iPhone_XR2` | collection | 8103 | 414 × 896 | **OFF** |
| `iPhone_11` | personal | 8101 | 375 × 812 | **ON** |

Role semantics:

- **`collection`** — part of the default 24h roster; eligible for unattended runs.
- **`personal`** — reserved for ad-hoc testing (Asher's `iPhone_11`); never
  grabbed by the default roster. Select it explicitly with `--personal` or
  `--devices iPhone_11`.

Display Zoom is load-bearing and enforced: `DeviceWorker.connect()` aborts on a
logical-size mismatch (the "dim guard" kills services rather than train on a
mis-sized screen). `iPhone_11` must keep Display Zoom **ON** (it reports
375 × 812); `iPhone_XR` / `iPhone_XR2` must keep it **OFF** (414 × 896). See
`GESTURES.md` for why the shared 19.5:9 aspect ratio makes normalised
coordinates device-agnostic regardless.

---

## 3. Operating modes

The rig runs in one of two modes. They are mutually exclusive on a given device
(both drive Appium/WDA on the same ports) and serve different goals.

### 3.1 Mode A — CMA-ES training

Evolutionary optimization of gesture parameters against a reward curriculum.

- **Entry point:** `scripts/run_training.py` — a self-healing supervisor that
  launches services, the status dashboard, and `train_cmaes.py`, wrapping
  everything in `caffeinate` while it runs.
- **Required flag:** `--curriculum <path>` (e.g. `curricula/360_flip.json`).
- **Common flags:** `--initial-mean <trick_library.json>` (warm start),
  `--max-evals` (default 100000), `--pop-size` (default 24), `--seed`
  (default 42), `--status-port` (default 8200), `--no-status-server`,
  `--no-caffeinate`.
- **Children supervised:** `launch_services.py` (Appium + WDA + iproxy per
  device, self-healing), `status_server.py` (dashboard on `:8200`),
  `train_cmaes.py` (restarts with a bumped seed).

Manual run (inside `tmux` so it survives an SSH drop):

```bash
cd /Users/training-server/trueskate-ai
source .venv/bin/activate
python scripts/run_training.py --curriculum curricula/360_flip.json \
    --initial-mean trick_libraries/<warm-start>.json
```

Single-phone test on the personal device:

```bash
python scripts/launch_services.py --personal
python scripts/train/train_cmaes.py --personal \
    --curriculum curricula/360_flip.json --max-evals 48
```

### 3.2 Mode B — SLS / XCTest data collection

Headless 30 fps frame→gesture corpus collection (no reward; independent of
CMA-ES). Fires a random SLS gesture mix and records the screen via Appium's
XCTest screen recording. See `CLAUDE.md` → "SLS Frame→Gesture Data Collection"
for the pipeline rationale.

- **Collector:** `scripts/data/collect_sls_xctest.py` — records bounded
  `--segment-min` `.mov` segments while logging a per-gesture manifest.
  **`--segment-min` must stay short** (default 1.0; ≤ ~90 s): `stop_and_save`
  returns the whole `.mov` as one base64 HTTP response, which fails above
  ~114 MB. Per-device, e.g.
  `python scripts/data/collect_sls_xctest.py --devices iPhone_XR2 --segment-min 1`.
- **Aligner:** `scripts/data/align_xctest_traces.py` — slices each gesture's
  frame window from the `.mov` into per-gesture sample dirs, then deletes the
  `.mov`. Spawned async after each segment by the collector by default.
- **Corpus filter:** `scripts/data/flag_menu_samples.py` — marks replay/menu
  samples with a `.menu` file. Training loaders must exclude any sample dir
  containing `.menu`.

**Critical prerequisite:** the root remotexpc tunnel daemon (§4.2) must be up,
or XCTest recording attachments accumulate until recording wedges with
`ScreenRecordingError Code=7 "Failed to write file"`. This is non-negotiable for
Mode B.

---

## 4. launchd jobs

This is the authoritative inventory of launchd jobs the rig uses. "In repo"
means a plist or installer is committed and the job is fully reproducible from
source; jobs marked otherwise are referenced operationally but their plists are
**not** committed and must be located on the rig itself.

| Label | Type | In repo | Purpose |
|---|---|---|---|
| `com.trueskate.training` | user LaunchAgent | ✅ `deploy/` | Mode A supervisor |
| `com.trueskate.remotexpc-tunnel` | root LaunchDaemon | ✅ `scripts/ops/` | Mode B recording prerequisite |
| `com.trueskate.watchdog.<LABEL>` | user LaunchAgent | ⚠️ logic only (`scripts/collection_watchdog.sh`); plist not committed | Mode B collector liveness alerts |
| `com.trueskate.services` | user LaunchAgent | ❌ not committed | Per CLAUDE.md: Appium + WDA + iproxy + health monitor |
| `com.trueskate.collect.<device>` | user LaunchAgent | ❌ not committed | Per CLAUDE.md: supervised Mode B collectors |

If a job is needed but not committed, treat that as a gap to close (commit the
plist/installer), not as license to assume it exists.

### 4.1 `com.trueskate.training` (Mode A)

Installed and reloaded by `deploy/install_launchd.sh`, which substitutes
`__PYTHON__` / `__REPO__` / `__CURRICULUM__` into
`deploy/com.trueskate.training.plist.template` and bootstraps it. `RunAtLoad` +
`KeepAlive`; with auto-login on, the rig resumes Mode A after a reboot.

```bash
deploy/install_launchd.sh curricula/360_flip.json   # install / update / reload
tail -f logs/supervisor.out.log

launchctl kickstart -k gui/$(id -u)/com.trueskate.training   # restart now
launchctl bootout    gui/$(id -u)/com.trueskate.training     # stop + unload
```

Changing the curriculum or warm start means re-running `install_launchd.sh`
(it rewrites and reloads the plist). For frequent experimentation, prefer the
manual `run_training.py` in `tmux`.

### 4.2 `com.trueskate.remotexpc-tunnel` (Mode B prerequisite, root)

The appium-xcuitest driver only auto-deletes on-device XCTest recording
attachments when its remotexpc tunnel registry is reachable, which requires a
**root** daemon. Plist: `scripts/ops/com.trueskate.remotexpc-tunnel.plist`.
Install (run as the user; will prompt for a password):

```bash
sudo cp /Users/training-server/trueskate-ai/scripts/ops/com.trueskate.remotexpc-tunnel.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.trueskate.remotexpc-tunnel.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.trueskate.remotexpc-tunnel.plist
```

Verify / stop:

```bash
sudo launchctl print system/com.trueskate.remotexpc-tunnel | head
tail -f /Users/training-server/trueskate-ai/logs/remotexpc_tunnel.log
sudo launchctl bootout system/com.trueskate.remotexpc-tunnel
```

If attachments have already backed up, clear them with
`scripts/recover_remotexpc_attachments.sh` (`--dry-run` / `--delete`). See
`CLAUDE.md` and memory `xctest-recording-attachments-accumulate`.

### 4.3 `com.trueskate.watchdog.<LABEL>` (Mode B)

`scripts/collection_watchdog.sh [DEVICE_TAG] [WDA_PORT] [LABEL]` alerts via ntfy
when a device's collector stops producing new `segment_*.json` for
`STALL_SECONDS` (default 600 s; healthy cadence ≈ 75 s). It catches both a dead
collector and a stuck-but-alive one (process up, `rec.start()` failing). The
script documents its own launchd label form (`com.trueskate.watchdog.<LABEL>`)
and `bootout` command; the plist that wires it is not committed.

---

## 5. Verifying what is actually loaded

This document cannot assert runtime state. Query the rig:

```bash
# user LaunchAgents for this rig
launchctl list | grep trueskate

# the root tunnel daemon
sudo launchctl print system/com.trueskate.remotexpc-tunnel | head

# what a specific agent is doing
launchctl print gui/$(id -u)/com.trueskate.training | sed -n '1,40p'
```

A label appearing here is the only proof it is loaded. Reconcile what you find
against the §4 inventory; an installed-but-uncommitted job is a gap to close.

---

## 6. Tailscale (remote access)

The configuration choices below are the ones that prevent "it died and I
couldn't get back in" on a headless, rarely-logged-in box:

1. **Disable key expiry** on the host (admin console → Machines → host → ⋯ →
   Disable key expiry). The top cause of a headless node silently dropping off
   the tailnet after months with nobody present to re-auth.
2. **Run `tailscaled` as a system service**, not the GUI App Store app (which
   only runs while logged into the GUI). Confirm with `sudo tailscale status`
   after a reboot with nobody logged in.
3. **Enable Tailscale SSH:** `sudo tailscale up --ssh`; connect with
   `tailscale ssh training-server@<host>`. No port-forwarding, no exposed ports.
4. **MagicDNS on** so the host is addressable by name, not a DHCP-roulette IP.
5. **Publish the dashboard** over the tailnet only:
   `tailscale serve --bg 8200` (undo with `tailscale serve --https=443 off`).
6. **Auto-login** on the host so it returns to a working GUI session after a
   power blip — required for LaunchAgents and for WDA/USB.

Post-setup sanity check from the operator Mac:

```bash
tailscale ssh training-server@<host> 'tailscale status; uptime'
```

---

## 7. Keeping the Mac awake

The supervisor wraps its run in `caffeinate -dimsu`, so active runs will not let
the Mac sleep. Belt-and-braces for a headless box:

```bash
sudo pmset -a sleep 0 disablesleep 1   # never sleep
sudo pmset -a autorestart 1            # auto power-on after a power loss
```

If running clamshell (lid closed), the Mac must be on AC power with an external
display or power source attached.

---

## 8. Hardware & per-phone iOS setup

These prevent the most common physical and overnight failures:

- **Powered USB hub (do this first).** Three to four iPhones on bus power will
  brown out and randomly disconnect — which is indistinguishable from flaky
  software. A quality *powered* hub is the single biggest uptime win. The XS has
  a degraded port; give it the most reliable cable/port. The supervisor waits
  for a vanished device to reappear and restarts only that device's stack.
- **Guided Access** per phone (Settings → Accessibility → Guided Access; triple-
  click to lock True Skate to the foreground). Hardware-enforces the scene guard
  so nothing (alarms, notifications) can steal focus.
- **Auto-Lock = Never** (Settings → Display & Brightness).
- **Disable** automatic iOS updates, notifications (Do Not Disturb / Focus),
  Low Power Mode, and all alarms.
- **Display Zoom** per the §2 table — `iPhone_11` ON; `iPhone_XR` / `iPhone_XR2`
  OFF. A mismatch aborts `DeviceWorker.connect()`.
- **Thermals.** Continuous GPU load plus charging for hours throttles the device
  and can drift trick timing. Maintain airflow; consider a charge-limit or
  duty-cycle for very long runs.

---

## 9. Monitoring

- **Mode A dashboard:** `status_server.py` on `:8200` (published via
  `tailscale serve`) — per-device alive/dead, evals & lands, evals/hr, last
  trick, best, and a **STALE** badge if `logs/status.json` stops updating
  (training stalled).
- **Mode B liveness:** `scripts/collection_watchdog.sh` (§4.3) ntfy alerts on a
  stalled collector. A separate live-stream dashboard exists in
  `scripts/train_dashboard.py` (`iPhone_XR` / `iPhone_XR2` streams).
- **Push (ntfy):** topic `NTFY_TOPIC` in `.env`. Mode A pushes run start/finish,
  a device going down, N consecutive zero-land generations, all-devices-dead
  abort, and supervisor stop.
- **Logs:** `logs/supervisor.{out,err}.log`; per-run JSONL under
  `logs/runs/<run>/`; `logs/remotexpc_tunnel.log` for the root daemon.

---

## 10. Scene-classifier guard (Mode A, optional)

The scene guard ships disabled. To enable automatic park recovery:

1. Collect negatives (home screen, menus, other apps) into `data/scene/negatives/`
   (see `experiments/scene_classifier_journal.md`).
2. `python scripts/data/build_scene_dataset.py --negatives-dir data/scene/negatives`
3. `python scripts/train/train_scene_classifier.py --manifest data/scene/manifest.json`
4. Add `SCENE_GUARD_MODEL=notebooks/models/scene_classifier.pth` to `.env`.

`DeviceWorker` then checks each eval and recovers into the park automatically.

---

## 11. Troubleshooting

| Symptom | Cause / action |
|---|---|
| One phone keeps dropping (commonly XS) | Supervisor restarts only that device's stack with decaying backoff and waits for USB to reappear; an ntfy fires. Reseat the cable / use a better powered-hub port. |
| Dashboard shows **STALE** | Mode A training stalled or died; supervisor relaunches it. Check `logs/supervisor.err.log`. |
| Recording wedges: `ScreenRecordingError Code=7 "Failed to write file"` | XCTest attachments accumulated — the remotexpc tunnel daemon (§4.2) is down or was never installed. Bring it up, then run `scripts/recover_remotexpc_attachments.sh --delete`. |
| ntfy "no new segment in N min" | A Mode B collector stalled (dead or `rec.start()` failing). Check the device's collector; a dropped WDA session needs one reconnect. |
| ntfy "no landed trick in N generations" (Mode A) | Likely left the skatepark or a bad warm start. The scene guard (§10) auto-recovers once trained. |
| All devices dead > 5 min | Mode A aborts cleanly with an urgent ntfy; supervisor relaunches the run. |
| Can't SSH in | Check Tailscale key expiry (§6.1) and that `tailscaled` runs without login (§6.2). |
| Screen-size mismatch on connect | Check Display Zoom against §2 (`iPhone_11` ON; `iPhone_XR`/`iPhone_XR2` OFF). |
| Unsure what's running | Query launchd directly (§5); do not assume from this document. |
