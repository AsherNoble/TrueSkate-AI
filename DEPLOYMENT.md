# 24h Training Rig — Deployment & Operations

How to run TrueSkate-AI as an unattended, self-healing, remotely-monitored
data-collection rig on the **Intel MacBook (`training-server`)** with the
iPhones plugged in 24/7, controlled and monitored from your **personal Mac**
over Tailscale.

```
 personal Mac  ──tailscale ssh──▶  training-server (Intel MacBook)
   (you code,                         ├─ run_training.py (supervisor, under caffeinate)
    you watch)  ◀──tailscale serve──  │   ├─ launch_services.py  → Appium+WDA+iproxy ×N (self-heals per device)
                                       │   ├─ status_server.py    → dashboard on :8200
                                       │   └─ train_cmaes.py      → CMA-ES (restarts w/ bumped seed)
                                       └─ iPhone XR + iPhone XS  (collection roster)
```

The iPhone 11 is tagged **personal** — never grabbed by the default roster, so
you can test on it on your own Mac while on the road.

---

## 1. Quick start (once home)

On `training-server`, from the repo root, inside `tmux` (so it survives SSH drop):

```bash
source .venv/bin/activate
# starts services + status server + training, under caffeinate, self-healing:
python scripts/run_training.py --curriculum curricula/360_flip.json \
    --initial-mean trick_libraries/<your-best-warmstart>.json
```

Then expose the dashboard once (persists across reboots):

```bash
tailscale serve --bg 8200
```

Open `https://training-server.<your-tailnet>.ts.net/` from anywhere. Push
alerts arrive via your existing `NTFY_TOPIC`.

To run it truly hands-off (auto-start on login/reboot), install the LaunchAgent
instead of the manual command — see §5.

Default roster is XR + XS. Override with `--devices iPhone_XR,iPhone_XS` or, for
a single-phone test, `--personal`.

---

## 2. Tailscale — make it bulletproof

These are the things that actually cause "it died and I couldn't get in":

1. **Disable key expiry on `training-server`.** Tailscale admin console →
   Machines → `training-server` → ⋯ → **Disable key expiry**. This is the #1
   cause of a headless node silently dropping off the tailnet after ~6 months
   with nobody logged in to re-auth.
2. **Run tailscaled as a system service, not just the GUI app.** The Mac App
   Store app only runs while you're logged into the GUI. Use the standalone
   `tailscaled` (or `sudo tailscale up`) so it's up before/without login.
   Confirm: `sudo tailscale status` after a reboot with nobody logged in.
3. **Enable Tailscale SSH:** `sudo tailscale up --ssh`. Then from your Mac:
   `tailscale ssh <user>@training-server` — no port-forwarding, no exposed
   ports. (This matches the note already in `.env`.)
4. **MagicDNS on** (admin console → DNS) so you use `training-server`, not a
   DHCP-roulette IP.
5. **`tailscale serve --bg 8200`** publishes the dashboard over HTTPS on your
   tailnet only. Undo with `tailscale serve --https=443 off`.
6. **Auto-login** on `training-server` (System Settings → Users & Groups →
   Automatically log in) so it returns to a working state after a power blip —
   required for the LaunchAgent and for WDA/USB (which need the GUI session).

Sanity check from your Mac after setup:
```bash
tailscale ssh <user>@training-server 'tailscale status; uptime'
```

---

## 3. Keep the Mac awake

The supervisor already wraps everything in `caffeinate -dimsu` while it runs, so
manual runs won't let the Mac sleep. Belt-and-braces for a headless box:

```bash
sudo pmset -a sleep 0 disablesleep 1   # never sleep
sudo pmset -a autorestart 1            # auto power-on after a power loss
```

If running clamshell (lid closed), it must be on AC power and external power.

---

## 4. Hardware & per-phone iOS setup

These prevent the most common physical/overnight failures:

- **Powered USB hub (do this first).** 3–4 iPhones on bus power will brown out
  and randomly disconnect — which looks exactly like flaky software. A quality
  *powered* hub is likely the single biggest uptime win. The old XS in
  particular has a "munted" port — give it the most reliable cable/port, and
  the supervisor will wait for it to reappear and restart just that device.
- **Guided Access** per phone (Settings → Accessibility → Guided Access; triple-
  click to lock True Skate to the foreground). This *hardware-enforces* the
  scene guard — nothing (Clock alarms, notifications) can steal focus.
- **Auto-Lock = Never** (Settings → Display & Brightness).
- **Disable**: automatic iOS updates, notifications (Do Not Disturb / Focus),
  Low Power Mode, and any alarms.
- **iPhone 11 Display Zoom must stay ON** (logical 375×812) — the code expects
  it; `DeviceWorker.connect()` aborts on a screen-size mismatch with a hint.
- **Thermals:** continuous GPU + charging for hours throttles and can drift
  trick timing. Keep airflow / a small fan; consider a charge-limit or
  duty-cycle for very long runs.

---

## 5. Running as a service (auto-start on login/reboot)

```bash
deploy/install_launchd.sh curricula/360_flip.json
tail -f logs/supervisor.out.log
```

This installs `~/Library/LaunchAgents/com.trueskate.training.plist` (RunAtLoad +
KeepAlive). With auto-login on, the rig comes back by itself after a reboot.

Manage it:
```bash
launchctl kickstart -k gui/$(id -u)/com.trueskate.training   # restart now
launchctl bootout    gui/$(id -u)/com.trueskate.training      # stop + unload
```

Note: edit the curriculum/warm-start by re-running `install_launchd.sh` with new
args (it rewrites and reloads the plist). For frequent experimentation, prefer
the manual `run_training.py` in tmux.

---

## 6. Monitoring

- **Dashboard:** `https://training-server.<tailnet>.ts.net/` — per-device
  alive/dead, evals & lands, evals/hr, last trick, best, and a **STALE** badge
  if the heartbeat (`logs/status.json`) stops updating (= training stalled).
- **Push (ntfy):** run start/finish, a device going down, **6 consecutive
  generations with zero landed tricks** (likely left the skatepark), all-devices-
  dead abort, and supervisor stop. Topic = `NTFY_TOPIC` in `.env`.
- **Logs:** `logs/supervisor.{out,err}.log`; per-run JSONL under
  `logs/runs/<run>/` (now includes `in_skatepark` per eval once the guard is on).

---

## 7. Testing on the road (personal iPhone 11)

On your personal Mac, with the iPhone 11 plugged in:

```bash
python scripts/launch_services.py --personal
python scripts/train/train_cmaes.py --personal --curriculum curricula/360_flip.json --max-evals 48
```

`--personal` selects only the `personal`-role device, so this never touches the
home roster config.

---

## 8. Finish the scene classifier (at home)

The guard ships disabled. To turn it on:

1. Collect negatives (home screen, menus, other apps) into
   `data/scene/negatives/` — see `experiments/scene_classifier_journal.md`.
2. `python scripts/data/build_scene_dataset.py --negatives-dir data/scene/negatives`
3. `python scripts/train/train_scene_classifier.py --manifest data/scene/manifest.json`
4. Add `SCENE_GUARD_MODEL=notebooks/models/scene_classifier.pth` to `.env`.

`DeviceWorker` then checks each eval and recovers into the park automatically.

---

## 9. Troubleshooting

| Symptom | What happens / what to do |
|---|---|
| One phone keeps dropping (XS) | Supervisor restarts only that device's stack with decaying backoff and waits for USB to reappear; you get an ntfy. Reseat cable / use a better hub port. |
| Dashboard shows **STALE** | Training process stalled or died; supervisor relaunches it. Check `logs/supervisor.err.log`. |
| ntfy "no landed trick in N generations" | Likely left the skatepark or bad params/warm-start. Once the scene guard is trained it auto-recovers. |
| All devices dead >5 min | Run aborts cleanly with an urgent ntfy; supervisor relaunches the whole training run. |
| Can't SSH in | Check Tailscale key expiry (§2.1) and that tailscaled runs without login (§2.2). |
| Screen-size mismatch on connect | Check Display Zoom (iPhone 11 = on, XS/XR per `DEVICES`). |
