# Bounded XR2 IPv4 recording diagnostic

`validate_ios_ipv4_recording.py` is an isolated hardware diagnostic, not a
collector or deployment launcher. It uses the rig's existing pairing record and
installed Appium dependencies. It does not rebuild WDA or change vendor modules.
Ordinary XR2 USB must be absent. Leave the powered camera adapter attached and
Pico disconnected until the final phase. Unlock XR2 in gameplay first.

**Live status (2026-10-07):** paired IPv4 and client-certificate TLS work when
XR2 is awake. XR2 closes the subsequent CDTunnel handshake at both tested MTUs.
No native TUN, WDA launch or recording has passed on this route. The Node TLS
forwarder is a diagnostic alternative; its packet interface has offline tests
but no live validation. The classic CoreDeviceProxy path is described as USB
in the [reference toolkit](https://github.com/jkcoxson/idevice/blob/master/idevice/src/services/core_device_proxy.rs);
network tunneling uses a different paired protocol. Do not infer wireless
recording support from TLS success alone.

Stage the Python coordinator, `ios_ipv4_tunnel.mjs` and
`ios_ipv4_node_tunnel.mjs` together under a fresh
`/Users/training-server/trueskate-ai-runtime/tmp/` directory. Keep the live checkout
unchanged. Use its existing `.venv`, `.env`, and source for gesture/calibration
contracts. Run the same command first with `--prepare` and a separate output
folder; this authenticates paired IPv4 and checks collection-off, USB absence,
root daemon health and an empty default tunnel registry without creating a TUN.

```sh
/Users/training-server/trueskate-ai/.venv/bin/python /ABS/STAGED/validate_ios_ipv4_recording.py \
  --repo /Users/training-server/trueskate-ai \
  --env-file /Users/training-server/trueskate-ai/.env \
  --out-dir /Users/training-server/trueskate-ai-runtime/tmp/UNIQUE-DIAGNOSTIC \
  --admin-prompt --pico-hover
```

Use desktop Terminal for `--admin-prompt`; macOS administrator authentication
from an SSH session can fail without displaying a usable prompt. Credentials go
only into macOS authentication. Without that flag the helper uses `sudo -n` and
aborts if authentication is unavailable. The output folder must not exist.
`administrator-command.txt` contains the exact bounded root-helper command.
Desktop authentication has a five-minute window; an expired prompt is cancelled.
The coordinator holds a 15-minute idle-sleep assertion during the live test and
releases it on exit. This does not override closing the rig's lid. Keep the rig
awake and reachable throughout the diagnostic.

The helper opens paired lockdown over IPv4, verifies the XR2 identity, starts
`com.apple.internal.devicecompute.CoreDeviceProxy`, negotiates CDTunnel over
paired Node TLS, creates a native TUN using the installed packet interface, discovers
required developer services and serves a registry on **127.0.0.1:42315**. It does
not alter the ordinary root daemon or the shared registry file. The unprivileged
coordinator leases that file, redirects new Appium sessions temporarily, and
restores its original bytes on exit. A separate guardian handles coordinator
process death and verifies the lease owner token before touching the shared
registry or its lock. A missing heartbeat revokes the root helper within approximately
two seconds; its absolute lifetime cap is 15 minutes. An external registry-file
change is reported rather than overwritten. Never start another diagnostic
concurrently or reclaim its lock without investigating the owning process.

An already-ready WDA is reused. Otherwise the toolkit launches only the installed
runner, after checking no runner already exists, with `killExisting=false`.
Cleanup stops only a runner launched by this diagnostic. No production service
restart, signing renewal or installation is attempted.

Cases run sequentially and abort on the first failure:

1. Five seconds at requested 30 fps, no gestures.
2. Five seconds at requested 60 fps, no gestures; checks repeat lifecycle cleanup.
3. One minute at 30 fps with eight fixed linear drags, seed 20261006.
4. One minute at 60 fps using the identical drags.
5. Optional 45-second 60 fps Pico hover movie. At `waiting-pico`, confirm the
   operator is ready, then create `pico-ready` in the output folder. Only after
   `recording-started` for this case should the operator plug in the Pico.
   Save insertion timing separately. Late insertion is inconclusive; do not
   automatically retry or infer input acceptance from charging.

Every recording requires foreground/gameplay/geometry checks, a live diagnostic
registry entry, the ordinary root daemon, an idle recorder and a supported dry
run reporting zero UUID-shaped attachments. Both pre/post attachment listings
are preserved. There is exactly one recorder start attempt per case and one stop
attempt, including uncertain-start recovery. Recovery movies remain diagnostic.
A failed stop is never hammered with another stop or start.

Original MOV files, start/stop metadata, SHA-256, full-decode frame counts, packet
counts and actual PTS are preserved. Counts must agree; codec/geometry must be
H.264 828×1792. Minute tests require at least 59 seconds, at least 98% of requested
fps and no frame gap over 100 ms. No gate is waived to make the test pass.
Two 50 ms centre controls bracket each gesture minute. Existing WDA timing and
strict two-anchor calibration checks run separately using actual movie PTS.
No aligned training clips or dataset admissions are produced.

Read-only transport probes can run without administrator authentication:
`probe` authenticates lockdown; `tls-probe --handshake yes --mtu 1280` (or 16000)
checks the actual CDTunnel response; `native-tls-probe` isolates the installed
OpenSSL forwarder. All accept the same absolute `--modules-root`, `--udid` and
`--host` arguments. They read pairing secrets only into memory. A TLS-only pass
is separate from a successful CoreDevice handshake. Keep the phone awake; a
62078 timeout invalidates later transport comparisons.

`events.jsonl` provides progress; `report.json` records each case, failure and
cleanup result. Native recording lifecycle, cadence/calibration and Pico mouse
acceptance are separate conclusions. The hover firmware cannot establish click
semantics, a programmable gesture bridge or a 1 ms game-input path. Model 1's
32-frame / 2.3-second input contract is unchanged.

Offline verification from the isolated worktree:

```sh
/ABS/REPO/.venv/bin/python -m pytest tests/test_ios_ipv4_recording.py tests/test_wda_action_timing.py tests/test_scene_settle.py -q
/usr/local/bin/node --test tests/ios_ipv4_tunnel.test.mjs
```
