# Private XR menu control

This standalone service mirrors XR1 and XR2 and sends a single touch gesture
on release. Tap, hold (up to five seconds), or draw a curved drag with the laptop
trackpad. It does not record, calibrate, launch collection, or share dashboard
port 8400. Full normalized screen coordinates map through `scale_to_device()`
to 414×896 logical points; gameplay sampling exclusions do not apply to menus.

## Prerequisites

- Existing healthy WDA/Appium and MJPEG forwarding on training-server:
  XR1 8100/4723/9100; XR2 8103/4726/9103. Nothing in this tool starts them.
- Collection and wrappers stopped. Any recognized collector process reserves
  both XRs conservatively, including between segments. No other Appium sessions.
- `.env` contains `IPHONE_XR_UDID` and `IPHONE_XR2_UDID`; identifiers stay on
  the server. Use the existing `.venv`; no new dependencies are required.
- Appium 3 must already expose `GET /appium/sessions` with
  `--allow-insecure='*:session_discovery'`. Older Appium 2 supports `/sessions`.
  Discovery denied/unavailable is a hard block, never interpreted as idle.
  This flag is a prerequisite for a separately scheduled Appium maintenance
  change, not permission for this launcher to restart Appium or healthy WDA.
  Keep Appium private. See the [Appium API](https://appium.io/docs/en/latest/reference/api/appium/).

## Launch

Use a separate committed checkout on the rig. Do not change its stable release
symlink. From the laptop:

```bash
tailscale ssh training-server@training-server
```

In that remote terminal, substitute the staged checkout's absolute path:

```bash
/Users/training-server/trueskate-ai/.venv/bin/python /absolute/staged-checkout/scripts/control/serve.py --env-file /Users/training-server/trueskate-ai-runtime/.env
```

Leave it in the foreground. It binds **only 127.0.0.1:8401**, prints a private
URL with a random per-launch token, and starts one shared MJPEG reader per XR.
Do not save the URL in shared logs. On the laptop, run from this checkout:

```bash
.venv/bin/python scripts/control/open.py
```

Paste only the part after `#` into its hidden prompt. The launcher uses OpenSSH
local forwarding through `tailscale nc`, because `tailscale ssh` does not expose
forwarding flags. SSH host-key verification remains enabled. It opens
`http://127.0.0.1:8401/` with the token in a fragment, then removes the fragment
from browser history. The token stays in tab-scoped session storage for reloads.
Keep this terminal open. All API reads require the token;
mutations additionally require the exact origin and a JSON body. No CORS access
or public hosting is provided.

Click **Connect** separately for each device, then **Open True Skate** if needed.
Connect attaches to running WDA with app launch, reset, force launch and app
termination disabled ([WDA attach guide](https://appium.github.io/appium-xcuitest-driver/latest/guides/attach-to-running-wda/)).
Only Open True Skate activates the game. Enlarge is optional. Drag paths appear
before release. Escape, blur, resizing, pointer cancellation, leaving the screen,
or exceeding five seconds discards a captured path. Multitouch is unsupported.

One upstream reader feeds authenticated latest-frame responses to all browser
clients; `/api/XR1/stream` (and XR2) also provides authenticated multipart
MJPEG with a 60-second client lifetime. The UI refreshes at up to roughly 12 fps. Every displayed image carries
a server sequence; a gesture must reference a frame received within two seconds.
Browser decoding/network age, stale status, and hidden tabs also disable input.
A frozen image cannot be made fresh just by fetching it again.

Commands serialize per device; a connection epoch and increasing submission
number reject duplicates and old tabs. Failed commands are never replayed.
Session ownership is checked before each command. This is cooperative rig
operation: do not start an external collector/client while controlling a phone;
Appium has no cross-client atomic reservation API. The service does not take
sessions from other clients or delete them.

## Shutdown and recovery

Click Disconnect for both devices, close the browser, then Ctrl-C in the laptop
forwarder and remote service terminals. Normal service shutdown closes only its
own idle sessions and stops video readers. It leaves True Skate and WDA running.
A browser/tunnel close alone does not disconnect immediately; idle Appium
sessions expire after 300 seconds. Read-only status probes detect expiry without
renewing the session; a command racing expiry is rejected; Connect creates a new epoch.

- **Stale video:** inspect existing forwarding/WDA, foreground and portrait
  orientation. The control service retries video reads only, never gestures.
- **Busy/collector:** stop using the UI until the owner finishes. Do not delete
  its session or restart services to gain control.
- **Session discovery unavailable:** use the prerequisite maintenance change
  above. Do not bypass this check using WDA's status or stale heartbeat data.
- **Command outcome unknown:** control is quarantined for that service launch.
  Inspect Appium/WDA and the device; a timed-out gesture may have executed.
  Resolve only this service's orphan session after confirming command completion,
  or wait for its idle timeout. Restart the control service only after resolving
  that uncertainty. It intentionally sends no touch release/replay on timeout.
- **Wrong geometry:** turn off Display Zoom and use portrait XR geometry;
  connection rejects other dimensions without applying offsets.
- **Tailscale “Failed to load preferences”:** in the agent sandbox this was an
  access restriction; the installed CLI worked outside that sandbox. Ordinary
  terminal `tailscale status` and `tailscale ssh` should succeed before launch.

Rollback: stop this standalone service and forwarding terminal. The rig release,
collector configuration, corpus, dashboard and WDA are unchanged. Remove only the
staged checkout if no longer needed; no migration is necessary.

## Verification and delivery evidence (2026-09-29)

```bash
python -m pytest -q tests/test_remote_control.py
node --test tests/remote_pointer.test.mjs
python scripts/ops/check_archive.py --base origin/main
```

Synthetic tests cover normalized/letterboxed coordinates, curved paths, taps,
holds, cancellation, validation limits, routing, busy ownership, timeouts,
duplicates, stale frames, reconnects and HTTP protections. No corpus or holdout is
used. The rig was reachable through Tailscale; WDA responded on both ports and
no collector process was found. Appium 3.3.1 denied session discovery on both
ports. The initial on-device validation was blocked by that prerequisite; no
service or gesture was deployed during that first verification pass. See the
subsequent authorized launch below.

For a complete navigation check on each XR: Connect,
Open True Skate, open the park menu, drag its list once, select a named installed
park, visually confirm its loaded scene, and Disconnect. Record the actual park
and results separately; never claim this checklist has run from offline tests.

Final verification: 367 Python tests passed, 2 skipped; all 3 Node pointer tests
passed; JavaScript syntax and archive integrity checks passed. Automated browser
visual verification was attempted against a synthetic local server but Chrome
computer access was not approved.

### Authorized launch (2026-09-29)

The operator requested launch and explicitly approved restarting only the two
Appium processes to enable touch control. Read-only checks confirmed all logged
sessions had ended and both recorders were idle. Appium was restarted with
`--address 127.0.0.1` and
`--allow-insecure=xcuitest:xctest_screen_record,*:session_discovery`; both discovery
endpoints then reported no sessions. Existing WDA remained healthy and was not
restarted. Collection stayed off.

Source revision `4380162` was staged at
`/Users/training-server/trueskate-ai-runtime/tmp/xr-control-4380162/source`.
The adjacent directory holds `control.pid`, private `control.log`,
`appium-XR1.pid`, `appium-XR2.pid`, their logs and `appium-rollback.json` with the
original Appium arguments/log paths. These runtime files are not source backups.
The control service and laptop SSH forward were started, the browser was opened,
and both panels connected. Final status showed each device available, not busy,
not quarantined, and receiving fresh video (40 ms and 2 ms frame age).

The operator reported the UI looked good and requested merge. No automated live
gesture or named-park navigation checklist result is claimed. The rig stable
release symlink was unchanged; the running UI remains on its staged source until
an explicit control-service restart. Stopping this UI does not undo the approved
Appium configuration; its saved original arguments are available for a separate
idle-time rollback if needed.
