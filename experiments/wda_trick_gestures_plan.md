# WDA Custom Gesture Endpoint — Journal

## Goal
Precise inter-gesture timing for trick execution: `delay=0` means g1 starts exactly when g0 ends (zero gap, no phantom touch), `delay>0` means a timed gap, `delay<0` means overlap.

---

## What We Built

### Custom WDA endpoint: `POST /wda/perform_trick_gestures`

Added `FBTrickGestureCommands.{h,m}` to our WDA fork (`~/Projects/WebDriverAgent`, branch `feature/trueskate-trick-gestures`). WDA auto-discovers it via `FBClassesThatConformsToProtocol` — no changes to `FBWebServer.m` needed.

Route registered with `.withoutSession` → final path is `/wda/perform_trick_gestures` (no `/session/:id` prefix). Python hits WDA directly at port 8101, **bypassing Appium entirely** (Appium 3.x doesn't proxy unknown routes).

Request format:
```json
{
  "gestures": [
    {"waypoints": [{"x": 207, "y": 486, "duration_ms": 0}, {"x": 207, "y": 560, "duration_ms": 50}]}
  ]
}
```
`duration_ms` on the first waypoint is ignored (it's the contact origin). Per-segment durations are pre-computed by `_easing_to_segment_durations` on the Python side.

---

## What Failed

### Attempt 1: Two `XCPointerEventPath` objects in one `XCSynthesizedEventRecord`
Each path in a record executes as a **parallel simultaneous finger track** from t=0. With two paths, True Skate sees two simultaneous contacts — even if path 1's `pressDownAtOffset:` is set far in the future. Phantom touch inevitable.

### Attempt 2: Single `XCPointerEventPath` with hover `moveToPoint:` between gestures
After `liftUpAtOffset:`, calling `moveToPoint:atOffset:` on the same path generates a **real UITouch event** — a hover move is still a touch event from the app's perspective. This produced the phantom swipe between g0 endpoint and g1 start that persisted for weeks.

### Routing detour: Appium session URL
Initial Python code POSTed to `http://127.0.0.1:4724/session/{appium_id}/wda/perform_trick_gestures`. Appium 3.x doesn't proxy truly unknown routes — always 404. Switched to direct WDA at port 8101.

WDA doesn't expose `/sessions` — attempted `GET /sessions` to get WDA session ID for a session-prefixed URL; that route doesn't exist. Solution: `.withoutSession` registration makes session ID unnecessary.

---

## What Worked

Two **separate** HTTP calls — one per gesture — with Python-controlled delay:

```python
def _fire_gesture(points, duration, easing, wda_url):
    payload = {"gestures": [{"waypoints": _wda_waypoints(points, duration, easing)}]}
    resp = requests.post(f"{wda_url.rstrip('/')}/wda/perform_trick_gestures",
                         json=payload, timeout=15)
    resp.raise_for_status()

def _execute_two_gestures(g0_points, g1_points, g0_duration, g1_duration,
                          delay, easing0, easing1, wda_url):
    t0 = time.perf_counter()
    _fire_gesture(g0_points, g0_duration, easing0, wda_url)
    elapsed = time.perf_counter() - t0
    remaining = delay - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _fire_gesture(g1_points, g1_duration, easing1, wda_url)
```

**Why it works:** `FBXCTestDaemonsProxy synthesizeEventWithRecord:error:` blocks via `FBRunLoopSpinner` until the gesture fully completes on device — so `requests.post` returns at exactly the moment g0 finishes. Each gesture is a completely independent `XCSynthesizedEventRecord` with no WDA-level connection between them. Phantom swipe eliminated.

Timing accuracy for `delay=0`: ~10–20ms gap (HTTP round-trip overhead). Acceptable.

---

## Architecture Summary

- **Push:** still executed via Appium `ActionChains` (standard W3C, no phantom issue)
- **Scoop + flick:** two sequential calls to `/wda/perform_trick_gestures` on WDA port 8101, bypassing Appium
- **Custom route lives in:** `~/Projects/WebDriverAgent/WebDriverAgentLib/Commands/FBTrickGestureCommands.{h,m}`
- **Python integration:** `src/trueskate_ai/sim/execute_trick.py` — `_fire_gesture`, `_wda_waypoints`, `_execute_two_gestures`

---

## Risks / Notes

- `XCSynthesizedEventRecord` and `XCPointerEventPath` are private XCTest APIs. Stable in practice (WDA itself depends on them), but could break with a major Xcode/iOS update.
- The multi-gesture loop in `FBTrickGestureCommands.m` is still present but effectively unused (Python always sends single-gesture payloads). Can be simplified or repurposed later.
