"""Lightweight ntfy.sh push notifications for unattended training runs.

Reads ``NTFY_TOPIC`` (and optional ``NTFY_SERVER``) from the environment / the
repo ``.env``. Used to surface device failures, all-dead aborts, throughput
collapse, and run start/stop on Asher's phone during 24h home collection.

Design rules:
  * Never raise. A notification problem must not crash or stall training.
  * Non-blocking by default (fires on a daemon thread).
  * No-op with a debug log when ``NTFY_TOPIC`` is unset, so dev runs stay quiet.
  * Never repeat. A notification whose title and text match one sent within the
    last hour (standalone numbers ignored, so "7.9GB" and "7.8GB" match but
    "XR1" and "XR2" do not) is suppressed across
    all processes; the next one that does go out reports how many were dropped.
  * Every send and suppression is appended to ``sent.log`` in the state dir
    (``NTFY_STATE_DIR``, default ``~/.trueskate-notify``).

Only the standard library is used (urllib) to avoid a new dependency.
"""
import fcntl
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from urllib import request

_TIMEOUT_S = 5.0
_REPO_ROOT = Path(__file__).resolve().parents[3]
_dotenv_loaded = False
DEDUPE_S = 3600.0
_STATE_RETENTION_S = 7 * 86400.0


def _ssl_context():
    """SSL context that works on python.org macOS builds (no system CA certs).

    Uses certifi's bundle when available, else the default context.
    """
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _ensure_env() -> None:
    """Best-effort load of the repo .env so NTFY_TOPIC is available.

    Callers usually load .env already (device connect, launchers). This is a
    safety net for entry points that don't, run at most once.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if os.environ.get("NTFY_TOPIC"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
    except Exception:
        pass


def is_configured() -> bool:
    _ensure_env()
    return bool(os.environ.get("NTFY_TOPIC"))


def _server() -> str:
    return os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")


def _state_dir() -> Path:
    return Path(os.environ.get("NTFY_STATE_DIR") or Path.home() / ".trueskate-notify")


def _repeat_key(title: str | None, message: str) -> str:
    """Same title and text with standalone numbers masked: a repeat, whatever it
    counts. Digits inside names (XR1 vs XR2) still distinguish alerts."""
    text = f"{title or ''}\n{re.sub(r'(?<![A-Za-z_])[0-9]+(?:[.][0-9]+)?', '#', message)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _admit(key: str, dedupe_s: float) -> tuple[bool, int]:
    """Decide under a cross-process lock whether a send may go out.

    Returns (send, repeats suppressed since the last send of this key). Any state
    error admits the send: a broken state file must not silence real alerts.
    """
    try:
        state_dir = _state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        with open(state_dir / "state.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = state_dir / "state.json"
            try:
                state = json.loads(path.read_text())
            except (OSError, ValueError):
                state = {}
            now = time.time()
            entry = state.get(key)
            if entry and now - entry["last_sent"] < dedupe_s:
                entry["suppressed"] += 1
                admitted, suppressed = False, entry["suppressed"]
            else:
                suppressed = entry["suppressed"] if entry else 0
                state[key] = {"last_sent": now, "suppressed": 0}
                admitted = True
            state = {k: v for k, v in state.items() if now - v["last_sent"] < _STATE_RETENTION_S}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(path)
            return admitted, suppressed
    except Exception as exc:
        logging.warning("ntfy dedupe state unavailable (%s); sending anyway", exc)
        return True, 0


def _log_event(event: str, title: str | None, message: str, priority, repeats: int) -> None:
    try:
        line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event,
                           "title": title, "priority": priority, "message": message,
                           "repeats": repeats})
        _state_dir().mkdir(parents=True, exist_ok=True)
        with open(_state_dir() / "sent.log", "a") as log:
            log.write(line + "\n")
    except Exception:
        pass


def notify(
    message: str,
    *,
    title: str | None = None,
    priority: str | int | None = None,
    tags: list[str] | str | None = None,
    actions: str | None = None,
    block: bool = False,
    dedupe_s: float = DEDUPE_S,
) -> None:
    """Push a notification to the configured ntfy topic.

    Args:
        message:  Body text.
        title:    Optional notification title.
        priority: ntfy priority (1-5 or 'min'..'urgent').
        tags:     ntfy tags/emoji (e.g. ['warning'] renders ⚠️).
        actions:  Raw ntfy ``Actions`` header value (e.g. from
                  ``confirm_button_action``) to add a tappable button.
        block:    Send synchronously (use for final shutdown messages where the
                  process may exit before a daemon thread flushes).
        dedupe_s: Suppress repeats of this title/text within this many seconds
                  (0 disables, e.g. for a deliberate re-prompt).
    """
    _ensure_env()
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        logging.debug("ntfy: NTFY_TOPIC unset; dropping notification: %s", message)
        return

    admitted, repeats = _admit(_repeat_key(title, message), dedupe_s) if dedupe_s > 0 else (True, 0)
    if not admitted:
        _log_event("suppressed", title, message, priority, repeats)
        return
    if repeats:
        message = f"{message}\n(+{repeats} identical alert{'s' if repeats > 1 else ''} suppressed)"
    _log_event("sent", title, message, priority, repeats)

    # HTTP headers are latin-1; drop any non-encodable chars (e.g. an emoji in a
    # title/label) so it can never break the whole notification. The body is utf-8.
    def _hdr(value: str) -> str:
        return value.encode("latin-1", "ignore").decode("latin-1")

    def _send() -> None:
        try:
            req = request.Request(
                f"{_server()}/{topic}", data=message.encode("utf-8")
            )
            if title:
                req.add_header("Title", _hdr(title))
            if priority is not None:
                req.add_header("Priority", str(priority))
            if tags:
                req.add_header(
                    "Tags", _hdr(tags if isinstance(tags, str) else ",".join(tags))
                )
            if actions:
                req.add_header("Actions", _hdr(actions))
            request.urlopen(req, timeout=_TIMEOUT_S, context=_ssl_context())
        except Exception as exc:  # never let a notification break training
            logging.warning("ntfy notification failed: %s", exc)

    if block:
        _send()
    else:
        threading.Thread(target=_send, daemon=True).start()


def _latch_path(latch: str) -> Path:
    return _state_dir() / "latches" / re.sub(r"[^A-Za-z0-9_.-]", "_", latch)


def notify_once(latch: str, message: str, **kwargs) -> bool:
    """Notify only when ``latch`` is not already set, then set it.

    For conditions that persist across processes (a collector restarted per
    segment): one alert per incident until :func:`clear_latch` marks it resolved.
    Returns True if this call raised the alert.
    """
    path = _latch_path(latch)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    except FileExistsError:
        return False
    except OSError as exc:
        logging.warning("ntfy latch %s unavailable (%s); sending anyway", latch, exc)
    notify(message, **kwargs)
    return True


def clear_latch(latch: str) -> bool:
    """Mark a latched condition resolved. Returns True if it was set."""
    try:
        _latch_path(latch).unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Tap-to-confirm: a notification carries an http action button that POSTs a
# token to a derived control topic; the rig polls that topic for the token.
# (Validated end-to-end on the iOS ntfy app.) The user can alternatively just
# send a title-less "done"/"switched"/"next" message to the main topic.
# ---------------------------------------------------------------------------

_CONFIRM_WORDS = ("done", "switched", "next", "ok", "go")


def control_topic() -> str | None:
    """The derived control topic the confirm button POSTs to (or None)."""
    _ensure_env()
    topic = os.environ.get("NTFY_TOPIC")
    return f"{topic}_ctl" if topic else None


def confirm_button_action(label: str, token: str = "SWITCHED") -> str | None:
    """An ``Actions`` header value: an http button that POSTs ``token`` to the
    control topic and clears the notification. None if ntfy is unconfigured."""
    ctl = control_topic()
    if not ctl:
        return None
    return f"http, {label}, {_server()}/{ctl}, method=POST, body={token}, clear=true"


def _poll_topic_messages(topic: str, since_ts: float) -> list[dict]:
    """One-shot poll of a topic's cached messages since ``since_ts`` (JSON API)."""
    import json

    url = f"{_server()}/{topic}/json?poll=1&since={int(since_ts)}"
    try:
        data = request.urlopen(url, timeout=_TIMEOUT_S, context=_ssl_context()).read().decode()
    except Exception as exc:
        logging.debug("ntfy poll failed: %s", exc)
        return []
    out = []
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") == "message":
            out.append(ev)
    return out


def poll_confirmation(token: str = "SWITCHED", *, since_ts: float) -> bool:
    """True if the user has confirmed since ``since_ts`` — either the button's
    ``token`` on the control topic, or a title-less confirm word typed to the
    main topic. Non-blocking single poll; returns False (never raises) on error."""
    _ensure_env()
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return False
    ctl = control_topic()
    if ctl and any(m.get("message") == token for m in _poll_topic_messages(ctl, since_ts)):
        return True
    # Fallback: a plain message (no title) the user typed to the main topic.
    for m in _poll_topic_messages(topic, since_ts):
        if not m.get("title") and str(m.get("message", "")).strip().lower() in _CONFIRM_WORDS:
            return True
    return False
