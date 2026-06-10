"""Lightweight ntfy.sh push notifications for unattended training runs.

Reads ``NTFY_TOPIC`` (and optional ``NTFY_SERVER``) from the environment / the
repo ``.env``. Used to surface device failures, all-dead aborts, throughput
collapse, and run start/stop on Asher's phone during 24h home collection.

Design rules:
  * Never raise. A notification problem must not crash or stall training.
  * Non-blocking by default (fires on a daemon thread).
  * No-op with a debug log when ``NTFY_TOPIC`` is unset, so dev runs stay quiet.

Only the standard library is used (urllib) to avoid a new dependency.
"""
import logging
import os
import threading
from pathlib import Path
from urllib import request

_TIMEOUT_S = 5.0
_REPO_ROOT = Path(__file__).resolve().parents[3]
_dotenv_loaded = False


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


def notify(
    message: str,
    *,
    title: str | None = None,
    priority: str | int | None = None,
    tags: list[str] | str | None = None,
    block: bool = False,
) -> None:
    """Push a notification to the configured ntfy topic.

    Args:
        message:  Body text.
        title:    Optional notification title.
        priority: ntfy priority (1-5 or 'min'..'urgent').
        tags:     ntfy tags/emoji (e.g. ['warning'] renders ⚠️).
        block:    Send synchronously (use for final shutdown messages where the
                  process may exit before a daemon thread flushes).
    """
    _ensure_env()
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        logging.debug("ntfy: NTFY_TOPIC unset; dropping notification: %s", message)
        return

    def _send() -> None:
        try:
            req = request.Request(
                f"{_server()}/{topic}", data=message.encode("utf-8")
            )
            if title:
                req.add_header("Title", title)
            if priority is not None:
                req.add_header("Priority", str(priority))
            if tags:
                req.add_header(
                    "Tags", tags if isinstance(tags, str) else ",".join(tags)
                )
            request.urlopen(req, timeout=_TIMEOUT_S, context=_ssl_context())
        except Exception as exc:  # never let a notification break training
            logging.warning("ntfy notification failed: %s", exc)

    if block:
        _send()
    else:
        threading.Thread(target=_send, daemon=True).start()
