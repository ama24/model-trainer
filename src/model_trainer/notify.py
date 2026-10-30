"""Phone notifications through ntfy (https://ntfy.sh): free, no account.

Install the ntfy app, subscribe to a private, hard-to-guess topic name (e.g. `mtrain-k39xq7`), and put
the same name in Settings. Anyone who knows the topic can read the messages, so keep it secret.
"""

from __future__ import annotations

import logging
import urllib.request
from collections.abc import Mapping
from typing import Any

log = logging.getLogger("model_trainer.notify")

PRIORITY = {
    "finished": "default",
    "failed": "high",
    "warning": "default",
    "paused": "low",
    "test": "default",
}
TAGS = {
    "finished": "white_check_mark",
    "failed": "x",
    "warning": "warning",
    "paused": "pause_button",
    "test": "bell",
}


def send(
    settings: Mapping[str, Any], event: str, title: str, message: str, click: str | None = None
) -> bool:
    """Send one notification if enabled for this event. Never raises (training must go on)."""
    n = settings.get("notify") or {}
    if event != "test" and (not n.get("enabled") or event not in (n.get("events") or [])):
        return False
    topic = (n.get("topic") or "").strip()
    if not topic:
        return False
    url = f"{(n.get('server') or 'https://ntfy.sh').rstrip('/')}/{topic}"
    headers = {
        "Title": title.encode("ascii", "replace").decode(),
        "Priority": PRIORITY.get(event, "default"),
        "Tags": TAGS.get(event, "robot"),
    }
    if click:
        headers["Click"] = click
    req = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # noqa: BLE001 - network problems must not stop a run
        log.warning("notification failed: %s", exc)
        return False
