"""
Fraud alerts waiting for the customer to call back.

Business-initiated WhatsApp calls are blocked for Nigerian business numbers, so
the flow inverts: the bank sends a message, the customer taps to call, and the
agent answers. That means when a call arrives we have to work out *why* this
person is ringing — which is what this module holds.

An alert is keyed by phone number and expires. A fraud briefing is only useful
while the risk is live; if someone rings back two days later they should reach a
human with fresh context, not an agent reading a stale script.

Alerts are mirrored to a small JSON file. Purely in-memory looked fine until
`uvicorn --reload` restarted on a file save between the alert going out and the
customer ringing back — the briefing vanished and a real caller reached the
"no alert pending" path for no reason they could see. The file makes a restart
survivable. A real deployment would use Redis or Mongo so it is also shared
across workers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("fraud_agent.pending_alerts")

# How long a raised alert stays answerable by the agent.
DEFAULT_TTL_MINUTES = 60

# Survives uvicorn --reload. Deliberately holds no card or account data — just
# which signal a caller is ringing about, so the agent can look the rest up.
STATE_FILE = Path(__file__).with_name(".pending_alerts.json")


def normalise_phone(number: str) -> str:
    """
    Reduce a number to its last 10 digits for matching.

    Meta hands back numbers in several shapes (+2348020812523, 2348020812523,
    08020812523) and they must all resolve to the same alert.
    """
    return "".join(c for c in number if c.isdigit())[-10:]


@dataclass
class PendingAlert:
    signal_id: str
    customer_id: str
    phone_number: str
    created_at: datetime
    expires_at: datetime
    notified_at: datetime | None = None
    answered_at: datetime | None = None

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def age_minutes(self) -> int:
        return int((datetime.now(timezone.utc) - self.created_at).total_seconds() // 60)

    def as_dict(self) -> dict:
        return {
            "signalId": self.signal_id,
            "customerId": self.customer_id,
            "phoneNumber": f"***{self.phone_number[-4:]}",
            "createdAt": self.created_at.isoformat(),
            "expiresAt": self.expires_at.isoformat(),
            "notified": self.notified_at is not None,
            "answered": self.answered_at is not None,
            "expired": self.expired,
        }


class PendingAlertStore:
    def __init__(
        self, ttl_minutes: int = DEFAULT_TTL_MINUTES, state_file: Path | None = None
    ) -> None:
        self._alerts: dict[str, PendingAlert] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()
        self._state_file = state_file
        if self._state_file:
            self._load()

    # ------------------------------------------------------------------
    # Persistence — so a uvicorn reload does not lose a live briefing
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._state_file or not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt state file must never stop the service starting; the
            # cost is a caller reaching a human, which is the safe direction.
            logger.warning("Could not read %s (%s) — starting empty", self._state_file, exc)
            return

        restored = 0
        for key, row in raw.items():
            try:
                alert = PendingAlert(
                    signal_id=row["signal_id"],
                    customer_id=row["customer_id"],
                    phone_number=row["phone_number"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=datetime.fromisoformat(row["expires_at"]),
                )
            except (KeyError, ValueError):
                continue
            if not alert.expired:
                self._alerts[key] = alert
                restored += 1
        if restored:
            logger.info("Restored %d pending alert(s) from disk", restored)

    def _save(self) -> None:
        if not self._state_file:
            return
        payload = {
            key: {
                "signal_id": a.signal_id,
                "customer_id": a.customer_id,
                "phone_number": a.phone_number,
                "created_at": a.created_at.isoformat(),
                "expires_at": a.expires_at.isoformat(),
            }
            for key, a in self._alerts.items()
            if not a.expired
        }
        try:
            # Write to a temp file and replace, so a crash mid-write cannot
            # leave a half-written file that fails to parse on restart.
            fd, tmp = tempfile.mkstemp(dir=str(self._state_file.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, self._state_file)
        except OSError as exc:
            logger.warning("Could not persist pending alerts: %s", exc)

    def raise_alert(self, *, signal_id: str, customer_id: str, phone_number: str) -> PendingAlert:
        """
        Record that this customer has a fraud briefing waiting.

        A newer signal replaces an older one for the same number — if two things
        fire in quick succession, the agent should brief on the latest.
        """
        now = datetime.now(timezone.utc)
        alert = PendingAlert(
            signal_id=signal_id,
            customer_id=customer_id,
            phone_number=phone_number,
            created_at=now,
            expires_at=now + self._ttl,
        )
        key = normalise_phone(phone_number)
        with self._lock:
            if key in self._alerts:
                logger.info("Replacing existing alert for ***%s", key[-4:])
            self._alerts[key] = alert
            self._save()
        logger.info(
            "Alert raised: %s for customer %s, answerable until %s",
            signal_id,
            customer_id,
            alert.expires_at.isoformat(timespec="seconds"),
        )
        return alert

    def get(self, phone_number: str) -> PendingAlert | None:
        """Find the live alert for a caller, if any. Expired ones do not count."""
        key = normalise_phone(phone_number)
        with self._lock:
            alert = self._alerts.get(key)
        if alert is None:
            logger.info("No pending alert for caller ***%s", key[-4:])
            return None
        if alert.expired:
            logger.info(
                "Alert %s for ***%s expired %d minutes ago — not serving it",
                alert.signal_id,
                key[-4:],
                alert.age_minutes,
            )
            return None
        return alert

    def mark_notified(self, phone_number: str) -> None:
        alert = self.get(phone_number)
        if alert:
            alert.notified_at = datetime.now(timezone.utc)

    def mark_answered(self, phone_number: str) -> None:
        alert = self.get(phone_number)
        if alert:
            alert.answered_at = datetime.now(timezone.utc)
            logger.info(
                "Alert %s answered after %d minutes", alert.signal_id, alert.age_minutes
            )

    def clear(self, phone_number: str) -> None:
        with self._lock:
            self._alerts.pop(normalise_phone(phone_number), None)
            self._save()

    def all(self) -> list[PendingAlert]:
        with self._lock:
            return list(self._alerts.values())


pending = PendingAlertStore(state_file=STATE_FILE)
