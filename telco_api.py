"""
Simulated Nigerian telco API.

Same shape as bank_api.py, and for the same reason: the authority boundary is
enforced in code, not merely requested in a prompt.

The boundary is different here because the risks are different. The dangerous
operation in Nigerian telecoms is not spending money — it is **SIM swap**, which
is how bank accounts get emptied: take over the line, receive the OTPs, drain
the account. No AI agent performs a SIM swap on a phone call, ever. Nor does it
change registered details, port a number out, or lift a regulatory bar.

What it can do is everything that makes an IVR maze pointless: read balances,
diagnose why data is not working, re-push a bundle that was paid for and never
delivered, and hand over cleanly when the answer is not in its reach.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import telco_data

logger = logging.getLogger("telco_agent.api")


class PermissionDenied(Exception):
    """Something tried to perform a human-only operation."""


class NotFound(Exception):
    """No such subscriber, plan or transaction."""


class InsufficientBalance(Exception):
    """Not enough airtime for the requested purchase."""


AGENT_PREAPPROVED = frozenset(
    {
        "get_subscriber",
        "get_airtime_balance",
        "get_data_balance",
        "get_transactions",
        "list_data_plans",
        "diagnose",
        "retry_bundle_activation",
        "credit_pending_recharge",
        "buy_data_bundle",
        "send_network_settings",
        "raise_ticket",
    }
)

# The line between these two sets is the whole design. Everything below either
# hands someone control of the number, moves money back, or touches a
# regulatory status — none of which an AI decides on a call.
HUMAN_ONLY = frozenset(
    {
        "sim_swap",
        "sim_replacement",
        "port_out",
        "change_registered_details",
        "link_nin",
        "unbar_line",
        "refund_transaction",
        "migrate_plan",
        "close_account",
    }
)


class SimulatedTelcoAPI:
    """In-memory stand-in for the telco's core systems."""

    def __init__(self) -> None:
        state = telco_data.fresh_state()
        self._subscribers: list[dict] = state["subscribers"]
        self._data_balances: list[dict] = state["dataBalances"]
        self._transactions: list[dict] = state["transactions"]
        self._data_plans: list[dict] = state["dataPlans"]
        self._tickets: list[dict] = []
        self.audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Audit and guard
    # ------------------------------------------------------------------

    def _audit(self, operation: str, actor: str, detail: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "actor": actor,
            **detail,
        }
        self.audit_log.append(entry)
        logger.info("AUDIT %s by %s: %s", operation, actor, detail)

    def _guard(self, operation: str, actor: str) -> None:
        if actor == "agent" and operation in HUMAN_ONLY:
            self._audit("DENIED:" + operation, actor, {"reason": "human-only operation"})
            raise PermissionDenied(
                f"'{operation}' cannot be done by the AI agent and must be handled "
                f"by a human in the care team."
            )

    # ------------------------------------------------------------------
    # Subscriber
    # ------------------------------------------------------------------

    def get_subscriber(self, subscriber_id: str) -> dict:
        for sub in self._subscribers:
            if sub["subscriberId"] == subscriber_id:
                return sub
        raise NotFound(f"No subscriber {subscriber_id}")

    def get_subscriber_by_msisdn(self, msisdn: str) -> dict:
        digits = "".join(c for c in msisdn if c.isdigit())[-10:]
        for sub in self._subscribers:
            if "".join(c for c in sub["msisdn"] if c.isdigit()).endswith(digits):
                return sub
        raise NotFound(f"No subscriber for number ending {digits[-4:]}")

    def get_airtime_balance(self, subscriber_id: str) -> dict:
        sub = self.get_subscriber(subscriber_id)
        return {
            "msisdn": sub["msisdn"],
            "airtimeBalance": sub["airtimeBalance"],
            "currency": "NAIRA",
            "plan": sub["plan"],
        }

    def get_data_balance(self, subscriber_id: str) -> list[dict]:
        return [b for b in self._data_balances if b["subscriberId"] == subscriber_id]

    def get_transactions(self, subscriber_id: str, last_n: int = 5) -> list[dict]:
        rows = [t for t in self._transactions if t["subscriberId"] == subscriber_id]
        rows.sort(key=lambda t: t["timestamp"], reverse=True)
        return rows[:last_n]

    def list_data_plans(self) -> list[dict]:
        return list(self._data_plans)

    # ------------------------------------------------------------------
    # Diagnosis — the part that replaces "press 1 for data"
    # ------------------------------------------------------------------

    def diagnose(self, subscriber_id: str) -> dict:
        """
        Work out what is actually wrong with this line.

        This is the heart of the thing. An IVR makes the customer guess which
        menu branch their problem lives under; this looks at the account and
        says what the problem is, in order of how likely it is to be the cause.
        """
        sub = self.get_subscriber(subscriber_id)
        findings: list[dict] = []

        if not sub["ninLinked"]:
            findings.append(
                {
                    "cause": "NIN_NOT_LINKED",
                    "detail": (
                        "The N I N is not linked to this SIM, so the line is "
                        f"{sub['simStatus'].replace('_', ' ').lower()}. This is a "
                        "regulatory bar and only the care team can lift it."
                    ),
                    "agentCanFix": False,
                }
            )

        for txn in self.get_transactions(subscriber_id, last_n=10):
            if txn["status"] == "PAID_NOT_CREDITED":
                findings.insert(
                    0,
                    {
                        "cause": "RECHARGE_NOT_CREDITED",
                        "detail": (
                            f"{txn['amount']} Naira was paid by "
                            f"{txn['channel'].replace('_', ' ').lower()} but never "
                            f"credited to the line. Reference {txn['reference']}."
                        ),
                        "agentCanFix": True,
                        "reference": txn["reference"],
                    },
                )
            if txn["status"] == "DEBITED_NOT_DELIVERED":
                findings.append(
                    {
                        "cause": "BUNDLE_PAID_NOT_DELIVERED",
                        "detail": (
                            f"{txn['amount']} Naira was taken for '{txn['narration']}' "
                            f"but the bundle never activated. Reference {txn['reference']}."
                        ),
                        "agentCanFix": True,
                        "reference": txn["reference"],
                    }
                )
            if txn["status"] == "DISPUTED":
                findings.append(
                    {
                        "cause": "DISPUTED_DEDUCTION",
                        "detail": (
                            f"{txn['amount']} Naira was deducted for "
                            f"'{txn['narration']}'. The subscriber disputes it. "
                            "A refund is a human decision."
                        ),
                        "agentCanFix": False,
                        "reference": txn["reference"],
                    }
                )

        for bundle in self.get_data_balance(subscriber_id):
            if bundle["status"] == "PENDING_ACTIVATION":
                findings.append(
                    {
                        "cause": "BUNDLE_PENDING",
                        "detail": f"'{bundle['bundleName']}' is bought but stuck pending activation.",
                        "agentCanFix": True,
                    }
                )
            elif bundle["status"] == "EXPIRED":
                findings.append(
                    {
                        "cause": "BUNDLE_EXPIRED",
                        "detail": f"'{bundle['bundleName']}' has expired.",
                        "agentCanFix": True,
                    }
                )
            elif bundle["status"] == "ACTIVE" and bundle["remainingGb"] < 0.5:
                findings.append(
                    {
                        "cause": "NEARLY_OUT_OF_DATA",
                        "detail": (
                            f"Only {bundle['remainingGb']} G B left on "
                            f"'{bundle['bundleName']}', which is why browsing is slow."
                        ),
                        "agentCanFix": True,
                    }
                )

        if not findings:
            findings.append(
                {
                    "cause": "NO_FAULT_FOUND",
                    "detail": "Nothing wrong is visible on the account.",
                    "agentCanFix": False,
                }
            )

        self._audit("diagnose", "agent", {"subscriberId": subscriber_id, "found": len(findings)})
        return {"subscriberId": subscriber_id, "findings": findings}

    # ------------------------------------------------------------------
    # Pre-approved fixes
    # ------------------------------------------------------------------

    def retry_bundle_activation(self, subscriber_id: str, actor: str = "agent") -> dict:
        """
        Push a paid-for bundle through again.

        Pre-approved: the subscriber already paid, so delivering what they bought
        takes nothing further from them and is the whole reason they called.
        """
        self._guard("retry_bundle_activation", actor)
        for bundle in self._data_balances:
            if bundle["subscriberId"] == subscriber_id and bundle["status"] == "PENDING_ACTIVATION":
                bundle["status"] = "ACTIVE"
                for txn in self._transactions:
                    if (
                        txn["subscriberId"] == subscriber_id
                        and txn["status"] == "DEBITED_NOT_DELIVERED"
                    ):
                        txn["status"] = "SUCCESS"
                self._audit(
                    "retry_bundle_activation",
                    actor,
                    {"subscriberId": subscriber_id, "bundle": bundle["bundleName"]},
                )
                return bundle
        raise NotFound("No pending bundle to activate for this subscriber")

    def credit_pending_recharge(self, subscriber_id: str, actor: str = "agent") -> dict:
        """
        Credit a recharge the subscriber paid for that never landed.

        Pre-approved, and deliberately so. This is not the agent giving money
        away — the subscriber already paid and the transaction record proves it.
        Refusing to fix it without a human would recreate the exact wait that
        makes people give up on care lines, for the commonest complaint there is.

        A *disputed* deduction is different and stays human-only: there the
        record says the money was legitimately taken and the customer disagrees,
        which is a judgement call rather than a stuck transaction.
        """
        self._guard("credit_pending_recharge", actor)
        sub = self.get_subscriber(subscriber_id)

        for txn in self._transactions:
            if (
                txn["subscriberId"] == subscriber_id
                and txn["status"] == "PAID_NOT_CREDITED"
            ):
                before = sub["airtimeBalance"]
                sub["airtimeBalance"] = f"{float(before) + float(txn['amount']):.2f}"
                txn["status"] = "SUCCESS"
                self._audit(
                    "credit_pending_recharge",
                    actor,
                    {
                        "subscriberId": subscriber_id,
                        "reference": txn["reference"],
                        "amount": txn["amount"],
                        "balanceBefore": before,
                        "balanceAfter": sub["airtimeBalance"],
                    },
                )
                return {
                    "reference": txn["reference"],
                    "amount": txn["amount"],
                    "newBalance": sub["airtimeBalance"],
                }
        raise NotFound("No uncredited recharge on this line")

    def buy_data_bundle(self, subscriber_id: str, plan_id: str, actor: str = "agent") -> dict:
        """
        Buy a bundle out of the subscriber's airtime.

        Pre-approved but only ever after the customer says yes out loud — this
        spends their money. The prompt requires explicit confirmation of the
        plan and the price before this is called.
        """
        self._guard("buy_data_bundle", actor)
        sub = self.get_subscriber(subscriber_id)
        plan = next((p for p in self._data_plans if p["planId"] == plan_id), None)
        if plan is None:
            raise NotFound(f"No data plan {plan_id}")

        if float(sub["airtimeBalance"]) < float(plan["price"]):
            raise InsufficientBalance(
                f"Balance is {sub['airtimeBalance']} Naira but "
                f"'{plan['name']}' costs {plan['price']} Naira."
            )

        sub["airtimeBalance"] = f"{float(sub['airtimeBalance']) - float(plan['price']):.2f}"
        reference = f"TXN-{uuid.uuid4().hex[:6].upper()}"
        self._transactions.append(
            {
                "subscriberId": subscriber_id,
                "reference": reference,
                "type": "DATA_PURCHASE",
                "amount": plan["price"],
                "channel": "VOICE_AGENT",
                "narration": plan["name"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "SUCCESS",
            }
        )
        self._data_balances.append(
            {
                "subscriberId": subscriber_id,
                "bundleName": plan["name"],
                "remainingGb": plan["sizeGb"],
                "expiresAt": telco_data._days(plan["validityDays"]),
                "status": "ACTIVE",
            }
        )
        self._audit(
            "buy_data_bundle",
            actor,
            {"subscriberId": subscriber_id, "plan": plan["name"], "price": plan["price"]},
        )
        return {
            "reference": reference,
            "plan": plan["name"],
            "price": plan["price"],
            "newAirtimeBalance": sub["airtimeBalance"],
        }

    def send_network_settings(self, subscriber_id: str, actor: str = "agent") -> dict:
        """Push APN/internet settings by SMS. Harmless and often the actual fix."""
        self._guard("send_network_settings", actor)
        sub = self.get_subscriber(subscriber_id)
        self._audit("send_network_settings", actor, {"subscriberId": subscriber_id})
        return {"msisdn": sub["msisdn"], "sent": True, "network": sub["network"]}

    def raise_ticket(
        self,
        subscriber_id: str,
        summary: str,
        actions_taken: list[str],
        needs_human: bool,
        actor: str = "agent",
    ) -> dict:
        self._guard("raise_ticket", actor)
        ticket = {
            "ticketId": f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "subscriberId": subscriber_id,
            "summary": summary,
            "actionsTaken": actions_taken,
            "needsHuman": needs_human,
            "status": "OPEN_FOR_CARE_TEAM" if needs_human else "RESOLVED_BY_AGENT",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        self._tickets.append(ticket)
        self._audit("raise_ticket", actor, {"ticketId": ticket["ticketId"]})
        return ticket

    def get_tickets(self) -> list[dict]:
        return list(self._tickets)

    # ------------------------------------------------------------------
    # Human-only
    # ------------------------------------------------------------------

    def sim_swap(self, subscriber_id: str, actor: str = "human") -> dict:
        """
        Move a number to a new SIM. Human-only, permanently.

        SIM swap is the single most abused operation in Nigerian telecoms: take
        the line, receive the bank OTPs, empty the account. An AI that can be
        talked into this on a phone call is a weapon.
        """
        self._guard("sim_swap", actor)
        sub = self.get_subscriber(subscriber_id)
        self._audit("sim_swap", actor, {"subscriberId": subscriber_id})
        return sub

    def link_nin(self, subscriber_id: str, nin: str, actor: str = "human") -> dict:
        """Human-only. Lifting a regulatory bar needs verified identity documents."""
        self._guard("link_nin", actor)
        sub = self.get_subscriber(subscriber_id)
        sub["ninLinked"] = True
        sub["simStatus"] = "ACTIVE"
        self._audit("link_nin", actor, {"subscriberId": subscriber_id})
        return sub

    def refund_transaction(self, reference: str, actor: str = "human") -> dict:
        """Human-only. Money going back is a person's decision."""
        self._guard("refund_transaction", actor)
        for txn in self._transactions:
            if txn["reference"] == reference:
                txn["status"] = "REFUNDED"
                self._audit("refund_transaction", actor, {"reference": reference})
                return txn
        raise NotFound(f"No transaction {reference}")

    def unbar_line(self, subscriber_id: str, actor: str = "human") -> dict:
        self._guard("unbar_line", actor)
        sub = self.get_subscriber(subscriber_id)
        sub["simStatus"] = "ACTIVE"
        self._audit("unbar_line", actor, {"subscriberId": subscriber_id})
        return sub

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    @staticmethod
    def get_issue(issue_id: str) -> dict:
        issue = telco_data.KNOWN_ISSUES.get(issue_id)
        if not issue:
            raise NotFound(f"No issue {issue_id}")
        return dict(issue)

    @staticmethod
    def list_issues() -> list[dict]:
        return [dict(i) for i in telco_data.KNOWN_ISSUES.values()]


telco = SimulatedTelcoAPI()
