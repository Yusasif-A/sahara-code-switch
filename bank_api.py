"""
Simulated bank API, shaped after the Open Banking Nigeria API Standard.

The standard groups its operations as Account (Get Account, Get Account Balance,
Get Account By Customer BVN, Block Account), Customer (Get By BVN, Get By Phone
Number), Card (Get Cards, Get Card Limit, Set Card Limit, Block) and Transaction
(GetStatement, Get Customer Transaction Limit). The methods below mirror those
names so that replacing this class with a real HTTP client is a transport change.

The important thing this module encodes is the *authority boundary*: which
operations the AI agent may perform on its own, and which belong to a human on
the bank's fraud desk. That split is enforced here, in code, not merely asked for
in a prompt — a jailbroken conversation still cannot reach an irreversible call.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import bank_data

logger = logging.getLogger("fraud_agent.bank_api")


class PermissionDenied(Exception):
    """Raised when something tries to perform a human-only operation."""


class NotFound(Exception):
    """Raised when a customer, account or card does not exist."""


# Operations the agent is pre-approved to perform unaided. Every one of these is
# protective and reversible by the bank.
AGENT_PREAPPROVED = frozenset(
    {
        "get_customer_by_phone",
        "get_customer_by_bvn",
        "get_account",
        "get_account_balance",
        "get_cards",
        "get_statement",
        "block_card",
        "set_card_limit",
        "flag_transaction",
        "create_fraud_case",
    }
)

# Operations that move money, remove access, or change the contact details a
# future verification would rely on. A human on the fraud desk decides these.
HUMAN_ONLY = frozenset(
    {
        "block_account",
        "unblock_card",
        "reverse_transaction",
        "recall_funds",
        "update_customer_information",
        "close_account",
    }
)


class SimulatedBankAPI:
    """In-memory stand-in for the bank's core system."""

    def __init__(self) -> None:
        state = bank_data.fresh_state()
        self._customers: list[dict] = state["customers"]
        self._accounts: list[dict] = state["accounts"]
        self._cards: list[dict] = state["cards"]
        self._transactions: list[dict] = state["transactions"]
        self._fraud_cases: list[dict] = []
        self.audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Audit
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
        """Refuse a human-only operation when the caller is the AI agent."""
        if actor == "agent" and operation in HUMAN_ONLY:
            self._audit("DENIED:" + operation, actor, {"reason": "human-only operation"})
            raise PermissionDenied(
                f"'{operation}' is not pre-approved for the AI agent and must be "
                f"performed by the bank's human fraud team."
            )

    # ------------------------------------------------------------------
    # Customer resources
    # ------------------------------------------------------------------

    def get_customer_by_phone(self, phone_number: str) -> dict:
        """Open Banking NG: Customer > Get By Phone Number."""
        digits = "".join(c for c in phone_number if c.isdigit())[-10:]
        for customer in self._customers:
            if "".join(c for c in customer["phoneNumber"] if c.isdigit()).endswith(digits):
                return customer
        raise NotFound(f"No customer for phone number ending {digits[-4:]}")

    def get_customer_by_id(self, customer_id: str) -> dict:
        """Open Banking NG: Customer > Get By CustomerId."""
        for customer in self._customers:
            if customer["customerId"] == customer_id:
                return customer
        raise NotFound(f"No customer {customer_id}")

    def get_customer_by_bvn(self, bvn: str) -> dict:
        """Open Banking NG: Customer > Get By BVN."""
        for customer in self._customers:
            if customer["bvn"] == bvn:
                return customer
        raise NotFound("No customer for that BVN")

    # ------------------------------------------------------------------
    # Account resources
    # ------------------------------------------------------------------

    def get_account(self, account_number: str) -> dict:
        """Open Banking NG: Account > Get Account."""
        for account in self._accounts:
            if account["accountNumber"] == account_number:
                return account
        raise NotFound(f"No account {account_number}")

    def get_account_balance(self, account_number: str) -> dict:
        """Open Banking NG: Account > Get Account Balance."""
        account = self.get_account(account_number)
        return {
            "accountNumber": account["accountNumber"],
            "availableBalance": account["availableBalance"],
            "ledgerBalance": account["ledgerBalance"],
            "currency": account["currency"],
        }

    def get_accounts_by_customer(self, customer_id: str) -> list[dict]:
        """Open Banking NG: Account > Get Account By Customer Id."""
        return [a for a in self._accounts if a["customerId"] == customer_id]

    def block_account(self, account_number: str, reason: str, actor: str = "human") -> dict:
        """
        Open Banking NG: Account > Block Account.

        Human-only. Blocking an account stops salary credits, standing orders and
        every channel the customer has — too blunt for an AI to decide on a call.
        """
        self._guard("block_account", actor)
        account = self.get_account(account_number)
        account["status"] = "BLOCKED"
        self._audit("block_account", actor, {"accountNumber": account_number, "reason": reason})
        return account

    # ------------------------------------------------------------------
    # Card resources
    # ------------------------------------------------------------------

    def get_cards(self, account_number: str) -> list[dict]:
        """Open Banking NG: Card > Get Cards."""
        return [c for c in self._cards if c["accountNumber"] == account_number]

    def get_card(self, card_id: str) -> dict:
        for card in self._cards:
            if card["cardId"] == card_id:
                return card
        raise NotFound(f"No card {card_id}")

    def get_cards_by_customer(self, customer_id: str) -> list[dict]:
        """
        Every card this customer holds.

        Needed for calls the customer starts themselves: a fraud alert names the
        card, but somebody ringing to say "I have lost my card" has not, and may
        hold more than one. Reading the list back by brand, type and last four
        lets them pick without the agent guessing.
        """
        return [c for c in self._cards if c["customerId"] == customer_id]

    def block_card(self, card_id: str, reason: str, actor: str = "agent") -> dict:
        """
        Open Banking NG: Card > Block.

        Pre-approved for the agent. A frozen card is the textbook reversible
        protective step: it stops the bleeding and the bank can lift it later.
        """
        self._guard("block_card", actor)
        card = self.get_card(card_id)
        if card["status"] == "BLOCKED":
            return card
        card["status"] = "BLOCKED"
        card["blockedAt"] = datetime.now(timezone.utc).isoformat()
        card["blockReason"] = reason
        self._audit(
            "block_card",
            actor,
            {"cardId": card_id, "last4": card["last4"], "reason": reason},
        )
        return card

    def unblock_card(self, card_id: str, actor: str = "human") -> dict:
        """Human-only. Unfreezing restores the exact risk we just contained."""
        self._guard("unblock_card", actor)
        card = self.get_card(card_id)
        card["status"] = "ACTIVE"
        self._audit("unblock_card", actor, {"cardId": card_id})
        return card

    def get_card_limit(self, card_id: str) -> dict:
        """Open Banking NG: Card > Get Card Limit."""
        card = self.get_card(card_id)
        return {"cardId": card_id, "dailyLimit": card["dailyLimit"]}

    def set_card_limit(self, card_id: str, daily_limit: str, actor: str = "agent") -> dict:
        """
        Open Banking NG: Card > Set Card Limit.

        Pre-approved only in the protective direction. Raising a limit during a
        suspected-fraud call is exactly what a social engineer would ask for, so
        an increase is refused here regardless of what the conversation said.
        """
        self._guard("set_card_limit", actor)
        card = self.get_card(card_id)
        if actor == "agent" and float(daily_limit) > float(card["dailyLimit"]):
            self._audit(
                "DENIED:set_card_limit",
                actor,
                {"cardId": card_id, "requested": daily_limit, "current": card["dailyLimit"]},
            )
            raise PermissionDenied("The agent may only lower a card limit, never raise it.")
        previous = card["dailyLimit"]
        card["dailyLimit"] = daily_limit
        self._audit(
            "set_card_limit", actor, {"cardId": card_id, "from": previous, "to": daily_limit}
        )
        return card

    # ------------------------------------------------------------------
    # Transaction resources
    # ------------------------------------------------------------------

    def get_statement(self, account_number: str, last_n: int = 5) -> list[dict]:
        """Open Banking NG: Transaction > GetStatement."""
        rows = [t for t in self._transactions if t["accountNumber"] == account_number]
        rows.sort(key=lambda t: t["bookDate"], reverse=True)
        return rows[:last_n]

    def get_transaction(self, reference_id: str) -> dict:
        for txn in self._transactions:
            if txn["referenceId"] == reference_id:
                return txn
        raise NotFound(f"No transaction {reference_id}")

    def flag_transaction(self, reference_id: str, note: str, actor: str = "agent") -> dict:
        """
        Mark a transaction as disputed by the customer.

        Pre-approved: flagging records the customer's account of events and queues
        the item for the fraud desk. It does not move money, so it is safe for the
        agent — the actual reversal is a human decision.
        """
        self._guard("flag_transaction", actor)
        txn = self.get_transaction(reference_id)
        txn["disputed"] = True
        txn["disputeNote"] = note
        self._audit("flag_transaction", actor, {"referenceId": reference_id, "note": note})
        return txn

    def reverse_transaction(self, reference_id: str, actor: str = "human") -> dict:
        """Human-only. Moving money back is irreversible from the agent's seat."""
        self._guard("reverse_transaction", actor)
        txn = self.get_transaction(reference_id)
        txn["reversed"] = True
        self._audit("reverse_transaction", actor, {"referenceId": reference_id})
        return txn

    # ------------------------------------------------------------------
    # Fraud cases
    # ------------------------------------------------------------------

    def create_fraud_case(
        self,
        customer_id: str,
        # None for a call the customer started: there is no signal to point at,
        # and inventing one would put a fraud incident on an account where none
        # was ever raised.
        signal_id: str | None,
        summary: str,
        actions_taken: list[str],
        needs_human: bool,
        actor: str = "agent",
    ) -> dict:
        """Open a case for the fraud desk. Always safe, always recorded."""
        self._guard("create_fraud_case", actor)
        case = {
            "caseId": f"CASE-{uuid.uuid4().hex[:8].upper()}",
            "customerId": customer_id,
            "signalId": signal_id,
            "summary": summary,
            "actionsTaken": actions_taken,
            "needsHuman": needs_human,
            "status": "OPEN_FOR_REVIEW" if needs_human else "AUTO_CONTAINED",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        self._fraud_cases.append(case)
        self._audit("create_fraud_case", actor, {"caseId": case["caseId"], "signalId": signal_id})
        return case

    def get_fraud_cases(self) -> list[dict]:
        return list(self._fraud_cases)

    # ------------------------------------------------------------------
    # Fraud signals (what the detection engine emits)
    # ------------------------------------------------------------------

    @staticmethod
    def get_fraud_signal(signal_id: str) -> dict:
        signal = bank_data.FRAUD_SIGNALS.get(signal_id)
        if not signal:
            raise NotFound(f"No fraud signal {signal_id}")
        return dict(signal)

    @staticmethod
    def list_fraud_signals() -> list[dict]:
        return [dict(s) for s in bank_data.FRAUD_SIGNALS.values()]

bank = SimulatedBankAPI()
