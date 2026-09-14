"""
Walk one fraud call end to end without LiveKit, WhatsApp or any credentials.

This exercises the real bank API, the real challenge flow and the real authority
boundary — everything except the voice transport — so you can see what the agent
would do, and what it would be stopped from doing.

Run with:  python demo.py
"""

from __future__ import annotations

import logging

from bank_api import PermissionDenied, SimulatedBankAPI
from config import settings
from prompts import build_opening_line, build_risk_briefing
from verification import ChallengeFlow

logging.basicConfig(level=logging.WARNING)

RULE = "─" * 74


def say(who: str, text: str) -> None:
    print(f"  {who:<9} {text}")


def main() -> None:
    bank = SimulatedBankAPI()
    signal = bank.get_fraud_signal("FRD-CARD-FOREIGN")
    customer = bank.get_customer_by_id(signal["customerId"])
    card = bank.get_card(signal["cardId"])

    print(RULE)
    print(f"  {settings.bank.name} — fraud signal raised")
    print(RULE)
    print(build_risk_briefing(signal, customer, card))
    print()

    print(RULE)
    print("  THE CALL")
    print(RULE)

    say(
        "agent",
        build_opening_line(
            first_name=customer["firstName"],
            agent_name=settings.bank.agent_display_name,
            bank_name=settings.bank.name,
        ),
    )
    say("customer", "Wait — how do I know this is really my bank?")
    say(
        "agent",
        "You are right to ask, and you should always ask. I have not asked you for "
        "anything secret and I never will. You can hang up and call this same number "
        "back, and what I am about to do will still be in place.",
    )

    # --- verification ---
    flow = ChallengeFlow(customer=customer, max_attempts=settings.bank.max_verification_attempts)
    question = flow.next_question()
    say("agent", question)

    correct = next(q["answer"] for q in customer["securityQuestions"] if q["question"] == question)
    say("customer", f"'{correct}'")
    assert flow.check(correct)
    say("system", "✓ verified via approved challenge flow (no PIN, no OTP, no BVN)")

    # --- explain ---
    txns = [bank.get_transaction(r) for r in signal["transactionRefs"]]
    total = sum(float(t["amount"]) for t in txns)
    say(
        "agent",
        f"In the last fifteen minutes there were {len(txns)} purchases on your card "
        f"ending {card['last4']}, totalling {total:,.0f} Naira, in Ukraine. Were those you?",
    )
    say("customer", "No! I've never been to Ukraine.")

    # --- the pre-approved protective step ---
    frozen = bank.block_card(card["cardId"], reason="customer does not recognise UA purchases", actor="agent")
    say("agent", f"Your card ending {frozen['last4']} is now frozen. It cannot be used.")
    say("system", f"✓ card {frozen['cardId']} status = {frozen['status']}  [pre-approved]")

    for txn in txns:
        bank.flag_transaction(txn["referenceId"], note="customer disputes", actor="agent")
    say("system", f"✓ {len(txns)} transactions flagged for the fraud desk  [pre-approved]")

    # --- what the agent is not allowed to do ---
    print()
    print(RULE)
    print("  WHERE THE AGENT IS STOPPED")
    print(RULE)
    say("customer", "Can you just reverse the charges and give me my money back?")

    for label, call in [
        ("reverse the transaction", lambda: bank.reverse_transaction("TRX-9003", actor="agent")),
        ("block the whole account", lambda: bank.block_account(signal["accountNumber"], "fraud", actor="agent")),
        ("raise the card limit", lambda: bank.set_card_limit(card["cardId"], "9000000.00", actor="agent")),
        ("unfreeze the card", lambda: bank.unblock_card(card["cardId"], actor="agent")),
    ]:
        try:
            call()
            say("system", f"✗ AGENT WAS ALLOWED TO {label.upper()} — THIS IS A BUG")
        except PermissionDenied:
            say("system", f"⛔ refused: agent may not {label} — human fraud desk only")

    say(
        "agent",
        "I cannot reverse a payment myself — that decision belongs to a person on our "
        "fraud team. I am connecting you to them now.",
    )

    case = bank.create_fraud_case(
        customer_id=customer["customerId"],
        signal_id=signal["signalId"],
        summary="Card-not-present fraud in UA; customer denies. Card frozen.",
        actions_taken=[f"froze card ending {card['last4']}", "flagged 2 transactions"],
        needs_human=True,
        actor="agent",
    )
    say("system", f"✓ case {case['caseId']} opened, status {case['status']}")

    # --- what the bank sees afterwards ---
    print()
    print(RULE)
    print("  AUDIT TRAIL")
    print(RULE)
    for entry in bank.audit_log:
        mark = "⛔" if entry["operation"].startswith("DENIED:") else "  "
        print(f"  {mark} {entry['timestamp'][11:19]}  {entry['actor']:<6} {entry['operation']}")

    print()
    print(f"  Card ending {card['last4']} is {bank.get_card(card['cardId'])['status']}.")
    print(f"  Account {signal['accountNumber']} is {bank.get_account(signal['accountNumber'])['status']}.")
    print("  The freeze happened. Nothing irreversible did.")
    print(RULE)


if __name__ == "__main__":
    main()
