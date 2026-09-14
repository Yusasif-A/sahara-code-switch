"""
Dummy data for the simulated Nigerian telco.

Modelled on how MTN, Glo, Airtel and 9mobile actually work for subscribers:
MSISDNs in 234 format, prepaid airtime in Naira, data bundles with expiry,
NIN-SIM linkage (mandatory since the NCC directive), and the specific
complaints that fill Nigerian call centres — data bought but not delivered,
airtime deducted twice, slow browsing, a line barred for NIN mismatch.

The point of the agent using this is that none of it should require the
customer to press 1, then 2, then 4, then hold. They say what is wrong and the
agent works out which of these records is the problem.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

NOW = datetime.now(timezone.utc)


def _iso(minutes_ago: int) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _days(days: int) -> str:
    return (NOW + timedelta(days=days)).isoformat()


NETWORKS = ("MTN", "GLO", "AIRTEL", "9MOBILE")

SUBSCRIBERS: list[dict] = [
    {
        "subscriberId": "SUB-200001",
        "msisdn": "+2348031234567",
        "firstName": "Yusuf",
        "lastName": "Ogunlesi",
        "network": "MTN",
        "accountType": "PREPAID",
        "airtimeBalance": "450.00",
        "plan": "MTN Pulse",
        "ninLinked": True,
        "simStatus": "ACTIVE",
        "preferredLanguage": "en",
        "registeredCity": "Lagos",
    },
    {
        "subscriberId": "SUB-200002",
        "msisdn": "+2348062345678",
        "firstName": "Chiamaka",
        "lastName": "Okonkwo",
        "network": "GLO",
        "accountType": "PREPAID",
        "airtimeBalance": "1250.00",
        "plan": "Glo Berekete",
        "ninLinked": True,
        "simStatus": "ACTIVE",
        "preferredLanguage": "ig",
        "registeredCity": "Enugu",
    },
    {
        "subscriberId": "SUB-200003",
        "msisdn": "+2349073456789",
        "firstName": "Musa",
        "lastName": "Abdullahi",
        "network": "AIRTEL",
        "accountType": "PREPAID",
        "airtimeBalance": "80.00",
        "plan": "Airtel SmartConnect",
        # The commonest reason a Nigerian line stops working.
        "ninLinked": False,
        "simStatus": "PARTIALLY_BARRED",
        "preferredLanguage": "ha",
        "registeredCity": "Kano",
    },
]

DATA_PLANS: list[dict] = [
    {"planId": "DP-1", "name": "1GB Daily", "sizeGb": 1.0, "price": "350.00", "validityDays": 1},
    {"planId": "DP-2", "name": "2.5GB Weekly", "sizeGb": 2.5, "price": "1000.00", "validityDays": 7},
    {"planId": "DP-3", "name": "10GB Monthly", "sizeGb": 10.0, "price": "3500.00", "validityDays": 30},
    {"planId": "DP-4", "name": "25GB Monthly", "sizeGb": 25.0, "price": "9000.00", "validityDays": 30},
]

DATA_BALANCES: list[dict] = [
    {
        "subscriberId": "SUB-200001",
        "bundleName": "10GB Monthly",
        "remainingGb": 0.42,
        "expiresAt": _days(11),
        "status": "ACTIVE",
    },
    {
        "subscriberId": "SUB-200002",
        "bundleName": "2.5GB Weekly",
        "remainingGb": 2.5,
        # Bought and never delivered — the classic complaint.
        "expiresAt": _days(6),
        "status": "PENDING_ACTIVATION",
    },
    {
        "subscriberId": "SUB-200003",
        "bundleName": "1GB Daily",
        "remainingGb": 0.0,
        "expiresAt": _days(-1),
        "status": "EXPIRED",
    },
]

TRANSACTIONS: list[dict] = [
    {
        "subscriberId": "SUB-200001",
        "reference": "TXN-5001",
        "type": "AIRTIME_TOPUP",
        "amount": "1000.00",
        "channel": "USSD",
        "narration": "Recharge via voucher",
        "timestamp": _iso(2880),
        "status": "SUCCESS",
    },
    {
        "subscriberId": "SUB-200001",
        "reference": "TXN-5002",
        "type": "DATA_PURCHASE",
        "amount": "3500.00",
        "channel": "APP",
        "narration": "10GB Monthly",
        "timestamp": _iso(1500),
        "status": "SUCCESS",
    },
    {
        "subscriberId": "SUB-200002",
        "reference": "TXN-5010",
        "type": "DATA_PURCHASE",
        "amount": "1000.00",
        "channel": "USSD",
        "narration": "2.5GB Weekly",
        "timestamp": _iso(35),
        # Money taken, bundle never activated.
        "status": "DEBITED_NOT_DELIVERED",
    },
    {
        "subscriberId": "SUB-200003",
        "reference": "TXN-5020",
        "type": "AIRTIME_TOPUP",
        "amount": "500.00",
        "channel": "POS",
        "narration": "Agent recharge",
        "timestamp": _iso(120),
        "status": "SUCCESS",
    },
    {
        # The single most common complaint on a Nigerian care line: money left
        # the customer, airtime never arrived. Usually a stuck transaction, not
        # a dishonest caller.
        "subscriberId": "SUB-200001",
        "reference": "TXN-5003",
        "type": "AIRTIME_TOPUP",
        "amount": "2000.00",
        "channel": "BANK_TRANSFER",
        "narration": "Recharge via bank app",
        "timestamp": _iso(22),
        "status": "PAID_NOT_CREDITED",
    },
    {
        "subscriberId": "SUB-200003",
        "reference": "TXN-5021",
        "type": "AIRTIME_DEDUCTION",
        "amount": "420.00",
        "channel": "SYSTEM",
        "narration": "Auto-renewal subscription",
        "timestamp": _iso(115),
        "status": "DISPUTED",
    },
]

# What the customer is actually ringing about. In production these would be
# inferred from the conversation; here they seed the scenarios.
KNOWN_ISSUES: dict[str, dict] = {
    "TEL-DATA-NOT-WORKING": {
        "issueId": "TEL-DATA-NOT-WORKING",
        "subscriberId": "SUB-200002",
        "category": "DATA_NOT_DELIVERED",
        "summary": (
            "Subscriber bought a 2.5GB weekly bundle 35 minutes ago via USSD. "
            "The airtime was deducted but the bundle is still PENDING_ACTIVATION, "
            "so they have no working data."
        ),
        "relatedTransactions": ["TXN-5010"],
    },
    "TEL-AIRTIME-DISPUTE": {
        "issueId": "TEL-AIRTIME-DISPUTE",
        "subscriberId": "SUB-200003",
        "category": "UNEXPLAINED_DEDUCTION",
        "summary": (
            "420 Naira was deducted for an auto-renewing subscription the "
            "subscriber says they never signed up for, five minutes after they "
            "recharged 500 Naira."
        ),
        "relatedTransactions": ["TXN-5021"],
    },
    "TEL-LINE-BARRED": {
        "issueId": "TEL-LINE-BARRED",
        "subscriberId": "SUB-200003",
        "category": "NIN_NOT_LINKED",
        "summary": (
            "The line is partially barred because the NIN is not linked to the "
            "SIM, so outgoing calls fail."
        ),
        "relatedTransactions": [],
    },
    "TEL-RECHARGE-NOT-REFLECTING": {
        "issueId": "TEL-RECHARGE-NOT-REFLECTING",
        "subscriberId": "SUB-200001",
        "category": "RECHARGE_NOT_CREDITED",
        "summary": (
            "The subscriber recharged 2,000 Naira from their bank app 22 minutes "
            "ago. The payment went through but the airtime was never credited, so "
            "their balance still reads 450 Naira."
        ),
        "relatedTransactions": ["TXN-5003"],
    },
    "TEL-SLOW-DATA": {
        "issueId": "TEL-SLOW-DATA",
        "subscriberId": "SUB-200001",
        "category": "SLOW_BROWSING",
        "summary": (
            "Subscriber reports very slow browsing. They have 0.42GB left of a "
            "10GB monthly bundle, so they are almost certainly throttled after "
            "the fair-use threshold rather than suffering a network fault."
        ),
        "relatedTransactions": [],
    },
}


def fresh_state() -> dict[str, list]:
    return {
        "subscribers": copy.deepcopy(SUBSCRIBERS),
        "dataBalances": copy.deepcopy(DATA_BALANCES),
        "transactions": copy.deepcopy(TRANSACTIONS),
        "dataPlans": copy.deepcopy(DATA_PLANS),
    }
