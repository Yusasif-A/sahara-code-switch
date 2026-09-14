"""
Dummy data for the simulated bank.

Field names follow the Open Banking Nigeria API Standard so that swapping this
module for a real bank integration is a change of transport, not of shape. In
particular the transaction records match the GetStatement response schema
(accountNumber, amount, bookDate, channel, currency, debitOrCredit, narration,
referenceId, transactionType, valueDate) and the card records follow the common
Nigerian PSP card object (bin, last4, expMonth, expYear, cardType, brand).

Amounts are Naira, held as strings the way the standard returns them.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

NOW = datetime.now(timezone.utc)


def _iso(minutes_ago: int) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


# --------------------------------------------------------------------------
# Customers, accounts, cards
# --------------------------------------------------------------------------

CUSTOMERS: list[dict] = [
    {
        "customerId": "CUS-100001",
        "bvn": "22134567890",
        "firstName": "Yusuf",
        "lastName": "Ogunlesi",
        "phoneNumber": "+2348031234567",
        "email": "adebayo.ogunlesi@example.ng",
        "dateOfBirth": "1985-04-12",
        "preferredLanguage": "en",
        "city": "Lagos",
        # Each customer sets their own questions at onboarding, so both the
        # question and the answer differ per customer. The agent picks one at
        # random per call, which means a fraudster who watched one call still
        # cannot predict the next challenge.
        "securityQuestions": [
            {"question": "What is your favourite food?", "answer": "amala"},
            {"question": "What is your favourite colour?", "answer": "green"},
            # Stored the way a customer would say it, not abbreviated — this is a
            # voice system, and speech-to-text returns "saint", never "st".
            {"question": "What was the name of your first school?", "answer": "green flower school"},
        ],
    },
    {
        "customerId": "CUS-100002",
        "bvn": "22198765432",
        "firstName": "Chiamaka",
        "lastName": "Okonkwo",
        "phoneNumber": "+2348062345678",
        "email": "chiamaka.okonkwo@example.ng",
        "dateOfBirth": "1992-11-03",
        "preferredLanguage": "en",
        "city": "Enugu",
        "securityQuestions": [
            {"question": "What is your favourite colour?", "answer": "blue"},
            {"question": "Which football club do you support?", "answer": "chelsea"},
            {"question": "What is the name of your home town?", "answer": "nsukka"},
        ],
    },
    {
        "customerId": "CUS-100003",
        "bvn": "22111223344",
        "firstName": "Musa",
        "lastName": "Abdullahi",
        "phoneNumber": "+2349073456789",
        "email": "musa.abdullahi@example.ng",
        "dateOfBirth": "1978-07-22",
        "preferredLanguage": "en",
        "city": "Kano",
        "securityQuestions": [
            {"question": "What is your favourite food?", "answer": "tuwo shinkafa"},
            {"question": "What was the name of your first pet?", "answer": "bingo"},
            {"question": "What is your favourite colour?", "answer": "white"},
        ],
    },
    # Real handsets used for live demos. The phone lookup matches on the last
    # ten digits, so 08142392322 dialled locally resolves to the same record as
    # +2348142392322 arriving from WhatsApp.
    {
        "customerId": "CUS-100004",
        "bvn": "22245678901",
        "firstName": "Amaka",
        "lastName": "Eze",
        "phoneNumber": "+2348142392322",
        "email": "amaka.eze@example.ng",
        "dateOfBirth": "1990-07-21",
        "preferredLanguage": "en",
        "city": "Abuja",
        "securityQuestions": [
            {"question": "What is your favourite food?", "answer": "jollof rice"},
            {"question": "What is your favourite colour?", "answer": "purple"},
            {"question": "What was the name of your first school?", "answer": "holy child school"},
        ],
    },
    {
        "customerId": "CUS-100005",
        "bvn": "22256789012",
        "firstName": "Tunde",
        "lastName": "Bakare",
        "phoneNumber": "+2348107538562",
        "email": "tunde.bakare@example.ng",
        "dateOfBirth": "1988-02-09",
        "preferredLanguage": "en",
        "city": "Ibadan",
        "securityQuestions": [
            {"question": "What is your favourite food?", "answer": "efo riro"},
            {"question": "What is your favourite colour?", "answer": "black"},
            {"question": "What was the name of your first school?", "answer": "command secondary school"},
        ],
    },
]

ACCOUNTS: list[dict] = [
    {
        "accountNumber": "0123456789",  # NUBAN is 10 digits
        "customerId": "CUS-100001",
        "accountName": "Yusuf Ogunlesi",
        "accountType": "SAVINGS",
        "currency": "NAIRA",
        "availableBalance": "845200.00",
        "ledgerBalance": "845200.00",
        "status": "ACTIVE",
    },
    {
        "accountNumber": "0987654321",
        "customerId": "CUS-100002",
        "accountName": "Chiamaka Okonkwo",
        "accountType": "CURRENT",
        "currency": "NAIRA",
        "availableBalance": "2310750.00",
        "ledgerBalance": "2310750.00",
        "status": "ACTIVE",
    },
    {
        "accountNumber": "0456789123",
        "customerId": "CUS-100003",
        "accountName": "Musa Abdullahi",
        "accountType": "SAVINGS",
        "currency": "NAIRA",
        "availableBalance": "156400.00",
        "ledgerBalance": "156400.00",
        "status": "ACTIVE",
    },
    {
        "accountNumber": "0142392322",
        "customerId": "CUS-100004",
        "accountName": "Amaka Eze",
        "accountType": "SAVINGS",
        "currency": "NAIRA",
        "availableBalance": "312450.00",
        "ledgerBalance": "312450.00",
        "status": "ACTIVE",
    },
    {
        "accountNumber": "0107538562",
        "customerId": "CUS-100005",
        "accountName": "Tunde Bakare",
        "accountType": "CURRENT",
        "currency": "NAIRA",
        "availableBalance": "1058900.00",
        "ledgerBalance": "1058900.00",
        "status": "ACTIVE",
    },
]

CARDS: list[dict] = [
    {
        "cardId": "CRD-77001",
        "accountNumber": "0123456789",
        "customerId": "CUS-100001",
        "bin": "539941",
        "last4": "4081",
        "maskedPan": "539941******4081",
        "expMonth": "09",
        "expYear": "2028",
        "cardType": "DEBIT",
        "brand": "verve",
        "status": "ACTIVE",
        "dailyLimit": "500000.00",
        "channelsEnabled": ["POS", "ATM", "WEB"],
    },
    {
        "cardId": "CRD-77002",
        "accountNumber": "0987654321",
        "customerId": "CUS-100002",
        "bin": "418742",
        "last4": "9032",
        "maskedPan": "418742******9032",
        "expMonth": "02",
        "expYear": "2027",
        "cardType": "DEBIT",
        "brand": "visa",
        "status": "ACTIVE",
        "dailyLimit": "1000000.00",
        "channelsEnabled": ["POS", "ATM", "WEB"],
    },
    {
        "cardId": "CRD-77003",
        "accountNumber": "0456789123",
        "customerId": "CUS-100003",
        "bin": "512345",
        "last4": "6677",
        "maskedPan": "512345******6677",
        "expMonth": "11",
        "expYear": "2026",
        "cardType": "DEBIT",
        "brand": "mastercard",
        "status": "ACTIVE",
        "dailyLimit": "300000.00",
        "channelsEnabled": ["POS", "ATM", "WEB"],
    },
    {
        "cardId": "CRD-77004",
        "accountNumber": "0142392322",
        "customerId": "CUS-100004",
        "bin": "539941",
        "last4": "2322",
        "maskedPan": "539941******2322",
        "expMonth": "11",
        "expYear": "2029",
        "cardType": "DEBIT",
        "brand": "verve",
        "status": "ACTIVE",
        "dailyLimit": "300000.00",
        "channelsEnabled": ["POS", "ATM", "WEB"],
    },
    {
        "cardId": "CRD-77005",
        "accountNumber": "0107538562",
        "customerId": "CUS-100005",
        "bin": "418742",
        "last4": "8562",
        "maskedPan": "418742******8562",
        "expMonth": "05",
        "expYear": "2028",
        "cardType": "DEBIT",
        "brand": "visa",
        "status": "ACTIVE",
        "dailyLimit": "750000.00",
        "channelsEnabled": ["POS", "ATM", "WEB"],
    },
]


# --------------------------------------------------------------------------
# Transactions — GetStatement response shape
# --------------------------------------------------------------------------

TRANSACTIONS: list[dict] = [
    # ---- Yusuf: normal history, then a suspicious foreign card burst ----
    {
        "accountNumber": "0123456789",
        "amount": "12500.00",
        "bookDate": _iso(4320),
        "valueDate": _iso(4320),
        "channel": "POS",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "POS PURCHASE SHOPRITE IKEJA LAGOS NG",
        "referenceId": "TRX-9001",
        "transactionType": "CARD_PURCHASE",
        "countryCode": "NG",
        "flagged": False,
    },
    {
        "accountNumber": "0123456789",
        "amount": "450000.00",
        "bookDate": _iso(2880),
        "valueDate": _iso(2880),
        "channel": "NIP",
        "currency": "NAIRA",
        "debitOrCredit": "CREDIT",
        "narration": "NIP TRANSFER FROM LAGOS STATE PAYROLL",
        "referenceId": "TRX-9002",
        "transactionType": "TRANSFER",
        "countryCode": "NG",
        "flagged": False,
    },
    {
        "accountNumber": "0123456789",
        "amount": "185000.00",
        "bookDate": _iso(14),
        "valueDate": _iso(14),
        "channel": "WEB",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "CARD PURCHASE ELECTRONICS STORE KYIV UA",
        "referenceId": "TRX-9003",
        "transactionType": "CARD_PURCHASE",
        "countryCode": "UA",
        "flagged": True,
    },
    {
        "accountNumber": "0123456789",
        "amount": "192000.00",
        "bookDate": _iso(11),
        "valueDate": _iso(11),
        "channel": "WEB",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "CARD PURCHASE ELECTRONICS STORE KYIV UA",
        "referenceId": "TRX-9004",
        "transactionType": "CARD_PURCHASE",
        "countryCode": "UA",
        "flagged": True,
    },
    # ---- Chiamaka: account takeover, large outbound transfer pending ----
    {
        "accountNumber": "0987654321",
        "amount": "35000.00",
        "bookDate": _iso(5760),
        "valueDate": _iso(5760),
        "channel": "MOBILE",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "AIRTIME PURCHASE MTN NG",
        "referenceId": "TRX-9010",
        "transactionType": "BILL_PAYMENT",
        "countryCode": "NG",
        "flagged": False,
    },
    {
        "accountNumber": "0987654321",
        "amount": "1850000.00",
        "bookDate": _iso(6),
        "valueDate": _iso(6),
        "channel": "NIP",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "NIP TRANSFER TO ADEYEMI J - NEW BENEFICIARY",
        "referenceId": "TRX-9011",
        "transactionType": "TRANSFER",
        "countryCode": "NG",
        "flagged": True,
    },
    # ---- Musa: ATM velocity burst ----
    {
        "accountNumber": "0456789123",
        "amount": "20000.00",
        "bookDate": _iso(9),
        "valueDate": _iso(9),
        "channel": "ATM",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "ATM WITHDRAWAL SABON GARI KANO NG",
        "referenceId": "TRX-9020",
        "transactionType": "CASH_WITHDRAWAL",
        "countryCode": "NG",
        "flagged": True,
    },
    {
        "accountNumber": "0456789123",
        "amount": "20000.00",
        "bookDate": _iso(7),
        "valueDate": _iso(7),
        "channel": "ATM",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "ATM WITHDRAWAL SABON GARI KANO NG",
        "referenceId": "TRX-9021",
        "transactionType": "CASH_WITHDRAWAL",
        "countryCode": "NG",
        "flagged": True,
    },
    {
        "accountNumber": "0456789123",
        "amount": "20000.00",
        "bookDate": _iso(5),
        "valueDate": _iso(5),
        "channel": "ATM",
        "currency": "NAIRA",
        "debitOrCredit": "DEBIT",
        "narration": "ATM WITHDRAWAL SABON GARI KANO NG",
        "referenceId": "TRX-9022",
        "transactionType": "CASH_WITHDRAWAL",
        "countryCode": "NG",
        "flagged": True,
    },
]


# --------------------------------------------------------------------------
# Fraud signals — what the bank's detection engine hands us to start a call
# --------------------------------------------------------------------------

FRAUD_SIGNALS: dict[str, dict] = {
    "FRD-CARD-FOREIGN": {
        "signalId": "FRD-CARD-FOREIGN",
        "riskType": "CARD_NOT_PRESENT_FOREIGN",
        "severity": "HIGH",
        "customerId": "CUS-100001",
        "accountNumber": "0123456789",
        "cardId": "CRD-77001",
        "summary": (
            "Two card-not-present purchases totalling 377,000 Naira were attempted "
            "in Ukraine within 3 minutes, on a card that has only ever been used in Nigeria."
        ),
        "transactionRefs": ["TRX-9003", "TRX-9004"],
        "recommendedAction": "FREEZE_CARD",
    },
    "FRD-ACCOUNT-TAKEOVER": {
        "signalId": "FRD-ACCOUNT-TAKEOVER",
        "riskType": "ACCOUNT_TAKEOVER",
        "severity": "CRITICAL",
        "customerId": "CUS-100002",
        "accountNumber": "0987654321",
        "cardId": "CRD-77002",
        "summary": (
            "Login from a new device in a new location, followed 4 minutes later by a "
            "1,850,000 Naira transfer to a beneficiary added in the same session."
        ),
        "transactionRefs": ["TRX-9011"],
        "recommendedAction": "FREEZE_CARD",
    },
    "FRD-ATM-VELOCITY": {
        "signalId": "FRD-ATM-VELOCITY",
        "riskType": "ATM_VELOCITY",
        "severity": "MEDIUM",
        "customerId": "CUS-100003",
        "accountNumber": "0456789123",
        "cardId": "CRD-77003",
        "summary": (
            "Three consecutive 20,000 Naira ATM withdrawals at the same terminal "
            "within 4 minutes, which is unusual for this account."
        ),
        "transactionRefs": ["TRX-9020", "TRX-9021", "TRX-9022"],
        "recommendedAction": "FREEZE_CARD",
    },
}


def fresh_state() -> dict[str, list]:
    """A deep copy of the seed data, so each bank instance mutates independently."""
    return {
        "customers": copy.deepcopy(CUSTOMERS),
        "accounts": copy.deepcopy(ACCOUNTS),
        "cards": copy.deepcopy(CARDS),
        "transactions": copy.deepcopy(TRANSACTIONS),
    }
