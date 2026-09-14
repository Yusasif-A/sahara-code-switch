"""
HTTP surface for the fraud-response agent.

Three jobs:
  - receive Meta's WhatsApp webhooks and relay the SDP to LiveKit
  - accept fraud signals from the bank's detection engine and place the call
  - expose the simulated bank so a demo can show what actually changed

Run it with:  uvicorn app:app --reload --port 8000
The agent worker runs separately:  python agent.py dev
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import whatsapp_connector
from bank_api import NotFound, bank
from caller import (
    CallFailed,
    accept_inbound_call,
    place_fraud_call,
    place_fraud_call_sip,
    raise_alert_and_notify,
)
from config import settings
from pending_alerts import pending

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_agent.app")

@asynccontextmanager
async def lifespan(_: FastAPI):
    problems = settings.validate()
    if problems:
        for problem in problems:
            logger.error("CONFIG: %s", problem)
        logger.error("Starting anyway, but calls will fail until these are fixed.")
    settings.log_summary()
    yield


app = FastAPI(
    lifespan=lifespan,
    title=f"{settings.bank.name} — Fraud Response Agent",
    description=(
        "Places an outbound call to a customer the moment a fraud signal is raised. "
        "The agent verifies via the approved challenge flow, freezes the card, and "
        "hands anything irreversible to the human fraud desk."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Meta WhatsApp webhook
# ----------------------------------------------------------------------


# Meta's console is fussy about the path you registered, and changing it there
# forces a re-verification round trip. Both paths are served so whichever URL is
# already in the console keeps working.
WEBHOOK_PATHS = ("/webhook/whatsapp", "/whatsapp")


async def verify_whatsapp_webhook(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
    hub_challenge: str = Query("", alias="hub.challenge"),
) -> Response:
    """Meta's one-time subscription handshake. Must echo the challenge back."""
    try:
        challenge = whatsapp_connector.verify_webhook_subscription(
            hub_mode, hub_verify_token, hub_challenge
        )
    except whatsapp_connector.WhatsAppError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return Response(content=challenge, media_type="text/plain")


async def receive_whatsapp_webhook(request: Request) -> dict:
    """
    Handle Meta's `calls` events.

    The one that matters is the SDP-bearing event: Meta has answered our dial and
    is offering its half of the media negotiation, which we relay straight to
    LiveKit. We always return 200 — a non-200 makes Meta retry the whole batch,
    which for a call event is worse than dropping one.
    """
    body = await request.json()

    # Log every payload. Meta's shapes vary by field and by event, and a silently
    # ignored webhook is indistinguishable from a broken one — which cost a live
    # call once already.
    logger.info("WEBHOOK RAW: %s", json.dumps(body)[:2000])

    events = whatsapp_connector.parse_call_events(body)
    if not events:
        fields = [
            change.get("field")
            for entry in body.get("entry", [])
            for change in entry.get("changes", [])
        ]
        logger.info("No call events extracted. Fields present: %s", fields or "none")
        return {"status": "ignored", "reason": "no call events", "fields": fields}

    for event in events:
        call_id, kind = event.get("call_id"), event.get("event")
        logger.info(
            "WhatsApp event %s for call %s (direction=%s status=%s sdp=%s)",
            kind,
            call_id,
            event.get("direction"),
            event.get("status"),
            "yes" if event.get("sdp") else "no",
        )

        try:
            direction = (event.get("direction") or "").upper()

            if event.get("sdp") and direction == "BUSINESS_INITIATED":
                # We dialled and Meta is completing the negotiation.
                await whatsapp_connector.connect_whatsapp_call(
                    call_id=call_id,
                    sdp=event["sdp"],
                    sdp_type=event.get("sdp_type") or "answer",
                )
                logger.info("Connected outbound call %s", call_id)

            elif event.get("sdp"):
                # Anything else carrying an SDP is the customer calling us.
                # Treating "not business-initiated" as inbound rather than
                # matching USER_INITIATED exactly means an unexpected or missing
                # direction still gets answered instead of silently ringing out.
                # The customer tapped the call button. This is the path that
                # works for Nigerian business numbers, where Meta blocks
                # business-initiated calls outright.
                result = await accept_inbound_call(
                    call_id=call_id,
                    from_number=event.get("from") or "",
                    sdp=event["sdp"],
                    sdp_type=event.get("sdp_type") or "offer",
                )
                logger.info(
                    "Accepted inbound call %s into %s (briefed=%s)",
                    call_id,
                    result["roomName"],
                    result["briefed"],
                )

            elif kind in ("terminate", "call_terminated"):
                logger.info("Call %s terminated by remote party", call_id)

        except Exception as exc:
            logger.error("Failed handling event %s for call %s: %s", kind, call_id, exc)

    return {"status": "ok", "handled": len(events)}


for _path in WEBHOOK_PATHS:
    app.add_api_route(_path, verify_whatsapp_webhook, methods=["GET"], tags=["whatsapp"])
    app.add_api_route(_path, receive_whatsapp_webhook, methods=["POST"], tags=["whatsapp"])


# ----------------------------------------------------------------------
# Fraud trigger — what the bank's detection engine calls
# ----------------------------------------------------------------------


class FraudAlert(BaseModel):
    signal_id: str = Field(..., description="Fraud signal id, e.g. FRD-CARD-FOREIGN")
    mode: str = Field(
        "notify",
        description=(
            "'notify' sends a WhatsApp alert and waits for the customer to call back — "
            "the only option that works from a Nigerian business number, since Meta "
            "blocks business-initiated calls there. 'call' dials them directly."
        ),
    )
    override_number: str | None = Field(
        None,
        description=(
            "Test only: use this number instead of the customer's. Never set this "
            "in production — it would read one customer's details to another number."
        ),
    )


@app.post("/fraud-alert")
async def trigger_fraud_call(alert: FraudAlert) -> dict:
    """
    Raise a fraud alert.

    In 'notify' mode the customer gets a WhatsApp message and the briefing is
    parked for their callback. In 'call' mode we dial them. Either way the
    conversation runs in the agent worker, not here.
    """
    if alert.mode not in ("notify", "call", "sip"):
        raise HTTPException(status_code=400, detail="mode must be 'notify', 'sip' or 'call'")

    try:
        if alert.mode == "notify":
            result = await raise_alert_and_notify(
                alert.signal_id, override_number=alert.override_number
            )
            return {"status": "notified", **result}

        if alert.mode == "sip":
            result = await place_fraud_call_sip(
                alert.signal_id, override_number=alert.override_number
            )
            return {"status": "calling", **result}

        result = await place_fraud_call(
            alert.signal_id, override_number=alert.override_number
        )
    except CallFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "calling", **result}


@app.get("/bank/pending-alerts")
async def list_pending_alerts() -> dict:
    """Alerts sent, waiting for the customer to call back."""
    return {"alerts": [a.as_dict() for a in pending.all()]}


# ----------------------------------------------------------------------
# Simulated bank — read-only views for the demo
# ----------------------------------------------------------------------


@app.get("/bank/signals")
async def list_signals() -> dict:
    """The fraud scenarios available to trigger."""
    return {"signals": bank.list_fraud_signals()}


@app.get("/bank/customers/{customer_id}")
async def get_customer(customer_id: str) -> dict:
    """Customer record with the security answers stripped out."""
    try:
        customer = dict(bank.get_customer_by_id(customer_id))
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    customer["securityQuestions"] = [
        q["question"] for q in customer.get("securityQuestions", [])
    ]
    return customer


@app.get("/bank/accounts/{account_number}")
async def get_account(account_number: str) -> dict:
    try:
        return {
            "account": bank.get_account(account_number),
            "cards": bank.get_cards(account_number),
            "recentTransactions": bank.get_statement(account_number, last_n=10),
        }
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/bank/cases")
async def list_cases() -> dict:
    """Fraud cases the agent opened, and what it did on each call."""
    return {"cases": bank.get_fraud_cases()}


@app.get("/bank/audit")
async def audit_trail() -> dict:
    """Every operation attempted against the bank, including refused ones."""
    return {"entries": bank.audit_log}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "bank": settings.bank.name,
        "agent": settings.bank.agent_display_name,
        "configProblems": settings.validate(),
    }
