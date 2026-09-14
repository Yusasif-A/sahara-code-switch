"""
Outbound call orchestration.

One function matters here: place_fraud_call. It takes a fraud signal id, works
out who to ring, creates the room, dispatches the agent with the briefing, and
places the WhatsApp call.

The ordering is deliberate. The agent is dispatched into the room *before* the
call is placed, so that it is already listening when the customer says "hello".
Dispatching after the answer loses the first second or two of the call, which on
a fraud call is the part where the customer is deciding whether to trust you.
"""

from __future__ import annotations

import json
import logging
import uuid

from livekit import api

import whatsapp_connector
from bank_api import NotFound, bank
from config import settings
from pending_alerts import pending

logger = logging.getLogger("fraud_agent.caller")


class CallFailed(Exception):
    """The call could not be placed."""


def _mask(number: str) -> str:
    """Never write a full customer number to a log line."""
    return f"***{number[-4:]}" if len(number) >= 4 else "***"


async def place_fraud_call(signal_id: str, *, override_number: str | None = None) -> dict:
    """
    Ring the customer named on a fraud signal.

    Args:
        signal_id: which signal from the bank's detection engine.
        override_number: dial this instead of the customer's number on file.
            For testing against your own WhatsApp — never set in production, as
            it would send one customer's account details to another number.
    """
    try:
        signal = bank.get_fraud_signal(signal_id)
        customer = bank.get_customer_by_id(signal["customerId"])
    except NotFound as exc:
        raise CallFailed(str(exc)) from exc

    to_number = override_number or customer["phoneNumber"]
    if override_number:
        logger.warning(
            "Dialling override number %s instead of the customer on file — test mode only",
            _mask(override_number),
        )

    room_name = f"fraud-{signal_id.lower()}-{uuid.uuid4().hex[:6]}"

    # The agent reads this to know which signal it is calling about. Only the id
    # travels; the agent re-reads the record from the bank itself, so nothing
    # sensitive sits in dispatch metadata.
    metadata = json.dumps({"signalId": signal_id, "roomName": room_name})

    logger.info(
        "Placing fraud call: signal=%s severity=%s customer=%s to=%s room=%s",
        signal_id,
        signal["severity"],
        customer["customerId"],
        _mask(to_number),
        room_name,
    )

    lkapi = api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )
    try:
        await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=120))

        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=settings.agent_name,
                metadata=metadata,
            )
        )
        logger.info(
            "Agent %s dispatched into %s on %s",
            settings.agent_name,
            room_name,
            settings.livekit.url,
        )

        dial_result = await whatsapp_connector.dial_whatsapp_call(
            to_number=to_number,
            room_name=room_name,
            agent_name=settings.agent_name,
            agent_metadata=metadata,
        )
    except Exception as exc:
        logger.error("Call failed for signal %s: %s", signal_id, exc)
        # Do not leave an empty room holding a dispatch behind.
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
        except Exception:
            logger.warning("Could not clean up room %s", room_name)
        raise CallFailed(f"Could not place call for {signal_id}: {exc}") from exc
    finally:
        await lkapi.aclose()

    return {
        "signalId": signal_id,
        "customerId": customer["customerId"],
        "roomName": room_name,
        "to": _mask(to_number),
        "severity": signal["severity"],
        "dial": dial_result,
    }


# ----------------------------------------------------------------------
# SIP (Twilio) — a real phone call
# ----------------------------------------------------------------------


async def place_fraud_call_sip(signal_id: str, *, override_number: str | None = None) -> dict:
    """
    Ring the customer on an actual phone number through a SIP trunk.

    This is the transport that works for a Nigerian bank. WhatsApp blocks
    business-initiated calls from Nigerian numbers outright, and its LiveKit
    connector is a closed beta — neither restriction applies to SIP. It is also
    what a customer expects: banks phone you, they do not WhatsApp-call you.

    The agent is dispatched into the room before the number is dialled, so it is
    already listening when the customer says hello.
    """
    if not settings.sip.configured:
        raise CallFailed(
            "No SIP trunk configured. Set SIP_OUTBOUND_TRUNK_ID in .env — "
            "run `python setup_twilio.py --create-trunk` to make one."
        )

    try:
        signal = bank.get_fraud_signal(signal_id)
        customer = bank.get_customer_by_id(signal["customerId"])
    except NotFound as exc:
        raise CallFailed(str(exc)) from exc

    to_number = override_number or customer["phoneNumber"]
    room_name = f"fraud-sip-{signal_id.lower()}-{uuid.uuid4().hex[:6]}"
    metadata = json.dumps({"signalId": signal_id, "transport": "sip"})

    logger.info(
        "SIP call: signal=%s to=%s room=%s trunk=%s",
        signal_id,
        _mask(to_number),
        room_name,
        settings.sip.outbound_trunk_id,
    )

    lkapi = api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )
    try:
        await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=120))
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name, agent_name=settings.agent_name, metadata=metadata
            )
        )
        logger.info("Agent dispatched into %s; dialling ...", room_name)

        participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=settings.sip.outbound_trunk_id,
                sip_call_to=to_number,
                room_name=room_name,
                participant_identity=to_number,
                participant_name=f"{customer['firstName']} {customer['lastName']}",
                # Block until the customer actually answers, so a failure to
                # connect surfaces here rather than as a silent empty room.
                wait_until_answered=True,
            )
        )
    except Exception as exc:
        logger.error("SIP call failed for %s: %s", signal_id, exc)
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
        except Exception:
            logger.warning("Could not clean up room %s", room_name)
        raise CallFailed(f"Could not place SIP call for {signal_id}: {exc}") from exc
    finally:
        await lkapi.aclose()

    logger.info("Customer answered. Participant %s in %s", participant.participant_identity, room_name)
    return {
        "transport": "sip",
        "signalId": signal_id,
        "customerId": customer["customerId"],
        "roomName": room_name,
        "to": _mask(to_number),
        "severity": signal["severity"],
    }


# ----------------------------------------------------------------------
# Inverted flow — message the customer, they call us back
# ----------------------------------------------------------------------


async def raise_alert_and_notify(signal_id: str, *, override_number: str | None = None) -> dict:
    """
    Send the customer a WhatsApp alert and park the briefing for their callback.

    Used where business-initiated calls are unavailable — Nigeria included. The
    customer taps the call button, which arrives as an inbound call that
    accept_inbound_call picks up with the right briefing already loaded.
    """
    try:
        signal = bank.get_fraud_signal(signal_id)
        customer = bank.get_customer_by_id(signal["customerId"])
        card = bank.get_card(signal["cardId"]) if signal.get("cardId") else None
    except NotFound as exc:
        raise CallFailed(str(exc)) from exc

    to_number = override_number or customer["phoneNumber"]

    alert = pending.raise_alert(
        signal_id=signal_id,
        customer_id=customer["customerId"],
        phone_number=to_number,
    )

    try:
        await whatsapp_connector.send_fraud_alert_message(
            to_number=to_number,
            first_name=customer["firstName"],
            last4=card["last4"] if card else None,
        )
    except whatsapp_connector.WhatsAppError as exc:
        pending.clear(to_number)
        raise CallFailed(f"Alert message failed: {exc}") from exc

    pending.mark_notified(to_number)
    logger.info(
        "Alert %s sent to %s; awaiting callback until %s",
        signal_id,
        _mask(to_number),
        alert.expires_at.isoformat(timespec="seconds"),
    )

    return {
        "mode": "await_callback",
        "signalId": signal_id,
        "customerId": customer["customerId"],
        "notified": _mask(to_number),
        "severity": signal["severity"],
        "answerableUntil": alert.expires_at.isoformat(),
    }


async def accept_inbound_call(
    *, call_id: str, from_number: str, sdp: str, sdp_type: str = "offer"
) -> dict:
    """
    Answer a customer who tapped the call button.

    Looks up why they are calling, spins up a room with the agent already
    briefed, then accepts the call so media flows into that room.
    """
    # Which agent answers?
    #
    # Both agents share one WhatsApp number, so something has to decide. The
    # rule is the honest one: a caller with a live fraud alert is calling back
    # about that alert — nothing else would be a coincidence at that moment.
    # Everyone else gets telco care, which is the general-purpose line.
    #
    # DEMO_AGENT=fraud|telco pins it for a demo, so you can show either without
    # having to arrange the right account state first.
    forced = settings.demo_agent
    alert = pending.get(from_number)

    if forced == "telco" or (forced != "fraud" and alert is None):
        agent_name = settings.telco.agent_name
        metadata = json.dumps({"msisdn": from_number, "inbound": True})
        room_name = f"telco-in-{uuid.uuid4().hex[:6]}"
        logger.info(
            "Inbound call from %s -> telco care agent%s",
            _mask(from_number),
            " (forced by DEMO_AGENT)" if forced else "",
        )
    elif alert:
        agent_name = settings.agent_name
        metadata = json.dumps({"signalId": alert.signal_id, "inbound": True})
        room_name = f"fraud-in-{alert.signal_id.lower()}-{uuid.uuid4().hex[:6]}"
        logger.info(
            "Inbound call from %s matches alert %s", _mask(from_number), alert.signal_id
        )
    else:
        # DEMO_AGENT=fraud with nothing pending. Answer anyway — someone ringing
        # the fraud line deserves a response — but tell the agent there is no
        # briefing so it greets and hands over rather than inventing a fraud.
        agent_name = settings.agent_name
        metadata = json.dumps({"phoneNumber": from_number, "inbound": True, "noAlert": True})
        room_name = f"fraud-in-unknown-{uuid.uuid4().hex[:6]}"
        logger.warning("Inbound call from %s with no pending alert", _mask(from_number))

    # The agent is dispatched through accept_whatsapp_call itself rather than as
    # a separate CreateAgentDispatch, so LiveKit brings the worker into the room
    # as part of accepting the call — one round trip, and no window where the
    # caller is connected to an empty room.
    try:
        await whatsapp_connector.accept_whatsapp_call(
            call_id=call_id,
            sdp=sdp,
            sdp_type=sdp_type,
            room_name=room_name,
            agent_name=agent_name,
            agent_metadata=metadata,
            participant_identity=from_number,
        )
        logger.info(
            "Accepted call into %s with agent '%s' on %s",
            room_name,
            agent_name,
            settings.livekit.url,
        )
    except Exception as exc:
        logger.error("Could not accept inbound call %s: %s", call_id, exc)
        raise CallFailed(f"Could not accept call {call_id}: {exc}") from exc

    if alert:
        pending.mark_answered(from_number)

    return {"callId": call_id, "roomName": room_name, "briefed": alert is not None}
