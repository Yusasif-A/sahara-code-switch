"""
WhatsApp transport, via LiveKit's native connector SDK.

How a WhatsApp call works on LiveKit (docs.livekit.io/telephony/connectors/whatsapp):
LiveKit and Meta negotiate media over SDP, and Meta delivers its half to *your*
webhook. So a call is a three-step dance:

    outbound:  dial_whatsapp_call  -> Meta webhook (SDP) -> connect_whatsapp_call
    inbound:   Meta webhook (SDP)  -> accept_whatsapp_call

LiveKit is not pre-configured with your WhatsApp account. Every RPC carries the
phone number id, access token and Cloud API version, and LiveKit uses them to
talk to Meta on your behalf.

A NOTE ON HOW THIS WAS DEBUGGED, because the failure mode is nasty: an earlier
version of this module hand-rolled the Twirp calls with guessed camelCase field
names (`whatsAppApiKey` and friends). The real fields are snake_case and
all-lowercase `whatsapp` — `whatsapp_api_key`. LiveKit silently ignores unknown
fields, so every call returned HTTP 200 with a body of "OK" and then simply
never bridged any media. The room stayed empty, session.start() blocked forever,
and the customer heard ringing. Using the generated protobuf request objects
below makes that class of mistake impossible: a wrong field name is now an
AttributeError at the call site instead of a silent no-op on a live call.

Requirements: WhatsApp calling is LiveKit Cloud only, the number must have
calling enabled with the `calls` webhook field subscribed, and business-initiated
calls are unavailable from numbers registered in Nigeria, the USA, Canada, Egypt
or Vietnam (inbound is fine everywhere).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp
from livekit import api

import pending_alerts
from config import settings

logger = logging.getLogger("fraud_agent.whatsapp")


class WhatsAppError(Exception):
    """A WhatsApp connector call failed."""


def _client() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )


def _cloud_api_version() -> str:
    """Meta's version without the leading 'v' — "23.0", not "v23.0"."""
    return settings.whatsapp.graph_version.lstrip("vV")


def _session_description(sdp: str, sdp_type: str) -> dict:
    """
    Wrap a raw SDP string as a livekit.SessionDescription.

    The `sdp` field on these requests is a message, not a string — passing the
    raw SDP fails with "expected <class 'rtc.SessionDescription'> got
    <class 'str'>". Protobuf accepts a dict for a message field, which is used
    here because livekit.SessionDescription is not exported from any public
    module path in this SDK version.

    sdp_type comes straight from Meta's webhook: "offer" on an inbound call,
    "answer" when completing one we dialled.
    """
    return {"type": sdp_type, "sdp": sdp}


# ----------------------------------------------------------------------
# Outbound
# ----------------------------------------------------------------------


async def dial_whatsapp_call(
    *,
    to_number: str,
    room_name: str,
    agent_name: str,
    agent_metadata: str,
) -> Any:
    """
    Ask LiveKit to place a WhatsApp call.

    Meta answers asynchronously on the webhook, so returning here means the call
    was *placed*, not answered. Blocked outright for Nigerian business numbers —
    use the inbound flow instead.
    """
    lkapi = _client()
    try:
        result = await lkapi.connector.dial_whatsapp_call(
            api.DialWhatsAppCallRequest(
                whatsapp_phone_number_id=settings.whatsapp.phone_number_id,
                whatsapp_to_phone_number=to_number,
                whatsapp_api_key=settings.whatsapp.token,
                whatsapp_cloud_api_version=_cloud_api_version(),
                room_name=room_name,
                agents=[
                    api.RoomAgentDispatch(agent_name=agent_name, metadata=agent_metadata)
                ],
                participant_identity=to_number,
            )
        )
    except Exception as exc:
        logger.error("dial_whatsapp_call failed: %s", exc)
        raise WhatsAppError(f"Could not dial {to_number[-4:]}: {exc}") from exc
    finally:
        await lkapi.aclose()

    logger.info("Dialled ***%s into room %s", to_number[-4:], room_name)
    return result


async def connect_whatsapp_call(*, call_id: str, sdp: str, sdp_type: str = "answer") -> Any:
    """Relay Meta's SDP answer so media can flow on an outbound call."""
    lkapi = _client()
    try:
        return await lkapi.connector.connect_whatsapp_call(
            api.ConnectWhatsAppCallRequest(
                whatsapp_call_id=call_id, sdp=_session_description(sdp, sdp_type)
            )
        )
    except Exception as exc:
        logger.error("connect_whatsapp_call failed: %s", exc)
        raise WhatsAppError(f"Could not connect call {call_id}: {exc}") from exc
    finally:
        await lkapi.aclose()


# ----------------------------------------------------------------------
# Inbound — the path that works from a Nigerian number
# ----------------------------------------------------------------------


async def accept_whatsapp_call(
    *,
    call_id: str,
    sdp: str,
    room_name: str,
    sdp_type: str = "offer",
    agent_name: str | None = None,
    agent_metadata: str = "",
    participant_identity: str | None = None,
) -> Any:
    """
    Answer a call the customer placed to us.

    The agent is dispatched through this request rather than separately, so
    LiveKit brings the worker into the room as part of accepting the call.
    """
    agents = (
        [api.RoomAgentDispatch(agent_name=agent_name, metadata=agent_metadata)]
        if agent_name
        else []
    )

    lkapi = _client()
    try:
        result = await lkapi.connector.accept_whatsapp_call(
            api.AcceptWhatsAppCallRequest(
                whatsapp_phone_number_id=settings.whatsapp.phone_number_id,
                whatsapp_api_key=settings.whatsapp.token,
                whatsapp_cloud_api_version=_cloud_api_version(),
                whatsapp_call_id=call_id,
                sdp=_session_description(sdp, sdp_type),
                room_name=room_name,
                agents=agents,
                participant_identity=participant_identity or "",
            )
        )
    except Exception as exc:
        logger.error("accept_whatsapp_call failed: %s", exc)
        raise WhatsAppError(f"Could not accept call {call_id}: {exc}") from exc
    finally:
        await lkapi.aclose()

    logger.info("Accepted call %s into room %s", call_id[-12:], room_name)
    return result


async def disconnect_whatsapp_call(*, call_id: str, reason: str = "COMPLETED") -> Any:
    """End a call from our side."""
    lkapi = _client()
    try:
        return await lkapi.connector.disconnect_whatsapp_call(
            api.DisconnectWhatsAppCallRequest(
                whatsapp_call_id=call_id,
                whatsapp_api_key=settings.whatsapp.token,
                disconnect_reason=reason,
            )
        )
    except Exception as exc:
        logger.warning("disconnect_whatsapp_call failed (call may already be over): %s", exc)
        return None
    finally:
        await lkapi.aclose()


# ----------------------------------------------------------------------
# Meta webhook parsing
# ----------------------------------------------------------------------


def parse_call_events(body: dict) -> list[dict]:
    """
    Pull call events out of a Meta webhook payload.

    Meta nests everything as entry[].changes[].value with a per-field shape, so
    this flattens to the handful of things we act on. Confirmed against live
    payloads: the SDP lives at calls[].session.sdp.
    """
    events: list[dict] = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "calls":
                continue
            value = change.get("value", {})
            for call in value.get("calls", []):
                session = call.get("session") or {}
                events.append(
                    {
                        "call_id": call.get("id"),
                        "event": call.get("event"),
                        "direction": call.get("direction"),
                        "from": call.get("from"),
                        "to": call.get("to"),
                        "status": call.get("status"),
                        "sdp": session.get("sdp"),
                        "sdp_type": session.get("sdp_type"),
                        "timestamp": call.get("timestamp"),
                    }
                )
    return events


def verify_webhook_subscription(mode: str, token: str, challenge: str) -> str:
    """
    Answer Meta's GET verification handshake.

    Meta calls this once when the webhook URL is registered, and delivers no
    events until the challenge comes back.
    """
    if mode == "subscribe" and token == settings.whatsapp.verify_token:
        logger.info("Meta webhook verification passed")
        return challenge
    logger.warning("Meta webhook verification FAILED (mode=%s)", mode)
    raise WhatsAppError("Webhook verification failed: token mismatch")


# ----------------------------------------------------------------------
# The inverted flow: message the customer, they call us
# ----------------------------------------------------------------------


# WhatsApp caps the call button label at 20 characters. The bank name is
# configurable, so this trims rather than trusting it to fit: a payload Meta
# rejects would cost the alert its button on the one message that matters.
CALL_BUTTON_MAX = 20


def _call_button_label() -> str:
    label = f"Call {settings.bank.name}"
    return label if len(label) <= CALL_BUTTON_MAX else "Call the bank"


async def send_fraud_alert_message(*, to_number: str, first_name: str, last4: str | None) -> dict:
    """
    Tell the customer something is wrong and ask them to call us.

    Needed because Meta blocks business-initiated calls from Nigerian numbers.
    It is also the better security posture: the customer initiates, so the call
    cannot be spoofed by someone impersonating the bank.

    The wording is deliberate: no links, and nothing secret requested. A fraud
    alert that behaves like a phishing message teaches exactly the wrong reflex.

    It does not tell them to ring the number on their card. This message comes
    from the bank's own verified WhatsApp number, so the number they should be
    calling is the one already in front of them. Sending them somewhere else adds
    a step during the minutes that matter, and implies this channel is the less
    trustworthy one when it is the verified one.

    The message carries a voice_call button so the customer taps once instead of
    hunting for the call icon. It falls back to plain text if the interactive
    send is refused: the button is a convenience, and an alert that never
    arrives is far worse than one without a button.
    """
    card_line = f" on your card ending {last4}" if last4 else ""
    text = (
        f"{settings.bank.written_name} security alert\n\n"
        f"Hello {first_name}, we have spotted a transaction{card_line} that does not "
        f"look like you, and we need to confirm it with you urgently.\n\n"
        f"Please tap the button below to speak to our automated security "
        f"assistant now. It can block the transaction and freeze your card if "
        f"this was not you.\n\n"
        f"We will never ask you for your PIN, your password or a one time code."
    )

    url = f"{settings.whatsapp.graph_url}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp.token}",
        "Content-Type": "application/json",
    }
    def envelope(extra: dict) -> dict:
        return {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            **extra,
        }

    interactive = envelope(
        {
            "type": "interactive",
            "interactive": {
                "type": "voice_call",
                "body": {"text": text},
                "action": {
                    "name": "voice_call",
                    "parameters": {
                        "display_text": _call_button_label(),
                        # Expire the button with the briefing behind it. A button
                        # that still works after the alert has gone reaches an
                        # agent that knows nothing about why they were called,
                        # which is worse than having no button at all.
                        "ttl_minutes": pending_alerts.DEFAULT_TTL_MINUTES,
                    },
                },
            },
        }
    )
    plain = envelope({"type": "text", "text": {"body": text}})

    async with aiohttp.ClientSession() as http:

        async def post(payload: dict) -> tuple[int, str]:
            async with http.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                return resp.status, await resp.text()

        status, body = await post(interactive)
        if status in (200, 201):
            logger.info("Fraud alert sent with call button to ***%s", to_number[-4:])
            return json.loads(body)

        # An interactive message needs calling enabled on the number and an open
        # service window; either can be missing. The alert itself still matters,
        # so send it plainly rather than losing it over a button.
        logger.warning(
            "Call-button message refused (%s), falling back to plain text: %s",
            status,
            body[:300],
        )
        status, body = await post(plain)
        if status not in (200, 201):
            logger.error("Fraud alert message failed %s: %s", status, body)
            raise WhatsAppError(f"Could not send alert message: {body}")

    logger.info("Fraud alert sent to ***%s (no button)", to_number[-4:])
    return json.loads(body)
