"""
Unified WhatsApp Router
Routes incoming WhatsApp webhooks to different agent endpoints based on phone number.

Two changes from the original, both about call events rather than messages:

1. Call events are forwarded in the background and Meta is acknowledged
   immediately. A `calls` connect webhook is time-critical — Meta is holding the
   call open waiting for an SDP answer while the customer listens to ringing —
   and making that wait on a downstream agent means one slow agent behind any
   mapping can stall calls for every other number sharing this router.

2. The forwarded response's headers are no longer copied back to Meta. They
   include content-length and transfer-encoding describing the *downstream*
   body, which conflicts with what this router actually sends and can produce a
   truncated or malformed response.
"""

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unified WhatsApp Router")

# WhatsApp verification token (shared across all agents)
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Phone number to endpoint mapping
ROUTING_MAP = {
    os.getenv("CHOPBETA_PHONE_NUMBER"): os.getenv("CHOPBETA_ENDPOINT"),
    os.getenv("AGENT2_PHONE_NUMBER"): os.getenv("AGENT2_ENDPOINT"),
    os.getenv("AGENT3_PHONE_NUMBER"): os.getenv("AGENT3_ENDPOINT"),
    os.getenv("AGENT4_PHONE_NUMBER"): os.getenv("AGENT4_ENDPOINT"),
    os.getenv("AGENT5_PHONE_NUMBER"): os.getenv("AGENT5_ENDPOINT"),
}

# Remove None values (for unconfigured agents)
ROUTING_MAP = {k: v for k, v in ROUTING_MAP.items() if k and v}

# Calls are forwarded without waiting; messages keep the original behaviour of
# returning whatever the downstream agent replied.
CALL_FORWARD_TIMEOUT = 20.0
MESSAGE_FORWARD_TIMEOUT = 30.0

logger.info("=" * 60)
logger.info("🔀 Unified WhatsApp Router Started")
logger.info(f"📱 Routing {len(ROUTING_MAP)} WhatsApp number(s)")
for phone, endpoint in ROUTING_MAP.items():
    logger.info(f"   {phone} → {endpoint}")
logger.info("=" * 60)


def extract_phone_number(data: dict) -> str | None:
    """Extract the WhatsApp Business Phone Number ID from webhook payload"""
    try:
        # The phone_number_id identifies which WhatsApp number received the
        # event. Present on `messages` and `calls` alike, so this routes both.
        return data["entry"][0]["changes"][0]["value"]["metadata"]["phone_number_id"]
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to extract phone number ID: {e}")
        return None


def extract_fields(data: dict) -> list[str]:
    """Which webhook fields this payload carries — 'messages', 'calls', etc."""
    try:
        return [
            change.get("field")
            for entry in data.get("entry", [])
            for change in entry.get("changes", [])
        ]
    except (AttributeError, TypeError):
        return []


async def forward_in_background(endpoint: str, body: bytes, signature: str) -> None:
    """
    Deliver a call event without holding up the acknowledgement to Meta.

    Failures are logged rather than raised: by the time this runs Meta has
    already had its 200, so there is nobody left to report an error to, and a
    raised exception here would only surface as an unhandled task warning.
    """
    try:
        async with httpx.AsyncClient(timeout=CALL_FORWARD_TIMEOUT) as client:
            response = await client.post(
                endpoint,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": signature,
                },
            )
        logger.info(f"✅ Call event forwarded to {endpoint}: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Call event forward to {endpoint} failed: {e}")


@app.get("/whatsapp")
async def whatsapp_verify(request: Request):
    """
    Handle WhatsApp webhook verification (GET request)
    Facebook sends this to verify the webhook URL
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return Response(content=challenge, status_code=200)

    logger.warning("❌ Webhook verification failed")
    return Response(content="Verification token mismatch", status_code=403)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Handle incoming WhatsApp webhooks (POST request)
    Routes to the appropriate agent endpoint based on phone number.
    """
    try:
        body = await request.body()
        headers = dict(request.headers)
        signature = headers.get("x-hub-signature-256", "")

        data = await request.json()

        phone_number_id = extract_phone_number(data)
        if not phone_number_id:
            logger.error("❌ Could not extract phone number from webhook")
            return Response(status_code=200)  # ACK to Facebook anyway

        target_endpoint = ROUTING_MAP.get(phone_number_id)
        if not target_endpoint:
            logger.warning(f"⚠️ No route configured for phone number: {phone_number_id}")
            return Response(status_code=200)  # ACK to Facebook anyway

        fields = extract_fields(data)

        # --- Call events: acknowledge Meta now, deliver in the background ---
        if "calls" in fields:
            logger.info(f"📞 Routing call event: {phone_number_id} → {target_endpoint}")
            asyncio.create_task(forward_in_background(target_endpoint, body, signature))
            return Response(status_code=200)

        # --- Everything else: forward and relay the downstream reply ---
        logger.info(f"📨 Routing {fields or 'webhook'}: {phone_number_id} → {target_endpoint}")
        async with httpx.AsyncClient(timeout=MESSAGE_FORWARD_TIMEOUT) as client:
            response = await client.post(
                target_endpoint,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": signature,
                },
            )

        logger.info(f"✅ Forwarded to {target_endpoint}: {response.status_code}")
        # Status and body only. Copying the downstream headers back would send
        # its content-length and transfer-encoding, which describe a different
        # response than the one this router is actually returning.
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    except Exception as e:
        logger.error(f"❌ Router error: {e}", exc_info=True)
        return Response(status_code=200)  # Always ACK to Facebook


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "router": "unified-whatsapp",
        "configured_numbers": len(ROUTING_MAP),
        "routes": list(ROUTING_MAP.keys()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
