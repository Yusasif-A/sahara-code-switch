"""
Ask a WhatsApp user for permission to call them.

WhatsApp does not let a business cold-call anyone. Before the fraud agent can
dial, the customer must have granted call permission, which they do by tapping
a button on a request the business sends them.

The rules that matter (Meta, WhatsApp Cloud API):
  - the user may grant temporary permission (7 calendar days) or permanent
  - you may send at most 1 permission request per 24 hours per user, and 2 per
    7 days — so do not spam this script while debugging
  - a free-form request only works inside an open customer service window, i.e.
    the user messaged you in the last 24 hours; outside that you need an
    approved template with a call permission button
  - 4 consecutive unanswered calls automatically revokes a granted permission

Usage:
    python grant_permission.py --to +2348020812523 --open-window
    python grant_permission.py --to +2348020812523

Do --open-window only if you have NOT messaged the business number from that
handset. The cleanest path is: message the business number from the phone first,
then run this without --open-window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import aiohttp

from config import settings


async def _graph_post(payload: dict) -> dict:
    url = f"{settings.whatsapp.graph_url}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp.token}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as http:
        async with http.post(
            url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            body = await resp.text()
            if resp.status not in (200, 201):
                print(f"\n  Meta returned {resp.status}:\n  {body}\n")
                sys.exit(1)
            return json.loads(body) if body else {}


async def open_window(to: str) -> None:
    """
    Send a plain text message.

    This does NOT open a customer service window on its own — only the user
    messaging you does that. It is here so you can check the token, the phone
    number id and the recipient are all wired up before dealing with permissions.
    """
    print(f"\nSending a test message to {to} ...")
    result = await _graph_post(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "body": (
                    f"{settings.bank.name} security check. Reply to this message so we "
                    f"can reach you if we spot anything unusual on your account."
                )
            },
        }
    )
    print(f"  sent: {result.get('messages', [{}])[0].get('id', '?')}")
    print("\n  Now REPLY from the handset — that is what opens the 24 hour window.\n")


async def request_call_permission(to: str) -> None:
    """Send the interactive call permission request the user taps to approve."""
    print(f"\nRequesting call permission from {to} ...")
    result = await _graph_post(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "call_permission_request",
                "body": {
                    "text": (
                        f"{settings.bank.name} may need to call you urgently if we detect "
                        f"a suspicious transaction on your account. Allow calls so we can reach "
                        f"you in time."
                    )
                },
                "action": {"name": "call_permission_request"},
            },
        }
    )
    message_id = result.get("messages", [{}])[0].get("id", "?")
    print(f"  sent: {message_id}")
    print(
        "\n  On the handset, tap Allow. Choose 'Always' for testing so it does not\n"
        "  expire after 7 days and burn your 2-requests-per-week budget.\n"
        "  Then: python trigger.py FRD-CARD-FOREIGN --to " + to + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a WhatsApp user for call permission.")
    parser.add_argument("--to", required=True, help="E.164 number, e.g. +2348020812523")
    parser.add_argument(
        "--open-window",
        action="store_true",
        help="also send a plain text message first (to sanity-check credentials)",
    )
    args = parser.parse_args()

    problems = settings.validate()
    whatsapp_problems = [p for p in problems if "WHATSAPP" in p]
    if whatsapp_problems:
        for problem in whatsapp_problems:
            print(f"  CONFIG: {problem}")
        sys.exit(1)

    async def run() -> None:
        if args.open_window:
            await open_window(args.to)
        await request_call_permission(args.to)

    asyncio.run(run())


if __name__ == "__main__":
    main()
