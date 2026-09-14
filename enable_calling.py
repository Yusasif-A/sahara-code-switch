"""
Turn on the WhatsApp Calling API for your business phone number.

A fresh WhatsApp Business number cannot place or receive calls until calling is
explicitly enabled on it. Until then every call attempt fails with:

    (#138000) Calling API not enabled. Enable calling for this phone number
    in WhatsApp Manager, or via the settings API

This script does it via the settings API, which is faster than clicking through
WhatsApp Manager and leaves a record of exactly what was set.

    python enable_calling.py --check     # show current settings
    python enable_calling.py --enable    # turn calling on
    python enable_calling.py --enable --callback-permission

ORDER OF OPERATIONS — this is the part that trips people up. Meta will refuse to
enable calling until the webhook is already live and subscribed:

    (#138018) WhatsApp Business calling cannot be enabled because technical
    pre-requisites are not met — you need to configure webhooks or set up SIP

So the sequence is webhook FIRST, calling second:

    1. uvicorn app:app --port 8000   and   ngrok http 8000
    2. Meta console: webhook URL = <ngrok>/webhook/whatsapp, subscribe to `calls`
    3. python enable_calling.py --enable
    4. python grant_permission.py --to +234...

Meta also wants the number's messaging limit at 2000/day or above before calling
functions. --check prints the tier so you can see where you stand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import aiohttp

from config import settings


async def _graph(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"https://graph.facebook.com/{settings.whatsapp.graph_version}/{path}"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp.token}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as http:
        async with http.request(
            method, url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            body = await resp.text()
            if resp.status not in (200, 201):
                print(f"\n  Meta returned {resp.status}:\n  {body}\n")
                sys.exit(1)
            return json.loads(body) if body else {}


async def list_numbers(waba_id: str) -> None:
    """
    Show every phone number on a WhatsApp Business Account.

    Useful when a new number has been added and you want its id and status
    without hunting through the console — and to confirm which numbers a token
    can actually see.
    """
    print(f"\nNumbers on WABA {waba_id}:\n")
    result = await _graph(
        "GET",
        f"{waba_id}/phone_numbers?fields=id,display_phone_number,verified_name,"
        "code_verification_status,quality_rating,platform_type,status",
    )
    entries = result.get("data", [])
    if not entries:
        print("  none visible to this token\n")
        return
    for entry in entries:
        marker = " <- in .env" if entry.get("id") == settings.whatsapp.phone_number_id else ""
        print(f"  {entry.get('display_phone_number','?')}  ({entry.get('verified_name','?')}){marker}")
        print(f"    id           : {entry.get('id')}")
        print(f"    status       : {entry.get('status', '?')}")
        print(f"    verification : {entry.get('code_verification_status', '?')}")
        print(f"    quality      : {entry.get('quality_rating', '?')}")
        print()


async def check(phone_id: str | None = None) -> None:
    """Show the number's current calling configuration and messaging limit."""
    phone_id = phone_id or settings.whatsapp.phone_number_id

    print(f"\nPhone number id: {phone_id}\n")

    profile = await _graph(
        "GET",
        f"{phone_id}?fields=display_phone_number,verified_name,quality_rating,"
        "messaging_limit_tier,code_verification_status,webhook_configuration,platform_type",
    )
    print("  Number       :", profile.get("display_phone_number", "?"))
    print("  Name         :", profile.get("verified_name", "?"))
    print("  Quality      :", profile.get("quality_rating", "?"))
    print("  Platform     :", profile.get("platform_type", "?"))

    verification = profile.get("code_verification_status", "?")
    print("  Verification :", verification)
    if verification != "VERIFIED":
        print(
            "    ^ not VERIFIED. This does NOT block enabling calling — that was\n"
            "      confirmed against a live EXPIRED number. Worth fixing anyway\n"
            "      (python verify_number.py --request), as other features gate on it."
        )

    webhook = (profile.get("webhook_configuration") or {}).get("application")
    print("  Webhook URL  :", webhook or "NOT SET")
    if not webhook:
        print("    ^ set this in the Meta console before calling can be enabled.")

    tier = profile.get("messaging_limit_tier", "unknown")
    print("  Messaging tier:", tier)
    if tier not in ("TIER_2K", "TIER_10K", "TIER_100K", "TIER_UNLIMITED"):
        print(
            "\n  WARNING: Meta requires a messaging limit of 2000/day or above for\n"
            "  calling to function. This number is below that, so calls may still\n"
            "  fail even after calling is enabled."
        )

    settings_response = await _graph("GET", f"{phone_id}/settings")
    calling = settings_response.get("calling", {})
    print("\n  Calling status:", calling.get("status", "NOT SET"))
    if calling:
        print("  Full calling config:")
        print("   ", json.dumps(calling, indent=2).replace("\n", "\n    "))

    if calling.get("status") in (None, "NOT_SET", "DISABLED"):
        print(
            "\n  If --enable returns 138018 ('technical pre-requisites are not met')\n"
            "  while the webhook URL above is set, the missing piece is the webhook\n"
            "  FIELD subscription. Verifying the callback URL is a separate thing from\n"
            "  subscribing to a field, and 138018 wants the field:\n\n"
            "    Meta App Dashboard -> your app -> WhatsApp -> Configuration\n"
            "    -> Webhooks -> Manage -> tick  calls\n\n"
            "  Having 'messages' ticked is not enough. This was the actual fix here.\n"
            "  The subscription cannot be read or set with a system user token (it\n"
            "  needs the app secret), so confirm it by eye in the console."
        )
    print()


async def enable(
    callback_permission: bool, show_call_icon: bool, phone_id: str | None = None
) -> None:
    """POST the minimal configuration that switches calling on."""
    phone_id = phone_id or settings.whatsapp.phone_number_id

    calling: dict = {"status": "ENABLED"}
    if callback_permission:
        # Prompts the user for callback permission when they call you and you
        # miss it — useful for a fraud desk, since the customer often calls back.
        calling["callback_permission_status"] = "ENABLED"
    if show_call_icon:
        # Puts the call icon in the chat so the customer can tap to call us.
        # This is what makes the inbound ("inverted") flow work: business-initiated
        # calls are blocked for Nigerian numbers, but user-initiated ones are not.
        calling["call_icon_visibility"] = "DEFAULT"

    print(f"\nEnabling calling on {phone_id} ...")
    print("  payload:", json.dumps({"calling": calling}))

    await _graph("POST", f"{phone_id}/settings", {"calling": calling})
    print("  accepted.")

    confirmed = await _graph("GET", f"{phone_id}/settings")
    status = confirmed.get("calling", {}).get("status", "UNKNOWN")
    print(f"\n  Calling status is now: {status}")

    if status == "ENABLED":
        print(
            "\n  Next:\n"
            "    1. message the business number from the handset\n"
            "    2. python grant_permission.py --to +234XXXXXXXXXX\n"
            "    3. tap Allow (choose 'Always')\n"
        )
    else:
        print("\n  Status is not ENABLED — re-run with --check and read the config.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enable the WhatsApp Calling API.")
    parser.add_argument("--check", action="store_true", help="show current settings")
    parser.add_argument(
        "--phone-number-id",
        help="check or enable this number instead of the one in .env",
    )
    parser.add_argument(
        "--list-numbers",
        metavar="WABA_ID",
        help="list every phone number on a WhatsApp Business Account",
    )
    parser.add_argument("--enable", action="store_true", help="turn calling on")
    parser.add_argument(
        "--callback-permission",
        action="store_true",
        help="also prompt users for callback permission (useful for a fraud desk)",
    )
    parser.add_argument(
        "--call-icon",
        action="store_true",
        help="show the call icon in chat so customers can tap to call in",
    )
    args = parser.parse_args()

    if not (args.check or args.enable or args.list_numbers):
        parser.print_help()
        sys.exit(0)

    problems = [p for p in settings.validate() if "WHATSAPP" in p]
    if problems:
        for problem in problems:
            print(f"  CONFIG: {problem}")
        sys.exit(1)

    async def run() -> None:
        if args.list_numbers:
            await list_numbers(args.list_numbers)
        if args.check:
            await check(args.phone_number_id)
        if args.enable:
            await enable(args.callback_permission, args.call_icon, args.phone_number_id)

    asyncio.run(run())


if __name__ == "__main__":
    main()
