"""
Re-verify ownership of the WhatsApp business phone number.

`enable_calling.py --check` reporting `Verification : EXPIRED` means Meta no
longer considers the number's ownership proven, and it gates features — calling
included — behind a verified number.

You can do this in WhatsApp Manager by clicking through Phone numbers -> verify,
but the Cloud API exposes the same two steps, which is faster and leaves the
result visible in --check straight away:

    POST /{phone_number_id}/request_code?code_method=SMS&language=en_US
    POST /{phone_number_id}/verify_code?code=123456

Usage:
    python verify_number.py --request                 # SMS code to the number
    python verify_number.py --request --method VOICE  # if SMS does not arrive
    python verify_number.py --code 123456             # submit what you received
    python verify_number.py --status                  # check where you stand

The code goes to +234 905 345 8146 itself — the business number, not your
personal handset. You need access to it to receive the SMS or the voice call.

A note on --register: after verification a Cloud API number must be registered
before it can send or receive. This script exposes it, but only run it if the
number is not already working, and be careful with the PIN — if two-step
verification was set up previously you must supply that same PIN, and repeated
wrong attempts lock the number for a period.
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
            method, url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=25)
        ) as resp:
            body = await resp.text()
            if resp.status not in (200, 201):
                try:
                    err = json.loads(body)["error"]
                    print(f"\n  Meta returned {resp.status}: {err.get('message')}")
                    if err.get("error_user_msg"):
                        print(f"  {err['error_user_msg']}")
                except Exception:
                    print(f"\n  Meta returned {resp.status}:\n  {body}")
                print()
                sys.exit(1)
            return json.loads(body) if body else {}


async def status() -> None:
    phone_id = settings.whatsapp.phone_number_id
    profile = await _graph(
        "GET",
        f"{phone_id}?fields=display_phone_number,verified_name,"
        "code_verification_status,quality_rating,status",
    )
    print(f"\n  Number       : {profile.get('display_phone_number', '?')}")
    print(f"  Name         : {profile.get('verified_name', '?')}")
    print(f"  Verification : {profile.get('code_verification_status', '?')}")
    print(f"  Status       : {profile.get('status', '?')}")
    print(f"  Quality      : {profile.get('quality_rating', '?')}\n")


async def request_code(method: str, language: str) -> None:
    phone_id = settings.whatsapp.phone_number_id
    print(f"\nRequesting a {method} verification code for {phone_id} ...")
    await _graph("POST", f"{phone_id}/request_code?code_method={method}&language={language}")
    print(
        "  Requested.\n\n"
        f"  A {method.lower()} with a 6 digit code is on its way to the BUSINESS number\n"
        "  (+234 905 345 8146), not your personal phone. When it arrives:\n\n"
        "      python verify_number.py --code 123456\n"
    )


async def verify_code(code: str) -> None:
    phone_id = settings.whatsapp.phone_number_id
    print(f"\nSubmitting code for {phone_id} ...")
    await _graph("POST", f"{phone_id}/verify_code?code={code}")
    print("  Accepted.")
    await status()
    print(
        "  If Verification now reads VERIFIED:\n"
        "      python enable_calling.py --enable\n"
    )


async def register(pin: str) -> None:
    phone_id = settings.whatsapp.phone_number_id
    print(f"\nRegistering {phone_id} for Cloud API ...")
    await _graph(
        "POST", f"{phone_id}/register", {"messaging_product": "whatsapp", "pin": pin}
    )
    print("  Registered.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-verify the WhatsApp business number.")
    parser.add_argument("--status", action="store_true", help="show verification status")
    parser.add_argument("--request", action="store_true", help="send a verification code")
    parser.add_argument(
        "--method", default="SMS", choices=["SMS", "VOICE"], help="how to deliver the code"
    )
    parser.add_argument("--language", default="en_US")
    parser.add_argument("--code", help="the 6 digit code you received")
    parser.add_argument(
        "--register",
        metavar="PIN",
        help="register for Cloud API with this 6 digit PIN (only if not already registered)",
    )
    args = parser.parse_args()

    if not any([args.status, args.request, args.code, args.register]):
        parser.print_help()
        sys.exit(0)

    problems = [p for p in settings.validate() if "WHATSAPP" in p]
    if problems:
        for problem in problems:
            print(f"  CONFIG: {problem}")
        sys.exit(1)

    async def run() -> None:
        if args.status:
            await status()
        if args.request:
            await request_code(args.method, args.language)
        if args.code:
            await verify_code(args.code)
        if args.register:
            await register(args.register)

    asyncio.run(run())


if __name__ == "__main__":
    main()
