"""
Manually fire a fraud alert — the stand-in for the bank's detection engine.

In production the bank's fraud system posts to /fraud-alert the moment it scores
a transaction as suspicious. For testing we skip the detection step and assert
that a signal has already fired, which is what this script does.

Usage:
    python trigger.py                                   # list the scenarios
    python trigger.py FRD-CARD-FOREIGN                  # call the customer on file
    python trigger.py FRD-CARD-FOREIGN --to +2348020812523
    python trigger.py --cases                           # what the agent did afterwards

app.py must already be running (uvicorn app:app --port 8000).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_SERVER = "http://localhost:8000"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.load(resp)


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        return json.load(resp)


def list_signals(server: str) -> None:
    signals = _get(f"{server}/bank/signals")["signals"]
    print("\nAvailable fraud scenarios:\n")
    for signal in signals:
        print(f"  {signal['signalId']:<24} [{signal['severity']}]")
        print(f"  {'':24} {signal['summary']}\n")
    print("Fire one with:  python trigger.py <SIGNAL_ID> --to +234XXXXXXXXXX\n")


def show_cases(server: str) -> None:
    cases = _get(f"{server}/bank/cases")["cases"]
    if not cases:
        print("\nNo cases yet — no call has completed.\n")
        return
    print("\nFraud cases opened by the agent:\n")
    for case in cases:
        print(f"  {case['caseId']}  {case['status']}")
        print(f"    signal  : {case['signalId']}")
        print(f"    actions : {', '.join(case['actionsTaken']) or 'none'}")
        print(f"    human   : {'yes' if case['needsHuman'] else 'no'}\n")


def fire(server: str, signal_id: str, to: str | None, mode: str) -> None:
    payload: dict = {"signal_id": signal_id, "mode": mode}
    if to:
        payload["override_number"] = to

    print(f"\nFiring {signal_id} [{mode}]" + (f" → {to}" if to else " → customer on file"))
    try:
        result = _post(f"{server}/fraud-alert", payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"\n  FAILED ({exc.code}): {detail}\n")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"\n  Cannot reach {server}: {exc.reason}")
        print("  Is app.py running?  uvicorn app:app --port 8000\n")
        sys.exit(1)

    if result.get("status") == "notified":
        print(f"\n  WhatsApp alert sent to {result['notified']}")
        print(f"  severity  : {result['severity']}")
        print(f"  answerable until: {result['answerableUntil'][11:19]} UTC")
        print(
            "\n  Now tap the call button in that WhatsApp chat. The agent picks up\n"
            "  already briefed on this signal.\n"
        )
    else:
        print(f"\n  Calling {result['to']}")
        print(f"  room     : {result['roomName']}")
        print(f"  severity : {result['severity']}")
        print("\n  Watch the agent worker terminal for the conversation.")
    print("  Afterwards:  python trigger.py --cases\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire a fraud alert to start a call.")
    parser.add_argument("signal_id", nargs="?", help="e.g. FRD-CARD-FOREIGN")
    parser.add_argument("--to", help="dial this number instead of the customer on file (testing only)")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--cases", action="store_true", help="show what the agent did")
    parser.add_argument(
        "--mode",
        default="notify",
        choices=["notify", "sip", "call"],
        help=(
            "sip: a real phone call through Twilio (works everywhere). "
            "notify: WhatsApp alert, customer calls back. "
            "call: direct WhatsApp dial (blocked for Nigerian business numbers)."
        ),
    )
    args = parser.parse_args()

    try:
        if args.cases:
            show_cases(args.server)
        elif not args.signal_id:
            list_signals(args.server)
        else:
            fire(args.server, args.signal_id, args.to, args.mode)
    except urllib.error.URLError as exc:
        print(f"\nCannot reach {args.server}: {exc.reason}")
        print("Is app.py running?  uvicorn app:app --port 8000\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
