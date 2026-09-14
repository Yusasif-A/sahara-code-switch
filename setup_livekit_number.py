"""
Route a LiveKit phone number to the telco care agent.

LiveKit Cloud includes one free US local number on every plan, and it handles
inbound calls — which is exactly what a customer care line is. The fraud agent
stays on WhatsApp, because it needs to call *out* and LiveKit numbers do not
support outbound yet.

    WhatsApp number  ->  fraud agent   (outbound alert, inbound callback)
    LiveKit number   ->  telco agent   (inbound care line)

WHAT YOU DO IN THE DASHBOARD FIRST
Number purchase is not in the Python SDK, so claim the free number by hand:

    cloud.livekit.io -> Telephony -> Phone Numbers -> Rent a number
    (or, with the LiveKit CLI:
        lk number search --country-code US --area-code 415
        lk number purchase --numbers +1415XXXXXXX)

THEN RUN THIS
    python setup_livekit_number.py --create-rule
    # prints a dispatch rule id

Finally assign the number to that rule, again in the dashboard:
    Phone Numbers -> your number -> ... -> Assign dispatch rule

    (or: lk number update --id <NUMBER_ID> --sip-dispatch-rule-id <RULE_ID>)

Then `python telco_agent.py start` and ring the number.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from livekit import api

from config import settings

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

RULE_NAME = "telco-care-inbound"


def _client() -> api.LiveKitAPI:
    return api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )


async def create_rule(trunk_ids: list[str] | None = None) -> None:
    """
    Create a dispatch rule that puts every caller in their own room with the
    telco agent already in it.

    Individual rather than direct: each caller needs a private room, since two
    strangers ringing customer care must never land in the same conversation.
    """
    lkapi = _client()
    try:
        rule = await lkapi.sip.create_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                name=RULE_NAME,
                # Attaching the number here is what makes calls to it reach the
                # agent. A rule with no trunk is valid, creates cleanly, and
                # answers nothing — which is a miserable thing to debug.
                trunk_ids=trunk_ids or [],
                rule=api.SIPDispatchRule(
                    dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                        room_prefix="telco-in",
                    )
                ),
                room_config=api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name=settings.telco.agent_name,
                            metadata="{}",
                        )
                    ]
                ),
            )
        )
    except Exception as exc:
        print(f"\n  Could not create the dispatch rule: {exc}\n")
        sys.exit(1)
    finally:
        await lkapi.aclose()

    print(f"\n  Created dispatch rule: {rule.sip_dispatch_rule_id}")
    print(f"  Agent               : {settings.telco.agent_name}")
    print(f"  Rooms               : telco-in-*\n")
    print("  Now assign your number to it:")
    print("    cloud.livekit.io -> Telephony -> Phone Numbers -> your number")
    print("    -> ... -> Assign dispatch rule -> pick this rule -> Save\n")
    print(f"    or: lk number update --id <NUMBER_ID> "
          f"--sip-dispatch-rule-id {rule.sip_dispatch_rule_id}\n")


async def list_rules() -> None:
    lkapi = _client()
    try:
        result = await lkapi.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
    finally:
        await lkapi.aclose()

    if not result.items:
        print("\n  No dispatch rules. Run --create-rule\n")
        return

    print("\n  Dispatch rules:\n")
    for rule in result.items:
        agents = []
        if rule.room_config and rule.room_config.agents:
            agents = [a.agent_name for a in rule.room_config.agents]
        print(f"    {rule.sip_dispatch_rule_id}  {rule.name or '(unnamed)'}")
        print(f"      agents  : {', '.join(agents) or 'none — calls will not be answered'}")
        if rule.inbound_numbers:
            print(f"      numbers : {', '.join(rule.inbound_numbers)}")
        print()


async def delete_rule(rule_id: str) -> None:
    lkapi = _client()
    try:
        await lkapi.sip.delete_dispatch_rule(
            api.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=rule_id)
        )
        print(f"\n  Deleted {rule_id}\n")
    except Exception as exc:
        print(f"\n  Could not delete: {exc}\n")
        sys.exit(1)
    finally:
        await lkapi.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route a LiveKit phone number to the telco care agent."
    )
    parser.add_argument("--create-rule", action="store_true", help="create the dispatch rule")
    parser.add_argument(
        "--trunk",
        action="append",
        metavar="ID",
        help="phone number / trunk id to attach, e.g. PN_xxxx (repeatable)",
    )
    parser.add_argument("--list", action="store_true", help="list existing dispatch rules")
    parser.add_argument("--delete", metavar="RULE_ID", help="delete a dispatch rule")
    args = parser.parse_args()

    if not any([args.create_rule, args.list, args.delete]):
        parser.print_help()
        sys.exit(0)

    async def run() -> None:
        if args.create_rule:
            await create_rule(args.trunk)
        if args.list:
            await list_rules()
        if args.delete:
            await delete_rule(args.delete)

    asyncio.run(run())


if __name__ == "__main__":
    main()
