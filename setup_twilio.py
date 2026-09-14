"""
Create the LiveKit side of a Twilio SIP trunk.

WHAT YOU DO IN TWILIO FIRST
---------------------------
1. Buy a phone number (Twilio Console > Phone Numbers > Manage > Buy a number).

   A Nigerian +234 caller ID is what you eventually want, but Twilio does not
   sell Nigerian mobile numbers to individuals, and Nigeria requires a
   regulatory bundle with an executed Letter of Authorization plus a real local
   address (no PO box, no virtual office). That is a business-account process
   measured in days.

   For testing, buy a US +1 number instead: it provisions instantly and needs no
   bundle. Outbound calls TO Nigerian numbers work fine from it — only *buying*
   a Nigerian number is restricted. The customer will see a foreign caller ID,
   which is fine for a technical test and worth fixing before any bank demo,
   since an unknown foreign number is exactly what a fraud victim distrusts.

2. Elastic SIP Trunking > Trunks > Create a trunk.

3. Termination tab (this is the outbound half — LiveKit calling out through Twilio):
     - Termination SIP URI: pick a unique subdomain,
       e.g. sahara-fraud.pstn.twilio.com
     - Authentication > Credential Lists > create one with a username and
       password. Note them down.

4. Origination tab (only needed if you want inbound calls too):
     - Origination URI: sip:<your-livekit-sip-host>;transport=tcp

5. Put these in .env:
     TWILIO_TERMINATION_URI=sahara-fraud.pstn.twilio.com
     TWILIO_SIP_USERNAME=...
     TWILIO_SIP_PASSWORD=...
     TWILIO_PHONE_NUMBER=+234...        # the caller ID the customer sees

THEN RUN
--------
    python setup_twilio.py --create-trunk
    # copy the printed SIP_OUTBOUND_TRUNK_ID into .env
    python setup_twilio.py --list

    python trigger.py FRD-CARD-FOREIGN --mode sip --to +2348020812523

The credentials must match on both sides exactly — a mismatch shows up as the
call simply failing to connect, with Twilio rejecting the INVITE.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from livekit import api

from config import settings


def _require(name: str, value: str) -> str:
    if not value:
        print(f"\n  {name} is not set in .env — see the notes at the top of this file.\n")
        sys.exit(1)
    return value


async def create_trunk() -> None:
    """Create the LiveKit outbound trunk that points at your Twilio termination URI."""
    uri = _require("TWILIO_TERMINATION_URI", settings.sip.termination_uri)
    username = _require("TWILIO_SIP_USERNAME", settings.sip.username)
    password = _require("TWILIO_SIP_PASSWORD", settings.sip.password)
    caller_id = _require("TWILIO_PHONE_NUMBER", settings.sip.caller_id)

    # Twilio expects a bare host here; a full sip: URI is a common paste error.
    address = uri.replace("sip:", "").strip("/")

    print(f"\nCreating LiveKit outbound trunk -> {address}")
    print(f"  caller id : {caller_id}")
    print(f"  username  : {username}")

    lkapi = api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )
    try:
        trunk = await lkapi.sip.create_sip_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(
                trunk=api.SIPOutboundTrunkInfo(
                    name=f"{settings.bank.name} fraud desk",
                    address=address,
                    numbers=[caller_id],
                    auth_username=username,
                    auth_password=password,
                )
            )
        )
    except Exception as exc:
        print(f"\n  Failed: {exc}\n")
        sys.exit(1)
    finally:
        await lkapi.aclose()

    print(f"\n  Created trunk: {trunk.sip_trunk_id}")
    print("\n  Add this line to .env:\n")
    print(f"      SIP_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}\n")


async def list_trunks() -> None:
    lkapi = api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )
    try:
        outbound = await lkapi.sip.list_sip_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        inbound = await lkapi.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())
    finally:
        await lkapi.aclose()

    print("\n  Outbound trunks:")
    for trunk in outbound.items:
        marker = "  <- in .env" if trunk.sip_trunk_id == settings.sip.outbound_trunk_id else ""
        print(f"    {trunk.sip_trunk_id}  {trunk.name}  -> {trunk.address}{marker}")
    if not outbound.items:
        print("    none — run --create-trunk")

    print("\n  Inbound trunks:")
    for trunk in inbound.items:
        print(f"    {trunk.sip_trunk_id}  {trunk.name}")
    if not inbound.items:
        print("    none")
    print()


async def test_call(to_number: str) -> None:
    """
    Dial a number through the trunk with no agent attached.

    This isolates the transport. If it rings, Twilio, the credentials and the
    trunk are all correct and any remaining silence is the agent's fault. If it
    fails, the SIP status tells you which prerequisite is missing — which is far
    easier to read here than tangled up in a full call.
    """
    if not settings.sip.configured:
        print("\n  SIP_OUTBOUND_TRUNK_ID is not set.\n")
        sys.exit(1)

    room_name = "sip-trunk-test"
    print(f"\nDialling {to_number} via {settings.sip.outbound_trunk_id}")
    print(f"  caller id: {settings.sip.caller_id}")
    print("  (no agent — you should hear silence when you answer)\n")

    lkapi = api.LiveKitAPI(
        settings.livekit.url, settings.livekit.api_key, settings.livekit.api_secret
    )
    try:
        await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, empty_timeout=60))
        participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=settings.sip.outbound_trunk_id,
                sip_call_to=to_number,
                room_name=room_name,
                participant_identity=to_number,
                wait_until_answered=True,
            )
        )
        print(f"  ANSWERED — participant {participant.participant_identity} joined.")
        print("  The trunk works. Any silence on a real call is the agent, not Twilio.\n")
    except Exception as exc:
        text = str(exc)
        print(f"  FAILED: {text}\n")
        # Twilio's numeric codes are the fastest route to the actual cause.
        # Twilio's numeric codes are exact where the SIP status is not — prefer
        # them. 32202 and 32100 both surfaced during setup here.
        if "32100" in text or "verified caller" in text.lower():
            print("  -> Authentication is working; Twilio is refusing the destination.\n")
            print("     First check the number is verified:")
            print("     https://console.twilio.com/us1/develop/phone-numbers/manage/verified\n")
            print("     IF IT IS ALREADY VERIFIED AND YOU STILL SEE THIS: you have hit")
            print("     the trial ceiling. Twilio's own documentation states that SIP")
            print("     trunking is only available after upgrading the account, and")
            print("     verifying the destination does not lift it — confirmed here")
            print("     against a verified number that still returned 32100.\n")
            print("     Upgrading (~$20) also removes the trial message Twilio plays")
            print("     before your agent speaks, which you need for any demo anyway.")
            print("     https://console.twilio.com/us1/billing/manage-billing/upgrade\n")
        elif "32202" in text:
            print("  -> Twilio rejected the credentials (error 32202).")
            print("     The username and password in .env must match the Credential")
            print("     List that is ATTACHED to the trunk's Termination tab.\n")
        elif "403" in text:
            # Twilio answers 403 for auth, geography and trial-verification
            # failures alike, so this cannot be narrowed from the SIP status
            # alone. Listed in the order they actually catch people out.
            print("  SIP 403 has three possible causes and Twilio does not")
            print("  distinguish them. Check in this order:\n")
            print("  1. CREDENTIAL LIST NOT ATTACHED (most common)")
            print("     Creating a Credential List is not the same as attaching it.")
            print("     Trunk > Termination > Authentication > Credential Lists —")
            print("     your list must actually appear there.\n")
            print("  2. GEOGRAPHIC PERMISSIONS")
            print("     Voice > Settings > Geographic Permissions > tick Nigeria.")
            print("     Off by default; calls to Nigeria are refused without it.\n")
            print("  3. UNVERIFIED NUMBER (trial accounts only)")
            print("     Phone Numbers > Manage > Verified Caller IDs > add the")
            print("     number you are dialling.\n")
            print("  The definitive answer is in Twilio's own log:")
            print("  https://console.twilio.com/us1/monitor/logs/calls\n")
        elif "401" in text or "auth" in text.lower():
            print("  -> Twilio rejected the credentials. The username and password in")
            print("     .env must match the Credential List attached to the trunk's")
            print("     Termination tab exactly.\n")
        elif "404" in text:
            print("  -> Twilio did not recognise the termination URI. Check")
            print(f"     TWILIO_TERMINATION_URI ({settings.sip.termination_uri}) matches")
            print("     the trunk's Termination SIP URI exactly.\n")
        sys.exit(1)
    finally:
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
        except Exception:
            pass
        await lkapi.aclose()


async def show_status() -> None:
    print(f"\n  LiveKit          : {settings.livekit.url}")
    print(f"  Trunk in .env    : {settings.sip.outbound_trunk_id or 'NOT SET'}")
    print(f"  Termination URI  : {settings.sip.termination_uri or 'NOT SET'}")
    print(f"  Caller ID        : {settings.sip.caller_id or 'NOT SET'}")
    print(f"  Ready for SIP    : {'yes' if settings.sip.configured else 'no'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the LiveKit side of a Twilio SIP trunk.")
    parser.add_argument("--create-trunk", action="store_true", help="create the outbound trunk")
    parser.add_argument("--list", action="store_true", help="list existing trunks")
    parser.add_argument("--status", action="store_true", help="show current configuration")
    parser.add_argument(
        "--test-call",
        metavar="NUMBER",
        help="dial a number through the trunk with no agent, to test the transport",
    )
    args = parser.parse_args()

    if not any([args.create_trunk, args.list, args.status, args.test_call]):
        parser.print_help()
        sys.exit(0)

    async def run() -> None:
        if args.status:
            await show_status()
        if args.create_trunk:
            await create_trunk()
        if args.list:
            await list_trunks()
        if args.test_call:
            await test_call(args.test_call)

    asyncio.run(run())


if __name__ == "__main__":
    main()
