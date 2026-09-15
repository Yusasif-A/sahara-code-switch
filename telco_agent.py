"""
The telco care voice agent.

Second of the two agents in this project. Where the fraud agent calls the
customer, this one answers them — replacing the press-one-press-two IVR that
makes a subscriber categorise their own problem before anyone will listen.

Run it with:  python telco_agent.py console
              python telco_agent.py start
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
)
from livekit.agents.llm import ChatContext
from livekit.plugins import noise_cancellation, openai, silero

import telco_prompts
import tts_router
from agent import sanitize_for_tts
from config import settings
from stt_providers import build_stt
from telco_api import InsufficientBalance, NotFound, PermissionDenied, telco
from verification import mentions_forbidden_credential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    force=True,
)
logger = logging.getLogger("telco_agent.agent")


class TelcoCareAgent(Agent):
    """Handles one inbound customer care call."""

    def __init__(
        self, *, subscriber: dict, issue: dict | None, caller_msisdn: str = ""
    ) -> None:
        self.subscriber = subscriber
        # The number they dialled from. The bank side needs it to tell a
        # customer from a stranger, and caller ID is the only thing we have.
        self.caller_msisdn = caller_msisdn
        self.issue = issue
        self.actions_taken: list[str] = []
        # One voice per language, built on first use. Without this a Yoruba
        # reply is read aloud by the English model and the caller hears noise.
        self._tts_router = (
            tts_router.LanguageRoutedTTS() if settings.tts.routing else None
        )
        self.handed_to_human = False
        self.blocked_utterances: list[str] = []
        self.participant_identity: str | None = None
        # True until they tell us they are calling about a different line. Once
        # false the agent may read but not act, because caller ID is the only
        # thing proving this person owns the number.
        self.calling_from_affected_line = True

        instructions = telco_prompts.build_system_prompt(
            agent_name=settings.telco.agent_display_name,
            telco_name=settings.telco.name,
            subscriber_briefing=telco_prompts.build_subscriber_briefing(subscriber, issue),
        )
        super().__init__(instructions=instructions, chat_ctx=ChatContext.empty())

        logger.info(
            "Agent briefed: subscriber=%s network=%s issue=%s",
            subscriber["subscriberId"],
            subscriber["network"],
            issue["issueId"] if issue else "none stated",
        )

    # ------------------------------------------------------------------
    # Output guard — same design as the fraud agent
    # ------------------------------------------------------------------

    async def tts_node(self, input, model_settings):
        """Sentence-buffer the output and refuse to speak credential requests."""
        import re

        async def guarded():
            buffer = ""

            def vet(raw: str) -> str:
                sentence = sanitize_for_tts(raw)
                term = mentions_forbidden_credential(sentence)
                if term is None:
                    return sentence
                logger.error("GUARD BLOCKED: %r (forbidden: %s)", sentence, term)
                self.blocked_utterances.append(sentence)
                return (
                    "I am sorry, I should not have started to ask that. "
                    "We will never ask you for a P I N or a one time code."
                )

            async for chunk in input:
                # Concatenate raw. Cleaning each chunk cannot work: the model
                # emits "<|channel>thought<channel|>" split across several of
                # them, so no single chunk contains the pattern and the markup
                # reached the caller, who heard the agent say "thought" before
                # every sentence. The old " " between chunks also split words
                # in half. Both go away by treating the whole sentence as the
                # unit, which is what the credential guard already does.
                buffer += chunk
                while (match := re.search(r"[.!?]\s", buffer)) is not None:
                    sentence, buffer = buffer[: match.end()], buffer[match.end() :]
                    if sentence.strip():
                        yield vet(sentence)
            if buffer.strip():
                yield vet(buffer)

        if self._tts_router is not None:
            async for frame in tts_router.speak_routed(self._tts_router, guarded()):
                yield frame
            return

        async for frame in super().tts_node(guarded(), model_settings):
            yield frame

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def _require_own_line(self) -> str | None:
        """
        Refuse to change anything for a caller who is not on the affected line.

        Enforced here rather than trusted to the prompt, because "it is my
        sister's line, just credit it" is a plausible-sounding request that a
        model will want to help with.
        """
        if self.calling_from_affected_line:
            return None
        return (
            "They are not calling from this line, so you cannot change anything on "
            "it. Explain what you can see, then either ask them to call back from "
            "the affected line, or call transfer_to_human_agent so a person can "
            "verify who they are."
        )

    @function_tool()
    async def switch_to_bank(self, ctx: RunContext) -> str:
        """
        Hand the caller to the bank line.

        Call this as soon as they say bank, or raise a card, a transaction, a
        transfer, a debit from their account, or fraud. Do not ask why first.

        If their number is on the bank's books they get the full security line,
        which can verify them and freeze a card. If it is not, they get the
        general bank line, which answers questions but cannot touch an account.
        """
        from agent import BankEnquiryAgent, FraudResponseAgent
        from bank_api import NotFound, bank

        try:
            customer = bank.get_customer_by_phone(self.caller_msisdn)
        except (NotFound, Exception):
            logger.info("Telecom -> bank line (caller not a bank customer)")
            return BankEnquiryAgent()

        accounts = bank.get_accounts_by_customer(customer["customerId"])
        cards = bank.get_cards_by_customer(customer["customerId"])
        logger.info("Telecom -> bank line for %s", customer["customerId"])
        return FraudResponseAgent(
            signal=None,
            customer=customer,
            account=accounts[0],
            card=cards[0] if len(cards) == 1 else None,
            transport=self.transport if hasattr(self, "transport") else "whatsapp",
        )

    @function_tool()
    async def switch_to_line(self, ctx: RunContext, msisdn: str) -> str:
        """
        Look at a different number from the one that called.

        Use when they say they are NOT calling from the affected line — from a
        friend's phone, a second SIM, or the office, which is common precisely
        because the broken line cannot make calls.

        After this you can look at the account but not change it.

        Args:
            msisdn: the affected number, e.g. "+2348062345678" or "08062345678".
        """
        try:
            subscriber = telco.get_subscriber_by_msisdn(msisdn)
        except NotFound:
            return (
                f"No account found for that number. Read it back to them one digit "
                f"at a time to check you heard it correctly, and try again."
            )

        self.subscriber = subscriber
        self.issue = None
        self.calling_from_affected_line = False
        logger.info(
            "Switched to line %s (caller is on a different number — read only now)",
            subscriber["subscriberId"],
        )
        return (
            f"Now looking at {subscriber['firstName']} {subscriber['lastName']}'s line "
            f"on {subscriber['network']}, plan {subscriber['plan']}. "
            "Because they are not calling from this line you may explain what you "
            "find but must not change anything on the account."
        )

    @function_tool()
    async def diagnose(self, ctx: RunContext) -> str:
        """
        Work out what is actually wrong with this line.

        Call this as soon as the subscriber has described their problem — often
        before they finish. It reads the account and returns the likely causes
        in order, so you never have to ask them to categorise the fault.
        """
        result = telco.diagnose(self.subscriber["subscriberId"])
        lines = []
        for finding in result["findings"]:
            fixable = "you can fix this" if finding["agentCanFix"] else "needs a human"
            lines.append(f"{finding['cause']}: {finding['detail']} ({fixable})")
        logger.info("Diagnosis: %s", [f["cause"] for f in result["findings"]])
        return "Findings, most likely first: " + " | ".join(lines)

    @function_tool()
    async def check_balances(self, ctx: RunContext) -> str:
        """Get the subscriber's airtime and data balances."""
        airtime = telco.get_airtime_balance(self.subscriber["subscriberId"])
        bundles = telco.get_data_balance(self.subscriber["subscriberId"])
        parts = [f"Airtime: {airtime['airtimeBalance']} Naira on {airtime['plan']}."]
        if bundles:
            for bundle in bundles:
                parts.append(
                    f"{bundle['bundleName']}: {bundle['remainingGb']} G B left, "
                    f"status {bundle['status'].replace('_', ' ').lower()}."
                )
        else:
            parts.append("No data bundle on the line.")
        return " ".join(parts)

    @function_tool()
    async def check_recent_transactions(self, ctx: RunContext) -> str:
        """Get recent airtime and data transactions, to explain a deduction."""
        rows = telco.get_transactions(self.subscriber["subscriberId"], last_n=5)
        if not rows:
            return "No recent transactions on this line."
        return " | ".join(
            f"{t['reference']}: {t['amount']} Naira, {t['type'].replace('_',' ').lower()}, "
            f"'{t['narration']}', status {t['status'].replace('_',' ').lower()}"
            for t in rows
        )

    # ------------------------------------------------------------------
    # Fixes the agent may perform
    # ------------------------------------------------------------------

    @function_tool()
    async def restore_paid_bundle(self, ctx: RunContext) -> str:
        """
        Re-activate a bundle the subscriber already paid for but never received.

        Use this the moment diagnose reports BUNDLE_PAID_NOT_DELIVERED or
        BUNDLE_PENDING. They have already been charged, so there is nothing to
        confirm and nothing further to pay — just fix it.
        """
        if blocked := self._require_own_line():
            return blocked
        try:
            bundle = telco.retry_bundle_activation(
                self.subscriber["subscriberId"], actor="agent"
            )
        except NotFound:
            return "There is no pending bundle on this line, so this is not the problem."
        except PermissionDenied as exc:
            return f"{exc} Call transfer_to_human_agent."

        self.actions_taken.append(f"restored {bundle['bundleName']}")
        return "Done. Tell them: " + telco_prompts.BUNDLE_RESTORED.format(
            bundle=bundle["bundleName"]
        )

    @function_tool()
    async def credit_pending_recharge(self, ctx: RunContext) -> str:
        """
        Credit a recharge the subscriber paid for that never reached their line.

        Call this as soon as diagnose reports RECHARGE_NOT_CREDITED. This is the
        most common complaint on this line, the transaction record already
        proves they paid, and there is nothing to confirm — just fix it and tell
        them their new balance.

        Do not ask them for the voucher number, do not tell them to wait, and do
        not send them off to dial a U S S D code.
        """
        if blocked := self._require_own_line():
            return blocked
        try:
            result = telco.credit_pending_recharge(
                self.subscriber["subscriberId"], actor="agent"
            )
        except NotFound:
            return (
                "There is no uncredited recharge on this line. If they insist they "
                "paid, check recent transactions with them and, if you still cannot "
                "see it, call transfer_to_human_agent rather than arguing."
            )
        except PermissionDenied as exc:
            return f"{exc} Call transfer_to_human_agent."

        self.actions_taken.append(f"credited {result['amount']} Naira recharge")
        return (
            f"Credited. Tell them: their {result['amount']} Naira recharge has now "
            f"gone through and their balance is {result['newBalance']} Naira. "
            f"Reference {result['reference']}."
        )

    @function_tool()
    async def list_data_plans(self, ctx: RunContext) -> str:
        """Get the data bundles available to buy, with prices."""
        plans = telco.list_data_plans()
        return " | ".join(
            f"{p['planId']}: {p['name']}, {p['price']} Naira, {p['validityDays']} days"
            for p in plans
        )

    @function_tool()
    async def buy_data_bundle(self, ctx: RunContext, plan_id: str) -> str:
        """
        Buy a data bundle using the subscriber's airtime.

        ONLY call this after you have said the plan name and the exact price out
        loud and the subscriber has clearly agreed. This spends their money.

        Args:
            plan_id: the plan identifier, e.g. "DP-2".
        """
        if blocked := self._require_own_line():
            return blocked
        try:
            result = telco.buy_data_bundle(
                self.subscriber["subscriberId"], plan_id, actor="agent"
            )
        except InsufficientBalance as exc:
            balance = telco.get_airtime_balance(self.subscriber["subscriberId"])
            return (
                f"Not enough airtime. Tell them: {exc} "
                f"They have {balance['airtimeBalance']} Naira. Offer a cheaper bundle "
                "or suggest they recharge first. Do not retry this purchase."
            )
        except (NotFound, PermissionDenied) as exc:
            return f"That did not work: {exc}"

        self.actions_taken.append(f"bought {result['plan']} for {result['price']}")
        return (
            f"Bought {result['plan']} for {result['price']} Naira. Their balance is now "
            f"{result['newAirtimeBalance']} Naira. Reference {result['reference']}."
        )

    @function_tool()
    async def send_network_settings(self, ctx: RunContext) -> str:
        """
        Push internet and A P N settings to the handset by S M S. Harmless, and
        frequently the actual fix when data is active but nothing loads.
        """
        result = telco.send_network_settings(self.subscriber["subscriberId"], actor="agent")
        self.actions_taken.append("sent network settings")
        return (
            f"Settings sent to the {result['network']} line by S M S. Tell them to open "
            "the message and save the settings, then restart the phone."
        )

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    @function_tool()
    async def transfer_to_human_agent(self, ctx: RunContext, reason: str) -> str:
        """
        Hand this call to a human in the care team.

        Use for SIM swap or replacement (always, without exception), refunds,
        N I N linking, unbarring a line, porting out, changing registered
        details, plan migration, an angry or distressed subscriber, a request
        for a human, or anything you are unsure about.

        Args:
            reason: why you are transferring, for the person picking it up.
        """
        self.handed_to_human = True
        ticket = telco.raise_ticket(
            subscriber_id=self.subscriber["subscriberId"],
            summary=f"Transferred to care team: {reason}",
            actions_taken=list(self.actions_taken),
            needs_human=True,
            actor="agent",
        )
        logger.info("Transferring to human. Ticket %s. Reason: %s", ticket["ticketId"], reason)

        wants_sim_swap = any(
            term in reason.lower() for term in ("sim swap", "sim replacement", "new sim")
        )
        line = (
            telco_prompts.SIM_SWAP_REFUSED
            if wants_sim_swap
            else telco_prompts.TRANSFER_TO_HUMAN
        )
        return (
            f"Ticket {ticket['ticketId']} raised. Say this, then stop talking: {line}"
        )

    @function_tool()
    async def end_call(self, ctx: RunContext) -> str:
        """End the call once the problem is solved or handed over."""
        if not self.handed_to_human:
            telco.raise_ticket(
                subscriber_id=self.subscriber["subscriberId"],
                summary=self.issue["summary"] if self.issue else "Handled on call",
                actions_taken=list(self.actions_taken),
                needs_human=False,
                actor="agent",
            )
        logger.info("Ending call. Actions: %s", self.actions_taken or ["none"])
        job = get_job_context()
        # ctx.wait_for_playout() does not exist on RunContext in this version.
        # Calling it raised, the model saw the tool fail and called end_call
        # again, and the call ran to "maximum number of function calls steps
        # reached" while opening a fraud case every time round. The speech
        # handle is what knows when the audio has finished.
        if ctx.speech_handle is not None:
            await ctx.speech_handle.wait_for_playout()
        await job.api.room.delete_room(api.DeleteRoomRequest(room=job.room.name))
        return "Call ended."


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.4,
        activation_threshold=0.45,
    )


def _resolve_context(metadata: str) -> tuple[dict, dict | None]:
    """Work out who is calling and, if known, what about."""
    data = json.loads(metadata) if metadata else {}

    issue = None
    if issue_id := data.get("issueId"):
        issue = telco.get_issue(issue_id)
        subscriber = telco.get_subscriber(issue["subscriberId"])
        return subscriber, issue

    if msisdn := data.get("msisdn"):
        try:
            return telco.get_subscriber_by_msisdn(msisdn), None
        except NotFound:
            # A number that is not in the simulated subscriber base — which is
            # every real handset used for testing. Falling through to the demo
            # subscriber keeps the call answerable instead of crashing the job,
            # and a real deployment would look the caller up in the live HLR
            # rather than a fixture.
            logger.warning(
                "Caller ***%s is not in the demo subscriber base; using the demo account",
                "".join(c for c in msisdn if c.isdigit())[-4:],
            )

    # Console mode, or an inbound caller we could not identify.
    fallback = settings.telco.demo_issue_id
    logger.warning("Using demo issue %s", fallback)
    issue = telco.get_issue(fallback)
    return telco.get_subscriber(issue["subscriberId"]), issue


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    await ctx.connect()

    subscriber, issue = _resolve_context(ctx.job.metadata)

    logger.info("=" * 70)
    logger.info("TELCO CARE CALL  room=%s", ctx.room.name)
    logger.info("Subscriber  %s %s", subscriber["firstName"], subscriber["lastName"])
    logger.info("Network     %s (%s)", subscriber["network"], subscriber["plan"])
    logger.info("Issue       %s", issue["issueId"] if issue else "not yet stated")
    logger.info("STT         %s", settings.stt.provider)
    logger.info("=" * 70)

    agent = TelcoCareAgent(
        subscriber=subscriber,
        issue=issue,
        # Caller ID, so switch_to_bank can tell a bank customer from a
        # stranger without asking them to identify themselves twice.
        caller_msisdn=json.loads(ctx.job.metadata or "{}").get("msisdn", ""),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        allow_interruptions=True,
        min_interruption_duration=0.3,
        llm=openai.LLM(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
        ),
        stt=build_stt(vad=ctx.proc.userdata["vad"]),
        tts=openai.TTS(
            base_url=settings.tts.base_url,
            model=settings.tts.model,
            voice=settings.tts.voice,
            api_key=settings.tts.api_key,
            response_format="wav",
        ),
    )

    @session.on("conversation_item_added")
    def _log_turn(ev) -> None:
        item = getattr(ev, "item", None)
        text = (getattr(item, "text_content", None) or "").strip()
        if not text:
            return
        who = "subscriber" if getattr(item, "role", "") == "user" else "agent"
        logger.info("[%s] %s", who, text[:160])

    if agent._tts_router is not None:
        asyncio.create_task(agent._tts_router.warm())

    try:
        await asyncio.wait_for(
            session.start(
                agent=agent,
                room=ctx.room,
                room_input_options=RoomInputOptions(
                    audio_enabled=True,
                    text_enabled=False,
                    noise_cancellation=noise_cancellation.BVCTelephony(),
                ),
                room_output_options=RoomOutputOptions(
                    audio_enabled=True, transcription_enabled=True
                ),
            ),
            timeout=45,
        )
    except asyncio.TimeoutError:
        logger.error("session.start() timed out; nobody joined room %s", ctx.room.name)
        return

    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=20)
        agent.participant_identity = participant.identity
        logger.info("Caller joined: %s", participant.identity)
    except asyncio.TimeoutError:
        logger.warning("No caller in room after 20s; speaking anyway")

    await session.say(
        telco_prompts.build_opening_line(
            agent_name=settings.telco.agent_display_name,
            telco_name=settings.telco.name,
        ),
        allow_interruptions=True,
    )

    async def _log_outcome() -> None:
        logger.info(
            "Call finished. actions=%s human=%s blocked=%d",
            agent.actions_taken or ["none"],
            agent.handed_to_human,
            len(agent.blocked_utterances),
        )

    ctx.add_shutdown_callback(_log_outcome)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=settings.telco.agent_name,
            num_idle_processes=settings.agent_idle_processes,
            # 8082, so this can run alongside the fraud agent on 8081.
            port=settings.telco.http_port,
        )
    )


def build_care_agent(msisdn: str = "") -> TelcoCareAgent:
    """A care agent for a caller we know nothing about, used when the bank line hands back."""
    issue = telco.get_issue(settings.telco.demo_issue_id)
    return TelcoCareAgent(
        subscriber=telco.get_subscriber(issue["subscriberId"]),
        issue=issue,
        caller_msisdn=msisdn,
    )
