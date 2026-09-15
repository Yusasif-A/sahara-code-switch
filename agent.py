"""
The fraud-response voice agent.

This is the LiveKit worker: it joins the room the outbound call lands in, is
briefed on the specific fraud signal that triggered the call, and talks the
customer through it.

The agent's authority is bounded in three places on purpose:
  - prompts.py tells it what it may and may not do
  - the tools below refuse to act before verification passes
  - bank_api.py refuses human-only operations regardless of what was said
A failure in any one of those does not by itself let the agent do something
irreversible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

import prompts
import tts_router
from bank_api import NotFound, PermissionDenied, bank
from config import settings
from stt_providers import build_stt
from verification import ChallengeFlow, mentions_forbidden_credential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    force=True,
)
logger = logging.getLogger("fraud_agent.agent")


def strip_channel_tokens(text: str) -> str:
    """
    Remove the model's reasoning-channel markup.

    gemma-4 emits harmony-style control tokens, and on a live call the customer
    heard the agent say "channel thought channel What is your favourite food".
    The real reply is whatever follows the last closing marker, so keep that and
    drop the scaffolding. Anything left over is stripped token by token, since
    reasoning leaking into a fraud call is worse than an odd clipped word.
    """
    if "<|" in text or "|>" in text:
        # Content after the final <channel|> is the actual utterance.
        tail = re.split(r"<\s*channel\s*\|\s*>", text)
        if len(tail) > 1:
            text = tail[-1]
        # Drop "<|channel>thought" style openers along with their channel label.
        text = re.sub(r"<\s*\|\s*channel\s*>\s*\w*", " ", text)
        # Any remaining control tokens of either orientation.
        text = re.sub(r"<\s*\|[^>]*>", " ", text)
        text = re.sub(r"<[^<>|]*\|\s*>", " ", text)
    return text


def sanitize_for_tts(text: str) -> str:
    """Strip control tokens and markdown so the TTS does not read them aloud."""
    text = strip_channel_tokens(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*{1,3}|_{1,3}", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class FraudResponseAgent(Agent):
    """Handles one outbound fraud-response call."""

    def __init__(
        self,
        *,
        signal: dict | None,
        customer: dict,
        account: dict,
        card: dict | None,
        transport: str = "whatsapp",
    ) -> None:
        # signal is None when the customer rang us with nothing flagged. That is
        # an ordinary call to the security line, not an error: most people who
        # phone their bank about a card have not been the subject of an alert.
        self.customer_initiated = signal is None
        self.signal = signal
        self.customer = customer
        self.account = account
        self.card = card

        self.challenge = ChallengeFlow(
            customer=customer,
            max_attempts=settings.bank.max_verification_attempts,
        )
        self.actions_taken: list[str] = []
        # One voice per language, built on first use. Without this a Yoruba
        # reply is read aloud by the English model and the caller hears noise.
        self._tts_router = (
            tts_router.LanguageRoutedTTS() if settings.tts.routing else None
        )
        self.handed_to_human = False
        # Anything the output guard refused to speak, for the audit trail.
        self.blocked_utterances: list[str] = []
        # Set once the caller is in the room. A warm transfer needs the identity
        # LiveKit actually knows them by, which is not the number on file.
        self.participant_identity: str | None = None
        self.transport = transport

        # The whole record goes to the model: accounts, cards and the security
        # questions this customer actually chose. Answers are the only thing
        # held back.
        accounts = bank.get_accounts_by_customer(customer["customerId"])
        cards = bank.get_cards_by_customer(customer["customerId"])
        briefing = (
            prompts.build_customer_initiated_briefing(customer, cards, accounts)
            if self.customer_initiated
            else prompts.build_risk_briefing(signal, customer, card, accounts, cards)
        )
        instructions = prompts.build_system_prompt(
            agent_name=settings.bank.agent_display_name,
            bank_name=settings.bank.name,
            risk_briefing=briefing,
            max_attempts=settings.bank.max_verification_attempts,
            customer_initiated=self.customer_initiated,
        )
        super().__init__(instructions=instructions, chat_ctx=ChatContext.empty())

        logger.info(
            "Agent briefed: signal=%s customer=%s card=%s",
            signal["signalId"] if signal else "none (customer-initiated)",
            customer["customerId"],
            card["cardId"] if card else "none",
        )

    # ------------------------------------------------------------------
    # Output guard
    # ------------------------------------------------------------------

    async def tts_node(self, input, model_settings):
        """
        Clean the text and refuse to speak any request for a credential.

        This buffers into sentences rather than passing chunks straight through.
        A forbidden phrase like "mother's maiden name" spans several chunks, so
        per-chunk checking cannot catch it — and on a live call this agent ran
        out of its listed questions, invented one, and asked the customer for
        exactly that. The prompt forbids it and the model did it anyway.

        An offending sentence is replaced, not merely logged. The small added
        latency of holding a sentence is a trivial price against an AI that
        phones bank customers and asks them for account-recovery answers.
        """

        async def guarded():
            buffer = ""

            def vet(raw: str) -> str:
                sentence = sanitize_for_tts(raw)
                term = mentions_forbidden_credential(sentence)
                if term is not None:
                    logger.error(
                        "GUARD BLOCKED: agent asked for %s in %r", term, sentence
                    )
                    self.blocked_utterances.append(sentence)
                    return prompts.CREDENTIAL_REQUEST_BLOCKED

                return sentence

            async for chunk in input:
                # Concatenate raw. Cleaning each chunk cannot work: the model
                # emits "<|channel>thought<channel|>" split across several of
                # them, so no single chunk contains the pattern and the markup
                # reached the caller, who heard the agent say "thought" before
                # every sentence. The old " " between chunks also split words
                # in half. Both go away by treating the whole sentence as the
                # unit, which is what the credential guard already does.
                buffer += chunk
                # Emit on sentence boundaries so each is vetted whole.
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
    # Verification
    # ------------------------------------------------------------------

    @function_tool()
    async def ask_security_question(self, ctx: RunContext) -> str:
        """
        Get the security question to ask this customer.

        Call this when you are ready to verify the customer. It returns one
        question that the customer themselves chose when they opened the account.
        Ask it to them word for word, then pass their reply to
        check_security_answer.
        """
        if self.challenge.verified:
            return "This customer is already verified. Do not ask again."

        # A question is already on the table. Hand back the same one rather than
        # drawing another: rotating mid-answer means check_security_answer
        # grades their reply against a question they were never asked.
        if self.challenge.pending_unanswered:
            return (
                "You have already asked this customer a question and they have "
                "not answered it yet. Ask this one, word for word, and do not "
                f"invent a different one: {self.challenge.pending_question}"
            )
        try:
            question = self.challenge.next_question()
        except RuntimeError as exc:
            logger.error("Cannot verify: %s", exc)
            return (
                "No security question is on file for this customer, so they cannot "
                "be verified on this call. Call transfer_to_human_agent now."
            )
        return f"Ask the customer this question, word for word: {question}"

    @function_tool()
    async def check_security_answer(
        self, ctx: RunContext, answer: str, matches: bool
    ) -> str:
        """
        Record whether the customer answered their security question correctly.

        You decide whether it matched. You are reading a speech transcript, so
        allow for it being misheard: judge what they clearly meant, not whether
        the spelling lines up. Only call this once you have decided.

        Args:
            answer: exactly what the customer said, as you heard it.
            matches: true if that is the answer on file, allowing for
                transcription errors; false if they said something genuinely
                different.
        """
        if self.challenge.verified:
            return "Already verified."

        if self.challenge.record(matches, answer):
            return (
                "Correct. The customer is now verified. You may explain what was seen "
                "on the account and offer a protective step."
            )

        if self.challenge.exhausted:
            return (
                "That was wrong and there are no attempts left. Do not ask again and "
                "do not hint at the answer. Say the following and then call "
                f"transfer_to_human_agent: {prompts.VERIFICATION_FAILED}"
            )
        return (
            f"Recorded as wrong. {self.challenge.attempts_remaining} attempt(s) "
            "remain. Ask a DIFFERENT question from the customer record. Do not "
            "reveal or hint at any answer."
        )

    # ------------------------------------------------------------------
    # Explaining the risk
    # ------------------------------------------------------------------

    @function_tool()
    async def describe_suspicious_activity(self, ctx: RunContext) -> str:
        """
        Get the specific transactions that triggered this call, so you can
        describe them to the customer. Safe to call before verification, but only
        describe them once the customer is verified.
        """
        if self.signal is None:
            return (
                "There is no alert on this account and no transactions were "
                "flagged. Do not describe anything. Ask the customer what they "
                "need instead."
            )
        refs = self.signal.get("transactionRefs", [])
        described = []
        for ref in refs:
            try:
                txn = bank.get_transaction(ref)
            except NotFound:
                continue
            described.append(
                f"{txn['amount']} Naira, {txn['debitOrCredit'].lower()}, via {txn['channel']}, "
                f"described as '{txn['narration']}', country {txn.get('countryCode', 'NG')}"
            )
        if not described:
            return self.signal["summary"]
        return self.signal["summary"] + " The transactions are: " + "; ".join(described)

    # ------------------------------------------------------------------
    # Pre-approved protective actions
    # ------------------------------------------------------------------

    def _require_verified(self) -> str | None:
        if not self.challenge.verified:
            return (
                "The customer is not verified yet, so no action can be taken. "
                "Verify them first with ask_security_question."
            )
        return None

    @function_tool()
    async def freeze_card(self, ctx: RunContext, reason: str) -> str:
        """
        Put a temporary freeze on the customer's card. This is the main protective
        step and the bank has pre-approved it. It is reversible by the bank.

        Only call this after the customer is verified and has confirmed the
        transaction was not theirs.

        Args:
            reason: short reason, e.g. "customer does not recognise Ukraine purchases".
        """
        if blocked := self._require_verified():
            return blocked
        if not self.card:
            return "There is no card on this signal. Call transfer_to_human_agent."

        try:
            card = bank.block_card(self.card["cardId"], reason=reason, actor="agent")
        except (PermissionDenied, NotFound) as exc:
            logger.error("freeze_card refused: %s", exc)
            return f"That could not be done: {exc}. Call transfer_to_human_agent."

        self.actions_taken.append(f"froze card ending {card['last4']}")
        logger.info("Card %s frozen: %s", card["cardId"], reason)
        return (
            "The card is now frozen. Tell the customer, in your own words: "
            + prompts.CARD_FROZEN.format(last4=card["last4"])
        )

    @function_tool()
    async def reduce_card_limit(self, ctx: RunContext, new_daily_limit: str) -> str:
        """
        Lower the card's daily limit. Pre-approved, and useful when the customer
        wants to keep using the card but the bank wants the exposure capped. The
        limit can only be lowered, never raised.

        Args:
            new_daily_limit: the new limit in Naira, e.g. "50000.00".
        """
        if blocked := self._require_verified():
            return blocked
        if not self.card:
            return "There is no card on this signal. Call transfer_to_human_agent."

        try:
            card = bank.set_card_limit(self.card["cardId"], new_daily_limit, actor="agent")
        except PermissionDenied as exc:
            return f"{exc} {prompts.NOT_PREAPPROVED}"
        except NotFound as exc:
            return f"That could not be done: {exc}. Call transfer_to_human_agent."

        self.actions_taken.append(f"lowered daily limit to {new_daily_limit}")
        return f"The daily limit on the card ending {card['last4']} is now {new_daily_limit} Naira."

    @function_tool()
    async def flag_transaction(self, ctx: RunContext, reference_id: str, note: str) -> str:
        """
        Record that the customer disputes a specific transaction. This queues it
        for the fraud team. It does not move any money — you cannot promise the
        customer a refund.

        Args:
            reference_id: the transaction reference, e.g. "TRX-9003".
            note: what the customer said about it.
        """
        if blocked := self._require_verified():
            return blocked
        try:
            bank.flag_transaction(reference_id, note=note, actor="agent")
        except (PermissionDenied, NotFound) as exc:
            return f"That could not be done: {exc}."
        self.actions_taken.append(f"flagged {reference_id} as disputed")
        return (
            f"Transaction {reference_id} is recorded as disputed and will be reviewed by "
            "the fraud team. Do not tell the customer the money will definitely be returned."
        )

    # ------------------------------------------------------------------
    # Escalation and ending
    # ------------------------------------------------------------------

    @function_tool()
    async def transfer_to_human_agent(self, ctx: RunContext, reason: str) -> str:
        """
        Hand this call to the bank's human fraud team.

        Call this for anything irreversible (reversing money, blocking or closing
        the account, unfreezing anything, changing contact details), if
        verification failed, if the customer asks for a person or is distressed,
        or any time you are not certain.

        Args:
            reason: why you are transferring, for the human picking it up.
        """
        self.handed_to_human = True
        case = bank.create_fraud_case(
            customer_id=self.customer["customerId"],
            signal_id=self.signal["signalId"] if self.signal else None,
            summary=f"Transferred to human: {reason}",
            actions_taken=list(self.actions_taken),
            needs_human=True,
            actor="agent",
        )
        logger.info("Transferring to human. Case %s. Reason: %s", case["caseId"], reason)

        # A warm transfer is a SIP operation. On a WhatsApp call the customer is
        # not a SIP participant and there is nothing to REFER, so there is no
        # transfer to make. Do not promise a callback either: nobody is going to
        # ring them, and a promise the bank will not keep is worse than telling
        # them where to go. Send them to a branch, which is a thing they can act
        # on themselves.
        if self.transport != "sip" or not settings.sip.configured:
            logger.info(
                "No SIP leg to transfer (transport=%s); directing to a branch",
                self.transport,
            )
            return (
                f"Case {case['caseId']} is open with the fraud team. You cannot "
                "transfer this call and nobody will ring them back, so do not say "
                "either. Tell the customer that anything further has to be done at "
                "a branch: they should visit any of our branches with a valid I D "
                "and the team there will take it from the case now on file. Then "
                "thank them and call end_call."
            )

        job = get_job_context()
        try:
            await job.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job.room.name,
                    participant_identity=self.participant_identity
                    or self.customer["phoneNumber"],
                    transfer_to=f"tel:{settings.bank.fraud_desk_number}",
                    play_dialtone=True,
                )
            )
        except Exception as exc:
            logger.error("SIP transfer failed: %s", exc)
            return (
                f"The transfer did not go through, but case {case['caseId']} is open. "
                "Tell the customer a member of the fraud team will call them back "
                "shortly, then call end_call. Do not claim they are being put through."
            )

        return (
            f"Transfer started, case {case['caseId']}. Say this and then stop talking: "
            + prompts.TRANSFER_TO_HUMAN
        )

    @function_tool()
    async def end_call(self, ctx: RunContext) -> str:
        """
        End the call. Only call this once the protective step is done and the
        customer has no further questions.
        """
        if not self.handed_to_human:
            bank.create_fraud_case(
                customer_id=self.customer["customerId"],
                signal_id=self.signal["signalId"] if self.signal else None,
                summary=(
                    self.signal["summary"]
                    if self.signal
                    else "Customer-initiated call to the security line."
                ),
                actions_taken=list(self.actions_taken),
                needs_human=False,
                actor="agent",
            )
        logger.info("Ending call. Actions taken: %s", self.actions_taken or ["none"])

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
# Worker entrypoint
# ----------------------------------------------------------------------


class BankEnquiryAgent(Agent):
    """
    The bank line for a caller we cannot identify.

    One number answers both the telecom line and the bank, so anyone may ask
    for the bank - including a number that has no account here. Hanging up on
    them, or handing them to a queue, wastes a call that could still be useful:
    most questions a bank gets are general, and the warning never to give a
    P I N to a caller is worth more to a stranger than to a customer.

    Deliberately toolless apart from ending the call and going back. With no
    customer there is nothing to act on, and an agent holding tools it cannot
    legitimately use is an agent looking for a reason to use them.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=prompts.build_bank_enquiry_prompt(
                agent_name=settings.bank.agent_display_name,
                bank_name=settings.bank.name,
            ),
            chat_ctx=ChatContext.empty(),
        )
        self._tts_router = (
            tts_router.LanguageRoutedTTS() if settings.tts.routing else None
        )
        self.blocked_utterances: list[str] = []

    async def tts_node(self, input, model_settings):
        async def cleaned():
            buffer = ""

            def vet(raw: str) -> str:
                sentence = sanitize_for_tts(raw)
                term = mentions_forbidden_credential(sentence)
                if term is None:
                    return sentence
                logger.error("GUARD BLOCKED: %s in %r", term, sentence)
                self.blocked_utterances.append(sentence)
                return prompts.CREDENTIAL_REQUEST_BLOCKED

            async for chunk in input:
                buffer += chunk
                while (match := re.search(r"[.!?]\s", buffer)) is not None:
                    sentence, buffer = buffer[: match.end()], buffer[match.end() :]
                    if sentence.strip():
                        yield vet(sentence)
            if buffer.strip():
                yield vet(buffer)

        if self._tts_router is not None:
            async for frame in tts_router.speak_routed(self._tts_router, cleaned()):
                yield frame
            return
        async for frame in super().tts_node(cleaned(), model_settings):
            yield frame

    @function_tool()
    async def switch_to_telecom(self, ctx: RunContext) -> str:
        """
        Hand the caller back to the telecom care line.

        Call this if they say they actually wanted their phone line, or raise
        anything about data, airtime, recharge, network or S I M.
        """
        from telco_agent import build_care_agent

        logger.info("Bank enquiry -> telecom care line")
        return build_care_agent()

    @function_tool()
    async def end_call(self, ctx: RunContext) -> str:
        """End the call once the caller has no further questions."""
        logger.info("Ending bank enquiry call")
        if ctx.speech_handle is not None:
            await ctx.speech_handle.wait_for_playout()
        job = get_job_context()
        await job.api.room.delete_room(api.DeleteRoomRequest(room=job.room.name))
        return "Call ended."


class NoBriefing(Exception):
    """An inbound caller with no live fraud alert waiting for them."""


def prewarm(proc: JobProcess) -> None:
    # Tuned for telephony, not browser audio. WhatsApp delivers narrowband Opus
    # (maxplaybackrate=16000) which is quieter and band-limited, so the 0.6
    # activation threshold that suits a laptop mic misses the onset of speech —
    # that clipped the customer's first syllables ("amala" transcribed as
    # "i.m.") and stopped interruptions registering at all.
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.4,
        activation_threshold=0.45,
    )


def _resolve_call_context(metadata: str) -> tuple[dict | None, dict, dict, dict | None]:
    """
    Turn the dispatch metadata into the signal, customer, account and card.

    A real call always carries a signalId. Console mode does not, so it falls
    back to a scenario — that is what makes `python agent.py console` usable for
    rehearsing the conversation without any telephony.
    """
    data = json.loads(metadata) if metadata else {}
    signal_id = data.get("signalId")

    if not signal_id and data.get("noAlert"):
        # The customer rang us with nothing flagged. Identify them from caller ID
        # and run the ordinary security line: this is most calls a bank gets, and
        # sending them all to a queue throws away the thing the agent is for.
        # The briefing carries no incident, so the agent has nothing to invent.
        phone = data.get("phoneNumber", "")
        try:
            customer = bank.get_customer_by_phone(phone)
        except NotFound:
            # An unrecognised number cannot be verified against anything, so
            # there is no safe action to take. This is the one case that still
            # goes straight to a person.
            raise NoBriefing(phone or "unknown") from None
        accounts = bank.get_accounts_by_customer(customer["customerId"])
        cards = bank.get_cards_by_customer(customer["customerId"])
        return None, customer, accounts[0], (cards[0] if len(cards) == 1 else None)

    if not signal_id:
        signal_id = settings.demo_signal_id
        logger.warning(
            "No signalId in dispatch metadata — falling back to %s. "
            "Expected in console mode; a bug on a real call.",
            signal_id,
        )

    signal = bank.get_fraud_signal(signal_id)
    customer = bank.get_customer_by_id(signal["customerId"])
    account = bank.get_account(signal["accountNumber"])
    card = bank.get_card(signal["cardId"]) if signal.get("cardId") else None
    return signal, customer, account, card


async def _greet_and_hand_off(ctx: JobContext) -> None:
    """
    Answer an unbriefed inbound caller, say so plainly, and transfer.

    Deliberately minimal: no tools, no LLM turn. There is nothing for the agent
    to reason about, and an idle fraud agent talking to a real customer is a
    liability rather than a feature.
    """
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
        ),
        stt=openai.STT(
            base_url=settings.stt.base_url,
            model=settings.stt.model,
            language=settings.stt.language,
            api_key=settings.stt.api_key,
        ),
        tts=openai.TTS(
            base_url=settings.tts.base_url,
            model=settings.tts.model,
            voice=settings.tts.voice,
            api_key=settings.tts.api_key,
            response_format="wav",
        ),
    )
    try:
        await asyncio.wait_for(
            session.start(
                agent=Agent(instructions="Say only what you are told to say, then stop."),
                room=ctx.room,
                room_input_options=RoomInputOptions(audio_enabled=True, text_enabled=False),
                room_output_options=RoomOutputOptions(audio_enabled=True),
            ),
            timeout=45,
        )
    except asyncio.TimeoutError:
        logger.error("session.start() timed out; caller never joined room %s", ctx.room.name)
        return

    # Wait for the caller to actually be in the room. Speaking before they join
    # plays the greeting to nobody, and the customer hears silence — which is
    # exactly how this looked on a live call.
    try:
        await asyncio.wait_for(ctx.wait_for_participant(), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("Caller never joined the room; nothing to greet")
        return

    await session.say(
        prompts.UNKNOWN_CALLER.format(
            bank_name=settings.bank.name,
            agent_name=settings.bank.agent_display_name,
        ),
        allow_interruptions=False,
    )
    logger.info("Unknown caller told they have no account here")


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    await ctx.connect()

    inbound = bool(json.loads(ctx.job.metadata or "{}").get("inbound"))

    try:
        signal, customer, account, card = _resolve_call_context(ctx.job.metadata)
    except NoBriefing as exc:
        # Someone rang the fraud line with nothing pending. Greet them honestly
        # and hand over rather than making something up.
        logger.warning("Inbound call with no briefing from %s", exc)
        await _greet_and_hand_off(ctx)
        return

    logger.info("=" * 70)
    logger.info(
        "%s  room=%s",
        "SECURITY LINE (customer called)" if signal is None else "FRAUD CALL",
        ctx.room.name,
    )
    # Printed on every call because a worker registered to a different LiveKit
    # project than the dispatcher is invisible otherwise — the call simply rings.
    logger.info("LiveKit     %s", settings.livekit.url)
    if signal is None:
        logger.info("Signal      none - nothing flagged, customer rang us")
    else:
        logger.info(
            "Signal      %s (%s, %s)",
            signal["signalId"],
            signal["riskType"],
            signal["severity"],
        )
    logger.info("Customer    %s %s", customer["firstName"], customer["lastName"])
    logger.info("Account     %s", account["accountNumber"])
    logger.info("=" * 70)

    transport = json.loads(ctx.job.metadata or "{}").get("transport", "whatsapp")
    agent = FraudResponseAgent(
        signal=signal, customer=customer, account=account, card=card, transport=transport
    )

    # Step-by-step logging from here to the first spoken word. A silent hang
    # between "briefed" and "speaking" is indistinguishable from a dead agent,
    # and the customer just hears ringing while you guess which of LLM, STT, TTS
    # or session start is at fault.
    logger.info("Building session: llm=%s stt=%s tts=%s",
                settings.llm.base_url, settings.stt.provider, settings.tts.base_url)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # A customer who is frightened will talk over the agent, and should be
        # able to. Shorter than the 0.5s default so cutting in feels immediate.
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
        who = "customer" if getattr(item, "role", "") == "user" else "agent"
        logger.info("[%s] %s", who, text[:160])

    # Warm the voice while the room is still filling. Sahara costs ~9s on the
    # first sentence of a session and ~3s after; without this the opening line
    # pays it, and the caller hears silence at the exact moment they answer.
    if agent._tts_router is not None:
        asyncio.create_task(agent._tts_router.warm())

    logger.info("Session built. Starting in room %s ...", ctx.room.name)

    start = session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            text_enabled=False,
            # BVCTelephony, not BVC. BVC is tuned for wideband browser audio;
            # this call arrives as narrowband telephony (the WhatsApp SDP offers
            # Opus at maxplaybackrate=16000). Using the wrong model on a phone
            # stream degrades recognition and can stall the input pipeline.
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
        room_output_options=RoomOutputOptions(audio_enabled=True, transcription_enabled=True),
    )

    # session.start() links the agent's audio to a participant, so it blocks
    # while the room stays empty. A silent forever-block looked exactly like a
    # crashed agent; fail loudly with the likely cause instead.
    try:
        await asyncio.wait_for(start, timeout=45)
    except asyncio.TimeoutError:
        logger.error(
            "session.start() timed out after 45s — nothing ever joined room %s. "
            "The WhatsApp leg was accepted by LiveKit but never bridged. Check the "
            "AcceptWhatsAppCall payload carries whatsAppApiKey / whatsAppPhoneNumberId "
            "/ whatsAppCloudApiVersion; without them LiveKit answers OK and does nothing.",
            ctx.room.name,
        )
        return

    logger.info("Session started. Waiting for the caller to join the room ...")

    # Wait for the customer to actually be on the call before speaking, otherwise
    # the opening line plays to a ringing handset. Console mode has no remote
    # participant, so don't block forever waiting for one.
    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=20)
        agent.participant_identity = participant.identity
        logger.info("Caller joined: identity=%s kind=%s",
                    participant.identity, getattr(participant, "kind", "?"))
    except asyncio.TimeoutError:
        # The WhatsApp leg never bridged into the LiveKit room. Speak anyway —
        # if media does arrive late the customer still hears something rather
        # than dead air, and the log says plainly what happened.
        logger.warning(
            "No caller in room after 20s. AcceptWhatsAppCall returned OK but the "
            "media leg did not join. Speaking regardless."
        )

    line = prompts.build_opening_line(
        first_name=customer["firstName"],
        agent_name=settings.bank.agent_display_name,
        bank_name=settings.bank.name,
        inbound=inbound,
        # No alert means we must not open with "we have seen something on your
        # account". Ask what they need instead.
        customer_initiated=signal is None,
    )
    logger.info("Speaking opening line (%d chars) ...", len(line))
    await session.say(line, allow_interruptions=True)
    logger.info("Opening line delivered.")


    async def _write_audit() -> None:
        logger.info(
            "Call finished. verified=%s actions=%s human=%s",
            agent.challenge.verified,
            agent.actions_taken or ["none"],
            agent.handed_to_human,
        )

    ctx.add_shutdown_callback(_write_audit)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=settings.agent_name,
            # `agent.py start` runs in production mode, where this defaults to 8.
            # On an 8-core laptop that prewarms eight job runners, each loading
            # its own Silero VAD, and the machine saturates before a call even
            # arrives. The symptom is brutal to diagnose: VAD blocks the event
            # loop for ~30s, the worker misses LiveKit's websocket ping, the
            # server drops the connection, and you get a reconnect storm with
            # m-line mismatches that all looks like a network fault.
            #
            # One idle runner is plenty for a single concurrent call. Raise
            # AGENT_IDLE_PROCESSES on a real server with cores to spare.
            num_idle_processes=settings.agent_idle_processes,
            # 8081 by default; the telco agent uses 8082 so both can run.
            port=settings.agent_http_port,
        )
    )
