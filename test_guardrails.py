"""
Tests for the parts that must not regress.

The conversational quality of the agent is judged by listening to it. What is
tested here is the authority boundary and the verification flow — the things
where a silent regression would mean an AI doing something irreversible to a
real customer's account.

Run with:  pytest test_guardrails.py -v
"""

from __future__ import annotations

import json
import re

import pytest

import bank_data
import prompts
from bank_api import HUMAN_ONLY, NotFound, PermissionDenied, SimulatedBankAPI
from config import settings
from verification import (
    ChallengeFlow,
    answers_match,
    mentions_forbidden_credential,
)


@pytest.fixture
def bank() -> SimulatedBankAPI:
    """A fresh bank per test, so one test's freeze does not leak into another."""
    return SimulatedBankAPI()


@pytest.fixture
def customer() -> dict:
    return dict(bank_data.CUSTOMERS[0])


# ----------------------------------------------------------------------
# The authority boundary
# ----------------------------------------------------------------------


class TestAgentAuthority:
    def test_agent_may_freeze_a_card(self, bank: SimulatedBankAPI) -> None:
        card = bank.block_card("CRD-77001", reason="customer disputes", actor="agent")
        assert card["status"] == "BLOCKED"

    def test_agent_may_lower_a_card_limit(self, bank: SimulatedBankAPI) -> None:
        card = bank.set_card_limit("CRD-77001", "50000.00", actor="agent")
        assert card["dailyLimit"] == "50000.00"

    def test_agent_may_not_raise_a_card_limit(self, bank: SimulatedBankAPI) -> None:
        """Raising a limit mid-call is the classic social-engineering ask."""
        with pytest.raises(PermissionDenied):
            bank.set_card_limit("CRD-77001", "9000000.00", actor="agent")
        assert bank.get_card("CRD-77001")["dailyLimit"] == "500000.00"

    def test_agent_may_not_block_an_account(self, bank: SimulatedBankAPI) -> None:
        with pytest.raises(PermissionDenied):
            bank.block_account("0123456789", reason="fraud", actor="agent")
        assert bank.get_account("0123456789")["status"] == "ACTIVE"

    def test_agent_may_not_reverse_a_transaction(self, bank: SimulatedBankAPI) -> None:
        with pytest.raises(PermissionDenied):
            bank.reverse_transaction("TRX-9003", actor="agent")

    def test_agent_may_not_unblock_a_card(self, bank: SimulatedBankAPI) -> None:
        """Undoing containment is a human decision even though it is 'just' a toggle."""
        bank.block_card("CRD-77001", reason="fraud", actor="agent")
        with pytest.raises(PermissionDenied):
            bank.unblock_card("CRD-77001", actor="agent")
        assert bank.get_card("CRD-77001")["status"] == "BLOCKED"

    @pytest.mark.parametrize("operation", sorted(HUMAN_ONLY))
    def test_every_human_only_operation_refuses_the_agent(
        self, bank: SimulatedBankAPI, operation: str
    ) -> None:
        with pytest.raises(PermissionDenied):
            bank._guard(operation, "agent")

    def test_human_may_do_what_the_agent_may_not(self, bank: SimulatedBankAPI) -> None:
        account = bank.block_account("0123456789", reason="confirmed fraud", actor="human")
        assert account["status"] == "BLOCKED"

    def test_refusals_are_audited(self, bank: SimulatedBankAPI) -> None:
        """A refused attempt must leave a trace, not vanish."""
        with pytest.raises(PermissionDenied):
            bank.block_account("0123456789", reason="fraud", actor="agent")
        assert any(e["operation"].startswith("DENIED:") for e in bank.audit_log)


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------


class TestVerification:
    def test_correct_answer_verifies(self, customer: dict) -> None:
        flow = ChallengeFlow(customer=customer)
        question = flow.next_question()
        expected = next(
            q["answer"] for q in customer["securityQuestions"] if q["question"] == question
        )
        assert flow.check(expected) is True
        assert flow.verified is True

    def test_wrong_answers_exhaust_and_do_not_verify(self, customer: dict) -> None:
        flow = ChallengeFlow(customer=customer, max_attempts=3)
        for _ in range(3):
            flow.next_question()
            assert flow.check("definitely not the answer") is False
        assert flow.exhausted is True
        assert flow.verified is False

    def test_attempts_are_capped(self, customer: dict) -> None:
        flow = ChallengeFlow(customer=customer, max_attempts=2)
        flow.next_question()
        flow.check("wrong")
        assert flow.attempts_remaining == 1
        flow.next_question()
        flow.check("wrong")
        assert flow.attempts_remaining == 0

    def test_questions_vary_between_customers(self) -> None:
        """A shared question across all customers would be guessable."""
        first = {q["question"] for q in bank_data.CUSTOMERS[0]["securityQuestions"]}
        answers = [
            {q["answer"] for q in c["securityQuestions"]} for c in bank_data.CUSTOMERS
        ]
        assert first  # non-empty
        # No two customers share an identical answer set.
        assert len({frozenset(a) for a in answers}) == len(answers)

    def test_customer_with_no_questions_cannot_be_verified(self) -> None:
        flow = ChallengeFlow(customer={"customerId": "CUS-X", "securityQuestions": []})
        with pytest.raises(RuntimeError):
            flow.next_question()

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("Amala", "amala"),
            ("  amala ", "amala"),
            ("it's green", "green"),
            ("the colour green", "green"),
            ("Saint Mary's", "saint mary"),  # possessive from STT must not break it
            ("green please", "green"),  # filler words around the answer
        ],
    )
    def test_spoken_answers_match_stored_ones(self, given: str, expected: str) -> None:
        assert answers_match(given, expected) is True

    @pytest.mark.parametrize("given", ["blue", "", "   ", "rice"])
    def test_wrong_answers_do_not_match(self, given: str) -> None:
        assert answers_match(given, "amala") is False


# ----------------------------------------------------------------------
# Never ask for credentials
# ----------------------------------------------------------------------


class TestCredentialGuard:
    @pytest.mark.parametrize(
        "utterance",
        [
            "Can you confirm your PIN for me?",
            "Please read out the OTP you just received",
            "What is your card number?",
            "I need your BVN to continue",
            "Tell me your password",
        ],
    )
    def test_credential_requests_are_detected(self, utterance: str) -> None:
        assert mentions_forbidden_credential(utterance) is not None

    @pytest.mark.parametrize(
        "utterance",
        [
            "What is your favourite colour?",
            "Your card ending four zero eight one is now frozen.",
            "I will never ask you for a code.",  # 'code' alone is not a solicitation
        ],
    )
    def test_safe_utterances_pass(self, utterance: str) -> None:
        assert mentions_forbidden_credential(utterance) is None


# ----------------------------------------------------------------------
# Bank data integrity
# ----------------------------------------------------------------------


class TestBankData:
    def test_every_signal_resolves_to_real_records(self, bank: SimulatedBankAPI) -> None:
        for signal in bank.list_fraud_signals():
            bank.get_customer_by_id(signal["customerId"])
            bank.get_account(signal["accountNumber"])
            bank.get_card(signal["cardId"])
            for ref in signal["transactionRefs"]:
                bank.get_transaction(ref)

    def test_nuban_account_numbers_are_ten_digits(self) -> None:
        for account in bank_data.ACCOUNTS:
            assert len(account["accountNumber"]) == 10
            assert account["accountNumber"].isdigit()

    def test_bvns_are_eleven_digits(self) -> None:
        for customer in bank_data.CUSTOMERS:
            assert len(customer["bvn"]) == 11
            assert customer["bvn"].isdigit()

    def test_no_full_pan_is_stored(self) -> None:
        """Only BIN and last4 — a full card number must never sit in this repo."""
        for card in bank_data.CARDS:
            assert "*" in card["maskedPan"]
            assert len(card["last4"]) == 4

    def test_lookup_by_phone_finds_the_customer(self, bank: SimulatedBankAPI) -> None:
        found = bank.get_customer_by_phone("+2348031234567")
        assert found["customerId"] == "CUS-100001"

    def test_unknown_phone_raises(self, bank: SimulatedBankAPI) -> None:
        with pytest.raises(NotFound):
            bank.get_customer_by_phone("+2340000000000")

    def test_statement_is_newest_first(self, bank: SimulatedBankAPI) -> None:
        rows = bank.get_statement("0123456789", last_n=5)
        assert rows == sorted(rows, key=lambda t: t["bookDate"], reverse=True)


# ----------------------------------------------------------------------
# Meta webhook
# ----------------------------------------------------------------------


class TestWebhook:
    """
    Meta's console was pointed at /whatsapp while the app served only
    /webhook/whatsapp, which 404'd the subscription handshake. Both paths are
    served now, and these tests keep it that way — a 404 here means Meta silently
    stops delivering call events.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import app as app_module

        return TestClient(app_module.app)

    @pytest.mark.parametrize("path", ["/whatsapp", "/webhook/whatsapp"])
    def test_handshake_echoes_the_challenge(self, client, path: str) -> None:
        from config import settings

        response = client.get(
            path,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": settings.whatsapp.verify_token,
                "hub.challenge": "1265650108",
            },
        )
        assert response.status_code == 200
        assert response.text == "1265650108"

    @pytest.mark.parametrize("path", ["/whatsapp", "/webhook/whatsapp"])
    def test_wrong_verify_token_is_rejected(self, client, path: str) -> None:
        response = client.get(
            path,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "not-the-token",
                "hub.challenge": "x",
            },
        )
        assert response.status_code == 403

    @pytest.mark.parametrize("path", ["/whatsapp", "/webhook/whatsapp"])
    def test_post_accepts_events(self, client, path: str) -> None:
        assert client.post(path, json={"entry": []}).status_code == 200

    def test_unknown_payload_never_500s(self, client) -> None:
        """A non-200 makes Meta retry the whole batch, which is worse than a no-op."""
        assert client.post("/whatsapp", json={"nonsense": True}).status_code == 200


# ----------------------------------------------------------------------
# The inverted flow
# ----------------------------------------------------------------------


class TestPendingAlerts:
    """
    Meta blocks business-initiated calls from Nigerian numbers, so the customer
    calls us. These cover the bookkeeping that lets the agent know why.
    """

    @pytest.fixture
    def store(self):
        from pending_alerts import PendingAlertStore

        return PendingAlertStore(ttl_minutes=60)

    def test_alert_is_found_by_the_number_that_calls(self, store) -> None:
        store.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        assert store.get("+2348020812523").signal_id == "FRD-CARD-FOREIGN"

    @pytest.mark.parametrize(
        "caller",
        ["+2348020812523", "2348020812523", "08020812523", "234 802 081 2523"],
    )
    def test_number_formats_all_match(self, store, caller: str) -> None:
        """Meta hands back numbers in several shapes; all must find the alert."""
        store.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        assert store.get(caller) is not None

    def test_unknown_caller_has_no_alert(self, store) -> None:
        assert store.get("+2349999999999") is None

    def test_expired_alert_is_not_served(self) -> None:
        """A stale fraud briefing should reach a human, not a scripted agent."""
        from pending_alerts import PendingAlertStore

        store = PendingAlertStore(ttl_minutes=0)
        store.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        assert store.get("+2348020812523") is None

    def test_newer_signal_replaces_older(self, store) -> None:
        for signal_id in ("FRD-CARD-FOREIGN", "FRD-ATM-VELOCITY"):
            store.raise_alert(
                signal_id=signal_id, customer_id="CUS-100001", phone_number="+2348020812523"
            )
        assert store.get("+2348020812523").signal_id == "FRD-ATM-VELOCITY"

    def test_cleared_alert_is_gone(self, store) -> None:
        store.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        store.clear("+2348020812523")
        assert store.get("+2348020812523") is None

    def test_alert_survives_a_restart(self, tmp_path) -> None:
        """
        uvicorn --reload restarts on every file save. An in-memory-only store
        dropped a live briefing between the alert going out and the customer
        ringing back, and they reached the 'no alert' path for no visible reason.
        """
        from pending_alerts import PendingAlertStore

        state = tmp_path / "alerts.json"
        first = PendingAlertStore(state_file=state)
        first.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )

        # A brand new process, as after a reload.
        second = PendingAlertStore(state_file=state)
        restored = second.get("+2348020812523")
        assert restored is not None
        assert restored.signal_id == "FRD-CARD-FOREIGN"

    def test_expired_alerts_are_not_restored(self, tmp_path) -> None:
        from pending_alerts import PendingAlertStore

        state = tmp_path / "alerts.json"
        first = PendingAlertStore(ttl_minutes=0, state_file=state)
        first.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        assert PendingAlertStore(state_file=state).get("+2348020812523") is None

    def test_corrupt_state_file_does_not_stop_startup(self, tmp_path) -> None:
        """Failing to start is worse than starting empty — empty reaches a human."""
        from pending_alerts import PendingAlertStore

        state = tmp_path / "alerts.json"
        state.write_text("{ this is not json", encoding="utf-8")
        assert PendingAlertStore(state_file=state).all() == []

    def test_cleared_alert_is_not_resurrected_by_restart(self, tmp_path) -> None:
        from pending_alerts import PendingAlertStore

        state = tmp_path / "alerts.json"
        first = PendingAlertStore(state_file=state)
        first.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        first.clear("+2348020812523")
        assert PendingAlertStore(state_file=state).get("+2348020812523") is None

    def test_persisted_file_holds_no_card_or_account_data(self, tmp_path) -> None:
        from pending_alerts import PendingAlertStore

        state = tmp_path / "alerts.json"
        store = PendingAlertStore(state_file=state)
        store.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        written = state.read_text(encoding="utf-8")
        for secret in ("539941", "4081", "0123456789", "22134567890"):
            assert secret not in written

    def test_masked_number_in_serialised_form(self, store) -> None:
        """Alert views are read over HTTP — no full numbers in them."""
        store.raise_alert(
            signal_id="FRD-CARD-FOREIGN",
            customer_id="CUS-100001",
            phone_number="+2348020812523",
        )
        rendered = store.all()[0].as_dict()
        assert rendered["phoneNumber"] == "***2523"
        assert "8020812523" not in json.dumps(rendered)


class TestOutputGuardBlocks:
    """
    On a live call the agent ran out of its listed security questions, invented
    one, and asked the customer for their mother's maiden name — the single most
    dangerous thing it could ask. The prompt forbade it and the model did it
    anyway, and the guard only logged. It now replaces the sentence.

    These simulate the sentence-buffering the TTS path does, because a forbidden
    phrase spans several streamed chunks and per-chunk checking never saw it.
    """

    @staticmethod
    def _vet_stream(chunks: list[str]) -> list[str]:
        import re as _re

        import agent as agent_module
        import prompts as prompts_module
        from verification import mentions_forbidden_credential

        out, buffer = [], ""

        def vet(sentence: str) -> str:
            if mentions_forbidden_credential(sentence):
                return prompts_module.CREDENTIAL_REQUEST_BLOCKED
            return sentence

        for chunk in chunks:
            buffer += agent_module.sanitize_for_tts(chunk) + " "
            while (m := _re.search(r"[.!?]\s", buffer)) is not None:
                sentence, buffer = buffer[: m.end()], buffer[m.end() :]
                if sentence.strip():
                    out.append(vet(sentence))
        if buffer.strip():
            out.append(vet(buffer))
        return out

    def test_maiden_name_split_across_chunks_is_blocked(self) -> None:
        """The exact failure: the phrase arrived in pieces, so chunks looked clean."""
        import prompts as prompts_module

        spoken = self._vet_stream(
            ["I am sorry, that was not correct. ", "What is your ", "mother's maiden ", "name?"]
        )
        assert prompts_module.CREDENTIAL_REQUEST_BLOCKED in spoken
        # The replacement deliberately names maiden name as something the agent
        # will NEVER ask for, so assert the *question* is gone rather than the
        # word — the reassurance is exactly what we want the customer to hear.
        assert not any(
            "maiden" in s and s is not prompts_module.CREDENTIAL_REQUEST_BLOCKED
            for s in spoken
        )
        assert not any(s.strip().endswith("maiden name?") for s in spoken)

    @pytest.mark.parametrize(
        "phrase",
        [
            "Can you confirm your PIN?",
            "Please read out the OTP.",
            "What is your card number?",
            "I need your BVN.",
        ],
    )
    def test_other_credential_requests_are_blocked(self, phrase: str) -> None:
        import prompts as prompts_module

        assert self._vet_stream([phrase]) == [prompts_module.CREDENTIAL_REQUEST_BLOCKED]

    def test_legitimate_questions_pass_through(self) -> None:
        spoken = self._vet_stream(["What is your favourite colour? ", "Take your time."])
        assert "What is your favourite colour?" in " ".join(spoken)
        assert "favourite" in " ".join(spoken)

    def test_card_freeze_confirmation_is_not_blocked(self) -> None:
        spoken = " ".join(self._vet_stream(["Your card ending 4081 is now frozen."]))
        assert "frozen" in spoken


class TestChannelTokenStripping:
    """
    On a live call the customer heard the agent read its own reasoning aloud:
    "channel thought channel What is your favourite food". gemma-4 emits
    harmony-style control tokens and they went straight to the TTS.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<|channel>thought\n<channel|>What is your favourite food?",
             "What is your favourite food?"),
            ("<|channel>thought\n<channel|>Your card is frozen.", "Your card is frozen."),
            ("<|channel>analysis\n<channel|>Hello there", "Hello there"),
        ],
    )
    def test_reasoning_channels_are_removed(self, raw: str, expected: str) -> None:
        import agent as agent_module

        assert agent_module.sanitize_for_tts(raw) == expected

    def test_clean_text_is_untouched(self) -> None:
        import agent as agent_module

        line = "Your card ending 4081 is now frozen."
        assert agent_module.sanitize_for_tts(line) == line

    def test_no_control_token_fragments_survive(self) -> None:
        import agent as agent_module

        out = agent_module.sanitize_for_tts("<|channel>thought<channel|>ok")
        for fragment in ("<|", "|>", "channel", "thought"):
            assert fragment not in out


class TestSpeechEndpoints:
    """
    The STT base URL was stored as a bare server root, so the OpenAI client built
    .../audio/transcriptions and every utterance 404'd — the agent could not hear
    the customer at all, silently. These pin the URL shape.
    """

    def test_stt_base_url_ends_in_v1(self) -> None:
        from config import settings

        assert settings.stt.base_url.endswith("/v1")

    def test_tts_base_url_ends_in_v1(self) -> None:
        from config import settings

        assert settings.tts.base_url.endswith("/v1")

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("https://host.net", "https://host.net/v1"),
            ("https://host.net/", "https://host.net/v1"),
            ("https://host.net/v1", "https://host.net/v1"),
            ("https://host.net/v1/", "https://host.net/v1"),
            ("https://host.net/en/v1", "https://host.net/en/v1"),
            ("", ""),
        ],
    )
    def test_v1_normalisation(self, given: str, expected: str) -> None:
        from config import _with_v1

        assert _with_v1(given) == expected


class TestWhatsAppRequestFields:
    """
    The connector RPCs were hand-rolled with guessed camelCase field names
    (`whatsAppApiKey`). The real fields are snake_case, all-lowercase
    `whatsapp` — and LiveKit ignores unknown fields, so every call returned
    200 "OK" and silently never bridged. These assert the generated protobuf
    request objects carry the credentials, which is what makes the difference
    between a connected call and a customer listening to ringing.
    """

    def test_accept_request_carries_meta_credentials(self) -> None:
        from livekit import api

        fields = {f.name for f in api.AcceptWhatsAppCallRequest.DESCRIPTOR.fields}
        for required in (
            "whatsapp_phone_number_id",
            "whatsapp_api_key",
            "whatsapp_cloud_api_version",
            "whatsapp_call_id",
            "sdp",
            "room_name",
            "agents",
        ):
            assert required in fields

    def test_dial_request_uses_to_phone_number(self) -> None:
        from livekit import api

        fields = {f.name for f in api.DialWhatsAppCallRequest.DESCRIPTOR.fields}
        assert "whatsapp_to_phone_number" in fields
        assert "whatsapp_api_key" in fields

    def test_connect_request_takes_no_api_key(self) -> None:
        """Connect is keyed off the call id alone — passing a key would error."""
        from livekit import api

        fields = {f.name for f in api.ConnectWhatsAppCallRequest.DESCRIPTOR.fields}
        assert fields == {"whatsapp_call_id", "sdp", "wait_until_answered"}

    def test_sdp_is_a_message_not_a_string(self) -> None:
        """
        Passing the raw SDP string fails with "expected SessionDescription got
        str" and drops the call. It must be wrapped with its type.
        """
        from livekit import api

        import whatsapp_connector

        req = api.AcceptWhatsAppCallRequest(
            whatsapp_call_id="wacid.TEST",
            sdp=whatsapp_connector._session_description("v=0\r\n", "offer"),
        )
        assert req.sdp.type == "offer"
        assert req.sdp.sdp.startswith("v=0")

    def test_raw_sdp_string_is_rejected(self) -> None:
        """Guards against anyone 'simplifying' _session_description away."""
        from livekit import api

        with pytest.raises(Exception):
            api.AcceptWhatsAppCallRequest(whatsapp_call_id="x", sdp="v=0\r\n")

    def test_cloud_api_version_drops_the_v_prefix(self) -> None:
        """Meta wants "23.0", not "v23.0"."""
        import whatsapp_connector

        assert not whatsapp_connector._cloud_api_version().lower().startswith("v")

    def test_sdp_is_read_from_session_object(self) -> None:
        """Confirmed against a live Meta payload: calls[].session.sdp."""
        import whatsapp_connector

        events = whatsapp_connector.parse_call_events(
            {
                "entry": [
                    {
                        "changes": [
                            {
                                "field": "calls",
                                "value": {
                                    "calls": [
                                        {
                                            "id": "wacid.ABC",
                                            "from": "2348020812523",
                                            "event": "connect",
                                            "direction": "USER_INITIATED",
                                            "session": {"sdp": "v=0...", "sdp_type": "offer"},
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ]
            }
        )
        assert len(events) == 1
        assert events[0]["sdp"] == "v=0..."
        assert events[0]["direction"] == "USER_INITIATED"

    def test_message_status_payloads_yield_no_call_events(self) -> None:
        """Meta sends sent/delivered/read on the `messages` field — not calls."""
        import whatsapp_connector

        assert (
            whatsapp_connector.parse_call_events(
                {"entry": [{"changes": [{"field": "messages", "value": {"statuses": [{}]}}]}]}
            )
            == []
        )


class TestRpcResponseParsing:
    """
    A live inbound call was dropped because AcceptWhatsAppCall answered 200 with
    a text/plain body and resp.json() raised on the mimetype. A successful
    response must never fail to parse — that hangs up on a real customer.
    """

    @staticmethod
    def _parse(body: str):
        """The parsing branch of whatsapp_connector._rpc, as exercised on a 200."""
        if not body.strip():
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    def test_empty_body_is_fine(self) -> None:
        assert self._parse("") == {}
        assert self._parse("   ") == {}

    def test_plain_text_body_is_kept_not_raised(self) -> None:
        assert self._parse("OK") == {"raw": "OK"}

    def test_json_body_is_parsed(self) -> None:
        assert self._parse('{"call_id": "abc"}') == {"call_id": "abc"}

    def test_non_object_json_is_wrapped(self) -> None:
        assert self._parse("[1, 2]") == {"result": [1, 2]}


class TestInboundBriefing:
    def test_unbriefed_inbound_call_refuses_to_improvise(self) -> None:
        """No alert pending means hand to a human, never invent a fraud story."""
        import agent as agent_module

        with pytest.raises(agent_module.NoBriefing):
            agent_module._resolve_call_context(
                json.dumps({"inbound": True, "noAlert": True, "phoneNumber": "+2349999999999"})
            )

    def test_briefed_inbound_call_resolves(self) -> None:
        import agent as agent_module

        signal, customer, _, _ = agent_module._resolve_call_context(
            json.dumps({"signalId": "FRD-ATM-VELOCITY", "inbound": True})
        )
        assert signal["signalId"] == "FRD-ATM-VELOCITY"
        assert customer["firstName"] == "Musa"

    def test_inbound_opening_does_not_say_calling_from(self) -> None:
        """The customer dialled us — 'calling from' would be wrong and confusing."""
        import prompts

        line = prompts.build_opening_line(
            first_name="Yusuf", agent_name="Noba", bank_name="Noba", inbound=True
        )
        assert "calling from" not in line.lower()
        assert "thank you for calling back" in line.lower()

    def test_both_openings_state_the_agent_is_ai(self) -> None:
        import prompts

        for inbound in (True, False):
            line = prompts.build_opening_line(
                first_name="Yusuf",
                agent_name="Noba",
                bank_name="Noba",
                inbound=inbound,
            ).lower()
            assert "a i assistant" in line
            assert "never ask you for your p i n" in line


# ---------------------------------------------------------------------------
# Language routing
#
# A wrong route is a silent failure: the call still produces audio, so nothing
# errors and nothing logs, but the customer hears a Yoruba sentence spelled out
# by an English model. These pin the routes that matter.
# ---------------------------------------------------------------------------

import tts_router


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Hello, am I speaking with Yusuf?", "en"),
        ("Your card ending four one two two is now frozen.", "en"),
        ("I will never ask you for your P I N or a one time code.", "en"),
        # Code-switched: the non-English half decides the voice.
        ("Emi ko ra nkan yen, it was not me.", "yo"),
        ("Jowo, freeze my card now.", "yo"),
        ("Ẹ jọwọ, mo fẹ ran mi lọwọ.", "yo"),
        ("Ba ni yi wannan ba, please freeze my card.", "ha"),
        ("Sannu, kudi na banki ya bata.", "ha"),
        ("Biko, nsogbu di na akaunti m.", "ig"),
        # Pidgin is English-lexified, so the English voice is the right one.
        ("Abeg, I no do that transaction at all.", "pcm"),
        ("Wetin dey happen with my data?", "pcm"),
    ],
)
def test_language_detection(sentence, expected):
    assert tts_router.detect_language(sentence) == expected


def test_pidgin_speaks_with_the_english_voice():
    assert tts_router.VOICE_FOR["pcm"] == "en"


def test_empty_text_does_not_crash_the_router():
    assert tts_router.detect_language("") == "en"
    assert tts_router.detect_language("   ") == "en"


def test_every_expected_language_has_a_voice():
    import code_switching

    for code in code_switching.EXPECTED_LANGUAGES:
        assert code in tts_router.VOICE_FOR, f"no voice mapping for {code}"


def test_non_english_languages_have_configured_endpoints():
    """A missing endpoint means that language silently falls back to English."""
    from config import settings

    for code in ("yo", "ha", "ig"):
        assert settings.tts.endpoint_for(code) is not None, (
            f"{code.upper()}_TTS_BASE_URL is not set — {code} would be spoken "
            "by the English voice"
        )


# ---------------------------------------------------------------------------
# The guard must fire on a REQUEST, never on a mention.
#
# It shipped matching the bare word, which gagged the agent's own safety line -
# "I will never ask you for your P I N, your password, or a one time code" - and
# replaced it with an apology for something it had not done. Censoring the
# warning that protects the customer is the exact opposite of the point.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "I will never ask you for your P I N, your password, or a one time code.",
        "I have not asked you for anything secret and I never will.",
        "You should never give your P I N to anyone who calls you.",
        "A real bank will not ask for your O T P.",
        "Never share your card number with anyone who calls you.",
        "Your card ending four zero eight one is now frozen.",
    ],
)
def test_guard_allows_warnings_and_reassurance(sentence):
    assert mentions_forbidden_credential(sentence) is None, (
        "the guard blocked a sentence that protects the customer"
    )


@pytest.mark.parametrize(
    "sentence,term",
    [
        ("What is your P I N?", "pin"),
        ("Please confirm your mother's maiden name.", "maiden name"),
        ("Can you give me the one time password sent to your phone?", "one time password"),
        ("I need your card number to proceed.", "card number"),
        ("Tell me your B V N.", "bvn"),
        ("Your C V V?", "cvv"),
        ("Please enter your password now.", "password"),
    ],
)
def test_guard_blocks_actual_requests(sentence, term):
    assert mentions_forbidden_credential(sentence) == term


def test_guard_sees_through_spaced_initialisms():
    """The agent spaces initialisms out for the TTS, so the guard must too."""
    assert mentions_forbidden_credential("Tell me your B V N.") == "bvn"
    assert mentions_forbidden_credential("Tell me your BVN.") == "bvn"


def test_the_real_opening_line_is_never_blocked():
    """Regression: this exact line was gagged on a live call."""
    line = prompts.build_opening_line(
        first_name="Yusuf",
        agent_name=settings.bank.agent_display_name,
        bank_name=settings.bank.name,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        assert mentions_forbidden_credential(sentence) is None, sentence


# ---------------------------------------------------------------------------
# The briefing must name this customer's own questions.
#
# It did not, and the prompt told the agent it could not know them. On a live
# call the model skipped the tool and asked Yusuf for the name of his first
# pet - which is a different customer's question, so no answer could ever have
# satisfied it. A model given no wording produces wording.
# ---------------------------------------------------------------------------


def test_briefing_lists_this_customers_questions():
    bank = SimulatedBankAPI()
    signal = bank.get_fraud_signal("FRD-CARD-FOREIGN")
    customer = bank.get_customer_by_id(signal["customerId"])
    briefing = prompts.build_risk_briefing(
        signal, customer, bank.get_card(signal["cardId"])
    )
    for q in customer["securityQuestions"]:
        assert q["question"] in briefing, q["question"]


def test_briefing_carries_this_customers_answers():
    """
    The agent judges the answer itself, so it needs them.

    String matching cannot tell "Sent mi Sent Mary" (Saint Mary, misheard) from
    "blue" (wrong). A reasoning model can, but only if it knows what it is
    comparing against. The prompt forbids ever saying one aloud.
    """
    bank = SimulatedBankAPI()
    for customer in bank_data.CUSTOMERS:
        profile = prompts.build_customer_profile(
            customer,
            bank.get_accounts_by_customer(customer["customerId"]),
            bank.get_cards_by_customer(customer["customerId"]),
        )
        for q in customer["securityQuestions"]:
            assert q["answer"] in profile, q["answer"]
        assert "NEVER SAY AN ANSWER ALOUD" in profile


def test_no_other_customers_data_reaches_the_prompt():
    """Each number gets its own record. Nothing about anyone else leaks in."""
    bank = SimulatedBankAPI()
    for customer in bank_data.CUSTOMERS:
        profile = prompts.build_customer_profile(
            customer,
            bank.get_accounts_by_customer(customer["customerId"]),
            bank.get_cards_by_customer(customer["customerId"]),
        )
        for other in bank_data.CUSTOMERS:
            if other["customerId"] == customer["customerId"]:
                continue
            assert other["customerId"] not in profile
            assert other["lastName"] not in profile
            assert other["phoneNumber"] not in profile
            mine = {q["question"] for q in customer["securityQuestions"]}
            for q in other["securityQuestions"]:
                if q["question"] not in mine:
                    assert q["question"] not in profile, q["question"]


def test_each_customer_gets_only_their_own_questions():
    """Yusuf must never be offered Musa's pet question."""
    bank = SimulatedBankAPI()
    for customer in bank_data.CUSTOMERS:
        briefing = prompts.build_customer_initiated_briefing(
            customer, bank.get_cards_by_customer(customer["customerId"])
        )
        mine = {q["question"] for q in customer["securityQuestions"]}
        for other in bank_data.CUSTOMERS:
            if other["customerId"] == customer["customerId"]:
                continue
            for q in other["securityQuestions"]:
                if q["question"] in mine:
                    continue
                assert q["question"] not in briefing, (
                    f"{customer['firstName']} was shown {other['firstName']}'s question"
                )


# ---------------------------------------------------------------------------
# The agent reads questions off the record, so it often asks one without going
# through ask_security_question. Both routes must work, and paraphrasing the
# question must not be treated as inventing one.
# ---------------------------------------------------------------------------


def _adebayo():
    return SimulatedBankAPI().get_customer_by_id("CUS-100001")


def test_correct_answer_accepted_when_asked_straight_from_the_record():
    flow = ChallengeFlow(customer=_adebayo(), max_attempts=3)
    assert flow.check("amala") is True
    assert flow.verified


def test_wrong_answer_still_counts_as_an_attempt_without_the_tool():
    flow = ChallengeFlow(customer=_adebayo(), max_attempts=3)
    assert flow.check("rice") is False
    assert flow.attempts == 1
    assert not flow.verified


def test_attempt_limit_holds_without_the_tool():
    flow = ChallengeFlow(customer=_adebayo(), max_attempts=3)
    for guess in ("rice", "blue", "kings college"):
        flow.check(guess)
    assert flow.exhausted


# ---------------------------------------------------------------------------
# Answer matching has to survive what speech-to-text does to a spoken answer.
#
# On a live call the customer said their school name three times and was refused
# three times, then transferred: STT returned "Sent mi Sent Mary", "Sent Maili"
# and "SANT.". The customer was right every time. That is the agent failing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heard",
    [
        "green flower school",
        "Green Flower school",
        "green flour school",
        "it is green flower school",
        "Green Flowers School.",
    ],
)
def test_answer_survives_transcription_noise(heard):
    assert answers_match(heard, "green flower school")


@pytest.mark.parametrize(
    "heard",
    ["green", "saint mary", "flower", "school"],
)
def test_a_partial_answer_is_not_enough(heard):
    """Every word of the stored answer must be present, or one overheard
    answer would unlock a different question."""
    assert not answers_match(heard, "green flower school")


def test_short_answers_still_work():
    assert answers_match("it is amala", "amala")
    assert answers_match("the colour green", "green")
    assert not answers_match("rice", "amala")
    assert not answers_match("blue", "green")


# ---------------------------------------------------------------------------
# Control tokens arrive split across streaming chunks.
#
# The model emits "<|channel>thought<channel|>" a few characters at a time, so
# no single chunk contains the pattern. Cleaning per chunk therefore caught
# nothing and the caller heard the agent say "thought" before every sentence.
# The fix is to clean the assembled sentence, which is the same unit the
# credential guard already works on.
# ---------------------------------------------------------------------------


def test_control_tokens_split_across_chunks_are_stripped():
    from agent import sanitize_for_tts

    chunks = ["<|", "channel", ">thought", "\n<channel", "|>", "Done. Your card is frozen."]
    assembled = "".join(chunks)
    spoken = sanitize_for_tts(assembled)
    assert "channel" not in spoken.lower()
    assert "<|" not in spoken and "|>" not in spoken
    assert "Done." in spoken


def test_chunks_are_joined_without_inserting_spaces():
    """A space after every chunk split words in half mid-utterance."""
    from agent import sanitize_for_tts

    assert sanitize_for_tts("".join(["Yus", "uf", ", your card"])).startswith("Yusuf,")


# ---------------------------------------------------------------------------
# One ambiguous word must not change the speaker.
#
# "abi" is as ordinary in Pidgin as in Yoruba, and it was in both marker sets.
# With the tie broken in favour of the non-Pidgin language, "You wan buy data
# abi?" routed to the Yoruba voice - a different speaker, in the middle of a
# Pidgin conversation. That is what a caller hears as the agent being swapped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "You wan buy data abi?",
        "Abi you don try am before?",
        "No wahala, I go check am for you.",
        "Oga, the money don comot from your account.",
        "Your data plan don enter now.",
        "Make I check your balance first.",
    ],
)
def test_pidgin_never_leaves_the_english_voice(line):
    lang = tts_router.detect_language(line)
    assert tts_router.VOICE_FOR[lang] == "en", f"{line!r} routed to {lang}"


@pytest.mark.parametrize(
    "line,voice",
    [
        ("Ẹ jọwọ, mo fẹ ran mi lọwọ.", "yo"),
        ("Emi ko ra nkan yen, jowo freeze the card.", "yo"),
        ("Sannu, kwana biyu, lafiya?", "ha"),
        ("Ba ni yi wannan ba, please freeze my card.", "ha"),
        ("Biko, nsogbu di na akaunti m.", "ig"),
    ],
)
def test_a_real_switch_still_switches(line, voice):
    """Guarding against false switches must not block the true ones."""
    assert tts_router.VOICE_FOR[tts_router.detect_language(line)] == voice


# ---------------------------------------------------------------------------
# Sahara reads punctuation aloud on short lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,spoken",
    [
        # Both marks were caught speaking themselves on a live call.
        ("Done.", "Done"),
        ("Done,", "Done"),
        ("Okay,", "Okay"),
        ("No wahala,", "No wahala"),
        ("One moment.", "One moment"),
        ("Okay, done.", "Okay, done"),
        # Only the trailing mark goes; punctuation inside the line is prosody.
        ("Yes, that is right.", "Yes, that is right"),
        # Question and exclamation marks carry the intonation of the whole
        # line and were never pronounced, so they stay.
        ("Yes?", "Yes?"),
        ("Which one?", "Which one?"),
        # Long enough that the normaliser behaves, so nothing is touched.
        ("I have frozen the card ending 4081 for you.",
         "I have frozen the card ending 4081 for you."),
        # A fragment that is nothing but punctuation must not be sent at all.
        (".", ""),
        (",", ""),
        ("  ", ""),
    ],
)
def test_short_lines_lose_trailing_punctuation(raw, spoken):
    from intron_tts import _spoken_text

    assert _spoken_text(raw) == spoken


# ---------------------------------------------------------------------------
# Short fragments are merged rather than spoken alone
# ---------------------------------------------------------------------------


def _routed(sentences, *, lang="en"):
    """Run speak_routed over a fixed list, capturing what each voice was asked
    to say. The engine is a stub: this is about text, not audio."""
    import asyncio

    spoken = []

    class _Engine:
        async def synthesize(self, text):
            spoken.append(text)
            return
            yield  # pragma: no cover - makes this an async generator

    class _Router:
        def for_text(self, text):
            return lang, _Engine()

    async def _feed():
        for s in sentences:
            yield s

    async def _drain():
        async for _ in tts_router.speak_routed(_Router(), _feed()):
            pass

    asyncio.run(_drain())
    return spoken


def test_a_short_sentence_leads_the_next_one():
    """Sahara speaks punctuation aloud on fragments sent by themselves."""
    assert _routed(["Okay. ", "I have frozen the card ending 4081 for you. "]) == [
        "Okay. I have frozen the card ending 4081 for you."
    ]


def test_several_short_sentences_collapse_into_one():
    assert _routed(["Done. ", "Okay. ", "No wahala. "]) == ["Done. Okay. No wahala."]


def test_a_lone_short_reply_is_still_spoken():
    """Merging must never swallow a fragment that has nothing to merge into."""
    assert _routed(["Done. "]) == ["Done."]


def test_long_sentences_are_not_merged():
    lines = [
        "I have frozen the card ending 4081 for you now. ",
        "Nothing else on the account has changed at all. ",
    ]
    assert _routed(lines) == [line.strip() for line in lines]


def test_card_digits_are_spoken_one_by_one():
    """A bare 4081 is read "four thousand eighty-one", which is not a card."""
    from prompts import spoken_digits

    assert spoken_digits("4081") == "4 0 8 1"
    assert spoken_digits(9032) == "9 0 3 2"


def test_the_frozen_card_line_spells_the_digits():
    import prompts

    line = prompts.CARD_FROZEN.format(last4=prompts.spoken_digits("4081"))
    assert "4 0 8 1" in line and "4081" not in line
