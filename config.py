"""
Central configuration for the fraud-response voice agent.

Everything the system needs is read from .env exactly once, here, so no other
module has to touch os.getenv. Call validate() at startup to fail loudly on a
missing credential rather than three minutes into a live call.
"""

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# override=True is deliberate. Without it python-dotenv leaves any variable that
# is already exported in the shell untouched, so a stale LIVEKIT_URL from an
# earlier session silently beats the .env file. That happened here: the agent
# worker registered to one LiveKit project while the app dispatched jobs into
# another, so calls rang out with no worker ever picking them up, and nothing in
# either log said why. The file on disk is the single source of truth.
load_dotenv(override=True)

logger = logging.getLogger("fraud_agent.config")


def _env(key: str, default: str = "") -> str:
    # .env in this project has values written as KEY = "value", so strip stray quotes.
    return (os.getenv(key) or default).strip().strip('"').strip("'")


@dataclass
class LiveKitSettings:
    url: str = field(default_factory=lambda: _env("LIVEKIT_URL"))
    api_key: str = field(default_factory=lambda: _env("LIVEKIT_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("LIVEKIT_API_SECRET"))

    @property
    def http_url(self) -> str:
        """LiveKit REST base — the same host as the websocket URL over https."""
        return self.url.replace("wss://", "https://").replace("ws://", "http://")


@dataclass
class WhatsAppSettings:
    token: str = field(default_factory=lambda: _env("WHATSAPP_TOKEN"))
    phone_number_id: str = field(default_factory=lambda: _env("WHATSAPP_PHONE_NUMBER_ID"))
    verify_token: str = field(default_factory=lambda: _env("WHATSAPP_VERIFY_TOKEN"))
    graph_version: str = field(default_factory=lambda: _env("WHATSAPP_GRAPH_VERSION", "v23.0"))

    @property
    def graph_url(self) -> str:
        return f"https://graph.facebook.com/{self.graph_version}/{self.phone_number_id}"


@dataclass
class LLMSettings:
    # Read LLM first, then FINETUNED_BASE_URL. The llama3-8b host is the one that
    # works; note it still advertises its model id as google/gemma-4-E4B-it, so
    # the host changes but MODEL does not.
    base_url: str = field(
        default_factory=lambda: _env("LLM") or _env("FINETUNED_BASE_URL")
    )
    model: str = field(default_factory=lambda: _env("MODEL", "google/gemma-4-E4B-it"))
    api_key: str = field(default_factory=lambda: _env("API_KEY"))


def _with_v1(url: str) -> str:
    """
    Ensure an OpenAI-compatible base URL ends in /v1.

    ENGLISH_STT_API_URL is stored as the bare server root. The OpenAI client
    appends only '/audio/transcriptions', so the bare root produces
    .../audio/transcriptions, which 404s — every utterance silently fails to
    transcribe and the agent never hears the customer. With /v1 the same path
    resolves. Verified against the live host: bare root 404, /v1 reaches it.
    """
    url = url.rstrip("/")
    if url and not url.endswith("/v1"):
        url += "/v1"
    return url


@dataclass
class STTSettings:
    """
    Speech recognition, switchable by provider.

    STT_PROVIDER selects one of:
      deepgram    — nova-2-phonecall, tuned for narrowband telephony
      intron      — the Sahara/Intron voice API (required by the challenge)
      linguacenter— the Nigerian-English endpoint (OpenAI-compatible)

    Kept swappable because the challenge requires benchmarking at least three
    speech models, and because provider quality varies enormously on the
    narrowband, code-switched audio this agent actually receives.
    """

    provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "deepgram").lower())
    # STT_BASE_URL wins when set. ENGLISH_STT_API_URL is linguacenter, which is
    # English-only — fine for an English call, useless the moment the caller
    # switches. Point STT_BASE_URL at the finetuned-ml endpoint to test that.
    base_url: str = field(
        default_factory=lambda: _with_v1(
            _env("STT_BASE_URL") or _env("ENGLISH_STT_API_URL")
        )
    )
    api_key: str = field(default_factory=lambda: _env("API_KEY"))
    model: str = field(default_factory=lambda: _env("ENGLISH_STT_MODEL", "whisper-1"))
    language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en"))

    deepgram_api_key: str = field(default_factory=lambda: _env("DEEPGRAM_API_KEY"))
    # nova-2-phonecall is trained on 8-16kHz call audio, which is exactly what
    # WhatsApp delivers. nova-3 is better on wideband but worse down here.
    deepgram_model: str = field(
        default_factory=lambda: _env("DEEPGRAM_MODEL", "nova-2-phonecall")
    )

    intron_api_key: str = field(default_factory=lambda: _env("intron_api"))
    elevenlabs_api_key: str = field(
        default_factory=lambda: _env("ELEVENLABS_API_KEY")
    )
    # Scribe is ElevenLabs' speech-to-text model; v1 is the multilingual one.
    elevenlabs_stt_model: str = field(
        default_factory=lambda: _env("ELEVENLABS_STT_MODEL", "scribe_v1")
    )
    # For the gated AfriSwitch datasets. HF_TOKEN is the name the huggingface
    # libraries read themselves, so setting it in .env also covers any tooling
    # that looks it up directly.
    hf_token: str = field(
        default_factory=lambda: _env("HF_TOKEN") or _env("HUGGINGFACE_TOKEN")
    )
    intron_stt_url: str = field(
        default_factory=lambda: _env("INTRON_STT_URL", "wss://infer.voice.intron.io/stt/v1/stream")
    )


def tts_provider() -> str:
    """
    Which engine speaks: intron (Sahara), finetuned-en, or finetuned-ml.

    Sahara by default. It is the only one of the three that reads a
    code-switched sentence in one voice - given "Abeg, no be me do that
    transaction. Your card don freeze now." it produced audio an independent
    transcriber read back with both halves intact - and at 3.3s warm per reply
    it is as fast as the others.

    The known risk: Sahara accepted about three sessions in quick succession in
    testing and hung up on the fourth. Whole replies are batched into a single
    session to spend as few as possible (see tts_router._speak_batched), and a
    closed socket is retried once, but a long call may still lose a line.
    TTS_PROVIDER=finetuned-en is the fallback if that happens on the day.
    """
    return _env("TTS_PROVIDER", "intron").lower()


def _tts_is_finetuned_en() -> bool:
    """
    the fine-tuned English voice carries English. Set TTS_PROVIDER=finetuned-ml to switch it back.

    Non-English sentences ignore this and go to their own finetuned-ml endpoint
    either way - the fine-tuned English voice has no Yoruba, Hausa or Igbo voice - so this only
    decides who speaks English and Pidgin.
    """
    return tts_provider() not in ("finetuned-ml", "intron", "sahara")


@dataclass
class TTSSettings:
    """
    Voice output. Defaults to the finetuned-ml endpoint; TTS_PROVIDER=finetuned-en
    switches English back to the fine-tuned English voice.

    Measured on the same sentences, three runs each:

                     short (9 words)   long (44 words)   first byte (long)
        finetuned-en            2.45s             4.95s             4.12s
        finetuned-ml       2.28s             4.19s             3.62s
        intron           (not measured)    10.03s             8.20s

    Three reasons finetuned-ml is the default:

      1. It is faster everywhere, and neither engine streams — first byte lands
         at 85-90% of total time, so the caller waits out the whole sentence.
      2. the fine-tuned English voice takes 5.2s to say what finetuned-ml says in 3.8s. That padding
         is dead air on a fraud call where a card is live.
      3. It is the same model family as the Yoruba, Hausa and Igbo endpoints,
         so the voice does not visibly change speaker when the agent switches
         language mid-call. With the fine-tuned English voice on English the jump is obvious and
         sounds like the call was handed to a different person.

    Intron TTS is benchmarked but not used live: 8.2s to first audio warm,
    12.5s cold. It is accurate, and far too slow to hold a conversation.

    One landmine: finetuned-ml returns HTTP 500 for response_format=pcm. It must
    be wav, which is what the agents send.
    """

    base_url: str = field(
        default_factory=lambda: (
            (_env("FINETUNED_TTS_BASE_URL") or _env("ENGLISH_TTS_BASE_URL"))
            if _tts_is_finetuned_en()
            else _env("ENGLISH_TTS_BASE_URL")
        )
    )
    api_key: str = field(default_factory=lambda: _env("API_KEY"))
    model: str = field(
        default_factory=lambda: (
            _env("FINETUNED_EN_TTS_MODEL", "tts-1")
            if _tts_is_finetuned_en()
            else _env("ENGLISH_TTS_MODEL", "nigerian-english-xtts")
        )
    )
    voice: str = field(
        default_factory=lambda: (
            _env("FINETUNED_EN_TTS_VOICE", "voice5")
            if _tts_is_finetuned_en()
            else _env("ENGLISH_TTS_VOICE", "female2")
        )
    )
    elevenlabs_api_key: str = field(
        default_factory=lambda: _env("ELEVENLABS_API_KEY")
    )
    # eleven_multilingual_v2 is the one that handles non-English text; the
    # monolingual models mangle Yoruba and Hausa outright.
    elevenlabs_tts_model: str = field(
        default_factory=lambda: _env("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    )
    elevenlabs_voice_id: str = field(
        default_factory=lambda: _env("ELEVENLABS_VOICE_ID", "")
    )
    intron_tts_url: str = field(
        default_factory=lambda: _env(
            "INTRON_TTS_URL", "wss://infer.voice.intron.io/tts/v1/stream"
        )
    )

    # Route each spoken sentence to the voice for its language. On by default:
    # the finetuned-ml endpoints are per-language (/en/, /yo/, /ha/, /ig/), so
    # without this a Yoruba reply is read by the English model and comes out as
    # gibberish. Set TTS_ROUTING=off to pin everything to one voice.
    routing: bool = field(
        default_factory=lambda: _env("TTS_ROUTING", "on").lower()
        not in ("off", "0", "false", "no")
    )

    @property
    def voice_languages(self) -> set[str]:
        """
        Which languages are allowed their own voice.

        English and Pidgin only, because they share one voice on Sahara and
        moving between them is inaudible. Yoruba, Hausa and Igbo are different
        speakers, so routing to them changes who the caller is talking to
        mid-call - which is heard as the agent being swapped, not as
        multilingual support. One ambiguous word was enough to trigger it.

        Set TTS_VOICE_LANGUAGES=en,pcm,yo,ha,ig to allow the rest once the
        speaker change is something you actually want.
        """
        raw = _env("TTS_VOICE_LANGUAGES", "en,pcm")
        return {part.strip().lower() for part in raw.split(",") if part.strip()}

    @property
    def is_intron(self) -> bool:
        """Sahara TTS speaks, rather than one of the our fine-tuned model endpoints."""
        return tts_provider() in ("intron", "sahara")

    def endpoint_for(self, lang: str) -> tuple[str, str, str] | None:
        """
        (base_url, model, voice) for a language code, or None if unconfigured.

        English deliberately returns whatever TTS_PROVIDER selected rather than
        forcing the finetuned-ml English endpoint — the fine-tuned English voice sounds better on
        English, and there is no reason to lose that just because the call may
        also contain Yoruba.
        """
        if lang in ("en", "pcm"):
            return self.base_url, self.model, self.voice
        prefix = {"yo": "YORUBA", "ha": "HAUSA", "ig": "IGBO"}.get(lang)
        if prefix is None:
            return None
        base = _env(f"{prefix}_TTS_BASE_URL")
        if not base:
            return None

        # Same voice name across every language, so the speaker does not change
        # when the agent code-switches. The point of this product is that
        # switching language mid-sentence is ordinary; if the voice changes with
        # it, the call sounds like it was handed to a different person, which
        # makes the ordinary thing sound like an event. All four finetuned-ml
        # endpoints accept the English voice name and render it distinctly.
        # Set TTS_PER_LANGUAGE_VOICES=on to go back to each endpoint's own.
        per_language = _env("TTS_PER_LANGUAGE_VOICES", "off").lower() in (
            "on",
            "1",
            "true",
            "yes",
        )
        voice = (
            _env(f"{prefix}_TTS_VOICE", "female") if per_language else self.voice
        )
        return (base, _env(f"{prefix}_TTS_MODEL", "tts-1"), voice)


@dataclass
class SIPSettings:
    """
    Twilio (or any SIP provider) trunk, used for real phone calls.

    This is the transport that actually works for a Nigerian bank. WhatsApp
    blocks business-initiated calls from Nigerian numbers, and its LiveKit
    connector is a closed beta — SIP has neither restriction.
    """

    outbound_trunk_id: str = field(default_factory=lambda: _env("SIP_OUTBOUND_TRUNK_ID"))
    # Twilio Elastic SIP Trunking, Termination tab
    termination_uri: str = field(default_factory=lambda: _env("TWILIO_TERMINATION_URI"))
    username: str = field(default_factory=lambda: _env("TWILIO_SIP_USERNAME"))
    password: str = field(default_factory=lambda: _env("TWILIO_SIP_PASSWORD"))
    # The number the customer sees calling them.
    caller_id: str = field(default_factory=lambda: _env("TWILIO_PHONE_NUMBER"))

    @property
    def configured(self) -> bool:
        return bool(self.outbound_trunk_id)

@dataclass
class BankSettings:
    """Identity the agent presents, and the limits it operates under."""

    name: str = field(default_factory=lambda: _env("BANK_NAME", "Noba"))
    # Both agents introduce themselves as Noba, matching the verified name on
    # the WhatsApp number, so the sender the customer sees and the voice they
    # hear are the same brand. Change AGENT_DISPLAY_NAME in .env to rebrand.
    agent_display_name: str = field(
        default_factory=lambda: _env("AGENT_DISPLAY_NAME", "Noba")
    )

    @property
    def written_name(self) -> str:
        """
        How the bank signs a written message to a customer.

        Spoken, "Noba" is enough - the caller already knows who rang. Written on
        WhatsApp among a hundred other chats, "Noba" alone is just a word;
        "Noba Bank" says what it is at a glance, which matters on a message the
        customer is meant to trust immediately. Set BANK_WRITTEN_NAME to
        override, or name the bank fully in BANK_NAME and this leaves it alone.
        """
        explicit = _env("BANK_WRITTEN_NAME")
        if explicit:
            return explicit
        return self.name if "bank" in self.name.lower() else f"{self.name} Bank"
    fraud_desk_number: str = field(default_factory=lambda: _env("FRAUD_DESK_NUMBER", "+2348000000000"))
    # A fraud call that drags on is a failed fraud call — cap it.
    max_call_seconds: int = field(default_factory=lambda: int(_env("MAX_CALL_SECONDS", "300")))
    max_verification_attempts: int = field(
        default_factory=lambda: int(_env("MAX_VERIFICATION_ATTEMPTS", "3"))
    )


@dataclass
class TelcoSettings:
    """The second agent: customer care for a Nigerian mobile network."""

    name: str = field(default_factory=lambda: _env("TELCO_NAME", "Naija Mobile"))
    agent_display_name: str = field(
        default_factory=lambda: _env("TELCO_AGENT_NAME", "Noba")
    )
    agent_name: str = field(
        default_factory=lambda: _env("TELCO_AGENT_WORKER", "telco-care-agent")
    )
    care_desk_number: str = field(
        default_factory=lambda: _env("TELCO_CARE_NUMBER", "+2348000000001")
    )
    demo_issue_id: str = field(
        default_factory=lambda: _env("TELCO_DEMO_ISSUE", "TEL-DATA-NOT-WORKING")
    )
    # Must differ from the fraud agent's, or the second worker to start cannot
    # bind its HTTP server and dies.
    http_port: int = field(default_factory=lambda: int(_env("TELCO_HTTP_PORT", "8082")))


@dataclass
class Settings:
    livekit: LiveKitSettings = field(default_factory=LiveKitSettings)
    whatsapp: WhatsAppSettings = field(default_factory=WhatsAppSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    stt: STTSettings = field(default_factory=STTSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    sip: SIPSettings = field(default_factory=SIPSettings)
    bank: BankSettings = field(default_factory=BankSettings)
    telco: TelcoSettings = field(default_factory=TelcoSettings)

    agent_name: str = field(default_factory=lambda: _env("AGENT_NAME", "fraud-response-agent"))
    # Scenario used when the agent starts without dispatch metadata, i.e. console mode.
    demo_signal_id: str = field(default_factory=lambda: _env("DEMO_SIGNAL_ID", "FRD-CARD-FOREIGN"))
    # Prewarmed job runners. LiveKit's production default is 8, which starves a
    # laptop — see the note in agent.py's WorkerOptions.
    agent_idle_processes: int = field(
        default_factory=lambda: int(_env("AGENT_IDLE_PROCESSES", "1"))
    )
    # Pins which agent answers an inbound call, for demos. Empty means route
    # automatically: a caller with a live fraud alert gets the fraud agent,
    # everyone else gets telco care.
    demo_agent: str = field(default_factory=lambda: _env("DEMO_AGENT", "").lower())
    # Each worker runs a small HTTP server for health and tracing, and in
    # production mode they all default to 8081 — so running both agents at once
    # fails with "only one usage of each socket address". They need separate
    # ports. (Dev mode picks a free port automatically, which is why this only
    # bites on `start`.)
    agent_http_port: int = field(default_factory=lambda: int(_env("AGENT_HTTP_PORT", "8081")))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list means good to go."""
        problems: list[str] = []
        required = [
            ("LIVEKIT_URL", self.livekit.url),
            ("LIVEKIT_API_KEY", self.livekit.api_key),
            ("LIVEKIT_API_SECRET", self.livekit.api_secret),
            ("WHATSAPP_TOKEN", self.whatsapp.token),
            ("WHATSAPP_PHONE_NUMBER_ID", self.whatsapp.phone_number_id),
            ("WHATSAPP_VERIFY_TOKEN", self.whatsapp.verify_token),
            ("LLM (or FINETUNED_BASE_URL)", self.llm.base_url),
            ("API_KEY", self.llm.api_key),
            ("ENGLISH_TTS_BASE_URL", self.tts.base_url),
        ]
        for key, value in required:
            if not value:
                problems.append(f"{key} is missing from .env")
        return problems

    def log_summary(self) -> None:
        logger.info("Bank            : %s", self.bank.name)
        logger.info("LiveKit         : %s", self.livekit.url)
        logger.info("WhatsApp number : %s", self.whatsapp.phone_number_id)
        logger.info("LLM             : %s @ %s", self.llm.model, self.llm.base_url)
        logger.info("STT             : %s (%s)", self.stt.model, self.stt.language)
        logger.info("TTS             : %s / voice=%s", self.tts.model, self.tts.voice)


settings = Settings()
