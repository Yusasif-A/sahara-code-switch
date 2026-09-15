"""
Speech-to-text providers, selectable at runtime.

The Sahara CodeSwitch Africa Challenge requires benchmarking at least three
speech models including the Sahara/Intron API, so recognition is kept behind one
factory rather than wired directly into the agent. Switching provider is an
environment variable, and benchmark.py drives the same factory.

Provider notes from testing on live WhatsApp calls:

  deepgram      nova-2-phonecall is trained on 8-16kHz call audio, which is what
                WhatsApp actually delivers (its SDP offers Opus at
                maxplaybackrate=16000). This is the default.
  linguacenter  OpenAI-compatible Nigerian-English endpoint. On live narrowband
                audio it returned "-.seint." for "saint mary" and "gain." for
                "green" — accurate enough on clean wideband, poor on the phone.
  intron        The Sahara API. Streaming WebSocket, not OpenAI-compatible, so
                it needs the adapter below rather than openai.STT.
"""

from __future__ import annotations

import logging

from livekit.agents import stt as agents_stt
from livekit.plugins import openai

from config import settings

# Imported at module scope, not lazily inside build_stt(). LiveKit registers a
# plugin as a side effect of importing it, and registration must happen on the
# main thread — but build_stt() runs inside a job, on a worker thread, so a
# lazy import there dies with "Plugins must be registered on the main thread"
# the moment a real call arrives. Wrapped in try/except so a missing optional
# plugin still leaves the others usable.
try:
    from livekit.plugins import deepgram as _deepgram
except ImportError:  # pragma: no cover - depends on which extras are installed
    _deepgram = None

logger = logging.getLogger("fraud_agent.stt")


def _env_intron_default() -> str:
    """Which code-switched pair to assume when STT_LANGUAGE says nothing useful."""
    import os

    return (os.getenv("INTRON_LANGUAGE") or "yo").strip().strip('"').lower()


def build_stt(provider: str | None = None, *, vad=None) -> agents_stt.STT:
    """
    Return a configured STT instance.

    Args:
        provider: override STT_PROVIDER. Used by the benchmark to build each
            model in turn without touching the environment.
        vad: the loaded Silero VAD. Only Intron uses it, and only to stream —
            given one it opens its socket at start-of-speech and feeds audio
            live, which roughly halves the wait after the caller stops talking.
            Without one it still works, just in batch.
    """
    name = (provider or settings.stt.provider).lower()

    if name == "deepgram":
        if _deepgram is None:
            raise RuntimeError(
                "STT_PROVIDER=deepgram but livekit-plugins-deepgram is not installed"
            )
        deepgram = _deepgram
        if not settings.stt.deepgram_api_key:
            raise RuntimeError("STT_PROVIDER=deepgram but DEEPGRAM_API_KEY is not set")
        logger.info("STT provider: deepgram (%s)", settings.stt.deepgram_model)
        return deepgram.STT(
            model=settings.stt.deepgram_model,
            language=settings.stt.language,
            api_key=settings.stt.deepgram_api_key,
            # Telephony audio is noisy and callers trail off; smart formatting
            # keeps numbers readable and punctuation helps the LLM.
            smart_format=True,
            punctuate=True,
        )

    if name == "intron":
        if not settings.stt.intron_api_key:
            raise RuntimeError("STT_PROVIDER=intron but intron_api is not set in .env")
        from intron_stt import CODE_SWITCHED, IntronSTT

        # STT_LANGUAGE defaults to "en" for the other providers, and "en" is the
        # one Intron code that has no code-switched model. Taking that default
        # literally here would quietly disable the exact capability Intron was
        # chosen for, so an unset language resolves to Yoruba-English instead.
        # An explicit code-switched code is always honoured.
        language = settings.stt.language
        if language.lower() in ("", "en", "auto", "multi"):
            language = _env_intron_default()
            logger.info(
                "STT_LANGUAGE was %r, which has no code-switched Intron model; "
                "using %r (Yoruba-English). Set STT_LANGUAGE to one of %s to "
                "choose a different pair.",
                settings.stt.language or "unset",
                language,
                ", ".join(sorted(CODE_SWITCHED)),
            )
        logger.info("STT provider: intron (Sahara API, language=%s)", language)

        return IntronSTT(
            api_key=settings.stt.intron_api_key,
            url=settings.stt.intron_stt_url,
            language=language,
            vad=vad,
        )

    if name == "elevenlabs":
        if not settings.stt.elevenlabs_api_key:
            raise RuntimeError(
                "STT_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is not set"
            )
        from elevenlabs_stt import ElevenLabsSTT

        logger.info("STT provider: elevenlabs (%s)", settings.stt.elevenlabs_stt_model)
        return ElevenLabsSTT(
            api_key=settings.stt.elevenlabs_api_key,
            model=settings.stt.elevenlabs_stt_model,
            # Blank means let Scribe detect, which is the right default on
            # code-switched audio - there is no single correct language to name.
            language="" if settings.stt.language.lower() in ("auto", "multi") else settings.stt.language,
        )

    if name in ("linguacenter", "openai", "finetuned"):
        # The finetuned-ml endpoints decide the language themselves. Pinning
        # STT_LANGUAGE=en makes them decode a Yoruba utterance as English, which
        # returns confident nonsense rather than an error — set STT_LANGUAGE to
        # "auto" (or blank) to let the model choose, which is what a
        # code-switched call needs.
        language = settings.stt.language
        kwargs = {} if language.lower() in ("", "auto", "multi") else {"language": language}
        logger.info(
            "STT provider: %s (%s, language=%s)",
            name,
            settings.stt.base_url,
            language or "auto",
        )
        return openai.STT(
            base_url=settings.stt.base_url,
            model=settings.stt.model,
            api_key=settings.stt.api_key,
            **kwargs,
        )

    raise RuntimeError(
        f"Unknown STT_PROVIDER '{name}'. "
        "Use deepgram, intron, elevenlabs, finetuned or linguacenter."
    )


AVAILABLE_PROVIDERS = (
    "deepgram",
    "intron",
    "elevenlabs",
    "finetuned",
    "linguacenter",
)
