"""
ElevenLabs Scribe speech-to-text as a LiveKit STT plugin.

The livekit-plugins-elevenlabs package ships TTS only - there is no STT class in
it - so this wraps the Scribe HTTP API directly.

WHY IT IS HERE
Scribe led the benchmark on code-switched Nigerian audio: WER 0.48 against
Intron's 0.53 and Deepgram's 0.79, and it was the only model that kept every
decision-critical term. It is also fast, around 2.5s against Intron's 4-6s.

What it does not do is keep both halves of a code-switched sentence as reliably
as Intron, which is the reason Intron is still the default. Use this when the
transcript quality matters more than the code-switching demonstration, or when
Intron is having a bad day.

Scribe authenticates with an xi-api-key header rather than a bearer token, and
takes audio as multipart. The language hint is sent only for plain English: the
Nigerian codes used elsewhere here (pcm, yo, ha, ig) are not all recognised, and
a wrong hint is worse than none. On genuinely mixed audio there is no single
correct language to declare anyway, so letting Scribe detect is the honest
setting.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

logger = logging.getLogger("elevenlabs-stt")

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

# ISO-639-3, which is what Scribe expects. Only codes it actually recognises
# appear here; everything else is left to detection.
LANGUAGE_HINTS = {"en": "eng", "yo": "yor", "ha": "hau", "ig": "ibo"}


class ElevenLabsSTT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "scribe_v1",
        language: str = "",
        timeout: float = 30.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        # Non-streaming: Scribe is a single HTTP request per utterance, so the
        # framework wraps this in its VAD-driven adapter.
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._api_key = api_key
        self._model = model
        self._language = language
        self._timeout = timeout
        self._session = http_session
        self._owns_session = http_session is None

        hint = LANGUAGE_HINTS.get(language.lower())
        logger.info(
            "ElevenLabs STT: %s, language hint %s",
            model,
            hint or "none (Scribe detects)",
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if isinstance(language, str) and language else self._language
        # to_wav_bytes() hands back a complete WAV at the frame's own rate.
        # Scribe accepts whatever it is given, so there is nothing to resample.
        frame = rtc.combine_audio_frames(buffer)
        audio = frame.to_wav_bytes()

        form = aiohttp.FormData()
        form.add_field("model_id", self._model)
        if hint := LANGUAGE_HINTS.get(lang.lower()):
            form.add_field("language_code", hint)
        form.add_field(
            "file", audio, filename="audio.wav", content_type="audio/wav"
        )

        try:
            async with self._ensure_session().post(
                ENDPOINT,
                data=form,
                headers={"xi-api-key": self._api_key},
                timeout=aiohttp.ClientTimeout(total=max(self._timeout, conn_options.timeout)),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    # A quota failure reads the same as a network failure unless
                    # the body is surfaced, and this key has run out before.
                    raise APIConnectionError(
                        f"ElevenLabs STT {resp.status}: {body[:200]}"
                    )
                text = (await resp.json()).get("text", "")
        except asyncio.TimeoutError as exc:
            raise APIConnectionError("ElevenLabs STT timed out") from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"ElevenLabs STT failed: {exc}") from exc

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=lang or "auto", text=text.strip())],
        )

    async def aclose(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None
        await super().aclose()
