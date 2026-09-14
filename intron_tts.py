"""
Intron (Sahara) text-to-speech as a LiveKit TTS plugin.

WHY THE PIDGIN VOICE
It reads code-switched text natively. Given "Abeg, no be me do that
transaction. Your card don freeze now." it produced audio that an independent
transcriber read back with both halves intact - one voice, no seam, no routing
between a Pidgin voice and an English one. Pidgin is English-lexified, so the
English inside a Pidgin sentence is not foreign to the model the way it is to a
monolingual Yoruba voice.

WHY THIS USES /generate AND NOT /stream
Intron offers two TTS endpoints. The streaming one allocates a *session* per
call, and sessions are a limited resource on the account: after roughly four,
every new connection is refused with close code 1002 in half a second - no
message, no SESSION_CREATED. On a live call that meant the agent spoke, spoke
again, and then went silent mid-conversation.

Closing properly helped and did not fix it. Measured under identical timing:

    shared HTTP session, autoclose off, no explicit close   1 call before 1002
    fresh session, autoclose on, explicit close             3 calls before 1002

So sessions were leaking, but there is a hard budget underneath that refills
slowly rather than per request. Batching a whole reply into one session bought
one session per turn instead of one per sentence, and still ran out.

POST /tts/v1/generate has no session at all. Nine calls back to back, two
seconds apart, zero failures. It also drops two limits the streaming protocol
imposes: no 10-100 character chunking, so "Done." works as-is, and no polling
for audio that has to be explicitly fetched.

The cost is latency: ~5.0s to generate plus ~0.7s to fetch the file, against
PrepAI's 4.1s and the streaming endpoint's 3.3s warm. Constant regardless of
length, which points at a fixed queue rather than per-character work. A second
and a half per reply is the price of Sahara speaking throughout the call
instead of intermittently.

The response carries a URL rather than audio, so every reply costs two round
trips.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)

logger = logging.getLogger("intron-tts")

GENERATE_URL = "https://infer.voice.intron.io/tts/v1/generate"

# Every Intron voice returns 22.05kHz mono PCM16.
SAMPLE_RATE = 22_050
NUM_CHANNELS = 1

# voice_accent must be one Intron offers for that language. Each language has a
# single accent matching its own name; "nigerian" is not a value.
ACCENTS = {
    "pcm": "pidgin",
    "yo": "yoruba",
    "ha": "hausa",
    "ig": "igbo",
    "en": "yoruba",
}


def pcm_from_wav(data: bytes) -> tuple[bytes, int]:
    """Return (pcm, sample_rate) from a WAV file, or the bytes unchanged."""
    if data[:4] != b"RIFF":
        return data, SAMPLE_RATE
    with wave.open(io.BytesIO(data)) as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate()


class IntronTTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str,
        url: str = GENERATE_URL,
        language: str = "pcm",
        gender: str = "female",
        accent: str | None = None,
        timeout: float = 90.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = api_key
        # .env carries the streaming URL; this endpoint is its sibling. Accept
        # either so nothing has to change there.
        self._url = GENERATE_URL if url.startswith("ws") else url
        self._language = language
        self._accent = accent or ACCENTS.get(language, "pidgin")
        self._gender = gender
        self._timeout = timeout
        self._session = http_session
        self._owns_session = http_session is None

        logger.info(
            "Intron TTS: voice_language=%s accent=%s gender=%s (generate endpoint)",
            self._language,
            self._accent,
            self._gender,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        return _IntronStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )

    async def aclose(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None


class _IntronStream(tts.ChunkedStream):
    async def _run(self) -> None:
        engine: IntronTTS = self._tts  # type: ignore[assignment]
        text = self._input_text.strip()
        if not text:
            return

        emitter = tts.SynthesizedAudioEmitter(
            event_ch=self._event_ch, request_id=utils.shortuuid()
        )

        try:
            audio = await self._generate(engine, text)
        except asyncio.TimeoutError as exc:
            raise APIConnectionError("Intron TTS timed out") from exc
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"Intron TTS failed: {exc}") from exc

        if not audio:
            return

        pcm, rate = pcm_from_wav(audio)
        bstream = utils.audio.AudioByteStream(
            sample_rate=rate, num_channels=NUM_CHANNELS
        )
        for frame in bstream.write(pcm):
            emitter.push(frame)
        for frame in bstream.flush():
            emitter.push(frame)
        emitter.flush()

    async def _generate(self, engine: IntronTTS, text: str) -> bytes:
        session = engine._ensure_session()
        timeout = aiohttp.ClientTimeout(total=engine._timeout)

        async with session.post(
            engine._url,
            headers={"Authorization": f"Bearer {engine._api_key}"},
            json={
                "text": text,
                "voice_language": engine._language,
                "voice_accent": engine._accent,
                "voice_gender": engine._gender,
                "output_audio_format": "wav",
            },
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                raise APIConnectionError(
                    f"Intron TTS {resp.status}: {(await resp.text())[:200]}"
                )
            payload = await resp.json()

        # generate queues the work and hands back where the file landed, so the
        # audio is a second request away.
        path = (payload.get("data") or {}).get("audio_path")
        if not path:
            logger.warning("Intron TTS returned no audio_path: %s", str(payload)[:200])
            return b""

        async with session.get(path, timeout=timeout) as resp:
            if resp.status != 200:
                raise APIConnectionError(f"Intron TTS fetch {resp.status}")
            return await resp.read()
