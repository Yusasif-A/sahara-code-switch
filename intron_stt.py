"""
Intron (Sahara) speech-to-text as a LiveKit STT plugin.

WHY THIS IS A SEPARATE MODEL CHOICE
Deepgram, ElevenLabs and the rest treat a call as being *in* one language. A
Nigerian caller is not. They speak English, drop into Yoruba mid-sentence, and
come back, and a monolingual model handles that by discarding whichever half it
was not expecting. Intron ships models trained on exactly that mixture, and they
are selected by language code:

    yo   Yoruba-English      ha   Hausa-English
    ig   Igbo-English        pcm  Pidgin-English

Those are not "Yoruba" and "Igbo" — they are the code-switched pairs. Passing
`en` gets a monolingual English model that silently throws the Yoruba away,
which reads as a bad transcript rather than as wrong configuration. Set
STT_LANGUAGE to the pair you expect, not to the language the call opens in.

STREAMING, AND WHY IT IS WORTH THE EXTRA CODE
The obvious way to wrap this API is to buffer a whole utterance, send it, COMMIT,
and wait — which is what the framework's own StreamAdapter does with any
non-streaming STT. Measured on a 9s Yoruba clip, that makes the caller wait 8.7s
of silence after they stop talking, because none of the transcription work
starts until they have finished.

Intron does not need that. It returns PARTIAL_TRANSCRIPT while audio is still
arriving, so if the socket is opened at start-of-speech and fed live, most of
the work is already done by the time the caller stops. The same clip then
returns its final transcript 4.58s after COMMIT — a little under half.

So this module streams when it is given a VAD, and falls back to the batch path
when it is not. The VAD decides where an utterance ends; Intron ends a session
on COMMIT, so it is one socket per utterance either way. The difference is
whether the audio was already sent.

PROTOCOL
Audio parameters in the query string; base64 PCM16 in INPUT_AUDIO_CHUNK; COMMIT;
read to COMMITTED_TRANSCRIPT. Chunks must be between 1 and 32 KB — a raw LiveKit
frame is ~320 bytes, so frames are accumulated before sending or the server
answers CHUNK_SIZE_TOO_SMALL and that audio is lost.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque

import aiohttp
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    stt,
    utils,
    vad as agents_vad,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

logger = logging.getLogger("intron-stt")

# Language codes that carry a code-switched model, per
# docs.voice.intron.io/docs/stt/supported-languages. Anything else is
# monolingual — a legitimate choice, but not what this project is for.
CODE_SWITCHED = frozenset(
    {"af", "ak", "am", "ha", "ig", "lg", "pcm", "rw", "sw", "wo", "yo", "zu"}
)

# Everything is resampled to this before it reaches Intron, so the session URL
# never has to change and narrowband call audio is not sent at 48kHz for nothing.
SAMPLE_RATE = 16_000

# 200ms at 16kHz mono PCM16. Comfortably over the 1 KB floor and under the 32 KB
# ceiling, and small enough that partials keep arriving during speech.
CHUNK_BYTES = SAMPLE_RATE * 2 // 5

# Audio kept from before the socket is ready, so the opening words are not lost.
# Two delays stack here: Silero reports start-of-speech only once it is
# confident, and Intron takes ~2s to return SESSION_CREATED. Three seconds
# covers both; anything shorter silently truncates the front of the utterance.
PREROLL_BYTES = SAMPLE_RATE * 2 * 3  # 3 seconds


class IntronSTT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        language: str = "yo",
        vad: agents_vad.VAD | None = None,
        timeout: float = 45.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        # With a VAD we can open the socket at start-of-speech and feed it live,
        # so we advertise streaming and do our own endpointing. Without one the
        # framework wraps us in its StreamAdapter, which is correct but waits
        # for the utterance to finish before sending anything.
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=vad is not None, interim_results=vad is not None
            )
        )
        self._api_key = api_key
        self._url = url
        self._language = language
        self._vad = vad
        # LiveKit's default API timeout is 10s, which Intron routinely exceeds:
        # it loads each language model on demand, so the first utterance in a
        # cold language can take half a minute. Using the framework default
        # turns that warm-up into four failed retries and a dead call.
        self._timeout = timeout
        self._session = http_session
        self._owns_session = http_session is None

        if language not in CODE_SWITCHED:
            logger.warning(
                "STT_LANGUAGE=%s has no code-switched model, so a caller who "
                "mixes in Yoruba will lose that half of the sentence. "
                "Code-switched codes: %s",
                language,
                ", ".join(sorted(CODE_SWITCHED)),
            )
        logger.info(
            "Intron STT: language=%s, mode=%s",
            language,
            "streaming" if vad is not None else "batch (no VAD supplied)",
        )

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def _session_url(self, language: str) -> str:
        return (
            f"{self._url}?sample_rate={SAMPLE_RATE}&bit_rate=16"
            f"&num_channels=1&use_language_asr_input={language}"
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def aclose(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None
        await super().aclose()

    # ------------------------------------------------------------------
    # Batch path — used when no VAD was supplied
    # ------------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if isinstance(language, str) and language else self._language
        # This livekit build has no remix_and_resample, so the session is opened
        # at whatever rate the frames arrive at rather than forcing 16kHz.
        frame = rtc.combine_audio_frames(buffer)
        timeout = max(self._timeout, conn_options.timeout)

        text = ""
        url = (
            f"{self._url}?sample_rate={frame.sample_rate}&bit_rate=16"
            f"&num_channels={frame.num_channels}&use_language_asr_input={lang}"
        )
        async with self._ensure_session().ws_connect(
            url, headers=self._headers, autoclose=False
        ) as ws:
            pcm = frame.data.tobytes()
            for index in range(0, len(pcm), CHUNK_BYTES):
                await _send_audio(ws, pcm[index : index + CHUNK_BYTES], index)
            await ws.send_json({"message_type": "COMMIT"})
            text = await _read_until_final(ws, timeout=timeout, language=lang)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=lang, text=text)],
        )

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        if self._vad is None:
            return super().stream(language=language, conn_options=conn_options)
        lang = language if isinstance(language, str) and language else self._language
        return IntronSpeechStream(
            stt=self, vad=self._vad, language=lang, conn_options=conn_options
        )


class IntronSpeechStream(stt.RecognizeStream):
    """
    One Intron socket per utterance, opened the moment the caller starts talking.

    The VAD is used only for boundaries. Audio goes to Intron as it arrives, so
    by the time end-of-speech fires the server has already transcribed most of
    the utterance and COMMIT returns quickly.
    """

    def __init__(
        self,
        *,
        stt: IntronSTT,
        vad: agents_vad.VAD,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        # sample_rate here makes the base class resample every frame for us, so
        # the session URL can hard-code 16kHz.
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._intron = stt
        self._vad_stream = vad.stream()
        self._language = language

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pump: asyncio.Task | None = None
        self._pending = bytearray()
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0
        self._acks = 0
        self._acked = 0
        self._drained = asyncio.Event()
        self._final: asyncio.Future[str] | None = None
        self._speaking = False
        self._opening: asyncio.Task | None = None
        # Set the instant COMMIT goes out. Audio keeps arriving from the
        # forwarding task while we wait for the final transcript, and Intron
        # rejects anything sent after a commit with INPUT_ERROR — which the
        # reader then treats as a failed utterance, losing a transcript that
        # was otherwise fine.
        self._committed = False

    async def _run(self) -> None:
        async def forward() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    self._vad_stream.flush()
                    continue
                self._vad_stream.push_frame(item)
                await self._on_frame(bytes(item.data))
            self._vad_stream.end_input()

        async def drive() -> None:
            async for event in self._vad_stream:
                if event.type is agents_vad.VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(stt.SpeechEventType.START_OF_SPEECH)
                    )
                    await self._start_utterance()
                elif event.type is agents_vad.VADEventType.END_OF_SPEECH:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(stt.SpeechEventType.END_OF_SPEECH)
                    )
                    await self._commit()

        # Open the first session before anyone speaks. A session costs ~2s to
        # establish, and on a short turn ("yes", "abeg help me") that is longer
        # than the turn itself — the socket was becoming ready just as COMMIT
        # went out, and Intron answered those commits with nothing at all. An
        # idle session survives comfortably, so the handshake is paid during
        # silence instead of inside the caller's latency budget.
        self._begin_open()

        tasks = [
            asyncio.create_task(forward(), name="intron_forward"),
            asyncio.create_task(drive(), name="intron_drive"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await self._close()
            await self._vad_stream.aclose()

    # -- audio in -------------------------------------------------------

    async def _on_frame(self, pcm: bytes) -> None:
        if not self._speaking or self._committed:
            # Between utterances. Keep a rolling window so the opening syllable
            # survives the VAD's decision delay, and so nothing is lost if the
            # socket is still being established.
            self._preroll.append(pcm)
            self._preroll_bytes += len(pcm)
            while self._preroll_bytes > PREROLL_BYTES:
                self._preroll_bytes -= len(self._preroll.popleft())
            return

        if self._ws is None:
            # Speaking, but the socket is not up yet. Hold it; _start_utterance
            # flushes this the moment the session is ready.
            self._pending += pcm
            return

        self._pending += pcm
        while len(self._pending) >= CHUNK_BYTES:
            block, self._pending = (
                bytes(self._pending[:CHUNK_BYTES]),
                self._pending[CHUNK_BYTES:],
            )
            await self._send(block)

    async def _send(self, block: bytes) -> None:
        if self._ws is None or self._ws.closed:
            return
        self._acks += 1
        self._drained.clear()
        try:
            await _send_audio(self._ws, block, self._acks)
        except (aiohttp.ClientError, ConnectionResetError) as exc:
            # Lose the rest of this turn, but not the call: open a replacement
            # straight away so the caller's next turn has a session waiting.
            logger.warning("Intron socket dropped mid-utterance: %s", exc)
            await self._close()
            self._begin_open()

    # -- session lifecycle ----------------------------------------------

    def _begin_open(self) -> None:
        """Start establishing a session in the background, if there is none."""
        if self._ws is None and self._opening is None:
            self._opening = asyncio.create_task(self._open(), name="intron_open")

    async def _start_utterance(self) -> None:
        """Speech has begun: make sure the socket is up, then send the preroll."""
        self._speaking = True
        self._begin_open()
        if self._opening is not None:
            try:
                await self._opening
            except Exception as exc:  # noqa: BLE001 — a dead socket must not kill the call
                logger.warning("Intron session unavailable for this turn: %s", exc)
                return

        preroll, self._preroll, self._preroll_bytes = b"".join(self._preroll), deque(), 0
        buffered, self._pending = bytes(self._pending), bytearray()
        await self._on_frame(preroll + buffered)

    async def _open(self) -> None:
        if self._ws is not None:
            return
        try:
            self._ws = await self._intron._ensure_session().ws_connect(
                self._intron._session_url(self._language),
                headers=self._intron._headers,
                autoclose=False,
            )
        except aiohttp.ClientError as exc:
            self._opening = None
            raise APIConnectionError(f"Intron connect failed: {exc}") from exc

        finally:
            self._opening = None

        self._acks = 0
        self._acked = 0
        self._drained.set()
        self._committed = False
        self._final = asyncio.get_running_loop().create_future()
        self._pump = asyncio.create_task(self._reader(), name="intron_reader")

    async def _reader(self) -> None:
        """Read partials as they arrive and resolve the final transcript."""
        assert self._ws is not None
        text = ""
        try:
            while True:
                msg = await self._ws.receive(timeout=self._intron._timeout)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    raise APIConnectionError(
                        f"socket closed (code={self._ws.close_code})"
                    )
                data = json.loads(msg.data)
                kind = data.get("message_type", "")
                if kind == "AUDIO_CHUNK_ACK":
                    self._acked += 1
                    if self._acked >= self._acks:
                        self._drained.set()
                    continue
                logger.debug("intron <- %s %s", kind, str(data)[:160])

                if kind == "PARTIAL_TRANSCRIPT":
                    text = data.get("transcript") or text
                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(language=self._language, text=text)
                                ],
                            )
                        )
                elif kind == "COMMITTED_TRANSCRIPT":
                    # transcript_text on the committed message; `transcript`
                    # only on partials. Reading the wrong key returns empty
                    # strings and makes a working model look broken.
                    final = data.get("transcript_text") or data.get("transcript") or text
                    _resolve(self._final, final)
                    return
                elif kind == "RESOURCE_EXHAUSTED":
                    # Intron loads each language model on demand; the first
                    # request for a cold language gets this and is told to wait
                    # ~30s. Mid-call there is nothing useful to wait for, so
                    # drop this utterance rather than stall the caller.
                    logger.warning(
                        "Intron warming up '%s'; dropped one utterance (%s)",
                        self._language,
                        data.get("message", ""),
                    )
                    _resolve(self._final, "")
                    return
                elif kind == "CHUNK_SIZE_TOO_SMALL":
                    # Audio was sent below the 1 KB floor and discarded. Not
                    # fatal, but it means a slice of speech never reached the
                    # model, so it should be visible rather than swallowed.
                    logger.warning("Intron rejected a chunk as too small")
                elif "ERROR" in kind:
                    logger.error("Intron %s: %s", kind, str(data)[:200])
                    _resolve(self._final, text)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — must not kill the call
            logger.warning("Intron reader stopped: %s", exc)
            _resolve(self._final, text)

    async def _commit(self) -> None:
        self._speaking = False
        if self._ws is None:
            self._pending = bytearray()
            return
        # Anything left under the chunk floor is padded rather than dropped;
        # the tail of an utterance is often where the verb lives.
        if self._pending:
            tail = bytes(self._pending)
            if len(tail) < 1024:
                tail += b"\x00" * (1024 - len(tail))
            await self._send(tail)
            self._pending = bytearray()

        # Wait for the audio to be acknowledged before committing. A COMMIT
        # that lands while Intron is still working through a burst of chunks is
        # silently dropped: partials keep coming, COMMITTED_TRANSCRIPT never
        # does, and the utterance is lost after a full timeout. This is only
        # ever a wait of a few hundred milliseconds, and it is the difference
        # between a transcript and nothing at all.
        if not self._drained.is_set():
            try:
                await asyncio.wait_for(self._drained.wait(), 5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Intron acked %d of %d chunks; committing anyway",
                    self._acked,
                    self._acks,
                )

        self._committed = True
        try:
            await self._ws.send_json({"message_type": "COMMIT"})
        except (aiohttp.ClientError, ConnectionResetError):
            pass

        text = ""
        if self._final is not None:
            try:
                text = await asyncio.wait_for(self._final, self._intron._timeout)
            except asyncio.TimeoutError:
                logger.warning("Intron did not return a transcript in time")

        await self._close()
        # Warm the next session now, during the silence while the agent is
        # replying, so the caller's next turn does not pay for the handshake.
        self._begin_open()

        if text.strip():
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=self._language, text=text)],
                )
            )

    async def _close(self) -> None:
        if self._pump is not None:
            await utils.aio.cancel_and_wait(self._pump)
            self._pump = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        self._final = None
        self._committed = False
        self._pending = bytearray()


# ----------------------------------------------------------------------
# Shared wire helpers
# ----------------------------------------------------------------------


async def _send_audio(ws, block: bytes, ack_id: int) -> None:
    await ws.send_json(
        {
            "message_type": "INPUT_AUDIO_CHUNK",
            "audio_base_64": base64.b64encode(block).decode(),
            "ack_id": ack_id,
        }
    )


async def _read_until_final(ws, *, timeout: float, language: str) -> str:
    text = ""
    while True:
        msg = await ws.receive(timeout=timeout)
        if msg.type is not aiohttp.WSMsgType.TEXT:
            raise APIConnectionError(f"socket closed (code={ws.close_code})")
        data = json.loads(msg.data)
        kind = data.get("message_type", "")
        if kind == "COMMITTED_TRANSCRIPT":
            return data.get("transcript_text") or data.get("transcript") or text
        if kind == "PARTIAL_TRANSCRIPT":
            text = data.get("transcript") or text
        elif kind == "RESOURCE_EXHAUSTED":
            logger.warning("Intron warming up '%s'; dropped one utterance", language)
            return ""
        elif "ERROR" in kind:
            raise APIConnectionError(f"{kind}: {str(data)[:200]}")


def _resolve(future: asyncio.Future[str] | None, value: str) -> None:
    if future is not None and not future.done():
        future.set_result(value)
