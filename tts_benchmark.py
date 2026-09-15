"""
TTS benchmark for the Sahara CodeSwitch Africa Challenge.

The organisers confirmed TTS should be benchmarked, and named five metrics:
hallucination, transcript loss, segment loss, WER, accuracy. The paper they
linked (ASR-FAIRBENCH, Interspeech 2025) is about ASR fairness and does not
define any of them, so the definitions below are the standard round-trip
formulations. They are stated explicitly here so a judge can see exactly what
was measured rather than having to infer it.

THE METHOD
Audio cannot be scored directly, so it is turned back into text: each TTS
synthesises the same phrase, and one ASR — held constant across every provider —
transcribes all of them. Differences in the resulting text are then attributable
to the TTS, because the only thing that changed is which model produced the
audio. This is the standard intelligibility proxy, and its limitation is worth
stating plainly: it measures how well a strong ASR can recover the words, not
how natural the voice sounds to a person. Naturalness needs human MOS ratings.

THE METRICS
  hallucination    Words present in the transcript but not in the input text,
                   over input length. The TTS produced sound that was never
                   asked for. Insertions.
  transcript loss  Words in the input text absent from the transcript, over
                   input length. The TTS dropped them. Deletions.
  segment loss     Fraction of phrases where synthesis effectively failed —
                   no audio, or a transcript under a quarter the expected
                   length. Whole utterances lost rather than words.
  WER              Round-trip word error rate against the input text.
  accuracy         1 - WER, floored at zero, so higher is better.

Usage:
    python tts_benchmark.py                       # all providers
    python tts_benchmark.py --providers intron,finetuned-en
    python tts_benchmark.py --report tts.json --keep-audio
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from benchmark import (
    PHRASE_SET,
    transcribe_deepgram,
    character_error_rate,
    language_for,
    load_samples,
    normalise,
    transcribe_elevenlabs,
    word_error_rate,
)
from config import settings

AUDIO_DIR = Path(__file__).with_name("tts_output")

# The ASR that scores every provider's audio back to text. Held constant on
# purpose — it is the control, and swapping it mid-run would make the
# comparison meaningless.
#
# It MUST be a scorer that can actually transcribe these languages, or it
# charges its own blindness to the TTS. Deepgram was tried as the default to
# save ElevenLabs credits and had to be reverted: the ASR benchmark measured it
# returning EMPTY STRINGS on most Hausa clips (WER 0.89, CER 0.76), so Yoruba
# and Hausa synthesis came back scored as "segment lost" when the audio was
# fine and the scorer simply could not read it.
#
# ElevenLabs Scribe is the only scorer here measured as competent on all four
# languages (WER 0.48, CER 0.21, 100% critical recall). That it is also a TTS
# provider is a cost problem, not a correctness one — use --providers to limit
# what you synthesise rather than weakening the scorer.
SCORERS = {
    "elevenlabs": ("elevenlabs/scribe_v1", transcribe_elevenlabs),
    # English-only in practice. Valid for an English-only comparison, and
    # meaningless on Nigerian-language audio.
    "deepgram": ("deepgram/nova-2-phonecall", transcribe_deepgram),
}
SCORING_ASR_NAME, SCORING_ASR = SCORERS["elevenlabs"]

# A transcript shorter than this fraction of the reference counts as a lost
# segment rather than merely a bad one.
SEGMENT_LOSS_THRESHOLD = 0.25


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def hallucination_rate(reference: str, hypothesis: str) -> float:
    """Words spoken that were never in the text, over reference length."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0
    ref_counts: dict[str, int] = {}
    for word in ref:
        ref_counts[word] = ref_counts.get(word, 0) + 1
    inserted = 0
    for word in hyp:
        if ref_counts.get(word, 0) > 0:
            ref_counts[word] -= 1
        else:
            inserted += 1
    return inserted / len(ref)


def transcript_loss(reference: str, hypothesis: str) -> float:
    """Words in the text that never made it into the audio, over reference length."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0
    hyp_counts: dict[str, int] = {}
    for word in hyp:
        hyp_counts[word] = hyp_counts.get(word, 0) + 1
    missing = 0
    for word in ref:
        if hyp_counts.get(word, 0) > 0:
            hyp_counts[word] -= 1
        else:
            missing += 1
    return missing / len(ref)


def is_segment_lost(reference: str, hypothesis: str) -> bool:
    """Did synthesis fail outright for this phrase?"""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return False
    return len(hyp) < len(ref) * SEGMENT_LOSS_THRESHOLD


# ----------------------------------------------------------------------
# TTS providers
# ----------------------------------------------------------------------


def _openai_tts(base_url: str, model: str, voice: str, api_key: str, text: str) -> bytes:
    payload = json.dumps(
        {"model": model, "voice": voice, "input": text, "response_format": "wav"}
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/audio/speech",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def synth_finetuned_en(text: str, language: str = "en") -> bytes:
    return _openai_tts(
        settings.tts.base_url,
        settings.tts.model,
        settings.tts.voice,
        settings.tts.api_key,
        text,
    )


def synth_multilingua(text: str, language: str = "en") -> bytes:
    """The per-language finetuned-ml endpoints already configured in .env."""
    import os

    hosts = {
        "en": (os.getenv("ENGLISH_TTS_BASE_URL"), os.getenv("ENGLISH_TTS_MODEL"), os.getenv("ENGLISH_TTS_VOICE")),
        "ha": (os.getenv("HAUSA_TTS_BASE_URL"), os.getenv("HAUSA_TTS_MODEL"), os.getenv("HAUSA_TTS_VOICE")),
        "ig": (os.getenv("IGBO_TTS_BASE_URL"), os.getenv("IGBO_TTS_MODEL"), os.getenv("IGBO_TTS_VOICE")),
        "yo": (os.getenv("YORUBA_TTS_BASE_URL"), os.getenv("YORUBA_TTS_MODEL"), os.getenv("YORUBA_TTS_VOICE")),
    }
    base, model, voice = hosts.get(language) or hosts["en"]
    if not base:
        raise RuntimeError(f"No finetuned-ml TTS endpoint configured for '{language}'")
    return _openai_tts(
        base.strip().strip('"'),
        (model or "tts-1").strip().strip('"'),
        (voice or "female").strip().strip('"'),
        settings.tts.api_key,
        text,
    )


async def _intron_tts_stream(text: str, language: str, gender: str = "female") -> bytes:
    """
    Intron/Sahara streaming TTS over WebSocket.

    The protocol has three message types and the order matters:

        INPUT_TEXT_CHUNK   submit 10-100 characters, get a TEXT_CHUNK_ACK
        FETCH_AUDIO_CHUNK  ask for the audio — it is NOT pushed to you
        COMMIT             finalise the session

    Two things cost real time to work out. Audio must be explicitly fetched,
    and the first fetch usually returns an empty audio_base_64 because
    synthesis is still running — so the fetch has to be polled until it comes
    back non-empty. Committing before the audio has been collected finalises
    the session with COMMITTED_AUDIO audio_len=0 and the audio is gone.
    """
    import aiohttp

    # The accent must be one Intron offers for that language; "nigerian" is not
    # one of them. For English it lists yoruba, igbo, hausa, swahili, luganda,
    # zulu, afrikaans, setswana, xhosa, sepedi.
    accents = {"en": "yoruba", "ha": "hausa", "ig": "igbo", "yo": "yoruba", "pcm": "yoruba"}
    accent = accents.get(language, "yoruba")
    # Pidgin has no voice_language of its own; English carries it.
    voice_language = "en" if language == "pcm" else language

    url = (
        f"{settings.tts.intron_tts_url}"
        f"?voice_language={voice_language}&voice_accent={accent}&voice_gender={gender}"
        f"&output_format=wav"
    )
    headers = {"Authorization": f"Bearer {settings.stt.intron_api_key}"}

    # Chunks must be 10-100 characters; split on words to stay inside that.
    chunks: list[str] = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 > 90:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    # A trailing fragment under 10 chars would be rejected; fold it back in.
    # pop() first: `chunks[-2] = f"... {chunks.pop()}"` evaluates the pop before
    # the index, so on a two-chunk split it indexes past the end of a list it
    # has just shortened.
    if len(chunks) > 1 and len(chunks[-1]) < 10:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"

    audio = bytearray()
    async with aiohttp.ClientSession() as http:
        # autoclose=False so a close frame does not tear the connection down
        # before the message explaining it has been read.
        async with http.ws_connect(
            url, headers=headers, timeout=180, autoclose=False
        ) as ws:
            opening = await ws.receive(timeout=60)
            if opening.type is aiohttp.WSMsgType.TEXT:
                data = json.loads(opening.data)
                if data.get("message_type") == "RESOURCE_EXHAUSTED":
                    raise IntronWarmingUp(data.get("message", "voice loading"))
                if "ERROR" in str(data.get("message_type", "")):
                    raise RuntimeError(str(data)[:220])

            for index, chunk in enumerate(chunks, start=1):
                await ws.send_json(
                    {"message_type": "INPUT_TEXT_CHUNK", "text": chunk, "ack_id": index}
                )
                await ws.receive(timeout=60)  # TEXT_CHUNK_ACK

            for chunk_id in range(1, len(chunks) + 1):
                for attempt in range(INTRON_FETCH_ATTEMPTS):
                    await ws.send_json(
                        {"message_type": "FETCH_AUDIO_CHUNK", "chunk_id": chunk_id}
                    )
                    msg = await ws.receive(timeout=90)
                    if msg.type is not aiohttp.WSMsgType.TEXT:
                        raise RuntimeError(
                            f"socket closed during fetch (code={ws.close_code})"
                        )
                    data = json.loads(msg.data)
                    if data.get("message_type") == "RESOURCE_EXHAUSTED":
                        raise IntronWarmingUp(data.get("message", "voice loading"))
                    encoded = data.get("audio_base_64") or ""
                    if encoded:
                        audio.extend(base64.b64decode(encoded))
                        break
                    # Still synthesising — give it a moment and ask again.
                    await asyncio.sleep(INTRON_FETCH_POLL_SECONDS)

            await ws.send_json({"message_type": "COMMIT"})

    if not audio:
        raise RuntimeError("Intron returned no audio after polling")
    return bytes(audio)


# The first fetch almost always returns empty while synthesis runs.
INTRON_FETCH_ATTEMPTS = 10
INTRON_FETCH_POLL_SECONDS = 3.0


class IntronWarmingUp(Exception):
    """Intron is loading the voice for this language; retry after a wait."""


# The server asks for 30 seconds when a voice is cold; give it 35.
INTRON_TTS_WARMUP_SECONDS = 35.0
INTRON_TTS_PACING_SECONDS = 8.0


def synth_intron(text: str, language: str = "en") -> bytes:
    last: Exception | None = None
    for attempt in range(4):
        try:
            audio = asyncio.run(_intron_tts_stream(text, language))
            time.sleep(INTRON_TTS_PACING_SECONDS)
            return audio
        except IntronWarmingUp as exc:
            last = exc
            print(f"        (intron warming up voice for '{language}', waiting 35s)")
            time.sleep(INTRON_TTS_WARMUP_SECONDS)
        except Exception as exc:
            last = exc
            time.sleep(10.0 * (attempt + 1))
    raise RuntimeError(f"intron TTS failed after 4 attempts: {last}")


# ElevenLabs multilingual TTS. Unlike finetuned-en (English only) and finetuned-ml
# (a separate endpoint per language), one model handles every language, so the
# language argument selects nothing here — it is accepted for a uniform
# provider signature. eleven_multilingual_v2 is the model that handles
# non-English text; the monolingual ones mangle Yoruba and Hausa outright.
ELEVENLABS_DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"  # Sarah — clear female voice


def _elevenlabs_voice_id() -> str:
    configured = settings.tts.elevenlabs_voice_id
    if configured:
        return configured
    return ELEVENLABS_DEFAULT_VOICE


def synth_elevenlabs(text: str, language: str = "en") -> bytes:
    """
    Synthesise with ElevenLabs.

    Requests PCM at 24kHz rather than the default mp3 so the bytes go straight
    into _ensure_wav without a decode step, and so the scoring ASR receives the
    same uncompressed format every other provider gives it — mp3 artefacts
    would otherwise be charged to the TTS rather than the codec.
    """
    voice = _elevenlabs_voice_id()
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        "?output_format=pcm_24000"
    )
    payload = json.dumps(
        {
            "text": text,
            "model_id": settings.tts.elevenlabs_tts_model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": settings.tts.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


SYNTHESISERS = {
    "intron": synth_intron,
    "finetuned-en": synth_finetuned_en,
    "finetuned-ml": synth_multilingua,
    "elevenlabs": synth_elevenlabs,
}


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


@dataclass
class TTSResult:
    provider: str
    phrase_id: str
    reference: str
    transcript: str
    seconds: float
    audio_bytes: int
    language: str = "en"
    error: str | None = None

    @property
    def wer(self) -> float:
        return word_error_rate(self.reference, self.transcript)

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.wer)

    @property
    def hallucination(self) -> float:
        return hallucination_rate(self.reference, self.transcript)

    @property
    def loss(self) -> float:
        return transcript_loss(self.reference, self.transcript)

    @property
    def segment_lost(self) -> bool:
        return is_segment_lost(self.reference, self.transcript)


def _ensure_wav(audio: bytes) -> bytes:
    """Wrap raw PCM in a WAV header if the provider streamed it bare."""
    if audio[:4] == b"RIFF":
        return audio
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(audio)
    return buffer.getvalue()


def collect_phrases(
    use_real: bool, per_language: int | None
) -> list[tuple[str, str, str]]:
    """
    The text each TTS is asked to speak, as (id, text, language).

    Real transcripts by default. The synthetic phrase set was only ever a
    smoke test: a TTS that handles "My favourite food is amala" says nothing
    about whether it can speak a sentence that switches language halfway,
    which is the whole question here.
    """
    if not use_real:
        return [(name, text, "en") for name, text in PHRASE_SET]

    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for name, _audio, reference in load_samples():
        language = language_for(name)
        if language == "en":
            continue
        grouped.setdefault(language, []).append((name, reference, language))

    phrases: list[tuple[str, str, str]] = []
    for language in sorted(grouped):
        rows = grouped[language]
        phrases.extend(rows[:per_language] if per_language else rows)
    return phrases


def run(
    providers: list[str],
    report: str | None,
    keep_audio: bool,
    phrases: list[tuple[str, str, str]],
) -> None:
    if keep_audio:
        AUDIO_DIR.mkdir(exist_ok=True)

    print(f"\n  {len(phrases)} phrases x {len(providers)} TTS providers")
    print(f"  Scoring ASR (held constant): {SCORING_ASR_NAME}\n")

    results: list[TTSResult] = []
    for provider in providers:
        synth = SYNTHESISERS[provider]
        print(f"  --- {provider}")
        for name, phrase, language in phrases:
            started = time.monotonic()
            try:
                audio = _ensure_wav(synth(phrase, language))
                transcript = SCORING_ASR(audio)
                error, size = None, len(audio)
            except Exception as exc:
                audio, transcript = b"", ""
                error, size = f"{type(exc).__name__}: {exc}", 0
            elapsed = time.monotonic() - started

            if keep_audio and audio:
                (AUDIO_DIR / f"{provider}_{name}.wav").write_bytes(audio)

            result = TTSResult(
                provider, name, phrase, transcript, elapsed, size, language, error
            )
            results.append(result)

            if error:
                print(f"      {name:<20} FAILED  {error[:64]}")
            else:
                flag = "  SEGMENT LOST" if result.segment_lost else ""
                print(
                    f"      {name:<20} wer={result.wer:4.2f} hall={result.hallucination:4.2f} "
                    f"loss={result.loss:4.2f} {elapsed:5.1f}s {transcript[:34]!r}{flag}"
                )
        print()

    summarise(results, providers)
    if report:
        Path(report).write_text(
            json.dumps(
                {
                    "scoringAsr": SCORING_ASR_NAME,
                    "method": "round-trip: TTS -> fixed ASR -> compare to input text",
                    "results": [
                        {
                            "provider": r.provider,
                            "phrase": r.phrase_id,
                            # Which clip the reference text came from, and where
                            # the synthesised audio was written. Without these a
                            # row is a number with no way back to the sound it
                            # describes, and nobody can check it by listening.
                            "language": r.language,
                            "sourceTranscript": f"samples/{r.phrase_id}.txt",
                            "sourceAudio": f"samples/{r.phrase_id}.wav",
                            "synthesisedAudio": (
                                f"{AUDIO_DIR.name}/{r.provider}_{r.phrase_id}.wav"
                            ),
                            "dataset": (
                                "AfriSwitch"
                                if r.phrase_id.startswith("afriswitch_")
                                else "AfriSwitchCare"
                                if r.phrase_id.startswith("afriswitchcare_")
                                else "synthetic"
                            ),
                            "reference": r.reference,
                            "transcript": r.transcript,
                            "wer": round(r.wer, 4),
                            "accuracy": round(r.accuracy, 4),
                            "hallucination": round(r.hallucination, 4),
                            "transcriptLoss": round(r.loss, 4),
                            "segmentLost": r.segment_lost,
                            "seconds": round(r.seconds, 2),
                            "audioBytes": r.audio_bytes,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  Written to {report}\n")


def summarise(results: list[TTSResult], providers: list[str]) -> None:
    print("  " + "=" * 82)
    print(
        f"  {'provider':<14}{'WER':>7}{'accuracy':>10}{'halluc':>9}"
        f"{'loss':>8}{'seg loss':>10}{'latency':>10}{'fails':>7}"
    )
    print("  " + "=" * 82)
    for provider in providers:
        rows = [r for r in results if r.provider == provider]
        ok = [r for r in rows if not r.error]
        if not ok:
            print(f"  {provider:<14}{'—':>7}{'—':>10}{'—':>9}{'—':>8}{'—':>10}{'—':>10}{len(rows):>7}")
            continue
        seg = sum(1 for r in ok if r.segment_lost) / len(ok)
        print(
            f"  {provider:<14}"
            f"{statistics.mean(r.wer for r in ok):>7.2f}"
            f"{statistics.mean(r.accuracy for r in ok):>9.0%}"
            f"{statistics.mean(r.hallucination for r in ok):>9.2f}"
            f"{statistics.mean(r.loss for r in ok):>8.2f}"
            f"{seg:>9.0%}"
            f"{statistics.mean(r.seconds for r in ok):>9.1f}s"
            f"{len(rows) - len(ok):>7}"
        )
    print("  " + "=" * 82)
    print(
        "\n  Round-trip scoring: every provider's audio is transcribed by the same\n"
        f"  ASR ({SCORING_ASR_NAME}), so differences are attributable to the TTS.\n"
        "  It measures intelligibility, not naturalness — how reliably the words\n"
        "  survive a phone line, which is what a voice agent depends on. Judging\n"
        "  how pleasant a voice sounds needs human MOS ratings.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TTS providers.")
    parser.add_argument(
        "--providers",
        # finetuned-ml only by default. finetuned-en is the ENGLISH voice endpoint
        # (finetuned-en-tts.finetuned.com) with no Hausa, Igbo or Yoruba voices, so
        # scoring it on code-switched Nigerian text measures nothing about the
        # model — the same mistake as running an English ASR on Hausa audio.
        # It stays selectable for an English-only comparison.
        #
        # intron is opt-in too: it rate-limits hard and warms a language model
        # per session, so it belongs in its own run.
        default="finetuned-ml,elevenlabs",
        help="comma separated: finetuned-ml,elevenlabs,finetuned-en,intron",
    )
    parser.add_argument("--report", help="write full results to this JSON file")
    parser.add_argument(
        "--keep-audio", action="store_true", help=f"save synthesised wavs to {AUDIO_DIR.name}/"
    )
    parser.add_argument(
        "--per-language",
        type=int,
        metavar="N",
        help="N real transcripts per language (default: all)",
    )
    parser.add_argument(
        "--scorer",
        choices=sorted(SCORERS),
        default="elevenlabs",
        help="ASR used to score the synthesised audio. Default elevenlabs — "
        "deepgram cannot transcribe Hausa/Yoruba/Igbo and will report good "
        "audio as lost",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the built-in English phrase set instead of real transcripts",
    )
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in providers if p not in SYNTHESISERS]
    if unknown:
        print(f"\n  Unknown provider(s): {', '.join(unknown)}\n")
        sys.exit(1)
    global SCORING_ASR, SCORING_ASR_NAME
    SCORING_ASR_NAME, SCORING_ASR = SCORERS[args.scorer]

    phrases = collect_phrases(not args.synthetic, args.per_language)
    if not phrases:
        print("\n  No phrases. Fetch samples first, or pass --synthetic.\n")
        sys.exit(1)
    run(providers, args.report, args.keep_audio, phrases)


if __name__ == "__main__":
    main()
