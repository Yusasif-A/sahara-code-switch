"""
Speech model benchmark for the Sahara CodeSwitch Africa Challenge.

The challenge requires benchmarking at least three speech models, one of which
must be the Sahara/Intron API. This runs the same audio through each and reports
word error rate, plus the two things that actually matter for a fraud call:

  - latency, because a hesitant agent on an urgent call loses the customer
  - keyword recall on the words the agent must act on — the security answers and
    the confirmations. A model can score a respectable WER while missing every
    word that decides whether a card gets frozen, which makes plain WER a
    misleading headline for a voice agent.

Usage:
    python benchmark.py --make-samples     # synthesise clips from the phrase set
    python benchmark.py                    # run every provider over samples/
    python benchmark.py --providers deepgram,intron
    python benchmark.py --report results.json

Put real recordings in samples/ as pairs -- clip.wav plus clip.txt holding the
ground truth. Real code-switched speech from real handsets is what the judges
care about; the synthesised set is only a smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import settings

# Windows consoles default to cp1252, which cannot encode Yoruba, Hausa or Igbo
# characters — printing a transcript containing "ọ" raises UnicodeEncodeError and
# kills the run. Since the entire point here is African-language text, force
# UTF-8 on stdout and fall back to replacement characters rather than crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

CRLF = chr(13) + chr(10)

SAMPLES_DIR = Path(__file__).with_name("samples")

# Phrases chosen for what this agent must actually hear: the security answers,
# the confirmations that trigger a card freeze, and code-switched utterances of
# the kind a Nigerian customer really produces under stress.
PHRASE_SET: list[tuple[str, str]] = [
    ("answer_food", "My favourite food is amala."),
    ("answer_colour", "My favourite colour is green."),
    ("answer_school", "My first school was Saint Mary."),
    ("deny_transaction", "No, that was not me. I have never been to Ukraine."),
    ("confirm_freeze", "Yes, please freeze the card now."),
    ("ask_human", "I want to speak to a human being please."),
    ("codeswitch_pidgin", "Abeg, I no do that transaction at all."),
    ("codeswitch_yoruba", "Emi ko ra nkan yen. It was not me."),
    ("codeswitch_hausa", "Ba ni yi wannan ba. Please freeze my card."),
    ("codeswitch_igbo", "O bughi m. Somebody used my card."),
]

# Words that change what the agent does. Missing "amala" fails verification;
# missing "freeze" loses the protective action.
CRITICAL_WORDS = {
    "amala", "green", "saint", "mary", "freeze", "card", "not", "no",
    "human", "ukraine", "yes",
}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def normalise(text: str) -> list[str]:
    keep = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return keep.split()


def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance over any sequence — words for WER, chars for CER."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                previous[j - 1]
                if r == h
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, normalized: bool = True) -> float:
    """
    Word error rate.

    Returns edits/reference_words, so 0.0 is perfect and values above 1.0 happen
    when a model emits more wrong words than were actually spoken.

    normalized=True lowercases and strips punctuation, which is the fair
    comparison across providers since some punctuate and some do not.
    Intron's AfriHealth MultiBench reports both, so both are available here.
    """
    if normalized:
        ref, hyp = normalise(reference), normalise(hypothesis)
    else:
        ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str, normalized: bool = True) -> float:
    """
    Character error rate — the second metric the official benchmark reports.

    CER matters far more than WER for code-switched African speech. Yoruba and
    Hausa carry meaning in diacritics and small morphemes, so a model can score
    a whole word wrong under WER for a single missing tone mark. CER shows how
    close it actually got, which is the difference between a usable transcript
    and a useless one.
    """
    if normalized:
        ref = " ".join(normalise(reference))
        hyp = " ".join(normalise(hypothesis))
    else:
        ref, hyp = reference, hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(list(ref), list(hyp)) / len(ref)


def critical_recall(reference: str, hypothesis: str) -> float | None:
    """
    Fraction of decision-critical words that survived transcription.

    None when the reference contains none, so it does not distort the average.
    """
    ref, hyp = set(normalise(reference)), set(normalise(hypothesis))
    wanted = ref & CRITICAL_WORDS
    if not wanted:
        return None
    return len(wanted & hyp) / len(wanted)


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------


@dataclass
class Result:
    provider: str
    sample: str
    reference: str
    hypothesis: str
    seconds: float
    error: str | None = None

    @property
    def wer(self) -> float:
        return word_error_rate(self.reference, self.hypothesis)

    @property
    def wer_raw(self) -> float:
        return word_error_rate(self.reference, self.hypothesis, normalized=False)

    @property
    def cer(self) -> float:
        return character_error_rate(self.reference, self.hypothesis)

    @property
    def cer_raw(self) -> float:
        return character_error_rate(self.reference, self.hypothesis, normalized=False)

    @property
    def recall(self) -> float | None:
        return critical_recall(self.reference, self.hypothesis)


# Deepgram has no Yoruba, Hausa, Igbo or Pidgin model, so it is given English -
# its best available shot at this audio, and the fairest way to score it.
#
# Hard-coded rather than read from STT_LANGUAGE, which now holds the Intron
# code-switch pair ("yo"). Passing that to Deepgram asks for a model it does not
# have, and a benchmark row would fail or come back empty for a reason that has
# nothing to do with the model's ability.
DEEPGRAM_BENCHMARK_LANGUAGE = "en"


def transcribe_deepgram(audio: bytes) -> str:
    url = (
        "https://api.deepgram.com/v1/listen"
        f"?model={settings.stt.deepgram_model}&smart_format=true&punctuate=true"
        f"&language={DEEPGRAM_BENCHMARK_LANGUAGE}"
    )
    req = urllib.request.Request(
        url,
        data=audio,
        headers={
            "Authorization": f"Token {settings.stt.deepgram_api_key}",
            "Content-Type": "audio/wav",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        out = json.load(resp)
    return out["results"]["channels"][0]["alternatives"][0]["transcript"]


def transcribe_linguacenter(audio: bytes) -> str:
    """OpenAI-compatible multipart upload."""
    boundary = "----benchmarkboundary"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
        f"{settings.stt.model}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
        audio,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        settings.stt.base_url.rstrip("/") + "/audio/transcriptions",
        data=b"".join(parts),
        headers={
            "Authorization": f"Bearer {settings.stt.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp).get("text", "")


async def _intron_stream(audio: bytes, language: str = "en") -> str:
    """
    Intron/Sahara streaming STT over WebSocket.

    Protocol per docs.voice.intron.io: connect with the audio parameters as query
    string, push base64 PCM16 in INPUT_AUDIO_CHUNK messages, then COMMIT and read
    until COMMITTED_TRANSCRIPT. Chunks must stay between 1 and 32 KB.
    """
    import aiohttp

    with wave.open(__import__("io").BytesIO(audio)) as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        pcm = wav.readframes(wav.getnframes())

    url = (
        f"{settings.stt.intron_stt_url}"
        f"?sample_rate={sample_rate}&bit_rate=16&num_channels={channels}"
        f"&use_language_asr_input={language}"
    )
    headers = {"Authorization": f"Bearer {settings.stt.intron_api_key}"}

    transcript = ""
    chunk_size = 16000  # comfortably inside the 32 KB ceiling
    async with aiohttp.ClientSession() as http:
        # autoclose=False so a close frame does not tear the connection down
        # before the message that explains it has been read. With the default
        # the RESOURCE_EXHAUSTED text was being lost and the failure surfaced
        # as a bare code-1002 close — which reads as a client protocol bug and
        # sent me hunting in the wrong place for hours. The server was always
        # saying why; this client was hanging up before listening.
        async with http.ws_connect(
            url, headers=headers, timeout=60, autoclose=False
        ) as ws:
            for index in range(0, len(pcm), chunk_size):
                await ws.send_json(
                    {
                        "message_type": "INPUT_AUDIO_CHUNK",
                        "audio_base_64": base64.b64encode(
                            pcm[index : index + chunk_size]
                        ).decode(),
                        "ack_id": index // chunk_size + 1,
                    }
                )
            await ws.send_json({"message_type": "COMMIT"})

            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                msg = await ws.receive(timeout=30)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    # Silently breaking here hid the real problem: the server
                    # was closing the socket straight after SESSION_CREATED and
                    # every sample returned an empty string, which looked like
                    # a transcription failure rather than a connection one.
                    raise RuntimeError(
                        f"socket closed before transcript "
                        f"({msg.type.name}, code={ws.close_code}, data={str(msg.data)[:120]})"
                    )
                data = json.loads(msg.data)
                kind = data.get("message_type", "")
                if kind == "RESOURCE_EXHAUSTED":
                    # Intron loads a language model on demand. Asking for a
                    # language it has not warmed up returns this and closes the
                    # socket, telling you to wait ~30s. It is a cold start, not
                    # a failure, so it gets its own exception and a longer wait.
                    raise IntronWarmingUp(data.get("message", kind))
                if "ERROR" in kind:
                    raise RuntimeError(f"{kind}: {data}")
                if kind == "COMMITTED_TRANSCRIPT":
                    # The committed message uses transcript_text; only the
                    # partial one uses `transcript`. Reading the wrong key
                    # returned empty strings and made Intron look broken when
                    # it was in fact transcribing correctly.
                    transcript = data.get("transcript_text") or data.get("transcript") or ""
                    break
                if kind == "PARTIAL_TRANSCRIPT":
                    transcript = data.get("transcript") or transcript
    return transcript


class IntronWarmingUp(Exception):
    """Intron is loading the language model; the call should be retried."""


def transcribe_intron(audio: bytes, language: str = "en") -> str:
    """
    Transcribe via Intron, pacing and retrying around its rate limit.

    Opening sessions back to back gets the socket closed with code 1002 straight
    after SESSION_CREATED. Run singly the same audio transcribes perfectly, so
    this is throughput limiting rather than anything wrong with the request.
    A short gap plus one backoff retry is enough for a benchmark run.
    """
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            result = asyncio.run(_intron_stream(audio, language))
            time.sleep(INTRON_PACING_SECONDS)
            return result
        except IntronWarmingUp as exc:
            # The server asks for 30 seconds; give it 35 and do not count this
            # against the ordinary backoff, since nothing is wrong.
            last_error = exc
            print(f"        (intron warming up '{language}', waiting 35s)")
            time.sleep(INTRON_WARMUP_SECONDS)
        except Exception as exc:
            last_error = exc
            time.sleep(INTRON_RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"intron failed after 4 attempts: {last_error}")


# Gap between Intron sessions, and extra backoff per retry. Measured, not
# guessed: at 3s roughly a third of a ten-sample run still hit code 1002.
INTRON_PACING_SECONDS = 8.0
INTRON_RETRY_BACKOFF = 10.0
# The server explicitly asks for 30 seconds when a language is cold.
INTRON_WARMUP_SECONDS = 35.0


def transcribe_elevenlabs(audio: bytes, language: str = "en") -> str:
    """
    ElevenLabs Scribe.

    Authenticates with an xi-api-key header rather than a bearer token, and
    takes the audio as multipart. Scribe accepts an ISO-639-3 language hint,
    but the Nigerian code-switched codes used elsewhere here (pcm, yo, ha, ig)
    are not all recognised — and a wrong hint is worse than none, so it is sent
    only for plain English and Scribe detects the rest itself. That is arguably
    the fairer test: on code-switched audio there is no single correct language
    to declare in the first place.
    """
    boundary = "----elevenlabsboundary"
    dash = "--"
    quote = chr(34)

    def field(name: str, value: str) -> bytes:
        head = dash + boundary + CRLF
        head += "Content-Disposition: form-data; name=" + quote + name + quote
        head += CRLF + CRLF + value + CRLF
        return head.encode()

    parts = [field("model_id", settings.stt.elevenlabs_stt_model)]
    if language == "en":
        parts.append(field("language_code", "eng"))

    file_head = dash + boundary + CRLF
    file_head += "Content-Disposition: form-data; name=" + quote + "file" + quote
    file_head += "; filename=" + quote + "a.wav" + quote + CRLF
    file_head += "Content-Type: audio/wav" + CRLF + CRLF
    parts.append(file_head.encode())
    parts.append(audio)
    parts.append((CRLF + dash + boundary + dash + CRLF).encode())

    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text",
        data=b"".join(parts),
        headers={
            "xi-api-key": settings.stt.elevenlabs_api_key,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp).get("text", "")


# PublicaAI's multilingual STT. The host is named stts-yoruba for historical
# reasons — it serves every language, selected by the path segment:
#
#     /en/v1   English        /ha/v1   Hausa
#     /yo/v1   Yoruba         /v1      Igbo  (no language code — Igbo is default)
#
# Igbo deliberately has no segment, which is why IGBO_STT_API_URL in .env ends
# at /v1 and is correct as written rather than missing something.
PUBLICAAI_STT_HOST = "https://stts-yoruba.publicaai.com"

# Nigerian Pidgin has no endpoint of its own; English is the closest model,
# Pidgin being English-lexified, rather than an arbitrary pick.
PUBLICAAI_LANGUAGE_PATH = {
    "en": "/en",
    "ha": "/ha",
    "yo": "/yo",
    "ig": "",
    "pcm": "/en",
}


def transcribe_publicaai(audio: bytes, language: str = "en") -> str:
    """PublicaAI multilingual STT, routed to the endpoint for the clip's language."""
    import os

    host = (os.getenv("PUBLICAAI_STT_HOST") or PUBLICAAI_STT_HOST).strip().strip(chr(34))
    segment = PUBLICAAI_LANGUAGE_PATH.get(language, "/en")
    base = host.rstrip("/") + segment + "/v1"

    boundary = "----publicaaiboundary"
    dash = "--"
    quote = chr(34)

    def field(name: str, value: str) -> bytes:
        head = dash + boundary + CRLF
        head += "Content-Disposition: form-data; name=" + quote + name + quote
        head += CRLF + CRLF + value + CRLF
        return head.encode()

    head_file = dash + boundary + CRLF
    head_file += "Content-Disposition: form-data; name=" + quote + "file" + quote
    head_file += "; filename=" + quote + "a.wav" + quote + CRLF
    head_file += "Content-Type: audio/wav" + CRLF + CRLF

    body = b"".join(
        [
            field("model", "whisper-1"),
            head_file.encode(),
            audio,
            (CRLF + dash + boundary + dash + CRLF).encode(),
        ]
    )
    req = urllib.request.Request(
        base + "/audio/transcriptions",
        data=body,
        headers={
            "Authorization": "Bearer " + settings.stt.api_key,
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp).get("text", "")


TRANSCRIBERS = {
    "deepgram": transcribe_deepgram,
    "linguacenter": transcribe_linguacenter,
    "intron": transcribe_intron,
    "elevenlabs": transcribe_elevenlabs,
    "publicaai": transcribe_publicaai,
}

# Intron is the only provider here that accepts a code-switched language code,
# and those codes are the point: `yo` means Yoruba mixed with English, not
# Yoruba alone. Sending everything as `en` throws away the one capability the
# challenge is judged on, so each sample is routed to its own language.
SAMPLE_LANGUAGE = {
    "codeswitch_pidgin": "pcm",
    "codeswitch_yoruba": "yo",
    "codeswitch_hausa": "ha",
    "codeswitch_igbo": "ig",
}


def language_for(sample_name: str) -> str:
    # A .lang file beside the clip wins — that is how AfriSwitch samples carry
    # their language through, rather than being guessed from the filename.
    lang_file = SAMPLES_DIR / f"{sample_name}.lang"
    if lang_file.exists():
        raw = lang_file.read_text(encoding="utf-8").strip().lower()
        return LANGUAGE_ALIASES.get(raw, raw[:3])
    return SAMPLE_LANGUAGE.get(sample_name, settings.stt.language)


# AfriSwitch spells languages out; Intron wants the ISO code.
LANGUAGE_ALIASES = {
    "yoruba": "yo",
    "hausa": "ha",
    "igbo": "ig",
    "pidgin": "pcm",
    "nigerian pidgin": "pcm",
    "english": "en",
    "swahili": "sw",
    "zulu": "zu",
}


# ----------------------------------------------------------------------
# Sample generation
# ----------------------------------------------------------------------


def make_samples() -> None:
    """Synthesise the phrase set so the harness runs with no recordings yet."""
    SAMPLES_DIR.mkdir(exist_ok=True)
    url = settings.tts.base_url.rstrip("/") + "/audio/speech"
    print(f"\nSynthesising {len(PHRASE_SET)} clips into {SAMPLES_DIR}/\n")

    for name, phrase in PHRASE_SET:
        wav_path = SAMPLES_DIR / f"{name}.wav"
        txt_path = SAMPLES_DIR / f"{name}.txt"
        if wav_path.exists():
            print(f"  {name}: exists, skipping")
            continue
        payload = json.dumps(
            {
                "model": settings.tts.model,
                "voice": settings.tts.voice,
                "input": phrase,
                "response_format": "wav",
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {settings.tts.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                wav_path.write_bytes(resp.read())
            txt_path.write_text(phrase, encoding="utf-8")
            print(f"  {name}: ok")
        except Exception as exc:
            print(f"  {name}: FAILED {exc}")

    print(
        "\n  These are synthetic. For the submission, replace them with real\n"
        "  recordings of real people code-switching on real handsets — that is\n"
        "  what the code-switching score is judged on.\n"
    )


AFRISWITCH_NIGERIAN = ("pidgin", "yoruba", "hausa", "igbo")


def fetch_afriswitch(
    limit: int, languages: tuple[str, ...] | None = None, split: str = "test"
) -> None:
    """
    Pull real code-switched clips from Intron's AfriSwitch.

    This is the dataset the challenge is built around — 16.6k instances of
    natural code-switching across 14 languages. Like AfriSwitchCare it is split
    by language config, and only four of those are Nigerian, so the rest are
    skipped by default: Amharic and Zulu are real code-switching but not this
    market, and they would dilute the per-language table with rows nobody is
    judging.

    Audio is loaded undecoded because the datasets version here demands
    torchcodec otherwise — see _write_audio_cell.
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        print(f"\n  Missing dependency: {exc}. pip install datasets soundfile\n")
        sys.exit(1)

    SAMPLES_DIR.mkdir(exist_ok=True)
    langs = languages or AFRISWITCH_NIGERIAN
    token = settings.stt.hf_token or None
    per_language = max(1, limit // len(langs))
    written = 0

    for language in langs:
        print(f"\n  Loading AfriSwitch [{language}] ...")
        try:
            data = load_dataset(
                "intronhealth/AfriSwitch",
                language,
                split=split,
                streaming=True,
                token=token,
            )
            data = data.cast_column("audio", Audio(decode=False))
        except Exception as exc:
            print(f"    could not load {language}: {str(exc).splitlines()[0][:100]}")
            continue

        taken = 0
        for row in data:
            if taken >= per_language:
                break
            audio = row.get("audio")
            text = next(
                (row[k] for k in ("transcript", "text", "sentence", "transcription") if row.get(k)),
                None,
            )
            if not audio or not text:
                continue

            name = f"afriswitch_{language}_{taken:03d}"
            if not _write_audio_cell(audio, SAMPLES_DIR / f"{name}.wav"):
                print(f"    {name}: audio unreadable, skipped")
                continue
            (SAMPLES_DIR / f"{name}.txt").write_text(str(text).strip(), encoding="utf-8")
            (SAMPLES_DIR / f"{name}.lang").write_text(language, encoding="utf-8")
            taken += 1
            written += 1
            print(f"    {name}: {str(text)[:56]!r}")

    print(f"\n  Wrote {written} real code-switched samples to {SAMPLES_DIR}/\n")


def _write_audio_cell(cell: dict, destination: Path) -> bool:
    """
    Write one dataset audio cell to a wav file.

    The datasets library version here refuses to decode audio without
    torchcodec, which drags in the whole of PyTorch for what is ultimately a
    format conversion. Loading the column undecoded gives the original bytes,
    and soundfile — already a dependency — reads wav, flac and ogg directly.
    Returns False rather than raising so one unreadable clip does not abandon
    the rest of the download.
    """
    import io

    import soundfile as sf

    raw = cell.get("bytes") if isinstance(cell, dict) else None
    if not raw:
        return False
    try:
        samples, rate = sf.read(io.BytesIO(raw))
        sf.write(destination, samples, rate)
        return True
    except Exception:
        return False


def fetch_afriswitchcare(limit: int, languages: tuple[str, ...] | None = None) -> None:
    """
    Pull real code-switched clips from Intron's AfriSwitchCare.

    Worth knowing why this exists alongside --fetch-afriswitch: AfriSwitch is
    gated "manual", so it waits on a human at Intron, while AfriSwitchCare is
    gated "auto" and is granted the moment you accept the terms. When the big
    set is still pending, this one gets real speakers into the benchmark today.

    It is medical dialogue rather than banking, so the vocabulary is off-domain
    for this agent — but code-switching behaviour is a property of the speaker,
    not the subject, so it measures the thing the challenge scores.

    Each clip is saved with a .lang file so Intron receives the right
    code-switched language code instead of everything being sent as English.
    """
    try:
        import soundfile as sf
        from datasets import Audio, load_dataset
    except ImportError as exc:
        print(f"\n  Missing dependency: {exc}. pip install datasets soundfile\n")
        sys.exit(1)

    SAMPLES_DIR.mkdir(exist_ok=True)
    langs = languages or AFRISWITCHCARE_NIGERIAN
    token = settings.stt.hf_token or None
    per_language = max(1, limit // len(langs))
    written = 0

    for language in langs:
        print(f"\n  Loading AfriSwitchCare [{language}] ...")
        try:
            data = load_dataset(
                "intronhealth/AfriSwitchCare",
                language,
                split="test",
                streaming=True,
                token=token,
            )
            data = data.cast_column("audio", Audio(decode=False))
        except Exception as exc:
            print(f"    could not load {language}: {str(exc).splitlines()[0][:100]}")
            continue

        taken = 0
        for row in data:
            if taken >= per_language:
                break
            audio = row.get("audio")
            text = next(
                (row[k] for k in ("transcript", "text", "sentence", "transcription") if row.get(k)),
                None,
            )
            if not audio or not text:
                continue

            name = f"afriswitchcare_{language}_{taken:03d}"
            if not _write_audio_cell(audio, SAMPLES_DIR / f"{name}.wav"):
                print(f"    {name}: audio unreadable, skipped")
                continue
            (SAMPLES_DIR / f"{name}.txt").write_text(str(text).strip(), encoding="utf-8")
            (SAMPLES_DIR / f"{name}.lang").write_text(language, encoding="utf-8")
            taken += 1
            written += 1
            print(f"    {name}: {str(text)[:56]!r}")

    print(f"\n  Wrote {written} real code-switched samples to {SAMPLES_DIR}/\n")


def load_samples() -> list[tuple[str, bytes, str]]:
    if not SAMPLES_DIR.exists():
        return []
    samples = []
    for wav_path in sorted(SAMPLES_DIR.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  skipping {wav_path.name}: no matching .txt ground truth")
            continue
        samples.append(
            (wav_path.stem, wav_path.read_bytes(), txt_path.read_text(encoding="utf-8").strip())
        )
    return samples


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def run(
    providers: list[str],
    report_path: str | None,
    code_switched_only: bool = False,
    limit: int | None = None,
    per_language: int | None = None,
) -> None:
    samples = load_samples()
    if code_switched_only:
        # The requirement is worded 'on code-switched audio input', so the
        # English-only clips are not what is being scored and only drag the
        # averages toward a question nobody asked.
        before = len(samples)
        samples = [row for row in samples if language_for(row[0]) != "en"]
        print(f"\n  Code-switched only: {len(samples)} of {before} samples")
    if per_language:
        # An equal number from each language. Unlike a flat limit this keeps
        # the per-language rows comparable: every language is scored on the
        # same amount of audio, so one language having more clips in the
        # dataset cannot quietly dominate the aggregate.
        grouped: dict[str, list] = {}
        for row in samples:
            grouped.setdefault(language_for(row[0]), []).append(row)
        samples = [
            row
            for language in sorted(grouped)
            for row in grouped[language][:per_language]
        ]
        counts = {k: len(v[:per_language]) for k, v in sorted(grouped.items())}
        summary = ", ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"\n  {len(samples)} samples — {summary}")
    elif limit:
        # Take the limit evenly across languages rather than the first N by
        # filename. Sorted order puts all the Hausa clips first, so a plain
        # head() produced ten Hausa samples and no Yoruba, Igbo or Pidgin at
        # all — which cannot answer the per-language question the challenge
        # actually asks.
        grouped: dict[str, list] = {}
        for row in samples:
            grouped.setdefault(language_for(row[0]), []).append(row)
        spread = []
        index = 0
        while len(spread) < limit and any(
            index < len(v) for v in grouped.values()
        ):
            for language in sorted(grouped):
                if index < len(grouped[language]) and len(spread) < limit:
                    spread.append(grouped[language][index])
            index += 1
        samples = spread
        counts = {}
        for row in samples:
            lang = language_for(row[0])
            counts[lang] = counts.get(lang, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"\n  Limited to {len(samples)} sample(s) — {summary}")
    if not samples:
        print("\n  No samples found. Run: python benchmark.py --make-samples\n")
        sys.exit(1)

    print(f"\n  {len(samples)} samples x {len(providers)} providers\n")
    results: list[Result] = []

    for provider in providers:
        transcriber = TRANSCRIBERS[provider]
        print(f"  --- {provider}")
        for name, audio, reference in samples:
            language = language_for(name)
            started = time.monotonic()
            try:
                if provider in ("intron", "elevenlabs", "publicaai"):
                    hypothesis = transcriber(audio, language)
                else:
                    hypothesis = transcriber(audio)
                error = None
            except Exception as exc:
                hypothesis, error = "", f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            result = Result(provider, name, reference, hypothesis, elapsed, error)
            results.append(result)
            if error:
                print(f"      {name:<20} FAILED  {error[:70]}")
            else:
                tag = f"[{language}]" if language != "en" else "    "
                print(
                    f"      {name:<20}{tag} wer={result.wer:5.2f}  {elapsed:4.1f}s  "
                    f"{hypothesis[:48]!r}"
                )
        print()

    summarise(results, providers)
    if report_path:
        Path(report_path).write_text(
            json.dumps(
                {
                    "generatedAt": datetime.now(timezone.utc).isoformat(),
                    "providers": providers,
                    "sampleCount": len({r.sample for r in results}),
                    "results": [
                    {
                        "provider": r.provider,
                        "sample": r.sample,
                        # Every result names the exact wav it came from, so a
                        # number in the table can always be traced back to the
                        # audio and listened to. Without this the report is a
                        # wall of figures nobody can audit.
                        "audioFile": f"samples/{r.sample}.wav",
                        "transcriptFile": f"samples/{r.sample}.txt",
                        "language": language_for(r.sample),
                        "dataset": (
                            "AfriSwitch"
                            if r.sample.startswith("afriswitch_")
                            else "AfriSwitchCare"
                            if r.sample.startswith("afriswitchcare_")
                            else "synthetic"
                        ),
                        "reference": r.reference,
                        "hypothesis": r.hypothesis,
                        "wer": round(r.wer, 4),
                        "cer": round(r.cer, 4),
                        "werRaw": round(r.wer_raw, 4),
                        "cerRaw": round(r.cer_raw, 4),
                        "criticalRecall": r.recall,
                        "seconds": round(r.seconds, 2),
                        "error": r.error,
                    }
                    for r in results
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  Written to {report_path}\n")


def summarise(results: list[Result], providers: list[str]) -> None:
    print("  " + "=" * 78)
    print(
        f"  {'provider':<14}{'WER':>7}{'CER':>7}{'WER raw':>9}{'CER raw':>9}"
        f"{'recall':>9}{'latency':>10}{'fails':>7}"
    )
    print("  " + "=" * 78)

    for provider in providers:
        rows = [r for r in results if r.provider == provider]
        ok = [r for r in rows if not r.error]
        if not ok:
            print(f"  {provider:<14}{'—':>7}{'—':>7}{'—':>9}{'—':>9}{'—':>9}{'—':>10}{len(rows):>7}")
            continue
        recalls = [r.recall for r in ok if r.recall is not None]
        print(
            f"  {provider:<14}"
            f"{statistics.mean(r.wer for r in ok):>7.2f}"
            f"{statistics.mean(r.cer for r in ok):>7.2f}"
            f"{statistics.mean(r.wer_raw for r in ok):>9.2f}"
            f"{statistics.mean(r.cer_raw for r in ok):>9.2f}"
            f"{(statistics.mean(recalls) if recalls else 0):>8.0%}"
            f"{statistics.mean(r.seconds for r in ok):>9.1f}s"
            f"{len(rows) - len(ok):>7}"
        )
    print("  " + "=" * 78)
    print(
        "\n  WER and CER, normalized and raw, are the metrics Intron's AfriHealth\n"
        "  MultiBench reports. CER is the one to watch on code-switched speech:\n"
        "  Yoruba and Hausa carry meaning in diacritics, so a single missing tone\n"
        "  mark scores a whole word wrong under WER while CER shows how close the\n"
        "  model actually got."
    )
    print(
        "\n  WER is the headline number, but critical recall is the one that\n"
        "  decides whether this agent works: it measures the words that trigger\n"
        "  a card freeze or pass verification. A model can look respectable on\n"
        "  WER while missing every one of them.\n"
    )

    # Per language per model, which is the shape the challenge asks results in.
    # An aggregate row hides the whole finding: Yoruba and Pidgin can differ by
    # a factor of two on the same model, and averaging them reports a number
    # that describes neither.
    by_language: dict[str, list[Result]] = {}
    for r in results:
        if r.error:
            continue
        language = language_for(r.sample)
        if language == "en":
            continue
        by_language.setdefault(language, []).append(r)

    if by_language:
        print("  Per language, code-switched audio only:")
        print()
        print(f"    {'language':<12}" + "".join(f'{p:>22}' for p in providers))
        print("    " + '-' * (12 + 22 * len(providers)))
        for language in sorted(by_language):
            row = f"    {language:<12}"
            for provider in providers:
                rows = [r for r in by_language[language] if r.provider == provider]
                if rows:
                    row += (
                        f"{statistics.mean(r.wer for r in rows):>11.2f}"
                        f"{statistics.mean(r.cer for r in rows):>11.2f}"
                    )
                else:
                    row += f"{'-':>11}{'-':>11}"
            print(row)
        print("    " + ' ' * 12 + "".join(f"{'WER':>11}{'CER':>11}" for _ in providers))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark speech models.")
    parser.add_argument("--make-samples", action="store_true", help="synthesise test clips")
    parser.add_argument(
        "--fetch-afriswitchcare",
        type=int,
        metavar="N",
        help="download N real clips from intronhealth/AfriSwitchCare (auto-gated)",
    )
    parser.add_argument(
        "--languages",
        help="comma separated configs for --fetch-afriswitchcare, e.g. yoruba,pidgin",
    )
    parser.add_argument(
        "--fetch-afriswitch",
        type=int,
        metavar="N",
        help="download N real code-switched clips from intronhealth/AfriSwitch",
    )
    parser.add_argument(
        "--providers",
        # linguacenter is deliberately not in the default set. It is an
        # English-only model, so scoring it on Hausa or Yoruba measures nothing
        # about the model and only produces a meaningless number — its WER of
        # 4.10 on Hausa is it transcribing speech it was never built for. Still
        # selectable by name for an English-only comparison.
        # intron is opt-in for the same reason as in tts_benchmark.py: it
        # rate-limits and warms a model per language, so it belongs in its
        # own run rather than holding up the fast providers.
        default="deepgram,elevenlabs,publicaai",
        help="comma separated: deepgram,intron,linguacenter,elevenlabs",
    )
    parser.add_argument("--report", help="write full results to this JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        help="only run the first N samples, to verify a provider cheaply",
    )
    parser.add_argument(
        "--per-language",
        type=int,
        metavar="N",
        help="run N samples of each language — the fair way to fill the "
        "per-language table",
    )
    parser.add_argument(
        "--code-switched-only",
        action="store_true",
        help="skip English-only clips; the challenge scores code-switched audio",
    )
    args = parser.parse_args()

    if args.make_samples:
        make_samples()
        return

    if args.fetch_afriswitch:
        langs = (
            tuple(l.strip() for l in args.languages.split(",") if l.strip())
            if args.languages
            else None
        )
        fetch_afriswitch(args.fetch_afriswitch, langs)
        return

    if args.fetch_afriswitchcare:
        langs = (
            tuple(l.strip() for l in args.languages.split(",") if l.strip())
            if args.languages
            else None
        )
        fetch_afriswitchcare(args.fetch_afriswitchcare, langs)
        return

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in providers if p not in TRANSCRIBERS]
    if unknown:
        print(f"\n  Unknown provider(s): {', '.join(unknown)}\n")
        sys.exit(1)
    run(
        providers,
        args.report,
        args.code_switched_only,
        args.limit,
        args.per_language,
    )


if __name__ == "__main__":
    main()
