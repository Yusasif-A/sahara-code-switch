"""
Pick the right voice for the language the agent is about to speak.

The TTS endpoints are per-language — /en/, /yo/, /ha/, /ig/ — so a single static
voice cannot answer a code-switched caller. Sending Yoruba text to the English
model does not produce accented Yoruba; it produces English letter-sounds read
off a Yoruba string, which is unintelligible. That is a silent failure: the call
sounds like it is working and the customer understands nothing.

Routing happens per sentence, inside each agent's existing tts_node, because
that is already where output is buffered into whole sentences for the credential
guard. One sentence is also the right unit linguistically: Nigerian callers
switch between sentences far more often than they switch mid-clause, and a
mid-clause switch is better served by one voice reading both halves than by two
voices stitched together.

MIXED SENTENCES GO TO THE NON-ENGLISH VOICE. "Emi ko ra nkan yen, it was not me"
routes to Yoruba. The Yoruba model reading English words produces Nigerian-
accented English, which is correct and normal here. The English model reading
Yoruba words produces noise.
"""

from __future__ import annotations

import asyncio

import logging
import re

logger = logging.getLogger("tts-router")

# Characters that appear in one language's orthography and nowhere else in the
# set. A single one of these is near-conclusive, so they outweigh word hits.
DIACRITICS = {
    "yo": "ẹọṣẸỌṢ",
    "ha": "ƙɓɗƴƘƁƊƳ",
    "ig": "ịụṅỊỤṄ",
}

# Words chosen for being distinctive, not for being common. "ni", "ka", "mo" and
# "na" are frequent in several of these languages *and* in English text, so they
# are deliberately absent: a false Yoruba route on an English sentence is worse
# than a missed Yoruba route, because English text read by the English model is
# always at least intelligible.
MARKERS = {
    "yo": {
        "emi", "iwo", "awon", "eyin", "temi", "tire", "nkan", "yen", "jowo",
        "jọwọ", "bawo", "sugbon", "ṣugbọn", "nitori", "kaadi", "ifowopamo",
        # "abi" is deliberately absent: it is as common in Pidgin as in
        # Yoruba, and letting it vote sent "You wan buy data abi?" to the
        # Yoruba voice mid-conversation.
        "pele", "pẹlẹ", "kilode", "kini", "beeni", "rara", "owo",
        "gbogbo", "ojo", "ọjọ", "mofe", "ranmilowo", "seun", "ṣeun",
    },
    "ha": {
        # "kudi" (money) and "banki" are deliberately absent: kudi is also the
        # telco agent's own name, and banki is shared with Yoruba. Ambiguous
        # markers cause false routes, which are worse than missed ones.
        "sannu", "kwana", "lafiya", "wannan", "yanzu", "gaskiya",
        "kudina", "yaya", "nawa", "zan", "kar", "toh", "aikin",
        "wayar", "asusun", "katin", "nagode", "madalla", "haba", "kada",
    },
    "ig": {
        "biko", "kedu", "daalu", "dalu", "nsogbu", "ego", "gini", "achoro",
        "achọrọ", "chere", "nwere", "adighi", "adịghị", "akaunti", "ndo",
        "maka", "ekwe", "onye", "ihe", "eme", "kwuru",
    },
    # Pidgin is English-lexified, so it is routed to the English voice — but it
    # is detected separately so the logs tell the truth about what was heard.
    "pcm": {
        "abeg", "wahala", "wetin", "dey", "sabi", "oga", "sef", "comot",
        "shey", "yawa", "japa", "gbese", "abi", "una", "dem", "waka",
    },
}

# Which endpoint actually speaks each detected language.
VOICE_FOR = {"en": "en", "pcm": "en", "yo": "yo", "ha": "ha", "ig": "ig"}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _brand_words() -> set[str]:
    """
    Words from our own product names, which must never vote on language.

    The telco agent was once called Kudi, and "kudi" is Hausa for money - so its
    own opening line, "This is Kudi, an A I assistant", scored as Hausa and was
    read aloud by the Hausa voice. Both agents are Noba now, but any brand name
    can collide like this, so strip them all rather than dropping a marker a
    real caller would genuinely say.
    """
    from config import settings

    names = (
        settings.telco.agent_display_name,
        settings.telco.name,
        settings.bank.agent_display_name,
        settings.bank.name,
    )
    return {w for name in names for w in _WORD.findall((name or "").lower())}


_BRANDS: set[str] | None = None


def detect_language(text: str) -> str:
    """Return the language code a sentence should be spoken in."""
    if not text or not text.strip():
        return "en"

    global _BRANDS
    if _BRANDS is None:
        try:
            _BRANDS = _brand_words()
        except Exception:  # config not loadable (bare unit test) - no brands
            _BRANDS = set()

    lowered = text.lower()
    words = set(_WORD.findall(lowered)) - _BRANDS

    scores: dict[str, int] = {}
    for lang, chars in DIACRITICS.items():
        hits = sum(1 for ch in text if ch in chars)
        if hits:
            scores[lang] = scores.get(lang, 0) + 3 * hits
    for lang, markers in MARKERS.items():
        hits = len(words & markers)
        if hits:
            scores[lang] = scores.get(lang, 0) + hits

    if not scores:
        return "en"

    best = max(scores, key=lambda k: (scores[k], k == "pcm"))

    # Changing voice costs more than getting the language slightly wrong.
    #
    # English and Pidgin share one voice, so moving between them is inaudible.
    # Yoruba, Hausa and Igbo are different speakers, and one ambiguous word
    # flipping a sentence to another speaker is heard as the agent being
    # replaced mid-conversation. So when a second language also scored, those
    # three need real evidence - a diacritic, or two distinctive words. A lone
    # unambiguous marker is still trusted: "Ba ni yi wannan ba" is Hausa and
    # nothing competes with it.
    if best in ("yo", "ha", "ig") and scores[best] < 2 and len(scores) > 1:
        return "pcm" if "pcm" in scores else "en"
    return best


class LanguageRoutedTTS:
    """
    Lazily-built pool of one TTS client per language.

    Clients are cached because each one holds an HTTP session; building a fresh
    one per sentence leaks connections and adds a handshake to every reply on a
    call where latency is the whole point.
    """

    def __init__(self, *, default_lang: str = "en") -> None:
        self._pool: dict[str, object] = {}
        self._default = default_lang

    def _build(self, lang: str):
        from livekit.plugins import openai

        from config import settings

        if settings.tts.is_intron:
            # Sahara reads code-switched text inside one voice, so there is no
            # seam to route around: the Pidgin voice speaks the English in a
            # Pidgin sentence natively. Routing still applies for a caller who
            # switches wholly into Yoruba or Hausa.
            from intron_tts import IntronTTS

            return IntronTTS(
                api_key=settings.stt.intron_api_key,
                language="pcm" if lang in ("en", "pcm") else lang,
            )

        endpoint = settings.tts.endpoint_for(lang)
        if endpoint is None:
            return None
        base_url, model, voice = endpoint
        logger.info("TTS voice for %s: %s (%s/%s)", lang, base_url, model, voice)
        return openai.TTS(
            base_url=base_url,
            model=model,
            voice=voice,
            api_key=settings.tts.api_key,
            response_format="wav",
        )

    def get(self, lang: str):
        """Return the TTS for `lang`, falling back to the default voice."""
        from config import settings

        target = VOICE_FOR.get(lang, "en")
        # Languages that are not switched on keep the default voice. Detecting
        # Yoruba is still useful - it goes in the log - but hearing a different
        # person say one sentence is worse than hearing the usual voice say it.
        if lang not in settings.tts.voice_languages:
            target = self._default
        if target not in self._pool:
            built = self._build(target)
            if built is None:
                # No endpoint configured for that language. Say it with the
                # default voice rather than dropping the sentence - a badly
                # pronounced warning still beats silence on a fraud call.
                logger.warning("No TTS endpoint for %s; using %s", target, self._default)
                target = self._default
                if target not in self._pool:
                    self._pool[target] = self._build(target)
            else:
                self._pool[target] = built
        return self._pool[target]

    def for_text(self, text: str):
        """Detect the language of `text` and return (lang, tts)."""
        lang = detect_language(text)
        return lang, self.get(lang)

    async def warm(self, lang: str = "en") -> None:
        """
        Synthesise a throwaway line so the first real sentence does not pay for
        a cold start. There is dead time while the room fills; spend it.
        """
        engine = self.get(lang)
        try:
            async for _ in engine.synthesize("Hello there."):
                pass
            logger.info("TTS warmed for %s", lang)
        except Exception as exc:  # noqa: BLE001 - a cold voice must not kill the call
            logger.warning("TTS warm-up failed for %s: %s", lang, exc)


async def _collect(engine, text: str) -> list:
    """Synthesise one sentence fully, so it can be generated ahead of time."""
    return [event.frame async for event in engine.synthesize(text)]


# A sentence shorter than this is held back and spoken together with the next
# one instead of being sent to the voice on its own.
#
# Sahara's text normaliser pronounces punctuation on very short input, and it
# does so intermittently - measured over five runs each, "Done." spoke "full
# stop" twice, "Done," spoke "comma" once and "full stop" once, and even bare
# "Done" with no punctuation at all spoke "full stop" once. Trimming the mark
# therefore reduces it but cannot remove it: the trigger is the shortness, not
# the character. "Done. The card is frozen." was clean every time.
#
# Merging costs nothing on latency and usually saves it. Sahara takes about five
# seconds per call regardless of length, so two fragments spoken as one line is
# one call instead of two, and the language model is far enough ahead of the
# voice that the next sentence is almost always already waiting.
MIN_SYNTHESIS_CHARS = 25


async def speak_routed(router: LanguageRoutedTTS, sentences):
    """
    Synthesize an async stream of sentences, each with its own language's voice.

    Yields audio frames, so this is a drop-in for `super().tts_node(...)` inside
    an Agent that already buffers its output into sentences.

    One sentence ahead is generated while the current one plays. Sahara takes
    about 5s to make roughly 3s of speech, so synthesising strictly in turn left
    a two-second hole between every sentence: the telco greeting is three
    sentences, nine seconds of audio, and took fifteen to deliver. Overlapping
    the two hides most of that, because the gap only shows when generation is
    slower than playback and nothing else is in flight.
    """
    inflight: tuple[str, asyncio.Task] | None = None
    held = ""

    async for sentence in sentences:
        text = (held + " " + sentence.strip()).strip() if held else sentence.strip()
        held = ""
        if not text:
            continue
        # Too short to send alone - keep it and lead the next sentence with it.
        if len(text) < MIN_SYNTHESIS_CHARS:
            held = text
            continue
        lang, engine = router.for_text(text)
        if lang != "en":
            logger.info("Speaking in %s: %s", lang, text[:80])

        # Start this one before playing the last, so the two overlap.
        task = asyncio.create_task(_collect(engine, text))
        if inflight is not None:
            for frame in await inflight[1]:
                yield frame
        inflight = (lang, task)

    # Nothing left to merge it into, so a short last fragment goes alone. The
    # trailing-punctuation trim in the voice client is what covers this case.
    if held:
        lang, engine = router.for_text(held)
        task = asyncio.create_task(_collect(engine, held))
        if inflight is not None:
            for frame in await inflight[1]:
                yield frame
        inflight = (lang, task)

    if inflight is not None:
        for frame in await inflight[1]:
            yield frame
