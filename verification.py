"""
The bank's approved challenge flow.

Verification here is deliberately narrow. It confirms that the person on the call
is the customer by asking them to repeat back a non-secret fact they themselves
chose at onboarding — a favourite food, a favourite colour, a first school. It
never asks for anything that would let the asker take over the account if the
call turned out to be the fraud: no PIN, no password, no OTP, no CVV, no BVN.

That asymmetry is the whole point. If a fraudster spoofs this call, the worst a
customer can lose by answering honestly is the name of their first pet.

The question is chosen at random from the customer's own set, so someone who
overheard one call cannot predict the next challenge.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

logger = logging.getLogger("fraud_agent.verification")


# Things the agent must never solicit. Kept here as well as in the prompt so the
# check survives a prompt edit.
FORBIDDEN_CREDENTIALS = (
    "pin",
    "password",
    "otp",
    "one time password",
    "one-time password",
    "cvv",
    "card number",
    "bvn",
    "maiden name",
)


def _normalise(text: str) -> str:
    """Lowercase, strip possessives and punctuation, collapse spaces, drop articles."""
    text = text.lower().strip()
    # Drop possessives before punctuation goes, or "Mary's" leaves a stray "s"
    # token behind that stops an otherwise good match.
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(the|a|an|my|its|it is)\s+", "", text)
    return text


# How close a heard word must be to a stored one to count as the same word.
# "saint" against "sent" scores 0.67, and that is the gap a phone line plus a
# code-switched model actually produces. Set it tighter and correct customers
# get refused; the failure mode of accepting a near-miss is a security question
# that was already only one factor, and the caller still cannot do anything
# irreversible.
TOKEN_SIMILARITY = 0.65


def _similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def answers_match(given: str, expected: str) -> bool:
    """
    Compare a spoken answer to the stored one, allowing for what STT does to it.

    Speech to text does not return the stored string. On a real call "green
    flower school" came back as "Sent mi Sent Mary", "Sent Maili" and "SANT." -
    the customer said the right thing three times and was refused three times,
    then transferred. That is the agent failing, not the customer.

    So this matches word by word and tolerantly: every meaningful word in the
    stored answer must have a close-enough word in what was heard. Extra words
    are ignored, since people answer in sentences and STT inserts noise.

    Requiring ALL stored words matters. "green" alone must not satisfy "green
    flower school", or a caller who overheard one answer could use it on a
    different question.
    """
    a, b = _normalise(given), _normalise(expected)
    if not a or not b:
        return False
    if a == b:
        return True

    heard = a.split()
    wanted = [w for w in b.split() if len(w) > 2] or b.split()

    for word in wanted:
        best = max((_similar(word, h) for h in heard), default=0.0)
        if best < TOKEN_SIMILARITY:
            return False
    return True


@dataclass
class ChallengeFlow:
    """Tracks one call's worth of verification state."""

    customer: dict
    max_attempts: int = 3

    attempts: int = 0
    verified: bool = False
    _asked: list[dict] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts and not self.verified

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts)

    def next_question(self) -> str:
        """Pick a question the customer set, preferring one not yet asked."""
        questions = self.customer.get("securityQuestions") or []
        if not questions:
            raise RuntimeError(
                f"Customer {self.customer.get('customerId')} has no security questions "
                "on file; this call cannot be verified and must go to a human."
            )
        unasked = [q for q in questions if q not in self._asked]
        chosen = random.choice(unasked or questions)
        self._asked.append(chosen)
        logger.info(
            "Challenge issued to %s: %s", self.customer["customerId"], chosen["question"]
        )
        return chosen["question"]

    @property
    def question_texts(self) -> list[str]:
        """Every question this customer set, in their own wording."""
        return [q["question"] for q in self.customer.get("securityQuestions") or []]

    @property
    def pending_unanswered(self) -> bool:
        """A question is outstanding that the customer has not yet answered."""
        return len(self._asked) > self.attempts

    @property
    def pending_question(self) -> str | None:
        """The question the customer is currently being asked, if any."""
        return self._asked[-1]["question"] if self._asked else None

    def record(self, matched: bool, answer: str = "") -> bool:
        """
        Record the agent's verdict on an answer and count the attempt.

        The agent decides whether the answer was right, because it is reading a
        speech transcript and only a reasoning model can tell "Sent mi Sent
        Mary" (Saint Mary, badly heard) from "blue" (simply wrong). String
        matching cannot: every tolerance added to it either refuses correct
        customers or waves through wrong ones, and on a live call it refused the
        right answer three times and transferred a verified customer.

        What stays here is the counting. The attempt limit is a control, not a
        judgement, so the agent does not get to decide how many tries it grants.
        """
        if self.verified:
            return True

        self.attempts += 1
        if matched:
            self.verified = True
            logger.info(
                "Verification PASSED for %s on attempt %d (agent judged %r a match)",
                self.customer["customerId"],
                self.attempts,
                answer[:40],
            )
            return True

        logger.warning(
            "Verification FAILED for %s on attempt %d of %d (agent judged %r wrong)",
            self.customer["customerId"],
            self.attempts,
            self.max_attempts,
            answer[:40],
        )
        return False

    def check(self, answer: str) -> bool:
        """
        Check an answer against whichever question was last asked.

        Returns True on success. Every call counts as an attempt, so a caller
        who keeps guessing runs out and is transferred.

        The agent reads the questions straight off the customer record now, so
        it often asks one without calling ask_security_question first and we
        have no record of which. Raising there would fail verification for a
        customer who answered correctly. Instead the answer is matched against
        any of theirs - still their own private facts, still capped by the
        attempt limit.
        """
        if self.verified:
            return True

        if not self._asked:
            self.attempts += 1
            for question in self.customer.get("securityQuestions") or []:
                if answers_match(answer, question["answer"]):
                    self._asked.append(question)
                    self.verified = True
                    logger.info(
                        "Verification PASSED for %s on attempt %d (question not "
                        "issued through the tool; matched %r)",
                        self.customer["customerId"],
                        self.attempts,
                        question["question"],
                    )
                    return True
            logger.warning(
                "Verification FAILED for %s on attempt %d of %d (no issued question)",
                self.customer["customerId"],
                self.attempts,
                self.max_attempts,
            )
            return False

        self.attempts += 1
        expected = self._asked[-1]["answer"]
        if answers_match(answer, expected):
            self.verified = True
            logger.info(
                "Verification PASSED for %s on attempt %d",
                self.customer["customerId"],
                self.attempts,
            )
            return True

        logger.warning(
            "Verification FAILED for %s on attempt %d of %d",
            self.customer["customerId"],
            self.attempts,
            self.max_attempts,
        )
        return False


# Phrases that mean the agent is REFUSING to ask, warning the customer, or
# reassuring them. The safety line the agent must say - "I will never ask you
# for your P I N, your password, or a one time code" - names three forbidden
# credentials in one breath, and a guard that matches on the word alone gags the
# single most important sentence on the call and replaces it with an apology for
# something the agent never did.
SAFE_CONTEXTS = (
    "never ask",
    "never asks",
    "will not ask",
    "won't ask",
    "would never ask",
    "do not ask",
    "does not ask",
    "not ask you",
    "never share",
    "never give",
    "do not share",
    "do not give",
    "should not give",
    "should never",
    "no one should",
    "nobody should",
    "anyone who asks",
    "anyone who calls",
    "i have not asked",
    "i did not ask",
    "without asking",
)

# Phrasings that actually solicit something. The agent may name a credential all
# day; it may not request one.
REQUEST_CUES = (
    "what is your",
    "what's your",
    "whats your",
    "tell me your",
    "tell me the",
    "give me your",
    "give me the",
    "provide your",
    "provide the",
    "confirm your",
    "confirm the",
    "enter your",
    "enter the",
    "type your",
    "read out your",
    "read out the",
    "read me your",
    "read me the",
    "read back the",
    "send me the",
    "send me your",
    "say your",
    "share your",
    "may i have your",
    "can i have your",
    "could i have your",
    "can you give me",
    "can you provide",
    "i need your",
    "i will need your",
    "please provide",
    "please confirm",
    "please enter",
    "what are the",
    "what is the",
)


_SPACED_LETTERS = re.compile(r"\b(?:[a-z]\s+){1,}[a-z]\b")


def _despace_initialisms(lowered: str) -> str:
    """
    Collapse "p i n" to "pin" before matching.

    The voice-formatting rule tells the agent to space initialisms out so the TTS
    reads them as letters, so the guard never sees the word it is looking for.
    That is not cosmetic: "Tell me your B V N" sailed straight through a matcher
    that only knew "bvn".
    """
    return _SPACED_LETTERS.sub(lambda m: m.group().replace(" ", ""), lowered)


def mentions_forbidden_credential(text: str) -> str | None:
    """
    Return the forbidden credential this text ASKS FOR, if any.

    A backstop on the agent's own output. The prompt already forbids asking, and
    the model mostly obeys - but on a live call it ran out of its listed security
    questions, invented one, and asked the customer for their mother's maiden
    name. A prompt is guidance; this is the thing that stops the sentence
    reaching the caller.

    It must fire on a request and stay silent on a mention. Blocking mentions
    censors the warnings that protect the customer, which is the opposite of the
    point.
    """
    lowered = _despace_initialisms(text.lower())

    # Longest first, so "one time password" is reported rather than "password".
    named = next(
        (t for t in sorted(FORBIDDEN_CREDENTIALS, key=len, reverse=True) if t in lowered),
        None,
    )
    if named is None:
        return None

    # Refusing, warning or reassuring. Let it through.
    if any(ctx in lowered for ctx in SAFE_CONTEXTS):
        return None

    if any(cue in lowered for cue in REQUEST_CUES):
        return named

    # No explicit cue. A question that names a credential is still a request
    # ("Your P I N?"), but a statement that merely mentions one is not.
    if "?" in lowered:
        return named

    return None
