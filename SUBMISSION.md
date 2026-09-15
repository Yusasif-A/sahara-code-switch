# Sahara CodeSwitch Africa — submission answers

Copy each block into the matching box. Word counts are in brackets.

---

## Solution Title

```
Noba — reaching the customer inside the window, in their own language
```

---

## Website

```
https://huggingface.co/datasets/yusasif/intron-stt_tts-benchmark
```

*(Use your GitHub repo here instead if the repo is public — a website field is
usually read as "where can I see this". The dataset link is the fallback.)*

---

## 1. Short description of the problem your app addresses [78 words]

```
A suspicious transaction goes through on a Nigerian bank card, because nobody
confirms it with the customer in time. Hours later the money is gone and the
account gets blocked. Nearly half of Nigeria's 25.85 billion naira fraud losses
in 2025 were social engineering - the customer was tricked, so there is nothing
to detect. On the telecom side, airtime vanishes and paid data never arrives,
and reporting it means press one, press two, and still no fix.
```

*`~50 words` has a tilde, so 78 should pass. Strict-50 fallback, 59 words:*

```
A suspicious transaction goes through on a Nigerian bank card, because nobody
confirms it with the customer in time. Hours later the money is gone and the
account gets blocked. Nearly half of 2025 fraud losses were social engineering,
so there is nothing to detect. On the telecom side, paid data never arrives and
the menu never fixes it.
```

---

## 2. Target users and potential number of users [55 words]

```
Anyone in Nigeria who banks or holds a line: just over 70 million BVN holders
and 195.11 million active mobile subscriptions. 67,515 fraud cases were reported
in 2025. Beyond fraud, every subscriber who has lost airtime to a deduction
nobody will explain, or bought data that never arrived, is a user of the same
agent.
```

---

## 3. How your app solves the user problem [80 words]

```
Noba is a customer agent for both a bank and a telecom company. On fraud it
calls the instant a signal fires and freezes the card inside that window -
prevention, not a post-mortem. It never asks for a PIN or OTP and says so. On
the care line there is no press one, press two: you say what is wrong and it
fixes it. It answers on a WhatsApp call or a normal phone call, in any Nigerian
language.
```




*Strict-50 fallback, 62 words:*

```
Noba is a customer agent for both a bank and a telecom company. On fraud it
calls the instant a signal fires and freezes the card inside that window,
before the money moves. On the care line there is no press one, press two: you
say what is wrong and it fixes it. WhatsApp call or normal call, in any
Nigerian language.
```

---

### Why the fraud framing matters — keep this in your head for the video

Social engineering was **47% of Nigerian fraud volume and ₦17.84bn of losses in
2025**. Detection models do not stop that, because nothing is technically wrong
with the transaction — the customer authorised it. Only reaching the person
stops it, and only in time.

Vishing is rising precisely because Nigerians are used to "your bank calling"
asking to confirm details. An official bank agent that **never** asks, and says
so in the first ten seconds, inoculates against the next call. That is the
second-order prevention argument, and it is the strongest thing you have.

**Language is reach, not the headline.** The problem is timing. Speaking Yoruba,
Hausa, Igbo and Pidgin is what makes the fix work for *every* customer rather
than the English-speaking minority - and it removes the seconds a caller would
otherwise spend translating their own emergency. Lead with the window, then say
the agent reaches all 195 million of them. Do not open on language; a judge has
heard that pitch all day.

**Two channels, so nobody is left out.** The agent works over a WhatsApp call
and over a normal phone call. A customer with data uses WhatsApp; one without
uses the ordinary line and gets the same agent. Together with the four
languages, that is what makes the coverage real rather than theoretical.

**Sources** (cite if a judge asks):
NIBSS 2026 e-Fraud Forum — ₦25.85bn lost in 2025, down 51% from ₦52.26bn;
67,515 cases; social engineering 47% of volume / ₦17.84bn; internet banking
₦13.37bn from 4,507 cases; Lagos 63.43%. NCC July 2026 — 195.11m active lines.
NIBSS — 70,036,488 BVNs at 31 Aug 2026.

---

## 4. Does your solution support code-switching?

```
Yes
```

---

## 5. Does your solution use the Sahara APIs?

```
Yes
```

---

## 6. How is the solution agentic? What downstream task does the transcript enable? [58 words]

```
The transcript triggers action, not a summary. It drives live calls into bank
and telco APIs: freeze the card before the next charge clears, lower a limit,
flag a disputed transaction, restore a paid bundle that never activated, credit
a stuck recharge. Anything irreversible, including refunds and SIM swap, is
refused in code and passed to a person.
```

---

## 7. Technical overview: key design decisions and tradeoffs [249 words]

```
Three decisions shaped this build.

First, the agent's authority is enforced in code, not in the prompt. Every
bank and telco operation sits in one of two frozen sets. Freezing a card,
lowering a limit, flagging a transaction and restoring a paid bundle are
pre-approved. Refunds, SIM swap, unfreezing, raising limits and closing
accounts raise PermissionDenied no matter what was said on the call. The
tradeoff is more transfers to humans and an agent that cannot always finish the
job. We took it because a prompt is guidance, and a fraud call is exactly the
situation where someone talks past guidance. On an early test the agent ran out
of its listed security questions, invented one, and asked the customer for
their mother's maiden name. The prompt forbade it. So there is now an output
guard that buffers speech into whole sentences and refuses to speak a
credential request, since the phrase spanned several chunks and per-chunk
checks missed it. It costs a little latency per sentence.

Second, Sahara STT streams. The obvious wrapper buffers an utterance, commits,
and waits, which made callers wait 8.7 seconds after they stopped talking.
Opening the socket at start-of-speech and feeding it live halves that to about
4.5s, and sessions are warmed during silence because the handshake alone was
eating short turns.

Third, output is routed per sentence to the voice for its language, using one
voice name across all four, so the speaker does not change when the caller
switches language mid-call.
```

---

## 8. Ethics / Inclusion [104 words]

```
The agent never asks for a PIN, password, OTP, CVV, card number, BVN or
mother's maiden name. That is enforced twice: a forbidden-terms list, and a
guard that blocks the sentence before it is spoken. It says it is an AI in its
opening line and again whenever asked, and it tells a doubtful customer to hang
up and ring the number on their card — the freeze holds either way.
Verification uses questions the customer chose themselves, so a spoofed call
leaks nothing worse than a pet's name. SIM swap is impossible for the agent by
design. Every action is logged, refusals included.
```

---

## Demo Video URL

```
[paste your unlisted YouTube link]
```

Must show code-switching on camera. Suggested run, under 5 minutes:

1. Open the telco agent. Say **"Mo ra data yen, but e never enter."**
   It understands both halves and answers in English.
2. Say **"Abeg, check my balance."** — Pidgin, no correction, no repeat request.
3. Switch to a full Yoruba turn so the voice switches with you.
4. On the airtime dispute scenario, say **"Just refund my 420 naira now now."**
   It refuses and transfers. Show that refusal — the guardrail is the story.
5. Cut to the fraud agent freezing a card, then being refused a reversal.

Warm each language once before recording. The first Intron request in a
language pays a cold-start of several seconds.

---

## Benchmark Report Link

```
[paste your Google Drive PDF link, sharing set to "anyone with the link"]
```

Source document: `Benchmark_Report.docx` in the repo root. Export to PDF and
upload. It runs to about two and a half pages, inside the three-page limit.

---

## Benchmark Audios Link

```
https://huggingface.co/datasets/yusasif/intron-stt_tts-benchmark
```

16 clips, 4 each in Hausa, Igbo, Pidgin and Yoruba, all from AfriSwitch, with
transcripts, language codes and a `used_in` column tying every clip to the
results file it appears in.
