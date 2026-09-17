# Noba

A Nigerian voice agent that handles bank fraud response and telecom customer
care, built for the Sahara CodeSwitch Africa Challenge.

## The problem

A Nigerian caller under stress does not speak one language. They move between
English, Pidgin, Yoruba, Hausa and Igbo inside a single sentence, without
noticing they are doing it. Every automated phone line in the country handles
one language at a time, so the caller has to translate their own emergency
before anyone will help them.

Two places where that costs the most:

**Fraud.** Banks wait for fraud to complete before flagging it. By the time a
customer reaches anyone, the money has gone and the remaining options are a
written dispute and weeks of waiting.

**Telecom care.** The path to a person runs through a menu — press 1, press 2,
press 3 — designed around the company's departments rather than the caller's
problem.

## What Noba does

One number answers both lines. It asks which one you need and routes you.

It understands a sentence that mixes languages and answers in the way the caller
spoke, rather than correcting them into English. It verifies a caller without
ever asking for a PIN, a password or a one-time code, takes a protective action
the bank has pre-approved, and hands anything irreversible to a human.

It reaches people on a normal phone call or on WhatsApp, so it works whether
or not the caller has a smartphone and data.

## What it can act on

The agent does not only talk. On the bank line it can verify the caller against
the questions the bank already holds, tell them what was seen on their account,
freeze a card, reduce a daily limit, and flag a transaction for review. On the
telecom line it can diagnose a fault on the line, check balances and recent
transactions, restore a bundle that was paid for but never delivered, credit a
recharge that did not land, and send network settings.

Every one of those is reversible, and every one is recorded against the account
with a note saying the agent did it.

## Where it stops

Verification never involves a secret. The agent does not ask for a PIN, a
password, a one-time code or a card number, and it will not accept one if it is
offered — it says so in the first line of the call, before anything is asked.

Anything that cannot be undone goes to a person. That means reversing money,
closing or blocking an account, unfreezing what was frozen, and changing the
contact details on file. The agent has no way to do any of it: the capability
does not exist for it to reach for.

It also hands over when verification fails, when the caller asks for a human,
when someone is distressed, and whenever it is not certain. A case is opened
with what the agent already did and why it stepped back, so the person picking
it up starts with the context instead of asking the customer to repeat
everything.

## Benchmarks

Speech recognition and text-to-speech were measured on code-switched Nigerian
audio — word and character error rates per language for ASR, and a
synthesise-then-transcribe round trip for TTS.

Results, scored clips and methodology:
**[huggingface.co/datasets/yusasif/intron-stt_tts-benchmark](https://huggingface.co/datasets/yusasif/intron-stt_tts-benchmark)**

The raw scores are also in this repository as `benchmark_results.json` (ASR) and
`tts_results.json` (TTS).

## Demo

A recorded walkthrough of both lines is linked from the challenge submission.
