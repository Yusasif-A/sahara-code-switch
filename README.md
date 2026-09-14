# Noba — Nigerian Voice Agents

An outbound voice agent that calls a bank customer the moment a fraud signal is
raised, verifies them through the bank's approved challenge flow, freezes the
card, and hands anything irreversible to a human.

Built on LiveKit Agents over WhatsApp. English only for now — the Hausa, Igbo and
Yoruba STT/TTS endpoints are already in `.env` for a later pass.

Both the fraud line and the telco care line introduce themselves as
`Noba`, matching the verified name on the WhatsApp number, so the sender the
customer sees and the voice they hear are the same brand. The name lives only
in `config.py` (`AGENT_DISPLAY_NAME`,
`TELCO_AGENT_NAME`, `BANK_NAME`, `TELCO_NAME`) — rebrand by editing `.env`.

## The idea

When a bank spots a stolen card or an account takeover, the window to act is
minutes. An IVR that makes the customer press 1, then 2, then hold, spends that
window. This agent calls them, speaks plainly, and does the one protective thing
it is allowed to do.

## Where the agent's authority stops

This is the part that matters. The agent may do only things the bank can undo:

| Pre-approved (agent) | Human fraud desk only |
| --- | --- |
| Freeze a card | Reverse or recall money |
| Lower a card limit | Block or close an account |
| Flag a disputed transaction | Unfreeze anything, raise any limit |
| Open a fraud case | Change phone, email or address |

That split is enforced in three independent places, so no single failure lets the
agent do something irreversible:

1. `prompts.py` — the instructions tell it what it may do
2. `agent.py` — tools refuse to act before verification passes
3. `bank_api.py` — `_guard()` refuses human-only operations regardless of what
   was said on the call

Test 3 by running `python demo.py` and watching the ⛔ lines.

## Verification, and what is never asked

The agent asks one non-secret question the customer chose themselves — favourite
food, favourite colour, first school — drawn at random from their own set, so
overhearing one call does not predict the next.

It never asks for a **PIN, password, OTP, CVV, full card number or BVN**. This is
not just courtesy: if a fraudster ever spoofs this call, the most an honest
customer loses by answering is the name of their first pet.

The prompt also tells the agent to *encourage* a suspicious customer to hang up
and call the number on the back of their card. A customer who does that has done
the right thing, and the freeze is already in place either way.

## Files

| File | Role |
| --- | --- |
| `config.py` | All `.env` loading and validation. Nothing else reads env vars. |
| `prompts.py` | System prompt, opening line, fixed wording for sensitive moments |
| `bank_data.py` | Dummy Nigerian customers, NUBAN accounts, cards, transactions, fraud signals |
| `bank_api.py` | Simulated bank, shaped after the Open Banking Nigeria standard |
| `verification.py` | The approved challenge flow |
| `agent.py` | The LiveKit agent and its tools — **run this as the worker** |
| `code_switching.py` | The language instructions both agents share |
| `tts_router.py` | Picks the voice for each sentence's language |
| `whatsapp_connector.py` | Meta ↔ LiveKit SDP relay |
| `caller.py` | Places the outbound call |
| `app.py` | FastAPI: Meta webhook + fraud trigger + demo views |
| `demo.py` | Full call flow, no credentials needed |
| `telco_agent.py` | The telco care agent — **run this as the second worker** |
| `telco_api.py` | Simulated network, with the same authority boundary |
| `telco_data.py` | Dummy subscribers, bundles, recharges and known issues |
| `telco_prompts.py` | System prompt and opening line for the care line |
| `stt_providers.py` | Chooses the speech model from `STT_PROVIDER` |
| `intron_stt.py` | Sahara speech-to-text, streaming, code-switched pairs |
| `elevenlabs_stt.py` | ElevenLabs Scribe speech-to-text |
| `intron_tts.py` | Sahara text-to-speech over the generate endpoint |
| `benchmark.py` | ASR benchmark: WER and CER per language, five models |
| `tts_benchmark.py` | TTS benchmark: hallucination, transcript loss, segment loss |
| `push_dataset.py` | Publishes the benchmarked clips to HuggingFace |
| `test_guardrails.py` | 162 tests over the authority boundary, verification, the credential guard and language routing |

The bank API mirrors the [Open Banking Nigeria](https://openbanking.readme.io/reference/overview)
resource names (`Get By Phone Number`, `GetStatement`, `Card > Block`,
`Set Card Limit`, `Block Account`), and the transaction records match its
`GetStatement` response schema. Swapping the simulator for a real bank is a
change of transport, not of shape.

## Running it

Everything runs in the **`publica` conda environment** (LiveKit is installed
there, not in the global Python):

```bash
conda activate publica
pip install -r requirements.txt   # only pytest was missing
```

## Testing, in three stages

Do these in order. Each one needs more setup than the last, so a failure tells
you exactly which layer broke.

### Stage 1 — logic only, no audio, no credentials

```bash
python demo.py                    # scripted call through the real bank logic
pytest test_guardrails.py -q      # 162 tests
```

### Stage 2 — talk to the agent through your laptop, no telephony

This is the fastest way to hear it working and tune the prompt. It uses your mic
and speakers; no WhatsApp, no phone, no webhook, no number to dial.

```bash
python agent.py console        # Noba, the fraud line
python telco_agent.py console  # Noba, the telco care line
```

Speak when it stops talking. `Ctrl+C` ends the call. Every turn is printed as
`[customer]` / `[agent]`, so you can read back what the STT actually heard —
which is usually where a confusing reply comes from.

With no dispatch metadata the fraud agent falls back to `DEMO_SIGNAL_ID`
(`FRD-CARD-FOREIGN`), so it opens as if it just caught the Ukraine card fraud.
Answer as Yusuf — his security answers are `amala`, `green`, `green flower school`.

#### Testing code-switching in the console

**The default config cannot pass this test, by design.** `STT_PROVIDER=deepgram`
with `STT_LANGUAGE=en` is the right choice for an English call and cannot hear
Yoruba at all, and the default PrepAI voice is English-only. Both have to move:

```dotenv
STT_PROVIDER = publicaai
STT_BASE_URL = https://stts-yoruba.publicaai.com/yo/v1
STT_LANGUAGE = auto
```

`stts-yoruba` is a naming accident — that endpoint is the multilingual one, and
it covers English, Yoruba, Hausa and Igbo. `STT_LANGUAGE=auto` omits the
language parameter entirely so the model decides; pinning it to `en` makes it
decode a Yoruba utterance *as English*, which returns confident nonsense rather
than an error.

Output routing needs nothing — `TTS_ROUTING` is on by default. Each sentence the
agent produces is language-detected and sent to that language's voice
(`multilingua-tts/yo/`, `/ha/`, `/ig/`), because those endpoints are per-language
and the English model reading Yoruba text produces noise rather than an accent.
The log line `Speaking in yo: ...` tells you a route fired.

**Things to say.** Each of these mixes two languages in one sentence, which is
the case that breaks single-language agents:

| Say this | What should happen |
| --- | --- |
| "Emi ko ra nkan yen, it was not me." | Understands both halves; replies mixing Yoruba and English; offers the freeze |
| "Abeg, I no do that transaction at all." | Answers in Pidgin, in the English voice |
| "Jowo, freeze my card now." | Freezes without asking you to repeat in English |
| "Ba ni yi wannan ba, please freeze my card." | Hausa route |
| "Wetin dey happen? My card don block." | Pidgin, no correction, no confusion |

**What counts as a pass** — the agent never says "please speak English", never
asks you to repeat, replies in the language you used, and the *safety* line
("I will never ask for your P I N") lands in that language too. A warning
delivered in a language the caller is straining to follow is not a warning.

Also worth trying, unrelated to language: refuse to answer the security question
and watch it transfer; ask "how do I know you're really my bank?"; ask it to
reverse the charges and watch it hand off; offer it your PIN unprompted and check
it stops you mid-sentence.

**Known weak spot.** On a heavily mixed utterance the multilingual STT tends to
transcribe the Yoruba cleanly and drop the trailing English clause. That matches
the benchmark, where ElevenLabs Scribe led on code-switched audio (WER 0.48) and
the rest trailed. It is an input-side limit, not a routing bug — the printed
`[customer]` line shows you exactly what was lost.

### Stage 3 — a real WhatsApp call (inverted flow)

**Meta blocks business-initiated WhatsApp calls from numbers registered in
Nigeria, the USA, Canada, Egypt and Vietnam.** The restriction keys off the
*business* number's country, not the customer's, and it cannot be configured
away. Attempting it returns:

```
(#138013) Business-initiated calling is not available.
```

Inbound (user-initiated) calls work fine in Nigeria, so the flow inverts:

```
fraud signal → WhatsApp alert message → customer taps call
            → we accept the call → agent answers, already briefed
```

This is arguably the better design for fraud anyway. The customer initiates, so
the call cannot be spoofed by someone impersonating the bank — and it avoids
training people to trust unsolicited "your bank calling" calls, which is the
exact reflex vishing relies on.

`pending_alerts.py` holds the briefing between the message and the callback, with
a 60-minute TTL. Ring back later than that and you reach a human instead, because
a stale fraud script is worse than no script.

Three processes:

```bash
python agent.py dev                        # 1. voice worker
uvicorn app:app --reload --port 8000       # 2. webhooks + trigger
ngrok http 8000                            # 3. public URL for Meta
```

Put the ngrok URL + `/webhook/whatsapp` into the Meta Developer Console, subscribe
to the **`calls`** field, and use `WHATSAPP_VERIFY_TOKEN` as the verify token.

#### The order matters — webhook, then calling, then permission

Meta refuses to enable calling until the webhook is already live and subscribed:

```
(#138018) WhatsApp Business calling cannot be enabled because technical
pre-requisites are not met
```

So do it in this order, and check your state at each step:

```bash
python enable_calling.py --check     # number, quality, verification, calling status
python enable_calling.py --enable    # only works once the webhook is subscribed
```

`--check` also prints the messaging tier. Meta wants **2000/day or above** for
calling to function — a new or unverified number starts lower, and calling can
read as enabled while calls still fail.

#### Then get call permission

WhatsApp does not let a business cold-call anyone. The customer must tap to allow
it. Message the business number **from the handset** to open a 24-hour service
window, then:

```bash
python grant_permission.py --to +2348020812523
```

Tap **Allow** on the phone, and pick *Always* rather than the 7-day option — you
only get 2 permission requests per user per week, so don't burn them.

Meta's limits, worth knowing before you debug a silent failure:

| Limit | Value |
| --- | --- |
| Permission requests | 1 per 24h, 2 per 7 days, per user |
| Business-initiated calls | 1 per day, 2 per week, per user (production) |
| Temporary permission | 7 calendar days |
| Auto-revoke | after 4 consecutive unanswered calls |

#### Then fire the trigger

Running `app.py` does **not** place a call — it only starts the server and waits.
The trigger is a POST to `/fraud-alert`, which stands in for the bank's detection
engine. Easiest way:

```bash
python trigger.py                                     # list scenarios
python trigger.py FRD-CARD-FOREIGN --to +2348020812523   # notify mode (default)
python trigger.py --cases                             # what the agent did
python trigger.py FRD-CARD-FOREIGN --mode call        # direct dial (blocked in NG)
```

In the default `notify` mode the customer gets the alert message and the briefing
is parked; tap the call button in that chat and the agent answers with the right
context loaded. `GET /bank/pending-alerts` shows what is waiting.

Or open http://localhost:8000/docs and click through the Swagger UI.

`--to` dials that number instead of the customer on file. **Testing only** — in
production it would read one customer's account details out to a different
person's phone. The agent will still address you as "Yusuf"; change `firstName`
in `bank_data.py` if you want your own name.

> WhatsApp calling is LiveKit Cloud only, and the connector must be enabled on
> your project.

### Fraud scenarios in the simulator

| Signal | What happened |
| --- | --- |
| `FRD-CARD-FOREIGN` | Two card-not-present purchases in Ukraine, ₦377,000, on a Nigeria-only card |
| `FRD-ACCOUNT-TAKEOVER` | New-device login, then ₦1,850,000 to a beneficiary added in the same session |
| `FRD-ATM-VELOCITY` | Three ₦20,000 ATM withdrawals at one terminal in four minutes |

Inspect what the agent did afterwards: `GET /bank/cases`, `GET /bank/audit`.

## WhatsApp setup

WhatsApp calling is **LiveKit Cloud only** — a self-hosted server cannot do it.
You also need a WhatsApp Business number with call permissions enabled in the
Meta console.

Calls are negotiated over SDP that Meta delivers to *your* webhook, so an
outbound call is a three-step dance:

1. `DialWhatsAppCall` — we ask LiveKit to place the call
2. Meta posts a `calls` event to `POST /webhook/whatsapp` carrying an SDP
3. `ConnectWhatsAppCall` — we relay that SDP back, media flows

In the Meta Developer Console, set the webhook URL to your public
`/webhook/whatsapp`, subscribe to the **`calls`** field, and use
`WHATSAPP_VERIFY_TOKEN` from `.env` as the verify token. Expose port 8000 with
ngrok or similar while developing.

### WhatsApp is blocked on a LiveKit closed beta

The full inverted flow works up to the final bridge: the alert sends, the
customer taps call, Meta delivers the SDP, `AcceptWhatsAppCall` answers `200 OK`
— and then nothing ever joins the room, so `session.start()` waits forever.

Two independent causes, neither fixable in this repo:

1. **The LiveKit WhatsApp connector is a closed beta** and must be enabled on
   your project by LiveKit staff. Until then the RPC accepts and silently does
   nothing. Fix: get your project id (`p_…`) from cloud.livekit.io, fill in the
   "Connector Interest Check & Beta Testing" form in LiveKit's Slack community,
   and ask them to enable it.
2. **`livekit-api` 1.0.2 has no connector service at all** — `LiveKitAPI` exposes
   only `room`, `sip`, `egress`, `ingress`, `agent_dispatch`. The native
   `accept_whatsapp_call` / `dial_whatsapp_call` methods arrived in a later
   version, which is why `whatsapp_connector.py` hand-rolls the Twirp calls.

Use **SIP mode** until both are resolved.

## Not built yet

- Per-language prompt variants — the agent code-switches, but its system
  prompt is written in English (`tts_router.py` handles the voice side)
- Real warm transfer to the fraud desk — `transfer_to_human_agent` calls
  `transfer_sip_participant`, which needs a SIP trunk configured
- Rate limiting on repeat calls to the same number
- Persisting the audit trail (currently in memory; MongoDB is already in `.env`)
