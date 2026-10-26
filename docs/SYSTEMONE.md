# The decision endpoint: `POST /v1/systemone`

Off by default. Turn it on with `"systemone": {"enabled": true}` in `strata-<model>.json`
and restart the server; `/v1/status` then lists `/v1/systemone` among the dialects. It
answers a decision request - a `state` and one to ten named `questions` - with one
answer per question, in the TypeSafe SystemOne shape: `noul` (yes/no), `choice` (one
of the options) and `score` (one of the ordered levels).

The endpoint is served by the model Strata already has loaded. There is no second
model, no second copy of the weights, and no training: the request is turned into a
prompt, and the model's own next-token distribution at the answer position is read
instead of sampled.

```sh
curl -s http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" \
  -d '{"state": {"plan": "enterprise", "days": 41},
       "questions": {"urgent": {"type": "noul", "instructions": "Is this an outage?"}}}'
```

```json
{"answers": {"urgent": {"type": "noul", "noul": 0.012, "confidence": 0.014}}, "calibrated": true, ...}
```

**What the numbers mean.** A greedy decode gives the argmax, so a pick would be
reported as 1.0 and every other option as 0.0. The endpoint reads the model's own
distribution instead, so the answer carries a real probability:

- `probabilities` is the distribution over the options, and it adds up to 1. It is
  the *closed* reading: the weight the model put on anything else (wanting to end
  the turn, writing a digit) is not counted.
- `confidence` is the model's *own* probability of the pick: the weight it put on
  that option out of the whole vocabulary. It is never larger than the
  distribution's maximum. Measured on a 235-item sample of the OpenDecisions set,
  it ranks a correct answer above a wrong one better than the closed reading does
  (the AUC against correctness 0.84 against 0.81 on the choice questions).
- A `noul` answer is the probability of "yes" among the two answers, so the value
  and the answer always agree. It carries a `confidence` too: the model's own
  probability of the side it picked. A confident "no" therefore reports a small
  value and a large confidence - `{"noul": 0.03, "confidence": 0.95}` means "the
  answer is no, and I am fairly sure of it". A consumer that does not know the
  field can ignore it.

**When the model was not going to answer with an option.** If the model's most
likely token at the answer position is not one of the options - it wanted to write
a digit, a phrase, a rate - the endpoint still answers: the model's own text when
that text is a valid answer (a yes/no given as a rate), and otherwise the option
the model would have picked, with the **confidence carrying the doubt** (the
weight it put on the option is small, so the confidence is small). The diagnostics
report it as `model_picked`.

A reply the model wrote that does not parse is a **502**, never a made-up answer -
unless the endpoint has the probabilities, in which case they are the better
answer.

**Several questions.** A request with more than one question is answered **one
question at a time**: each question runs the same path a one-question request runs,
and the answers are merged into one response. So every answer in the response carries
the model's own probability, exactly like a one-question request does, and the
response says `"calibrated": true`.

**A question that cannot be answered does not cost the others their answers.** The
response shape has no per-question error field, and a made-up answer is worse than a
missing one, so a question the model failed to answer is simply **left out** of
`answers`; the other questions keep theirs and the request is still a **200**. The
reason is in the server's log, and in the diagnostics when they are on. When *no*
question could be answered there is nothing to return, and the request is a **502** -
the same answer a one-question request gives in the same situation.

The old shared prompt is still there: put `"multi": "shared"` in the same block
(`"systemone": {"enabled": true, "multi": "shared"}`) and the questions are answered
by one prompt again. Its answers carry no probability (`"calibrated": false`), and it
is not the default.

**What it costs.** One prefill, the same as a chat turn: about 2 s for a new state
and about 0.2 s for a state the conversation cache already holds. A request with
several questions is one engine turn per question: the state is in every question's
prompt, so the prefill is repeated per question (the conversation cache can reuse the
state's tokens, but that has not been measured). When two of the options start with
the same word, one more row is read - the one where the options are told apart -
which takes a few hundred milliseconds, and only for those.

Add `"diagnostics": true` to the same block to get the alternative reading beside
the answer: the model's own probability of every option, the weight it put on
ending the turn, and the weight it put on everything that is not an option. It is
a diagnostic only - the answer does not change.

**Not parking the conversation.** Each decision question reads its prompt from
scratch and nothing later extends it, so the engine request carries `park=0`: the
conversation cache does not park that conversation and short decisions do not evict
the chat they were interleaved with. A conversation the request **restored** from
the cache is parked as before: the key suppresses only the park of one the request
read from scratch. Absent, every conversation is parked as before, and an engine
without the key skips it, so a version skew is safe. The log line is
`conversation cache: skip parking (the request that read it said park=0; parked=P)`.
The server sends the key on the decision path **by default** since the measurement in
[conversation_cache_oneshot.md](research/conversation_cache_oneshot.md) passed
(§11-§12); `STRATA_DECISION_PARK=0` in the server's environment turns it off and
restores the old call. The cache itself: [DETAILS.md](DETAILS.md#conversation-cache).

**A decision in the Monitor.** A decision is a request like any other, so it shows
up in the web app's Monitor tab: the state card says **Deciding** while the engine
reads the state, and the request list has a **Decision** row with the questions it
answered, the tokens it read and the time it took. The API request monitor
(`/api-monitor`, [DETAILS.md](DETAILS.md#api-request-monitor)) keeps the same row and
shows what the decision answered in its Output pane. The answer itself carries the
`timings` the chat path reports, so a client can see what the decision cost.

**A decision in the web app.** The web app has a **Decide** tab, next to Chat, Images,
Monitor and About. It is a client of the same endpoint, over the same API key: you fill
in the state and the questions, the page shows the exact request body, and it sends it
to `POST /v1/systemone`. The answer comes back as one card per question: the pick, the
distribution over the options, the confidence, and - with `"diagnostics": true` - the
numbers behind them, each with a plain-language explanation. The tab is hidden when the
endpoint is off, and the page never invents an answer: a request the server refuses is
shown as the server wrote it.

The measurements, the design and what is still open are in
[docs/research/shared_backbone_decision_policy.md](research/shared_backbone_decision_policy.md).
