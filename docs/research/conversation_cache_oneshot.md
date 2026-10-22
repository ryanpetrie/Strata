# The conversation cache and a decision request's parked state

**2026-10-22.** Status: plan. No engine change, no build, no request to the live
engine. Every number below is from the user's engine log
(`strata-unsloth-ud-iq4_xs.log`, 211,642 lines, 64 engine lifetimes, last start
2026-10-22 02:37) or from a named line of the code. Anything read from the code
and not run is marked **static analysis**.

## 1. The problem

The decision endpoint (`/v1/systemone`) sends very short prompts, one per
question. Each one leaves a parked state behind in the conversation cache. The
cache has two slots in the current launch config
(`strata-unsloth-ud-iq4_xs.json`: `--conversation-cache-mib 8192
--conversation-cache-slots 2`). Two decision requests fill both slots, and the
third evicts the oldest entry - the user's chat. The user's proposal is a
minimum length: never park anything under about 2k tokens.

**The proposal works, and it is already in the shipped engine**
(`--conversation-cache-min-tokens`, added 2026-10-08 in `624070c5`, an ancestor
of this branch's tip; the string is in the binary at
`C:\tools\strata\engine\strata.exe`). But the log says length is a proxy, not
the cause: the cause is that a request which will never be continued is parked
at all. The precise lever is a request that says "do not park me".

## 2. What the log measured

The log holds 64 engine lifetimes. 36 started with `--conversation-cache-slots
2` and 19 with 4 (line 1 is a 4-slot start; line 211,629 is the last, 2 slots).
The churn is almost all in the 2-slot lifetimes: 25,738 of the 26,179 parks.
The numbers below are the 2-slot lifetimes unless said otherwise.

| what | value | how it was counted |
|---|---:|---|
| requests that parked a conversation | 25,738 | `conversation cache: parked ...` lines |
| parked under 2,000 tokens | 25,311 (**98.3%**) |  |
| parked under 500 tokens | 9,893 (38.4%) |  |
| parked tokens: median / p90 / max | 819 / 1,628 / 13,026 |  |
| snapshot size: median / min / max | **237.2** / 226.1 / 960.7 MiB | `snapshot_bytes=` |
| a 22-token park still cost | 225.5 MiB | line 2,336 |
| parks with `reused_kv_bytes=0` | 25,093 (**97.5%**) |  |
| restores (a cache hit) | 649 (**2.5%** of parks) | `conversation cache: restored` |
| ... of those, `(live)`: the same conversation continuing | 591 |  |
| ... `(checkpoint)`: a checkpoint inside a conversation | 58 |  |
| superseded copies dropped | 12,700 | `dropped N superseded` |
| evictions (sum of each lifetime's counter) | **12,325** | `evictions=` |
| worst single lifetime | 6,379 parks, **1,497 evictions**, 150 restores | lines 139,621+ |
| prefill rate, all prompt lines | median 100.4 tok/s (p10 79.3, p90 191.3) | `prompt N tokens = ...` |

Two consequences, both visible in single lines:

* A chat turn that reads from scratch: line 2,344,
  `prompt 4847 tokens = 0 reused + 4847 read in 7601 ms`, 108 generated. It is a
  chat (108 tokens out, 2 checkpoints) and it re-read 4,847 tokens in **7.6 s**.
* A chat turn that hit: line 10,883,
  `prompt 3515 tokens = 3390 reused + 125 read in 1534 ms`. The parked state
  saved most of a 6-14 s read.

And the flood, lines 2,690-2,757: nine decision requests in a row. Each one read
its whole prompt (`0 reused`) and each one parked, with `evictions` climbing
5 → 13 one per request:

```
2690 : prompt 351 tokens = 0 reused + 351 read in 1700 ms, 6 generated, 1 checkpoints
2696 : conversation cache: parked 358 tokens ... parked=4 evictions=5 ... reused_kv_bytes=0
2697 : prompt 292 tokens = 0 reused + 292 read in 1545 ms, 5 generated, 1 checkpoints
2703 : conversation cache: parked 299 tokens ... parked=4 evictions=6 ... reused_kv_bytes=0
...
2756 : conversation cache: parked 302 tokens ... parked=4 evictions=13
```

(That window is a 4-slot lifetime. At 2 slots the chat leaves after the second
decision instead of the fourth.)

## 3. The mechanism

### 3.1 Where a conversation is parked

Parking is a **switch-time** decision, taken by the request that arrives, not by
the one that leaves. At the start of a request the engine calls
`park_current(...)` (`src/program/generate.cpp:12441-12445`) before it restores
anything. `park_current_body` (`:9129`) copies the live session - tokens,
checkpoints and K/V - into host memory as one cache entry, and `put()`
(`include/strata/core/conversation_cache.hpp:422-431`) appends it.

The entry costs one slot and about 237 MiB of **host RAM** whatever its length:
`src/program/generate.cpp:9128` says all parked images live in ordinary host
vectors, and the admission line reads `/proc/meminfo` (`:9243`). The 8 GiB
budget is host RAM, not VRAM. At 2 slots the byte budget never binds (2 × 237
MiB of 8,192 MiB), so **the slot count is the only constraint**.

### 3.2 The third conversation evicts the first

`make_room()` (`conversation_cache.hpp:315-333`) loops while
`entries_.size() >= slots_` and calls `evict_oldest()` (`:341-351`), which
removes the oldest entry that holds no pinned prefix. So with two slots: A, B,
then C evicts A. With four: A-D, then E evicts A. The eviction counter in the
log (`evictions=`) is that call.

The counter for the 2-slot lifetimes sums to 12,325 against 25,738 parks: **one
eviction for every two parks.** That is the "eviction counter climbs through a
long run" of `shared_backbone_decision_policy.md` §12.4, now with numbers.

### 3.3 Why a one-shot decision request parks at all

`serve/server.py:2168` sends `LOGITS <k> <max_new> <ids>` with no keys. The
engine's `ckpt` key defaults to 1 (`generate.cpp:11983`), so after the request
the session is marked continuable: `live_ok = o.prompt_cache > 0 && req_ckpt`
(`:14934`). The next request - the next decision, a chat turn, anything - finds
that conversation live and parks it at `:12441`. The engine has no way to know
that the decision endpoint will never extend it.

The engine already has the opposite switch: `ckpt=0` (`:11979-11983`) is its
one-shot mode, and it means *both* "no checkpoint at the turn boundary" and
"nothing to continue or park" (`:14934`, and `phase2_engine_lora_plan.md` S0.1).
That coupling is why the server cannot simply send `ckpt=0` (option b1 below).

### 3.4 What depends on that parking

1. **The trie walk.** A walk step asks for the row at `ids + node`
   (`server.py:4563`). It continues the session the base request left: of the
   14,393 walk-shaped requests in the log, only **535 (3.7%)** follow a
   `restored` line; 12,723 follow a `park` line and still reuse the session
   (median `1,140 reused + 10 read`), and 1,128 follow no cache line at all.
   So the walk's fast path is the *session's* state, and a parked entry serves
   it in at most one step in 27 - and even then `ckpt` (not `park`) is what
   keeps it. This is also why `ckpt=0` is not a candidate (option b1).
2. **An exact repeat of the same question** (a retry, or the same item again in
   a run). The turn-boundary checkpoint matches and the base prompt is reused.
   The log's 591 `(live)` restores include these.
3. Nothing else. The next decision has a different question, so its prompt
   diverges before the parked checkpoint; and with prompts under
   `--prompt-cache-root 2048` (`:741`) there is no checkpoint inside the shared
   system+state prefix to match either.

### 3.5 Is the parked state ever reused by the decision path?

**No, essentially never.**

* 649 restores against 25,738 parks (2.5%), and 591 of those are the same
  conversation continuing.
* Prompts under 500 tokens: 10,611 lines, **8,616 with `0 reused`** (81.2%).
* The decision windows show `0 reused` on every request (lines 2,690-2,757).

So the decision's parked state is written, occupies a slot, evicts the chat, and
is thrown away. The walk does not save it either: 96% of its steps reuse the
session's own state, and the parked entry is involved in at most 3.7% of them.

### 3.6 What `reused_kv_bytes=0` means, and the end of that mystery

`reused_kv_bytes` is the bytes of K/V the outgoing conversation **already held
from a restore** that the new snapshot can carry over
(`generate.cpp:9260-9262`, `conversation_snapshot_save(..., &reused_bytes)`).
It is not "how much of the cache this snapshot reuses".

* 0 = the conversation read its prompt from scratch and then parked. That is
  97.5% of parks.
* Non-zero = the conversation had been restored before. The log's non-zero
  values run 2.4 MiB to 174.4 MiB, i.e. a partial carry-over.

§12.4 called this "sometimes 0 and sometimes not, so what a restore actually
reuses is not obvious". It is now: a 0 is a conversation the cache never served.
That is the same 2.5% as the restore count, seen from the other side.

## 4. The options

| | what changes | who is hurt | cost | what could go wrong |
|---|---|---|---|---|
| **(a) minimum length** | config only: `--conversation-cache-min-tokens 2048`. Already shipped (`generate.cpp:2541`, `conversation_cache.hpp:217`, `DETAILS.md:1181-1188`) | a chat under 2k loses its reuse: ~10 s per turn at 100 tok/s; a short repeated decision loses a rare hit | none: no code, no latency, no memory | the number is a guess; and the skip returns before `drop_superseded` (`:9143` vs `:9151`), so a conversation that *shrank* (compaction) can leave a superseded copy in a slot |
| **(b1) `ckpt=0` on decisions** | `serve/server.py:2168` only, no engine change | **the walk**: with `live_ok=false` (`:14934`) every walk step re-reads the whole prompt. 96% of the log's 14,393 walk steps reuse the session; the median walk prompt is 1,140 tokens at ~100 tok/s ≈ 11 s a step, 1 step for 50 of the 72 colliding items and 7 for the worst (`server.py:4533-4534`) → up to ~77 s a decision | 0 memory, +11 s per walk step | decisions go from 3.1-4.0 s to tens of seconds. **Rejected** |
| **(b2) a `park=0` request key** | engine: a second flag beside `ckpt` (`generate.cpp:11983`, parse `:12013`, guard `:9130`, set `:14934`, restore path `:12587`, help `:883`); server: one argument on `Engine.logits` (`server.py:2168`) | an exact repeat of the same decision re-reads ~300 tokens (~1.7 s); nothing else | 0 latency, 0 memory, chat hit rate unchanged at every length | per-session state in a long serve loop; and a conversation *taken* from the cache must still be re-parked (a decision never takes a chat entry, but the rule must be explicit) |
| **(c) separate budget or priority** | a `oneshot` bit on `SavedConversation` + `evict_oldest` (`conversation_cache.hpp:341`) preferring it, plus (b2)'s plumbing to carry the bit from the request to the entry | nobody | more code than (b2) for a worse outcome: the decisions still park and still take a slot between turns | interacts with `pinned`, `guard`, `drop_superseded` and the physical-RAM eviction loop (`generate.cpp:9246`); no benefit once (b2) exists |
| **(d1) more slots (2 → 4)** | config; the roadmap's "maybe" | 20 decisions still flush 4 slots | +~474 MiB host RAM | not a fix, only a delay; the roadmap records it as VRAM-blocked |
| **(d2) `pin=N`** | server only: send the shared prefix length | the pinned entry **can never be evicted** (`conversation_cache.hpp:317, 344`), so it holds one of two slots forever | 0 code in the engine | wrong for one-question-per-state calls: the pin buys nothing when the state text is new each time |
| **(d3) evict by last use, not last park** | engine eviction policy | - | - | does not help: the chat is not *used* between decisions either, only parked, so its stamp is as old as the first decision's. **static analysis** |
| **(d4) `ckpt=0` on the last engine request of a decision** | server only | the non-walk items (the server cannot know before the base request whether a walk follows: the decision depends on the base row, `server.py:4679`) | 0 | covers only the walk items. The fallback if no engine build is possible |

## 5. The recommendation

**Build (b2), and use (a) as the stopgap until the build lands.**

1. **Today, no code:** set `--conversation-cache-min-tokens 2048` in the launch
   config. It removes 98.3% of the observed parks, it is documented for exactly
   this case (`DETAILS.md:1181-1188`), and it reverts by deleting one flag. The
   log line it adds is
   `conversation cache: skip parking (N tokens, below the M minimum; parked=P)`.
2. **The fix:** the decision path sends `park=0`, so its conversations are never
   parked, while `ckpt` stays 1 and the walk keeps the live session. Then set
   the minimum back to 0 and re-measure: the chat should keep its reuse at every
   length, including its first two turns, which (a) gives up.

**Is the 2k threshold the right lever? Not as the fix.** It is right as a
stopgap, and for a measurable reason: 98.3% of the parks are under 2k, so the
threshold removes almost exactly the traffic that hurts. But length is standing
in for "will this conversation be continued", and the two are not the same
thing:

* A 1,500-token chat *will* be continued, and (a) makes every one of its turns
  re-read ~10 s of history. That is a real regression the current churn hides.
* A 5,000-token decision prompt would still park, and still evict, under (a).

(b2) states the real condition and has neither failure. It costs one engine key
and one server argument, and it is opt-in per request, so the default path is
untouched.

**(b1) loses** because `ckpt=0` is not just "do not park": it also stops the
session from being continued (`:14934`), and 96% of the walk's steps continue
it. The log's walk numbers say what that costs: 14,342 requests reusing a median
1,140 tokens and reading 10.

**(c) loses** because it keeps the useless parks and only ranks them. It is
strictly more code than (b2) with a strictly worse outcome.

**(d1)-(d4) lose** as described in the table: a delay, a permanent pin, a
policy that does not bind, and a partial version of (b2).

## 6. The measurement

Needs the engine, so it waits until the user frees it. Nothing here is a
substitute for the real run.

### 6.1 The reproduction (before), and the proof (after)

One scripted session against the live endpoint, on a **freshly restarted
engine** (the log shows 64 lifetimes and the cache state carries across
requests, so a cold start is part of the test):

1. Four chat turns over `/v1/chat/completions` that build to **over 2k tokens**
   (so (a) does not skip it). Record turn 4's line:
   `prompt N4 tokens = M4 reused + K4 read`. Expect M4 to be most of N4.
2. **20 decision requests** over `/v1/systemone`, one short question each,
   under 500 tokens, fresh state text.
3. Turn 5 of the chat. Read `prompt N5 tokens = M5 reused + K5 read`.

| | before | after |
|---|---:|---:|
| the chat's `M5` | **0** (evicted; re-reads all of N5) | **≥ 0.9 × N5** |
| the chat's turn-5 read time | 5-30 s at 100-640 tok/s | ~0.2 s |
| `parked=` after the 20 decisions | 2 (the last two decisions) | 1 (the chat's entry) |
| `evictions=` after the 20 decisions | ≥ 18 | 0 |
| `skip parking` lines | 0 | 20-21 (arm (a) only) |

Run it **3 times per arm**, alternating before/after/before/after, and report
the median of the three. A single run is not enough: turn 4's own M varies with
the chat text.

### 6.2 The decision path's own cost

The same 20 decisions, timed end to end, 3 runs per arm, interleaved. The
observed decision medians are 3.1-4.0 s across runs (`shared_backbone...` §10.8:
3.2 s median / 5.4 s p95 cold), so:

* **pass**: median within **+0.2 s** and p95 within **+0.4 s** of the same-day
  before run;
* **fail**: anything larger, because that would mean a lost hit (the repeat
  case) rather than noise.

Also count `prompt` lines per decision: the walk's steps must still show
`~N-10 reused + 10 read`. If they show `0 reused + N read`, the change broke
the live continuation - that is the b1 failure and the test catches it directly.

### 6.3 The answers

The 235-item OpenDecisions sample, 3 runs per arm:

* **fast-path rate must stay 235/235** every run (the doc's own figure, and the
  same in every arm it measured: `shared_backbone...` §10.8);
* the answers must be **identical** to the before run's JSON (the same strings,
  the same confidence values). Parking changes no arithmetic, so a changed answer
  is a bug, not noise;
* the Capability score must stay inside the sample's run-to-run spread of
  **0.6-0.9 points** (§10.8, "the noise floor").

Cost: ~3.2 s median per item → ~13 min a pass, ~40 min an arm, ~80 min for both
arms. Say so before starting, not after.

### 6.4 What would make the fix not worth it

* The chat's `M5` is already 0 in the **before** run for another reason: the
  chat's prompt shares no checkpoint (a changed system prompt), the cvec differs
  (`want_cvec != cvec_cached`, `generate.cpp:12592`), or the chat is under the
  threshold. Then the fix changes nothing and the eviction was not the cause.
* `M5` does not rise in the **after** run: the park was not what kept the chat.
* The decision median rises past the margin: the repeat hit was worth more than
  the chat's (it should not be - the log says repeats are rare).
* The 235 answers change, or the fast-path rate drops below 235/235: the change
  reached the numerics, and it must be withdrawn.

## 7. Risk and blockers

**This is the first change in this program that touches the engine.** The
options above are ranked partly by how little engine they touch. (b2) is the
only one that needs a build.

* **The build.** `src/program/generate.cpp` is the `strata` target
  (`CMakeLists.txt:724`), compiled as CUDA and, with `STRATA_ENABLE_HIP`, as HIP
  from the same sources (`generate.cpp:9094` has the HIP branch). SYCL and Metal
  are separate trees (`sycl/`, `metal/`); `generate.cpp` has no
  `STRATA_USE_METAL` (**static analysis**: I did not build either). Per the
  repo's rule, build CUDA and HIP before review. A CPU-only compile check is not
  enough for a change in the serve loop.
* **Memory.** The cache is host RAM, not VRAM (§3.1). 2 slots ≈ 474 MiB of the
  8,192 MiB budget; the slot count binds, not bytes. The roadmap's "4 slots ≈
  484 MiB below the 1024 MiB threshold" is the same figure, but framed as VRAM;
  that framing should be checked, because the parked bytes are host RAM and the
  1,024 MiB threshold is the free-VRAM warning (`generate.cpp:11689`, and log
  line 2,639 shows 196 MiB free). The change does not add memory; (a) and (b2)
  both *reduce* it.
* **The `reused_kv_bytes` mystery** is closed (§3.6), so it is no longer a
  reason to avoid the cache's eviction path. §12.6 listed that path as one of
  two live candidates for the long-run wrong answers; this document removes it
  as a *mystery*, not as a candidate.
* **Default or opt-in.** The repo's rule is that a new feature is opt-in and the
  default path stays byte-identical. (b2) is opt-in by construction: absent, the
  engine behaves exactly as today, and an older engine ignores the unknown key,
  so a version skew is safe. The server sends it **only** on the decision path,
  behind `STRATA_DECISION_PARK=1` for the first measurement, then on by default
  if the numbers hold. The config flag (a) is the user's choice, not a code
  default, so it does not touch the rule at all.
* **The one thing that could bite:** a request that takes a parked conversation
  out of the cache must put it back. `park=0` must therefore never suppress the
  re-park of a conversation that was restored (`generate.cpp:12587`), only of
  one this request read from scratch. A decision never takes a chat entry today
  (different system prompt), so the user is safe either way, but the rule must be
  in the code, not in the caller.

## 8. The steps

Each is separately reviewable and testable.

| # | what | files | risk | test |
|---|---|---|---|---|
| 1 | this plan | `docs/research/conversation_cache_oneshot.md` | none | read |
| 2 | the stopgap: the minimum length | the user's `strata-unsloth-ud-iq4_xs.json` (not this repo) | none; revert = delete the flag | §6.1 before/after; `skip parking` lines appear for the 20 decisions |
| 3 | the `park=0` key | `src/program/generate.cpp` (`:11983`, `:12013`, `:9130`, `:12587`, `:14934`, `:883`), `docs/DETAILS.md` near `:1181` | medium: per-session state in the serve loop; needs CUDA + HIP builds | a manual `LOGITS ... park=0` then another request → no `parked` line; `park=1` → a `parked` line; the walk still shows `N-10 reused + 10 read` |
| 4 | the server sends it | `serve/server.py:2168` (an argument to `Engine.logits`), behind `STRATA_DECISION_PARK` | low: one call site, decision-only | `serve/test_decision_engine_turn.py`: the line written to the engine carries `park=0`; the chat's `GEN` lines do not |
| 5 | the measurement | a new `tools/decision_policy/cache_churn.py` | none | §6.1-6.3, both arms, 3 runs each |
| 6 | make it the default | `serve/server.py` (drop the gate), this doc (the numbers) | low, but only after step 5 passes | the same measurement, one more time |

Step 2 can ship alone and should, because it helps the user today. Step 3 does
not depend on it. Step 6 must not happen before step 5.

## 9. What this document does not measure

* I did not run the engine, restart the server or send a request. The log is the
  user's, from their own workload.
* The 235-item sample's run-to-run spread (0.6-0.9 Capability points) and its
  latency figures (3.2 s median / 5.4 s p95) are quoted from
  `shared_backbone_decision_policy.md` §10.8, not re-measured here.
* The walk's step counts (50 of 72 colliding items need one extra request, seven
  for the worst) are from `serve/server.py:4533-4534`. `trie_costs.py` measures
  them, but the sample file is not in the repository, so I did not run it.
* Which request a `parked` line belongs to is read from the log's line order: the
  park and the prompt line are adjacent for these requests (as in the window at
  lines 5,799-5,800). The counts in §3.4 depend on that reading. A request the
  engine parks and then also serves from its own session is the case this
  document is about; the code at `:12441` decides it, and I did not trace one
  such request in the engine.
* The SYCL and Metal build claim is **static analysis**: I did not build them.
