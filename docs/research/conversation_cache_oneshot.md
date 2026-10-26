# The conversation cache and a decision request's parked state

**2026-10-22.** Status: the plan's steps 3 and 4 are built and step 5's measurement has been
run (§11): the fix passes its pre-registered rule on three runs an arm. The engine was not
touched on disk; `C:\tools\strata\engine\strata.exe` and the live config are as they were.
Every number in §1-§6 is from the user's engine log (`strata-unsloth-ud-iq4_xs.log`, 211,642
lines, 64 engine lifetimes, last start 2026-10-22 02:37) or from a named line of the code;
the numbers in §10-§11 are from the runs named there. Anything read from the code and not
run is marked **static analysis**.

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

## 10. As built

**2026-10-22, `feat/cache-oneshot`.** Steps 3 and 4 of §8. At the time of this record the
engine had not been run; §11 is the run that followed. The default path is unchanged, byte
for byte (§10.4), and the tests that say so are in §10.3.

### 10.1 What changed, and where it is

| file | line | what |
|---|---|---|
| `src/program/generate.cpp` | `9041` | `live_parkable`: whether the session the queue holds may be parked by the next request. One flag, per session; true at start |
| | `9141-9146` | the guard at the top of `park_current_body`: not parkable means the same non-error skip the minimum-length and budget paths take, and the log line `skip parking (the request that read it said park=0; parked=P)` |
| | `12005-12006` | the request key: `bool req_park = true;` beside `ckpt`, and `bool took_parked = false;` |
| | `12037` | the parse, `park` beside `ckpt`. Unknown keys are still skipped, so an older engine ignores it |
| | `12613` | `took_parked = true;` after a parked conversation was restored into the session |
| | `14964` | at the end of a finished request: `live_parkable = conversation_park_allowed(req_park, took_parked)` |
| | `14974` | the same for a request stopped between two prompt windows, which keeps its conversation (#1620) |
| | `11933` | `LOAD` of a session file sets it back to true (no request key there) |
| `include/strata/core/conversation_cache.hpp` | `201-206` | `conversation_park_allowed(park_key, took_parked)`: the one rule, `park_key || took_parked`. In the header so it can be tested without a GPU |
| `src/core/conversation_cache_test.cpp` | `561-573` | the engine test of that rule and of the slot it saves (§10.3) |
| `serve/server.py` | `312-317` | `decision_park_off()`: `STRATA_DECISION_PARK` set and not `0` or empty |
| | `2154-2184` | `StrataEngine.logits(..., park=True)`; `park=False` writes ` park=0` between `max_new` and the ids |
| | `4514-4522` | `Service._engine_logits`, the one call the decision path uses. Off, it calls `self.engine.logits(ids, k, max_new, cancel)`, argument for argument today's |
| | `4534` | the trie walk's steps take it (`_logits_row`) |
| | `4685-4686` | the base request takes it |
| `docs/DETAILS.md` | `1189-1197` | the key, what it does, what it does not do, and the log line |
| `serve/test_decision_engine_turn.py` | `285-371` | the server's tests: off by default, on with the switch, on every walk step, never on a chat's `GEN` |
| `serve/test_logits_first_token.py` | `911-921` | the wire format of `LOGITS <k> <max_new> park=0 <ids>` |

The state is per session because parking is not the request's own decision: the request
that *reads* a conversation decides whether it may be parked, and the request that
*replaces* it does the parking. So the key is remembered in `live_parkable` at the end of
the request that read it.

### 10.2 Which request lines carry the key

```
LOGITS <k> <max_new> park=0 <id,id,...>          the base request, one per question
LOGITS <k> 1 park=0 <id,id,...>,<node tokens>   each trie-walk step
```

Both of the decision path's LOGITS calls go through `_engine_logits`, so both carry it.
**The walk's steps carry it too**, for one reason: `live_parkable` is set at the end of
*every* request (`:14964`), so a walk step without the key would set it back to true, and
the step that matters - the last one, the one that leaves the conversation behind - would
be parkable again. The walk keeps its checkpoint: no decision request sends `ckpt`, so
`ckpt` is 1 on all of them and the step still resumes the live session (§3.4). The key
only stops the park.

The key comes **before** the ids, not after: the engine's key loop (`:12022-12064`) ends at
the first token without `=`, and everything from there is parsed as the token ids. A key
after the ids is an invalid token id (`ERR bad request`).

Nothing else gets the key. `GEN`, `GENI` and `BGEN` lines are built by `sampling_keys`, which
this change does not touch, so a chat request cannot carry it (§10.3's last test).

### 10.3 The tests and the build

**The engine's test that needs no GPU** is `conversation_cache_test` (CMake option
`STRATA_BUILD_CONVERSATION_TESTS`). The new case is the rule and its effect on a slot: a
conversation read from scratch with `park=0` takes no slot, so the chat parked before it is
still the one a new request restores; a conversation taken from the cache is parked again
even with the key.

```
cmake -S . -B build-cache -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build build-cache --target conversation_cache_test
build-cache\conversation_cache_test.exe
  -> conversation_cache_test: 4778 checks passed
```

(Run from a Visual Studio developer shell, so `cl` is on the PATH. The artifact stays in the
worktree's `build-cache\`; no binary the user runs is touched.)

**The compile check of `generate.cpp`**, with CUDA configured as the release build is:

```
cmake -S . -B build-cache-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build build-cache-cuda --target CMakeFiles/strata.dir/src/program/generate.cpp.obj
  -> build-cache-cuda\CMakeFiles\strata.dir\src\program\generate.cpp.obj (20,097,389 bytes)
```

The object compiles; the target was not linked and nothing was written to `engine\`.
**HIP was not built**: this machine has no ROCm or `hipcc`. That is the one item of §7's
"build CUDA and HIP before review" that this record cannot claim, and it is the one a
reviewer with an AMD card must do.

**The server's tests, no GPU, mock engine** (`C:\tools\strata\.venv\Scripts\python.exe`,
system Python has no `jinja2`):

| command | result |
|---|---|
| `python -m unittest serve.test_decision_engine_turn serve.test_logits_first_token` | 14 + 70 tests, OK |
| `python -m unittest serve.test_systemone serve.test_decision_monitor` | 142 tests together with the first file, OK |
| `python -m unittest serve.test_server serve.test_logit_bias serve.test_model_card_sampling serve.test_cache_drop serve.test_prompt_encoder serve.test_request_accounting serve.test_structured` | 390 tests, OK (6 skipped) |
| `python -m unittest serve.test_parallel` | 42 tests, OK |

The four new server cases:

1. **Not sent by default.** `STRATA_DECISION_PARK=0`: every LOGITS call the decision makes
   carries `park=True`, and the wire-format test pins the line itself,
   `LOGITS 64 32 11,22,33` - the same string the test asserted before this change.
2. **Sent when the switch asks.** With `STRATA_DECISION_PARK=1`, no decision call carries
   `park=True`.
3. **Every walk step carries it.** The options share a first token, so the walk runs; more
   than one engine request is made and none of them is parkable.
4. **A chat's `GEN` line never carries it**, with the switch off and on:
   `GEN 8 1,2,3`, byte for byte.

The smoke test the plan's step 3 asked for (a manual `park=0`, then another request) needs
the engine; §11's runs are the equivalent with the real endpoint, and their logs carry
exactly the lines §10.6 names.

### 10.4 When the key is absent

The engine: `req_park` stays true and `took_parked` false, so `live_parkable` is set to
true at the end of every request and the guard at `:9141` is never reached. No `GEN` or
`LOGITS` line changes, no log line is added, `park_current_body` runs exactly its old path,
and no memory or latency is added. An engineer reading the diff can check this from
`conversation_park_allowed(true, false) == true`.

The server: `_engine_logits` calls `self.engine.logits(ids, k, max_new, cancel)` with the
switch off - today's arguments, not today's arguments plus a default - and only passes
`park=False` when the variable is set. So an older engine, or a test double with the old
four-argument `logits`, is never handed the new keyword on the default path.

Version skew both ways is safe: a new server with an old engine sends nothing (switch off);
with the switch on, the old engine skips the unknown key (`:12064`) and parks as before.
A new engine with any server parks as before.

### 10.5 What differs from the plan

1. **The walk's steps.** Step 4 said "an argument to `Engine.logits`". As built it is an
   argument to one helper that both LOGITS call sites use, and the walk's steps carry the
   key. §10.2 has the reason: with per-request state, a step without it undoes the base
   request's key.
2. **The restore path.** Setting `live_parkable = true` at `:12587` alone would not work:
   the end-of-request assignment (`:14964`) runs later and would overwrite it. The restore
   records `took_parked`, and the end-of-request rule ORs it. That is the plan's trap
   (§7), expressed once in `conversation_park_allowed` and tested without a GPU.
3. **No help-text line.** The plan listed `:883` (the `--help` text). There is no
   command-line flag: this is a request key, so it is documented in `docs/DETAILS.md`
   beside `--conversation-cache-min-tokens`, and in the comment at the parse.
4. **`park=0` and an exact repeat.** A repeat of the same question restores the previous
   decision's entry (a hit) and, because it `took_parked`, parks it again at its end. So the
   key does not throw away §3.4's second case; it only stops a conversation that read from
   scratch from taking a slot.
5. **The switch is read per request** (`decision_park_off()` at each call), like the engine's
   other switches. The two arms need no code change: arm A is the server without the
   variable, arm B with it (a launch config can carry it as `env`).

### 10.6 The measurement, ready to paste

**Run:** §11 is the measurement that followed these commands, with the differences (the
resident copy, the decision size, no latency) said there.

The engine has to be free first. Build and stage the new binary from the worktree (this is
the step this record did **not** do, because the user's engine is serving):

```
cmake --build C:\tools\strata-wt-cache\build-cache-cuda --target strata
# artifact: C:\tools\strata-wt-cache\build-cache-cuda\strata.exe
```

The arms, 3 runs each, alternating A B A B A B, a **fresh engine start per run** (§6.1).
Arm A is today's server and engine; arm B is the new engine and the server started with
`STRATA_DECISION_PARK=1`. `$D` is the OpenDecisions sample directory §9 names.

```powershell
$D   = "C:\Users\ryanp\AppData\Local\Temp\opencode\opendecisions"
$PY  = "C:\tools\strata\.venv\Scripts\python.exe"
$LOG = "C:\tools\strata\strata-unsloth-ud-iq4_xs.log"     # the user's engine log
$env:STRATA_DECISION_PARK = "1"                            # arm B only; unset it for arm A
```

1. Four chat turns over `/v1/chat/completions` that build past 2k tokens: the user's chat
   client, or step 5's script. Record turn 4's line
   `prompt N4 tokens = M4 reused + K4 read`.
2. Twenty decision requests over `/v1/systemone`, one short fresh-state question each. The
   endpoint's harness is the `opendecisions.py` command below.
3. Turn 5 of the chat. Read `prompt N5 tokens = M5 reused + K5 read`.
4. From the log, after the twenty decisions. The launch appends to one log across engine
   lifetimes, so take the arm's own window first (`$FROM` is the log's line count before
   the arm's first request):

```powershell
$FROM = (Get-Content $LOG).Count     # before the arm; write the arm's lines to its own file
Get-Content $LOG | Select-Object -Skip $FROM | Set-Content measure\b1\arm.log

# arm B: 20-21 of these; arm A: none
(Select-String measure\b1\arm.log -Pattern 'skip parking \(the request that read it said park=0').Count
# arm B: 1 (the chat's entry); arm A: 2 (the last two decisions)
(Select-String measure\b1\arm.log -Pattern 'conversation cache: parked').Count
(Select-String measure\b1\arm.log -Pattern 'evictions=' | Select-Object -Last 1).Line
# the walk's steps: these must still read "~N-10 reused + 10 read"
Select-String measure\b1\arm.log -Pattern 'prompt \d+ tokens = \d+ reused' | Select-Object -Last 5
```

The answers and the decision's own cost (§6.2, §6.3): one pass over the 235 items per arm,
three passes each, from the same sample and the official scorer:

```powershell
& $PY C:\tools\strata-wt-cache\tools\decision_policy\opendecisions.py `
      --items "$D\opendecisions.jsonl" `
      --out "measure\b1\predictions.jsonl" --log "measure\b1\runlog.jsonl"
& $PY "$D\score.py" "measure\b1\predictions.jsonl" --data "$D\sample_items.jsonl"
```

`opendecisions.py` is `--host 127.0.0.1 --port <the server's> --api-key <if set>`; the
runlog has each item's latency, which is where the 3.2 s median / 5.4 s p95 of §6.2 is
compared. The chat arm's script does not exist yet: it is step 5's
`tools/decision_policy/cache_churn.py`, and until it does, §6.1's first and third steps are
done by hand.

**The proof that the key reached the engine**: arm B's log carries
`conversation cache: skip parking (the request that read it said park=0; parked=P)` once per
decision switch, and the decision's `LOGITS` requests leave no `conversation cache: parked`
line. Neither line exists in arm A.

## 11. The measurement, run (2026-10-22)

**What was run.** §6.1's recipe, on the engine §10 built: `build-cache-cuda\strata.exe`
(75,546,112 bytes, 13:46), a copy of the live config whose `exe` is that binary, arm A with
`STRATA_DECISION_PARK` unset and arm B with it set to `1`, three runs each alternating
A B A B A B, a fresh server and engine per run. Each run: four chat turns that build the
conversation to **7,258 tokens** (turn 1; turns 2-4 add 30-60), **20 decisions** with fresh
state text, then chat turn 5. The server is the branch's `serve/server.py`; the scripts are
`C:\Users\ryanp\AppData\Local\Temp\opencode\cache_measure\` (`run_cache.py`, `cache_arm.py`,
`analyze.py`), and each run's engine log is beside them under `runs\A1` ... `runs\B3`.

**The host, as it was.** An RTX 4090 with **9,890 MiB in use by other work** (14,249 MiB
free) and 63.7 GB of RAM, of which another session held 9.6 GB. Two consequences, both
recorded rather than hidden:

* **The RAM copy had to come down.** With the live config's `--resident-budget-gib 36` the
  engine held 34.5 GB and the host had **261 MiB** available; the engine's own admission
  then refused every park - `skip parking (physical RAM admission; need 671 MiB plus 2560
  MiB floor, 261 MiB available)`, measured in the first attempt - so arm A could not park,
  which is not today's behaviour and cannot measure the fix. Every arm's config copy
  therefore sets **`--resident-budget-gib 20`** and nothing else. The 20 GiB copy leaves
  ≈7 GB and the floor is met: the chat's snapshot is 704 MiB at 7,258 tokens and a
  decision's 263 MiB at 1,822.
* **Latency is not a result here.** The card is shared and the arms are not interleaved for
  speed; the readouts below are the cache's own (tokens reused, evictions, parks), which do
  not depend on the clock. §6.2's latency margin is **not run**.

**One change from §6.1's decision prompts.** The 20 decisions are **1,815-1,881 tokens**,
not "under 500". The live config keeps `--conversation-cache-min-tokens 1024` (the shipping
state, as asked), and a 300-token decision is already skipped by that minimum: arm A would
not evict the chat and the two arms would differ in nothing. The decisions are over the
minimum so that the shipping config parks them. Five of the 20 are choice questions whose
options (`rule_1`, `rule_2`) share their first token, so the trie walk runs and its reuse
can be read in both arms.

### 11.1 The six runs

| run | arm | chat 5: prompt = reused + read | M5 / N5 | evictions | parked | `skip parking (park=0)` |
|---|---|---|---:|---:|---:|---:|
| A1 | A | 7,467 = **0** + 7,467 | 0.0% | **19** | 2 | 0 |
| A2 | A | 7,477 = **0** + 7,477 | 0.0% | **19** | 2 | 0 |
| A3 | A | 7,458 = **0** + 7,458 | 0.0% | **19** | 2 | 0 |
| B1 | B | 7,453 = **7,423** + 30 | **99.6%** | **0** | 1 | 25 |
| B2 | B | 7,459 = **7,429** + 30 | **99.6%** | **0** | 1 | 25 |
| B3 | B | 7,461 = **7,430** + 31 | **99.6%** | **0** | 1 | 25 |

The engine's line, not the client's usage: `strata serve: prompt N tokens = M reused + K
read` (the server's `cached_tokens` agrees on all six). Every run answered all 25 requests.

* **Arm A is the plan's `before`, exactly.** The four chat turns reuse 7,281-7,394 tokens
  each; the 20 decisions each read their whole prompt (`0 reused`), park, and by the third
  the oldest entry goes: **19 evictions**, 26 park lines, 2 left. Chat turn 5 then finds
  nothing and reads all 7,467 tokens: **M5 = 0**.
* **Arm B is the plan's `after`.** The decisions still read their whole prompt (the key
  does not make a fresh conversation reusable) but they are not parked: **0 evictions** in
  every run, the chat's entry stays the only one (`parked=1`), and turn 5 resumes it:
  **99.6% reused**. The parks that remain are the chat's own: the chat path parks and
  restores its own conversation between turns (1, 2 and 3 parks in the three runs), which is
  today's behaviour and is not the key's doing.
* **The skip count is 25, not 20.** The 20 decision switches, plus 5 more: each trie-walk
  step resumes from a checkpoint 10 tokens back rather than from the live session, so it
  tries to park the base request's conversation too, and the key suppresses that as well.
  That is the fix working on the walk, not a fault.
* **The walk is untouched.** Every run's walk steps read
  `1,877 = 1,867 reused + 10 read` and `1,879 = 1,878 reused + 1 read` (99.5% and 99.9%),
  identical in both arms. `ckpt` stays 1 throughout, so the walk's fast path is what it was.

### 11.2 The 235, on one run per arm

| run | items | fast path | Capability | Intelligence | Calibration | answers moved vs `fix_serial.jsonl` |
|---|---:|---|---:|---:|---:|---:|
| stored `fix_serial.jsonl` | 235 | 235/235 | 79.8 | 78.0 | 81.7 | - |
| arm A (`q235\A1`) | 235 | 235/235 | **79.8** | 78.1 | 81.4 | 2 (choice 1, score 1) |
| arm B (`q235\B1`) | 235 | 235/235 | **78.3** | 76.4 | 80.3 | 6 (choice 2, noul 2, score 2) |

Both arms answered every item and both are inside the sample's measured **1.8-point**
run-to-run spread (A is equal, B is 1.5 below). The 6 moves in B against a stored run of a
different engine state are the engine's own wobble (the two stored base runs span 1.8
Capability points and differ in 2 noul items, §23); the key changes no arithmetic, and the
235's answers are produced by the same rows either way.

### 11.3 The verdict, against §6's pre-registered rule

* **The fix works.** Arm B: `M5 >= 0.9 x N5` (99.6% in all three) **and** `evictions = 0`
  (0 in all three). Arm A: `M5 = 0` (0 in all three) **and** `evictions >= 18` (19 in all
  three). The walk's reuse is intact in both arms (99.5% / 99.9%).
* **Not "not worth it"** by §6.4: arm A's `M5` is 0 *because the chat was evicted* - its
  entry was the one the decisions pushed out, and arm B keeps it - not for a changed system
  prompt or a cvec mismatch, and `M5` does rise in B on the same conversation.
* **The cost** §6.2 would measure is not measured (shared card). What the run does say is
  that the decisions pay the same read they paid before (`0 reused` in both arms) and the
  key only removes the park and the eviction; it cannot slow their arithmetic.

### 11.4 The limits, and what was not run

* One model, one card, one engine build; 3 runs per arm; 2 cache slots and
  `--conversation-cache-min-tokens 1024`, the shipping values. Another slot count or no
  minimum changes which requests park, not what `park=0` does.
* The decisions are 1,815-1,881 tokens. The plan's 300-token decisions would not park at
  all under the shipping minimum, so they cannot show a difference between the arms; with
  the minimum set back to 0 (the plan's step 2 revert) they would, and that arm was not run.
* The `--resident-budget-gib 20` change is the one departure from the live config, forced by
  the host's RAM (above). It changes what the engine keeps in RAM, not the cache's rules.
* Not run: the §6.2 latency comparison; the §6.3 three runs per arm (one each); a live
  default-on arm (§8's step 6); any AMD or Mac engine; a second model. The 235's own
  run-to-run spread is quoted from §23, not re-measured here.


