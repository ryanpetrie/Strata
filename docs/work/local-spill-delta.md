# The local spill delta: one disk tier, and every upstream line it touches

Branch `local/spill-unify`, from `dbbc4d32b` (the 0.2.21.1 merge into `local/integration`; before that `aa9722de4`, the
0.2.20 merge). This file exists so the next merge of `upstream/main` can replay the local delta from a short list instead
of reading a diff: every upstream file and function we touch, why, and the flag/format migration. Keep it current when the
delta changes.

## What changed, in one paragraph

The branch carried two disk caches for parked conversations: upstream's `ConversationSpill` (the "SSD tier") and the
local `ConversationDiskStore` (the "L2 tier"). They shared the payload format, the identity and the matching rule but not
the index, the admission or the telemetry, and running both could restore the same prompt twice, write every eviction
twice, and hand adapted K/V cells to a base-model request. `ConversationSpill` is now the **one** disk tier. The local
store's four real advantages were ported into it: a streaming save from live K/V sources, a streaming restore with
bounded staging, the content-addressed page pool with copy-on-write, and its telemetry, timed flush, RAM guard and
adapter key. The store, the DirectStorage fast path and their tests are deleted. The local flag family became hidden
deprecated aliases of `--conversation-spill-*`.

## Merge rules this delta follows

- Upstream's declarations and default behaviour are intact: `put()`, `take_serial()`, `best(prompt, images, cvec)`,
  `open(root, cap, error, cleaned, persist, adopted)` and `BackgroundWriter` all still compile and behave as upstream
  with their upstream arguments. Local additions are optional, defaulted or new methods.
- No parallel index, matcher, LRU, budget or namespace: everything reuses the upstream objects (`SpillBudget`,
  `SpillNamespace`, `ConversationSpill::Entry`, `BackgroundWriter`, `SessionFileIdentity`, `ConversationCache::Match`).
- New code lives in new files (`spill_pages.hpp`, `spill_pages.cpp`, the two new tests). Edits inside upstream files are
  the smallest anchors that work, listed below.
- No default changed. `--conversation-spill-mib` is still `-1` (auto), the tier is still on in a normal serve, and the
  page backend and the flush are still off unless asked for.
- Upstream test files are untouched. The behaviour worth keeping from `conversation_disk_test.cpp` moved into the two
  new test files.

## Upstream files touched, and where

### `engine/include/strata/core/conversation_spill.hpp` (header-only upstream file)

| Where | Change | Why |
| --- | --- | --- |
| includes | `spill_pages.hpp`, `<chrono>`, `<memory>` | the page backend, timings, `unique_ptr` readers |
| before `class ConversationSpill` | new: a "local additions" note, `SpillPageOptions`, `SpillSaveReport`, `SpillStats` | value types with no behaviour |
| `Entry` | + `lora`, `paged`, `kv_bytes`, `layout` (all defaulted) | the adapter key, and a paged entry's page list |
| `open(...)` | + defaulted `const SpillPageOptions* page_options`; opens `<dir>/pool`; resets the guard/pool | one entry point, so adoption can see the pool |
| `put(...)` | the inline eviction loop calls `make_room_locked`; records `image.lora`; `min_free_` in the write options; counts a save; `trim_locked()` | the guard, the slot cap, the adapter key, telemetry |
| `best(...)` | + defaulted `ConversationLora lora = kLoraNone`, and the filter | never match across adapter states (upstream callers unaffected) |
| `clear()` | clears the guard first, counts a clearance | `#879` must not spin on a guarded entry |
| `close()` | resets the page pool and the guard | the pool goes with the namespace folder |
| new public methods | `put_streaming`, `take_layers`, `guard`/`unguard`/`guarded`, `stats`, `note_restore`, `note_restore_failure`, `set_min_free_bytes`, `set_max_entries`, `paged`, `page_open_error`, `pool`, `pool_bytes`, `pool_dead_bytes`, `compact`, `compact_needed` | the four ported advantages, behind wrappers |
| private helpers | `make_room_locked`, `trim_locked`, `lru_locked` (skips the guard), `release_pages`, `pages_path`, `layout_kv_bytes`, `put_page_layout`, `write_page_index`, `read_page_index` | eviction with a guard, the page sidecar |
| `superseded_by` | + `e.lora != image.lora` | an adapter state is its own conversation |
| `drop_superseded(...)` (new upstream in 0.2.21) | + defaulted `ConversationLora lora = kLoraNone`, copied into the image it builds | the same rule when the tier drops a snapshot the outgoing conversation extends |
| `meta_write` / `adopt_locked` | magic `SPM1` → `SPM2`: adapter state, paged flag, logical bytes; a paged entry needs its sidecar and the pool; `pool/` is not swept | a restart filters on the same key matching does; an old cache is discarded |
| `drop_locked` | refuses a guarded entry; releases the pool's pages; removes the sidecar | a restore in flight must survive a save |

### `engine/include/strata/core/conversation_snapshot.hpp` (upstream)

- `#include "strata/core/conversation_ds.hpp"` removed, and the 12-line `ConversationDsPage` POD defined in its place.
- `ConversationFastWrite` and `conversation_snapshot_restore_layers`' signature are unchanged: the portable path is
  untouched, and with no DirectStorage caller every restore takes it (the `fast` default is empty).

### `engine/src/program/generate.cpp` (upstream, the serve loop)

| Where | Change |
| --- | --- |
| includes | `conversation_disk.hpp` → `spill_pages.hpp` |
| `Options` (the disk block) | `conversation_disk_*` → `conversation_spill_*` (only/only min-tokens/staging/park-gain/slots/pages/whole-file/compact-*/flush-min/flush-min-tokens); upstream's `conversation_spill_mib/dir/persist` untouched |
| `usage()` (the disk block) | the `--conversation-spill-*` lines (the family now has help text at all) |
| argument parser | the canonical spill flags; one `else if (apply_deprecated_disk_flag(...))`; that helper sits above `main` |
| start-up | the "budget needs a folder" refusal is gone (the folder is derived) |
| the L2 setup block, the demote hook, `park_to_disk`, `spill_image`, the old flush cycle | deleted (~550 lines) |
| the tier's open block (upstream's) | + the page options, `set_min_free_bytes`, `set_max_entries`, the pool/INFO log lines, the flush latch |
| after `spill_clear` | new local block: `spill_stream` (streaming put), `spill_live` (the live-source save), `spill_flush_push`, `conversation_flush_cycle`, `conversation_flushed_at`, `SpillRestoreMark` |
| `spill_write` (upstream's) | streams from the image's host buffers through `put_streaming`, with `put()` as the fallback |
| `park_current_body` (3 call sites) and the disk-only branch | `park_to_disk()` → `spill_live()`; `--conversation-cache-disk-only` → `--conversation-spill-only` |
| the restore's selection | `spill.best(..., want_lora_key)` + `spill.take_layers(...)`: a layer-at-a-time reader instead of `take_serial`'s whole image |
| the restore's pre-park check | the reader is already open; the file's own cvec/lora are re-checked and a mismatch drops the entry |
| the park's supersede block (upstream's, new in 0.2.21) | one block: the RAM cache's `drop_superseded(..., src.lora, &gone)`, then the tier's `drop_superseded(..., src.lora)` into the same `gone` list, then the per-snapshot log line | keeps upstream's snapshot drop and the local adapter key |
| the restore's apply | `conversation_snapshot_restore_layers` on the reader (the local L2 body, retargeted); the DirectStorage branch removed |
| the flush's safe point | `spill_restoring`, `o.conversation_spill_flush_min` |
| `#879` sites | unchanged: they already call `spill_clear()`, which now clears the only disk tier |

### Build files

- `engine/cmake/strata_gpu_engine.cmake`: the engine's source list loses `conversation_disk.cpp`,
  `conversation_pages.cpp`, `conversation_ds.cpp` and gains `spill_pages.cpp`; the DirectStorage include/define block and
  `src/kernels/cuda/ds_pages.cu` are gone; the `ds_pages_test` target is gone.
- `engine/CMakeLists.txt`: `conversation_disk_test` removed; `conversation_spill_test` links `spill_pages.cpp`;
  `spill_pages_test` and `spill_stream_test` added.
- `engine/sycl/CMakeLists.txt`: `conversation_spill_test` links `spill_pages.cpp` (header-only until now).

### New files

| File | What |
| --- | --- |
| `engine/include/strata/core/spill_pages.hpp` | the page pool, ported from `conversation_pages.hpp` (class names kept); `kSpillPoolFormatVersion` = 2 |
| `engine/src/core/spill_pages.cpp` | the pool's implementation; a pool of another version is discarded on the writable open, a miss on a read-only one |
| `engine/src/core/spill_pages_test.cpp` | 56 checks: append/dedup/read, the persisted index, a rebuilt index, a torn tail, a corrupt page, release+compaction, the writer/reader with a one-page rewrite, the version discard |
| `engine/src/core/spill_stream_test.cpp` | 63 checks: the adapter key in `put`/`best`/supersede, the whole-file streaming save and layered restore, the paged save/CoW/restore/adoption, the SPM1 discard, the slot cap, the guard, `clear` and the counters, the background writer's queue |
| `docs/work/local-spill-delta.md` | this file |

### Deleted

`engine/include/strata/core/conversation_disk.hpp`, `engine/src/core/conversation_disk.cpp`,
`engine/include/strata/core/conversation_pages.hpp`, `engine/src/core/conversation_pages.cpp`,
`engine/include/strata/core/conversation_ds.hpp`, `engine/src/core/conversation_ds.cpp`,
`engine/include/strata/kernels/ds_pages.hpp`, `engine/src/kernels/cuda/ds_pages.cu`,
`engine/src/core/ds_pages_test.cpp`, `engine/src/core/conversation_disk_test.cpp`,
`tools/conversation_disk_cache_test.py`.

### Deliberately not deleted

`engine/include/strata/core/conversation_segment_pool.hpp` with `conversation_segment_pool_test.cpp`, and
`engine/src/core/conversation_buffer_pool_test.cpp`. They cover the RAM tier's park buffers (the `STRATA_PARK_SEGPOOL_MIB`
segment pool and the park buffer pool), not the disk tier: no disk-tier behaviour moved out of them, so deleting them
would remove coverage rather than replace it. If the intent was to drop the segment pool too, that is a separate change
with its own default (`STRATA_PARK_SEGPOOL_MIB`, off) to retire.

## The flag migration

Old names are hidden aliases: accepted, mapped, one warning line each, and **not** in `--help`. A preset or script that
still names them keeps working and says why it should be updated.

| Retired flag | Canonical flag | Effect of the alias |
| --- | --- | --- |
| `--conversation-cache-disk-mib N` | `--conversation-spill-mib N` | sets the cap (`0` = off, unset = auto) |
| `--conversation-cache-spill-dir DIR`, `--conversation-cache-disk DIR` | `--conversation-spill-dir DIR` | sets the folder |
| `--conversation-cache-disk-only` | `--conversation-spill-only` | sets it |
| `--conversation-cache-disk-min-tokens N` | `--conversation-spill-min-tokens N` | sets it |
| `--conversation-cache-disk-park-gain N` | `--conversation-spill-park-gain N` | sets it |
| `--conversation-cache-disk-min-free-mib N` | `--conversation-spill-min-free-mib N` | sets it |
| `--conversation-cache-disk-staging-mib N` | `--conversation-spill-staging-mib N` | sets it |
| `--conversation-cache-disk-slots N` | `--conversation-spill-slots N` | sets it |
| `--conversation-cache-disk-cow` | `--conversation-spill-pages` | sets it |
| `--conversation-cache-disk-whole-file` | `--conversation-spill-whole-file` | sets it |
| `--conversation-cache-disk-compact-min-mib N` | `--conversation-spill-compact-min-mib N` | sets it |
| `--conversation-cache-disk-compact-min-pages N` | `--conversation-spill-compact-min-pages N` | sets it |
| `--conversation-cache-flush-min MIN` | `--conversation-spill-flush-min MIN` | sets it |
| `--conversation-cache-flush-min-tokens N` | `--conversation-spill-flush-min-tokens N` | sets it |
| `--conversation-cache-disk-demote` | — | accepted, nothing set: the eviction sink is always on |
| `--conversation-cache-disk-fast-path` | — | accepted, nothing set: the DirectStorage fast path was removed with the disk store |

### Adopted as canonical, new in `--help`

`--conversation-spill-mib`, `--conversation-spill-dir`, `--conversation-spill-persist` were upstream "tester flags, no
help line". They, and the local additions above, are now in `--help`.

### Config files migrated

Every config, preset, script and doc that prescribed the old names now names the canonical ones: the root and `terminal/`
presets (`strata-unsloth-ud-iq4_xs.json`, `terminal/strata-nvfp4-nvfp4.json`, `terminal/strata-unsloth-ud-iq4_xs.json`),
the evidence run configs under `docs/work/evidence/` (`*-config.json`, the two `make-*-config.py`, `disk-fastpath-e2e-drive.py`),
`terminal/serve/web/app.js` (the Monitor's note), `terminal/tools/conversation_cache_disabled.py` and its test,
`terminal/tools/logit_bias_english_example.py`, `tools/research/tools/live-probes-2026-10/logit_bias_http_probe.py`,
`docs/DISK_CACHE.md`, `docs/DETAILS.md`, `docs/work/DISK_CACHE_STATUS.md`, `docs/work/DISK_CONVERSATION_CACHE_PLAN.md`,
and the two evidence write-ups. Captured engine logs (`*.txt`, `*.log`) were left alone as historical records.

## The on-disk formats, and what an upgrade costs

Everything here is a **cache**, not user data: a version bump forgets, it never migrates.

| What | Before | Now | What an upgrade does |
| --- | --- | --- | --- |
| Spill index beside a persistent entry (`conv-N.meta`) | `SPM1` | `SPM2` (adds the adapter state, the paged flag, the logical bytes) | the entry is not adopted and its files are deleted |
| Page pool (`pool/pages.bin` header, `pages.idx`) | version 1 | `kSpillPoolFormatVersion` = 2 | a writable open deletes the pool and makes an empty one; a read-only open is a miss |
| Paged entry's page list (`conv-N.pages`) | — | `SPG1`, written by `put_streaming` | a paged entry without it is deleted |
| Entry payload | a session file, unchanged | the same session file (+ a head with no K/V layers for a paged entry) | unchanged |

A user upgrading loses their spill cache once and re-reads the next prompt. No conversation, model or setting is
touched. Immortalising the old formats was not worth it: the local tier had shipped nothing, and the spill folder is
recreated on the first write.

## Telemetry the product reads

Upstream's lines are kept where the server parses or a user reads them:

- `strata serve: conversation cache: spilled N tokens to disk in X ms; file=… MiB tier=disk …` (saves; the server's
  `DISK_SPILL` regex),
- `strata serve: conversation cache: restored N tokens from disk in X ms (Y MiB/s); tier=disk parked=… bytes=…`
  (`DISK_RESTORE`),
- `strata serve: conversation cache: disk restore failed (…); reading the prompt`,
- `strata serve: conversation cache: disk evicted N entries (LRU); …` and `disk compacted the page pool in X ms; …`,
- `strata serve: conversation cache: not spilling N tokens (…): …` (the cost rule),
- the upstream SSD-tier lines (start line, "SSD tier in …", "could not spill", "read … back from the SSD tier").

The INFO keys are unchanged on purpose: they are the Monitor card's contract (`conversation_disk_mib`,
`conversation_disk_dir`, `conversation_disk_slots`, `conversation_disk_bytes`, `conversation_disk_entries`,
`conversation_disk_paged`, `conversation_disk_pool_bytes`, `conversation_disk_dead_bytes`,
`conversation_disk_fast_path` — the last is always 0 now). Renaming them would have meant a server change for no
product gain; `docs/DISK_CACHE.md` says what they mean.

## What the next merge should expect to conflict on

1. `conversation_spill.hpp`: upstream edits `put`, `best`, `Entry` and `meta_write` often. Take upstream's side and
   re-apply the fields/methods above (they are additive; `best`'s new argument is defaulted).
2. `generate.cpp`'s spill open block and restore block: upstream changes both. The local parts are contiguous and
   marked `// ---- local:` / `// (local`; re-apply them as blocks.
3. `conversation_snapshot.hpp`: if upstream reintroduces `conversation_ds.hpp`, either take it back (and the fast path
   returns) or keep the local POD.
4. `engine/CMakeLists.txt`, `strata_gpu_engine.cmake`, `sycl/CMakeLists.txt`: re-add `spill_pages.cpp` to
   `conversation_spill_test` and to `strata_engine`; keep the two new test targets.
5. `docs/DETAILS.md` §"Disk tier" and `docs/DISK_CACHE.md`: upstream documents the spill tier there now; merge by
   keeping upstream's text and folding in the local flags.

## The 0.2.21.1 replay, for the next merge's benefit

`f87121058` (0.2.21) put the snapshot drop in three places, and each met the local delta differently:

- `conversation_cache.hpp`: upstream added `dropped_tokens` to `drop_superseded()`, the `live_extended_by()` rule, and
  a call from `put()`. Resolution: upstream's rule and out-parameter plus the local `lora` key, in the order
  `(ids, imgs, checks, cvec, lora, dropped_tokens = nullptr)`, so every older call still compiles.
- `conversation_spill.hpp`: upstream added `drop_superseded(ids, imgs, cvec)`. Resolution: upstream's function with the
  local defaulted `lora`.
- `generate.cpp` and `sycl/generate.cpp`: upstream replaced the outgoing-chain call with a `gone` block that also drains
  the tier. Resolution: one block, `src.lora` passed to both drops.

Three upstream test expectations changed with it, and upstream's side is the right one: an equal snapshot now replaces
the older one (the cache holds one, not two), and an entry with no checkpoints is superseded by its live length. Those
conflict with the 0.2.20-era local assertions in `conversation_cache_test.cpp`; the local assertions were dropped.
