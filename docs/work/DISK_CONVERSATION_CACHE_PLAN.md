# Disk-backed conversation cache — implementation plan

**Status:** hand-off draft for implementation.
**Branch:** `feat/disk-conversation-cache` (worktree `C:\tools\strata-disk`), based on `origin/main` `4b9e05a2`
(Strata 0.2.6, engine 0.1.51).
**Combined test branch:** `local/integration` (worktree `C:\tools\strata`) has the decision endpoint, LoRA, and
research work; merge the disk branch into it for combined testing, but keep the PR branch independent.
**Owner:** the implementing agent. Read `AGENTS.md` and `CONTRIBUTING.md` before writing code or a PR.

This plan is deliberately concrete about the seams in the current engine. Line numbers are approximate and drift; use
the named functions and files.

---

## 0. TL;DR for the implementer

Build a **second cache tier**: the existing host-RAM conversation cache stays as L1, and a new opt-in disk tier holds
the conversations that are too large for RAM. Save the engine's existing `SavedConversation` to a private directory,
restore it layer-by-layer (never as one 5 GB host image), and add LRU/size limits and atomic writes. Keep the default
path off and byte-identical.

The first useful milestone is **Phase 1**: a portable, single-GPU disk tier that makes a 200K-token conversation
alternating with small sub-agents restore in seconds instead of re-reading 250–280 s of prompt. DirectStorage is a
later optimization, not the feature.

**Do not start with DirectStorage.** Start with the measurement spike, then the portable disk tier.

---

## 1. Why this exists: the measured problem

### 1.1 The live case (measured 2026-10-25)

A Windows host with a 24 GB NVIDIA card, 63.7 GiB RAM, `Qwen3.8-Flash-Next` UD-IQ4_XS, `--max-context 262144`,
`--kv int8 --kv-resident 32768`, `--resident-budget-gib 36`, `--conversation-cache-mib 8192`,
`--conversation-cache-slots 4`. One main agent (~200K tokens) alternated with small sub-agent requests.

From the server's `/metrics` and the engine log:

| Prompt tokens | Reused | Prompt time | Notes |
|---:|---:|---:|---|
| 203,236 | 0 | 250.2 s | main, cold after engine start |
| 8,263 | 0 | 12.3 s | sub-agent |
| 205,878 | **0** | 281.1 s | main returns |
| 14,632 | 8,434 | 13.8 s | sub-agent, restored from RAM |
| 26,661 | 0 | 49.3 s | sub-agent |
| 207,676 | **0** | 279.8 s | main returns |
| 26,952 | 26,815 | 3.4 s | sub-agent, restored from RAM |
| 15,760 | 0 | 29.6 s | sub-agent |
| 208,225 | **0** | 280.1 s | main returns |
| 18,368 | 16,886 | 10.4 s | sub-agent, restored from RAM |

- 10 requests, 3 reused. 52,135 of 935,651 prompt tokens reused (**5.6%**).
- The four ~200K requests read **825,015 prompt tokens** from scratch: **~1,091 s of prefill**.
- The small conversations reused correctly (restores took 90–154 ms), which proves the cache's matching and restore
  path works. The main conversation is the failure.

### 1.2 The engine said exactly why

```
conversation cache: skip parking (physical RAM admission; need 3788 MiB plus 2560 MiB floor,
                    5832 MiB available, source meminfo; evicted 0, 0 still parked)
conversation cache: skip parking (... need 3796 MiB plus 2560 MiB floor, 5587 MiB available ...)
conversation cache: skip parking (... need 3823 MiB plus 2560 MiB floor, 5191 MiB available;
                    evicted 1, 0 still parked)
conversation cache: skip parking (... need 3826 MiB plus 2560 MiB floor, 5474 MiB available;
                    evicted 1, 0 still parked)
```

The engine refuses to **park** the 200K conversation because the snapshot needs ~3.8 GiB of *new* host RAM plus the
2.56 GiB safety floor, and only ~5.2–5.8 GiB is available. It evicts the parked small conversations and still fails.
The 8 GiB cache budget is a **cap**, not a reservation; the physical-RAM admission gate is the binding constraint.

### 1.3 Why it does not fit

- The host has 63.7 GiB total and 4.7–6.7 GiB free. The engine's private memory was 63.77 GB (working set 43.1 GB).
- `--resident-budget-gib 36` holds 36 GiB of experts in RAM. Add the 262K int8 KV streamed to RAM, engine buffers,
  the 8 GiB cache, the server, the OS, and the apps.
- A 200K snapshot is about **4–5 GB**. Observed snapshots in the same log: 8,434 tokens → 460.7 MiB,
  15,719 → 679.4 MiB, 16,886 → 583.8 MiB, 26,815 → 841.1 MiB, 27,018 → 1,002.9 MiB, 75,659 → 1,891.4 MiB.
  That is roughly **25–56 KB per token**, so a 200K conversation is ~5 GB.
- The 8 GiB budget can hold about one such conversation, not two, and cannot hold it at all while the host is out of
  RAM.
- This matches prior evidence: `docs/DETAILS.md` records a 64 GB PC with `--resident-budget-gib 40`, an 8 GiB cache,
  and a 256K context stalling while parking; `docs/research/conversation_cache_oneshot.md` §11 (on
  `local/integration`) records the same host with the 36 GiB budget leaving ~261 MiB available and refusing every
  park, and uses 20 GiB in its measurement copy.

### 1.4 The product trade-off

The current workaround is to use one conversation at a time. That is fast but defeats the point of firing off
multiple agents and coming back later. Lowering `--resident-budget-gib` frees RAM for the cache but costs decode
speed. **The disk tier removes that trade-off**: the 36 GiB decode setting stays, and the 200K conversation lives on
NVMe instead of competing with the experts for RAM.

---

## 2. Goals, non-goals, acceptance criteria

### 2.1 Goals

1. An opt-in disk tier for parked conversations, configurable by directory and size.
2. A 200K-token conversation that alternates with smaller requests reuses its prefix on every turn.
3. Restore time of seconds, not minutes, on a normal NVMe drive; measured, not assumed.
4. Bounded host RAM and VRAM during save and restore. A 5 GB snapshot must not require a 5 GB host allocation or a
   5 GB VRAM staging allocation.
5. No partial decode on a missing, corrupt, or foreign snapshot. Fail closed and fall back to a full prompt read.
6. The default path (disk tier off) remains byte-identical to the current release.
7. The Monitor shows the disk tier's hits, misses, refusals, bytes, and last restore time.

### 2.2 Non-goals for v1

- DirectStorage or GPUDirect Storage. That is Phase 3.
- Concurrent/batched requests (`--parallel`, `--batch`). The engine serves one request at a time.
- Layer splits and peer devices. The disk tier is single-GPU first; stage images can come later.
- Encryption at rest. Document the plaintext cache and keep it private.
- A new chat export format. The file holds engine state and token identity, not a user-facing transcript.
- Making the cache on by default. It stays opt-in until measured.

### 2.3 Acceptance criteria

| Area | Criterion |
|---|---|
| Correctness | A restored 200K conversation produces the same next tokens (greedy) and the same `STATE_HASH` as the live state. |
| Reuse | Every follow-up turn of the main conversation reuses ≥ 99% of its prompt when the disk tier is on. |
| TTFT | On the reference host, a 200K main turn returns in ≤ 10 s (restore + new tail), against ~250–280 s cold. |
| Host RAM | Restore peak increase ≤ 512 MiB over the normal engine footprint; save peak ≤ 512 MiB. |
| VRAM | Staging increase ≤ 256 MiB over the normal engine footprint. |
| Disk throughput | Report MB/s and ms for save and restore; target ≥ 1 GB/s on NVMe, but report what is measured. |
| Safety | Corrupt, truncated, foreign, or missing files fall back to a full prompt read; no decode from partial state. |
| Disk full | A full or read-only cache directory logs a refusal and leaves the RAM cache working; no crash. |
| Default | `--conversation-cache-disk-mib 0` (default) is byte-identical to the current release; the comparison tools pass. |
| Platform | Windows CUDA first; build and test HIP; SYCL/Metal only if the session-file code supports them. |

### 2.4 Reference workload

Use the measured case above: Qwen3.8-Flash-Next UD-IQ4_XS, 262K context, int8 KV with `--kv-resident 32768`,
36 GiB resident experts, one ~200K main conversation and 8–27K sub-agent conversations, alternating. Any other
hardware is welcome, but report it separately.

---

## 3. Current architecture and where to attach

### 3.1 The RAM cache

`include/strata/core/conversation_cache.hpp`:

- `SavedConversation` (~line 158): `geometry`, `layer_lo/hi`, `live`, `checkpoints`, `kv`, `cvec`,
  `stage_images`. This is the object to persist.
- `ConversationCache` (~line 201): `budget_`, `slots_`, `entries_`, `serials_`, retained K/V.
- `best()` (~line 250): exact token-prefix and image match; returns a `Match`.
- `take()` (~line 268): removes an entry and moves it into the session.
- `borrows()` (~line 283): a new conversation starting at a checkpoint inside a parked one.
- `make_room()` (~line 315): evicts before a park.
- `evict_oldest()` (~line 341): LRU by insertion order, pinned entries protected.
- `put()` (~line 460), `drop_superseded()`, `clear()`, `slots()`.
- `retain()` (~line 231): keeps restored K/V buffers so a re-park writes only the delta.

The cache is host RAM only. There is no disk field in `ConversationCache`.

### 3.2 The snapshot and session-file machinery

`include/strata/core/conversation_snapshot.hpp`:

- `conversation_snapshot_bytes()` / `capture_bytes()`: estimates for admission.
- `conversation_snapshot_save()`: captures a `SavedConversation` from the live session.
- `conversation_snapshot_sources()` (~line 104): **the key save hook** — produces a `SavedConversation` *metadata*
  plus one `SessionKvSource` per layer, without a full host copy. The caller synchronizes the device and must not run
  the session until the file is written.
- `conversation_snapshot_restore()` (~line 112): restores a full `SavedConversation`.
- `conversation_snapshot_restore_prefix()` (~line 118): restores only the K/V prefix for a checkpoint borrow.
- `conversation_kv_source()` (~line 47): the per-layer streamed source used by save.
- `conversation_session_read_limits()`: bounded read preflight.

`include/strata/core/conversation_file.hpp`:

- `SessionFileIdentity` (~line 27) and `SessionConfig` (~line 73): engine/backend/KV/context/rope/cvec/adapter
  fingerprint. Reuse this; do not invent a weaker key.
- `SessionWriteOptions` (~line 152): `min_free_bytes` preflight, `durable`, progress/phase callbacks.
- `session_file_write()` (~line 170/186): temp file, flush, atomic rename, mode 0600 on POSIX, free-space preflight.
- `SessionKvSource` (~line 178): per-layer `read(part, offset, dst, n)`.
- `SessionReadLimits` (~line 208) and `session_file_read()` (~line 234): full validation and bounded parse.
- The file saves the running state, the **deepest** checkpoint, every K/V layer, and the draft K/V.

`docs/DETAILS.md` §“Conversation cache” and §“Session files (disk)” are the public descriptions. The session-file
feature (`--slot-save-path`) is **manual, single-slot, and has different semantics**. Reuse its format and helpers;
do not make it the disk cache.

### 3.3 The parking and restore flow

`src/program/generate.cpp`:

- Option field `conversation_cache_mib` (~line 732), parser (~line 2646), auto-sizing (~line 9399), cache
  construction (~line 9418).
- `park_current_body(held)` (~line 9496): the whole parking policy. It checks `live_parkable`, `conversation_cache-
  min-tokens`, `drop_superseded`, computes the estimate, calls `make_room`, does the physical-RAM admission,
  captures the snapshot, and calls `put`.
- The restore side (`conversations.take()` ~line 12729, `conversation_snapshot_restore()` ~line 12900,
  `restore_prefix()` ~line 12874, `retain()` ~line 12949).
- The engine's INFO line (~line 11117) reports `conversation_cache_mib` and `conversation_cache_slots`.

`serve/server.py`:

- `ConvCacheLog` (~line 757) parses `parked`/`skipped`/`restored` lines. It does **not** parse
  `skip parking (physical RAM admission...)`, which is why the Monitor cannot explain refusals.
- `conversation_cache_view()` (~line 827) builds the `/metrics` card.
- Metrics are attached at ~line 4647.

### 3.4 Existing test tools to reuse

- `tools/parking_test.py` — live vs parked token parity.
- `tools/conversation_cache_parity.py`, `isolation.py`, `growth.py`, `disabled.py`.
- `tools/conversation_restore_speed.py` — restore-speed template.
- `tools/conversation_cache_http_smoke.py` — HTTP smoke and INFO fields.
- `tools/harness_reuse_probe.py` — MAIN/SUB alternating probe; the exact pattern that fails today.
- `tools/test_conversation_cache_disabled.py` — default-off comparison.

The last two already mention future disk flags (`--conversation-cache-spill-dir`, `--conversation-cache-disk-mib`,
`--conversation-cache-disk-only`, and `--conversation-cache-disk` in one test). Keep the parser, docs, and tools in
sync; decide whether `--conversation-cache-disk` is an alias or update the tool.

---

## 4. Proposed design

### 4.1 Two tiers

```
request
  -> L1: host-RAM ConversationCache (fast, small: 8 GiB, 4 slots)
  -> L2: disk ConversationDiskCache (large: 32 GiB, LRU)
  -> miss: read the prompt from scratch
```

- L1 stays exactly as it is today.
- L2 is opt-in and independent. `--conversation-cache-disk-only` disables L1 parking and uses L2 for everything above
  the minimum.
- Default policy: try L1; if the park is refused or the entry is evicted, write it to L2. On restore, prefer the
  longest match; prefer L1 for equal prefixes (faster).
- Phase 1 can be simpler: L2 is used when L1 refuses a park. Demoting L1 entries to L2 on eviction is a Phase 2
  optimization (it needs a callback from `evict_oldest`).

### 4.2 On-disk layout

Phase 1 (whole-file snapshots):

```
<dir>/
  manifests/
    <id>.json or .bin        # small index/manifest: identity, tokens, checkpoints, file name, bytes, timestamps
  conversations/
    <id>.snap                # the versioned session file (session_file_write/read format)
  tmp/                       # short-lived .tmp files; removed on startup
```

Phase 2 (content-addressed pages):

```
<dir>/
  manifests/<id>.manifest    # header, token prefix, checkpoint metadata, page hash list, checksum
  pool/pages.bin             # 4 KiB-aligned KV pages, append-only
  pool/pages.idx             # page hash -> offset/length
  pool/compact.tmp           # compaction target, renamed into place
```

The manifest is the commit point. A payload without a valid manifest is garbage; a manifest with missing/corrupt
pages is refused. Rebuild the in-RAM index by scanning manifests at startup and deleting `*.tmp`.

### 4.3 `DiskEntry` and the in-RAM index

Keep the matching metadata in RAM, the heavy K/V on disk.

```cpp
struct DiskEntry {
    uint64_t serial = 0;
    std::vector<int32_t> ids;                 // live token prefix, for exact matching
    std::vector<ConversationImageKey> imgs;
    std::vector<int32_t> checkpoint_ids;      // or per-checkpoint metadata
    bool cvec = true;
    // ConversationLora lora;                 // when merged with feat/lora-decoding
    uint64_t bytes = 0;
    uint64_t last_used = 0;                   // monotonic clock
    bool pinned = false;
    std::string file;                         // manifest/payload path
};
```

A 200K token prefix is ~800 KB per entry; a handful of entries is a few MB, acceptable. If that grows, store a
prefix hash + length in RAM and read the manifest to verify; do not optimize before measuring.

### 4.4 Matching

Reuse `conversation_prefix()` and the `best()`/`borrows()` rules:

- Exact token prefix and image keys, `cvec`, and (after LoRA) adapter state must match.
- Prefer the longest prefix; for equal prefixes prefer L1.
- A match on the entry's live state is a continuation; a match on an interior checkpoint is a borrow (restore the
  prefix, keep the entry). The disk tier must preserve the `pinned` checkpoint rule.
- A `park=0` request must **not** be written to disk. The decision endpoint uses this; without it, one-shot decisions
  would fill the disk. Reuse `conversation_park_allowed()`.

### 4.5 Save path

1. In `park_current_body`, after the existing checks and after L1 `put` fails (or when `disk-only`), call the disk
   store.
2. Build the metadata and streamed K/V sources:
   `conversation_snapshot_sources(meta, sources, view, ss, g, draft, err)`.
   This avoids a full host copy; the writer reads each layer through `SessionKvSource`.
3. Compute the cache key from the model/config identity (`SessionFileIdentity`) plus the token prefix and image keys.
4. `session_file_write(path, meta, sources, identity, bytes, err, options)`:
   - `min_free_bytes` = the payload size plus a free-space floor;
   - `durable = true` (temp + flush + rename);
   - progress/phase callbacks so the engine watchdog and the server see “parking to disk”.
5. The caller has synchronized the device and must not run the session until the write finishes. This is a
   **synchronous** pause in v1; measure it. For a 5 GB file at 5 GB/s it is ~1 s, which is acceptable against a
   250 s prefill. A background writer is Phase 2.
6. On success, add the `DiskEntry` to the index and log:
   `conversation cache: spilled N tokens to disk in X ms; file=Y MiB tier=disk entries=...`
7. On failure (no space, permission, allocation), leave the live state unchanged, remove the temp file, log the
   reason, and continue.

Do not use `session_file_write` with a captured `SavedConversation` for a 5 GB image if that would first allocate
5 GB on the host. Use the `SessionKvSource` overload.

### 4.6 Restore path

**The critical rule: never call `session_file_read` into a full 5 GB host image for a large snapshot.** The existing
reader builds a full `SavedConversation`; `ConversationBuffer` owns its segments and cannot map an external file.

Phase 1 needs a **layer-at-a-time restore**:

1. Read and validate the header/identity/checksum.
2. Read the small running state and checkpoint metadata into the session.
3. For each K/V layer in file order (main layers, then the draft):
   - read that layer's parts into a temporary `ConversationKv` (one layer is tens of MB, not gigabytes);
   - call the existing `conversation_kv_restore()` for that layer;
   - release the temporary layer.
4. Synchronize, verify the state hashes in a debug build, and continue the prompt from the restored prefix.

Add a new API in `conversation_snapshot.hpp`, e.g.:

```cpp
// Restores a session file layer-by-layer, with a bounded host buffer.
// The caller has validated the identity and synchronized before/after as for the whole restore.
ConversationRestore conversation_snapshot_restore_stream(
    SessionFileReader& reader, SessionState& session, const ModelGeometry& g, const QsaState& draft,
    size_t staging_bytes, std::string& error);
```

The exact shape is the implementer's choice. The contract is: peak host RAM is `staging_bytes` plus one layer, not
the whole conversation; on any failure, no partial state is used.

If a layer is still too large for the staging budget, split the layer by part (`k`, `v`, `k_scale`, `v_scale`,
`pooled`) or by video-memory-sized chunks. Do not fall back to a full host image on the reference host.

### 4.7 Eviction, limits, and compaction

- **L2 budget:** `--conversation-cache-disk-mib` (default 0 = off; example 32768).
- **L2 minimum:** `--conversation-cache-disk-min-tokens` so small side requests do not churn the disk.
- **Park gain:** do not rewrite a conversation that advanced fewer than N tokens since its last disk copy. Reuse or
  mirror `--conversation-cache-park-gain`.
- **Free space:** `--conversation-cache-disk-min-free-mib` (default 4096, like the session-file floor).
- **Eviction:** LRU by `last_used`; pinned entries are protected; if only pinned entries remain, refuse the save.
  Prefer evicting smaller/shorter entries before the expensive 200K one if the measurement supports it. Do not evict
  a large entry for a save that then fails its free-space preflight.
- **Compaction (Phase 2):** append-only pages leave dead space. Trigger compaction on dead pages/bytes/fragmentation
  thresholds (borrow ninfer's ≥256 dead pages, ≥2 GiB dead, ≥25% fragmentation) and use a low-watermark, not every
  turn. Write to a temp pool, flush, rename, and rebuild the index.
- **Startup:** scan manifests, drop `*.tmp`, drop entries whose payload is missing or whose identity no longer
  matches, and rebuild the index.

### 4.8 Admission and failure handling

- Save admission: free disk space + metadata RAM + (v1) the synchronous pause. If it fails, leave L1 and the live
  state unchanged.
- Restore admission: identity, file size, checksum, `SessionReadLimits` (max tokens/checkpoints/layers/part sizes),
  staging RAM. If it fails, delete/quarantine the entry and read the prompt from scratch.
- A failed restore is not a fatal engine error. It is a cache miss. Log the reason and continue.
- Do not decode from a partially restored session. `conversation_snapshot_restore` already has this contract; the
  streaming variant must preserve it.
- A failed save must not evict L1 or L2 entries. This is the current bug pattern: the RAM admission loop evicts
  parked small conversations before it knows the big park will succeed. The disk path must preflight first.

### 4.9 Identity and invalidation

Reuse `SessionFileIdentity` / `SessionConfig` exactly:

- model fingerprint (paths and sampled bytes), engine version, backend, KV format, `max_context`, `kv_resident`,
  `mtp_window`, `STRATA_KV_ROT`, rope, control vector, and (after LoRA) the adapter state and arithmetic switches.
- Store the identity in the manifest; refuse a mismatch before reading the payload.
- Invalidate on engine upgrade, model change, KV/context/rope change. Do not try to migrate engine-state bytes.
- The default-off path must remain byte-identical; the identity is only for the disk tier.

### 4.10 Concurrency and the watchdog

- v1: one request at a time; the disk save is synchronous. Use `SessionPhase`/`session_phase_limit_s` so the engine
  and server can call a stuck write stuck, named “parking to disk”.
- Restore is on the critical path; stream it and report progress.
- Phase 2: a background writer can take ownership of a stable copy. The hard part is not the thread; it is keeping
  the K/V stable while the next request runs. Options:
  - stream to disk synchronously (v1);
  - copy one layer at a time into a bounded queue and write in the background while the next request waits for the
    copies it needs;
  - use the retained K/V delta so only the changed pages are copied.
  Do not let a background write read K/V buffers that the next request is overwriting.
- Use a mutex around the index and pool. The engine's serve loop already serializes requests.

### 4.11 Cross-platform I/O

- Phase 1: use the existing `conversation_file.cpp` buffered 16 MiB writer/reader. It is portable and already handles
  Windows wide paths, POSIX modes, temp+rename, and free space.
- Measure buffered I/O first. If the restore is disk-bound, try `FILE_FLAG_NO_BUFFERING`/`O_DIRECT` and large aligned
  reads. Do not add a dependency before measuring.
- Phase 3 (optional): DirectStorage on Windows/CUDA and GPUDirect Storage on Linux/CUDA; fallback everywhere else.
- The disk store is host code; the K/V source/sink adapters are backend-specific. Build CUDA and HIP at minimum.
  Confirm the session-file code's SYCL/Metal status before claiming support.

### 4.12 Security and privacy

- The cache holds prompt state and token identity. Treat it as sensitive.
- Directory created 0700, files 0600 on POSIX; on Windows, inherit a private per-user folder ACL. Document it.
- Reject path traversal, symlinks, reparse points, Windows device names, control characters, leading/trailing dots or
  spaces. Reuse the session-file path rules.
- The cache directory must not be a shared temp folder by default. Default to the user's local app data / Strata data
  directory, or require an explicit dir.
- No encryption in v1. Document that the files are plaintext; a disk-level encryption option is future work.
- The uninstaller should not leave the cache behind silently; either remove it as an app-owned folder or document it.

### 4.13 Configuration

Proposed engine flags (names chosen to match the existing test tooling):

| Flag | Meaning | Default |
|---|---|---|
| `--conversation-cache-disk-mib N` | Disk budget; `0` = off | `0` |
| `--conversation-cache-spill-dir DIR` | Cache directory | none; required when the budget > 0 |
| `--conversation-cache-disk-only` | Skip L1 parking; disk only | off |
| `--conversation-cache-disk-min-tokens N` | Do not write shorter conversations | `0` or the L1 value |
| `--conversation-cache-disk-min-free-mib N` | Free-space floor | `4096` |
| `--conversation-cache-disk-staging-mib N` | Restore/save staging buffer | `256` |
| `--conversation-cache-disk-park-gain N` | Skip a rewrite until N new tokens | reuse L1 value |

Notes:

- `--conversation-cache-disk` appears in `tools/test_conversation_cache_disabled.py`. Either implement it as an alias
  for `--conversation-cache-spill-dir` or update the tool in the same PR. Pick one and keep it consistent.
- The engine INFO line must report `conversation_disk_mib`, `conversation_disk_dir`, and `conversation_disk_slots` so
  the server and the disabled-cache comparison can see it. The test tool already expects `conversation_disk_mib`.
- `strata launch`/the app can add a friendly option later; the first PR is engine-only.

### 4.14 Metrics and the Monitor

Engine log lines (new, parseable):

```
conversation cache: spilled N tokens to disk in X ms; file=Y MiB tier=disk entries=K bytes=B
conversation cache: restored N tokens from disk in X ms (Z MiB/s); tier=disk
conversation cache: skip disk save (free space; need ... available ...)
conversation cache: skip disk save (identity; ...)
conversation cache: disk evicted K entries (LRU) ...
conversation cache: disk restore failed (...); reading the prompt
```

`serve/server.py`:

- Extend `ConvCacheLog` to parse these and the existing `skip parking (physical RAM admission...)` reasons.
- Extend `conversation_cache_view()` with:
  `disk_enabled`, `disk_budget_mib`, `disk_bytes`, `disk_entries`, `disk_saves`, `disk_restores`,
  `disk_evictions`, `disk_refusals`, `disk_errors`, `disk_last_save_ms`, `disk_last_restore_ms`,
  `disk_last_restore_mib_s`, `last_tier`.
- The Monitor card should show L1 and L2 separately: slots, bytes, hit rate, last restore, refused saves. Today the
  card shows 1/4 slots and 0.7/8 GB with no explanation; that is the visibility gap to close.

### 4.15 Interaction with existing features

- **`park=0` / decision endpoint:** never write a `park=0` conversation to disk. Reuse `conversation_park_allowed()`.
- **`--conversation-cache-min-tokens` / `park-gain`:** apply to both tiers, or give the disk tier its own values.
- **Pinned prefix (`strata_prefix`):** protect pinned entries from disk eviction; a pinned prefix may stay in L1
  even when the rest goes to disk.
- **Borrowing:** a new sub-agent that starts at a checkpoint inside a disk entry should restore the prefix and keep
  the entry, exactly as `borrows()` does in RAM. This is a big win for sibling sub-agents.
- **Layer split / stage images:** `SavedConversation::stage_images` exists. v1 may refuse `--layer-split` with a
  clear startup message or fall back to L1-only; add disk support only after the single-GPU path is measured.
- **Batch/parallel/peer:** v1 refuses or falls back; the engine is single-request.
- **MTP draft KV:** included in `SavedConversation::kv` (main layers + draft). The streamed save/restore must include
  it.
- **Vision:** the snapshot holds image keys/hashes, not pixels; the token IDs include the image tokens. Include the
  image keys in the manifest and matching.
- **Control vector:** `cvec` is part of the key; a disk entry saved with one control vector must not match another.
- **LoRA (on `local/integration`):** `feat/lora-decoding` adds `ConversationLora` to `SavedConversation` and the cache
  key. When the branches meet, the disk manifest must include the adapter state; otherwise adapted and base state
  could be mixed. If the disk lands first, the LoRA PR must extend the manifest; if LoRA lands first, rebase the disk
  branch and add the field before testing.
- **Session files (`--slot-save-path`):** share the `SavedConversation` format and helpers, not the lifecycle. The
  session file is manual and single-slot; the disk cache is automatic and multi-conversation.

### 4.16 What to borrow from ninfer, and what to do differently

ninfer-4090 (`https://github.com/UDPSendToFailed/ninfer-4090`, branch `feat/rtx-4090-sm89-native`) is a useful
blueprint:

- `src/core/disk_state_cache.h/.cpp`: content-addressed 64-token KV pages in a shared pool, per-conversation
  manifests, CoW deltas, a background writer, LRU pruning, and compaction guards.
- `src/core/direct_storage_engine.h/.cpp`: DirectStorage 1.3 with a D3D12 shared buffer imported into CUDA, a shared
  fence imported as a CUDA external semaphore, and 32 MiB chunked reads.
- `RELEASE_NOTES_1.1.0.md` §1 and `RELEASE_NOTES_1.2.0.md` §5: the design and the bounded 32-page staging lesson.
- Their published claim: 77,615 prompt tokens / 1.51 GiB restored in 150 ms (10.1 GB/s), cold TTFT 52.6 s → 1.86 s.
  Treat this as their measurement on their hardware, not ours.
- License: Apache-2.0; Strata is MIT. Compatible if the Apache license/NOTICE is preserved for copied files, but a
  clean reimplementation of the architecture is likely cleaner because Strata's state layout differs.

Borrow the ideas: page pool, CoW, manifests, LRU/compaction, bounded staging, atomic publish, background writer.
Do not copy the 64-token page assumption: Strata's K/V is per-layer with `page_size`, `k/v/k_scale/v_scale/pooled`,
recurrent state, checkpoints, MTP draft K/V, and stage images. Use the engine's native page geometry.

---

## 5. Roadmap and phases

### Phase 0 — measurement spike (1–3 days)

**Goal:** prove the disk is fast enough and the existing save/restore path is a viable base, before building the cache.

- Build the engine from `feat/disk-conversation-cache` with CUDA and the conversation tests:
  `cmake -S . -B build-cache-cuda -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON`
  then `cmake --build build-cache-cuda --target strata conversation_file_test conversation_snapshot_test`.
  (Use the repo's normal configure for the card; add the conversation-test option.)
- Create a real ~200K-token conversation on the reference host.
- Save it with the existing `--slot-save-path` / `/slots/0?action=save` path (or a small harness around
  `conversation_snapshot_sources` + `session_file_write`), then restore it.
- Measure: file size, save ms, restore ms, MB/s, peak host RAM, peak VRAM, token parity, `STATE_HASH` parity.
- Repeat on the actual NVMe and, if available, a SATA SSD, to see the range.
- Answer: is the portable layer-at-a-time restore fast enough? What is the real per-token snapshot size? What is the
  save pause?

**Exit:** a measured report with the numbers and a go/no-go for Phase 1. No cache code yet.

### Phase 1 — portable disk tier (the useful release)

**Goal:** a 200K conversation alternating with sub-agents reuses its prefix, with bounded RAM/VRAM.

Scope:

- `ConversationDiskCache`/`ConversationDiskStore` with a manifest and one payload file per conversation.
- Save via `conversation_snapshot_sources` + `session_file_write` (no full host copy).
- Restore layer-by-layer (new streaming reader; bounded staging), not a full `session_file_read`.
- LRU, size cap, free-space floor, startup scan, atomic publish, identity/checksum validation.
- Integration with `park_current_body` and the restore path; honor `park=0`, min-tokens, park-gain, pinned prefixes,
  borrowing.
- Opt-in flags; default off; engine INFO line; server metrics and Monitor.
- Unit/integration/parity/performance tests.

Deliverables:

- `include/strata/core/conversation_disk.hpp`, `src/core/conversation_disk.cpp`,
  `src/core/conversation_disk_test.cpp`.
- A streaming restore reader, likely in `conversation_file.*`/`conversation_snapshot.*`.
- Changes in `generate.cpp`, `conversation_cache.hpp`, `serve/server.py`, `docs/DETAILS.md`, and the relevant tools.
- A measured report: 200K restore, reuse %, host/VRAM high-water, MB/s.

**Exit:** on the reference host, a 200K main turn is ≤ 10 s (restore + new tail), every follow-up reuses ≥ 99%, and
the default-off path is byte-identical. This is the milestone that solves the user's problem.

### Phase 2 — CoW page pool and write reduction

**Goal:** cut SSD writes and save time when a conversation advances a little each turn.

- Split the K/V snapshot into content-addressed pages at the engine's native page size; hash each page; store each
  page once in a shared pool.
- Per-turn manifest lists page hashes; write only missing/delta pages.
- Store the recurrent state and checkpoint metadata in the manifest.
- Add compaction guards and a low-watermark; rebuild the index on startup.
- Optionally demote L1 entries to L2 on RAM eviction via a callback from `evict_oldest`.
- Measure write amplification per turn, save time, SSD bytes written, and restore time.

**Exit:** a 2K-token advance writes ≤ 200 MB (target; measure), restore is unchanged or faster, and a 20-turn run
does not rewrite the whole 5 GB each turn. If CoW is too complex for the first PR, ship Phase 1 with `park-gain` and
make Phase 2 the next PR.

### Phase 3 — optional fast paths

**Goal:** reduce restore latency on supported hardware, with the portable path as the fallback.

- **Windows + CUDA:** DirectStorage 1.3. D3D12 default-heap buffer, shared NT handle, CUDA external memory import,
  shared D3D12 fence imported as a CUDA external semaphore, 32 MiB chunks, bounded staging (ninfer's 32-page lesson).
  Scatter the pages into Strata's K/V layout on the GPU.
- **Linux + CUDA:** GPUDirect Storage (`cuFile`) if the environment supports it; otherwise portable.
- **Windows AMD / Linux AMD / macOS:** portable; no DirectStorage/GDS dependency.
- Benchmark each fast path against the portable path on the same file. Do not assume 10 GB/s; report the drive and
  the measured MB/s. Keep the fast path opt-in or automatic only when the capability check passes.

**Exit:** a measured comparison; the portable path remains the default and the fallback.

### Phase 4 — hardening, docs, PR

- Failure injection: disk full, read-only dir, deleted file mid-restore, truncated file, corrupt page, identity
  mismatch after model/config change, engine restart with stale manifests.
- Cross-platform builds: CUDA, HIP; SYCL/Metal if supported.
- Docs: `docs/DETAILS.md` (public settings and behavior), a short `docs/` page for users, the Monitor card.
- PR: one problem per PR; title with `[Engine]` (and `[GUI]`/`[Other]` for the Monitor); lead with the problem and the
  measured result; state what passed, failed, and was not run; keep the default path byte-identical.
- Do not submit measurements you did not produce. Mark static analysis as unconfirmed.

---

## 6. Detailed implementation steps

### Step 1 — options and configuration

- Add fields in `generate.cpp` next to `conversation_cache_mib` (~line 732):
  `conversation_disk_mib`, `conversation_disk_dir`, `conversation_disk_only`, `conversation_disk_min_tokens`,
  `conversation_disk_min_free_mib`, `conversation_disk_staging_mib`.
- Parse in the existing `--conversation-cache-*` block (~line 2646); validate ranges and require a dir when the budget
  is > 0.
- Refuse or warn at startup for unsupported configurations (`--parallel`, `--batch`, `--layer-split`, `--peer-device`,
  `--prompt-cache 0`) with a clear message. Decide whether to refuse or fall back to L1-only; document it.
- Include `conversation_disk_mib`/`conversation_disk_dir` in the INFO line (~line 11117) and in the engine's `info`
  dict for the server.

### Step 2 — manifest and index

- Define a versioned manifest format (little-endian, fixed-width, checksum):
  magic, version, identity hash, token count, checkpoint metadata (ids, image keys, state sizes), payload file name,
  payload bytes, page count (Phase 2), pinned flag, created/last-used timestamps, checksum.
- Read/write helpers with bounds checks; reuse `SessionHasher`/`SessionIdentityBuilder` if it fits.
- Startup: scan `manifests/`, drop `*.tmp`, validate identity/checksum, rebuild `DiskEntry`s.
- Path safety: reuse the session-file rules; no user-controlled file names.

### Step 3 — disk store (Phase 1)

- `save_current(...)`: build the key, call `conversation_snapshot_sources`, `session_file_write` to a temp name,
  publish, add the index entry.
- `restore(...)`: validate, stream layer-by-layer, restore state, return a result. On failure, remove the entry and
  return a miss.
- `evict_until_fits(...)`: LRU, pinned protection, free-space floor; never evict for a save that cannot complete.
- `clear()`, `bytes()`, `entries()`, `slots()`, `stats()`.

### Step 4 — save integration

- In `park_current_body`, after the L1 path decides it cannot park (or when `disk-only`), call the disk store.
- Keep the existing `live_parkable`, `wants`, `drop_superseded`, `park_gain` checks. A `park=0` conversation is not
  written.
- Log the spill line and update the metrics.
- If the save is synchronous, the next request waits; measure. If the pause is too long, move to a background writer
  in Phase 2.

### Step 5 — restore integration

- In the restore path, after L1 `best()` returns no match (or a shorter match), ask the disk store.
- Prefer the longer prefix; for ties prefer L1.
- Validate identity/checksum, stream layer-by-layer, set `live`, `checks`, `cvec`, `took_parked`, and `retain` if
  applicable.
- On failure, delete/quarantine and read the prompt from scratch.
- Log the restore line with ms and MB/s.

### Step 6 — streaming restore

- Add a `SessionFileReader`-style API that parses the header/metadata and yields one K/V layer at a time.
- Restore each layer with the existing `conversation_kv_restore`; free the temporary layer before the next.
- Keep the peak host allocation to `staging_mib` + one layer.
- Tests: a 200K file with a 256 MiB staging budget; assert peak RSS/`ConversationBuffer` bytes and token parity.
- If the whole-layer buffer is still too large, split by part or chunk; do not load the whole image.

### Step 7 — CoW pages (Phase 2)

- Page the K/V at the engine's native `page_size`; one hash per (layer, part, page).
- Pool: append-only, 4 KiB-aligned; `pages.idx` maps hash → offset/length.
- Manifest: live token ids, checkpoints, state, page hash list.
- Save: hash the pages, write missing ones, publish the manifest.
- Restore: resolve page hashes, read pages, scatter into the K/V layout (or into a layer buffer and use the existing
  restore).
- Compaction: dead-page/bytes/fragmentation thresholds; temp pool + rename; rebuild.
- Tests: same page reused across turns, delta save, compaction, torn manifest, orphan pages.

### Step 8 — metrics and Monitor

- Extend `ConvCacheLog` (serve/server.py ~line 757) to parse disk lines and the existing physical-RAM refusals.
- Extend `conversation_cache_view()` (~line 827) and `/metrics` (~line 4647).
- Update the Monitor card (`serve/web/app.js` `renderConvCache`, `apps/host` monitor dashboard) to show L1/L2 bytes,
  slots, saves, restores, evictions, refusals, last restore ms/MB/s.
- Add a "why" line: refused saves by reason (RAM, disk space, identity, min tokens).

### Step 9 — tests

See §7. Add the disk tests to CMake next to `STRATA_BUILD_CONVERSATION_TESTS` and the Python tools.

### Step 10 — docs and PR

- Update `docs/DETAILS.md` §Conversation cache and §Session files; add a short user-facing page if the flag is
  shipped.
- Update the tools that strip `--conversation-cache-*` flags (`tools/conversation_cache_disabled.py`,
  `tools/logit_bias_http_probe.py`, `tools/test_conversation_cache_disabled.py`) so the default-off comparison still
  works.
- PR description: problem, result, approach, validation (passed/failed/not run), limits, and the measured numbers
  with the hardware.

---

## 7. Test plan

### 7.1 Unit (C++)

| Test | Proves |
|---|---|
| `conversation_disk_test` | manifest round-trip, version refusal, checksum, identity mismatch, path safety, LRU, free-space refusal, startup scan |
| `conversation_file_test` | existing save/read round-trip, fault injection, atomic publish, free-space preflight |
| `conversation_snapshot_test` | capture/restore parity, `conversation_snapshot_sources` streaming, layer-at-a-time restore, peak bytes |
| `conversation_cache_test` | L1 behavior unchanged; disk entry matching/borrowing; `park=0`; pinned entries |

### 7.2 Integration and parity

- `tools/parking_test.py` with the disk tier on: live vs parked token parity, including a 200K conversation.
- `tools/conversation_cache_http_smoke.py`: INFO fields, `/metrics`, HTTP smoke.
- `tools/harness_reuse_probe.py`: MAIN/SUB alternation; every follow-up reuses > 50% (target ≥ 99% for the main).
- `tools/conversation_cache_parity.py`, `isolation.py`, `growth.py`, `disabled.py`: existing cache behavior and the
  default-off comparison.
- `tools/conversation_restore_speed.py`: restore speed with the disk tier.
- A new `tools/conversation_disk_cache_test.py`: build a 200K conversation, alternate, measure save/restore, assert
  parity and peak memory, write a JSON report.

### 7.3 Failure and safety

- Disk full / read-only directory / permission denied: save refused, L1 and the live state unchanged.
- Truncated file, corrupt page, wrong checksum, wrong identity, missing payload: restore refused, full prefill, entry
  removed/quarantined.
- File deleted between index scan and restore: miss, no crash.
- Engine restart with a stale manifest: identity/version check, prune.
- A failed big save must not evict L1/L2 entries (preflight first).
- `park=0` conversation never written to disk.

### 7.4 Performance and memory

- Reference host, 200K conversation, 2K-token advances, 10 alternating turns.
- Measure: file size, save ms, restore ms, MB/s, host RAM high-water, VRAM high-water, TTFT, reuse %, `STATE_HASH`
  parity.
- Run with `--conversation-cache-mib 8192` and with `0` (disk-only for the main) to show the trade.
- Compare `--resident-budget-gib 36` (the user's decode setting) and a lower setting; the disk tier should let the 36
  stay.
- Repeat on a SATA SSD if available; report the range. Do not generalize from one drive.

### 7.5 Build matrix

- Windows CUDA (reference), Linux CUDA, HIP. SYCL and Metal only if the session-file code supports them; state it.
- `STRATA_BUILD_CONVERSATION_TESTS=ON` and the normal test suite.
- `python -m pytest serve/test_server.py serve/test_systemone.py ...` for the server changes.
- The default path: `git diff` shows no behavior change when the disk flags are absent; the disabled-cache comparison
  passes.

---

## 8. Measurement plan and report format

Use the existing report style: date, commit, engine version, model, config (KV, context, resident budget, cache
settings), drive model and interface, file size, save/restore ms, MB/s, peak host/VRAM, TTFT, reuse %, parity.

A minimal report table:

| Run | Conversation | File | Save | Restore | MB/s | Host peak | VRAM peak | TTFT | Reuse | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| disk, 200K main | 208,225 tokens | ~5 GB | X ms | X ms | X | X MiB | X MiB | X s | X% | pass |
| RAM, 200K main (baseline) | 208,225 tokens | — | refused | — | — | — | — | 280 s | 0% | — |
| disk, 26K sub | 26,952 tokens | ~0.9 GB | X ms | X ms | X | X MiB | X MiB | X s | X% | pass |

Always say what was **not** run. Do not claim 10 GB/s or 150 ms from ninfer's hardware.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Restore needs a 5 GB host image | Layer-at-a-time reader; bounded staging; never `session_file_read` a large file whole |
| VRAM staging pressure | 256 MiB (or configurable) staging; stream; do not allocate the whole payload |
| SSD wear from full rewrites | Phase 2 CoW; Phase 1 `park-gain`; measure bytes/turn |
| Slow or busy disk | Fall back to full prefill; report the measured speed; do not block the engine forever |
| Corrupt/truncated files | Checksums, temp+rename, fail closed, prune at startup |
| Wrong model/config state | Reuse `SessionFileIdentity`; refuse mismatches |
| Failed big save evicts small entries | Preflight free space and admission; do not evict until the save can succeed |
| Background writer races the next request | v1 synchronous; Phase 2 stable copies/bounded queue |
| DirectStorage dependency | Portable path first; optional fast path with fallback; no Windows-only feature requirement |
| Security of prompt state on disk | Private dir, 0700/0600, path checks, no encryption in v1 (documented) |
| Scope creep into layer split/batch | v1 single-GPU; refuse or fall back; add later with tests |
| LoRA/decision merge conflicts | Include `lora` and `park=0` in the manifest/key; coordinate with `local/integration` |

---

## 10. PR plan

Keep the default path byte-identical and the feature opt-in.

1. **`[Engine] disk conversation cache: store, manifest, LRU, opt-in flags`**
   Phase 1 save + layer-at-a-time restore, single-GPU, tests, INFO/metrics. Minimum useful PR.
2. **`[Engine] disk conversation cache: streaming restore for large snapshots`**
   If not in PR 1, split it out to keep the review small. It is required for the 200K case.
3. **`[Engine] disk conversation cache: CoW page pool and compaction`**
   Phase 2; write amplification and SSD wear.
4. **`[Engine] disk conversation cache: Windows DirectStorage fast path`**
   Phase 3, optional; benchmark against the portable path.
5. **`[GUI]`/`[Other] Monitor: L1/L2 cache card and refused-save reasons`**
   The visibility fix; can land with PR 1 or separately.

Each PR: one problem, title with the area prefix, build every backend touched, tests, docs, and a measured result.
Do not make the cache on by default in the first PR.

---

## 11. Hand-off checklist

### First day

- [ ] Read `AGENTS.md`, `CONTRIBUTING.md`, `docs/DETAILS.md` §Conversation cache and §Session files.
- [ ] Read the current code: `conversation_cache.hpp`, `conversation_snapshot.hpp`, `conversation_file.hpp`,
      `park_current_body` and the restore path in `generate.cpp`.
- [ ] Build the baseline engine and run `conversation_cache_test`, `conversation_file_test`,
      `conversation_snapshot_test`.
- [ ] Reproduce the 200K diagnosis on the reference host if it is available; otherwise use the numbers in §1.
- [ ] Do the Phase 0 spike and write the measured report.

### Before writing the cache

- [ ] Decide the flag names and the `--conversation-cache-disk` alias/tool update.
- [ ] Decide the v1 scope: single-GPU only, no parallel/batch/layer split.
- [ ] Decide the streaming restore API shape (layer-at-a-time is the recommended minimum).
- [ ] Decide the default directory and the privacy/ACL story.
- [ ] Confirm the session-file code's backend support (CUDA/HIP/SYCL/Metal).

### Before the PR

- [ ] Default off is byte-identical; the disabled-cache comparison passes.
- [ ] A 200K conversation reuses ≥ 99% on every follow-up; TTFT ≤ 10 s on the reference host.
- [ ] Host/VRAM peaks are bounded and measured.
- [ ] Corrupt/missing/foreign files fall back safely.
- [ ] CUDA and HIP build; tests pass; docs and Monitor updated.
- [ ] No unmeasured claims; report passed/failed/not run separately.

### Do not

- Do not make DirectStorage a dependency for the feature.
- Do not load a 5 GB snapshot into a full host `SavedConversation` to restore it.
- Do not evict parked entries for a save that cannot succeed.
- Do not write `park=0` conversations to disk.
- Do not change the default path or the existing L1 behavior.
- Do not copy ninfer's code without preserving its Apache-2.0 license/NOTICE; a clean reimplementation is preferred.
- Do not claim ninfer's 10.1 GB/s or 150 ms as ours.

---

## Appendix A — ninfer reference

- Repository: `https://github.com/UDPSendToFailed/ninfer-4090`, branch `feat/rtx-4090-sm89-native` (Apache-2.0).
- `src/core/disk_state_cache.h/.cpp`: content-addressed 64-token KV pages, shared pool, per-conversation manifests,
  CoW deltas, background writer, LRU pruning, compaction guards.
- `src/core/direct_storage_engine.h/.cpp`: D3D12 default-heap staging buffer, shared NT handle, CUDA external memory
  import, shared D3D12 fence as a CUDA external semaphore, DirectStorage queue, 32 MiB chunks.
- `RELEASE_NOTES_1.1.0.md` §1: design and measured table (77,615 tokens / 1.51 GiB / 150 ms / 10.1 GB/s; cold TTFT
  52.6 s → 1.86 s).
- `RELEASE_NOTES_1.2.0.md` §5: bounded 32-page staging to avoid OOM on a 24 GB card; MTP KV stride fix; chunked save.
- Difference from Strata: ninfer's K/V is uniform 64-token text pages; Strata has per-layer K/V with
  `page_size`/`parts`, recurrent state, checkpoints, MTP draft K/V, vision image keys, control vectors, and stage
  images. Use Strata's native page geometry.

## Appendix B — code map (approximate, `4b9e05a2`)

| File | What to read |
|---|---|
| `include/strata/core/conversation_cache.hpp` | `SavedConversation` (~158), `ConversationCache` (~201), `best` (~250), `take` (~268), `borrows` (~283), `make_room` (~315), `evict_oldest` (~341), `put` (~460), `retain` (~231) |
| `include/strata/core/conversation_snapshot.hpp` | `conversation_kv_source` (~47), `conversation_snapshot_save` (~97), `conversation_snapshot_sources` (~104), `conversation_snapshot_restore` (~112), `restore_prefix` (~118) |
| `include/strata/core/conversation_file.hpp` | `SessionFileIdentity` (~27), `SessionConfig` (~73), `SessionWriteOptions` (~152), `session_file_write` (~170/186), `SessionKvSource` (~178), `SessionReadLimits` (~208), `session_file_read` (~234) |
| `src/program/generate.cpp` | option field (~732), parser (~2646), auto size (~9399), cache construction (~9418), `park_current_body` (~9496), INFO (~11117), restore/take (~12729/12900), `retain` (~12949) |
| `serve/server.py` | `ConvCacheLog` (~757), `conversation_cache_view` (~827), metrics (~4647) |
| `docs/DETAILS.md` | Conversation cache (~608), Session files (~643) |
| `CMakeLists.txt` | `STRATA_BUILD_CONVERSATION_TESTS` (~124), conversation test targets |
| `tools/` | `parking_test.py`, `conversation_cache_parity.py`, `isolation.py`, `growth.py`, `disabled.py`, `conversation_restore_speed.py`, `conversation_cache_http_smoke.py`, `harness_reuse_probe.py`, `test_conversation_cache_disabled.py` |

## Appendix C — glossary

- **Park**: copy a conversation's running state, checkpoints, and K/V into host RAM so another request can use the
  engine.
- **Snapshot**: the `SavedConversation` image of a parked conversation.
- **L1 / L2**: host-RAM cache / disk cache.
- **Prefill**: reading the prompt tokens into the model; the expensive part when a cache misses.
- **TTFT**: time to first token, including the queue, prompt read, and first decode.
- **Physical-RAM admission**: the engine's check that `MemAvailable` can hold the snapshot plus the floor.
- **Floor**: `--conversation-cache-min-free-mib`, RAM the engine leaves for the OS.
- **CoW**: copy-on-write; write only the pages that changed since the previous snapshot.
- **Manifest**: the small per-conversation metadata that points at the payload/pages.
- **Pinned prefix**: a checkpoint a request asked the cache to keep for good (`strata_prefix`).
- **`park=0`**: a request that tells the engine not to park its conversation (used by one-shot decisions).
