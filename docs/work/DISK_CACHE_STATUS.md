# Disk conversation cache — implementation status

Date: 2026-10-28
Branch: `feat/disk-conversation-cache` (worktree `C:\tools\strata-disk`), based on `origin/main` `4b9e05a2`.
Plan: [DISK_CONVERSATION_CACHE_PLAN.md](DISK_CONVERSATION_CACHE_PLAN.md).

This is the resume point: what is implemented, what the reference-host run measured, and what remains. The Phase 1
measurement was made from commit `07f2b9fb` on the sm_89 build; the follow-up commits only change the staging default
and record the evidence. Phase 2 (the copy-on-write page pool) is implemented and unit-tested, but not measured on the
reference host.

## Phase 2: the page pool (copy-on-write, opt-in)

With `--conversation-cache-disk-cow` the disk tier stores K/V in one shared, append-only pool of content-addressed
64 KiB pages (`pool/pages.bin` + a rebuildable `pool/pages.idx`). Each conversation keeps a small manifest (tokens,
checkpoints, identity) and a small metadata head in the session-file format with no K/V layers. The manifest lists the
hash and length of each page, so:

- a turn that advanced a little writes only the pages whose content changed; unchanged pages are reused by hash;
- manifests never change when the pool is compacted, because a page is found by its hash, not by an offset;
- a pool page is referenced by every conversation that names it, and pages with no reference are reclaimed by a
  compaction (guards: at least 256 dead pages, at least 2 GiB dead, at least 25% of the pool);
- a missing or corrupt page refuses the restore before the session is touched; every page's hash is checked as it is
  read;
- restore still happens one layer at a time, and now one page at a time: host staging is the head's metadata, one
  layer and one page.

Implementation:

| File | What it holds |
| --- | --- |
| `include/strata/core/conversation_pages.hpp`, `src/core/conversation_pages.cpp` | The pool, the page writer, the layer-at-a-time page reader, the index and the compaction. |
| `include/strata/core/conversation_disk.hpp`, `src/core/conversation_disk.cpp` | Manifest v2 (paged), `save_paged`, `open_paged`, page refcounts, startup scan, compaction stats, LRU and the free-space preflight. |
| `include/strata/core/conversation_file.hpp` | `ConversationKvLayers`, the reader interface shared by a session file and the page pool. |
| `include/strata/core/conversation_snapshot.hpp`, `src/core/conversation_state.cpp` | `conversation_snapshot_restore_layers` (the generic restore) and `conversation_kv_image_sources` (demotion from a parked L1 entry). |
| `src/program/generate.cpp` | `--conversation-cache-disk-whole-file`, `--conversation-cache-disk-demote`, the compaction flags, the paged save and restore, the write/reuse log line, the demotion hook. |
| `include/strata/core/conversation_cache.hpp` | `set_demote`: an optional callback before `evict_oldest` drops an entry. |
| `serve/server.py`, `serve/web/app.js` | Write/reuse, pool and compaction fields in `/metrics` and the Monitor card. |

Flags added: `--conversation-cache-disk-cow` (opt in to the page pool), `--conversation-cache-disk-whole-file`
(explicit earlier format; the default), `--conversation-cache-disk-demote` (needs `-cow`, off by default),
`--conversation-cache-disk-compact-min-mib` (default 2048), `--conversation-cache-disk-compact-min-pages`
(default 256). The default path (the disk tier off) is unchanged: no pool, no pages, no new I/O. With the disk tier on
but `-cow` absent, the Phase 1 one-file save/restore is used.

### What was validated

- `conversation_disk_test` (CPU): 195 checks, including a paged save, layer-at-a-time restore, an unchanged turn
  writing zero page bytes, a one-byte change writing exactly one page, a restart, a missing-index rebuild, eviction of a
  superseded entry, a compaction, a restart after compaction and a corrupt page refused. Passed.
- `conversation_file_test`, `conversation_stream_test`, `conversation_cache_test` (CPU): passed.
- `conversation_validation_test` (CUDA, host-only checks) and `conversation_snapshot_test` (CUDA, 4139 checks):
  passed.
- The engine (`build-cuda`, Windows CUDA 13.4, sm_89) builds with the new files and the changed restore path.

### What was not run

- No reference-host run of the page pool: write amplification per turn, save time, SSD bytes written and restore
  time are **not measured**. The Phase 1 numbers below are for the one-file format.
- The server tests (`serve/test_monitor.py`, `test_conversation_cache_disabled`) were not run here: the Python
  environment has no `pytest`/`jinja2`. `serve/server.py` compiles.
- HIP, SYCL and Metal were not built (no AMD/Mac hardware here).
- Failure injection at the pool level (disk full mid-page, read-only pool, a page deleted between the scan and the
  read) is covered by the preflight and hash checks, not by an injected run.
- L1 demotion (`--conversation-cache-disk-demote`) is implemented and its sources are unit-tested only through the
  shared page writer; no engine run exercised it.

## Commits

| Commit | What it adds |
| --- | --- |
| `d74859bd` | Phase A: CPU-only `ConversationDiskStore` — manifest, in-RAM index, matching/borrowing, LRU, budget, free-space preflight, atomic publish, startup scan, path safety, unit tests. |
| `fe69eeee` | Phase B1: `SessionFileStream` — validates a session file and yields one K/V layer at a time, bounded host staging, payload hash checked in `finish()`. |
| `6e30702c` | Phase B2: `conversation_snapshot_restore_stream` — restores a streamed file layer by layer, full and prefix (borrow) modes, tested against the whole-image restore on CUDA. |
| `0543dc9c` | Phase C1: `--conversation-cache-disk-*` flags, startup identity/store, INFO fields, and the save path. The disk preflight runs before `make_room` can evict parked conversations. |
| `6924e4e1` | Phase C2: restore integration — early validation, L1 tie preference, donor protection, full/borrow adoption, fail-closed reset to a cold read. |
| `d1d505bd` | Reader admission on the file's actual metadata (the old pre-parse bound over-charged checkpoints and refused a valid restore). |
| `f2f60c34` | Phase D: `/metrics` and Monitor L1/L2 card, refusal reasons, disk eviction log, startup bytes/entries in INFO, tools and docs. |
| `07f2b9fb` | Implementation status and the unrun-validation record that preceded the reference-host run. |
| `9eea39bc` | Measured evidence, test-tool fixes, and the staging default raised from 512 to 1024 MiB. |
| `217c5288` | Phase 2: the content-addressed page pool, CoW saves, compaction, the page reader, demotion, flags and tests. |
| `55de6d9c` | Phase 2 Monitor: write/reuse, pool and compaction fields. |
| this commit | Phase 2 docs, status and tool updates. |

## Reference-host measurement

Host: Windows, RTX 4090 24 GB, 63.7 GiB RAM, CUDA 13.4, Samsung 980 PRO 2TB NVMe. Model: Qwen3.8-Flash-Next,
`unsloth-ud-iq4_xs`, 262K context, int8 KV, 36 GiB resident experts. Commit `07f2b9fb`, engine `0.1.51`.

The build was **reconfigured for sm_89** first: the original `build-cuda` was sm_75 only and every prefill of about
7.8K tokens or more crashed with `prefill copy_i32: unspecified launch failure`. The same source built for sm_89 ran.
This was a build-architecture setting, not a source change.

### 194K main + small sub, disk tier taking the conversation

| Turn | Prompt | Reused | Reuse | HTTP total | Restore | Restore rate | Spill | File |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAIN 1 cold | 194,382 | 0 | — | 166.55 s | — | — | 2.09 s | 3,056 MiB |
| MAIN 2 | 194,483 | 194,389 | 99.952% | **4.14 s** | 1.73 s | 1,770 MiB/s | 1.92 s | 3,057 MiB |
| MAIN 3 | 194,583 | 194,490 | 99.952% | 4.1-4.7 s | 1.48 s | 2,065 MiB/s | 1.58 s | 3,059 MiB |

- The restored HTTP turn is **4.14 s** without the `STRATA_STATE_HASH` diagnostic, against 166.55 s cold. With the
  hash enabled the same turn was 14.5-14.8 s; that diagnostic costs about 9.6 s after a 194K request.
- The disk file is **3.06 GiB**, not the plan's ~5 GB estimate.
- Saves wrote at 1,463-1,932 MiB/s; restores at 1,770-2,065 MiB/s.
- The actual prompt was 194.4K tokens, not 200K exactly.
- Parity: the live and disk-restored 194K states matched **field for field** in `STATE_HASH` (including gdn, ple,
  tail, pooled, kv, mtp, stale, dead, pooled_full, ple_prev) and produced identical text. The small direct-engine
  parking test was not reproducible (one IDENTICAL, one DIFFER) because the engine's own greedy A answers differed
  between arms; the 194K hash comparison is the parity evidence.
- VRAM: +172 MiB for the disk run, +106-240 MiB across the runs, inside the 256 MiB target.

### The product path: L1 enabled, L1 refuses, disk takes it

- With `--conversation-cache-mib 2048` (the 194K snapshot is 3.62 GiB), the engine logged
  `skip parking (snapshot 3622 MiB exceeds available budget)` then `spilled ... to disk`, and the next turn
  `restored ... from disk`. Three saves, two restores, zero errors, 99.952% reuse. The small sub-agent conversations
  stayed in RAM. This is the intended shape.
- With the L1 floor raised (`--conversation-cache-min-free-mib 16000`), the physical-RAM path also spilled and
  restored at the same reuse.
- With the reference `--conversation-cache-mib 8192` the 3.62 GiB snapshot fit in RAM, so the disk tier stayed idle.
  That is correct behavior: the disk tier is for conversations RAM cannot hold.

### Host memory: private passes, working set does not

| Point | Working set | Private |
|---|---:|---:|
| End of MAIN 1 (live 194K) | 41,939 MiB | 68,089 MiB |
| Spill peak | 44,841 MiB (**+2,890**) | 68,446 MiB (**+357**) |
| Restore peak | 44,407 MiB (+2,456) | 67,996 MiB (-93) |
| Sustained after MAIN 2 | 44,099 MiB (+2,148) | 67,682 MiB (-407) |

The cache's **private** memory rose 357 MiB during the spill and returned below the live footprint; the restore's own
transient was 461 MiB. The **working set** rose 2.9 GiB because the OS caches the 3.0 GiB file being written; that
cache is reclaimable (free physical recovered from 4.3 GiB to 7.3 GiB after the spill). A strict working-set reading
of the plan's 512 MiB bound fails; a private/committed reading passes. Both are in the evidence.

## Acceptance criteria (plan §2.3)

| Criterion | Measured result |
| --- | --- |
| Same next tokens / `STATE_HASH` | **Pass** at 194K (field-for-field hash, identical text). The 3.7K direct-engine test was flaky for engine-determinism reasons, not a restore mismatch. |
| Every follow-up reuses ≥ 99% | **Pass**: 99.952%. |
| 194K main turn ≤ 10 s | **Pass** on the actual HTTP path: 4.14 s restored (4.53-4.95 s by restore + tail in the hash run). |
| Restore in seconds | **Pass**: 1.48-1.79 s at 1.77-2.07 GiB/s. |
| Host save/restore peak ≤ 512 MiB | **Pass on private memory** (+357 MiB save, -93 MiB restore); **fail on process working set** (+2.89 GiB at the spill peak), which is dominated by reclaimable file cache from the 3.0 GiB write. |
| VRAM staging ≤ 256 MiB | **Pass**: +106-172 MiB (max +240 in the L1 run). |
| Disk full / read-only leaves RAM working | Preflight and fake-writer unit tests only; real disk-full was not injected. |
| Corrupt/truncated/foreign/missing falls back | Unit-tested at the store and reader level; a live corrupt-file request was not run. |
| Default off unchanged | No store I/O, no fingerprint, no directory; the only visible change is the new INFO keys. Unit tests pass; not re-measured end to end in this run. |
| Windows CUDA; HIP | Windows CUDA built and run. HIP/SYCL/Metal not built (no AMD GPU). |

## Remaining gaps and decisions

- **Staging default is now 1024 MiB**, not 512: an 11.7K restore asked for 537 MiB (metadata + one layer + the 16 MiB
  block), so 512 refused it. The measured private peak is 357 MiB, so the default is an admission bound, not a
  measured peak. A user-set smaller value still refuses with a message that names the limit.
- **The strict 512 MiB working-set bound is not met** because the OS caches the file write. If that must be reduced,
  the next step is unbuffered I/O (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux) for the payload, or
  streaming the running-state arrays instead of the current host copies. The restore transient already passes.
- **`SNAPSHOT_VERIFY` is not logged for disk restores.** The 194K state hash includes the draft fields and matched, so
  the draft K/V was correct; the explicit read-back log line exists only on the RAM path.
- **One deepest checkpoint per file.** Interior checkpoints are not stored, so interior borrowing from a disk entry
  is not available yet.
- **Phase 2's page pool is implemented but unmeasured.** The one-file format is still available with
  `--conversation-cache-disk-whole-file`; interior checkpoints are still not stored; DirectStorage/GDS is Phase 3.
- **No-buffer I/O, failure injection (disk full/read-only), HIP/SYCL/Metal, and the default-off end-to-end comparison**
  remain. The unit suites cover the mechanisms.
- The `build-cuda` directory must be configured for the card's architecture. On this 4090:
  `cmake -S . -B build-cuda -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89`.

## Evidence

- [evidence/disk-cache-200k.md](evidence/disk-cache-200k.md) — the main run, parity, host/VRAM samples.
- [evidence/disk-cache-200k-followup.md](evidence/disk-cache-200k-followup.md) — phase-aligned host cost, L1-refusal
  path, and the no-hash TTFT.
- The tool JSONs, engine logs, sampler CSVs, configs and harness scripts are in [evidence/](evidence/).
- Unit tests: CPU `conversation_file`, `conversation_stream`, `conversation_disk`, `conversation_cache`,
  `conversation_memory`; CUDA `conversation_snapshot_test` (4139 checks); server `test_monitor` and
  `test_conversation_cache_disabled` (22 tests). All passed on the branch.

The numbers above are one model, one drive and one host. Do not quote them as another configuration's performance.
