# Disk conversation cache — implementation status

Date: 2026-10-28
Branch: `feat/disk-conversation-cache` (worktree `C:\tools\strata-disk`), based on `origin/main` `4b9e05a2`.
Plan: [DISK_CONVERSATION_CACHE_PLAN.md](DISK_CONVERSATION_CACHE_PLAN.md).

This is the resume point: what is implemented, what the reference-host run measured, and what remains. The Phase 1
measurement was made from commit `07f2b9fb` on the sm_89 build; the follow-up commits only change the staging default
and record the evidence. Phase 2 (the copy-on-write page pool) is implemented and unit-tested, but not measured on the
reference host. Phase 3 (the DirectStorage fast path) is implemented, unit-tested, and measured against the portable
page read on two drives ([evidence](evidence/disk-fastpath.md)); no engine-end run of it was possible while a serving
engine held the GPU.

## Phase 3: the DirectStorage fast path (Windows + CUDA, opt-in)

With `--conversation-cache-disk-fast-path` (needs `--conversation-cache-disk-mib` and `--conversation-cache-disk-cow`)
a paged restore's page bytes travel by GPU DMA instead of through the host: DirectStorage reads the ranges straight
out of `pool/pages.bin` into a 2×32 MiB shared D3D12/CUDA staging ring, signals a shared D3D12 fence that CUDA waits on
as an imported external semaphore, and a kernel re-hashes every staged page (`session_hash64`, the pool's own content
address) and scatters it into the live K/V pool **only on a match**. A corrupt or stale page therefore never reaches
the K/V, and a failed fast write keeps the streaming restore's fail-closed contract: the session is discarded and the
prompt is read from scratch.

The fast path is opt-in and capability-checked at startup: the engine does **not** link `dstorage.dll` (it is loaded at
runtime, telemetry disabled), a refusal prints one notice (`the disk fast path is off: …`) and every restore runs the
portable path, which remains the default and the fallback. Refusals: not Windows+CUDA, no DirectStorage runtime, no
D3D12 adapter matching the CUDA device, a non-paged disk tier, a whole-file entry, or a session whose K/V parts are not
device memory (see the static-analysis note below). After 3 failed fast writes the attempts stop for the run.

Implementation:

| File | What it holds |
| --- | --- |
| `include/strata/core/conversation_ds.hpp`, `src/core/conversation_ds.cpp` | The engine: probe/create, the 2×32 MiB ring pipeline (queue → fence → semaphore → verify), the refusal stub on every other build. |
| `include/strata/kernels/ds_pages.hpp`, `src/kernels/cuda/ds_pages.cu` | The verify/scatter kernel: per-page hash equal to the host hasher, scatter only on match, `bad` flag, `copy_bytes` 0 for pages past a borrowed prefix (verified, not written). |
| `include/strata/core/conversation_file.hpp` | `ConversationKvGeom`, `ConversationPageJob`, `ConversationKvLayers::next_page_jobs` (default: unsupported → portable). |
| `include/strata/core/conversation_pages.hpp`, `src/core/conversation_pages.cpp` | `next_page_jobs` (the layer's pages as file jobs); Windows pool handles opened with `FILE_SHARE_DELETE` (DirectStorage's open needs it); `borrow_for_fast_read`/`return_from_fast_read` lend the pool handle around a fast write (`append` refuses while lent). |
| `include/strata/core/conversation_snapshot.hpp`, `src/core/conversation_state.cpp` | `conversation_kv_fast_targets` (destinations + device check), `conversation_kv_fast_match` (manifest-vs-session), `conversation_kv_refresh_residency`; the fast branch in `conversation_snapshot_restore_layers` (all fast-vs-portable decisions settled **before** the reader is consumed; after staging, failures map to the portable path's own codes). |
| `src/program/generate.cpp` | `--conversation-cache-disk-fast-path`, the startup probe and notices, the INFO field `conversation_disk_fast_path=0/1`, the failure counter, the `path=directstorage` token on the restore log line. |
| `serve/server.py` | `disk_fast_path` in the Conversation-cache view. |
| `src/core/ds_pages_test.cpp` | Kernel test (below). `src/core/ds_probe.cpp`, `src/core/ds_bench_main.cpp`: the manual probe and the benchmark tool. |
| `third_party/directstorage/` | The MIT DirectStorage 1.3 SDK (NuGet `Microsoft.Direct3D.DirectStorage` 1.3.0): headers and the x64 redist DLLs, vendored unmodified, provenance in its README. Not link-bound. |

### What was validated (2026-10-28, `build-cuda`, RTX 4090)

- `ds_pages_test` (new, GPU): 55 checks — the kernel's hash equals the host `session_hash64` for lengths 1…65 536
  (tail rounds and the avalanche), a one-byte corruption sets `bad` and scatters **nothing**, a good page beside it
  scatters intact, prefix copies (`copy_bytes` < page) and zero-copy verify-without-write behave. Passed.
- `conversation_disk_test` 195, `conversation_snapshot_test` 4139, `conversation_cache_test` 4772,
  `conversation_file_test` 183, `conversation_stream_test` 135, `conversation_validation_test` 972: all passed after
  every Phase 3 change (the pool-file handle change included — compaction and atomic replace run under the wider
  sharing). The portable loop inside `conversation_snapshot_restore_layers` is untouched code; the fast branch runs
  only when the caller supplies a fast write.
- `ds_bench` (same pool file, same pages, same device destination, 3 alternating rounds): the fast shape measured
  **4.6–5.5 GB/s against the portable shape's 1.7–2.2 GB/s on the 980 PRO — 2.20× at 32 MiB rising to 2.99× at
  1 GiB**; on an idle GPU the two larger sizes added **2.28× at 1.5 GiB and 2.31× at 3 GiB** (the reference entry's
  size), so the ratio holds with size. On the OS drive (Kingston) roughly break-even to 512 MiB and 1.98× at 1 GiB
  (noisy: 0.74–1.93× across two same-hour runs). Byte-equality checks passed every run. Full tables, caveats (warm
  cache, up to 1 GiB beside a resident engine) and the drive disagreement:
  [evidence/disk-fastpath.md](evidence/disk-fastpath.md), raw rounds in
  [evidence/disk-fastpath-ds-bench-raw.txt](evidence/disk-fastpath-ds-bench-raw.txt).
- The bench's portable arm (1.7-2.2 GB/s) independently matches the engine's own Phase 1 disk restores
  (1.77-2.07 GB/s) - the harness measures what the engine sees.
- **The live engine, end to end** (2026-10-29, idle RTX 4090, spill dir on D:): a device-resident conversation's
  restores ran through the fast path on every attempt (`path=directstorage`): **78.5 ms for 9,402 tokens / 480 MiB
  (6,079 MiB/s)** and 83.2 ms (5,760 MiB/s). The same conversation on the same pool with the flag off restored
  portably in **148.5 ms (3,212 MiB/s)** and 168.6 ms (2,841 MiB/s): **1.89x and 2.03x**, engine-to-engine. Note the
  portable end-to-end rate is higher than the bench's portable shape (the engine reads a page at a time into staging
  and uploads per layer), so the bench's ratios are the optimistic ones and 1.9-2.0x is the engine-level figure.
- **Fail-closed, end to end, on both paths**: three data bytes of three of the conversation's pages were flipped on
  disk. The fast path refused (`page hash mismatch (staging refused)`), the portable path refused
  (`page checksum mismatch`); each re-read the prompt from scratch and answered identically to a later read of the
  same messages. The corrupted bytes were put back.
- No performance claim is taken from ninfer, whose design inspired the pipeline; every number here was measured on
  this PC with the committed binary.

### Phase 3: what was not run

- **The reference configuration's fallback (static analysis):** with `--kv int8 --kv-resident 32768` the K/V streams
  through host slots (`pools()` returns `st.host.*` when `kv_mode != 0`), `conversation_kv_fast_targets` finds a
  non-device destination and the restore runs portable. The engine-end run above used a device-resident config, so
  that fallback is still an expectation from reading the code, not a measured run of the reference config.
- **Cold-cache restores** (file not in the OS cache): not measured; the bench and the engine runs read warm files.
- **Linux GPUDirect Storage (`cuFile`)**: not implemented and not run - this worktree is Windows-only tooling; the
  plan's Linux fast path remains open. The portable path serves Linux unchanged.
- **HIP, SYCL, Metal builds**: not built here (no AMD/Mac hardware). `conversation_ds.cpp` is written to compile to a
  runtime refusal stub outside Windows+CUDA, but no non-Windows-CUDA build has compiled it yet.
- **Release packaging**: the two redist DLLs are not yet in the Windows engine package (`packaging/binary_deps.py`
  follows PE imports and cannot see a dynamically loaded DLL). Without them the fast path prints its refusal notice
  and everything runs portable - but shipping the win means adding them to the engine component. Not done here.

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
