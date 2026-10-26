# Disk conversation cache — implementation status

Date: 2026-10-26
Branch: `feat/disk-conversation-cache` (worktree `C:\tools\strata-disk`), based on `origin/main` `4b9e05a2`.
Plan: [DISK_CONVERSATION_CACHE_PLAN.md](DISK_CONVERSATION_CACHE_PLAN.md).

This is the resume point. It says what is implemented, what the tests prove, and what has **not** been run. No
performance number is claimed: the 200K end-to-end measurement has not been made.

## Commits

| Commit | What it adds |
| --- | --- |
| `d74859bd` | Phase A: CPU-only `ConversationDiskStore` — manifest, in-RAM index, matching/borrowing, LRU, budget, free-space preflight, atomic publish, startup scan, path safety, unit tests. |
| `fe69eeee` | Phase B1: `SessionFileStream` — validates a session file and yields one K/V layer at a time, bounded host staging, payload hash checked in `finish()`. |
| `6e30702c` | Phase B2: `conversation_snapshot_restore_stream` — restores a streamed file layer by layer, full and prefix (borrow) modes, tested against the whole-image restore on CUDA. |
| `0543dc9c` | Phase C1: `--conversation-cache-disk-*` flags, startup identity/store, INFO fields, and the save path. The disk preflight runs before `make_room` can evict parked conversations. |
| `6924e4e1` | Phase C2: restore integration — early validation, L1 tie preference, donor protection, full/borrow adoption, fail-closed reset to a cold read. |
| `d1d505bd` | Reader admission on the file's actual metadata (the old pre-parse bound over-charged checkpoints and refused a valid 512 MiB restore). |
| `f2f60c34` | Phase D: `/metrics` and Monitor L1/L2 card, refusal reasons, disk eviction log, startup bytes/entries in INFO, tools and docs. |

## Tests run on this host

Host: Windows, RTX 4090 24 GB, 63.7 GiB RAM, MSVC 14.51, CUDA 13.4. No AMD GPU; no HIP build.

| Command | Result |
| --- | --- |
| `cmake --build build-cuda --target strata conversation_snapshot_test` | built |
| `ctest --test-dir build-cuda -R conversation_snapshot_test` | 1/1 passed; direct run: 4139 checks |
| `ctest --test-dir build-cpu -R "conversation_(file\|stream\|disk\|cache\|memory)_test"` | 5/5 passed |
| `python -m unittest serve.test_monitor tools.test_conversation_cache_disabled` (repo venv) | 22 tests, OK |
| `node --check serve/web/app.js` | OK |
| `strata --help`; `--conversation-cache-disk-mib 64` with no folder | flags listed; exit 2 with a clear message |
| `git diff --check` | clean |

The unit tests cover: manifest round-trip/version/checksum/identity, LRU and pinned protection, budget and free-space
refusals without eviction, startup pruning, layer-at-a-time file reads and restores, full/prefix parity with the
whole-image restore, no-MTP files, and the Monitor's parsing and view.

## Acceptance criteria (plan §2.3)

| Criterion | Status |
| --- | --- |
| Same next tokens and `STATE_HASH` as the live state | **Not run end to end.** The GPU test proves the streamed restore produces the same K/V and running state as the whole-image restore. |
| Every follow-up reuses ≥ 99% | **Not run.** |
| 200K main turn ≤ 10 s | **Not run.** |
| Restore/save host peak ≤ 512 MiB | Bounded by design and the reader test (metadata + one layer + 16 MiB block); **not measured on a model**. |
| VRAM staging ≤ 256 MiB | **Not measured.** The restore writes one layer at a time; no whole-image VRAM staging exists. |
| Disk MB/s and ms reported | The engine logs ms and MiB/s per restore; **no measurement has been made**. |
| Corrupt/truncated/foreign/missing falls back | Unit-tested at the store and reader level; the engine's fallback is compiled but **not exercised on a live request**. |
| Disk full / read-only leaves the RAM cache working | Preflight and fake-writer tests; **real disk-full not injected**. |
| Default off behaves as the release | No directory, no fingerprint, no store I/O when the buffer is 0. The only visible change is the new INFO keys (`conversation_disk_*` = 0/empty/0/0/0). Existing tests pass. |
| Windows CUDA; HIP build | CUDA built and tested. **HIP not built** (no AMD GPU here). SYCL/Metal untouched. |

## Known limitations and decisions

- **Single GPU, one request at a time.** `--layer-split`, `--batch` and `--peer-device` turn the tier off with one
  warning and the RAM cache serves alone.
- **Synchronous save.** The next request waits for the file write. A background writer is Phase 2.
- **One deepest checkpoint per file.** v1 stores the live state and the deepest checkpoint, as the session-file save
  does. Interior checkpoints are not stored, so a new conversation cannot borrow an interior prefix from a disk entry
  yet. The RAM cache still borrows as before.
- **Full-file rewrite per save.** Phase 2 adds the content-addressed page pool and CoW deltas. No DirectStorage/GDS
  (Phase 3).
- **Plaintext files.** A spill directory is required; the engine does not pick a default folder. POSIX creates
  0700/0600; Windows inherits the user profile's ACL.
- **Save staging default is 512 MiB**, not the plan's 256: the admission figure is the live state plus two checkpoint
  copies (~371 MiB on the reference model), so 256 refused every spill. A user-set smaller value refuses with a
  message that names the limit.
- **The disk decision is strictly longer than L1's.** Equal prefixes stay in RAM.
- **A failed restore is a cache miss.** The entry is removed, the retained K/V is dropped, the live state is reset, and
  the prompt is read from token 0. The cold read's `session_zero` wipes the partial state.
- **The reader's admission is after metadata parsing.** A valid file is admitted on the metadata it actually holds plus
  one layer and the 16 MiB block. A corrupt but identity-matching file can allocate up to the caller's read limits
  before the refusal; that is bounded by the engine's maxima, not by the whole file.
- **LRU touches are in RAM.** A restart orders entries by the save-time stamps. Startup evictions are not logged as
  events; INFO reports the starting bytes and entries.

## What to run next (Phase 0 and the Phase 1 exit)

The host's GPU was held by a pre-existing engine (PID 27792) throughout this work, so no model was loaded and no
end-to-end request was sent. The next session should:

1. Use an idle GPU, then start the reference configuration with the disk tier:
   `--conversation-cache-mib 8192 --conversation-cache-disk-mib 32768 --conversation-cache-spill-dir <NVMe dir>`.
2. Run `tools/conversation_disk_cache_test.py` (added, **not run**) to alternate a long MAIN with a short SUB.
3. Measure: file size, save ms, restore ms, MiB/s, save/restore host peaks, VRAM peak, TTFT, reuse %, and `STATE_HASH`
   parity. Record the drive model and interface and the exact commit.
4. Repeat with `--conversation-cache-mib 0` (disk-only) and with a lower `--resident-budget-gib` to show the trade.
5. Build HIP before review; state SYCL/Metal support (the session-file code does not claim them).

Do not present the unit results above as a performance measurement. They prove the mechanisms, not the speed.
