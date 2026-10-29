# Disk fast path (Phase 3) — DirectStorage restore measured against the portable page read

Date: 2026-10-28. Branch `feat/disk-conversation-cache`, worktree `C:\tools\strata-disk`.
Tool: `ds_bench` (`src/core/ds_bench_main.cpp`, manual-run target beside `ds_probe`; never a ctest test).
Raw output of every round: [disk-fastpath-ds-bench-raw.txt](disk-fastpath-ds-bench-raw.txt).

## What was measured, and what it isolates

`ds_bench` builds a real page pool with the engine's own writer (`ConversationPagePool::append`:
64 KiB pages, content hashes, 4 KiB-aligned records), then runs the two restore I/O shapes over the
**same pool file, the same pages, and the same device destination**, in one process, three alternating
rounds each, and checks both shapes' bytes against the source:

- **portable** — the paged restore's shape: `ConversationPagePool::read` (hash-verified page reads)
  into host staging, copied to pinned memory, then one host-to-device copy;
- **fast** — `ConversationDs::write`: DirectStorage DMAs the page ranges into a 2×32 MiB shared VRAM
  ring, a kernel re-hashes every staged page and scatters it to the destination only on a match.

Nothing here goes through a session or a model: it isolates the bytes moved. A full engine restore adds
the checkpoint read and the per-layer residency refresh on top — small compared to the K/V bytes.

## Host

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, driver 617.14 (CUDA build, `build-cuda`) |
| Engine build | `strata.exe` 2026-10-28 22:11 (same tree as this commit) |
| DirectStorage | `dstorage.dll` / `dstoragecore.dll` 1.3.2506.2501 (NuGet 1.3.0, MIT, vendored) |
| Pool drive 1 | **D:** Samsung SSD 980 PRO 2TB (NVMe, data drive) |
| Pool drive 2 | **C:** KINGSTON SNV2S1000G (NVMe, OS drive) |
| GPU state | a strata engine was serving (≈22.7 GiB resident) through every run |

## D: — Samsung SSD 980 PRO 2TB (best of 3 rounds; every round in the raw log)

| Pool bytes | portable | fast | ratio |
| --- | --- | --- | --- |
| 32 MiB | 14.8 ms (2,155 MB/s) | 6.7 ms (4,746 MB/s) | **2.20×** |
| 64 MiB | 29.9 ms (2,144 MB/s) | 12.5 ms (5,120 MB/s) | **2.39×** |
| 128 MiB | 57.9 ms (2,209 MB/s) | 24.3 ms (5,271 MB/s) | **2.39×** |
| 256 MiB | 114.8 ms (2,230 MB/s) | 48.0 ms (5,335 MB/s) | **2.39×** |
| 512 MiB | 238.4 ms (2,148 MB/s) | 94.1 ms (5,443 MB/s) | **2.53×** |
| 1,024 MiB | 566.7 ms (1,807 MB/s) | 189.7 ms (5,398 MB/s) | **2.99×** |

The fast shape holds 4.6–5.5 GB/s across the whole range; the portable shape stays under ~2.2 GB/s.
Round-to-round spread is under 8% except one fast round at 1 GiB (5,004 MB/s).

## C: — KINGSTON SNV2S1000G (OS drive; best of 3 rounds)

| Pool bytes | portable | fast | ratio |
| --- | --- | --- | --- |
| 32 MiB | 17.0 ms (1,883 MB/s) | 19.1 ms (1,675 MB/s) | 0.89× |
| 64 MiB | 34.5 ms (1,854 MB/s) | 36.9 ms (1,732 MB/s) | 0.93× |
| 128 MiB | 73.2 ms (1,749 MB/s) | 72.3 ms (1,770 MB/s) | 1.01× |
| 256 MiB | 125.9 ms (2,034 MB/s) | 142.7 ms (1,794 MB/s) | 0.88× |
| 512 MiB | 265.3 ms (1,930 MB/s) | 286.0 ms (1,790 MB/s) | 0.93× |
| 1,024 MiB | 598.6 ms (1,711 MB/s) | 302.2 ms (3,389 MB/s) | **1.98×** |

On this drive the fast path is roughly break-even up to 512 MiB and about 2× at 1 GiB. The drive is the
busy OS drive and noisy: an earlier run from the same hour on the same binary ranged 0.74× (32 MiB) to
1.93× (1 GiB) with the crossover at ~128 MiB instead of ~1 GiB. Both runs agree that on this drive small
restores gain nothing and large ones gain about 2×.

## What the portable baseline's level means

The portable shape here measured 1.7–2.2 GB/s. The engine's own whole-file disk restores on this host
measured 1.77–2.07 GB/s in the Phase 1 run (1.48–1.73 s for a 3.06 GiB file,
`disk-cache-200k-*.log`). The harness's portable arm matches the engine's real restores, so the fast
arm's gain is removal of that ~2 GB/s ceiling, not a caching advantage.

## Caveats

- **Warm file cache.** Each size reads its pool file repeatedly; no attempt was made to drop the system
  file cache (both arms read the same cached file — the steady state of a serve that keeps restoring the
  same entry). Cold-cache behavior is not measured.
- **Contended GPU.** A serving engine held ~22.7 GiB through every run. That is why the largest size is
  1 GiB (the bench's destination buffer could not grow past the free memory), and why some rounds vary;
  PCIe was otherwise idle.
- **1 GiB is the largest measured size.** The reference 194K entry is 3.06 GiB; the D: ratio trend
  (2.20× → 2.99× over 32 MiB → 1 GiB) is **not** extended to 3 GiB as a claim — that size was not run.
- **The drive matters.** D: and C: disagree; the fast path's win is a property of drive + queue behavior,
  not guaranteed on every machine. This is why the fast path is opt-in.
- **Config scope (static analysis, unconfirmed by an end-to-end run).** The fast path stages only into
  device-resident K/V pools (`conversation_kv_fast_targets` checks every part with
  `cudaPointerGetAttributes`). The reference configuration (`--kv int8 --kv-resident 32768`) streams its
  K/V through host slots (`pools()` returns `st.host.*` whenever `kv_mode != 0`), so its restores are
  expected to refuse the fast path and run portable. Reading code, not measurement: no engine-end run
  exercised either outcome.

## Claims kept apart

- Numbers here are all measured on this PC, this hour, with the committed binary. All rounds are in the
  raw log, passed and slow alike.
- **Nothing is claimed from ninfer's published numbers.** The implemented mechanisms (adapter match by
  LUID, shared default-heap buffer, fence-as-semaphore, 32 MiB ring) are engineering choices; no
  performance statement of theirs is reused or echoed as evidence.
- Correctness of the staging kernel (hash equals the host hasher; a corrupt page scatters nothing) is
  measured by `ds_pages_test` (55 checks, passes); this bench confirms byte-equality of both shapes
  end-of-pipe (2 checks per run).
- Not run: engine end-to-end `path=directstorage` restore, 3 GiB sizes, cold cache, Linux GDS (cuFile),
  AMD/Mac builds (they compile to a refusal stub). Listed in
  [DISK_CACHE_STATUS.md](../DISK_CACHE_STATUS.md).
