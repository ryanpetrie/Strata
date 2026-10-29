# Disk fast path (Phase 3) — DirectStorage restore measured against the portable page read

Date: 2026-10-28. Branch `feat/disk-conversation-cache`, worktree `C:\tools\strata-disk`.
Tool: `ds_bench` (`src/core/ds_bench_main.cpp`, manual-run target beside `ds_probe`; never a ctest test).
Raw output of every round: [disk-fastpath-ds-bench-raw.txt](disk-fastpath-ds-bench-raw.txt).
The live-serve run's inputs and outputs are beside this file: `disk-fastpath-e2e-config.json` (fast path on) and
`disk-fastpath-e2e-portable-config.json` (the same config with the flag removed), `disk-fastpath-e2e-drive.py` (the
driver), `disk-fastpath-e2e-engine-log.txt` / `disk-fastpath-e2e-report.json` and their `-portable-` counterparts (the logs are copies of the engine's, renamed because `*.log` is ignored), plus
each server's stdout (`*-server.out`).

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

### Full sizes, GPU idle (same day, the serving engine stopped)

The sizes up to 1 GiB above were run beside a serving engine (see caveats). With the GPU free the
two larger sizes were added; first-round fast writes show a warm-up (GPU clocks, a fresh 3 GiB
destination's page tables), later rounds are steady:

| Pool bytes | portable (best) | fast (best) | ratio | fast, slowest round |
| --- | --- | --- | --- | --- |
| 1,536 MiB | 660.4 ms (2,326 MB/s) | 289.8 ms (5,300 MB/s) | **2.28×** | 616.4 ms (2,492 MB/s) |
| 3,072 MiB | 1,457.4 ms (2,108 MB/s) | 630.9 ms (4,869 MB/s) | **2.31×** | 1,014.9 ms (3,027 MB/s) |

3 GiB is the reference entry's size (194K tokens of int8 K/V, 3.06 GiB). The ratio stays at ~2.3×
through 3 GiB: on this drive the fast path does not degrade with size.

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

## End-to-end through the live serve (fast path on)

A real serve on this PC (same build, RTX 4090), `--conversation-cache-disk-only --conversation-cache-disk-cow
--conversation-cache-disk-fast-path`, K/V device-resident (no `--kv`; `--kv-resident 0` is the default and means
"all of it in VRAM"), spill dir on **D:** (the 980 PRO). Two conversations alternating so that each one is parked and
restored; every restore is the fast path (`conversation_disk_fast_path=1` in the INFO line):

| Turn | Engine log | Wall |
| --- | --- | --- |
| A restore 1 | `restored 9402 tokens from disk in 78.5 ms (6079 MiB/s); tier=disk parked=0 bytes=0 path=directstorage` | 4.39 s, prefill 537 ms for the 27 new tokens |
| A restore 2 | `restored 9453 tokens from disk in 83.2 ms (5760 MiB/s); tier=disk parked=0 bytes=0 path=directstorage` | 4.20 s, prefill 293 ms for the 7 new tokens |

78.5 ms for a 480 MiB entry is 6.1 GB/s of page bytes through the whole engine path (queue, fence, kernel hash,
scatter), consistent with `ds_bench`. (The turn's own 4 s is prefill of the conversation's new tail plus generation
and the vision-free chat path, not the restore.)

### The same restore with the fast path off (portable), same pool, same conversation

The engine was restarted with the same config and spill dir, `--conversation-cache-disk-fast-path` removed, and the
same driver run again:

| Restore | fast path | portable | ratio |
| --- | --- | --- | --- |
| A restore 1 | 78.5 ms (6,079 MiB/s), `path=directstorage` | 148.5 ms (3,212 MiB/s) | **1.89×** |
| A restore 2 | 83.2 ms (5,760 MiB/s), `path=directstorage` | 168.6 ms (2,841 MiB/s) | **2.03×** |

Both runs read the same ~480 MiB of pages of the same conversation from the same pool on D:, warm in the OS cache;
the pairs are engine-to-engine, minutes apart. The portable end-to-end rate here (2.8–3.2 GB/s) is higher than the
bench's portable shape (1.7–2.2 GB/s): the engine's paged restore reads a page at a time into host staging and
uploads per layer, while `ds_bench` copies each page separately into pinned memory. So the bench's portable arm is
the slower of the two, its ratios are slightly optimistic, and the engine-level 1.9–2.0× is the number to quote.
Both runs are also warm-cache; see the caveats.

```text
portable:  strata serve: conversation cache: restored 9404 tokens from disk in 148.5 ms (3212 MiB/s); tier=disk parked=0 bytes=0
fast:      strata serve: conversation cache: restored 9402 tokens from disk in  78.5 ms (6079 MiB/s); tier=disk parked=0 bytes=0 path=directstorage
```

### Fail-closed end to end

Three data bytes of three different pages of the conversation's manifest were flipped on disk, then the next turn
asked for a restore:

```
strata serve: conversation cache: disk restore failed (disk fast path: page hash mismatch (staging refused)); reading the prompt
strata serve: prompt 9460 tokens = 0 reused + 9460 read in 8134 ms (1163.1 tok/s), 2 generated
```

The kernel's hash caught each corrupt page, the restore was refused, the disk entry was dropped, the session was
discarded and the prompt was read from scratch — and the answer was the same as a later request for the same
messages (`A-corrupt reply equals A-repeat reply: True`). The failure counter now reads one; the fast path keeps
serving (the next restore succeeded).

The portable run above was given the same corruption: its restore refused it too
(`disk restore failed (page checksum mismatch); reading the prompt`), re-read the prompt, and answered the same
(`True`). So the fail-closed contract holds on both paths at the engine level, and the corrupted pages were put
back afterwards.

An observation from the run, by reading the code (not a defect): when a request is about to read a disk entry, the
outgoing conversation is parked **into RAM only** (`DiskRestoreMark` suppresses the disk save), because that save
could evict or supersede the entry being read. With the RAM cache off (`--conversation-cache-mib 0`) the outgoing
conversation is therefore dropped, and the small second conversation used to force the parking was never itself
saved — its turns in this driver were cold reads by design.

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
- **1 GiB is the largest size measured with the engine resident**; the two larger sizes (1.5 and 3 GiB) were run
  with the GPU idle and are reported above. The reference 194K entry is 3.06 GiB, so its size is covered; an
  end-to-end restore of a real 194K entry through the fast path is the remaining gap (below).
- **The drive matters.** D: and C: disagree; the fast path's win is a property of drive + queue behavior,
  not guaranteed on every machine. This is why the fast path is opt-in.
- **Config scope.** The fast path stages only into device-resident K/V pools
  (`conversation_kv_fast_targets` checks every part with `cudaPointerGetAttributes`). The engine-end run above used
  a device-resident configuration and the fast path fired; the reference configuration
  (`--kv int8 --kv-resident 32768`) streams its K/V through host slots (`pools()` returns `st.host.*` whenever
  `kv_mode != 0`), so **those** restores are expected to refuse the fast path and run portable — static analysis,
  not measured, for that configuration.

## Claims kept apart

- Numbers here are all measured on this PC, this hour, with the committed binary. All rounds are in the
  raw log, passed and slow alike.
- **Nothing is claimed from ninfer's published numbers.** The implemented mechanisms (adapter match by
  LUID, shared default-heap buffer, fence-as-semaphore, 32 MiB ring) are engineering choices; no
  performance statement of theirs is reused or echoed as evidence.
- Correctness of the staging kernel (hash equals the host hasher; a corrupt page scatters nothing) is
  measured by `ds_pages_test` (55 checks, passes); this bench confirms byte-equality of both shapes
  end-of-pipe (2 checks per run).
- Not run: cold-cache restores, Linux GDS (cuFile), AMD/Mac builds (they compile to a refusal stub). The
  reference `--kv int8 --kv-resident 32768` configuration is expected to fall back to the portable path
  (host-mapped K/V) and was not exercised at the engine end. Listed in
  [DISK_CACHE_STATUS.md](../DISK_CACHE_STATUS.md).
