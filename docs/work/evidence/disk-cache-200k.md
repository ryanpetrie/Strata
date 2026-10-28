# Disk conversation cache — end-to-end validation on the reference host

Date: 2026-10-26 (UTC; local timezone UTC-06:00).
Commit: `07f2b9fb` (`feat/disk-conversation-cache`), engine `0.1.51`.
Host: Windows, NVIDIA RTX 4090 24 GB, 63.7 GiB RAM, CUDA 13.4, MSVC 14.51.
Model: Qwen3.8-Flash-Next, pack `unsloth-ud-iq4_xs` (`C:\tools\Strata-data`), tokenizer from the same pack.
Spill drive: `D:` = Samsung SSD 980 PRO 2TB, NVMe, NTFS (710,671,220,736 bytes free after the runs).

This report contains only numbers produced on this host during this session. Where a target was not met, the
measured value is given. Nothing here is a projection.

## Build note (read this first)

`build-cuda\strata.exe` was configured with `CMAKE_CUDA_ARCHITECTURES=75` (sm_75 SASS + PTX only) when the session
started. On this sm_89 card the tiny (55-token) prompt ran, but every prompt of about 7.8K tokens or more killed the
engine:

```
prefill copy_i32: unspecified launch failure
[strata] the engine stopped unexpectedly (exit code 1)
```

`C:\tools\strata\engine\strata.exe` (the 0.1.50 reference engine, sm_89 SASS) served the same 7,826-token prompt in
8.14 s on the same host, so the crash was in the build, not the config or the model. The same commit was rebuilt for
this card:

```
cmake -S . -B build-cuda -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --target strata --parallel 12     # inside vcvars64
```

`cuobjdump --list-elf build-cuda\strata.exe` then listed `strata.1.sm_89.cubin`; `--version` stayed `0.1.51`. No
engine source was changed. All numbers below are from the sm_89 rebuild.

## Configuration

The reference config `C:\tools\strata\strata-unsloth-ud-iq4_xs.json` was copied and these keys changed: `exe` ->
`C:\tools\strata-disk\build-cuda\strata.exe`, `cwd` -> `C:\tools\strata-disk`, `log` -> `build-cuda\...log`,
`host` -> `127.0.0.1`, `port` -> 8099 (200K) / 8098 (smoke). `before_load`, `before_load_strict` and `systemone` were
removed. `env` gained `STRATA_STATE_HASH=1` and `STRATA_SNAPSHOT_VERIFY=1` beside `STRATA_HC_Q8=1`.

Engine args (200K disk-only run; the L1+L2 run differs only in `--conversation-cache-mib 8192` and no
`--conversation-cache-disk-only`):

```
--pack C:\tools\Strata-data\packs\unsloth-ud-iq4_xs
--native C:\tools\Strata-data\models\unsloth-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
--expert-profile C:\tools\strata\data\expert-profile.bin --expert-cache auto --prefill auto
--spec 4 --mtp C:\tools\Strata-data\mtp\rt --max-context 262144 --kv int8 --kv-resident 32768
--resident-budget-gib 36 --vision --vram-reserve-mib 1536 --pcie-frac 0.35 --spec-min-p 0.50 --pool-workers 7
--conversation-cache-mib 0 --conversation-cache-slots 4 --conversation-cache-min-free-mib 2560
--conversation-cache-disk-only --conversation-cache-disk-mib 32768
--conversation-cache-spill-dir D:\strata-disk-cache-200k
--conversation-cache-disk-min-free-mib 4096 --conversation-cache-disk-staging-mib 1024
```

**Deviation from the brief:** `--conversation-cache-disk-staging-mib` is 1024, not 512. At 512 the very first disk
restore was refused (see “Track 1”); the value was raised to 1024 for every run after that. The measured host peaks
in this report are with the 1024 limit.

Configs used (copies in this folder): `disk-cache-smoke-config.json`, `disk-cache-200k-config.json` (L1+L2),
`disk-cache-200k-diskonly-config.json`, `parity-200k-live-config.json`, `parity-200k-disk-config.json`.

## Calibration

Measured offline with the pack tokenizer and chat template (the tool's filler, one system + one user turn):

| main_words | system tokens | prompt tokens | tokens/word |
|---:|---:|---:|---:|
| 5,000 | 19,346 | 19,442 | 3.888 |
| 50,000 | 194,286 | 194,382 | 3.888 |

`--main-words 50000` was used. The engine reported **194,382 prompt tokens** for MAIN turn 1, which matches the
offline figure exactly. The runs are 194.4K-194.6K prompts, not “200K” exactly; they are called 194K below.

`--step-words 20` was added to `tools/conversation_disk_cache_test.py` (default still `main_words // 8`, at least 16)
so the MAIN user turns stay tiny at this prompt size.

## Track 1 — disk-only smoke (port 8098)

`build-cuda\disk-cache-smoke-config.json`: disk-only, RAM cache 0, staging 1024. Command:

```
C:\tools\strata\.venv\Scripts\python.exe -m serve.server --engine strata --config build-cuda\disk-cache-smoke.json --host 127.0.0.1 --port 8098
C:\tools\strata\.venv\Scripts\python.exe tools\conversation_disk_cache_test.py http://127.0.0.1:8098 qwen3.8-flash-next-unsloth-ud-iq4_xs --key REDACTED --output build-cuda\disk-cache-smoke-report.json --turns 2 --main-words 3000 --sub-words 300 --step-words 20 --run
```

Result: **PASS** (`disk-cache-smoke-report.json`).

| Turn | Prompt tokens | Reused | Reused % | HTTP total s | Disk event |
|---|---:|---:|---:|---:|---|
| MAIN 1 | 11,693 | 0 | 0 | 12.34 | spill 11,697 tokens, 283.8 ms, 395 MiB |
| SUB 1 | 1,414 | 0 | — | 4.42 | — |
| MAIN 2 | 11,788 | 11,697 | 99.23 | 2.42 | restore 11,697 tokens, 136.0 ms, 2,904 MiB/s |
| SUB 2 | 1,729 | 0 | — | 4.35 | — |

`/metrics` after: `disk_saves=2, disk_restores=1, disk_errors=0, disk_entries=1, disk_bytes=416,276,384`.
File: `snap-d375333846c219c3.payload` = 416,276,384 bytes (396.99 MiB). It was a clean server start with an empty
spill directory.

### The 512 MiB staging refusal

With the brief's `--conversation-cache-disk-staging-mib 512` the same smoke failed on MAIN turn 2, with the engine
falling back to a cold read (log kept as `disk-cache-smoke-engine.log`; report
`disk-cache-smoke-report-staging512.json` in `build-cuda`):

```
strata serve: conversation cache: spilled 11697 tokens to disk in 283.8 ms; file=395 MiB tier=disk entries=1 bytes=414810276
strata serve: conversation cache: disk restore failed (session file: the disk restore needs 537 MiB of host staging, the limit is 512 MiB (--conversation-cache-disk-staging-mib)); reading the prompt
strata serve: prompt 11788 tokens = 0 reused + 11788 read in 11357 ms
strata serve: conversation cache: spilled 1418 tokens to disk in 171.4 ms; file=245 MiB
```

`/metrics` then read `disk_saves=2, disk_restores=0, disk_errors=1`; MAIN turn 2 reused 0. An 11.7K-token snapshot
needs 537 MiB of reader staging (metadata + one layer + the 16 MiB block), so a 512 MiB limit admits the spill but
refuses to restore it. This is a real finding about the default, not a host problem.

## Track 2 — 194K MAIN alternating with a small SUB

### L1+L2 run (RAM cache 8192 MiB, disk tier on, port 8099)

The disk tier was **never used**: the 194K snapshot (3,798,014,676 bytes, 3.62 GiB) fits the 8,192 MiB RAM cache, so
every park and restore stayed in RAM. The tool refuses to claim a disk result (`disk_saves` did not move), and its
report is `disk-cache-200k-l1-only-refused.json`.

Measured from the engine log (`disk-cache-200k-l1-engine.log`):

| Turn | Prompt | Reused | Reused % | Prompt read ms | Park/restore |
|---|---:|---:|---:|---:|---|
| MAIN 1 | 194,382 | 0 | 0 | 147,031 (cold) | park 194,389 in 1,284.5 ms, 3.62 GiB |
| SUB 1 | 1,414 | 0 | — | 5,160 | park 1,418 in 55.9 ms |
| MAIN 2 | 194,483 | 194,389 | 99.95 | 2,730 | restore (RAM) 194,389 in 1,221.9 ms |
| SUB 2 | 1,729 | 1,418 | 82.0 | 2,494 | restore 1,418 in 99.0 ms |
| MAIN 3 | 194,583 | 194,490 | 99.95 | 2,392 | restore 194,490 in 910.5 ms |

`SNAPSHOT_VERIFY` lines are present for these RAM restores, e.g.
`SNAPSHOT_VERIFY draft=a41ed0c909ee57a5 cells=1420 mode=2 source=ram resident=32856`.
This run shows the trade: an 8 GiB L1 keeps a 194K conversation and the disk tier is idle. To exercise L2, the RAM
cache must be 0 (or the L1 must refuse the park).

### Disk-only run (RAM cache 0, disk tier on, port 8099)

Command:

```
C:\tools\strata\.venv\Scripts\python.exe -m serve.server --engine strata --config build-cuda\disk-cache-200k-diskonly.json --host 127.0.0.1 --port 8099
C:\tools\strata\.venv\Scripts\python.exe tools\conversation_disk_cache_test.py http://127.0.0.1:8099 qwen3.8-flash-next-unsloth-ud-iq4_xs --key REDACTED --output build-cuda\disk-cache-200k-diskonly-report.json --turns 3 --main-words 50000 --step-words 20 --sub-words 300 --run
```

Result: **PASS** (`disk-cache-200k-diskonly-report.json`). First MAIN is a cold ~194K prefill (151.3 s); the SUB
requests park MAIN to disk, so MAIN turns 2 and 3 restore from disk.

| Turn | Prompt tokens | Reused | Reused % | HTTP total s | Prompt read ms | Restore ms | Restore MiB/s | Spill ms | File MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAIN 1 | 194,382 | 0 | 0 | 163.18 | 151,277 | — | — | 2,088.6 | 3,056 |
| SUB 1 | 1,414 | 0 | — | 6.33 | 5,997 | — | — | — | — |
| MAIN 2 | 194,483 | 194,389 | 99.952 | 14.77 | 3,219 | 1,726.7 | 1,770 | 1,923.2 | 3,057 |
| SUB 2 | 1,729 | 0 | — | 6.22 | 5,880 | — | — | — | — |
| MAIN 3 | 194,583 | 194,490 | 99.952 | 14.53 | 3,052 | 1,480.7 | 2,065 | 1,583.2 | 3,059 |
| SUB 3 | 2,044 | 0 | — | 6.01 | 5,648 | — | — | — | — |

- `reused` is the tool's delta of `/metrics conversation_cache.reused_tokens`; `request_reused` in the JSON matches.
  MAIN turn 2 reused 194,389 of 194,483 prompt tokens; turn 3 reused 194,490 of 194,583.
- **TTFT estimate** = restore + tail prefill: 1.727 + 3.219 = **4.95 s** (MAIN 2), 1.481 + 3.052 = **4.53 s**
  (MAIN 3), plus one decode step (~0.03 s). The cold MAIN 1 TTFT is the 151.3 s prompt read.
- **The HTTP turn total is 14.77 s / 14.53 s**, i.e. more than restore + prefill. The engine's own phase timers
  account for restore + prefill + 8-token decode + the synchronous spill of that same conversation at the end of the
  turn: 1.73 + 3.22 + 0.22 + 1.92 = 7.09 s (MAIN 2) and 1.48 + 3.05 + 0.15 + 1.58 = 6.26 s (MAIN 3). The remaining
  **7.7-8.3 s per turn** is the `STRATA_STATE_HASH=1` pass that runs after generation and is not one of the phase
  timers; it scales with tokens (about 0.2 s at 1,414 tokens, 9.6 s after the 194K cold read). The attribution to
  the hash is arithmetic plus that scaling; **a timing run with `STRATA_STATE_HASH` unset was not made**.
- Spills: 194,389 tokens in 2,088.6 ms (3,056.4 MiB, 1,463 MiB/s), 194,490 in 1,923.2 ms (3,057.9 MiB,
  1,590 MiB/s), 194,587 in 1,583.2 ms (3,059.3 MiB, 1,932 MiB/s). Saves are synchronous and part of the turn.
- Restores: 194,389 in 1,726.7 ms at 1,770 MiB/s; 194,490 in 1,480.7 ms at 2,065 MiB/s.
- `disk_errors=0`, `disk_refusals=0`, no `skip parking` / `skip disk save` / `disk restore failed` lines.

`/metrics` after the run: `disk_saves=3, disk_restores=2, disk_errors=0, disk_entries=1,
disk_bytes=3,207,875,828, last_tier=disk`, 6 requests, 388,879 reused of 588,635 prompt tokens (66.0% over the whole
alternation).

Files on disk after the run:

```
D:\strata-disk-cache-200k\conversations\snap-039fe4afcdd81612.payload   3,207,875,828 B  (3,059.3 MiB)
D:\strata-disk-cache-200k\manifests\snap-039fe4afcdd81612.bin               1,556,781 B
```

One entry, because v1 stores one file per conversation and each later park supersedes the previous file.

## Host and VRAM peaks

Sampler: `build-cuda\sample_memory.py`, every 250 ms requested (nvidia-smi call overhead made the real cadence about
305 ms), recording UTC time, `nvidia-smi memory.used`, the `strata.exe` engine's `WorkingSet64` and
`PrivateMemorySize64`, the sum over all `strata*` processes, and `Win32_OperatingSystem.FreePhysicalMemory`.
CSV: `disk-cache-200k-diskonly-sampler.csv` (baseline + run phases), `disk-cache-200k-l1-sampler.csv`.

Disk-only run (idle baseline after READY, before the first request):

| Metric | Baseline | Run peak | Change |
|---|---:|---:|---:|
| Engine working set | 39,236.9 MiB | 44,847.8 MiB | **+5,610.9 MiB** |
| Engine private bytes | 66,021.4 MiB | 68,432.5 MiB | +2,411.1 MiB |
| All `strata*` working set | 39,857.1 MiB | 45,467.9 MiB | +5,610.8 MiB |
| GPU used | 22,639 MiB | 22,811 MiB | **+172 MiB** |
| Host free physical | 8,718-8,765 MiB | 3,928.7 MiB (min) | -4,789 MiB |

The idle baseline is not the right reference for the 512 MiB rule, because a live 194K conversation needs host K/V
memory. During MAIN turn 1's prefill, before any park, the engine working set reached 41,858.4 MiB (still climbing)
at 13:23:29 UTC; the park/restore material then pushed it to 44,847.8 MiB. **The cache's own peak is about
+2,990 MiB over the live 194K footprint**, and it stays about 2.6 GiB above the pre-restore level after the turn
(44.4 GiB sustained). That is far above the plan's 512 MiB save/restore bound.

**Corrected in the follow-up** ([disk-cache-200k-followup.md](disk-cache-200k-followup.md)): a clean cache-off A/B
gives a live 194K footprint of 41,950.8 MiB and a spill peak of 44,840.7 MiB, so the cost is **+2,889.9 MiB**, not
+2,990. The 41,858.4 MiB value above was a real 2.7 s plateau, but 92.4 MiB below the clean live figure (the
"still climbing" wording was wrong); the peak itself reproduced within 7.1 MiB. A 512 MiB staging control run
(44,852.2 MiB peak) shows the cost is not the staging copy. The 512 MiB verdict does not change.

L1 run (for comparison): peak engine working set 48,428.5 MiB (+9,191.7 over idle), private 72,288.2 MiB, GPU
22,811 MiB (+240 over idle), free physical min 3,205.9 MiB. The extra ~3.5 GiB over the disk-only peak is the second
194K snapshot in RAM (the cache budget).

VRAM staging stayed **+172 MiB (disk run)** and **+240 MiB (L1 run)** over the loaded engine, inside the 256 MiB
target. The baseline includes the vision encoder, which the reference config loads.

## Parity

### 194K, live vs disk-restored (STATE_HASH, HTTP) — PASS

Same conversation sent twice in two server sessions. P1 is the 194K system prompt + first user turn; P2 is P1 + a
fixed assistant turn `ok` + a second user turn. Arm A (cache off) sends P1 then P2, so the engine continues the live
session. Arm B (disk-only) sends P1, an unrelated SUB (which parks P1 to disk), then P2, which is restored. Both arms
ran with `STRATA_STATE_HASH=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0`, `--pcie-frac 0`,
`--adapt-every 1000000` (the settings the engine needs to be reproducible; see the note below). Results
(`parity-200k-live.json`, `parity-200k-disk.json`):

| Step | Live hash | Disk hash | Live text | Disk text |
|---|---|---|---|---|
| P1 (L=194389) | `kv=e6a4962191ebb2e5` | `kv=e6a4962191ebb2e5` | `m10m11m1` | `m10m11m1` |
| P2 (L=194481) | `kv=37402fc68c49a6e4` | `kv=37402fc68c49a6e4` | `ok` | `ok` |

The full `STATE_HASH` line matched field for field (gdn, ple, tail, pooled, kv, mtp, stale, dead, pooled_full,
ple_prev), not only `kv`. Both P2 replies reused 194,375 prompt tokens. The disk arm's restore line:

```
strata serve: conversation cache: restored 194375 tokens from disk in 1756.1 ms (1740 MiB/s); tier=disk parked=0 bytes=0
```

P2 took 12.90 s live and 14.41 s restored (the difference includes the restore; both include the same STATE_HASH
overhead). File: 3,204,882,488 bytes.

The parity script wrote its result to the config's own file name, so the two `parity-200k-*-config.json` files here
were reconstructed from the `engine started:` line the server logged (that line is in
`parity-200k-live-engine.log` and `parity-200k-disk-engine.log`); the engine arguments above match it flag for
flag.

### 3,682-token direct-engine parking test (disk-only)

`tools/parking_test.py` was run with the disk-only config and `--repeat 60`. Two Windows fixes were needed: the log
path is now `tempfile.gettempdir()/batch_test_engine.log` (or `BATCH_TEST_LOG`), and a disk restore counts as a pass
even though `reused_kv_bytes=0` with disk-only. Command (deterministic settings added; see below):

```
set STRATA_ROUTE_TAIL_SKIP=0 & set STRATA_PREFILL_CPU_SHARE=0 & set BATCH_TEST_LOG=build-cuda\parity-engine.log
python tools\parking_test.py --exe build-cuda/strata.exe --config build-cuda/disk-cache-smoke.json --repeat 60 ^
  --extra "--conversation-cache-disk-only --conversation-cache-mib 32768 ^
  --conversation-cache-spill-dir D:\strata-disk-cache-parity --conversation-cache-disk-min-free-mib 4096 ^
  --pcie-frac 0 --adapt-every 1000000"
```

| Run | Restored | Follow-up token IDs | Evidence |
|---|---|---|---|
| 1 | 3,780 tokens, 46.6 ms, 6,006 MiB/s | **IDENTICAL** (96/96) | `parking-parity-engine-det.log` (engine log; console output is the first block in this section) |
| 2, same settings | 3,675 tokens, 47.5 ms, 5,894 MiB/s | DIFFER at token 66 (96/95) | `parking-parity.txt` (console), `parking-parity-engine-run2.log` (engine log); the parked snapshot was 3,783 tokens, not 3,780 |

The brief's original command (no exactness flags) gave `DIFFER at 0` (86/97 follow-up tokens). Adding the
docstring's `--pcie-frac 0 --adapt-every 1000000` still gave `DIFFER at 0` (89/87): both arms generated A with the
same first 8 tokens but the two complete 99-token A answers had different SHA-256 hashes
(`9cf1cfa81e7d5c10` vs `fab91d774e4cbaed`; log `parity-engine-exact-flags-A-differed.log`). That is engine
run-to-run variation, not the restore: the follow-up prompt contains A's answer, so a different answer changes the
prompt from the first token. Adding `STRATA_ROUTE_TAIL_SKIP=0` and `STRATA_PREFILL_CPU_SHARE=0` made A bit-identical
between the arms, and the two runs in the table followed.

The repeat that differed also restored a shorter prefix (3,675 vs 3,780 tokens) because the parked state itself was
3 tokens longer (3,783) and the matcher fell back to an earlier checkpoint. With the checkpoint tail recomputed,
the follow-up diverged at token 66. This direct-engine gate is therefore **not reproducible on this build**: one
IDENTICAL, one DIFFER under identical settings. The 194K HTTP hash comparison above is the parity evidence used for
this report. No 194K direct-engine parking run was made.

### SNAPSHOT_VERIFY

`SNAPSHOT_VERIFY draft=... source=ram` lines appeared after RAM restores in the L1 run. **No `SNAPSHOT_VERIFY` line
appeared for any disk restore** in the disk-only 200K run, the 200K parity run, or the direct-engine disk test, even
with `STRATA_SNAPSHOT_VERIFY=1` in the engine environment. Either the disk path does not perform the draft-ring
read-back or it does not log it; that check was not exercised on the disk path.

## Phase 1 exit criteria

| Criterion | Target | Measured | Verdict |
|---|---|---|---|
| MAIN turn 2/3 reuse | ≥99% | 99.952% / 99.952% (194,389/194,483; 194,490/194,583) | **pass** |
| 194K main turn TTFT | ≤10 s | restore + tail prefill 4.95 s / 4.53 s; HTTP turn total 14.77 s / 14.53 s (includes the STATE_HASH diagnostic) | **pass on the restore+tail definition; fail on the HTTP total**; **follow-up: 4.141 s without the hash, pass** (the 4.95 s double-counted the restore: the prompt-read timer includes it) |
| Restore finishes in seconds | seconds | 1.73 s / 1.48 s at 1,770 / 2,065 MiB/s | **pass** |
| Host save/restore peak | ≤512 MiB over the normal footprint | +2,990 MiB over the live 194K footprint (+5,611 MiB over idle); **follow-up: +2,889.9 MiB** over a clean live 41,950.8 MiB, restore +2,455.8, sustained +2,148.3 | **fail** |
| VRAM staging | ≤256 MiB | +172 MiB (disk run), +240 MiB (L1 run) | **pass** |
| Disk throughput reported | report it | save 1,463-1,932 MiB/s; restore 1,770-2,065 MiB/s (194K), 6,006 MiB/s (3.7K) | reported |
| Same next tokens / STATE_HASH | exact | 194K STATE_HASH identical (all fields) and text identical; 3.7K direct test 1 of 2 identical | **pass at 194K; small direct test flaky** |
| Default off unchanged | byte-identical | not re-run in this session (unit suites from the branch report are not repeated here) | not run |

## What failed, what was not run

- **sm_75 build crashed** on every prompt ≥7.8K (`prefill copy_i32: unspecified launch failure`). Rebuilt the same
  commit for sm_89; no further crash in any run. This is a build-architecture issue, not a code change.
- **512 MiB staging refused the first restore** (537 MiB needed for an 11.7K/395 MiB snapshot). Raised to 1024 MiB
  for all runs; this is a deviation from the brief and is visible in every config and log.
- **Host peak is about 3 GiB, not 512 MiB.** The 194K save/restore holds a multi-GiB image in the engine's working
  set. Measured, not inferred: baseline 39.24 GiB, live-194K 41.86 GiB, peak 44.85 GiB.
- **An 8 GiB RAM cache never reaches the disk tier at 194K** (the 3.62 GiB snapshot fits; `disk_saves=0`). The
  disk-only run was needed to exercise L2. A production default of 8192 therefore gives L1-only behaviour for
  conversations of this size.
- **The direct-engine parity test is not reproducible**: one IDENTICAL, one DIFFER at token 66 with a shorter
  restored prefix, under identical deterministic settings; the parked snapshot length itself varied (3,780 vs
  3,783 tokens). The brief's original command also gave DIFFER at 0, and with its exactness flags the engine's two
  greedy A answers themselves differed (different SHA-256), so the engine is not run-to-run reproducible without
  `STRATA_ROUTE_TAIL_SKIP=0` and `STRATA_PREFILL_CPU_SHARE=0`.
- **No timing run without `STRATA_STATE_HASH`.** The 7.7-9.6 s residual per 194K turn is attributed to the hash by
  arithmetic and scaling, not by an A/B timing measurement.
- **No SNAPSHOT_VERIFY line for disk restores** (see above).
- **HIP, SYCL and Metal were not built.** Only Windows CUDA was exercised.
- **Default-off byte-identity** was not re-measured in this session.

## Artifacts in this folder

- Reports: `disk-cache-smoke-report.json`, `disk-cache-200k-diskonly-report.json`,
  `disk-cache-200k-l1-only-refused.json` (the tool refused because L1 held everything).
- Engine logs: `disk-cache-smoke-engine.log`, `disk-cache-200k-l1-engine.log`,
  `disk-cache-200k-diskonly-engine.log`, `parity-200k-live-engine.log`, `parity-200k-disk-engine.log`.
- Server output: `disk-cache-smoke-server.out`, `disk-cache-200k-l1-server.out`,
  `disk-cache-200k-diskonly-server.out`, `parity-200k-live-server.out`, `parity-200k-disk-server.out`.
- Samplers: `disk-cache-200k-diskonly-sampler.csv`, `disk-cache-200k-l1-sampler.csv` (columns documented above).
- Parity: `parity-200k-live.json`, `parity-200k-disk.json`, `parking-parity.txt`,
  `parking-parity-engine-det.log`, `parking-parity-engine-run2.log`,
  `parity-engine-exact-flags-A-differed.log`.
- Configs: `disk-cache-smoke-config.json`, `disk-cache-200k-config.json`, `disk-cache-200k-diskonly-config.json`,
  `parity-200k-live-config.json`, `parity-200k-disk-config.json` (the two parity configs are reconstructions, as
  noted in the parity section).

Spill directories were left in place: `D:\strata-disk-cache-smoke` (397 MiB),
`D:\strata-disk-cache-200k` (3.0 GiB), `D:\strata-disk-cache-parity` (280 MiB),
`D:\strata-disk-cache-parity-200k` (3.0 GiB). All servers were stopped at the end; no `strata.exe` remained and the
GPU returned to 759 MiB used (desktop baseline).

## Follow-up (later session, 2026-10-26)

A later session re-measured the host cost with per-request sampler phases and a clean cache-off A/B, forced the L1
refusal with both the physical-RAM and the budget trigger, and timed the restored HTTP turn without
`STRATA_STATE_HASH`. The full report, configs, logs, raw `/metrics` blocks, sampler CSVs and client JSONs are in
**[disk-cache-200k-followup.md](disk-cache-200k-followup.md)**. Corrected numbers:

| Item | This report | Follow-up |
|---|---:|---:|
| Live 194K before the spill | 41,858.4 (parent plateau) | **41,950.8** (clean cache-off A/B) |
| First spill peak | 44,847.8 | **44,840.7** (reproduced within 7.1 MiB) |
| Spill cost over live | +2,990 | **+2,889.9** |
| Restore peak over live | not separated | **+2,455.8** (+461.5 over the post-spill level) |
| Sustained after the restored turn | "about 2.6 GiB" | **+2,148.3** |
| Staging 512 control | not run | peak **44,852.2**, spill still succeeds: the cost is **not** the staging copy |
| L1 enabled, refuses, disk takes it | not exercised (L1 held the 194K) | **both triggers pass**: B1 (16,000 MiB floor) and B2 (2,048 MiB L1 budget), 3 saves, 2 restores, 0 errors, 99.95% reuse; B2 keeps the small SUBs in L1 |
| Restored TTFT, no state hash | 14.77 s with the hash | **4.141 s** HTTP (prompt read 3,903 ms incl. restore 1,792 ms): **passes ≤10 s** |

The 512 MiB host verdict is unchanged: the first spill costs 2,889.9 MiB, 2,377.9 MiB over the bound. The restore
path's own transient (461.5 MiB) is inside the bound; the write side is not. The likely source is the streamed
3.0 GiB K/V write becoming resident, not the configured staging copy; the exact allocation was not isolated (no
profiler run).
