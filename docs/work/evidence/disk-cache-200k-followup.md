# Disk conversation cache — follow-up: phase-aligned host cost, L1 refusal, no-hash TTFT

Date: 2026-10-26 (UTC; local timezone UTC-06:00), later session than
[disk-cache-200k.md](disk-cache-200k.md).
Commit: `07f2b9fb` (`feat/disk-conversation-cache`), engine `0.1.51`, `build-cuda\strata.exe` built for
`CMAKE_CUDA_ARCHITECTURES=89` (same binary as the parent report; `--version` 0.1.51).
Host: Windows, NVIDIA RTX 4090 24 GB, 63.7 GiB RAM, CUDA 13.4, MSVC 14.51.
Model: Qwen3.8-Flash-Next, pack `unsloth-ud-iq4_xs` (`C:\tools\Strata-data`).
Spill drive: `D:` Samsung SSD 980 PRO 2TB, NVMe, NTFS.

This file contains only numbers produced in this session. The parent report's runs are referred to as
"the parent run". No engine source was changed. No `STRATA_STATE_HASH` was set in any run below (it was removed from
the config's `env` and popped from the server environment).

## Bottom line

| Question | Measured answer |
|---|---|
| A. Is +2,990 MiB real? | **Real, not a sampling artifact.** Clean A/B: live 194K 41,950.8 MiB; spill peak 44,840.7 (**+2,889.9**); restore peak 44,406.6 (**+2,455.8**); sustained 44,099.1 (**+2,148.3**). The parent peak 44,847.8 reproduced within 7.1 MiB; only its live reference was 92 MiB low. |
| A. Is it the staging copy? | **No.** With `--conversation-cache-disk-staging-mib 512` the 194K spill still succeeded and the peak was 44,852.2 (**+2,901.4**, 11.5 MiB above the 1,024 MiB run). The explicit save copy fits in 512 MiB. |
| B. L1 enabled, L1 refuses, disk takes it? | **Yes, both triggers.** B1 (RAM floor 16,000 MiB): saves 3, restores 2, reuse 99.95%, errors 0. B2 (L1 2,048 MiB): saves 3, restores 2, reuse 99.95%, errors 0. B2 is the cleaner product path: L1 keeps the 1.4K/1.7K SUBs, disk holds the 194K. |
| C. Restored TTFT without the hash? | **4.141 s** HTTP for MAIN 2 (prompt read 3,903 ms, of which restore 1,791.8 ms). **Passes ≤10 s.** The parent's 14.77 s was the `STRATA_STATE_HASH` diagnostic. |
| 512 MiB host bound | **Fail**, by **2,377.9 MiB** at the spill peak (2,889.9 − 512). Even the retained working set after the turn is 2,148.3 MiB over live. |

## Method

The parent sampler CSV had only `baseline` and `run` markers, so its requests could not be separated. For this
session the sampler was driven by a phase file written by the client (`build-cuda/followup_harness.py`), one label
per request (`main1`, `after_main1`, `sub1`, `after_sub1`, `main2`, `after_main2`, `sub2`, `after_sub2`). The
sampler records the label on every tick, so the windows below are exact, not inferred from timestamps.

Runs (all on port 8099, one engine at a time, a fresh spill folder for each):

| Run | Config | Client | Purpose |
|---|---|---|---|
| baseline | `followup-baseline.json` (RAM cache 0, no disk flags) | harness, MAIN 1 + SUB 1 | live 194K footprint with no cache at all |
| disk | `followup-disk-nohash.json` (RAM cache 0, disk-only, staging 1,024) | harness, 2 turns (MAIN/SUB twice) | A + C |
| staging512 | `followup-staging512.json` (same, staging 512) | harness, MAIN 1 + SUB 1 | is the peak the staging copy? |
| b1 | `followup-b1-ram-refusal.json` (L1 8,192, min-free 16,000, disk on) | `tools/conversation_disk_cache_test.py --turns 3` | physical-RAM refusal |
| b2 | `followup-b2-budget-refusal.json` (L1 2,048, min-free 2,560, disk on) | same tool, `--turns 3` | budget refusal |

All runs: `--main-words 50000 --step-words 20 --sub-words 300`, 194,382-token MAIN 1, `--key REDACTED`,
`--resident-budget-gib 36`, `--max-context 262144`, `--kv int8 --kv-resident 32768`, `--vision`.
The server was started with the same command as the parent:
`python -m serve.server --engine strata --config <config> --host 127.0.0.1 --port 8099`.
Each run's `run.json` records the UTC start, ready time, client exit, server exit and the post-stop process check.

## A. Phase-aligned host working set

Sampler: `build-cuda/sample_memory.py`, 250 ms requested (about 0.36 s real cadence), columns as in the parent.
Unit: MiB of `WorkingSet64` for `strata.exe`.

### Clean A/B

| Run | Point | UTC | Engine WS | Private | Free physical | GPU |
|---|---|---:|---:|---:|---:|---:|
| baseline | idle after READY | 13:54:57.398 | 39,237.3 | 66,026.9 | 8,833.8 | 22,653 |
| baseline | **end of MAIN 1 (live 194K)** | 13:57:35.589 | **41,950.8** | 68,084.8 | 7,175.4 | 22,803 |
| baseline | SUB 1 peak (no cache, no spill) | 13:57:44.296 | 41,405.0 | 67,522.6 | 7,705.4 | 22,803 |
| disk | idle after READY | 13:58:57.918 | 39,224.1 | 66,031.0 | 8,777.8 | 22,585 |
| disk | **end of MAIN 1 (pre-spill live)** | 14:01:49.765 | **41,939.1** | 68,089.1 | 6,719.1 | 22,799 |
| disk | **SUB 1 peak (spill)** | 14:01:53.372 | **44,840.7** | 68,446.2 | 4,348.4 | 22,799 |
| disk | after SUB 1 (post-spill) | 14:02:01.461 | 43,945.1 | 67,532.8 | 7,323.0 | 22,799 |
| disk | **MAIN 2 peak (restore)** | 14:02:03.858 | **44,406.6** | 67,996.1 | 6,991.0 | 22,819 |
| disk | **after MAIN 2 (sustained)** | 14:02:07.808 | **44,099.1** | 67,682.4 | 7,195.4 | 22,819 |
| disk | SUB 2 peak (second spill) | 14:02:11.102 | 44,456.1 | 68,038.7 | 4,415.8 | 22,819 |
| disk | after SUB 2 | 14:02:16.037 | 44,003.3 | 67,575.2 | 7,299.4 | 22,819 |

Exact CSV rows behind the key points (`disk-cache-200k-followup-disk-sampler.csv`; omitted columns are
`engine_pid`, `all_strata_*` and `strata_processes`):

```
utc                     phase        engine_ws_mib  engine_private_mib  free_physical_mib  gpu_used_mib
2026-10-26T14:01:49.765Z after_main1       41939.1             68089.1             6719.1         22799
2026-10-26T14:01:51.554Z sub1              41994.9             68202.3             6506.6         22799
2026-10-26T14:01:53.372Z sub1              44840.7             68446.2             4348.4         22799
2026-10-26T14:02:01.461Z after_sub1        43945.1             67532.8             7323.0         22799
2026-10-26T14:02:03.858Z main2             44406.6             67996.1             6991.0         22819
2026-10-26T14:02:07.808Z after_main2       44099.1             67682.4             7195.4         22819
2026-10-26T14:02:11.102Z sub2              44456.1             68038.7             5417.7         22819
2026-10-26T14:02:16.037Z after_sub2        44003.3             67575.2             7299.4         22819
```

Baseline rows (`disk-cache-200k-followup-baseline-sampler.csv`):

```
2026-10-26T13:54:57.398Z baseline          39237.3             66026.9             8833.8         22653
2026-10-26T13:57:35.589Z main1             41950.8             68084.8             7175.9         22803
2026-10-26T13:57:37.375Z after_main1       41950.8             68084.8             7175.4         22803
2026-10-26T13:57:44.296Z sub1              41405.0             67522.6             7705.4         22803
```

Deltas over the clean live footprint (41,950.8 MiB):

| Window | Peak | Delta over live | Delta over the window's own base |
|---|---:|---:|---:|
| End of MAIN 1, disk vs baseline live | 41,939.1 vs 41,950.8 | **−11.7** | the two live measurements agree within 0.03% |
| SUB 1, first spill | 44,840.7 | **+2,889.9** | +2,901.6 over the disk run's own 41,939.1 |
| MAIN 2, restore | 44,406.6 | **+2,455.8** | **+461.5** over the post-spill 43,945.1 |
| After MAIN 2, sustained | 44,099.1 | **+2,148.3** | +154.0 over the post-spill level |
| SUB 2, second spill | 44,456.1 | +505.3 | **+357.0** over the sustained level |

The exact CSV rows for the peak and the windows are in
`disk-cache-200k-followup-disk-sampler.csv` (phase labels) and the client's per-request UTC windows are in
`disk-cache-200k-followup-disk-client.json`.

By the brief's literal "SUB-window peaks, disk vs baseline" comparison the spill costs
44,840.7 − 41,405.0 = **+3,435.7 MiB**. That is not the cache's own cost: the baseline SUB window is 545.8 MiB
**below** the live MAIN footprint because the uncached SUB request drops MAIN's state. The live-referenced figure,
**+2,889.9 MiB**, is the smaller and the correct one for the 512 MiB bound. Even that smaller figure is 5.6× the
bound; the literal SUB-window comparison is 6.7×.

The baseline arm has no MAIN 2 (cache off, so no restore), so the restore cost is measured against the live
footprint (**+2,455.8 MiB**) and against the disk run's own post-spill level (**+461.5 MiB**). The second number is
the restore operation's own transient; the first is what it adds to the conversation's footprint.

### Why the parent's live reference was 92 MiB low

The parent CSV has no request labels. Aligning it by the run marker (13:20:56), the logged 151,277 ms prompt read
and the 44,847.8 peak gives this sequence:

```
13:23:25.065  41,179.0   last sample before the last pre-spill jump
13:23:26.268  41,228.2   last pre-spill jump starts
13:23:27.159  41,838.8   flat until 13:23:29.853  <-- the live plateau
13:23:29.552  41,858.4   the value the parent report used (still on the plateau)
13:23:30.152  41,916.6   spill ramp starts
13:23:33.463  42,532.3
13:23:39.477  43,873.8
13:23:42.762  44,847.8   peak (spill)
13:23:44.863  43,830.4   post-spill
```

The 41,858.4 MiB was a real 2.7 s plateau, not a mid-ramp sample; the parent wording "still climbing" was
inaccurate. It is 92.4 MiB below the clean A/B live (41,950.8), which is run-to-run variation (the parent run also
had `STRATA_STATE_HASH=1`, this one does not). The peak itself is 7.1 MiB below this session's 44,840.7. **The
+2,990 MiB was therefore real cache cost; the corrected figure is +2,889.9 MiB, not a sampling artifact.**

### Is the peak the staging copy? No (512 MiB control)

`followup-staging512.json` changes only `--conversation-cache-disk-staging-mib` from 1024 to 512. The log still
shows a successful spill:

```
strata serve: conversation cache: disk tier on: dir=D:\strata-disk-cache-followup-staging512 budget=32768 MiB
  slots=0 staging=512 MiB min_free=4096 MiB entries=0
strata serve: conversation cache: spilled 194389 tokens to disk in 2338.5 ms; file=3056 MiB tier=disk
  entries=1 bytes=3204882488
```

| Run | Live after MAIN 1 | Spill peak | Delta | Post-spill |
|---|---:|---:|---:|---:|
| disk, staging 1,024 | 41,939.1 | 44,840.7 | +2,901.6 | 43,945.1 |
| staging 512 | 41,950.6 | 44,852.2 | +2,901.4 | 43,959.1 |
| difference | +11.5 | +11.5 | +0.2 | +14.0 |

A 512 MiB cut in the staging limit changed the peak by 11.5 MiB. The 2.9 GiB is **not** the configured staging
copy, and it is not the checkpoint copy: the save's own admission (`session_save_peak_bytes`: write block + one
deepest checkpoint copy + live state + token list) fits in 512 MiB, or the spill would have been refused.

### What the host cost is, and what it is not

Evidence pointing to the streamed K/V write path, stated with its limits:

- The peak is **+2,901.6 MiB** over the disk run's own live 41,939.1, and the first file is **3,056.4 MiB**
  (3,204,882,488 bytes); later saves are 3,057.9 and 3,059.3 MiB. The peak is 94.9% of the file size.
- The working-set ramp follows the write rate: in the disk run it rises from 41,994.9 to 44,840.7 MiB over about
  1.8 s, about 1,565 MiB/s; the logged write is 2,393.9 ms at about 1,339 MiB/s.
- **Private bytes rise only +357.1 MiB** at the peak (68,089.1 → 68,446.2). A new 2.9 GiB private allocation
  would show in `PrivateMemorySize64`; it does not.
- A control 3 GiB `WriteFile` in a separate Python process raised that process's working set by **4.0 MiB**, so
  Windows does not charge an ordinary file write to the writer's working set. The engine's rise is engine memory,
  not file-cache accounting.
- The cost is one-time: the second spill (SUB 2, same 3.0 GiB file) adds only **+357.0 MiB** over the sustained
  level. The first spill leaves about 2.0 GiB resident; later spills reuse it.
- The metadata is far too small: the manifest is 1,556,781 bytes (1.5 MiB) and the id list is 194,389 × 4 =
  777,556 bytes (0.74 MiB). The peak cannot be "the metadata".

What was **not** isolated: the exact allocation or mapping behind the 2.9 GiB. The source streams the K/V
(`Out::source` copies from the live K/V pools straight into a 16 MiB file block; the reader reads one layer at a
time), so a whole-image `SavedConversation` is not built. The most likely explanation is that the streamed K/V
transfer makes roughly the file's size of host pages resident (the pools and/or a CUDA transfer staging buffer) and
that the first spill pays for it. This is consistent with the ramp, the size match, the flat private bytes and the
one-time behaviour, but it is **static analysis plus the sampler, not an allocation trace**. No profiler or heap
trace was run. The report does not claim more than that.

### Corrected 512 MiB verdict

The criterion is "host save/restore peak ≤512 MiB over the normal footprint".

| Quantity | Value | Against 512 MiB |
|---|---:|---|
| First spill peak over live | **2,889.9 MiB** (44,840.7 − 41,950.8) | **fail by 2,377.9 MiB** (5.64×) |
| Restore peak over live | 2,455.8 MiB | fail by 1,943.8 MiB |
| Restore transient over the post-spill level | **461.5 MiB** | **pass** (90.1% of 512) |
| Sustained after MAIN 2 | 2,148.3 MiB | fail by 1,636.3 MiB |
| Second spill transient | 357.0 MiB | pass |
| VRAM peak over idle | +234 MiB (22,819 − 22,585) | pass (≤256) |
| Free physical minimum | **3,388.9 MiB** (disk run; parent reported 3,928.7) | the host did not OOM or swap |

So the read (restore) side is already inside the 512 MiB bound: its transient is 461.5 MiB, and the code reads one
layer at a time. The **write (spill) side is not**: it costs 2.9 GiB on the first spill and leaves 2.0 GiB resident.
The parent's 512 MiB verdict stands, with the number corrected from 2,990 to 2,890 MiB and the mechanism
narrowed to the write stream, not the configured staging limit.

## B. L1 enabled, L1 refuses, disk takes it

Both runs used the real tool, `tools/conversation_disk_cache_test.py --turns 3 --main-words 50000 --step-words 20
--sub-words 300 --key REDACTED`. Both passed.

### B1 — physical-RAM admission (`--conversation-cache-min-free-mib 16000`)

L1 = 8,192 MiB, disk = 32,768 MiB, staging 1,024 MiB, `--conversation-cache-min-free-mib 16000`,
`--resident-budget-gib 36` unchanged. The 194K snapshot is about 3,622 MiB, which fits the 8,192 MiB L1 budget; the
host floor is what refuses it. On this host MemAvailable is about 7.5 GiB at park time, far below 3.6 + 16 GiB.

Exact log lines (engine log `disk-cache-200k-followup-b1-engine.log`):

```
strata serve: conversation cache: spilled 194389 tokens to disk in 2320.8 ms; file=3056 MiB tier=disk entries=1 bytes=3204882488
strata serve: conversation cache: skip parking (physical RAM admission; need 245 MiB plus 16000 MiB floor, 7548 MiB available, source meminfo; evicted 0, 0 still parked)
strata serve: conversation cache: restored 194389 tokens from disk in 1828.9 ms (1671 MiB/s); tier=disk parked=0 bytes=0
strata serve: conversation cache: spilled 194490 tokens to disk in 1867.6 ms; file=3057 MiB tier=disk entries=1 bytes=3206409696
strata serve: conversation cache: skip parking (physical RAM admission; need 250 MiB plus 16000 MiB floor, 7465 MiB available, source meminfo; evicted 0, 0 still parked)
strata serve: conversation cache: restored 194490 tokens from disk in 1568.4 ms (1949 MiB/s); tier=disk parked=0 bytes=0
strata serve: conversation cache: spilled 194587 tokens to disk in 1619.6 ms; file=3059 MiB tier=disk entries=1 bytes=3207875828
```

**The exact sequence the brief expected, "skip parking (physical RAM admission)" then "spilled ... to disk", is
not the line order for the 194K conversation here.** The 194K park takes the *pre-check* in
`src/program/generate.cpp` (lines 9904-9911): when `conversation_memory_admit` fails and the disk tier is on, the
code calls `park_to_disk()` and returns **before** the later admission loop that prints the skip line. So the 194K
spill at SUB 1 is logged only as `spilled ... to disk`. The two `skip parking (physical RAM admission)` lines above
are the outgoing 1,414- and 1,729-token SUB conversations at the start of MAIN 2 and MAIN 3, when a disk restore is
in flight (the code reserves the tier for the restore, so the small conversation is dropped, not spilled). This is
static reading of the current source plus the log order; the refusal is also visible in `/metrics`
(`refusals.ram = 2`, `last_refusal = "physical RAM admission"`).

The 194K path is nevertheless unambiguously on the disk tier: three 194K spills, two 194K restores, both with
`tier=disk`, and the `refusals` block. The L1 never held the 194K conversation (disk-only `bytes=0` on the restore
lines; the previous L1 run with `min-free 2560` parked it in RAM and had `disk_saves=0`).

`/metrics` after (`disk-cache-200k-followup-b1-report.json`):

```
disk_saves=3  disk_restores=2  disk_errors=0  disk_evictions=0  disk_refusals=0
disk_entries=1  disk_bytes=3,207,875,828  last_tier=disk
refusals: ram=2, budget=0, min_tokens=0, disk_space=0, identity=0, disk_other=0
last_refusal="physical RAM admission"
```

Host/VRAM peaks (sampler `disk-cache-200k-followup-b1-sampler.csv`, whole run): idle 39,237.5 MiB; peak engine
working set **44,856.1 MiB** at 14:07:45.703; peak private 68,461.9 MiB; free physical minimum **4,009.7 MiB**; GPU
peak **22,835 MiB** (+120 over the 22,715 idle). The peak is +2,905.3 MiB over the clean live 41,950.8, i.e. the
same first-spill cost as the disk-only run; L1 did not add a second 194K snapshot.

### B2 — L1 budget refusal (`--conversation-cache-mib 2048`)

L1 = 2,048 MiB, disk = 32,768 MiB, staging 1,024 MiB, `--conversation-cache-min-free-mib 2560` (the parent value),
`--resident-budget-gib 36` unchanged. The 194K snapshot (3,622 MiB) is larger than the L1 budget, so `make_room`
refuses and the disk takes it. This is the literal sequence the brief asked for, twice:

```
strata serve: conversation cache: skip parking (snapshot 3622 MiB exceeds available budget)
strata serve: conversation cache: spilled 194389 tokens to disk in 2377.5 ms; file=3056 MiB tier=disk entries=1 bytes=3204882488
strata serve: conversation cache: parked 1418 tokens in 51.6 ms; parked=1 bytes=257817084 evictions=0 snapshot_bytes=257817084 reused_kv_bytes=0
strata serve: conversation cache: restored 194389 tokens from disk in 1751.8 ms (1745 MiB/s); tier=disk parked=1 bytes=257817084
strata serve: conversation cache: skip parking (snapshot 3171 MiB exceeds available budget)
strata serve: conversation cache: spilled 194490 tokens to disk in 1990.3 ms; file=3057 MiB tier=disk entries=1 bytes=3206409696
strata serve: conversation cache: restored 1418 tokens (live) in 46.6 ms; parked=0 bytes=21679224
strata serve: conversation cache: parked 1733 tokens in 77.0 ms; parked=1 bytes=382587136 evictions=0 snapshot_bytes=382587136 reused_kv_bytes=21613824
strata serve: conversation cache: restored 194490 tokens from disk in 1605.1 ms (1905 MiB/s); tier=disk parked=1 bytes=382587136
strata serve: conversation cache: skip parking (snapshot 3172 MiB exceeds available budget)
strata serve: conversation cache: spilled 194588 tokens to disk in 1618.4 ms; file=3059 MiB tier=disk entries=1 bytes=3207881976
strata serve: conversation cache: restored 1733 tokens (live) in 53.1 ms; parked=0 bytes=28377720
```

This is the full product policy in one run: L1 **parked** the 1,418- and 1,733-token SUB conversations (51.6 and
77.0 ms), **restored** them (46.6 and 53.1 ms), and **refused** the 194K snapshot on budget, which the disk tier
then **spilled** and **restored**. The small SUB reuse shows up as `reused 1418` and `reused 1733` on SUB 2/3.

`/metrics` after (`disk-cache-200k-followup-b2-report.json`):

```
disk_saves=3  disk_restores=2  disk_errors=0  disk_evictions=0  disk_refusals=0
disk_entries=1  disk_bytes=3,207,881,976  last_tier=l1
refusals: ram=0, budget=3, min_tokens=0, disk_space=0, identity=0, disk_other=0
last_refusal="snapshot 3172 MiB exceeds available budget"
```

`last_tier=l1` is correct: the final restore of the run is the small SUB from L1; the two 194K restores are the
`tier=disk` lines above.

Host/VRAM peaks (sampler `disk-cache-200k-followup-b2-sampler.csv`, whole run): idle 39,237.0 MiB; peak engine
working set **44,965.1 MiB** at 14:12:37.461; peak private 68,543.3 MiB; free physical minimum **3,639.1 MiB**; GPU
peak **22,821 MiB** (+106 over the 22,715 idle). The peak is +3,014.3 MiB over live, 109.0 MiB above B1's
+2,905.3 and 124.4 MiB above the disk-only +2,889.9; the difference is smaller than the two L1 snapshots (246 and
365 MiB) that were parked at the time, but the L1 copy is the only extra state in this run.

### Reuse and HTTP, both B runs

| Run | Turn | Prompt | Reused | Reused % | HTTP total s | Restore ms |
|---|---|---:|---:|---:|---:|---:|
| B1 | MAIN 2 | 194,483 | 194,389 | **99.952** | 4.596 | 1,828.9 |
| B1 | MAIN 3 | 194,583 | 194,490 | **99.952** | 3.327 | 1,568.4 |
| B2 | MAIN 2 | 194,483 | 194,389 | **99.952** | 4.699 | 1,751.8 |
| B2 | MAIN 3 | 194,583 | 194,490 | **99.952** | 4.390 | 1,605.1 |

Both runs: `disk_saves = 3`, `disk_restores = 2`, `disk_errors = 0`, MAIN 2/3 reuse ≥ 99.95%, no
`disk restore failed`, no `skip disk save`, no `disk evicted`. All assertions the brief asked for pass.

## C. TTFT without `STRATA_STATE_HASH`

The disk-only run (`followup-disk-nohash.json`) had `STRATA_STATE_HASH` removed from the config and popped from the
server environment; `STRATA_SNAPSHOT_VERIFY=1` stayed. Engine log
`disk-cache-200k-followup-disk-engine.log`:

```
strata serve: prompt 194382 tokens = 0 reused + 194382 read in 166286 ms (1169.0 tok/s), 8 generated in 176 ms (45.6 tok/s), drafts accepted 5 of 6, 6 checkpoints
strata serve: conversation cache: spilled 194389 tokens to disk in 2393.9 ms; file=3056 MiB tier=disk entries=1 bytes=3204882488
strata serve: prompt 1414 tokens = 0 reused + 1414 read in 9735 ms (145.2 tok/s), 3 generated in 157 ms (19.1 tok/s), drafts accepted 3 of 3, 1 checkpoints
strata serve: conversation cache: restored 194389 tokens from disk in 1791.8 ms (1706 MiB/s); tier=disk parked=0 bytes=0
strata serve: prompt 194483 tokens = 194389 reused + 94 read in 3903 ms (24.1 tok/s), 8 generated in 202 ms (39.7 tok/s), drafts accepted 4 of 5, 2 checkpoints
strata serve: conversation cache: spilled 194490 tokens to disk in 1919.7 ms; file=3057 MiB tier=disk entries=1 bytes=3206409696
strata serve: prompt 1729 tokens = 0 reused + 1729 read in 6106 ms (283.2 tok/s), 3 generated in 92 ms (32.6 tok/s), drafts accepted 3 of 3, 1 checkpoints
```

| Turn | Prompt | Reused % | **HTTP total s** | Engine prompt read ms | Restore ms | Decode ms |
|---|---:|---:|---:|---:|---:|---:|
| MAIN 1 (cold) | 194,382 | 0 | **166.550** | 166,286 | — | 176 |
| SUB 1 (parks MAIN) | 1,414 | 0 | 9.910 | 9,735 (includes the 2,393.9 ms spill) | — | 157 |
| **MAIN 2 (restored)** | 194,483 | **99.952** | **4.141** | **3,903** (includes the restore) | **1,791.8** | 202 |
| SUB 2 (parks MAIN 2) | 1,729 | 0 | 6.202 | 6,106 (includes the 1,919.7 ms spill) | — | 92 |

The HTTP total of MAIN 2 is 4.141 s; the engine's prompt read is 3.903 s, the restore is 1.792 s and the 8-token
decode is 0.202 s. The prompt read **includes** the restore (3.903 − 1.792 = 2.111 s of tail prefill), which is why
restore + prompt read is larger than the HTTP total; the previous report's "restore + tail prefill = 4.95 s"
double-counted that overlap. The first-token latency is therefore **at most 4.14 s** on the actual HTTP path,
well inside the **≤10 s** Phase 1 target.

The parent's 14.77 s / 14.53 s for MAIN 2/3 included the state-hash pass. Removing it removes about **10.6 s**
(14.77 → 4.14). The parent's estimate of 7.7-8.3 s was low because its phase arithmetic added the end-of-turn spill
(1.92 s) to the MAIN 2 phase sum; the spill is actually charged to the **next** request (the metrics after MAIN 2
still show `disk_saves=1`, and it becomes 2 only after SUB 2). The no-hash measurement is the reliable figure:
**4.141 s**.

Cold MAIN 1 is not a TTFT target and varies with prefill: 154.3 s (staging512), 157.9 s (baseline, cache off),
158.3 s (B1), 166.6 s (disk), 168.6 s (B2). The 8.6 s difference between the baseline and disk cold turns is
prefill variance, not a cache effect; the restored turns are the only ones that matter for the target.

The B runs repeat the restored-turn result without the hash at 3.3-4.7 s (table in section B), so 4.141 s is not a
one-off.

## Phase 1 exit criteria, corrected

| Criterion | Target | Measured in this session | Verdict |
|---|---|---|---|
| MAIN turn 2/3 reuse | ≥99% | 99.952% / 99.952% in disk, B1 and B2 | **pass** |
| 194K restored TTFT | ≤10 s | **4.141 s** HTTP (disk, no hash); 3.327-4.699 s in B1/B2 | **pass** |
| Restore finishes in seconds | seconds | 1.568-1.829 s at 1,671-1,949 MiB/s | **pass** |
| Host save/restore peak | ≤512 MiB over live | **+2,889.9 MiB** spill, +2,455.8 restore, +2,148.3 sustained; restore **transient** +461.5 | **fail** (spill/sustained); restore transient passes |
| VRAM staging | ≤256 MiB | +234 MiB (disk), +120 (B1), +106 (B2) | **pass** |
| Disk throughput | report it | save **1,277-1,890 MiB/s** at 194K (first saves 2,320.8-2,393.9 ms; later saves 1,619.6-1,990.3 ms); restore 1,671-1,949 MiB/s | reported |
| L1 enabled, refuses, disk takes it | works | B1 and B2 both: 3 saves, 2 restores, 0 errors, 99.95% reuse | **pass** |
| Default off unchanged | byte-identical | not re-run | not run |

The spill/restore throughput differs slightly from the parent (save 1,463-1,932 MiB/s; restore 1,770-2,065 MiB/s)
because the parent's save timings included different host conditions; these are same-host, same-file-size
measurements and are not a speed claim. The file sizes match the parent to the byte for the first save
(3,204,882,488 bytes).

## What failed or was not run

- **The parent's "live 194K" label was wrong, not the peak.** 41,858.4 MiB was a 2.7 s plateau 92.4 MiB below the
  clean cache-off live (41,950.8). The corrected cost is +2,889.9 MiB, not +2,990; the peak reproduced within
  7.1 MiB (44,840.7 vs 44,847.8). The 512 MiB verdict does not change.
- **The physical-RAM skip line is not printed for the 194K park in B1.** The 194K takes the pre-check and spills
  before the skip-line code path; the visible `skip parking (physical RAM admission)` lines are the small outgoing
  SUBs. The 194K refusal is visible in `refusals` only indirectly (it does not increment `refusals.ram`; that count
  is 2, both SUBs). B2 prints the budget skip line for the 194K as expected. If the log line is a requirement, the
  pre-check needs its own line; that is a code change, not made here.
- **The exact 2.9 GiB allocation was not isolated.** The staging 512 control rules out the configured staging
  copy and the checkpoint copy; the write-rate ramp, file-size match, flat private bytes and one-time behaviour
  point to the streamed K/V transfer path. No profiler or heap trace was run, so the mechanism is stated as the
  likely source, not proven.
- **512 MiB restore at 194K was not re-run.** The 512 run exercised the save only. The parent's 11.7K restore
  refusal at 512 MiB (needed 537 MiB) is still the evidence that 512 is not a viable restore setting at this
  conversation size.
- **Cold prefill varied by 14.4 s across runs** (154.0-168.4 s) with the same prompt and config family. This is
  host/SSD/page-cache variation, not the cache; the restored turns are stable at 3.3-4.7 s.
- **Free physical reached 3,388.9 MiB** in the disk run (the parent reported 3,928.7). The host did not OOM or
  swap, but the margin during a spill is smaller than the parent's minimum suggests.
- **No HIP, SYCL or Metal build; no default-off byte-identity run** in this session.
- **No second engine was run at once.** Each run's `run.json` records `strata_pids_after: []`; the GPU returned to
  the desktop baseline after each run.

## Artifacts

In this folder:

- Report: `disk-cache-200k-followup.md`.
- Configs: `disk-cache-200k-followup-{baseline,disk,b1,b2,staging512}-config.json`.
- Engine logs: `disk-cache-200k-followup-{baseline,disk,b1,b2,staging512}-engine.log`.
- Server output: `disk-cache-200k-followup-{baseline,disk,b1,b2,staging512}-server.out`.
- Samplers: `disk-cache-200k-followup-{baseline,disk,b1,b2,staging512}-sampler.csv` (phase column, 250 ms request).
- Client reports: `disk-cache-200k-followup-baseline-client.json`,
  `disk-cache-200k-followup-disk-client.json`, `disk-cache-200k-followup-b1-report.json`,
  `disk-cache-200k-followup-b2-report.json`, `disk-cache-200k-followup-staging512-client.json`.
- Run summaries (UTC windows, exits, post-stop process check): `disk-cache-200k-followup-*-run.json`.
- Harness and runner: `disk-cache-200k-followup-harness.py`, `disk-cache-200k-followup-run.py`,
  `disk-cache-200k-followup-make-configs.py`, `disk-cache-200k-followup-make-staging512-config.py`.

Spill folders left in place: `D:\strata-disk-cache-followup-disk`, `-b1`, `-b2`, `-staging512` (about 3.0 GiB
each; `-b1` and `-b2` hold one conversation file each, the later turn supersedes the earlier one).

## Summary for the parent report

- Corrected host cost: **+2,889.9 MiB** at the first spill peak (was +2,990), **+2,455.8** at the restore peak,
  **+2,148.3** sustained. The 512 MiB bound is missed by 2,377.9 MiB at the peak; the restore's own transient is
  461.5 MiB and does meet it.
- The peak is real: 44,840.7 MiB, within 7.1 MiB of the parent's 44,847.8. The staging 512 control (44,852.2)
  rules out the staging copy.
- L1 refusal works: B1 (16,000 MiB floor) and B2 (2,048 MiB L1 budget) both spill and restore the 194K with
  99.95% reuse, 3 saves, 2 restores, 0 errors. B2 is the cleaner product shape: L1 keeps the small SUBs, disk
  takes the 194K.
- No-hash restored TTFT: **4.141 s** HTTP (prompt read 3,903 ms, restore 1,792 ms); **passes ≤10 s**. The parent's
  14.77 s was the state-hash diagnostic.
