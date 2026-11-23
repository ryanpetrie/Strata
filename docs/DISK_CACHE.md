# Disk conversation cache

Strata keeps the last conversation's state, so a follow-up reads only what is new. With
`--conversation-cache-mib` it also keeps several conversations in RAM, and switching between them is quick.

A long conversation can be too big for RAM. The disk cache is a second place for those conversations: a conversation
the RAM cache cannot keep is written to disk, and read back when it returns. Then a long chat that takes turns with
small sub-agent requests does not read its whole prompt again.

The K/V can be kept as content-addressed pages in one shared pool (`--conversation-spill-pages`): a turn that
advanced a little then writes only the pages whose bytes changed. Without that flag the tier keeps one session
file per conversation, the earlier format.

There is one disk tier. Until version 0.2.20 the branch carried a second one with its own flags
(`--conversation-cache-disk-*`); it was folded into the spill tier and those names are now hidden aliases that print
a warning and map onto `--conversation-spill-*` ([work/local-spill-delta.md](work/local-spill-delta.md)).

The disk cache is **on by default** in a normal serve, with a cap of the smaller of 32 GiB and a quarter of the free
space. It has been measured on the reference host below; your model, context and drive will differ, so set a budget
you are comfortable with.

## Turn it on (or off)

The tier needs no flag in a normal serve. To cap it, or to move it, add engine arguments to the run config's `args`,
the same place as `--conversation-cache-mib`:

```json
"args": [
  "--conversation-spill-mib", "32768",
  "--conversation-spill-dir", "D:\\strata-cache"
]
```

- `--conversation-spill-mib` is the cap in MiB. Unset means min(32 GiB, a quarter of the free space); `0` turns the
  disk tier off.
- `--conversation-spill-dir` is the folder. Without it the tier uses `<data>/conversation-spill` when `--pack` is
  under `<data>/packs`. Use a fast local drive, not a network share or a folder that syncs to the cloud.
- Every engine writes into a folder of its own inside that root (`<pid>-<hex>`), with a lock, and cleans the folders
  of engines that ended. A crashed engine's files go at the next start.
- By default the entries go when the engine stops cleanly. `--conversation-spill-persist` keeps them across a restart
  in a stable folder per model fingerprint; a restart adopts them.

Optional: `--conversation-spill-only` keeps no conversations in RAM and uses only the disk tier; the K/V is then
streamed from the live session instead of through a host copy. It is slower for short conversations and useful for
testing. `--conversation-spill-slots N` limits how many conversations are kept.
`--conversation-spill-staging-mib N` is the host-memory admission bound for a save or a restore (default 1024); a
smaller value can refuse a restore, and the log says the limit by name. `--conversation-spill-pages` turns on the
shared page pool; without it each conversation is one session file, and `--conversation-spill-whole-file` says so
explicitly. Every conversation the RAM cache evicts is already written to the tier, so there is no separate "demote"
switch. `--conversation-spill-min-tokens`,
`--conversation-spill-park-gain` and `--conversation-spill-min-free-mib` bound what is written, as the RAM cache's own
minimum, park gain and floor do.

The pool is compacted when it has at least 2 GiB and 256 dead pages and they are at least a quarter of it
(`--conversation-spill-compact-min-mib`, `--conversation-spill-compact-min-pages`). The pool's format is versioned:
an upgrade that changes it discards the old pool and its entries, because a spill folder is a cache, not user data.

`--conversation-spill-flush-min MIN` writes conversations that grew since their copy at the engine's safe points,
through the tier's background writer, so a clean stop re-reads at most one interval's turns. It is off by default and
costs one host copy of each flushed conversation.

## What was measured

On a Windows host with an RTX 4090 (24 GB), 63.7 GiB RAM and a Samsung 980 PRO NVMe, with Qwen3.8-Flash-Next
UD-IQ4_XS and a 194.4K-token main conversation alternating with a small sub-agent conversation:

- 99.95% of each follow-up prompt was reused.
- A restore took 1.48-1.79 s at 1.77-2.07 GiB/s; the restored turn took 4.14 s end to end, against 166.6 s for the
  cold read.
- The file was 3.06 GiB. Saves wrote at 1.46-1.93 GiB/s.
- The cache's own private memory rose 357 MiB during the save. The process working set rose about 2.9 GiB while the
  3.0 GiB file was written, because the OS caches the file; that cache is reclaimable and the private count returned
  below the live level. VRAM rose 106-172 MiB.

The full measurements, including the cases where the RAM cache refused the park, are in
[work/evidence/disk-cache-200k.md](work/evidence/disk-cache-200k.md). They used the one-session-file format
(`--conversation-spill-whole-file`); the shared page pool's write amplification and restore speed have not been
measured yet. They are one model on one drive, not a promise for another configuration.

## What it costs in disk

Budget by the size of the conversations you keep. The measured 194.4K-token conversation above wrote 3.06 GiB; the
implementation plan's measured 75,659-token snapshot was 1,891.4 MiB, about 25 KB per token. That is one model and one
KV format: measure your own model and context, then set the budget above what you expect to need. The engine also
keeps 4 GiB free on the drive by default (`--conversation-spill-min-free-mib`).

The shared pool stores a page once however many conversations use it, and a turn rewrites only the changed pages. The
budget counts what each conversation names, not the pool's physical size; the pool can hold a little more until it is
compacted. That saves SSD writes but has not been measured on the reference host yet; a save's log line reports the
bytes written and reused.

When the budget is full, the oldest conversation goes first. A save that does not fit is refused, and the conversations
already on disk stay. A conversation that is not worth the round trip (a short one, whose prompt is cheaper to read
again) is not written at all: the tier measures the engine's own prompt and disk speeds and compares them.

Each entry keeps the conversation's live state and its deepest checkpoint. It does not keep every interior checkpoint
yet, so a new conversation cannot start from a middle checkpoint of a disk entry; it can resume at the deepest one.

## Privacy

The files are plaintext. They hold the engine's state and token identity, not an encrypted archive; anyone who can read
the folder can read the conversation's tokens. The folder is created for your user only (0700, files 0600 on Linux and
macOS; on Windows it inherits your user profile's permissions). Keep it private, do not put it on a shared drive, and
delete the folder to clear the cache. The page pool, the metadata heads and the entries are all in that folder.

## See it in the Monitor

Open **Monitor** and look at the **Conversation cache** card. When the disk tier is on, a second block shows:

- **Conversations on disk** and **Their memory (disk)**, against the slot count (or "unlimited") and the budget.
- **Disk saved / restored**, **Disk evictions**, **Last disk restore** (milliseconds and MiB/s), and **Refused saves**
  with the reasons in plain words (RAM, budget, min tokens, disk space, model or settings).
- **Last disk write**: the MiB written and reused by the last save, and the share that was new.
- **Page pool**: the pool's size and how many compactions it has had.

The same values are in `/metrics` under `"conversation_cache"`, as `disk_enabled`, `disk_budget_mib`, `disk_dir`,
`disk_slots`, `disk_bytes`, `disk_entries`, `disk_saves`, `disk_restores`, `disk_evictions`, `disk_refusals`,
`disk_errors`, `disk_last_save_ms`, `disk_last_restore_ms`, `disk_last_restore_mib_s`, `disk_paged`,
`disk_pool_mib`, `disk_last_write_mib`, `disk_last_reuse_mib`, `disk_last_write_share`, `disk_written_mib`,
`disk_reused_mib`, `disk_compactions`, `disk_compacted_mib`, `last_tier`, `refusals` and `last_refusal`.

See [DETAILS.md](DETAILS.md#disk-tier-l2) for every argument, and [TEST_REQUESTS.md](TEST_REQUESTS.md) if you want to
contribute a measurement.
