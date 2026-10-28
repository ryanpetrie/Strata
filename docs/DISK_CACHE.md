# Disk conversation cache

Strata keeps the last conversation's state, so a follow-up reads only what is new. With
`--conversation-cache-mib` it also keeps several conversations in RAM, and switching between them is quick.

A long conversation can be too big for RAM. The disk cache is a second place for those conversations: a conversation
the RAM cache cannot keep is written to disk, and read back when it returns. Then a long chat that takes turns with
small sub-agent requests does not read its whole prompt again.

The K/V can be kept as content-addressed pages in one shared pool (`--conversation-cache-disk-cow`): a turn that
advanced a little then writes only the pages whose bytes changed. Without that flag the disk tier keeps one session
file per conversation, the earlier format.

The disk cache is opt-in and off by default. It has been measured on the reference host below; your model, context
and drive will differ, so try a small budget on your own hardware first.

## Turn it on

Add two engine arguments to the run config's `args`, the same place as `--conversation-cache-mib`:

```json
"args": [
  "--conversation-cache-disk-mib", "32768",
  "--conversation-cache-spill-dir", "D:\\strata-cache"
]
```

- `--conversation-cache-disk-mib` is the size of the cache in MiB. `0`, the default, is off.
- `--conversation-cache-spill-dir` is the folder. Use a fast local drive, not a network share or a folder that syncs
  to the cloud. The engine refuses to start when the budget is above 0 and the folder is missing. Use one folder for
  one engine: two engines writing the same pool would corrupt it.

Optional: `--conversation-cache-disk-only` keeps no conversations in RAM and uses only the disk tier. It is slower for
short conversations and useful for testing. `--conversation-cache-disk-slots N` limits how many conversations are kept.
`--conversation-cache-disk-staging-mib N` is the host-memory admission bound for a save or a restore (default 1024); a
smaller value can refuse a restore, and the log says the limit by name. `--conversation-cache-disk-cow` turns on the
shared page pool (Phase 2); without it each conversation is one session file, and `--conversation-cache-disk-whole-file`
says so explicitly. `--conversation-cache-disk-demote` also writes conversations the RAM cache evicts, and needs
`--conversation-cache-disk-cow`.
The pool is compacted when it has at least 2 GiB and 256 dead pages and they are at least a quarter of it
(`--conversation-cache-disk-compact-min-mib`, `--conversation-cache-disk-compact-min-pages`). Switching between the
page pool and the one-file format removes the entries of the other format at startup: the two formats are not read by
one another.

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
(`--conversation-cache-disk-whole-file`); the shared page pool's write amplification and restore speed have not been
measured yet. They are one model on one drive, not a promise for another configuration.

## What it costs in disk

Budget by the size of the conversations you keep. The measured 194.4K-token conversation above wrote 3.06 GiB; the
implementation plan's measured 75,659-token snapshot was 1,891.4 MiB, about 25 KB per token. That is one model and one
KV format: measure your own model and context, then set the budget above what you expect to need. The engine also keeps
4 GiB free on the drive by default (`--conversation-cache-disk-min-free-mib`).

The shared pool stores a page once however many conversations use it, and a turn rewrites only the changed pages. The
budget counts what each conversation names, not the pool's physical size; the pool can hold a little more until it is
compacted. That saves SSD writes but has not been measured on the reference host yet; a save's log line reports the
bytes written and reused.

When the budget is full, the oldest conversation goes first. A save that does not fit is refused, and the conversations
already on disk stay.

Each entry keeps the conversation's live state and its deepest checkpoint. It does not keep every interior checkpoint
yet, so a new conversation cannot start from a middle checkpoint of a disk entry; it can resume at the deepest one.

## Privacy

The files are plaintext. They hold the engine's state and token identity, not an encrypted archive; anyone who can read
the folder can read the conversation's tokens. The folder is created for your user only (0700, files 0600 on Linux and
macOS; on Windows it inherits your user profile's permissions). Keep it private, do not put it on a shared drive, and
delete the folder to clear the cache. The shared pool's manifests, metadata heads and pages are all in that folder.

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
`disk_reused_mib`, `disk_compactions`, `disk_compacted_mib`, `disk_demotions`, `last_tier`, `refusals` and
`last_refusal`.

See [DETAILS.md](DETAILS.md#disk-tier-l2) for every argument, and [TEST_REQUESTS.md](TEST_REQUESTS.md) if you want to
contribute a measurement.
