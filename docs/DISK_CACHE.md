# Disk conversation cache

Strata keeps the last conversation's state, so a follow-up reads only what is new. With
`--conversation-cache-mib` it also keeps several conversations in RAM, and switching between them is quick.

A long conversation can be too big for RAM. The disk cache is a second place for those conversations: a conversation
the RAM cache cannot keep is written to one file on disk, and read back when it returns. Then a long chat that takes
turns with small sub-agent requests does not read its whole prompt again.

The disk cache is opt-in and off by default. It is new: its files, manifest and restore have unit tests, but the whole
path has not been measured on a model yet. Try a small budget on your own hardware first.

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
  to the cloud. The engine refuses to start when the budget is above 0 and the folder is missing.

Optional: `--conversation-cache-disk-only` keeps no conversations in RAM and uses only the disk tier. It is slower for
short conversations and useful for testing. `--conversation-cache-disk-slots N` limits how many conversations are kept.

## What it costs in disk

Budget by the size of the conversations you keep. The measurement behind this feature, on the reference host in the
[implementation plan](work/DISK_CONVERSATION_CACHE_PLAN.md#13-why-it-does-not-fit), was a 75,659-token snapshot of
1,891.4 MiB, about 25 KB per token; a 200K-token conversation is therefore several GB. That figure is for one model and
one KV format: measure your own model and context, then set the budget above what you expect to need. The engine also
keeps 4 GiB free on the drive by default (`--conversation-cache-disk-min-free-mib`).

When the budget is full, the oldest conversation goes first. A save that does not fit is refused, and the conversations
already on disk stay.

Each file keeps the conversation's live state and its deepest checkpoint. It does not keep every interior checkpoint
yet, so a new conversation cannot start from a middle checkpoint of a disk entry; it can resume at the deepest one.

## Privacy

The files are plaintext. They hold the engine's state and token identity, not an encrypted archive; anyone who can read
the folder can read the conversation's tokens. The folder is created for your user only (0700, files 0600 on Linux and
macOS; on Windows it inherits your user profile's permissions). Keep it private, do not put it on a shared drive, and
delete the folder to clear the cache.

## See it in the Monitor

Open **Monitor** and look at the **Conversation cache** card. When the disk tier is on, a second block shows:

- **Conversations on disk** and **Their memory (disk)**, against the slot count (or "unlimited") and the budget.
- **Disk saved / restored**, **Disk evictions**, **Last disk restore** (milliseconds and MiB/s), and **Refused saves**
  with the reasons in plain words (RAM, budget, min tokens, disk space, model or settings).

The same values are in `/metrics` under `"conversation_cache"`, as `disk_enabled`, `disk_budget_mib`, `disk_dir`,
`disk_slots`, `disk_bytes`, `disk_entries`, `disk_saves`, `disk_restores`, `disk_evictions`, `disk_refusals`,
`disk_errors`, `disk_last_save_ms`, `disk_last_restore_ms`, `disk_last_restore_mib_s`, `last_tier`, `refusals` and
`last_refusal`.

See [DETAILS.md](DETAILS.md#disk-tier-l2) for every argument, and [TEST_REQUESTS.md](TEST_REQUESTS.md) if you want to
contribute a measurement.
