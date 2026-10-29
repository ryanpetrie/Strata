// CPU-only disk tier for parked conversations: the L2 cache beside the host-RAM ConversationCache.
//
// One manifest per conversation (small: identity, the live token prefix, image keys and checkpoint metadata) and one
// opaque payload file written by the caller's Writer (a session file, in the engine).  The manifest is the commit
// point: a payload without a manifest is garbage, a manifest whose payload is missing or a different size is dropped
// at startup, and a manifest with another magic, version, checksum or identity is deleted, never read.
//
// The store never accepts a file name or a path below its directory from the caller: it makes "snap-<16 hex>.bin"
// and "snap-<16 hex>.payload" itself.  The directory must be absolute and private; a link or reparse point there is
// refused, and nothing is followed.
//
// Manifest layout (little endian; the file is refused on a big-endian build):
//   magic "STRDISK\x01" (8), u32 version (2), u32 flags (1 = cvec, 2 = pinned, 4 = adapter state OFF,
//   8 = adapter state ON; neither = no adapter; both = refused), u64 identity model,
//   u64 identity config, u64 created, u64 last_used, u64 payload bytes, u32 name length, payload file name,
//   u32 live ids, i32 ids[], u32 live images, (i64 start, u64 hash) images[], u32 checkpoints,
//   per checkpoint: u32 ids, i32 ids[], u32 images, (i64 start, u64 hash) images[], u64 used, u32 flags (1 = pinned);
//   u64 checksum of every byte before it.
// Version 1 had no adapter state in the flags; it is refused, not migrated - the disk tier has shipped nothing.
// The payload keeps its own checksum (its format's), so the manifest binds its name and size instead of hashing
// gigabytes of K/V a second time.  Every length is checked against the bytes left before it is allocated.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_pages.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationDiskOptions {
    std::string dir;                 // required when the budget is not 0: absolute and private
    uint64_t budget_bytes = 0;       // 0 = disabled
    uint64_t min_free_bytes = 4ull << 30;
    int64_t min_tokens = 0;          // like --conversation-cache-min-tokens, for the disk tier
    size_t slots = 0;                // 0 = unlimited
    // Phase 2 (opt-in): write the K/V as content-addressed pages in a shared pool, so a turn that advanced a little
    // rewrites only the pages that changed.  The default keeps the Phase 1 format (one session file per conversation);
    // `--conversation-cache-disk-cow` turns this on.
    bool cow = false;
    // Compaction thresholds: reclaim pages no conversation names only when all three are passed.  The defaults are
    // ninfer's published guards (>=256 dead pages, >=2 GiB dead, >=25% of the pool), with a low-watermark: a
    // compaction leaves almost no dead space, so the next one waits for the whole quota again.
    uint64_t compact_min_bytes = 2ull << 30;
    uint64_t compact_min_pages = 256;
    double compact_fragmentation = 0.25;
};

// What matching needs, and nothing else: the live token prefix and image keys, and each checkpoint's ids/images with
// its LRU stamp and pin.  No K/V bytes, no checkpoint state arrays.
struct ConversationDiskMeta {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;   // ids, imgs, used and pinned only
    bool cvec = true;
    // the adapter state the cells were computed with, the third part of every cache key
    // (`ConversationLora`, conversation_cache.hpp): matching on it is what stops the disk tier handing
    // base-model cells to an adapted request when the RAM cache correctly refused them.
    ConversationLora lora = kLoraNone;
};

// conversation_prefix() (conversation_cache.hpp) reads a checkpoint; a manifest keeps the live state's ids and images
// apart.  Same rule: the whole saved list is a prefix of the prompt, the last prompt token is never covered, and the
// prompt's images inside it are exactly the saved ones.  conversation_disk_test compares both.
template<class Token>
int64_t conversation_disk_live_prefix(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                                      const std::vector<Token>& prompt,
                                      const std::vector<ConversationImageKey>& images) {
    const size_t n = ids.size();
    if (n == 0 || n >= prompt.size() || !std::equal(ids.begin(), ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == imgs.size() || !(imgs[j++] == image)) return 0;
    }
    if (j != imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationDiskStore {
public:
    struct Entry {
        uint64_t serial = 0;
        ConversationDiskMeta meta;
        uint64_t bytes = 0;
        uint64_t last_used = 0;
        bool pinned = false;
        std::string payload;    // whole-file entries: the absolute path, generated by the store
        std::string manifest;   // absolute path, generated by the store
        // Phase 2 (paged) entries: a metadata head and the K/V page list.  `payload` is empty for these.
        bool paged = false;
        std::string head;
        uint64_t head_bytes = 0, kv_bytes = 0;
        ConversationPageLayout layout;
    };
    struct Match {
        size_t index = 0;       // position in the index at match time; serial is the identity across changes
        int64_t tokens = 0;
        bool live = false;
        uint64_t serial = 0;
    };
    struct Stats {
        uint64_t saves = 0, evictions = 0, refusals = 0, errors = 0, restores = 0, restore_failures = 0;
        uint64_t last_save_ms = 0, last_restore_ms = 0, last_restore_bytes = 0;
        // Phase 2: what the last save wrote and reused, and the pool's write amplification and compaction
        uint64_t last_save_written_bytes = 0, last_save_reused_bytes = 0;
        uint64_t last_save_pages_written = 0, last_save_pages_reused = 0;
        uint64_t pages_written = 0, pages_reused = 0, bytes_written = 0, bytes_reused = 0;
        uint64_t compactions = 0, bytes_compacted = 0, last_compaction_ms = 0, last_compacted_bytes = 0;
        uint64_t pool_bytes = 0, pool_live_bytes = 0, pool_dead_bytes = 0, pool_dead_pages = 0;
    };
    // Writes the payload to `path` (the store's own name) and reports its bytes.  The Writer owns its temporary file
    // and its rename, so a failure leaves neither a half payload at `path` nor the previous file.
    using Writer = std::function<bool(const std::string& path, uint64_t& bytes, std::string& error)>;

    // `error` is empty on success.  A disabled store (budget 0) does not touch the directory and is not an error;
    // any other setup failure, a refused directory included, leaves enabled() false with the reason in `error`.
    ConversationDiskStore(const ConversationDiskOptions& options, const SessionFileIdentity& identity,
                          std::string& error);

    ConversationDiskStore(const ConversationDiskStore&) = delete;
    ConversationDiskStore& operator=(const ConversationDiskStore&) = delete;

    bool enabled() const;
    bool wants(int64_t tokens) const;
    bool paged() const { return pool_ != nullptr; }   // Phase 2: K/V pages in the shared pool
    size_t bytes() const;      // the payload bytes in the index
    size_t entries() const;    // the index size
    size_t slots() const;      // the configured cap (0 = unlimited)
    Stats stats() const;

    // The longest match in the index, by the rule of ConversationCache::best(): the control vector's setting and the
    // adapter state must agree, only a strict longer prefix replaces the match, and the index runs newest to oldest
    // so a tie keeps the most recently used entry.  `index` is the entry's position at match time.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
               ConversationLora lora) const {
        std::lock_guard<std::mutex> lock(mu_);
        Match best;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& e = entries_[i];
            if (e.meta.cvec != cvec || e.meta.lora != lora) continue;
            const int64_t live = conversation_disk_live_prefix(e.meta.ids, e.meta.images, prompt, images);
            if (live > best.tokens) best = {i, live, true, e.serial};
            for (const auto& c : e.meta.checkpoints) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, false, e.serial};
            }
        }
        return best;
    }

    // ConversationCache::borrows(): a new conversation starting at an interior checkpoint of a parked one restores
    // from it and leaves it in place.  A live match, a stale serial, a pinned checkpoint at or before the match, or
    // the deepest checkpoint do not borrow.
    bool borrows(const Match& match) const;

    // Valid until the store changes (the engine serializes requests, as for the RAM cache).
    const Entry* find(uint64_t serial) const;

    // Preflights `estimate_bytes` against the budget and the free-space floor BEFORE evicting anything; a refusal
    // evicts nothing.  Then evicts LRU unpinned entries until the budget and slots fit, calls `writer` without the
    // index lock, writes the manifest to a temporary name, and renames it; only then is the entry indexed.
    bool save(const ConversationDiskMeta& meta, uint64_t estimate_bytes, const Writer& writer, uint64_t& bytes,
              std::string& error);
    // Phase 2: the K/V as content-addressed pages in the shared pool, and `head` (metadata, running state and
    // checkpoints; its kv must be empty) as one small session file.  The store writes both; the caller's K/V is
    // streamed through the sources, one page at a time.  `written`/`reused` report the page bytes of this save.
    bool save_paged(const ConversationDiskMeta& meta, const SavedConversation& head,
                    const std::vector<SessionKvSource>& layers, uint64_t& bytes, uint64_t& written,
                    uint64_t& reused, std::string& error);
    // Opens a paged entry for a layer-at-a-time restore.  The reader resolves every page before it returns.
    bool open_paged(uint64_t serial, ConversationPageReader& reader, const SessionReadLimits& limits,
                    std::string& error, SessionStatus* status = nullptr) const;
    // Reclaims the pool's dead pages when the thresholds in the options are passed.  Manifests never change.
    bool compact(std::string& error);
    bool compact_needed() const;
    // The page pool, or null in whole-file mode.  The disk fast path lends its file handle around a DirectStorage
    // write (ConversationPagePool::borrow_for_fast_read); valid while this store is open and paged().
    ConversationPagePool* pool() { return pool_.get(); }
    uint64_t pool_bytes() const;
    uint64_t pool_dead_bytes() const;
    // Moves an entry to the most recently used end of the index.  The order is kept in RAM, as the RAM cache's is: a
    // restart orders the entries by the last_used each manifest was written with.
    bool touch(uint64_t serial);
    // The manifest goes first, so an interrupted delete cannot come back at the next start.
    bool remove(uint64_t serial, std::string& error);
    void note_restore(uint64_t ms, uint64_t bytes);
    void note_restore_failure();
    size_t clear();

private:
    bool valid_ = false;
    ConversationDiskOptions options_;
    SessionFileIdentity identity_;
    std::string manifests_, conversations_, tmp_;
    mutable std::mutex mu_;
    std::deque<Entry> entries_;   // least recently used first
    uint64_t bytes_ = 0, clock_ = 0, next_serial_ = 1;
    bool saving_ = false;
    Stats stats_;
    std::unique_ptr<ConversationPagePool> pool_;   // null in whole-file mode or when the tier is off

    bool setup(std::string& error);
    bool scan(std::string& error);
    // lock held
    bool wants_locked(int64_t tokens) const { return options_.min_tokens <= 0 || tokens >= options_.min_tokens; }
    bool evict_until_fits(uint64_t need, std::string& why);
    void enforce_limits();
    void erase_files(const Entry& entry);
    void release_pages(const Entry& entry);
    void note_pool_stats();
};

} // namespace strata::core
