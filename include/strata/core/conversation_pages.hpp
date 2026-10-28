// Content-addressed page pool for the disk tier (Phase 2: copy-on-write K/V).
//
// A conversation's K/V is stored as pages in one append-only pool shared by every parked conversation.  A page is
// identified by the hash of its bytes and its length, so a conversation that advanced a little rewrites only the pages
// that changed: the rest are looked up and reused.  The per-conversation manifest lists the page hashes; it never
// changes when the pool is compacted, because a page is found by its hash, not by an offset.
//
// On disk:
//   <root>/pages.bin   a 4 KiB header block, then 4 KiB-aligned page records
//   <root>/pages.idx   the hash -> offset index (a cache: it is rebuilt from pages.bin when it is missing or stale)
//
// A page record's data starts at a 4 KiB boundary; its 16-byte header (magic, length, hash) starts 16 bytes before the
// data, in the padding of the record before it.  Every record is written completely before the index names it, so the
// index is the commit point for a page.  A torn record at the end of the file is ignored, and the next append starts
// after it.
//
// The hashes detect accidental corruption; they do not authenticate a file.  Restore trusted files only.
#pragma once

#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

// One K/V page.  64 KiB keeps the page directory small (about 1 MB for a 3 GiB conversation) while a 2K-token advance
// rewrites only the tail page of each part plus the new pages.
inline constexpr uint32_t kConversationPageBytes = 64u << 10;
inline constexpr uint64_t kConversationPageAlign = 4u << 10;

// A page as a manifest records it: its content hash and length.  The offset is the pool's business.
struct ConversationPageRef {
    uint64_t hash = 0;
    uint32_t bytes = 0;
    bool operator==(const ConversationPageRef&) const = default;
};

// One Q/V layer in the pool: the session-file layer header, the size of each part, and the page list of each part.
struct ConversationPageLayer {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<uint64_t, 5> sizes{};
    std::array<std::vector<ConversationPageRef>, 5> parts;
    uint64_t kv_bytes() const {
        uint64_t n = 0;
        for (uint64_t size : sizes) n += size;
        return n;
    }
};

struct ConversationPageLayout {
    std::vector<ConversationPageLayer> layers;
    uint64_t kv_bytes() const {
        uint64_t n = 0;
        for (const auto& layer : layers) n += layer.kv_bytes();
        return n;
    }
};

struct ConversationPageStats {
    uint64_t pages = 0, live_pages = 0, dead_pages = 0;
    uint64_t pool_bytes = 0, live_bytes = 0, dead_bytes = 0;
    uint64_t pages_written = 0, pages_reused = 0;
    uint64_t bytes_written = 0, bytes_reused = 0;
    uint64_t compactions = 0, bytes_compacted = 0, last_compaction_ms = 0, last_compacted_bytes = 0;
};

// The shared pool.  All methods are safe to call from one thread; the mutex only guards the index and the append
// position (the engine serves one request at a time).
class ConversationPagePool {
public:
    struct Page {
        uint64_t offset = 0;    // where the data starts (4 KiB aligned)
        uint32_t bytes = 0;
        uint32_t refs = 0;      // how many indexed conversations name this page
    };

    // `root` is the pool folder (<cache>/pool).
    explicit ConversationPagePool(std::string root);
    ~ConversationPagePool();
    ConversationPagePool(const ConversationPagePool&) = delete;
    ConversationPagePool& operator=(const ConversationPagePool&) = delete;

    // Opens (or creates) pages.bin and loads pages.idx, rebuilding it by scanning the pool when it is missing,
    // stale, or refuses its checksum.  `writable` opens the pool for appending.
    bool open(bool writable, std::string& error);
    bool writable() const { return writable_; }

    bool lookup(uint64_t hash, uint32_t bytes, Page& out) const;
    // Reads a page and verifies its hash.  `offset` comes from lookup(); a caller that cached it must not use it
    // across a compaction.
    bool read(uint64_t hash, uint32_t bytes, uint64_t offset, void* dst, std::string& error) const;
    // Appends a page unless its (hash, bytes) is already present.  `wrote` says whether new bytes were written.
    // The index is not persisted until persist() succeeds: that is the page's commit point.
    bool append(uint64_t hash, const void* data, uint32_t bytes, ConversationPageRef& ref, bool& wrote,
                std::string& error);
    // Drops a page whose hash/length is not in the index: the caller already has the ref.
    void reference(uint64_t hash, uint32_t bytes);
    void release(uint64_t hash, uint32_t bytes);

    // Writes the index atomically.  A save calls this after its pages are written and before its manifest.
    bool persist(std::string& error);

    // Reclaims pages no indexed conversation names, when the dead pages/bytes/fragmentation pass the thresholds.
    // The pool is unchanged when the thresholds are not met.  Manifests never change: a page is found by its hash.
    bool compact(uint64_t min_dead_bytes, uint64_t min_dead_pages, double min_fragmentation, std::string& error);
    bool compact_needed(uint64_t min_dead_bytes, uint64_t min_dead_pages, double min_fragmentation) const;

    ConversationPageStats stats() const;
    uint64_t generation() const;
    const std::string& path() const { return pool_path_; }

private:
    struct Key {
        uint64_t hash = 0;
        uint32_t bytes = 0;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = k.hash ^ (uint64_t(k.bytes) * 0x9E3779B97F4A7C15ull);
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdull;
            h ^= h >> 33;
            return (size_t) h;
        }
    };
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string pool_path_, index_path_, tmp_path_;
    bool writable_ = false;
    uint64_t generation_ = 0, next_offset_ = 0;
    mutable std::mutex mu_;
    std::unordered_map<Key, Page, KeyHash> index_;
    mutable ConversationPageStats stats_;
    bool index_dirty_ = false;
};

// Writes the K/V sources of a conversation into the pool, one page at a time, and reports the page list of each layer.
// The sources are read through SessionKvSource::read, so a capture holds one page plus one layer, never the whole K/V.
struct ConversationPageWriteResult {
    ConversationPageLayout layout;
    uint64_t bytes_written = 0;   // new page bytes written to the pool
    uint64_t bytes_reused = 0;    // page bytes already in the pool
    uint64_t pages_written = 0, pages_reused = 0;
};
bool conversation_pages_write(ConversationPagePool& pool, const std::vector<SessionKvSource>& sources,
                              ConversationPageWriteResult& out, std::string& error);

// Restores a paged conversation one K/V layer at a time, so peak host RAM is the head's metadata, one layer and one
// page, never the whole conversation.  The head is a session file with no K/V layers (the metadata and the running
// checkpoint); the pages come from the pool.
class ConversationPageReader : public ConversationKvLayers {
public:
    ConversationPageReader();
    ~ConversationPageReader() override;
    ConversationPageReader(ConversationPageReader&&) noexcept;
    ConversationPageReader& operator=(ConversationPageReader&&) noexcept;
    ConversationPageReader(const ConversationPageReader&) = delete;
    ConversationPageReader& operator=(const ConversationPageReader&) = delete;

    // `limits` applies to the head and to the caller's own staging: the reader asks limits.admit for the head's
    // metadata plus the largest layer plus one page before it allocates any of them.
    bool open(const std::string& head_path, const ConversationPageLayout& layout, const ConversationPagePool& pool,
              const SessionFileIdentity& identity, const SessionReadLimits& limits, std::string& error,
              SessionStatus* status = nullptr, uint64_t head_bytes = 0);

    const SavedConversation& meta() const override;
    size_t layers() const override;
    bool next(ConversationKv& layer, std::string& error) override;
    bool finish(std::string& error) override;
    uint64_t bytes() const override;

    // The head's bytes plus the K/V bytes the manifest names (the logical size, not the pool's physical size).
    uint64_t logical_bytes() const;
    // High-water host staging since open(): the head's metadata and the largest layer read, plus one page.
    uint64_t peak_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
