#pragma once
// include/strata/core/conversation_spill.hpp - the SSD tier of the conversation cache (phase P2 of the admission design,
// RESULTS-ENG-1116.md): where a parked conversation is written when RAM refuses it, and how much may go there.
//
// Ownership: every engine process writes into a folder of its own, `<root>/<pid>-<16 hex nonce>`, and holds a lock on
// `<that folder>/.lock` for its whole life (POSIX flock on an open file; Windows: the file opened with no sharing).  At start a
// process removes the folders under `<root>` whose lock nobody holds - and nothing else: only folders named exactly that way,
// never a link, never a file of the root, never another process's live folder.  Two engines on one data folder therefore
// never delete or overwrite each other's files (#6034 wiped every .conv file of a shared folder at start), and a crashed
// engine's files go at the next start.  The files themselves are session files (conversation_file.hpp: a temporary name,
// then a rename; bounded reads), so an entry exists only once it is complete.
//
// Budget: min(32 GiB, 25% of the free space), never below the free-space floor; a write reserves its estimate first and
// the reservation counts until the write ends (temporary files included).

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/spill_pages.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#endif
#endif

namespace strata::core {

/// The SSD tier's size: min(32 GiB, 25% of `free_bytes`), and never more than what keeps `floor_bytes` free.
inline uint64_t spill_default_budget(uint64_t free_bytes, uint64_t floor_bytes) {
    if (free_bytes <= floor_bytes) return 0;
    uint64_t b = free_bytes / 4;
    if (b > (uint64_t(32) << 30)) b = uint64_t(32) << 30;
    if (b > free_bytes - floor_bytes) b = free_bytes - floor_bytes;
    return b;
}

/// What the tier holds and has promised: `used` (written entries) + `reserved` (writes in flight) never pass `cap`.
struct SpillBudget {
    uint64_t cap = 0, used = 0, reserved = 0;
    bool reserve(uint64_t n) {
        if (n > cap || used + reserved > cap - n) return false;
        reserved += n;
        return true;
    }
    /// A write ended: its reservation goes; `written` bytes stay (0 when it failed).  A file larger than its reservation is
    /// the caller's to remove when used would pass the cap (fits_after).
    void finish(uint64_t reserved_n, uint64_t written) {
        reserved -= reserved_n < reserved ? reserved_n : reserved;
        used += written;
    }
    bool over() const { return used + reserved > cap; }
    void drop(uint64_t n) { used -= n < used ? n : used; }
};

namespace spill_detail {
inline bool is_namespace_name(const std::string& n) {
    const size_t dash = n.find('-');
    if (dash == std::string::npos || dash == 0 || n.size() != dash + 17) return false;
    for (size_t i = 0; i < dash; ++i) if (n[i] < '0' || n[i] > '9') return false;
    for (size_t i = dash + 1; i < n.size(); ++i)
        if (!((n[i] >= '0' && n[i] <= '9') || (n[i] >= 'a' && n[i] <= 'f'))) return false;
    return true;
}
/// `fp-<16 hex>`: the stable folder of one model fingerprint (--conversation-spill-persist)
inline std::string persist_name(uint64_t fingerprint) {
    char b[32];
    std::snprintf(b, sizeof b, "fp-%016llx", (unsigned long long) fingerprint);
    return b;
}
inline bool is_persist_name(const std::string& n) {
    if (n.size() != 19 || n.compare(0, 3, "fp-") != 0) return false;
    for (size_t i = 3; i < n.size(); ++i)
        if (!((n[i] >= '0' && n[i] <= '9') || (n[i] >= 'a' && n[i] <= 'f'))) return false;
    return true;
}
#if defined(_WIN32)
using LockHandle = HANDLE;
inline LockHandle no_lock() { return INVALID_HANDLE_VALUE; }
inline LockHandle try_lock(const std::filesystem::path& p) {
    return CreateFileW(p.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}
inline void unlock(LockHandle h) { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
inline unsigned long own_pid() { return (unsigned long) _getpid(); }
#else
using LockHandle = int;
inline LockHandle no_lock() { return -1; }
inline LockHandle try_lock(const std::filesystem::path& p) {
    const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); return -1; }
    return fd;
}
inline void unlock(LockHandle fd) { if (fd >= 0) ::close(fd); }
inline unsigned long own_pid() { return (unsigned long) ::getpid(); }
#endif
/// The calling thread gives way (the SSD tier's background writer, P2 step 2c): on Linux nice +10 and the lowest best-effort I/O
/// level (not the idle class, which a steady stream of expert reads could starve - and a starved writer would hold the serve loop
/// at its next push); on Windows a below-normal thread priority; elsewhere nothing.  Best effort: a refusal changes nothing.
inline void lower_thread_priority() {
#if defined(_WIN32)
    (void) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    const long tid = ::syscall(SYS_gettid);
    (void) ::setpriority(PRIO_PROCESS, (id_t) tid, 10);
    constexpr int kWhoProcess = 1, kClassBestEffort = 2, kLowest = 7, kClassShift = 13;
    (void) ::syscall(SYS_ioprio_set, kWhoProcess, (int) tid, (kClassBestEffort << kClassShift) | kLowest);
#endif
}
}  // namespace spill_detail

/// Whether a conversation is worth writing to the SSD tier: reading it back (and writing it first) must cost less than reading its
/// prompt again.  A parked image has a fixed-size running state (about 236 MB on Qwen3.8-Flash-Next) whatever its length, so a short
/// conversation is cheaper to re-read than to round-trip through the disk, and a long one is not (l620, RTX PRO 5000: a 51-token
/// conversation re-read in about 220 ms against 120 ms written + 220 ms read back; an 8K one 1.6 s against about 0.5 s).  The speeds
/// are this engine's own, measured as it runs (moving averages; defaults until the first measurement: 2 tokens per ms - a mid-range
/// card's prompt speed - and 1 GB/s each way).  The prompt speed is learned only from passes of at least kMinTokens fresh tokens, so
/// a request's fixed costs do not make re-reading look slow.
struct SpillCost {
    static constexpr int64_t kMinTokens = 512;
    static constexpr double kAlpha = 0.3, kMargin = 1.25;
    double reread_tokens_per_ms = 2.0;
    double write_bytes_per_ms = 1.0e6, read_bytes_per_ms = 1.0e6;
    static void blend(double& avg, double sample) {
        if (sample > 0.0 && sample < 1e300) avg = avg * (1.0 - kAlpha) + sample * kAlpha;
    }
    void saw_prompt(int64_t tokens, double ms) { if (tokens >= kMinTokens && ms > 0.0) blend(reread_tokens_per_ms, (double) tokens / ms); }
    void saw_write(uint64_t bytes, double ms) { if (bytes > 0 && ms > 0.0) blend(write_bytes_per_ms, (double) bytes / ms); }
    void saw_read(uint64_t bytes, double ms) { if (bytes > 0 && ms > 0.0) blend(read_bytes_per_ms, (double) bytes / ms); }
    double reread_ms(int64_t tokens) const { return tokens > 0 ? (double) tokens / reread_tokens_per_ms : 0.0; }
    double round_trip_ms(uint64_t bytes) const { return (double) bytes / write_bytes_per_ms + (double) bytes / read_bytes_per_ms; }
    bool worth(int64_t tokens, uint64_t bytes) const { return reread_ms(tokens) > kMargin * round_trip_ms(bytes); }
};

/// This process's spill folder.  open() makes `<root>`, cleans the dead folders and takes its own; the destructor gives the
/// lock back and removes its own folder (an orderly end; after a crash the next start's clean_dead does it).
class SpillNamespace {
public:
    SpillNamespace() = default;
    SpillNamespace(const SpillNamespace&) = delete;
    SpillNamespace& operator=(const SpillNamespace&) = delete;
    ~SpillNamespace() { close(); }

    /// `persist_fp`: the model fingerprint whose stable folder `<root>/fp-<hex>` is taken instead of a per-process one (it and its
    /// files outlive the process).  The folders of other fingerprints nobody holds are removed; when the stable folder is held by a
    /// live engine, this one falls back to its own per-process folder (persistent() false).
    bool open(const std::filesystem::path& root, std::string& error, size_t* cleaned = nullptr,
              const uint64_t* persist_fp = nullptr) {
        close();
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec || !std::filesystem::is_directory(root, ec) || std::filesystem::is_symlink(root, ec)) {
            error = "spill folder " + root.string() + " cannot be used" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        const std::string keep = persist_fp ? spill_detail::persist_name(*persist_fp) : std::string();
        const size_t n = clean_dead(root, persist_fp != nullptr, keep);
        if (cleaned) *cleaned = n;
        if (persist_fp) {
            const std::filesystem::path pdir = root / keep;
            std::error_code e3;
            if (!std::filesystem::is_symlink(pdir, e3) && (std::filesystem::is_directory(pdir, e3) || std::filesystem::create_directory(pdir, e3))) {
                const spill_detail::LockHandle h = spill_detail::try_lock(pdir / ".lock");
                if (h != spill_detail::no_lock()) {
                    lock_ = h;
                    root_ = root;
                    dir_ = pdir;
                    persistent_ = true;
                    return true;
                }
            }
        }
        std::random_device rd;
        char name[64];
        std::snprintf(name, sizeof name, "%lu-%08x%08x", spill_detail::own_pid(), (unsigned) rd(), (unsigned) rd());
        const std::filesystem::path dir = root / name;
        if (!std::filesystem::create_directory(dir, ec) || ec) {
            error = "spill folder " + dir.string() + " cannot be made" + (ec ? ": " + ec.message() : std::string());
            return false;
        }
        lock_ = spill_detail::try_lock(dir / ".lock");
        if (lock_ == spill_detail::no_lock()) {
            std::filesystem::remove_all(dir, ec);
            error = "spill folder " + dir.string() + ": its lock cannot be taken";
            return false;
        }
        root_ = root;
        dir_ = dir;
        return true;
    }
    void close() {
        if (dir_.empty()) return;
        spill_detail::unlock(lock_);
        lock_ = spill_detail::no_lock();
        if (!persistent_) {
            std::error_code ec;
            std::filesystem::remove_all(dir_, ec);
        }
        persistent_ = false;
        dir_.clear();
    }
    bool persistent() const { return persistent_; }
    bool is_open() const { return !dir_.empty(); }
    const std::filesystem::path& dir() const { return dir_; }
    /// the file of parked conversation `serial` (a session file; written to a temporary name first by session_file_write)
    std::filesystem::path file_for(uint64_t serial) const {
        return dir_ / ("conv-" + std::to_string(serial) + ".session");
    }

    /// Removes the folders under `root` that are named like a namespace, are real folders (not links) and whose lock can be
    /// taken - no process holds them.  Returns how many went.  Everything else under `root` is left alone.
    /// With `persist`, the unheld `fp-<hex>` folders other than `keep` go too (another model's spilled conversations).
    static size_t clean_dead(const std::filesystem::path& root, bool persist = false, const std::string& keep = {}) {
        size_t n = 0;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            const std::filesystem::path p = it->path();
            std::error_code e2;
            if (it->is_symlink(e2) || !it->is_directory(e2) || !(spill_detail::is_namespace_name(p.filename().string()) ||
                (persist && spill_detail::is_persist_name(p.filename().string()) && p.filename().string() != keep))) continue;
            const spill_detail::LockHandle h = spill_detail::try_lock(p / ".lock");
            if (h == spill_detail::no_lock()) continue;   // a live engine holds it (or it cannot be checked): kept
            spill_detail::unlock(h);
            std::filesystem::remove_all(p, e2);
            if (!e2) ++n;
        }
        return n;
    }

private:
    std::filesystem::path root_, dir_;
    spill_detail::LockHandle lock_ = spill_detail::no_lock();
    bool persistent_ = false;
};

// ---------------------------------------------------------------------------------------------------------------
// Local additions (branch local/spill-unify; docs/work/local-spill-delta.md).  Everything below this line is
// optional and default-off: the tier behaves exactly as upstream when the engine uses put()/take_serial()/best()
// with their upstream arguments.
//
// The local branch's separate disk tier (ConversationDiskStore) was folded in here, keeping four things it did
// better: a streaming save from live K/V sources (put_streaming), a streaming restore with bounded staging
// (take_layers), the content-addressed page pool as the spill file's backend (spill_pages.hpp), and its product
// surface (stats(), the page/eviction counters, guard(), the timed flush that feeds the background writer).
// ---------------------------------------------------------------------------------------------------------------

/// The page backend's settings (the `--conversation-spill-pages*` flags).  `enabled` false keeps upstream's
/// whole-file behaviour: one session file per entry, no pool, no extra folder.
struct SpillPageOptions {
    bool enabled = false;
    uint64_t compact_min_bytes = 2ull << 30;   // all three guards must pass before a compaction reclaims dead pages
    uint64_t compact_min_pages = 256;
    double compact_fragmentation = 0.25;
};

/// What one save wrote, for the engine's log line: `file_bytes` is the entry's logical size (the head plus the K/V
/// it names), `bytes_written` the new bytes on the disk (a shared page is not written again).
struct SpillSaveReport {
    uint64_t file_bytes = 0;
    uint64_t bytes_written = 0;
    uint64_t bytes_reused = 0;
    uint64_t pages_written = 0, pages_reused = 0;
};

/// The spill tier's counters (the server's Monitor card reads them from the engine's INFO fields).
struct SpillStats {
    uint64_t saves = 0, save_failures = 0, save_refusals = 0;
    uint64_t restores = 0, restore_failures = 0;
    uint64_t evictions = 0, clearances = 0;
    uint64_t bytes_written = 0, bytes_read = 0;
    uint64_t last_save_ms = 0, last_restore_ms = 0, last_restore_bytes = 0;
    uint64_t pages_written = 0, pages_reused = 0;        // this run, all saves
    uint64_t compactions = 0, bytes_compacted = 0, last_compaction_ms = 0, last_compacted_bytes = 0;
};

/// The SSD tier's entries: parked conversations written as session files into this process's namespace, oldest first.
/// What stays in RAM per entry is only what matching needs: the token ids and picture keys of its live state and of each
/// checkpoint (no running state, no K/V).  put() deletes the oldest entries when the budget needs the room, and never
/// leaves a partial file behind; take() reads an entry back under the caller's read limits and removes it either way (a
/// file that does not read back is not tried again).
/// Thread-safe (P2 step 2c: a background writer puts while the serve loop matches and takes): the entry list and the budget are
/// under one lock, a file is written and read outside it, and an entry is reached by its serial (an index can shift when the
/// writer drops the oldest).  An entry is listed only once its file is complete.
class ConversationSpill {
public:
    struct Entry {
        uint64_t serial = 0;
        std::filesystem::path file;
        uint64_t bytes = 0;
        bool cvec = false;
        uint64_t used = 0;                               // last use (written, matched-and-read): the least recently used goes first
        ConversationCheckpoint live;                     // ids and images only
        std::vector<ConversationCheckpoint> checkpoints;   // ids and images only
        // local: the adapter state the cells were computed with, the third part of every cache key
        // (ConversationLora, conversation_cache.hpp).  Matching on it is what stops the tier handing base-model
        // cells to an adapted request when the RAM cache correctly refused them (the bug the disk tier fixed).
        ConversationLora lora = kLoraNone;
        // local: a paged entry keeps its K/V in the shared pool (spill_pages.hpp) and `file` is a head: a session
        // file with no K/V layers.  `bytes` is logical (the head plus the K/V it names), as the disk tier counted it.
        bool paged = false;
        uint64_t kv_bytes = 0;
        std::vector<ConversationPageLayer> layout;
    };

    /// `persist`: the tier lives in the stable folder of persist->model and keeps its entries across restarts: those of the files
    /// whose identity is `*persist` (model and config) are adopted (`*adopted` counts them, oldest use first), any other file of
    /// that folder is deleted.  The folders of other fingerprints go.  Reads leave the entries in place; a per-process folder
    /// (persist null, or the stable one held by another engine) behaves as before.
    bool open(const std::filesystem::path& root, uint64_t cap, std::string& error, size_t* cleaned = nullptr,
              const SessionFileIdentity* persist = nullptr, size_t* adopted = nullptr,
              const SpillPageOptions* page_options = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        entries_.clear();
        pages_.reset();
        page_error_.clear();
        guarded_ = 0;
        page_opts_ = page_options ? *page_options : SpillPageOptions{};
        budget_ = SpillBudget{cap, 0, 0};
        tick_ = 0;
        next_serial_ = 1;
        if (adopted) *adopted = 0;
        const bool ok = ns_.open(root, error, cleaned, persist ? &persist->model : nullptr);
        if (ok && page_opts_.enabled) {
            // The K/V backend lives inside this process's namespace folder (`<dir>/pool`), so it goes with it.
            auto pool = std::make_unique<ConversationPagePool>((ns_.dir() / "pool").string());
            std::string perr;
            if (pool->open(true, perr)) {
                pages_ = std::move(pool);
            } else {
                page_error_ = perr.empty() ? "the page pool could not be opened" : perr;   // whole-file entries still work
            }
        }
        if (ok && ns_.persistent()) {
            const size_t n = adopt_locked(*persist);
            if (adopted) *adopted = n;
        }
        return ok;
    }
    bool persistent() const { std::lock_guard<std::mutex> lk(mu_); return ns_.persistent(); }
    bool enabled() const { std::lock_guard<std::mutex> lk(mu_); return enabled_locked(); }
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return entries_.size(); }
    uint64_t bytes() const { std::lock_guard<std::mutex> lk(mu_); return budget_.used; }
    SpillBudget budget() const { std::lock_guard<std::mutex> lk(mu_); return budget_; }
    Entry entry(size_t i) const { std::lock_guard<std::mutex> lk(mu_); return entries_.at(i); }
    /// the entry `serial` (a copy), if it is still listed
    std::optional<Entry> find(uint64_t serial) const {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t i = index_locked(serial);
        return i == SIZE_MAX ? std::nullopt : std::optional<Entry>(entries_[i]);
    }
    std::filesystem::path dir() const { std::lock_guard<std::mutex> lk(mu_); return ns_.dir(); }

    /// Writes `image` as an entry (its serial in `*serial_out`).  False (with `error`) when the tier is off, the image is larger
    /// than the whole tier, or the write failed; nothing of this call is then left on the disk.  The file is written outside the
    /// lock, under a budget reserved for it.
    bool put(const SavedConversation& image, const SessionFileIdentity& id, std::string& error, uint64_t* serial_out = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t need = (uint64_t) image.bytes() + (uint64_t(1) << 20);   // the payload and the file's own tables
        uint64_t serial = 0;
        std::filesystem::path f;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!enabled_locked()) { error = "the SSD tier is off"; return false; }
            if (need > budget_.cap) { error = "larger than the whole SSD tier"; return false; }
            if (!make_room_locked(need, image, error)) { ++stats_.save_refusals; return false; }
            serial = next_serial_++;
            f = ns_.file_for(serial);
        }
        size_t written = 0;
        SessionWriteOptions wo;
        wo.min_free_bytes = min_free_;
        const bool ok = session_file_write(f.string(), image, id, written, error, wo);
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
        }
        Entry e;
        if (ok) {
            e.serial = serial; e.file = f; e.bytes = written; e.cvec = image.cvec; e.lora = image.lora;
            e.live = stub(image.live);
            for (const auto& c : image.checkpoints) e.checkpoints.push_back(stub(c));
        }
        std::lock_guard<std::mutex> lk(mu_);
        budget_.finish(need, ok ? written : 0);
        if (ok && ns_.persistent() && !meta_write(f, e, id)) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
            budget_.drop(written);
            error = "the entry's index file could not be written";
            return false;
        }
        if (!ok) { ++stats_.save_failures; return false; }
        e.used = ++tick_;
        // the same conversation grown: an entry whose tokens are a prefix of this one's is served by this one in full
        for (size_t i = entries_.size(); i-- > 0;)
            if (superseded_by(entries_[i], image)) drop_locked(i);
        entries_.push_back(std::move(e));
        while (budget_.over() && entries_.size() > 1) drop_locked(lru_locked());   // (a file larger than its estimate)
        trim_locked();
        ++stats_.saves;
        stats_.bytes_written += written;
        stats_.last_save_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t0).count();
        if (serial_out) *serial_out = serial;
        return true;
    }

    /// The entry and the prefix it gives `prompt`, by the RAM cache's rule (ConversationCache::best): its live state or a
    /// checkpoint, same pictures, same steering; ties prefer the newest.  Reach it by `serial` (find / take_serial).
    /// `lora` is the RAM cache's third key (local addition; the upstream call means kLoraNone, which is what a tier
    /// written by a build without adapters holds): an entry whose cells were computed under another adapter state is
    /// never a match, in either direction.
    template <class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
                                  ConversationLora lora = kLoraNone) const {
        std::lock_guard<std::mutex> lk(mu_);
        ConversationCache::Match m;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& e = entries_[i];
            if (e.cvec != cvec || e.lora != lora) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > m.tokens) m = {i, n, live, e.serial};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return m;
    }

    /// Reads entry `serial` back into `out` (bounded by `limits`, the identity checked) and removes the entry.  False: `out` is
    /// unchanged and the entry is gone (or was not there).  The file is read outside the lock: the entry leaves the list first.
    bool take_serial(uint64_t serial, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out,
                     std::string& error) {
        std::filesystem::path f;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t i = index_locked(serial);
            if (i == SIZE_MAX) { error = "no such SSD entry"; return false; }
            f = entries_[i].file;
            if (ns_.persistent()) {   // kept: a restart finds it again; the next spill that extends it replaces it
                entries_[i].used = ++tick_;
                std::error_code ec;
                std::filesystem::last_write_time(meta_path(f), std::filesystem::file_time_type::clock::now(), ec);
            } else {
                budget_.drop(entries_[i].bytes);
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            }
        }
        const bool keep = persistent_now();
        SavedConversation image;
        size_t read = 0;
        const bool ok = session_file_read(f.string(), id, image, read, error, limits);
        if (!keep) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
        } else if (!ok) {
            drop_serial(serial);   // (a file that does not read back is not tried again)
        }
        if (ok) out = std::move(image);
        return ok;
    }
    bool take(size_t i, const SessionFileIdentity& id, const SessionReadLimits& limits, SavedConversation& out, std::string& error) {
        uint64_t serial = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (i >= entries_.size()) { error = "no such SSD entry"; return false; }
            serial = entries_[i].serial;
        }
        return take_serial(serial, id, limits, out, error);
    }

    void drop(size_t i) { std::lock_guard<std::mutex> lk(mu_); drop_locked(i); }
    /// Drops entry `serial` and its file; false when it was not listed.
    bool drop_serial(uint64_t serial) {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t i = index_locked(serial);
        if (i == SIZE_MAX) return false;
        drop_locked(i);
        return true;
    }
    /// A snapshot of the same conversation is being parked, grown past these tokens: deletes the entries (file and index file)
    /// whose live tokens are a prefix of `image_live` (same steering and pictures) - they would never be read again.  Returns the
    /// token counts of the deleted entries.  `lora` is the RAM cache's third key (local addition, defaulted): an entry whose
    /// cells were computed under another adapter state is never superseded by this one, as superseded_by() refuses it.
    std::vector<size_t> drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec,
                                       ConversationLora lora = kLoraNone) {
        SavedConversation image_live;   // (tokens and pictures only)
        image_live.live.ids = ids;
        image_live.live.imgs = imgs;
        image_live.cvec = cvec;
        image_live.lora = lora;
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<size_t> out;
        for (size_t i = entries_.size(); i-- > 0;)
            if (superseded_by(entries_[i], image_live)) { out.push_back(entries_[i].live.ids.size()); drop_locked(i); }
        return out;
    }
    /// Drops every entry and keeps the tier open (#879: after non-finite logits no parked conversation is kept).  Returns how many.
    size_t clear() {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t n = entries_.size();
        guarded_ = 0;   // (#879 wins over a restore in flight: the state it reads must not be kept either)
        while (!entries_.empty()) drop_locked(entries_.size() - 1);
        stats_.clearances += n;
        return n;
    }
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!ns_.persistent()) while (!entries_.empty()) drop_locked(entries_.size() - 1);   // (a stable folder keeps its entries)
        entries_.clear();
        pages_.reset();
        guarded_ = 0;
        ns_.close();
    }

    // ---- local additions: the streaming save, the layered restore, the page backend, telemetry (see the note above) ----

    /// The streaming save: `head` carries everything but the K/V (`head.kv` must be empty) and `sources` supply each
    /// layer as the file is written, so a conversation this host could not capture whole is still spilled - the disk
    /// tier's strongest case.  With the page pool attached the K/V goes to the pool as content-addressed pages (only
    /// the pages that changed are written) and `head` becomes a small session file beside them; without it the file
    /// is written whole, exactly as put() writes it.  put(image) stays the whole-image wrapper and is unchanged.
    bool put_streaming(const SavedConversation& head, const std::vector<SessionKvSource>& sources,
                       const SessionFileIdentity& id, std::string& error, uint64_t* serial_out = nullptr,
                       SpillSaveReport* report = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t kv = 0;
        for (const auto& s : sources)
            for (uint64_t n : s.sizes) kv += n;
        const uint64_t need = (uint64_t) head.bytes() + kv + (uint64_t(1) << 20);
        uint64_t serial = 0;
        std::filesystem::path f;
        bool paged = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!enabled_locked()) { error = "the SSD tier is off"; return false; }
            if (need > budget_.cap) { error = "larger than the whole SSD tier"; ++stats_.save_refusals; return false; }
            if (!make_room_locked(need, head, error)) { ++stats_.save_refusals; return false; }
            serial = next_serial_++;
            f = ns_.file_for(serial);
            paged = pages_ != nullptr;
        }
        SpillSaveReport rep;
        Entry e;
        bool ok = false;
        SessionWriteOptions wo;
        wo.min_free_bytes = min_free_;
        if (paged) {
            size_t head_bytes = 0;
            ok = session_file_write(f.string(), head, id, head_bytes, error, wo);   // the head: no K/V layers
            if (ok) {
                ConversationPageWriteResult wr;
                ok = conversation_pages_write(*pages_, sources, wr, error) && pages_->persist(error) &&
                     write_page_index(f, id, wr.layout.layers, error);
                if (ok) {
                    rep.file_bytes = (uint64_t) head_bytes + wr.layout.kv_bytes();
                    rep.bytes_written = (uint64_t) head_bytes + wr.bytes_written;
                    rep.bytes_reused = wr.bytes_reused;
                    rep.pages_written = wr.pages_written;
                    rep.pages_reused = wr.pages_reused;
                    e.paged = true;
                    e.kv_bytes = wr.layout.kv_bytes();
                    e.layout = std::move(wr.layout.layers);
                }
            }
        } else {
            size_t written = 0;
            ok = session_file_write(f.string(), head, sources, id, written, error, wo);
            rep.file_bytes = rep.bytes_written = (uint64_t) written;
        }
        if (!ok) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
            std::filesystem::remove(pages_path(f), ec);
            std::lock_guard<std::mutex> lk(mu_);
            budget_.finish(need, 0);
            ++stats_.save_failures;
            return false;
        }
        e.serial = serial;
        e.file = f;
        e.bytes = rep.file_bytes;
        e.cvec = head.cvec;
        e.lora = head.lora;
        e.live = stub(head.live);
        for (const auto& c : head.checkpoints) e.checkpoints.push_back(stub(c));
        std::lock_guard<std::mutex> lk(mu_);
        budget_.finish(need, rep.file_bytes);   // logical bytes: the head plus the K/V this entry names (as the disk tier)
        if (ns_.persistent() && !meta_write(f, e, id)) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
            std::filesystem::remove(pages_path(f), ec);
            budget_.drop(rep.file_bytes);
            error = "the entry's index file could not be written";
            ++stats_.save_failures;
            return false;
        }
        if (e.paged)
            for (const auto& l : e.layout)
                for (const auto& part : l.parts)
                    for (const auto& ref : part) pages_->reference(ref.hash, ref.bytes);
        e.used = ++tick_;
        for (size_t i = entries_.size(); i-- > 0;)
            if (superseded_by(entries_[i], head)) drop_locked(i);
        entries_.push_back(std::move(e));
        while (budget_.over() && entries_.size() > 1) drop_locked(lru_locked());
        trim_locked();
        ++stats_.saves;
        stats_.bytes_written += rep.bytes_written;
        stats_.pages_written += rep.pages_written;
        stats_.pages_reused += rep.pages_reused;
        stats_.last_save_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t0).count();
        if (serial_out) *serial_out = serial;
        if (report) *report = rep;
        return true;
    }

    /// The streaming restore: opens entry `serial` as a layer-at-a-time reader (a session file, or the page pool), so
    /// peak host staging is the metadata, one layer and one page - never the whole conversation.  The entry STAYS
    /// (its use is bumped; a later save that extends it supersedes it): the reader is still reading it, and a paged
    /// entry's next save then writes only the pages that changed.  A failed open counts a failed restore and the
    /// caller drops the entry (drop_serial); take_serial() stays upstream's whole-image read.
    bool take_layers(uint64_t serial, const SessionFileIdentity& id, const SessionReadLimits& limits,
                     std::unique_ptr<ConversationKvLayers>& out, std::string& error) {
        Entry e;
        bool persistent = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t i = index_locked(serial);
            if (i == SIZE_MAX) { error = "no such SSD entry"; return false; }
            entries_[i].used = ++tick_;
            e = entries_[i];
            persistent = ns_.persistent();
            if (persistent) {
                std::error_code ec;
                std::filesystem::last_write_time(meta_path(e.file), std::filesystem::file_time_type::clock::now(), ec);
            }
        }
        if (e.paged) {
            if (!pages_) { error = "the page pool of this entry is not open"; return false; }
            ConversationPageLayout layout;
            layout.layers = std::move(e.layout);
            auto r = std::make_unique<ConversationPageReader>();
            if (!r->open(e.file.string(), layout, *pages_, id, limits, error)) {
                note_restore_failure();
                return false;
            }
            out = std::move(r);
        } else {
            auto r = std::make_unique<SessionFileStream>();
            if (!r->open(e.file.string(), id, limits, error)) {
                note_restore_failure();
                return false;
            }
            out = std::move(r);
        }
        (void) persistent;
        return true;
    }

    /// The entry a restore is reading right now: a save never evicts it (it reports "no room" when only it is left).
    /// 0 clears the guard.  The disk tier's DiskRestoreMark, kept by the same name and rule.
    void guard(uint64_t serial) { std::lock_guard<std::mutex> lk(mu_); guarded_ = serial; }
    void unguard() { std::lock_guard<std::mutex> lk(mu_); guarded_ = 0; }
    uint64_t guarded() const { std::lock_guard<std::mutex> lk(mu_); return guarded_; }

    SpillStats stats() const { std::lock_guard<std::mutex> lk(mu_); return stats_; }
    void note_restore(uint64_t ms, uint64_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.restores;
        stats_.last_restore_ms = ms;
        stats_.last_restore_bytes = bytes;
        stats_.bytes_read += bytes;
    }
    void note_restore_failure() { std::lock_guard<std::mutex> lk(mu_); ++stats_.restore_failures; }
    /// The free-space preflight a save uses (the session writer's `min_free_bytes`); 0 keeps upstream's (none).
    void set_min_free_bytes(uint64_t n) { std::lock_guard<std::mutex> lk(mu_); min_free_ = n; }

    /// At most N entries (--conversation-spill-slots; 0 = no count limit).  Trims the least recently used entries at
    /// once and after every save: the disk tier's slot cap, which upstream's bytes-only budget does not have.
    void set_max_entries(size_t n) { std::lock_guard<std::mutex> lk(mu_); max_entries_ = n; trim_locked(); }
    bool paged() const { std::lock_guard<std::mutex> lk(mu_); return pages_ != nullptr; }
    const std::string& page_open_error() const { return page_error_; }
    ConversationPagePool* pool() { std::lock_guard<std::mutex> lk(mu_); return pages_.get(); }
    uint64_t pool_bytes() const { std::lock_guard<std::mutex> lk(mu_); return pages_ ? pages_->stats().pool_bytes : 0; }
    uint64_t pool_dead_bytes() const {
        std::lock_guard<std::mutex> lk(mu_);
        return pages_ ? pages_->stats().dead_bytes : 0;
    }
    bool compact_needed() const {
        std::lock_guard<std::mutex> lk(mu_);
        return pages_ && pages_->compact_needed(page_opts_.compact_min_bytes, page_opts_.compact_min_pages,
                                               page_opts_.compact_fragmentation);
    }
    /// Reclaims the pool's dead pages when all three guards pass; entries never change (a page is found by its hash).
    bool compact(std::string& error) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!pages_) { error = "the page pool is off"; return false; }
        if (!pages_->compact(page_opts_.compact_min_bytes, page_opts_.compact_min_pages,
                             page_opts_.compact_fragmentation, error))
            return false;
        const auto s = pages_->stats();
        ++stats_.compactions;
        stats_.bytes_compacted = s.bytes_compacted;
        stats_.last_compaction_ms = s.last_compaction_ms;
        stats_.last_compacted_bytes = s.last_compacted_bytes;
        return true;
    }

private:
    bool enabled_locked() const { return ns_.is_open() && budget_.cap > 0; }
    size_t index_locked(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].serial == serial) return i;
        return SIZE_MAX;
    }
    bool persistent_now() const { std::lock_guard<std::mutex> lk(mu_); return ns_.persistent(); }
    static std::filesystem::path meta_path(std::filesystem::path f) { return f.replace_extension(".meta"); }
    size_t lru_locked() const {
        size_t b = SIZE_MAX;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].serial == guarded_) continue;   // (a restore is reading it)
            if (b == SIZE_MAX || entries_[i].used < entries_[b].used) b = i;
        }
        return b;
    }
    /// Evicts until `need` fits (mu_ held): an entry the new image extends goes first (it would go anyway), then the
    /// least recently used.  Never the entry a restore is reading (guard).  False: only guarded entries are left.
    bool make_room_locked(uint64_t need, const SavedConversation& image, std::string& error) {
        while (!budget_.reserve(need)) {
            size_t victim = SIZE_MAX;
            for (size_t i = 0; i < entries_.size() && victim == SIZE_MAX; ++i)
                if (superseded_by(entries_[i], image)) victim = i;
            if (victim == SIZE_MAX) victim = lru_locked();
            if (victim == SIZE_MAX) { error = "no room in the SSD tier"; return false; }
            ++stats_.evictions;
            drop_locked(victim);
        }
        return true;
    }
    /// The slot cap (--conversation-spill-slots; mu_ held and the guard respected, as for any eviction).
    void trim_locked() {
        while (max_entries_ > 0 && entries_.size() > max_entries_) {
            const size_t victim = lru_locked();
            if (victim == SIZE_MAX) break;
            ++stats_.evictions;
            drop_locked(victim);
        }
    }
    /// `e` is a prefix (equal or shorter) of `image`'s live tokens, same steering and pictures: `image` serves every resume `e` could
    static bool superseded_by(const Entry& e, const SavedConversation& image) {
        const size_t n = e.live.ids.size();
        if (n == 0 || e.cvec != image.cvec || e.lora != image.lora ||   // (an adapter state is its own conversation)
            n > image.live.ids.size() ||
            !std::equal(e.live.ids.begin(), e.live.ids.end(), image.live.ids.begin())) return false;
        for (const auto& k : e.live.imgs) {   // its pictures are the first ones of the image's
            bool found = false;
            for (const auto& m : image.live.imgs) if (m == k) { found = true; break; }
            if (!found) return false;
        }
        size_t within = 0;
        for (const auto& m : image.live.imgs) if (m.start < (int64_t) n) ++within;
        return within == e.live.imgs.size();
    }
    // The index file beside a persistent entry (`conv-N.meta`): what matching needs, so a restart adopts the entry without reading
    // its session file.  Written after the session file is complete, by a temporary name and a rename.
    template <class T> static void put_pod(std::string& b, const T& v) { b.append((const char*) &v, sizeof v); }
    static void put_ckpt(std::string& b, const ConversationCheckpoint& c) {
        put_pod(b, (uint64_t) c.ids.size());
        b.append((const char*) c.ids.data(), c.ids.size() * sizeof(int32_t));
        put_pod(b, (uint64_t) c.imgs.size());
        for (const auto& k : c.imgs) { put_pod(b, k.start); put_pod(b, k.hash); }
    }
    static bool meta_write(const std::filesystem::path& f, const Entry& e, const SessionFileIdentity& id) {
        // SPM2 (local): the adapter state and the paged flag/bytes joined the index, so a restart's adoption filters on
        // the same key matching does.  An SPM1 index file is refused and deleted at adoption: this folder is a cache.
        std::string b = "SPM2";
        put_pod(b, id.model); put_pod(b, id.config);
        put_pod(b, (uint8_t) e.cvec); put_pod(b, (uint8_t) e.paged); put_pod(b, (uint8_t) e.lora); put_pod(b, (uint8_t) 0);
        put_pod(b, e.bytes); put_pod(b, (uint64_t) e.kv_bytes);
        put_ckpt(b, e.live);
        put_pod(b, (uint64_t) e.checkpoints.size());
        for (const auto& c : e.checkpoints) put_ckpt(b, c);
        const std::filesystem::path m = meta_path(f);
        std::filesystem::path t = m;
        t += ".tmp";
        {
            std::FILE* fp = std::fopen(t.string().c_str(), "wb");
            if (!fp) return false;
            const bool ok = std::fwrite(b.data(), 1, b.size(), fp) == b.size();
            if (std::fclose(fp) != 0 || !ok) { std::error_code ec; std::filesystem::remove(t, ec); return false; }
        }
        std::error_code ec;
        std::filesystem::rename(t, m, ec);
        if (ec) { std::filesystem::remove(t, ec); return false; }
        return true;
    }
    struct Rd {
        const std::string& b; size_t at = 0; bool bad = false;
        template <class T> T pod() { T v{}; if (bad || at + sizeof v > b.size()) { bad = true; return v; } std::memcpy(&v, b.data() + at, sizeof v); at += sizeof v; return v; }
    };
    static bool get_ckpt(Rd& r, ConversationCheckpoint& c, uint64_t file_bytes) {
        const uint64_t n = r.pod<uint64_t>();
        if (r.bad || n > file_bytes || r.at + n * sizeof(int32_t) > r.b.size()) return false;
        c.ids.resize((size_t) n);
        std::memcpy(c.ids.data(), r.b.data() + r.at, (size_t) n * sizeof(int32_t));
        r.at += (size_t) n * sizeof(int32_t);
        const uint64_t m = r.pod<uint64_t>();
        if (r.bad || m > r.b.size()) return false;
        for (uint64_t i = 0; i < m; ++i) { ConversationImageKey k; k.start = r.pod<int64_t>(); k.hash = r.pod<uint64_t>(); c.imgs.push_back(k); }
        return !r.bad;
    }
    /// Lists the entries a previous run left in the stable folder (mu_ held): a session file whose index file reads back, whose size
    /// is the one written and whose identity is `id` is adopted, oldest use (the index file's time) first; everything else in the
    /// folder - another identity, a half-written or orphaned file - is deleted.
    size_t adopt_locked(const SessionFileIdentity& id) {
        namespace fs = std::filesystem;
        struct Found { Entry e; fs::file_time_type t; };
        std::vector<Found> found;
        std::error_code ec;
        std::vector<fs::path> all;
        for (fs::directory_iterator it(ns_.dir(), ec), end; !ec && it != end; it.increment(ec)) all.push_back(it->path());
        auto serial_of = [](const fs::path& p, uint64_t& out) {
            const std::string n = p.stem().string();   // conv-<serial>
            if (n.size() < 6 || n.compare(0, 5, "conv-") != 0) return false;
            out = 0;
            for (size_t i = 5; i < n.size(); ++i) { if (n[i] < '0' || n[i] > '9') return false; out = out * 10 + (uint64_t)(n[i] - '0'); }
            return true;
        };
        std::vector<uint64_t> seen;
        for (const auto& p : all) {
            if (p.extension() != ".meta") continue;
            uint64_t serial = 0;
            std::error_code e2;
            const fs::path sess = fs::path(p).replace_extension(".session");
            bool good = serial_of(p, serial) && fs::is_regular_file(sess, e2) && fs::is_regular_file(p, e2);
            Found f;
            if (good) {
                std::string b;
                if (std::FILE* fp = std::fopen(p.string().c_str(), "rb")) {
                    char buf[65536];
                    size_t n;
                    while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) { b.append(buf, n); if (b.size() > (size_t(1) << 28)) break; }
                    std::fclose(fp);
                } else good = false;
                Rd r{b};
                good = good && b.compare(0, 4, "SPM2") == 0;
                if (good) {
                    r.at = 4;
                    const uint64_t model = r.pod<uint64_t>(), config = r.pod<uint64_t>();
                    f.e.cvec = r.pod<uint8_t>() != 0;
                    f.e.paged = r.pod<uint8_t>() != 0;
                    f.e.lora = (ConversationLora) r.pod<uint8_t>();
                    (void) r.pod<uint8_t>();
                    f.e.bytes = r.pod<uint64_t>();
                    f.e.kv_bytes = r.pod<uint64_t>();
                    const bool known_lora = f.e.lora == kLoraNone || f.e.lora == kLoraOn || f.e.lora == kLoraOff;
                    good = !r.bad && known_lora && model == id.model && config == id.config &&
                           fs::file_size(sess, e2) != 0 && !e2 && get_ckpt(r, f.e.live, f.e.bytes);
                    const uint64_t nc = good ? r.pod<uint64_t>() : 0;
                    for (uint64_t i = 0; good && i < nc; ++i) { ConversationCheckpoint c; good = get_ckpt(r, c, f.e.bytes); if (good) f.e.checkpoints.push_back(std::move(c)); }
                    good = good && !r.bad;
                    // A paged entry needs its page list and the pool; a whole-file one names the session file's size.
                    if (good && f.e.paged) {
                        std::string why;
                        good = pages_ != nullptr && read_page_index(sess, id, f.e.layout, why) &&
                               layout_kv_bytes(f.e.layout) == f.e.kv_bytes;
                    } else if (good) {
                        good = fs::file_size(sess, e2) == f.e.bytes;
                    }
                }
                if (good) { f.e.serial = serial; f.e.file = sess; f.t = fs::last_write_time(p, e2); good = !e2; }
            }
            if (good) { seen.push_back(serial); found.push_back(std::move(f)); }
            else { fs::remove(p, e2); fs::remove(sess, e2); }
        }
        for (const auto& p : all) {   // whatever is not an adopted entry's file (a .session without index, a leftover temporary)
            if (p.filename() == ".lock") continue;
            if (p.filename() == "pool") continue;   // (the page pool: it follows its own entries and its version)
            uint64_t serial = 0;
            const bool listed = (p.extension() == ".meta" || p.extension() == ".session" || p.extension() == ".pages") &&
                                serial_of(p, serial) && std::find(seen.begin(), seen.end(), serial) != seen.end();
            if (!listed) { std::error_code e2; fs::remove_all(p, e2); }
        }
        std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t != b.t ? a.t < b.t : a.e.serial < b.e.serial; });
        for (auto& f : found) {
            f.e.used = ++tick_;
            if (f.e.serial >= next_serial_) next_serial_ = f.e.serial + 1;
            budget_.used += f.e.bytes;
            entries_.push_back(std::move(f.e));
        }
        while (budget_.over() && entries_.size() > 1) drop_locked(lru_locked());   // (the cap is now smaller than what was kept)
        return entries_.size();
    }
    void drop_locked(size_t i) {
        if (i >= entries_.size() || entries_[i].serial == guarded_) return;   // (a restore is reading it)
        const Entry& e = entries_[i];
        if (e.paged && pages_) release_pages(e);
        std::error_code ec;
        std::filesystem::remove(meta_path(e.file), ec);
        std::filesystem::remove(e.file, ec);
        if (e.paged) std::filesystem::remove(pages_path(e.file), ec);
        budget_.drop(e.bytes);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
    }
    /// Drops this entry's claims on the pool's pages (the pages themselves stay until a compaction reclaims them).
    void release_pages(const Entry& e) {
        for (const auto& l : e.layout)
            for (const auto& part : l.parts)
                for (const auto& ref : part) pages_->release(ref.hash, ref.bytes);
        std::string ignored;
        pages_->persist(ignored);
    }
    static ConversationCheckpoint stub(const ConversationCheckpoint& c) {
        ConversationCheckpoint s;
        s.ids = c.ids;
        s.imgs = c.imgs;
        return s;
    }

    // ---- local: the page entry's sidecar -------------------------------------------------------------------------
    // `conv-<serial>.pages` beside the head: magic "SPG1", the identity, the page list, a checksum.  Written by a
    // temporary name and a rename after the head and the pool are complete, so a paged entry exists only when all
    // three do.  It is the paged entry's own index - the meta (SPM2) carries only the flag and the logical size.
    static std::filesystem::path pages_path(std::filesystem::path f) { return f.replace_extension(".pages"); }
    static uint64_t layout_kv_bytes(const std::vector<ConversationPageLayer>& layers) {
        uint64_t n = 0;
        for (const auto& l : layers)
            for (uint64_t s : l.sizes) n += s;
        return n;
    }
    static void put_page_layout(std::string& b, const std::vector<ConversationPageLayer>& layers) {
        put_pod(b, (uint64_t) layers.size());
        for (const auto& l : layers) {
            put_pod(b, (int64_t) l.format);
            put_pod(b, (int64_t) l.cells); put_pod(b, (int64_t) l.heads); put_pod(b, (int64_t) l.head_dim);
            put_pod(b, (int64_t) l.page_size); put_pod(b, (int64_t) l.pooled_rows); put_pod(b, (int64_t) l.idx_dim);
            for (uint64_t s : l.sizes) put_pod(b, s);
            for (const auto& part : l.parts) {
                put_pod(b, (uint64_t) part.size());
                for (const auto& r : part) { put_pod(b, r.hash); put_pod(b, (uint32_t) r.bytes); }
            }
        }
    }
    bool write_page_index(const std::filesystem::path& f, const SessionFileIdentity& id,
                          const std::vector<ConversationPageLayer>& layers, std::string& error) const {
        std::string b = "SPG1";
        put_pod(b, id.model); put_pod(b, id.config);
        put_page_layout(b, layers);
        put_pod(b, session_hash64(b.data(), b.size(), 0));
        const std::filesystem::path p = pages_path(f);
        std::filesystem::path t = p;
        t += ".tmp";
        {
            std::FILE* fp = std::fopen(t.string().c_str(), "wb");
            if (!fp) { error = "the entry's page index cannot be created"; return false; }
            const bool ok = std::fwrite(b.data(), 1, b.size(), fp) == b.size();
            if (std::fclose(fp) != 0 || !ok) {
                std::error_code ec; std::filesystem::remove(t, ec);
                error = "the entry's page index cannot be written";
                return false;
            }
        }
        std::error_code ec;
        std::filesystem::rename(t, p, ec);
        if (ec) { std::filesystem::remove(t, ec); error = "the entry's page index cannot be published"; return false; }
        return true;
    }
    static bool read_page_index(const std::filesystem::path& f, const SessionFileIdentity& id,
                                 std::vector<ConversationPageLayer>& out, std::string& why) {
        std::string b;
        {
            std::FILE* fp = std::fopen(pages_path(f).string().c_str(), "rb");
            if (!fp) { why = "no page index"; return false; }
            char buf[65536];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) { b.append(buf, n); if (b.size() > (size_t(1) << 26)) break; }
            std::fclose(fp);
        }
        if (b.size() < 8 + 16 + 8 || b.compare(0, 4, "SPG1") != 0) { why = "not a page index"; return false; }
        uint64_t sum = 0;
        std::memcpy(&sum, b.data() + b.size() - 8, 8);
        if (sum != session_hash64(b.data(), b.size() - 8, 0)) { why = "page index checksum mismatch"; return false; }
        Rd r{b};
        r.at = 4;
        const uint64_t model = r.pod<uint64_t>(), config = r.pod<uint64_t>();
        if (model != id.model || config != id.config) { why = "page index: another identity"; return false; }
        const uint64_t nl = r.pod<uint64_t>();
        if (r.bad || nl > 64) { why = "page index: bad layer count"; return false; }
        out.clear();
        for (uint64_t i = 0; i < nl; ++i) {
            ConversationPageLayer l;
            l.format = (int) r.pod<int64_t>();
            l.cells = r.pod<int64_t>(); l.heads = r.pod<int64_t>(); l.head_dim = r.pod<int64_t>();
            l.page_size = r.pod<int64_t>(); l.pooled_rows = r.pod<int64_t>(); l.idx_dim = r.pod<int64_t>();
            for (uint64_t& s : l.sizes) s = r.pod<uint64_t>();
            for (auto& part : l.parts) {
                const uint64_t n = r.pod<uint64_t>();
                if (r.bad || n > (b.size() / 12)) { why = "page index: bad page count"; return false; }
                part.resize((size_t) n);
                for (auto& ref : part) { ref.hash = r.pod<uint64_t>(); ref.bytes = r.pod<uint32_t>(); }
            }
            if (r.bad) { why = "page index: truncated"; return false; }
            out.push_back(std::move(l));
        }
        if (r.at != b.size() - 8) { why = "page index: trailing bytes"; return false; }
        return true;
    }

    mutable std::mutex mu_;
    SpillNamespace ns_;
    SpillBudget budget_;
    std::vector<Entry> entries_;
    uint64_t next_serial_ = 1, tick_ = 0;
    // local additions
    std::unique_ptr<ConversationPagePool> pages_;   // null: whole-file entries only (page_options off or refused)
    SpillPageOptions page_opts_;
    std::string page_error_;
    uint64_t guarded_ = 0;    // the entry a restore is reading: never evicted (guard())
    uint64_t min_free_ = 0;   // the session writer's free-space preflight for a save
    size_t max_entries_ = 0;  // the slot cap (set_max_entries; 0 = no count limit)
    SpillStats stats_;
};

/// The best prefix a parked image gives `prompt`, by the RAM cache's rule (its live state or a checkpoint, same pictures, same
/// steering): for the images the background writer still holds in RAM.  {tokens, live}; tokens 0 = none.
template <class Token>
std::pair<int64_t, bool> conversation_image_prefix(const SavedConversation& s, const std::vector<Token>& prompt,
                                                   const std::vector<ConversationImageKey>& images, bool cvec) {
    std::pair<int64_t, bool> m{0, false};
    if (s.cvec != cvec) return m;
    const int64_t l = conversation_prefix(s.live, prompt, images);
    if (l > m.first) m = {l, true};
    for (const auto& c : s.checkpoints) {
        const int64_t n = conversation_prefix(c, prompt, images);
        if (n > m.first) m = {n, false};
    }
    return m;
}

/// P2 step 2c: writes parked conversations to the SSD tier on its own thread, so a request's park does not wait for the disk.
/// At most one image is written at a time and one waits behind it (push() blocks past that): what the writer holds in RAM is at
/// most two images over the RAM tier's budget, and the physical-RAM gate sees it (it is still allocated).  A written image is not
/// freed on the writer thread: it is handed back (take_done) to be freed - or reused, it is still in RAM - on the caller's
/// thread, the one that owns the cache's allocators.  `write` runs on the writer thread and returns the SSD serial (0: failed).
template <class Item>
class BackgroundWriter {
public:
    struct Done {
        Item item;
        uint64_t serial = 0;   // its SSD entry; 0: the write failed (the image is all there is)
    };
    /// `on_start`: runs first on the writer thread (the serve loop lowers its priority there: spill_detail::lower_thread_priority)
    explicit BackgroundWriter(std::function<uint64_t(const Item&)> write, std::function<void()> on_start = {})
        : write_(std::move(write)), on_start_(std::move(on_start)), th_([this] { if (on_start_) on_start_(); run(); }) {}
    BackgroundWriter(const BackgroundWriter&) = delete;
    BackgroundWriter& operator=(const BackgroundWriter&) = delete;
    /// what is queued is not written (the namespace goes with the process); a write in flight finishes first
    ~BackgroundWriter() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            queue_.clear();
        }
        cv_.notify_all();
        th_.join();
    }
    void push(Item&& it) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return queue_.empty() || stop_; });
        if (stop_) return;
        queue_.push_back(std::move(it));
        cv_.notify_all();
    }
    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return (queue_.empty() && !busy_) || stop_; });
    }
    bool idle() const { std::lock_guard<std::mutex> lk(mu_); return queue_.empty() && !busy_; }
    /// the written ones, to free or reuse on this thread
    std::vector<Done> take_done() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Done> out;
        out.swap(done_);
        return out;
    }
    /// the queued ones (not the one in flight) back, unwritten - #879 clears them with the cache
    std::vector<Item> take_queued() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Item> out;
        for (auto& q : queue_) out.push_back(std::move(q));
        queue_.clear();
        cv_.notify_all();
        return out;
    }
    /// f(const Item&) over every image not yet handed back - queued, in flight, written - under the lock (the writer only
    /// reads an image in flight, so reading it here as well is safe)
    template <class F>
    void for_each_held(F f) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& q : queue_) f(q);
        if (inflight_) f(*inflight_);
        for (const auto& d : done_) f(d.item);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            inflight_.emplace(std::move(queue_.front()));
            queue_.pop_front();
            busy_ = true;
            cv_.notify_all();   // (a push waiting for room)
            lk.unlock();
            uint64_t serial = 0;
            try { serial = write_(*inflight_); } catch (...) { serial = 0; }
            lk.lock();
            done_.push_back(Done{std::move(*inflight_), serial});
            inflight_.reset();
            busy_ = false;
            cv_.notify_all();
        }
    }
    std::function<uint64_t(const Item&)> write_;
    std::function<void()> on_start_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> queue_;
    std::optional<Item> inflight_;
    std::vector<Done> done_;
    bool busy_ = false, stop_ = false;
    std::thread th_;   // (last: it starts once the rest exists)
};

}  // namespace strata::core
