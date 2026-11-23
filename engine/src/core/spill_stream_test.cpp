// CPU-only tests for the spill tier's local additions (include/strata/core/conversation_spill.hpp): the streaming save
// (put_streaming), the layered restore (take_layers), the page backend, the adapter key, the slot cap, the guard and the
// counters.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model, no network.
// The checks the retired conversation_disk_test.cpp held that still describe the spill tier live here
// (docs/work/local-spill-delta.md).
#include "strata/core/conversation_spill.hpp"

#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

SessionFileIdentity identity(uint64_t model = 7, uint64_t config = 9) { return {model, config}; }

fs::path unique_root() {
    static int n = 0;
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path p = fs::temp_directory_path() /
                       ("strata-spill-tier-" + std::to_string(++n) + "-" + std::to_string(ticks));
    std::error_code ec;
    fs::remove_all(p, ec);
    return p;
}

std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void write_all(const fs::path& p, const std::string& bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), (std::streamsize) bytes.size());
}

std::vector<std::vector<uint8_t>> parts_of(int seed) {
    std::vector<std::vector<uint8_t>> parts(2);
    parts[0].resize(kConversationPageBytes * 2 + kConversationPageBytes / 2);
    parts[1].resize(kConversationPageBytes);
    for (size_t i = 0; i < parts[0].size(); ++i) parts[0][i] = (uint8_t) (i * 7 + (i >> 9) + seed);
    for (size_t i = 0; i < parts[1].size(); ++i) parts[1][i] = (uint8_t) (i * 11 + (i >> 10) + seed);
    return parts;
}

SessionKvSource make_source(const std::vector<std::vector<uint8_t>>& p) {
    SessionKvSource src;
    src.format = 2;
    src.cells = 128;
    src.heads = 4;
    src.head_dim = 64;
    src.page_size = 4;
    src.pooled_rows = 0;
    src.idx_dim = 128;
    src.sizes = {p[0].size(), p[1].size(), 0, 0, 0};
    src.read = [p](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= p.size() || offset > p[part].size() || n > p[part].size() - offset) return false;
        std::memcpy(dst, p[part].data() + offset, n);
        return true;
    };
    return src;
}

SavedConversation head_for(std::initializer_list<int32_t> ids, bool cvec = true, ConversationLora lora = kLoraNone) {
    SavedConversation head;
    head.geometry = {128, 2, 1, 0, 0, 0, 0, 0, 0, 4, 4, 64, 4, 128, 0, 0, 0, 0};
    head.layer_lo = 0;
    head.layer_hi = 0;
    head.live.ids = ids;
    head.live.imgs = {};
    head.cvec = cvec;
    head.lora = lora;
    return head;
}

// An image with no K/V: whole-image put()/take_serial() (upstream's path) stay coverable without a GPU.
SavedConversation image_for(std::initializer_list<int32_t> ids, bool cvec = true, ConversationLora lora = kLoraNone) {
    return head_for(ids, cvec, lora);
}

bool buffer_bytes(const ConversationBuffer& buffer, std::vector<uint8_t>& out) {
    out.resize(buffer.size());
    return buffer.size() == 0 || buffer.read(out.data(), 0, buffer.size());
}

// One layered restore: the single layer's first two parts must be `want`.
bool restore_layers(ConversationSpill& tier, uint64_t serial, const SessionFileIdentity& id,
                    const std::vector<std::vector<uint8_t>>& want, std::string& why) {
    std::unique_ptr<ConversationKvLayers> reader;
    if (!tier.take_layers(serial, id, SessionReadLimits{}, reader, why)) return false;
    if (reader->layers() != 1) { why = "unexpected layer count"; return false; }
    ConversationKv layer;
    if (!reader->next(layer, why)) return false;
    std::vector<uint8_t> a, b;
    if (!buffer_bytes(layer.k, a) || !buffer_bytes(layer.v, b)) { why = "a restored part is unreadable"; return false; }
    if (a != want[0] || b != want[1]) { why = "the restored K/V bytes differ"; return false; }
    return reader->finish(why);
}

} // namespace

int main() {
    const auto parts = parts_of(3);
    const std::vector<SessionKvSource> sources = {make_source(parts)};

    // ---- the adapter key: put/best/supersede never cross an adapter state (the bug the disk tier fixed) ----
    {
        const fs::path root = unique_root();
        std::string err;
        {
            ConversationSpill tier;
            check(tier.open(root, 4u << 20, err), "the tier opens");
            uint64_t a = 0, b = 0;
            check(tier.put(image_for({1, 2, 3, 4}), identity(), err, &a), "save a base-model entry");
            check(tier.put(image_for({1, 2, 3, 4}, true, kLoraOn), identity(), err, &b), "save the same prefix adapted");
            const std::vector<int32_t> prompt = {1, 2, 3, 4, 9};
            check(tier.best(prompt, {}, true).serial == a && tier.best(prompt, {}, true).tokens == 4,
                  "a base request matches the base entry, not the newer adapted one");
            check(tier.best(prompt, {}, true, kLoraOn).serial == b, "an adapted request matches its own entry");
            check(tier.best(prompt, {}, true, kLoraOff).tokens == 0, "the other adapter state matches nothing");
            // the same conversation grown under another state does not supersede the base entry
            check(tier.put(image_for({1, 2, 3, 4, 5}, true, kLoraOff), identity(), err), "save a grown entry, off");
            check(tier.find(a).has_value() && tier.find(b).has_value(), "neither earlier entry was superseded");
            // the grown base one does supersede the base entry
            check(tier.put(image_for({1, 2, 3, 4, 5}), identity(), err), "save the grown base entry");
            check(!tier.find(a).has_value(), "the grown base entry supersedes the shorter base one");
            check(tier.find(b).has_value(), "the adapted entry survives the base save");
            check(tier.stats().saves == 4, "every save is counted");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- streaming save and layered restore, whole-file backend (no page pool) ----
    {
        const fs::path root = unique_root();
        std::string err;
        {
            ConversationSpill tier;
            check(tier.open(root, 16u << 20, err), "the tier opens for the whole-file streaming save");
            check(!tier.paged(), "the page pool is off without its option");
            SpillSaveReport rep;
            uint64_t serial = 0;
            check(tier.put_streaming(head_for({1, 2, 3, 4}), sources, identity(), err, &serial, &rep),
                  "a streaming save writes the head and the streamed K/V");
            check(serial != 0 && rep.pages_written == 0, "a whole-file save reports no pages");
            check(rep.file_bytes > parts[0].size() + parts[1].size(), "the entry's logical size covers the K/V");
            check(restore_layers(tier, serial, identity(), parts, err), "the streamed bytes restore layer by layer");
            check(tier.find(serial).has_value(), "a layered restore keeps the entry for the next save");
            check(tier.stats().saves == 1 && tier.stats().restores == 0, "the counters see one save");
            // a failed restore of an unknown serial is a miss, not an error state
            std::unique_ptr<ConversationKvLayers> reader;
            check(!tier.take_layers(999999, identity(), SessionReadLimits{}, reader, err), "an unknown serial is a miss");
            // another identity cannot read it
            check(!tier.take_layers(serial, identity(1, 2), SessionReadLimits{}, reader, err),
                  "another identity cannot open the entry");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- the page backend: a CoW save, an unchanged save, one changed page, a layered restore, adoption ----
    {
        const fs::path root = unique_root();
        std::string err;
        const SessionFileIdentity id = identity();
        SpillPageOptions pages;
        pages.enabled = true;
        pages.compact_min_bytes = 0;
        pages.compact_min_pages = 1;
        pages.compact_fragmentation = 0.0;
        auto owner = std::make_unique<ConversationSpill>();
        ConversationSpill& tier = *owner;
        size_t adopted = 0;
        check(tier.open(root, 16u << 20, err, nullptr, &id, &adopted, &pages), "the tier opens with the page pool");
        check(tier.paged() && tier.persistent(), "the page pool is on and the folder is stable");
        SpillSaveReport first;
        uint64_t serial = 0;
        check(tier.put_streaming(head_for({1, 2, 3, 4}), sources, identity(), err, &serial, &first),
              "a paged save writes a head and pages");
        check(first.pages_written == 4 && first.bytes_reused == 0, "the first paged save writes every page");
        auto match = tier.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
        check(match.tokens == 4 && match.live && match.serial == serial, "the paged entry matches its live state");
        check(tier.find(serial)->paged && tier.find(serial)->bytes == first.file_bytes,
              "the entry counts its head and the K/V it names");
        check(restore_layers(tier, serial, identity(), parts, err), "the paged entry restores layer by layer");

        SpillSaveReport second;
        check(tier.put_streaming(head_for({1, 2, 3, 4, 5}), sources, identity(), err, nullptr, &second),
              "the CoW save succeeds");
        check(second.pages_written == 0 && second.bytes_reused == parts[0].size() + parts[1].size(),
              "an unchanged K/V writes no page bytes");
        auto changed = parts;
        changed[0][kConversationPageBytes + 10] ^= 0xff;
        const std::vector<SessionKvSource> changed_sources = {make_source(changed)};
        SpillSaveReport third;
        uint64_t grown = 0;
        check(tier.put_streaming(head_for({1, 2, 3, 4, 5, 6}), changed_sources, identity(), err, &grown, &third),
              "a changed page saves");
        check(third.pages_written == 1, "only the changed page is written");
        check(third.bytes_reused == parts[0].size() - kConversationPageBytes + parts[1].size(),
              "the pages before and after it are reused");
        check(tier.pool_bytes() > 0, "the pool reports its bytes");
        check(restore_layers(tier, grown, identity(), changed, err), "the changed entry restores the changed bytes");

        // A persistent restart adopts the paged entries (the meta holds the adapter state and the page flag).
        const size_t entries_before = tier.size();
        const uint64_t pool_before = tier.pool_bytes();
        owner.reset();   // one tier per pool at a time (Windows keeps a rename-blocking handle)
        {
            ConversationSpill again;
            size_t reopened_adopted = 0;
            check(again.open(root, 16u << 20, err, nullptr, &id, &reopened_adopted, &pages),
                  "the persistent paged tier opens again");
            check(again.persistent() && reopened_adopted == entries_before,
                  "every paged entry is adopted from the last run");
            check(again.pool_bytes() == pool_before, "the restart keeps the pool's bytes");
            check(restore_layers(again, grown, id, changed, err), "the adopted paged entry restores the changed bytes");
            check(again.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 9}, {}, true).tokens == 6,
                  "the adopted entry matches its live state");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- the format version: an SPM1 index file (the previous spill cache) is discarded, not migrated ----
    {
        const fs::path root = unique_root();
        std::string err;
        const SessionFileIdentity id = identity();
        size_t adopted = 0;
        {
            ConversationSpill tier;
            check(tier.open(root / "ns", 8u << 20, err, nullptr, &id, &adopted) && tier.persistent(),
                  "a persistent tier opens for the version check");
            check(tier.put(image_for({1, 2, 3}), id, err), "a persistent entry is written");
            // rewrite its index file with the old magic: the next open must delete it rather than read it
            std::error_code ec;
            size_t rewritten = 0;
            for (fs::directory_iterator it(tier.dir(), ec), end; !ec && it != end; it.increment(ec)) {
                if (it->path().extension() != ".meta") continue;
                std::string bytes = read_all(it->path());
                if (bytes.size() >= 4) std::memcpy(bytes.data(), "SPM1", 4);
                write_all(it->path(), bytes);
                ++rewritten;
            }
            check(rewritten == 1, "the entry's index file was rewritten as SPM1");
        }
        {
            ConversationSpill tier;
            check(tier.open(root / "ns", 8u << 20, err, nullptr, &id, &adopted) && adopted == 0,
                  "an SPM1 index file is discarded at adoption");
            check(tier.size() == 0, "the old entry is gone");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- the slot cap, the restore guard, clear and the counters ----
    {
        const fs::path root = unique_root();
        std::string err;
        ConversationSpill tier;
        check(tier.open(root, 4u << 20, err), "the tier opens for the slot cap");
        tier.set_max_entries(2);
        check(tier.put(image_for({1, 2}), identity(), err), "save A");
        check(tier.put(image_for({3, 4}), identity(), err), "save B");
        check(tier.put(image_for({5, 6}), identity(), err), "save C, one over the cap");
        check(tier.size() == 2, "the slot cap trims the third entry");
        check(tier.stats().evictions == 1, "the slot trim is counted as an eviction");
        check(tier.best(std::vector<int32_t>{1, 2, 9}, {}, true).tokens == 0, "the oldest entry went");

        // The guard: a save never evicts the entry a restore is reading.
        const uint64_t keep = tier.best(std::vector<int32_t>{5, 6, 9}, {}, true).serial;
        tier.guard(keep);
        check(tier.guarded() == keep, "the tier reports the guarded entry");
        check(tier.put(image_for({7, 8}), identity(), err), "a save beside the guarded entry succeeds");
        check(tier.find(keep).has_value(), "the guarded entry survives");
        tier.unguard();
        check(tier.guarded() == 0, "unguard clears the guard");

        const size_t n = tier.size();
        check(tier.clear() == n && tier.size() == 0, "clear drops every entry and reports how many");
        check(tier.stats().clearances == n, "the clearance is counted");
    }

    // ---- the background writer: the timed flush's trigger (one image in flight, one queued) ----
    {
        const fs::path root = unique_root();
        std::string err;
        const SessionFileIdentity id = identity();
        ConversationSpill tier;
        check(tier.open(root, 8u << 20, err), "the tier opens for the writer");
        std::atomic<int> written{0};
        {
            BackgroundWriter<SavedConversation> writer(
                [&](const SavedConversation& img) -> uint64_t {
                    std::string e;
                    uint64_t serial = 0;
                    const bool ok = tier.put(img, id, e, &serial);
                    if (ok) ++written;
                    return ok ? serial : 0;
                });
            writer.push(image_for({1, 2, 3}));
            writer.push(image_for({4, 5, 6}));
            writer.wait_idle();
            auto done = writer.take_done();
            check(done.size() == 2 && written == 2, "both queued images were written");
            check(done[0].serial != 0 && done[1].serial != 0, "each done item carries its serial");
            check(tier.size() == 2, "the tier lists both");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    std::fprintf(stderr, "spill_stream_test: %d checks passed\n", checks);
    return 0;
}
