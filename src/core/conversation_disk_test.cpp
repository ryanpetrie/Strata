// CPU-only tests for the disk tier of the conversation cache (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model, no network.
#include "strata/core/conversation_disk.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
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

ConversationDiskOptions options(const fs::path& dir, uint64_t budget, size_t slots = 0) {
    ConversationDiskOptions o;
    o.dir = dir.string();
    o.budget_bytes = budget;
    o.min_free_bytes = 0;
    o.slots = slots;
    return o;
}

ConversationDiskMeta meta(std::initializer_list<int32_t> ids, bool cvec = true,
                          ConversationLora lora = kLoraNone) {
    ConversationDiskMeta m;
    m.ids = ids;
    m.cvec = cvec;
    m.lora = lora;
    return m;
}

ConversationCheckpoint checkpoint(std::initializer_list<int32_t> ids, uint64_t used = 0, bool pinned = false) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.used = used;
    c.pinned = pinned;
    return c;
}

// Writes `n` bytes and reports them.  `fail` leaves the partial file behind after the write, the way an interrupted
// writer would, so the store has to remove it.
ConversationDiskStore::Writer writer(size_t n, std::string* path = nullptr, bool fail = false) {
    return [n, path, fail](const std::string& p, uint64_t& bytes, std::string& error) {
        if (path) *path = p;
        {
            std::ofstream f(p, std::ios::binary | std::ios::trunc);
            if (!f) { error = "cannot open"; return false; }
            const std::string data(n, 'x');
            f.write(data.data(), (std::streamsize) data.size());
            f.close();
            if (!f) { error = "short write"; return false; }
        }
        if (fail) { error = "injected failure"; return false; }
        bytes = n;
        return true;
    };
}

fs::path unique_root() {
    static int n = 0;
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path p = fs::temp_directory_path() /
                       ("strata-conversation-disk-" + std::to_string(++n) + "-" + std::to_string(ticks));
    std::error_code ec;
    fs::remove_all(p, ec);
    return p;
}

std::vector<fs::path> files_with_suffix(const fs::path& dir, const std::string& suffix) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            out.push_back(it->path());
    }
    return out;
}

std::vector<fs::path> manifest_files(const fs::path& root) { return files_with_suffix(root / "manifests", ".bin"); }

std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool buffer_bytes(const ConversationBuffer& buffer, std::vector<uint8_t>& out) {
    out.resize(buffer.size());
    return buffer.size() == 0 || buffer.read(out.data(), 0, buffer.size());
}

// One synthetic paged entry: its single layer's first two parts are the saved bytes.
bool paged_restore(ConversationDiskStore& store, uint64_t serial, const std::vector<std::vector<uint8_t>>& want,
                   std::string& why) {
    ConversationPageReader reader;
    SessionReadLimits limits;
    if (!store.open_paged(serial, reader, limits, why)) return false;
    if (reader.layers() != 1) { why = "unexpected layer count"; return false; }
    ConversationKv layer;
    if (!reader.next(layer, why)) return false;
    std::vector<uint8_t> a, b;
    if (!buffer_bytes(layer.k, a) || !buffer_bytes(layer.v, b)) { why = "a restored part is unreadable"; return false; }
    if (a != want[0] || b != want[1]) { why = "the restored K/V bytes differ"; return false; }
    return reader.finish(why);
}

void write_all(const fs::path& p, const std::string& bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), (std::streamsize) bytes.size());
}

void poke(const fs::path& p, size_t offset, uint8_t value) {
    std::string bytes = read_all(p);
    check(offset < bytes.size(), "the poke is inside the manifest");
    bytes[offset] = (char) value;
    write_all(p, bytes);
}

std::string make_tmp(const fs::path& dir, const std::string& name, const std::string& data = "x") {
    write_all(dir / name, data);
    return (dir / name).string();
}

} // namespace

int main() {
    // The live-state prefix rule is conversation_prefix() with the two lists kept apart; both must agree.
    {
        ConversationCheckpoint c;
        c.ids = {1, 2, 3, 4};
        c.imgs = {{2, 7}};
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        check(conversation_disk_live_prefix(c.ids, c.imgs, prompt, std::vector<ConversationImageKey>{{2, 7}}) ==
                  conversation_prefix(c, prompt, std::vector<ConversationImageKey>{{2, 7}}),
              "the live prefix rule matches conversation_prefix");
        check(conversation_disk_live_prefix(c.ids, c.imgs, prompt, std::vector<ConversationImageKey>{{2, 8}}) ==
                  conversation_prefix(c, prompt, std::vector<ConversationImageKey>{{2, 8}}),
              "both refuse a different image key");
        check(conversation_disk_live_prefix(c.ids, c.imgs, prompt, {}) == 0, "a missing image key is refused");
        check(conversation_disk_live_prefix(c.ids, c.imgs, c.ids, {{2, 7}}) == 0,
              "the last prompt token is never covered");
    }

    // ---- the manifest round-trip: live ids, images, checkpoints, cvec, pinned and last_used survive a restart ----
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        o.min_tokens = 1;
        std::string err;
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.enabled() && err.empty(), "the store opens on an absolute directory");
            check(s.slots() == 0 && s.entries() == 0 && s.bytes() == 0, "a new store is empty and unlimited");
            check(s.wants(1) && !s.wants(0), "min_tokens gates the disk tier too");

            ConversationDiskMeta m = meta({1, 2, 3, 4, 5});
            m.images = {{2, 0xabc}};
            m.checkpoints.push_back(checkpoint({1, 2}, 5));
            m.checkpoints.push_back(checkpoint({1, 2, 3, 4}, 9, true));
            m.cvec = false;
            uint64_t bytes = 0;
            check(s.save(m, 400, writer(400), bytes, err), "save writes the payload and the manifest");
            check(bytes == 400 && s.bytes() == 400 && s.entries() == 1, "save reports and counts its payload");
            check(s.stats().saves == 1 && s.stats().evictions == 0, "the first save evicts nothing");

            const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6};
            const std::vector<ConversationImageKey> images = {{2, 0xabc}};
            auto match = s.best(prompt, images, false, kLoraNone);
            check(match.tokens == 5 && match.live, "the live state matches");
            check(s.best(prompt, images, true, kLoraNone).tokens == 0, "cvec is part of the key");
            auto without_image = s.best(prompt, {}, false, kLoraNone);
            check(without_image.tokens == 4 && !without_image.live, "the live image keys are part of its key");
            const auto* e = s.find(match.serial);
            check(e != nullptr, "find returns the saved entry");
            check(e != nullptr && e->meta.ids == m.ids && e->meta.images == m.images && !e->meta.cvec,
                  "the live metadata round-trips in RAM");
            check(e != nullptr && e->meta.checkpoints.size() == 2, "both checkpoints are indexed");
            check(e != nullptr && e->meta.checkpoints[0].ids == m.checkpoints[0].ids &&
                      e->meta.checkpoints[0].used == 5 && !e->meta.checkpoints[0].pinned,
                  "a checkpoint keeps its ids, stamp and pin");
            check(e != nullptr && e->meta.checkpoints[1].used == 9 && e->meta.checkpoints[1].pinned,
                  "the pinned checkpoint keeps its pin");
            check(e != nullptr && e->meta.checkpoints[0].gdn.empty() && e->meta.checkpoints[0].tails.empty() &&
                      e->meta.checkpoints[0].stage_parts.empty(),
                  "no checkpoint state bytes are kept in RAM");
            check(e != nullptr && e->pinned && e->bytes == 400 && !e->payload.empty() && !e->manifest.empty(),
                  "the entry knows its files and its size");
            check(fs::exists(e->payload) && fs::exists(e->manifest), "the payload and the manifest exist");
            const uint64_t last_used = e->last_used;
            const std::string payload = e->payload;

            // The adapted twin: the same conversation's prefix under another adapter state.  Before the state
            // joined the key this save silently replaced the one above for every later request.
            ConversationDiskMeta twin = m;
            twin.lora = kLoraOn;
            check(s.save(twin, 400, writer(400), bytes, err), "the same prefix saves under another adapter state");
            check(s.best(prompt, images, false, kLoraNone).serial == match.serial,
                  "the older entry still answers its own state, though the twin is the newer tie");
            check(s.best(prompt, images, false, kLoraOn).tokens == 5, "the twin answers its state");
            check(s.best(prompt, images, false, kLoraOff).tokens == 0, "off is neither none nor on");

            ConversationDiskStore reopened(o, identity(), err);
            check(reopened.enabled() && err.empty(), "the startup scan opens the same directory");
            check(reopened.entries() == 2 && reopened.bytes() == 800, "the startup scan rebuilds the index");
            auto again = reopened.best(prompt, images, false, kLoraNone);
            check(again.tokens == 5 && again.live && again.serial != 0, "the rebuilt index matches");
            const auto* r = reopened.find(again.serial);
            check(r != nullptr && r->meta.ids == m.ids && r->meta.images == m.images && !r->meta.cvec && r->pinned,
                  "the manifest round-trips ids, images, cvec and the pin");
            check(r != nullptr && r->meta.lora == kLoraNone, "the manifest round-trips the adapter state");
            check(r != nullptr && r->meta.checkpoints.size() == 2 &&
                      r->meta.checkpoints[1].ids == m.checkpoints[1].ids && r->meta.checkpoints[1].used == 9 &&
                      r->meta.checkpoints[1].pinned,
                  "the manifest round-trips the checkpoints");
            check(r != nullptr && r->last_used == last_used, "the manifest round-trips last_used");
            check(r != nullptr && r->payload == payload && r->bytes == 400, "the rebuilt entry names the same payload");
            const auto* on = reopened.find(reopened.best(prompt, images, false, kLoraOn).serial);
            check(on != nullptr && on->meta.lora == kLoraOn && on->payload != payload,
                  "the twin survives the restart under its own state and payload");
            check(reopened.best(prompt, images, false, kLoraOff).tokens == 0, "off still matches neither after restart");
        }
        check(manifest_files(root).size() == 2, "both manifests were written");
        fs::remove_all(root);
    }

    // ---- version, checksum and truncation are refused, and the bad files are deleted ----
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        std::string err;
        uint64_t bytes = 0;
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.save(meta({1, 2, 3}), 100, writer(100), bytes, err), "save before the version test");
        }
        check(manifest_files(root).size() == 1, "one manifest before the version test");
        poke(manifest_files(root)[0], 8, 4);   // u32 version, after the 8-byte magic: 4 is no build's version
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.enabled() && s.entries() == 0, "an unknown manifest version is refused");
            check(s.stats().errors == 1, "the refused manifest is counted");
        }
        check(manifest_files(root).empty() && files_with_suffix(root / "conversations", ".payload").empty(),
              "the refused manifest and its payload are deleted");
        fs::remove_all(root);
    }

    // ---- the flags word: a state that names two adapters and a bit this build does not know are both refused ----
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        std::string err;
        uint64_t bytes = 0;
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.save(meta({1, 2, 3}, true, kLoraOn), 100, writer(100), bytes, err), "save before the flags test");
        }
        poke(manifest_files(root)[0], 12, 1 | 4 | 8);   // u32 flags, after the magic and version: cvec and both states
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.enabled() && s.entries() == 0, "a manifest naming two adapter states is refused");
            check(s.stats().errors == 1, "the refused flags manifest is counted");
        }
        check(manifest_files(root).empty() && files_with_suffix(root / "conversations", ".payload").empty(),
              "the flags-refused manifest and its payload are deleted");
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.save(meta({1, 2, 3}, true, kLoraOn), 100, writer(100), bytes, err), "save before the unknown bit");
        }
        poke(manifest_files(root)[0], 12, 1 | 16);   // cvec plus a flag this build does not know
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.enabled() && s.entries() == 0, "an unknown manifest flag is refused");
            check(s.stats().errors == 1, "the unknown-flag manifest is counted");
        }
        check(manifest_files(root).empty(), "the unknown-flag manifest is deleted");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        std::string err;
        uint64_t bytes = 0;
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.save(meta({1, 2, 3}), 100, writer(100), bytes, err), "save before the checksum test");
        }
        const fs::path manifest = manifest_files(root)[0];
        const std::string good = read_all(manifest);
        std::string bad = good;
        bad[bad.size() - 1] = (char) (bad.back() ^ 0x5a);
        write_all(manifest, bad);
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.entries() == 0, "a bad checksum is refused");
        }
        check(manifest_files(root).empty(), "the bad-checksum manifest is deleted");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        std::string err;
        uint64_t bytes = 0;
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.save(meta({1, 2, 3}), 100, writer(100), bytes, err), "save before the truncation test");
        }
        std::error_code ec;
        fs::resize_file(manifest_files(root)[0], 20, ec);
        check(!ec, "the manifest is truncated");
        {
            ConversationDiskStore s(o, identity(), err);
            check(s.entries() == 0, "a truncated manifest is refused");
        }
        check(manifest_files(root).empty(), "the truncated manifest is deleted");
        fs::remove_all(root);
    }

    // ---- identity: a foreign model/config is pruned, a matching one is kept ----
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        std::string err;
        uint64_t bytes = 0;
        const std::string payload = [&] {
            ConversationDiskStore s(o, identity(1, 2), err);
            check(s.save(meta({1, 2, 3}), 100, writer(100), bytes, err), "save under the first identity");
            auto m = s.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true, kLoraNone);
            return s.find(m.serial)->payload;
        }();
        {
            ConversationDiskStore s(o, identity(1, 3), err);
            check(s.entries() == 0, "a different config identity is not indexed");
        }
        check(manifest_files(root).empty() && !fs::exists(payload), "the foreign manifest and payload are deleted");
        {
            ConversationDiskStore s(o, identity(1, 2), err);
            check(s.entries() == 0, "the pruned entry does not come back");
        }
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        ConversationDiskOptions o = options(root, 1u << 20);
        std::string err;
        uint64_t bytes = 0;
        {
            ConversationDiskStore s(o, identity(1, 2), err);
            check(s.save(meta({1, 2, 3}), 100, writer(100), bytes, err), "save under a matching identity");
        }
        {
            ConversationDiskStore s(o, identity(1, 2), err);
            check(s.entries() == 1, "a matching identity is kept at startup");
        }
        check(manifest_files(root).size() == 1 && files_with_suffix(root / "conversations", ".payload").size() == 1,
              "a kept entry keeps both files");
        fs::remove_all(root);
    }

    // ---- matching: longest prefix, cvec, images, and the most recently used entry on a tie ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1u << 20), identity(), err);
        check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save A");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        check(s.save(meta({1, 2, 3, 4}), 100, writer(100), bytes, err), "save B");
        const uint64_t b = s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraNone).serial;
        check(a != 0 && b != 0 && a != b, "each entry has its own serial");
        check(s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraNone).tokens == 4, "the longest prefix wins");
        check(s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).tokens == 2, "the shorter entry still matches");
        check(s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, false, kLoraNone).tokens == 0, "cvec must agree");

        check(s.save(meta({1, 2, 3, 4}, false), 100, writer(100), bytes, err), "save C with the other cvec");
        check(s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, false, kLoraNone).tokens == 4, "C answers the other cvec");
        check(s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraNone).serial == b, "a longer entry still wins over C");

        check(s.save(meta({1, 2, 3, 4}, true, kLoraOff), 100, writer(100), bytes, err), "save D, the same prefix off the adapter");
        check(s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraNone).serial == b,
              "a newer entry under another adapter state does not steal B's tie");
        const uint64_t d = s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraOff).serial;
        const uint64_t b2 = s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraNone).serial;
        check(d != b && d != 0, "D answers the off request");
        check(b2 == b, "B still answers the none request");
        check(s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraOn).tokens == 0, "an on request reads none of them");

        check(s.save(meta({30, 31}), 100, writer(100), bytes, err), "save E for the tie");
        const uint64_t e_serial = s.best(std::vector<int32_t>{30, 31, 99}, {}, true, kLoraNone).serial;
        check(s.save(meta({30, 31}), 100, writer(100), bytes, err), "save F with the same prefix");
        const uint64_t f_serial = s.best(std::vector<int32_t>{30, 31, 99}, {}, true, kLoraNone).serial;
        check(f_serial != e_serial, "a tie keeps the newest entry");
        check(s.touch(e_serial), "touch the older equal entry");
        check(s.best(std::vector<int32_t>{30, 31, 99}, {}, true, kLoraNone).serial == e_serial,
              "a touched entry wins the next tie");
        check(!s.touch(123456), "touch of an unknown serial fails");
        check(s.find(e_serial) != nullptr && s.find(999999) == nullptr, "find looks up by serial");
        fs::remove_all(root);
    }

    // ---- image keys: a live state saved with one only matches the same one ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1u << 20), identity(), err);
        ConversationDiskMeta m = meta({1, 2, 3, 4});
        m.images = {{3, 0x1234}};
        check(s.save(m, 100, writer(100), bytes, err), "save a conversation with an image key");
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 9};
        check(s.best(prompt, {{3, 0x1234}}, true, kLoraNone).tokens == 4, "the same image key matches");
        check(s.best(prompt, {{3, 0x9999}}, true, kLoraNone).tokens == 0, "another image hash does not match");
        check(s.best(prompt, {}, true, kLoraNone).tokens == 0, "a prompt without the image does not match");
        fs::remove_all(root);
    }

    // ---- borrowing: an interior checkpoint is borrowed, the live state and the deepest one are taken ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1u << 20), identity(), err);
        ConversationDiskMeta m = meta({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        m.checkpoints.push_back(checkpoint({1, 2, 3, 4, 5}, 5));
        m.checkpoints.push_back(checkpoint({1, 2, 3, 4, 5, 6, 7, 8}, 9));
        check(s.save(m, 100, writer(100), bytes, err), "save a conversation with two checkpoints");

        auto interior = s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 99}, {}, true, kLoraNone);
        check(interior.tokens == 5 && !interior.live, "an interior checkpoint matches");
        check(s.borrows(interior), "an interior checkpoint is borrowed");
        auto live = s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99}, {}, true, kLoraNone);
        check(live.tokens == 10 && live.live, "the live state matches");
        check(!s.borrows(live), "the live state is taken, not borrowed");
        auto deepest = s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 99}, {}, true, kLoraNone);
        check(deepest.tokens == 8 && !deepest.live, "the deepest checkpoint matches");
        check(!s.borrows(deepest), "the deepest checkpoint is taken, not borrowed");
        auto stale = interior;
        stale.serial += 999;
        check(!s.borrows(stale), "a serial that is no longer present does not borrow");
        ConversationDiskStore::Match empty;
        check(!s.borrows(empty), "an empty match does not borrow");

        ConversationDiskMeta pinned = meta({10, 11, 12, 13});
        pinned.checkpoints.push_back(checkpoint({10, 11, 12}, 1, true));
        check(s.save(pinned, 100, writer(100), bytes, err), "save a pinned checkpoint");
        auto blocked = s.best(std::vector<int32_t>{10, 11, 12, 99}, {}, true, kLoraNone);
        check(blocked.tokens == 3 && !blocked.live, "the pinned checkpoint matches");
        check(!s.borrows(blocked), "a pinned checkpoint at the match blocks the borrow");
        fs::remove_all(root);
    }

    // ---- LRU, touch, pinned protection and the slot cap ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1000), identity(), err);
        check(s.save(meta({1, 2}), 400, writer(400), bytes, err), "save A for LRU");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        check(s.save(meta({3, 4}), 400, writer(400), bytes, err), "save B for LRU");
        const uint64_t b = s.best(std::vector<int32_t>{3, 4, 9}, {}, true, kLoraNone).serial;
        check(s.save(meta({5, 6}), 400, writer(400), bytes, err), "save C over the budget");
        check(s.find(a) == nullptr && s.find(b) != nullptr, "the least recently used entry is evicted");
        check(s.entries() == 2 && s.bytes() == 800 && s.stats().evictions == 1, "eviction keeps the byte count right");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1000), identity(), err);
        check(s.save(meta({1, 2}), 400, writer(400), bytes, err), "save A before the touch");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        check(s.save(meta({3, 4}), 400, writer(400), bytes, err), "save B before the touch");
        const uint64_t b = s.best(std::vector<int32_t>{3, 4, 9}, {}, true, kLoraNone).serial;
        check(s.touch(a), "touch A before the eviction");
        check(s.save(meta({5, 6}), 400, writer(400), bytes, err), "save C after touching A");
        check(s.find(a) != nullptr && s.find(b) == nullptr, "the touched entry survives, the other is evicted");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 800), identity(), err);
        ConversationDiskMeta pinned = meta({1, 2, 3});
        pinned.checkpoints.push_back(checkpoint({1, 2}, 1, true));
        check(s.save(pinned, 500, writer(500), bytes, err), "save a pinned entry");
        const uint64_t p = s.best(std::vector<int32_t>{1, 2, 3, 9}, {}, true, kLoraNone).serial;
        std::string path;
        check(!s.save(meta({10, 11}), 400, writer(400, &path), bytes, err), "only pinned entries left: refuse");
        check(s.entries() == 1 && s.find(p) != nullptr, "the pinned entry is never evicted");
        check(path.empty() || !fs::exists(path), "the refused save wrote no payload");
        check(s.stats().refusals == 1 && s.stats().evictions == 0, "the refusal evicts nothing");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1000), identity(), err);
        ConversationDiskMeta pinned = meta({1, 2, 3});
        pinned.checkpoints.push_back(checkpoint({1, 2}, 1, true));
        check(s.save(pinned, 500, writer(500), bytes, err), "save the pinned entry beside a normal one");
        check(s.save(meta({4, 5, 6}), 300, writer(300), bytes, err), "save a normal entry 1");
        const uint64_t n1 = s.best(std::vector<int32_t>{4, 5, 6, 9}, {}, true, kLoraNone).serial;
        check(s.save(meta({7, 8, 9}), 300, writer(300), bytes, err), "save a normal entry 2 over the budget");
        check(s.find(n1) == nullptr && s.best(std::vector<int32_t>{1, 2, 3, 9}, {}, true, kLoraNone).tokens == 3,
              "a normal entry is evicted before the pinned one");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1u << 20, 2), identity(), err);
        check(s.slots() == 2, "the store reports its slot cap");
        check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save A for the slot cap");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        check(s.save(meta({3, 4}), 100, writer(100), bytes, err), "save B for the slot cap");
        check(s.save(meta({5, 6}), 100, writer(100), bytes, err), "save C over the slot cap");
        check(s.entries() == 2 && s.find(a) == nullptr, "the slot cap evicts the oldest entry");
        fs::remove_all(root);
    }

    // ---- preflight: an estimate over the budget, or a huge free-space floor, evicts nothing ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1000), identity(), err);
        check(s.save(meta({1, 2}), 400, writer(400), bytes, err), "save before the estimate refusal");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        const std::string payload = s.find(a)->payload;
        std::string path;
        check(!s.save(meta({5, 6}), 1200, writer(1200, &path), bytes, err), "an estimate over the budget is refused");
        check(s.entries() == 1 && s.bytes() == 400 && s.find(a) != nullptr, "the estimate refusal evicts nothing");
        check(fs::exists(payload), "the existing payload stays");
        check(path.empty(), "the refused save does not even name a payload");
        check(s.stats().refusals == 1 && s.stats().evictions == 0, "the estimate refusal is counted");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        {
            ConversationDiskStore s(options(root, 1u << 20), identity(), err);
            check(s.save(meta({1, 2}), 400, writer(400), bytes, err), "save before the free-space refusal");
        }
        ConversationDiskOptions o = options(root, 1u << 20);
        o.min_free_bytes = UINT64_MAX;
        ConversationDiskStore s(o, identity(), err);
        check(s.entries() == 1, "the entry is loaded for the free-space refusal");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        std::string path;
        check(!s.save(meta({5, 6}), 100, writer(100, &path), bytes, err), "a huge free-space floor is refused");
        check(s.entries() == 1 && s.find(a) != nullptr && path.empty(), "the free-space refusal evicts nothing");
        check(s.stats().refusals == 1 && s.stats().evictions == 0, "the free-space refusal is counted");
        fs::remove_all(root);
    }

    // ---- a failing writer leaves no entry, no payload, and the entries before it ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1u << 20), identity(), err);
        check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save before the writer failure");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        std::string path;
        check(!s.save(meta({3, 4}), 100, writer(100, &path, true), bytes, err), "a failing writer refuses the save");
        check(bytes == 0 && s.entries() == 1 && s.find(a) != nullptr, "a failed save adds no entry");
        check(!path.empty() && !fs::exists(path), "a failed save removes its partial payload");
        check(manifest_files(root).size() == 1, "a failed save leaves one manifest");
        check(s.stats().errors == 1 && s.stats().saves == 1, "the writer failure is counted as an error");
        check(!s.save(meta({3, 4}), 100, ConversationDiskStore::Writer{}, bytes, err), "a save with no writer fails");
        fs::remove_all(root);
    }

    // ---- startup: tmp files, a missing payload and a corrupt manifest are dropped, the rest survives ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        std::string a_payload, b_payload;
        {
            ConversationDiskStore s(options(root, 1u << 20), identity(), err);
            check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save A for the startup scan");
            a_payload = s.find(s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial)->payload;
            check(s.save(meta({3, 4}), 100, writer(100), bytes, err), "save B for the startup scan");
            b_payload = s.find(s.best(std::vector<int32_t>{3, 4, 9}, {}, true, kLoraNone).serial)->payload;
        }
        make_tmp(root / "manifests", "stray.tmp");
        make_tmp(root / "conversations", "stray.tmp");
        make_tmp(root / "conversations", "snap-0000000000000000.payload");
        make_tmp(root / "tmp", "junk.tmp");
        check(fs::remove(b_payload), "the missing-payload entry's payload is deleted");
        {
            ConversationDiskStore s(options(root, 1u << 20), identity(), err);
            check(s.entries() == 1 && s.bytes() == 100, "the entry with a missing payload is dropped");
            check(s.find(s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial)->payload == a_payload,
                  "the other entry survives");
        }
        check(!fs::exists(root / "conversations" / "snap-0000000000000000.payload"),
              "a payload without a manifest is removed at startup");
        check(files_with_suffix(root / "manifests", ".tmp").empty(), "a stray manifest .tmp is removed");
        check(files_with_suffix(root / "conversations", ".tmp").empty(), "a stray conversation .tmp is removed");
        check(files_with_suffix(root / "tmp", ".tmp").empty(), "the startup scan empties tmp");
        check(manifest_files(root).size() == 1, "the dropped entry's manifest is gone");
        fs::remove_all(root);
    }
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        std::string a_manifest, a_payload;
        {
            ConversationDiskStore s(options(root, 1u << 20), identity(), err);
            check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save A before the corruption");
            const auto* e = s.find(s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial);
            a_manifest = e->manifest;
            a_payload = e->payload;
            check(s.save(meta({3, 4}), 100, writer(100), bytes, err), "save B before the corruption");
        }
        poke(a_manifest, 40, 0x7f);   // inside the identity/timestamp fields: the checksum no longer matches
        {
            ConversationDiskStore s(options(root, 1u << 20), identity(), err);
            check(s.entries() == 1, "a corrupt manifest is dropped and the other entry survives");
            check(s.stats().errors == 1, "the corrupt manifest is counted");
            check(s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).tokens == 0, "the corrupt entry does not match");
        }
        check(!fs::exists(a_manifest) && !fs::exists(a_payload), "the corrupt entry's files are deleted");
        check(manifest_files(root).size() == 1, "one manifest survives the corruption");
        fs::remove_all(root);
    }

    // ---- Phase 2: the shared page pool, a CoW save, dedup, restore and compaction ----
    {
        const fs::path root = unique_root();
        {
        ConversationDiskOptions o = options(root, 16u << 20);
        o.cow = true;
        o.min_free_bytes = 0;
        o.min_tokens = 1;
        std::string err;
        auto s_owner = std::make_unique<ConversationDiskStore>(o, identity(), err);
        ConversationDiskStore& s = *s_owner;
        check(s.enabled() && s.paged(), "the page pool opens with the store");

        // Two synthetic layers: part 0 is 2.5 pages, part 1 is one page.
        std::vector<std::vector<uint8_t>> parts(2);
        parts[0].resize(kConversationPageBytes * 2 + kConversationPageBytes / 2);
        parts[1].resize(kConversationPageBytes);
        for (size_t i = 0; i < parts[0].size(); ++i) parts[0][i] = (uint8_t) (i * 7 + (i >> 9) + 3);
        for (size_t i = 0; i < parts[1].size(); ++i) parts[1][i] = (uint8_t) (i * 11 + (i >> 10) + 5);
        auto make_source = [](const std::vector<std::vector<uint8_t>>& p) {
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
        };
        auto head_for = [](const std::vector<int32_t>& ids) {
            SavedConversation head;
            head.geometry = {128, 2, 1, 0, 0, 0, 0, 0, 0, 4, 4, 64, 4, 128, 0, 0, 0, 0};
            head.layer_lo = 0;
            head.layer_hi = 1;
            head.live.ids = ids;
            head.live.imgs = {};
            head.cvec = true;
            return head;
        };
        ConversationDiskMeta dm;
        dm.ids = {1, 2, 3, 4};
        dm.cvec = true;
        uint64_t bytes = 0, written = 0, reused = 0;
        const std::vector<SessionKvSource> sources = {make_source(parts)};
        const SavedConversation head = head_for(dm.ids);
        const bool saved = s.save_paged(dm, head, sources, bytes, written, reused, err);
        check(saved, "a paged save writes a head and pages");
        auto match = s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true, kLoraNone);
        check(match.tokens == 4 && match.live, "the paged entry matches its live state");
        const uint64_t serial = match.serial;
        check(s.find(serial) != nullptr && s.find(serial)->paged && s.find(serial)->bytes == bytes &&
                  bytes > parts[0].size() + parts[1].size(),
              "the paged entry counts its head and K/V bytes");
        check(written == parts[0].size() + parts[1].size() && reused == 0,
              "the first paged save writes every page");

        // Restore: one layer at a time, from the pool.
        {
            ConversationPageReader reader;
            SessionReadLimits limits;
            check(s.open_paged(serial, reader, limits, err), "the paged entry opens");
            check(reader.meta().live.ids == dm.ids && reader.meta().cvec == dm.cvec,
                  "the head's metadata round-trips");
            check(reader.layers() == 1 && reader.bytes() == bytes, "the reader reports one layer and its bytes");
            ConversationKv layer;
            check(reader.next(layer, err), "the paged entry restores layer by layer");
            std::vector<uint8_t> a, b;
            check(buffer_bytes(layer.k, a) && buffer_bytes(layer.v, b), "the restored parts are readable");
            check(a == parts[0] && b == parts[1], "the restored K/V bytes are the saved bytes");
            check(reader.finish(err), "the paged reader finishes");
        }

        // A new turn with the same bytes but one more token: every page is reused.
        ConversationDiskMeta dm2;
        dm2.ids = {1, 2, 3, 4, 5};
        dm2.cvec = true;
        check(s.save_paged(dm2, head_for(dm2.ids), sources, bytes, written, reused, err), "the CoW save succeeds");
        check(written == 0 && reused == parts[0].size() + parts[1].size(),
              "an unchanged K/V writes no page bytes");

        // Change one byte in the second page of part 0: exactly that page is written.
        std::vector<std::vector<uint8_t>> changed = parts;
        changed[0][kConversationPageBytes + 10] ^= 0xff;
        ConversationDiskMeta dm3;
        dm3.ids = {1, 2, 3, 4, 5, 6};
        dm3.cvec = true;
        const std::vector<SessionKvSource> changed_sources = {make_source(changed)};
        check(s.save_paged(dm3, head_for(dm3.ids), changed_sources, bytes, written, reused, err),
              "a changed page saves");
        check(written == kConversationPageBytes, "only the changed page is written");
        check(reused == parts[0].size() - kConversationPageBytes + parts[1].size(),
              "the pages before and after it are reused");

        // A restart rebuilds the index, the refcounts and the restore path from the manifests and the pool.
        const uint64_t pool_before = (uint64_t) s.pool_bytes();
        const size_t entries_before = s.entries();
        {
            ConversationDiskStore again(o, identity(), err);
            check(again.enabled() && again.entries() == entries_before, "the startup scan loads the paged entries");
            auto m = again.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 9}, {}, true, kLoraNone);
            check(m.tokens == 6 && m.live, "the rebuilt index matches the newest paged entry");
            check(paged_restore(again, m.serial, changed, err), "the rebuilt entry restores the changed K/V");
            check(again.pool_bytes() == pool_before, "the restart keeps the pool's bytes");
        }
        // Without pages.idx the pool is scanned and the index is rebuilt: the index is a cache, not the truth.
        {
            std::error_code rec;
            check(fs::remove(root / "pool" / "pages.idx", rec) && !rec, "the page index is deleted");
            ConversationDiskStore rebuilt(o, identity(), err);
            check(rebuilt.enabled() && rebuilt.entries() == entries_before, "a missing index is rebuilt by scanning");
            check(paged_restore(rebuilt, serial, parts, err), "the scanned pool restores");
            check(fs::exists(root / "pool" / "pages.idx"), "the scan writes a new page index");
        }

        // Remove the newest entry: the older one still restores, and its pages survive.
        auto newest = s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 9}, {}, true, kLoraNone);
        check(s.remove(newest.serial, err), "remove the newest paged entry");
        check(s.entries() == entries_before - 1 && s.bytes() > 0, "the older paged entry stays");
        check(s.find(serial) != nullptr, "the original entry survives a remove of another");
        check(paged_restore(s, serial, parts, err), "the surviving entry restores unchanged");
        s_owner.reset();   // one store per pool at a time (Windows keeps a rename-blocking handle)

        // Compaction: the removed entry's pages are dead; the pool is rewritten and the survivor still restores.
        ConversationDiskOptions tight = o;
        tight.compact_min_bytes = 0;
        tight.compact_min_pages = 1;
        tight.compact_fragmentation = 0.0;
        ConversationDiskStore compacting(tight, identity(), err);
        check(compacting.entries() == entries_before - 1, "the compacting store loads the survivor");
        const uint64_t before = compacting.pool_bytes();
        const bool compacted = compacting.compact(err);
        check(compacted, "the pool compacts");
        check(compacting.stats().compactions == 1 && compacting.pool_bytes() < before,
              "compaction shrinks the pool");
        check(compacting.stats().last_compacted_bytes > 0, "the compaction reports the bytes it reclaimed");
        check(paged_restore(compacting, serial, parts, err), "the survivor restores after compaction");
        {
            ConversationDiskStore reopened(tight, identity(), err);
            check(reopened.entries() == entries_before - 1 && paged_restore(reopened, serial, parts, err),
                  "the compacted pool restores after a restart");
        }

        // Corruption: a changed byte in a page is caught when the page is read.
        {
            const fs::path pool = root / "pool" / "pages.bin";
            std::ifstream in(pool, std::ios::binary);
            std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            check(data.size() > kConversationPageBytes + 32, "the pool has a page to corrupt");
            data[kConversationPageBytes + 32] ^= 0x5a;   // inside the first page's data (after the 16-byte header)
            std::ofstream out(pool, std::ios::binary | std::ios::trunc);
            out.write(data.data(), (std::streamsize) data.size());
        }
        {
            ConversationDiskStore corrupted(o, identity(), err);
            check(corrupted.entries() == entries_before - 1,
                  "the corrupt pool still indexes the entries: the manifest is the commit");
            ConversationPageReader reader;
            SessionReadLimits limits;
            check(corrupted.open_paged(serial, reader, limits, err), "opening resolves the page offsets");
            ConversationKv layer;
            check(!reader.next(layer, err) && !err.empty(), "a corrupt page refuses the restore");
        }
        }   // the stores are closed before their folder is removed (Windows keeps open files)
        std::error_code rec;
        fs::remove_all(root, rec);
    }

    // ---- path safety ----
    {
        ConversationDiskOptions o;
        o.dir = "relative-cache";
        o.budget_bytes = 1u << 20;
        o.min_free_bytes = 0;
        std::string err;
        ConversationDiskStore s(o, identity(), err);
        check(!s.enabled() && !err.empty(), "a relative directory is refused");
    }
    {
        ConversationDiskOptions o;
        o.dir = (fs::temp_directory_path() / std::string("strata-bad\x01dir")).string();
        o.budget_bytes = 1u << 20;
        o.min_free_bytes = 0;
        std::string err;
        ConversationDiskStore s(o, identity(), err);
        check(!s.enabled() && !err.empty(), "a directory with a control byte is refused");
        check(!fs::exists(o.dir), "a refused directory is not created");
    }
    {
        ConversationDiskOptions o;
        o.budget_bytes = 0;
        std::string err;
        ConversationDiskStore s(o, identity(), err);
        check(!s.enabled() && err.empty(), "budget 0 disables the store without touching a directory");
    }

    // ---- remove, clear and the restore counters ----
    {
        const fs::path root = unique_root();
        std::string err;
        uint64_t bytes = 0;
        ConversationDiskStore s(options(root, 1u << 20), identity(), err);
        check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save for remove");
        const uint64_t a = s.best(std::vector<int32_t>{1, 2, 9}, {}, true, kLoraNone).serial;
        const std::string payload = s.find(a)->payload;
        check(s.remove(a, err), "remove an entry");
        check(s.entries() == 0 && s.bytes() == 0 && !fs::exists(payload), "remove deletes the payload");
        check(!s.remove(a, err) && !err.empty(), "remove of a gone serial fails");
        check(s.save(meta({1, 2}), 100, writer(100), bytes, err), "save for clear");
        check(s.save(meta({3, 4}), 100, writer(100), bytes, err), "save a second entry for clear");
        check(s.clear() == 2, "clear returns the number of entries");
        check(s.entries() == 0 && s.bytes() == 0, "clear empties the index");
        check(files_with_suffix(root / "manifests", ".bin").empty() &&
                  files_with_suffix(root / "conversations", ".payload").empty(),
              "clear deletes both files of each entry");

        s.note_restore(12, 345);
        auto st = s.stats();
        check(st.restores == 1 && st.last_restore_ms == 12 && st.last_restore_bytes == 345,
              "note_restore records the time and the bytes");
        s.note_restore_failure();
        check(s.stats().restore_failures == 1, "note_restore_failure counts the miss");
        fs::remove_all(root);
    }

    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
