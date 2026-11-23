// CPU-only tests for the conversation spill tier's page pool (include/strata/core/spill_pages.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model, no network.  These are the page-pool checks the
// retired conversation_disk_test.cpp held, moved to the file the pool lives in now (docs/work/local-spill-delta.md).
#include "strata/core/spill_pages.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

fs::path unique_root() {
    static int n = 0;
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path p = fs::temp_directory_path() /
                       ("strata-spill-pages-" + std::to_string(++n) + "-" + std::to_string(ticks));
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

// Two synthetic layers: part 0 is 2.5 pages, part 1 is one page.
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

bool read_back(ConversationPagePool& pool, const ConversationPageRef& ref, std::vector<uint8_t>& out) {
    ConversationPagePool::Page page;
    if (!pool.lookup(ref.hash, ref.bytes, page)) return false;
    out.resize(ref.bytes);
    std::string why;
    return pool.read(ref.hash, ref.bytes, page.offset, out.data(), why);
}

} // namespace

int main() {
    // ---- the pool: append, content-addressed dedup, read back, release and compact ----
    {
        const fs::path root = unique_root();
        std::string err;
        {
            ConversationPagePool pool(root.string());
            check(pool.open(true, err), "the pool opens writable");
            std::vector<uint8_t> data(kConversationPageBytes);
            for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) (i * 3 + 1);
            ConversationPageRef ref;
            bool wrote = false;
            check(pool.append(session_hash64(data.data(), data.size(), 0), data.data(), (uint32_t) data.size(), ref,
                              wrote, err),
                  "a page appends");
            check(wrote && ref.bytes == data.size(), "the append reports what it wrote");
            check(pool.persist(err), "the index persists");
            std::vector<uint8_t> back;
            check(read_back(pool, ref, back) && back == data, "the page reads back byte for byte");

            ConversationPageRef again;
            check(pool.append(ref.hash, data.data(), (uint32_t) data.size(), again, wrote, err) && !wrote,
                  "the same page is not written twice");
            check(again.hash == ref.hash && again.bytes == ref.bytes, "the dedup returns the same ref");

            ConversationPageRef corrupt;
            std::vector<uint8_t> other = data;
            other[0] ^= 0xff;
            check(pool.append(session_hash64(other.data(), other.size(), 0), other.data(), (uint32_t) other.size(),
                              corrupt, wrote, err) && wrote,
                  "a different page appends");
            check(pool.persist(err), "the second append persists");
            const auto stats = pool.stats();
            check(stats.pages == 2 && stats.bytes_written > 0, "the pool counts its pages and writes");
        }
        // Reopen: the index is loaded, not rescanned, and a page reads back.
        {
            ConversationPagePool pool(root.string());
            check(pool.open(false, err), "the pool opens read-only");
            check(pool.stats().pages == 2, "the persisted index holds both pages");
        }
        // No index: the pool is scanned and the index rebuilt (the index is a cache, not the truth).
        {
            std::error_code ec;
            fs::remove(root / "pages.idx", ec);
            ConversationPagePool pool(root.string());
            check(pool.open(true, err) && pool.stats().pages == 2, "a missing index is rebuilt by scanning");
            check(fs::exists(root / "pages.idx"), "the scan writes a new index");
        }
        // Release one page, then compact: the pool shrinks and stays readable.
        {
            ConversationPagePool pool(root.string());
            check(pool.open(true, err), "the pool opens for the release");
            const auto parts = parts_of(3);
            (void) parts;
            std::vector<uint8_t> data(kConversationPageBytes);
            for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t) (i * 3 + 1);
            std::vector<uint8_t> other = data;
            other[0] ^= 0xff;
            check(pool.stats().pages == 2, "two pages before the release");
            const uint64_t live_hash = session_hash64(data.data(), data.size(), 0);
            pool.reference(live_hash, (uint32_t) data.size());   // (a conversation names this page)
            pool.release(session_hash64(other.data(), other.size(), 0), (uint32_t) other.size());
            check(pool.persist(err), "the release persists");
            const uint64_t before = pool.stats().pool_bytes;
            check(pool.compact(0, 1, 0.0, err), "the pool compacts with all guards open");
            check(pool.stats().pool_bytes < before, "compaction reclaims the released page");
            ConversationPagePool::Page page;
            check(pool.lookup(session_hash64(data.data(), data.size(), 0), (uint32_t) data.size(), page),
                  "the live page is still indexed");
            std::vector<uint8_t> back(data.size());
            check(pool.read(session_hash64(data.data(), data.size(), 0), (uint32_t) data.size(), page.offset, back.data(), err) &&
                      back == data,
                  "the live page reads back after compaction");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- the writer and the reader: layers in, layers out, and a changed page costs one page ----
    {
        const fs::path root = unique_root();
        std::string err;
        auto owner = std::make_unique<ConversationPagePool>(root.string());
        ConversationPagePool& pool = *owner;
        check(pool.open(true, err), "the pool opens for the writer");

        const auto parts = parts_of(3);
        const std::vector<SessionKvSource> sources = {make_source(parts)};
        ConversationPageWriteResult out;
        check(conversation_pages_write(pool, sources, out, err), "the writer stores the K/V as pages");
        check(out.layout.layers.size() == 1, "one layer is described");
        check(out.bytes_written == parts[0].size() + parts[1].size(), "the first write stores every byte");
        check(out.pages_written == 4 && out.pages_reused == 0, "2.5 + 1 pages are three full pages and a half");
        check(out.layout.kv_bytes() == parts[0].size() + parts[1].size(), "the layout names the K/V bytes");
        check(pool.persist(err), "the index persists");

        // The same bytes again: nothing is written.
        ConversationPageWriteResult again;
        check(conversation_pages_write(pool, sources, again, err), "an unchanged K/V writes again");
        check(again.bytes_written == 0 && again.bytes_reused == parts[0].size() + parts[1].size(),
              "an unchanged K/V reuses every page");

        // One changed byte in the second page of part 0: exactly that page is written.
        auto changed = parts;
        changed[0][kConversationPageBytes + 10] ^= 0xff;
        const std::vector<SessionKvSource> changed_sources = {make_source(changed)};
        ConversationPageWriteResult third;
        check(conversation_pages_write(pool, changed_sources, third, err), "a changed page writes");
        check(third.bytes_written == kConversationPageBytes, "only the changed page is written");
        check(third.bytes_reused == parts[0].size() - kConversationPageBytes + parts[1].size(),
              "the pages before and after it are reused");
        check(pool.persist(err), "the index persists after the change");
        owner.reset();

        // Compaction: the superseded pages are dead, the live ones survive, and the pool shrinks.
        {
            ConversationPagePool compacting(root.string());
            check(compacting.open(true, err), "the pool opens for compaction");
            // the newest entry names these pages (the tier calls reference() for each; a bare pool does not)
            for (const auto& layer : third.layout.layers)
                for (const auto& part : layer.parts)
                    for (const auto& ref : part) compacting.reference(ref.hash, ref.bytes);
            check(compacting.persist(err), "the live references persist");
            const uint64_t before = compacting.stats().pool_bytes;
            check(compacting.compact(0, 1, 0.0, err), "the pool compacts");
            check(compacting.stats().compactions == 1 && compacting.stats().pool_bytes < before,
                  "compaction shrinks the pool");
            check(compacting.stats().last_compacted_bytes > 0, "the compaction reports the bytes it reclaimed");
            // Every live page still reads back with its own hash.
            for (const auto& layer : third.layout.layers)
                for (const auto& part : layer.parts)
                    for (const auto& ref : part) {
                        std::vector<uint8_t> back;
                        check(read_back(compacting, ref, back) && back.size() == ref.bytes,
                              "a live page survives compaction");
                    }
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- a torn tail is ignored, and a corrupt page is refused on read ----
    {
        const fs::path root = unique_root();
        std::string err;
        std::vector<uint8_t> data(kConversationPageBytes, 0x5a);
        {
            ConversationPagePool pool(root.string());
            check(pool.open(true, err), "the pool opens for the torn tail");
            ConversationPageRef ref;
            bool wrote = false;
            check(pool.append(session_hash64(data.data(), data.size(), 0), data.data(), (uint32_t) data.size(), ref,
                              wrote, err) && pool.persist(err),
                  "one page is written");
        }
        // Append garbage past the last record: it must be ignored and trimmed, not indexed.
        {
            std::ofstream f(root / "pages.bin", std::ios::binary | std::ios::app);
            const std::string junk(64, '\x7f');
            f.write(junk.data(), (std::streamsize) junk.size());
        }
        {
            ConversationPagePool pool(root.string());
            check(pool.open(true, err), "a pool with a torn tail opens");
            check(pool.stats().pages == 1, "the torn tail is not indexed");
        }
        // Corrupt the page's data in place: the read must refuse it.
        {
            std::string bytes = read_all(root / "pages.bin");
            check(bytes.size() > 4096 + 16, "the pool holds the page");
            bytes[4096 + 32] ^= 0x5a;
            write_all(root / "pages.bin", bytes);
            std::error_code ec;
            fs::remove(root / "pages.idx", ec);   // force a rescan: the corrupt record ends the scan
            ConversationPagePool pool(root.string());
            check(pool.open(true, err), "the corrupt pool still opens (its index is a cache)");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    // ---- the format version: a pool of another version is a CACHE, discarded on the writable open ----
    {
        const fs::path root = unique_root();
        std::string err;
        {
            ConversationPagePool pool(root.string());
            check(pool.open(true, err), "the pool opens before the version bump");
            std::vector<uint8_t> data(kConversationPageBytes, 0x11);
            ConversationPageRef ref;
            bool wrote = false;
            check(pool.append(session_hash64(data.data(), data.size(), 0), data.data(), (uint32_t) data.size(), ref,
                              wrote, err) && pool.persist(err),
                  "a page is written before the version bump");
        }
        {
            std::string bytes = read_all(root / "pages.bin");
            check(bytes.size() > 12, "the pool has a header");
            const uint32_t old = kSpillPoolFormatVersion - 1;
            std::memcpy(bytes.data() + 8, &old, 4);   // pretend the pool was written by the previous format
            write_all(root / "pages.bin", bytes);
            // the header checksum no longer matches, which the discard path must not care about
            ConversationPagePool pool(root.string());
            check(pool.open(false, err) == false, "a read-only open of another version is a miss");
            check(pool.open(true, err), "a writable open of another version discards it and starts empty");
            check(pool.stats().pages == 0, "the old pool's pages are gone");
            std::vector<uint8_t> data(kConversationPageBytes, 0x22);
            ConversationPageRef ref;
            bool wrote = false;
            check(pool.append(session_hash64(data.data(), data.size(), 0), data.data(), (uint32_t) data.size(), ref,
                              wrote, err),
                  "the new pool accepts pages");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    std::fprintf(stderr, "spill_pages_test: %d checks passed\n", checks);
    return 0;
}
