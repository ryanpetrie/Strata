// CPU-only tests for the layer-at-a-time session-file reader (SessionFileStream, conversation_file.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model, no network.
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
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

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 13u + seed);
    return v;
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> flatten(const ConversationBuffer& b) {
    std::vector<uint8_t> v(b.size());
    b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t at) {
        std::memcpy(v.data() + at, p, c);
        return true;
    });
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(100 + i));
    c.imgs = {{3, 0xabc0000012345678ull ^ seed}};
    c.gdn = bytes_of(257, seed);
    c.ple = bytes_of(33, seed + 1);
    c.tails = bytes_of(91, seed + 2);
    c.dead = bytes_of(16, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 7 + seed;
    return c;
}

ConversationKv kv_layer(int index, size_t k, size_t v, size_t k_scale, size_t v_scale, size_t pooled) {
    ConversationKv kv;
    kv.format = 2 + index;
    kv.cells = 512 + index;
    kv.heads = 2;
    kv.head_dim = 128;
    kv.page_size = 64;
    kv.pooled_rows = pooled ? 9 : 0;
    kv.idx_dim = 128;
    kv.k = pattern(k, uint8_t(index));
    kv.v = pattern(v, uint8_t(index + 10));
    kv.k_scale = pattern(k_scale, uint8_t(index + 20));
    kv.v_scale = pattern(v_scale, uint8_t(index + 30));
    kv.pooled = pattern(pooled, uint8_t(index + 40));
    return kv;
}

// Three layers of different sizes; the last is draft-like (no pooled rows, no pooled bytes).
SavedConversation small_image() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(7 + i);
    s.layer_lo = 0;
    s.layer_hi = 27;
    s.cvec = true;
    s.live = checkpoint(40, 1);
    s.checkpoints = {checkpoint(8, 2), checkpoint(20, 3)};
    s.kv.push_back(kv_layer(0, 3000, 2200, 512, 256, 900));
    s.kv.push_back(kv_layer(1, 4096, 1024, 0, 256, 300));
    s.kv.push_back(kv_layer(2, 2048, 2048, 128, 128, 0));
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same_kv(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim && a.k == b.k &&
           a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}

// The limits an engine that saved this image would set: every count and byte size exact.
SessionReadLimits exact_limits(const SavedConversation& s) {
    SessionReadLimits l;
    l.max_tokens = s.live.ids.size();
    for (const auto& c : s.checkpoints) l.max_tokens = std::max<uint64_t>(l.max_tokens, c.ids.size());
    l.max_checkpoints = s.checkpoints.size();
    l.max_kv_layers = s.kv.size();
    l.geometry = s.geometry;
    l.layer_range = std::make_pair(s.layer_lo, s.layer_hi);
    l.max_state_bytes = {s.live.gdn.size(), s.live.ple.size(), s.live.tails.size(), s.live.dead.size(),
                         s.live.block_pos.size()};
    for (const auto& k : s.kv)
        l.max_kv_bytes.push_back({k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()});
    return l;
}

void check_meta(const SavedConversation& meta, const SavedConversation& original) {
    check(meta.kv.empty(), "meta holds no K/V");
    check(meta.geometry == original.geometry, "meta: geometry");
    check(meta.layer_lo == original.layer_lo && meta.layer_hi == original.layer_hi, "meta: layer range");
    check(meta.cvec == original.cvec, "meta: cvec");
    check(same_checkpoint(meta.live, original.live), "meta: live state");
    check(meta.checkpoints.size() == original.checkpoints.size(), "meta: checkpoint count");
    for (size_t i = 0; i < meta.checkpoints.size(); ++i)
        check(same_checkpoint(meta.checkpoints[i], original.checkpoints[i]), "meta: checkpoint state");
}

// ---- 1. the image overload
void round_trip_image(const fs::path& dir, const SessionFileIdentity& id) {
    const SavedConversation original = small_image();
    const fs::path p = dir / "image.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(p.string(), original, id, written, error), "image: write");
    check(error.empty(), "image: write leaves no error");

    SessionFileStream stream;
    SessionStatus st;
    uint64_t reports = 0, last_done = 0, last_total = 0;
    SessionReadLimits limits = exact_limits(original);
    limits.progress = [&](uint64_t done, uint64_t total) { ++reports; last_done = done; last_total = total; };
    check(stream.open(p.string(), id, limits, error, &st), "image: open");
    check(st.error == SessionError::none && error.empty(), "image: open succeeds without an error");
    check(reports >= 1 && last_done == written && last_total == written,
          "image: progress follows the file's blocks");
    check(stream.bytes() == written && stream.bytes() == fs::file_size(p), "image: size is the file's");
    check(stream.layers() == original.kv.size(), "image: layer count");
    check_meta(stream.meta(), original);

    ConversationKv got;
    for (size_t i = 0; i < stream.layers(); ++i) {
        check(stream.next(got, error), "image: next reads a layer");
        check(error.empty(), "image: next leaves no error");
        check(same_kv(got, original.kv[i]), "image: layer geometry and bytes");
    }
    check(stream.finish(error), "image: finish accepts the file");
    check(error.empty(), "image: finish leaves no error");
    check(!stream.next(got, error), "image: next after finish fails");
    check(got.k.size() == 0 && got.v.size() == 0 && got.pooled.size() == 0, "image: no layer after finish");

    // The whole-file read still accepts the same file unchanged.
    SavedConversation back;
    size_t read = 0;
    check(session_file_read(p.string(), id, back, read, error), "regression: session_file_read succeeds");
    check(read == written && back.kv.size() == original.kv.size(), "regression: same size and layer count");
    check(same_checkpoint(back.live, original.live), "regression: live state unchanged");
    for (size_t i = 0; i < back.kv.size(); ++i) check(same_kv(back.kv[i], original.kv[i]), "regression: layer bytes");
}

// ---- 2. the streamed-sources overload
void round_trip_sources(const fs::path& dir, const SessionFileIdentity& id) {
    const SavedConversation original = small_image();
    SavedConversation meta = original;
    meta.kv.clear();
    // each part lives in a byte vector the source reads from, as the engine's device read would
    struct Stored {
        std::array<std::vector<uint8_t>, 5> part;
    };
    std::vector<SessionKvSource> sources;
    for (const auto& k : original.kv) {
        auto held = std::make_shared<Stored>();
        const ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t i = 0; i < 5; ++i) held->part[i] = flatten(*parts[i]);
        SessionKvSource s;
        s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
        s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
        s.sizes = {held->part[0].size(), held->part[1].size(), held->part[2].size(), held->part[3].size(),
                   held->part[4].size()};
        s.read = [held](size_t part, size_t offset, void* dst, size_t n) {
            if (part >= 5 || offset > held->part[part].size() || n > held->part[part].size() - offset) return false;
            std::memcpy(dst, held->part[part].data() + offset, n);
            return true;
        };
        s.error = [] { return std::string("source read failed"); };
        sources.push_back(std::move(s));
    }
    const fs::path p = dir / "sources.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(p.string(), meta, sources, id, written, error), "sources: write");

    SessionFileStream stream;
    SessionStatus st;
    check(stream.open(p.string(), id, exact_limits(original), error, &st), "sources: open");
    check(stream.bytes() == written && stream.layers() == original.kv.size(), "sources: size and layer count");
    check_meta(stream.meta(), original);
    ConversationKv got;
    for (size_t i = 0; i < stream.layers(); ++i) {
        check(stream.next(got, error), "sources: next");
        check(same_kv(got, original.kv[i]), "sources: layer geometry and bytes");
    }
    check(stream.finish(error), "sources: finish");
}

// ---- 3. one layer at a time, and the peak
void peak_and_reuse(const fs::path& dir, const SessionFileIdentity& id) {
    SavedConversation big;
    for (size_t i = 0; i < big.geometry.size(); ++i) big.geometry[i] = int64_t(7 + i);
    big.layer_lo = 0;
    big.layer_hi = 27;
    big.cvec = false;
    big.live = checkpoint(16, 5);
    big.checkpoints = {checkpoint(8, 6)};
    for (int i = 0; i < 3; ++i) big.kv.push_back(kv_layer(i, 450000, 450000, 60000, 60000, i == 2 ? 0 : 30000));
    const fs::path p = dir / "peak.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(p.string(), big, id, written, error), "peak: write");
    check(written > (2u << 20), "peak: the file holds all three layers");

    SessionFileStream stream;
    SessionStatus st;
    check(stream.open(p.string(), id, exact_limits(big), error, &st), "peak: open");
    check(stream.peak_bytes() < (1u << 20), "peak: opening holds the metadata only");
    ConversationKv reuse;   // the same layer, read three times
    for (size_t i = 0; i < stream.layers(); ++i) {
        check(stream.next(reuse, error), "peak: next");
        check(same_kv(reuse, big.kv[i]), "peak: layer bytes");
        // one layer and the metadata, never the whole file: the two other layers are not held
        check(stream.peak_bytes() < (8u << 20), "peak: one layer at a time stays under 8 MiB");
        check(stream.peak_bytes() < written / 2, "peak: well under the whole file's bytes");
    }
    // the last layer has no pooled part: the previous layer's pooled bytes were dropped, not kept
    check(big.kv.back().pooled.size() == 0 && reuse.pooled.size() == 0, "peak: the previous layer's part is gone");
    check(stream.finish(error), "peak: finish");
}

// ---- 4/5. refusals, and the read's own state machine
bool refused(const fs::path& p, const SessionFileIdentity& id, const SessionReadLimits& limits, const char* expect,
             SessionError kind) {
    SessionFileStream stream;
    std::string error;
    SessionStatus st;
    if (stream.open(p.string(), id, limits, error, &st)) return false;
    if (error.find(expect) == std::string::npos) {
        std::fprintf(stderr, "  unexpected error text: %s (wanted %s)\n", error.c_str(), expect);
        return false;
    }
    if (st.error != kind) return false;
    // a refusal leaves nothing usable: no metadata, no layers, and no layer for the caller
    if (!stream.meta().kv.empty() || stream.layers() != 0 || stream.bytes() != 0) return false;
    ConversationKv layer;
    layer.k.resize(64);
    std::string again;
    return !stream.next(layer, again) && layer.k.size() == 0;
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<char>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize) d.size());
}

void refusals(const fs::path& dir, const SessionFileIdentity& id) {
    const SavedConversation original = small_image();
    const fs::path good = dir / "good.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(good.string(), original, id, written, error), "refusals: write");
    const std::vector<char> image = slurp(good);
    const SessionReadLimits exact = exact_limits(original);

    check(refused(good, {id.model ^ 1, id.config}, exact, "model", SessionError::invalid),
          "refused: another model");
    check(refused(good, {id.model, id.config ^ 1}, exact, "config", SessionError::invalid),
          "refused: another configuration");
    {
        auto d = image; d[56] ^= 0x10;
        const fs::path p = dir / "badhash.bin"; spit(p, d);
        check(refused(p, id, exact, "header checksum", SessionError::invalid), "refused: corrupt header hash");
    }
    {
        const fs::path p = dir / "short.bin"; spit(p, std::vector<char>(40, '\0'));
        check(refused(p, id, exact, "below the minimum", SessionError::invalid), "refused: a short file");
    }
    {
        const fs::path p = dir / "truncated.bin";
        spit(p, std::vector<char>(image.begin(), image.end() - 100));
        check(refused(p, id, exact, "size", SessionError::invalid), "refused: a truncated file");
    }
    {
        auto d = image; d.push_back('x');
        const fs::path p = dir / "trailing.bin"; spit(p, d);
        check(refused(p, id, exact, "size", SessionError::invalid), "refused: extra bytes after the trailer");
    }
}

void refuse_limits(const fs::path& dir, const SessionFileIdentity& id) {
    const SavedConversation original = small_image();
    const fs::path good = dir / "limits.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(good.string(), original, id, written, error), "limits: write");
    const SessionReadLimits exact = exact_limits(original);

    SessionReadLimits l = exact;
    l.max_tokens = original.live.ids.size() - 1;
    check(refused(good, id, l, "limit", SessionError::invalid), "refused: too many tokens");

    l = exact;
    l.max_kv_layers = original.kv.size() - 1;
    check(refused(good, id, l, "limit", SessionError::invalid), "refused: too many K/V layers");

    l = exact;
    l.max_checkpoints = original.checkpoints.size() - 1;
    check(refused(good, id, l, "limit", SessionError::invalid), "refused: too many checkpoints");

    // a layer part over its cap: open() stops before the K/V, so the refusal comes from next()
    l = exact;
    l.max_kv_bytes[0][0] = original.kv[0].k.size() - 1;
    SessionFileStream stream;
    SessionStatus st;
    check(stream.open(good.string(), id, l, error, &st), "limits: open stops before the K/V");
    check(error.empty() && stream.layers() == original.kv.size(), "limits: metadata read, layers counted");
    ConversationKv layer;
    check(!stream.next(layer, error), "refused: a K/V part over its limit");
    check(error.find("limit") != std::string::npos, "refused: the part's limit is named");
    check(layer.k.size() == 0 && layer.v.size() == 0 && layer.k_scale.size() == 0 && layer.pooled.size() == 0,
          "refused: no half layer is handed back");
    check(!stream.finish(error), "refused: the stream cannot finish after a broken layer");

    // the second layer's V over its cap: the first layer still reads, the next one is refused and cleared
    l = exact;
    l.max_kv_bytes[1][1] = original.kv[1].v.size() - 1;
    SessionFileStream cut;
    check(cut.open(good.string(), id, l, error, &st), "limits: second part cap");
    ConversationKv got;
    check(cut.next(got, error), "limits: the first layer reads");
    check(!cut.next(got, error) && got.v.size() == 0, "refused: the second layer's V over its limit");

    // the read's own state machine
    SessionFileStream s;
    check(s.open(good.string(), id, exact, error, &st), "state: open");
    check(!s.finish(error), "state: finish before the layers fails");
    check(error.find("not read") != std::string::npos, "state: finish says layers are missing");
    for (size_t i = 0; i < s.layers(); ++i) check(s.next(got, error), "state: next");
    check(!s.next(got, error), "state: next past the last layer fails");
    check(error.find("no more") != std::string::npos, "state: next past the end says so");
    check(got.k.size() == 0 && got.v.size() == 0, "state: no layer past the end");
    check(s.finish(error), "state: finish after every layer");
    check(s.finish(error), "state: finish again is harmless");
    check(!s.next(got, error), "state: next after finish fails");
}

void refuse_trailer(const fs::path& dir, const SessionFileIdentity& id) {
    const SavedConversation original = small_image();
    const fs::path good = dir / "trailer.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(good.string(), original, id, written, error), "trailer: write");
    const std::vector<char> image = slurp(good);
    const SessionReadLimits exact = exact_limits(original);

    // a corrupted payload hash in the trailer, and a corrupted end marker: both pass open()
    auto read_all = [&](const fs::path& p, SessionFileStream& s) {
        SessionStatus st;
        if (!s.open(p.string(), id, exact, error, &st)) return false;
        ConversationKv layer;
        for (size_t i = 0; i < s.layers(); ++i)
            if (!s.next(layer, error)) return false;
        return true;
    };
    for (int which = 0; which < 2; ++which) {
        auto d = image;
        if (which == 0) d[d.size() - 16] ^= 0x20; else d[d.size() - 1] ^= 0x20;
        const fs::path p = dir / (which == 0 ? "badhash2.bin" : "badend.bin");
        spit(p, d);
        SessionFileStream s;
        check(read_all(p, s), "trailer: the layers still read");
        check(!s.finish(error), which == 0 ? "refused: corrupt payload hash" : "refused: bad end marker");
        check(error.find(which == 0 ? "checksum" : "end marker") != std::string::npos,
              which == 0 ? "trailer: the checksum is named" : "trailer: the end marker is named");
    }
}

// ---- the admission ask is a bound on the metadata plus one layer, not on the file
void admission(const fs::path& dir, const SessionFileIdentity& id) {
    auto build = [&](size_t layers) {
        SavedConversation s;
        for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(7 + i);
        s.layer_lo = 0;
        s.layer_hi = 27;
        s.live = checkpoint(16, 8);
        s.checkpoints = {checkpoint(8, 9)};
        for (size_t i = 0; i < layers; ++i) s.kv.push_back(kv_layer((int) i, 450000, 450000, 60000, 60000, 30000));
        return s;
    };
    auto ask = [&](const SavedConversation& s, const char* name) {
        const fs::path p = dir / (std::string(name) + ".bin");
        size_t written = 0;
        std::string error;
        check(session_file_write(p.string(), s, id, written, error), name);
        SessionReadLimits l = exact_limits(s);
        uint64_t asked = 0;
        bool admitted = false;
        l.admit = [&](uint64_t n, std::string&) { asked = n; admitted = true; return true; };
        SessionFileStream stream;
        SessionStatus st;
        check(stream.open(p.string(), id, l, error, &st) && admitted, name);
        check(asked >= (16u << 20), "admission: the 16 MiB block is counted");
        check(asked < (20u << 20), "admission: the ask is metadata plus one layer plus the block");
        return std::make_pair(asked, (uint64_t) written);
    };
    const auto three = ask(build(3), "ask3");
    const auto six = ask(build(6), "ask6");
    // six layers and twice the file: the ask is unchanged, because it counts one layer, not the file
    check(six.second > three.second + (2u << 20), "admission: the second file is larger");
    check(six.first == three.first, "admission: the ask does not grow with the file's layers");

    // unbounded limits make no claim it cannot keep
    SessionFileStream s;
    std::string error;
    SessionStatus st;
    const fs::path p = dir / "ask3.bin";
    uint64_t asked = 0;
    SessionReadLimits l;
    l.admit = [&](uint64_t n, std::string&) { asked = n; return true; };
    check(s.open(p.string(), id, l, error, &st), "admission: open with open limits");
    check(asked == UINT64_MAX, "admission: open limits report no bound");
}

// ---- a file with one checkpoint is admitted on the metadata it holds, not on max_checkpoints of them
void admission_of_one_checkpoint(const fs::path& dir, const SessionFileIdentity& id) {
    // A ~1.2 MiB running state (and the same in one checkpoint) with small K/V layers: the file's metadata plus one
    // layer is a few MiB, while the old pre-parse estimate charged 6 checkpoints (7 x the state) and asked ~27 MiB.
    auto state = [](size_t tokens, uint8_t seed, size_t gdn, size_t tails) {
        ConversationCheckpoint c = checkpoint(tokens, seed);
        c.gdn = bytes_of(gdn, seed);
        c.tails = bytes_of(tails, seed + 1);
        return c;
    };
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(7 + i);
    s.layer_lo = 0;
    s.layer_hi = 27;
    s.live = state(16, 21, 600000, 600000);
    s.checkpoints.push_back(state(16, 31, 575000, 575000));   // exactly one checkpoint: the normal save
    for (int i = 0; i < 3; ++i) s.kv.push_back(kv_layer(i, 2600000, 0, 0, 0, 0));
    const fs::path p = dir / "one-checkpoint.bin";
    size_t written = 0;
    std::string error;
    check(session_file_write(p.string(), s, id, written, error), "one checkpoint: write");

    SessionReadLimits l = exact_limits(s);
    l.max_checkpoints = 6;   // the engine's allowance, though this file holds one: it must not be charged six times
    // the restore's budget: 8 MiB for the metadata and the layer being read, plus the fixed 16 MiB file block
    const uint64_t layer_budget = 8u << 20, block = 16u << 20;
    uint64_t asked = 0;
    bool admitted = false;
    l.admit = [&](uint64_t need, std::string& why) {
        asked = need;
        admitted = true;
        if (need <= layer_budget + block) return true;
        why = "no room for " + std::to_string(need >> 20) + " MiB";
        return false;
    };
    SessionFileStream stream;
    SessionStatus st;
    check(stream.open(p.string(), id, l, error, &st) && admitted, "one checkpoint: open is admitted");
    check(st.error == SessionError::none && error.empty(), "one checkpoint: open succeeds without an error");
    check(stream.meta().checkpoints.size() == 1, "one checkpoint: the file's own count");
    std::printf("one checkpoint: metadata %llu KiB, largest layer %llu KiB, asked %llu KiB, budget %llu KiB\n",
                (unsigned long long) (stream.meta().bytes() >> 10),
                (unsigned long long) ((asked - (uint64_t) stream.meta().bytes() - block) >> 10),
                (unsigned long long) (asked >> 10), (unsigned long long) ((layer_budget + block) >> 10));
    check(asked >= (uint64_t) stream.meta().bytes() + block, "one checkpoint: the ask counts the real metadata");
    check(asked <= layer_budget + block, "one checkpoint: the ask fits the staging budget");
    check(stream.peak_bytes() < (8u << 20), "one checkpoint: the real peak is under 8 MiB");
    ConversationKv got;
    for (size_t i = 0; i < stream.layers(); ++i) {
        check(stream.next(got, error), "one checkpoint: next");
        check(same_kv(got, s.kv[i]), "one checkpoint: layer bytes");
    }
    check(stream.finish(error), "one checkpoint: finish");
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-stream-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1122334455667788ull, 0x99aabbccddeeff00ull};

    round_trip_image(dir, id);
    round_trip_sources(dir, id);
    peak_and_reuse(dir, id);
    refusals(dir, id);
    refuse_limits(dir, id);
    refuse_trailer(dir, id);
    admission(dir, id);
    admission_of_one_checkpoint(dir, id);

    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("conversation_stream_test: %d checks passed\n", checks);
    return 0;
}
