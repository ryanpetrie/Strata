// src/core/lora_pack.cpp - reading and checking a LoRA adapter's flat index (`include/strata/core/lora_pack.hpp`).
//
// The rule this file enforces is the plan's 3.5: the adapter is REJECTED at load time with the offending name in
// the message, never partially loaded, never warned about, never skipped.  The reason is the format's own failure
// mode - a transposed or truncated adapter produces plausible numbers and a silently wrong model, and the only
// place that can be caught is here.
#include "strata/core/lora_pack.hpp"

#include "strata/core/family_layout.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
namespace {

constexpr const char* kVersion = "# strata lora index v1";
constexpr uint64_t kFp16 = 2;

/// Decimal MB/GB, because the sizes in the refusal are compared with the plan's own table (3.3: 1.52 GB fused,
/// 7.55 GB per expert) and those are decimal.  A refusal that says 10.1 MB where the plan says 10.6 MB is a
/// second opinion nobody asked for.
std::string human_bytes(double b) {
    char out[64];
    if (b >= 1e9) std::snprintf(out, sizeof out, "%.2f GB", b / 1e9);
    else std::snprintf(out, sizeof out, "%.1f MB", b / 1e6);
    return out;
}

bool read_all(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    out.clear();
    uint8_t buf[1 << 16];
    for (;;) {
        const size_t n = std::fread(buf, 1, sizeof buf, f);
        if (n > 0) out.insert(out.end(), buf, buf + n);
        if (n < sizeof buf) break;
    }
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) { err = "cannot read " + path; return false; }
    return true;
}

/// Every name this model has, with its GGUF shape, over the trunk and any in-file MTP blocks.
using ShapeTable = std::map<std::string, std::vector<int64_t>>;

ShapeTable shapes_of(const ModelGeometry& g) {
    ShapeTable t;
    for (const TensorWant& w : global_tensors(g)) t.emplace(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const TensorWant& w : layer_tensors(g, l)) t.emplace("blk." + std::to_string(l) + "." + w.name, w.shape);
    return t;
}

/// "3, 7, 11, 15, ... 47" - so a refusal can say which layers DO have the tensor.
std::string qsa_layers(const ModelGeometry& g) {
    std::string s;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!is_qsa_layer(g, l)) continue;
        s += (s.empty() ? "" : ", ") + std::to_string(l);
    }
    return s;
}

/// The suffix a blk.N tensor carries, or "" when `name` is not a blk.N tensor.
std::string blk_suffix(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return std::string();
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos) return std::string();
    return name.substr(dot + 1);
}

/// The engine's rule for a matrix in this format: element (i, j) sits at `i + ne0 * j`.
double at(const std::vector<uint16_t>& m, int64_t ne0, int64_t i, int64_t j) {
    return (double) kernels::f32_from_f16(m[(size_t) (i + ne0 * j)]);
}

struct Row {
    std::string base;  // the base tensor's engine name
    bool b = false;    // .lora_b
    int64_t rank = 0;
    double alpha = 0.0;
    LoraMatrix m;
    int line = 0;
};

bool parse_row(char* line, int lineno, Row& r, std::string& err) {
    char name[256] = {0};
    long long rank = 0, ne0 = 0, ne1 = 0;
    double alpha = 0.0;
    unsigned long long off = 0, bytes = 0;
    // The rows are written by tools/lora_pack.py and are space separated with no quoting; a name with a space
    // would break the parse, which is why a row that does not carry exactly its seven fields is REFUSED.
    char extra[8] = {0};
    const int n = std::sscanf(line, "%255s %lld %lf %lld %lld %llu %llu %7s", name, &rank, &alpha, &ne0, &ne1, &off,
                              &bytes, extra);
    if (n != 7) {
        err = "index.txt line " + std::to_string(lineno) + " is not `name rank alpha ne0 ne1 offset bytes`";
        return false;
    }
    std::string base(name);
    const std::string a_suffix = ".lora_a", b_suffix = ".lora_b";
    if (base.size() > b_suffix.size() && base.compare(base.size() - b_suffix.size(), b_suffix.size(), b_suffix) == 0) {
        r.b = true;
        base.resize(base.size() - b_suffix.size());
    } else if (base.size() > a_suffix.size() && base.compare(base.size() - a_suffix.size(), a_suffix.size(), a_suffix) == 0) {
        r.b = false;
        base.resize(base.size() - a_suffix.size());
    } else {
        err = "index.txt line " + std::to_string(lineno) + ": `" + name + "` does not end in .lora_a or .lora_b";
        return false;
    }
    r.base = base;
    r.rank = rank;
    r.alpha = alpha;
    r.m.ne0 = ne0;
    r.m.ne1 = ne1;
    r.m.offset = off;
    r.m.bytes = bytes;
    r.line = lineno;
    if (r.rank < 1) {
        err = base + ": rank " + std::to_string(r.rank) + " is not a rank";
        return false;
    }
    if (r.m.ne0 < 1 || r.m.ne1 < 1) {
        err = base + ": the row's shape is [" + std::to_string(r.m.ne0) + ", " + std::to_string(r.m.ne1) + "]";
        return false;
    }
    const uint64_t want = (uint64_t) (r.m.ne0 * r.m.ne1) * kFp16;
    if (r.m.bytes != want) {
        err = base + (r.b ? ".lora_b" : ".lora_a") + ": the row says " + std::to_string(r.m.bytes) + " bytes but [" +
              std::to_string(r.m.ne0) + ", " + std::to_string(r.m.ne1) + "] fp16 is " + std::to_string(want);
        return false;
    }
    if (r.rank > kMaxLoraRank) {
        err = base + ": rank " + std::to_string(r.rank) + " is above this engine's maximum (" +
              std::to_string(kMaxLoraRank) + ")";
        return false;
    }
    return true;
}

/// A routed-expert base tensor is refused with BOTH of its sizes, because the answer is a capacity decision and
/// not a formatting one (the plan's 3.3, whose two numbers this reproduces from the geometry in the name's shape).
bool refuse_routed_expert(const std::string& name, const std::vector<int64_t>& shape, int64_t rank,
                          const ModelGeometry& g, std::string& err) {
    const int64_t out = shape[0], in = shape[1], ne = shape[2];
    const double fused = (double) rank * (double) (in * ne + out) * kFp16;
    const double per_expert = (double) ne * (double) rank * (double) (in + out) * kFp16;
    const int64_t layers = std::max<int64_t>(1, g.n_layers);
    err = name + ": a routed-expert adapter is refused.  At rank " + std::to_string(rank) + " one adapter over the " +
          std::to_string(ne) + " experts' fused weights is " + human_bytes(fused) + " for this tensor (" +
          human_bytes(fused * layers) + " over the " + std::to_string(layers) + " layers), and one adapter per " +
          "expert is " + human_bytes(per_expert) + " per tensor (" + human_bytes(per_expert * layers) + " over all " +
          std::to_string(layers) + "); the engine holds the adapter beside the base weights, inside the headroom " +
          "the plan measured at 58 MiB.  The MoE-compatible adapter is the attention-only one (plan 3.3).";
    return false;
}

}  // namespace

bool lora_load(const std::string& dir, const ModelGeometry& g, LoraAdapter& out, std::string& err) {
    const std::string index = dir + "/index.txt";
    std::FILE* f = std::fopen(index.c_str(), "rb");
    if (!f) { err = "cannot open " + index + " (run tools/lora_pack.py)"; return false; }

    std::vector<Row> rows;
    char line[1024];
    int lineno = 0;
    bool versioned = false;
    while (std::fgets(line, sizeof line, f)) {
        ++lineno;
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] == '#') {
            // The version line is the format's own guard: a pack's index.txt would otherwise parse as garbage.
            if (!versioned) {
                if (std::strncmp(line, kVersion, std::strlen(kVersion)) != 0) {
                    err = index + " does not start with `" + kVersion +
                          "` - this is not a LoRA index (a pack's index.txt is not one)";
                    std::fclose(f);
                    return false;
                }
                versioned = true;
            }
            continue;
        }
        if (!versioned) {
            err = index + " has no `" + kVersion + "` header before its first row";
            std::fclose(f);
            return false;
        }
        Row r;
        if (!parse_row(line, lineno, r, err)) { std::fclose(f); return false; }
        rows.push_back(std::move(r));
    }
    std::fclose(f);
    if (!versioned) { err = index + " is empty (no `" + kVersion + "` header)"; return false; }
    if (rows.empty()) { err = index + " names no tensor"; return false; }

    const ShapeTable shapes = shapes_of(g);
    LoraAdapter made;
    std::map<std::string, bool> seen_a, seen_b;

    for (const Row& r : rows) {
        bool& seen = r.b ? seen_b[r.base] : seen_a[r.base];
        if (seen) {
            err = r.base + (r.b ? ".lora_b" : ".lora_a") + " appears twice (line " + std::to_string(r.line) + ")";
            return false;
        }
        seen = true;

        const auto it = shapes.find(r.base);
        if (it == shapes.end()) {
            const std::string suffix = blk_suffix(r.base);
            if (!suffix.empty()) {
                // A name that a layer of the OTHER kind has: the commonest mistake is a QSA projection on a GDN
                // layer, and saying which layers do have it is the whole value of the message.
                bool on_other_kind = false;
                for (int64_t l = 0; l < g.n_layers && !on_other_kind; ++l) {
                    if (is_qsa_layer(g, l) == is_qsa_layer(g, 0)) continue;
                    for (const TensorWant& w : layer_tensors(g, l))
                        if (w.name == suffix) { on_other_kind = true; break; }
                }
                const size_t dot = r.base.find('.', 4);
                const std::string layer = r.base.substr(4, dot - 4);
                if (on_other_kind) {
                    const bool qsa_only = suffix.rfind("attn_", 0) == 0 || suffix == "indexer" ||
                                          suffix.rfind("indexer.", 0) == 0;
                    err = r.base + ": layer " + layer + " is a " + (qsa_only ? "GDN" : "full-attention") +
                          " layer and has no `" + suffix + "`" +
                          (qsa_only ? "; the full-attention layers are " + qsa_layers(g) : std::string());
                    return false;
                }
            }
            err = r.base + ": this model does not have that tensor (the layout table names every tensor of every "
                  "layer and every global one)";
            return false;
        }
        const std::vector<int64_t>& shape = it->second;
        if (shape.size() > 2) return refuse_routed_expert(r.base, shape, r.rank, g, err);
        if (shape.size() != 2) {
            err = r.base + ": the adapter names a " + std::to_string(shape.size()) +
                  "-dimensional tensor; a LoRA targets a projection";
            return false;
        }
        const int64_t out_dim = shape[0], in_dim = shape[1];
        const int64_t want0 = r.b ? out_dim : r.rank;
        const int64_t want1 = r.b ? r.rank : in_dim;
        if (r.m.ne0 != want0 || r.m.ne1 != want1) {
            // The message shows the adapter's A and B TOGETHER, because it is the PAIR that is wrong.
            std::string a_shape = "?", b_shape = "?";
            for (const Row& o : rows)
                if (o.base == r.base) (o.b ? b_shape : a_shape) =
                    "[" + std::to_string(o.m.ne0) + ", " + std::to_string(o.m.ne1) + "]";
            err = r.base + ": the adapter's A is " + a_shape + " and B is " + b_shape + "; the model's tensor is [" +
                  std::to_string(out_dim) + ", " + std::to_string(in_dim) + "]";
            return false;
        }

        LoraTarget* t = nullptr;
        for (LoraTarget& existing : made.targets)
            if (existing.base == r.base) { t = &existing; break; }
        if (!t) {
            made.targets.push_back(LoraTarget{});
            t = &made.targets.back();
            t->base = r.base;
            t->out = out_dim;
            t->in = in_dim;
            t->rank = r.rank;
            t->alpha = r.alpha;
        } else if (t->rank != r.rank || t->alpha != r.alpha) {
            err = r.base + ": its A and B disagree on rank/alpha (" + std::to_string(t->rank) + "/" +
                  std::to_string(t->alpha) + " against " + std::to_string(r.rank) + "/" + std::to_string(r.alpha) +
                  ")";
            return false;
        }
        (r.b ? t->b : t->a) = r.m;
    }

    for (const LoraTarget& t : made.targets) {
        if (!seen_a[t.base]) { err = t.base + ".lora_a is missing (the index has its B only)"; return false; }
        if (!seen_b[t.base]) { err = t.base + ".lora_b is missing (the index has its A only)"; return false; }
    }

    // The blob must be exactly the bytes the index describes: a leftover from another adapter is the one failure
    // a size check catches for free.
    const std::string blob_path = dir + "/lora.bin";
    std::FILE* blob = std::fopen(blob_path.c_str(), "rb");
    if (!blob) { err = "cannot open " + blob_path; return false; }
    uint64_t blob_bytes = 0;
#if defined(_WIN32)
    if (_fseeki64(blob, 0, SEEK_END) == 0) blob_bytes = (uint64_t) _ftelli64(blob);
#else
    if (fseeko(blob, 0, SEEK_END) == 0) blob_bytes = (uint64_t) ftello(blob);
#endif
    std::fclose(blob);
    uint64_t covered = 0;
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    for (const LoraTarget& t : made.targets) {
        for (const LoraMatrix& m : {t.a, t.b}) {
            spans.emplace_back(m.offset, m.offset + m.bytes);
            covered += m.bytes;
        }
    }
    std::sort(spans.begin(), spans.end());
    uint64_t high = 0;
    for (const auto& s : spans) {
        if (s.first < high) { err = blob_path + ": two rows overlap at byte " + std::to_string(s.first); return false; }
        high = s.second;
    }
    if (high != blob_bytes) {
        err = blob_path + " is " + std::to_string(blob_bytes) + " bytes but the index describes " +
              std::to_string(high) + " (a stale blob from another adapter?)";
        return false;
    }
    if (covered != blob_bytes) {
        err = blob_path + " is " + std::to_string(blob_bytes) + " bytes but its rows use " +
              std::to_string(covered) + " of them";
        return false;
    }
    made.blob_bytes = blob_bytes;

    out = std::move(made);
    return true;
}

bool lora_read_f16(const std::string& dir, const LoraMatrix& m, std::vector<uint16_t>& out, std::string& err) {
    const std::string path = dir + "/lora.bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
#if defined(_WIN32)
    const int seek = _fseeki64(f, (long long) m.offset, SEEK_SET);
#else
    const int seek = fseeko(f, (off_t) m.offset, SEEK_SET);
#endif
    if (seek != 0) { std::fclose(f); err = path + ": seek to " + std::to_string(m.offset) + " failed"; return false; }
    out.assign((size_t) (m.ne0 * m.ne1), 0);
    const size_t got = std::fread(out.data(), kFp16, out.size(), f);
    std::fclose(f);
    if (got != out.size()) {
        err = path + ": short read at " + std::to_string(m.offset) + " (" + std::to_string(got) + " of " +
              std::to_string(out.size()) + " fp16 values)";
        return false;
    }
    return true;
}

bool lora_delta_f32(const std::string& dir, const LoraTarget& t, const float* x, float* dy, std::string& err) {
    std::vector<uint16_t> a, b;
    if (!lora_read_f16(dir, t.a, a, err)) return false;
    if (!lora_read_f16(dir, t.b, b, err)) return false;
    // The two products, in double, then one rounding: the reference exists to decide whether a kernel's float
    // arithmetic is right, so its own error has to be far below the thing it measures.
    std::vector<double> mid((size_t) t.rank, 0.0);
    for (int64_t k = 0; k < t.rank; ++k) {
        double s = 0.0;
        for (int64_t j = 0; j < t.in; ++j) s += at(a, t.a.ne0, k, j) * (double) x[j];
        mid[(size_t) k] = s;
    }
    const double scale = t.scale();
    for (int64_t i = 0; i < t.out; ++i) {
        double s = 0.0;
        for (int64_t k = 0; k < t.rank; ++k) s += at(b, t.b.ne0, i, k) * mid[(size_t) k];
        dy[i] += (float) (scale * s);
    }
    return true;
}

bool lora_digest(const std::string& dir, const LoraAdapter& a, uint64_t& out, std::string& err) {
    // FNV-1a over what makes the adapter the same adapter: its rows (name, rank, alpha, shape, scaling) and the
    // bytes they point at.  One byte of weight or one digit of alpha changes it, which is what a saved
    // conversation needs to refuse a resume against a different adapter.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* p, size_t n) {
        const uint8_t* q = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) { h ^= q[i]; h *= 1099511628211ull; }
    };
    std::vector<uint8_t> blob;
    if (!read_all(dir + "/lora.bin", blob, err)) return false;
    for (const LoraTarget& t : a.targets) {
        mix(t.base.data(), t.base.size());
        const int64_t shape[4] = {t.out, t.in, t.rank, (int64_t) (t.alpha * 1000.0)};
        mix(shape, sizeof shape);
        for (const LoraMatrix& m : {t.a, t.b}) {
            const uint64_t mm[2] = {m.offset, m.bytes};
            mix(mm, sizeof mm);
            if (m.offset + m.bytes > blob.size()) { err = "the adapter's rows point past lora.bin"; return false; }
            mix(blob.data() + m.offset, (size_t) m.bytes);
        }
    }
    out = h;
    return true;
}

}  // namespace strata::core
