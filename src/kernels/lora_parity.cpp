// src/kernels/lora_parity.cpp - the LoRA delta kernel (`strata/kernels/lora.hpp`) against the host reference.
//
//   (a) flag off: every destination buffer is BIT-IDENTICAL to its pre-call bytes - the kernel returns on the
//       device before writing one of them (docs/research/phase2_engine_lora_plan.md 5.4, the mechanical form of
//       "the chat path is byte-identical").
//   (b) flag on: `core::lora_delta_f32` (double internally) on a fixed synthetic adapter and a ten-token input,
//       for non-square shapes and all three activations, within an fp32 tolerance.
//   (c) the flag moves on the device: off writes nothing, on writes again AND reproduces the first on-run bit
//       for bit (the accumulation order does not wander between calls).
//   (d) one call per layer applies every entry of that layer, and only that layer's entries.  The QSA layer is
//       the case the API had to grow for: `attn_q`, `attn_k` and `attn_v` all read the layer's input but write
//       three different tensors (30, 10 and 10 wide here), and `attn_output` reads a fourth activation - four
//       outputs, four buffers, one launch.  Two outputs have the SAME width on purpose: a kernel that mixed them
//       up by width or position would still have to reproduce each tensor's own delta to pass.
//   (e) a missing activation or a missing output returns false and writes nothing.
//
// The adapter is a real directory (`index.txt` + `lora.bin`) loaded by `lora_load`, because the kernel's
// dimensions and the reference's both come from the loader's `LoraTarget`; a hand-built description would let
// the two disagree about the orientation the format exists to pin (`include/strata/core/lora_pack.hpp`).
#include "strata/core/lora_pack.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/lora.hpp"

#include "cuda/lora_vram.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace core = strata::core;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;

/// The tolerance for (b): the reference sums in double and the kernel in fp32 with fma over rank <= 10 terms
/// of O(1) magnitude.  The measured worst error is printed beside each check; 2e-5 is two orders above it and
/// well below the smallest delta the adapter produces here, so it can only pass on a correct kernel.
constexpr double kTol = 2e-5;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

template <typename T>
void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

/// The same small, asymmetric geometry the host test uses (`src/core/lora_pack_test.cpp`): layer 0 is a GDN
/// layer, layer 3 the full-attention one, and every projection's in and out differ, so a transposed kernel or a
/// swapped stride cannot pass by symmetry.  The widths are hand-checked:
///
///   blk.0.attn_qkv.weight    {in n_embd 8,  out ssm_conv_channels 300}  (out > kThreads: the output loop strides)
///   blk.0.ssm_out.weight     {in ssm_value_dim 10, out n_embd 8}
///   blk.3.attn_q.weight      {in n_embd 8,  out 2*n_head*head_dim 30}
///   blk.3.attn_k.weight      {in n_embd 8,  out n_head_kv*head_dim 10}
///   blk.3.attn_v.weight      {in n_embd 8,  out n_head_kv*head_dim 10}  (k and v the same width, on purpose)
///   blk.3.attn_output.weight {in n_head*head_dim 15, out n_embd 8}
core::ModelGeometry tiny_geometry() {
    core::ModelGeometry g;
    g.n_embd = 8;
    g.n_layers = 4;
    g.qsa_interval = 4;
    g.n_head = 3;
    g.head_dim = 5;
    g.n_head_kv = 2;
    g.n_vocab = 16;
    g.ssm_state_size = 2;
    g.ssm_k_heads = 1;
    g.ssm_v_heads = 2;
    g.ssm_d_conv = 2;
    g.ssm_conv_channels = 300;
    g.ssm_value_dim = 10;
    g.n_expert = 2;
    g.n_ff = 3;
    g.n_ff_shexp = 3;
    return g;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-lora-parity-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
    std::string str() const { return path.string(); }
};

std::string row(const std::string& base, int64_t rank, double alpha, int64_t ne0, int64_t ne1, uint64_t off) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s %lld %.1f %lld %lld %llu %llu", base.c_str(), (long long) rank, alpha,
                  (long long) ne0, (long long) ne1, (unsigned long long) off,
                  (unsigned long long) ((uint64_t) ne0 * (uint64_t) ne1 * 2));
    return buf;
}

bool write_adapter(const fs::path& dir, const std::vector<std::string>& rows, const std::vector<uint16_t>& blob) {
    std::ofstream idx(dir / "index.txt", std::ios::binary);
    idx << "# strata lora index v1\n# name rank alpha ne0 ne1 offset bytes\n";
    for (const std::string& r : rows) idx << r << "\n";
    idx.close();
    std::ofstream bin(dir / "lora.bin", std::ios::binary);
    bin.write((const char*) blob.data(), (std::streamsize) (blob.size() * 2));
    return idx.good() && bin.good();
}

struct Spec {
    const char* base;
    int64_t rank;
    double alpha;
    int64_t in, out;   // the model's GGUF shape: the projection's input, then its output
};

struct Adapter {
    std::vector<std::string> rows;
    std::vector<uint16_t> blob;
};

/// Writes PEFT's A [rank, in] and B [out, rank] into the format's transposed layout: A^T [in][rank] then
/// B^T [rank][out], exactly as tools/lora_pack.py does.
Adapter make_adapter(const std::vector<Spec>& specs, std::mt19937& rng) {
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    Adapter ad;
    uint64_t off = 0;
    for (const Spec& s : specs) {
        std::vector<uint16_t> ab, bb;
        ab.reserve((size_t) (s.rank * s.in));
        bb.reserve((size_t) (s.out * s.rank));
        for (int64_t j = 0; j < s.in; ++j)
            for (int64_t k = 0; k < s.rank; ++k) ab.push_back(k::f16_from_f32(ud(rng)));
        for (int64_t k = 0; k < s.rank; ++k)
            for (int64_t i = 0; i < s.out; ++i) bb.push_back(k::f16_from_f32(ud(rng)));
        ad.rows.push_back(row(std::string(s.base) + ".lora_a", s.rank, s.alpha, s.rank, s.in, off));
        off += (uint64_t) ab.size() * 2;
        ad.rows.push_back(row(std::string(s.base) + ".lora_b", s.rank, s.alpha, s.out, s.rank, off));
        off += (uint64_t) bb.size() * 2;
        ad.blob.insert(ad.blob.end(), ab.begin(), ab.end());
        ad.blob.insert(ad.blob.end(), bb.begin(), bb.end());
    }
    return ad;
}

/// One destination buffer: T rows of `out` floats at stride `ld` (wider than `out`, so a wrong stride or a
/// write past the row shows up as a moved padding value).
struct Buf {
    int64_t out = 0, ld = 0;
    std::vector<float> host;
    float* dev = nullptr;
};

struct Metrics {
    double worst = 0.0;    // max |kernel - reference|
    double biggest = 0.0;  // max |delta|, so the check can be read against what it is measuring
    bool pad_ok = true;    // the row padding kept its sentinel
};

}  // namespace

int main() {
    const core::ModelGeometry g = tiny_geometry();
    constexpr int64_t T = 10;   // more than one token tile (kTokensPerBlock is 8), at a padded row stride
    std::mt19937 rng(20261023);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::string err;

    // Layer 0 (GDN): two entries, two activations.  Layer 3 (QSA): four entries, three of them on the layer's
    // input and writing three different tensors - the q/k/v case, with k and v the same width.
    const std::vector<Spec> spec = {
        {"blk.0.attn_qkv.weight", 10, 4.0, 8, 300},   // kLoraActInput, rank > kWarps and out > kThreads
        {"blk.0.ssm_out.weight", 2, 3.0, 10, 8},      // kLoraActSsm
        {"blk.3.attn_q.weight", 3, 2.0, 8, 30},       // kLoraActInput
        {"blk.3.attn_k.weight", 2, 4.0, 8, 10},       // kLoraActInput, same width as v
        {"blk.3.attn_v.weight", 4, 8.0, 8, 10},       // kLoraActInput, same width as k
        {"blk.3.attn_output.weight", 5, 2.0, 15, 8},  // kLoraActAttn
    };

    // ------------------------------------------------------------------ the adapter, through the real loader
    std::printf("fixture\n");
    TempDir dir;
    {
        const Adapter ad = make_adapter(spec, rng);
        check(write_adapter(dir.path, ad.rows, ad.blob), "the index and the fp16 blob are written");
    }
    core::LoraAdapter adapter;
    if (!core::lora_load(dir.str(), g, adapter, err)) {
        std::fprintf(stderr, "lora_load: %s\n", err.c_str());
        return 2;
    }
    check(adapter.targets.size() == spec.size(), "six targets load against the fixture geometry");
    bool shapes_ok = true;
    for (const Spec& s : spec) {
        const core::LoraTarget* t = adapter.find(s.base);
        shapes_ok = shapes_ok && t != nullptr && t->in == s.in && t->out == s.out;
    }
    check(shapes_ok, "every target's in/out is the layout table's, in GGUF order");
    if (!k::lora_upload(adapter, dir.str(), err)) {
        std::fprintf(stderr, "lora_upload: %s\n", err.c_str());
        return 2;
    }
    check(k::lora().loaded() && k::lora().n_entries == (int64_t) spec.size() && k::lora().n_layers == 4,
          "the device table holds every entry and stops at the last adapted layer");
    check(k::lora().covers(0) && k::lora().covers(3) && !k::lora().covers(1) && !k::lora().covers(4),
          "covers(): the two adapted layers, no others, and not past the table");
    check(k::lora_enabled(), "a fresh upload starts ON");
    if (!k::lora_replicate(err)) { std::fprintf(stderr, "lora_replicate: %s\n", err.c_str()); return 2; }
    check(k::lora().loaded(), "lora_replicate on the device that already holds the tables is a no-op");

    // `lora_find` names each entry of a layer, so the test fills `outs` the way a call site does instead of
    // assuming the adapter's order.  A name the layer does not target is -1.
    const int i_q = k::lora_find(3, "attn_q.weight");
    const int i_k = k::lora_find(3, "attn_k.weight");
    const int i_v = k::lora_find(3, "attn_v.weight");
    const int i_o = k::lora_find(3, "attn_output.weight");
    const int i_qkv0 = k::lora_find(0, "attn_qkv.weight");
    const int i_ssm0 = k::lora_find(0, "ssm_out.weight");
    {
        std::vector<int> got = {i_q, i_k, i_v, i_o};
        std::sort(got.begin(), got.end());
        check(got == std::vector<int>({0, 1, 2, 3}),
              "lora_find gives layer 3's four entries four distinct indices in 0..3");
    }
    check(i_qkv0 >= 0 && i_ssm0 >= 0 && i_qkv0 != i_ssm0, "and layer 0's two entries two indices");
    check(k::lora_find(3, "blk.3.attn_k.weight") == i_k, "the full `blk.<layer>.<tensor>` spelling also resolves");
    check(k::lora_find(3, "attn_qkv.weight") == -1 && k::lora_find(0, "attn_q.weight") == -1,
          "a tensor the layer does not target is -1 (a GDN tensor on a QSA layer and the reverse)");
    check(k::lora_find(1, "attn_q.weight") == -1 && k::lora_find(99, "attn_q.weight") == -1 &&
              k::lora_find(3, nullptr) == -1,
          "a layer with no entries, a layer past the table, and a null name are all -1");

    // The activation widths, taken from the adapter so a fixture change cannot silently desynchronize the
    // kernel's n_in from the buffer the test passes.
    const int64_t W_in = adapter.find("blk.0.attn_qkv.weight")->in;
    const int64_t W_attn = adapter.find("blk.3.attn_output.weight")->in;
    const int64_t W_ssm = adapter.find("blk.0.ssm_out.weight")->in;
    const int64_t LD_IN = W_in + 1, LD_ATTN = W_attn + 1, LD_SSM = W_ssm + 1;
    check(adapter.find("blk.3.attn_q.weight")->in == W_in && adapter.find("blk.3.attn_k.weight")->in == W_in &&
              adapter.find("blk.3.attn_v.weight")->in == W_in,
          "every input-read projection of both layers has the same input width");
    check(W_in != W_attn && W_in != W_ssm && W_attn != W_ssm, "the three activations have three different widths");

    std::map<std::string, Buf> ybuf;
    for (size_t i = 0; i < spec.size(); ++i) {
        const Spec& s = spec[i];
        Buf b;
        b.out = s.out;
        b.ld = s.out + 2 + (int64_t) i;   // a different stride per entry, so one swapped stride cannot hide
        b.host.assign((size_t) (T * b.ld), 1234.5f);
        ybuf.emplace(s.base, std::move(b));
    }
    for (auto& kv : ybuf) kv.second.dev = dalloc<float>(kv.second.host.size());

    std::vector<float> hxin((size_t) (T * LD_IN)), hxatt((size_t) (T * LD_ATTN)), hxssm((size_t) (T * LD_SSM));
    for (float& v : hxin) v = ud(rng);
    for (float& v : hxatt) v = ud(rng);
    for (float& v : hxssm) v = ud(rng);
    for (int64_t t = 0; t < T; ++t) {   // a padding cell a wrong stride would read
        hxin[(size_t) (t * LD_IN + W_in)] = 999.0f;
        hxatt[(size_t) (t * LD_ATTN + W_attn)] = 999.0f;
        hxssm[(size_t) (t * LD_SSM + W_ssm)] = 999.0f;
    }
    float* dxin = dalloc<float>(hxin.size());
    float* dxatt = dalloc<float>(hxatt.size());
    float* dxssm = dalloc<float>(hxssm.size());
    up(dxin, hxin);
    up(dxatt, hxatt);
    up(dxssm, hxssm);

    // one activation per slot; the outputs are per entry and filled through `lora_find` below
    auto io_for = [&](bool input, bool attn, bool ssm) {
        std::array<k::LoraIo, k::kLoraActCount> io{};
        if (input) io[k::kLoraActInput] = k::LoraIo{dxin, LD_IN};
        if (attn) io[k::kLoraActAttn] = k::LoraIo{dxatt, LD_ATTN};
        if (ssm) io[k::kLoraActSsm] = k::LoraIo{dxssm, LD_SSM};
        return io;
    };
    auto outs_for = [&](int64_t layer, const std::vector<const char*>& names) {
        std::vector<k::LoraOut> o(names.size());
        for (const char* n : names) {
            const int idx = k::lora_find(layer, n);
            if (idx < 0) continue;   // the checks above already report it; leave a null output so apply refuses
            const Buf& b = ybuf.at(n);
            o[(size_t) idx] = k::LoraOut{b.dev, b.ld};
        }
        return o;
    };
    const auto io0 = io_for(true, false, true);
    const auto io3 = io_for(true, true, false);
    const auto outs0 = outs_for(0, {"blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"});
    const auto outs3 = outs_for(3, {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                                   "blk.3.attn_output.weight"});

    auto fill = [&](bool base_values) {
        const float scale = base_values ? 1.0f : 3.0f;   // the off-run only needs bytes, not magnitudes
        for (auto& kv : ybuf) {
            Buf& b = kv.second;
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t i = 0; i < b.out; ++i)
                    b.host[(size_t) (t * b.ld + i)] = ud(rng) * scale;
                for (int64_t i = b.out; i < b.ld; ++i) b.host[(size_t) (t * b.ld + i)] = 1234.5f;
            }
            up(b.dev, b.host);
        }
    };
    auto download = [&] {
        for (auto& kv : ybuf) kv.second.host = down(kv.second.dev, kv.second.host.size());
    };
    auto apply = [&](int64_t layer, const std::array<k::LoraIo, k::kLoraActCount>& io,
                     const std::vector<k::LoraOut>& outs) {
        const bool ok = k::lora_apply_layer(layer, T, io.data(), outs.data(), nullptr);
        ck(cudaDeviceSynchronize(), "apply");
        return ok;
    };

    // the same act mapping the upload applies, so the reference is fed the buffer that entry reads
    auto act_of = [](const std::string& base) {
        if (contains(base, "attn_output")) return (int) k::kLoraActAttn;
        if (contains(base, "ssm_out")) return (int) k::kLoraActSsm;
        return (int) k::kLoraActInput;
    };

    // ------------------------------------------------------------------ (a) switched off
    std::printf("switched off\n");
    k::lora_set_enabled(false);
    check(!k::lora_enabled(), "lora_set_enabled(false) clears the host flag");
    fill(/*base_values=*/false);
    std::map<std::string, std::vector<float>> pre;
    for (const auto& kv : ybuf) pre.emplace(kv.first, kv.second.host);
    check(apply(0, io0, outs0), "layer 0 launches with the flag off");
    check(apply(3, io3, outs3), "layer 3 launches with the flag off");
    check(k::lora_apply_layer(1, T, io0.data(), outs0.data(), nullptr),
          "a layer the adapter does not touch returns true");
    ck(cudaDeviceSynchronize(), "untouched layer");
    download();
    {
        bool same = true;
        for (const auto& kv : ybuf)
            same = same && std::memcmp(kv.second.host.data(), pre.at(kv.first).data(),
                                       kv.second.host.size() * sizeof(float)) == 0;
        check(same, "off: every destination is BIT-IDENTICAL to its pre-call bytes");
    }

    // ------------------------------------------------------------------ (b) the delta, against the reference
    std::printf("the delta\n");
    k::lora_set_enabled(true);
    check(k::lora_enabled(), "lora_set_enabled(true) arms the device flag");
    fill(/*base_values=*/true);
    std::map<std::string, std::vector<float>> before;
    for (const auto& kv : ybuf) before.emplace(kv.first, kv.second.host);
    check(apply(0, io0, outs0), "one call applies layer 0's two entries");
    check(apply(3, io3, outs3), "one call applies layer 3's four entries (q, k, v and attn_output)");
    download();
    std::map<std::string, Metrics> on_metrics;
    for (const auto& kv : ybuf) {
        const std::string& base = kv.first;
        const core::LoraTarget* t = adapter.find(base);
        const Buf& b = kv.second;
        const int act = act_of(base);
        const std::vector<float>& xs = act == k::kLoraActAttn ? hxatt : (act == k::kLoraActSsm ? hxssm : hxin);
        const int64_t ldx = act == k::kLoraActAttn ? LD_ATTN : (act == k::kLoraActSsm ? LD_SSM : LD_IN);
        Metrics m;
        for (int64_t tok = 0; tok < T; ++tok) {
            std::vector<float> xr((size_t) t->in);
            std::copy(xs.begin() + tok * ldx, xs.begin() + tok * ldx + t->in, xr.begin());
            std::vector<float> want(before.at(base).begin() + tok * b.ld,
                                    before.at(base).begin() + tok * b.ld + b.out);
            if (!core::lora_delta_f32(dir.str(), *t, xr.data(), want.data(), err)) {
                std::fprintf(stderr, "lora_delta_f32: %s\n", err.c_str());
                return 2;
            }
            for (int64_t i = 0; i < b.out; ++i) {
                const float got = b.host[(size_t) (tok * b.ld + i)];
                const float had = before.at(base)[(size_t) (tok * b.ld + i)];
                m.worst = std::fmax(m.worst, std::fabs((double) got - want[(size_t) i]));
                m.biggest = std::fmax(m.biggest, std::fabs((double) want[(size_t) i] - had));
            }
            for (int64_t i = b.out; i < b.ld; ++i)
                m.pad_ok = m.pad_ok && b.host[(size_t) (tok * b.ld + i)] == 1234.5f;
        }
        std::printf("    %-30s max |err| %.3g (largest delta %.3g)\n", base.c_str(), m.worst, m.biggest);
        check(m.worst < kTol && m.pad_ok, base + ": the delta matches core::lora_delta_f32, strides respected");
        on_metrics.emplace(base, m);
    }
    // every entry moved its own output, so a "matches" result cannot come from an untouched buffer
    {
        bool moved = true;
        for (const auto& kv : on_metrics) moved = moved && kv.second.biggest > 1e-3;
        check(moved, "every entry moved its own output by a measurable amount");
    }
    std::map<std::string, std::vector<float>> on_result;
    for (const auto& kv : ybuf) on_result.emplace(kv.first, kv.second.host);
    // The two same-width outputs (k and v) carry different deltas, so neither was written into the other's
    // buffer: a shared output for the three input entries cannot pass the row checks above OR this one.
    {
        const auto dq = on_metrics.at("blk.3.attn_q.weight");
        const auto dk = on_metrics.at("blk.3.attn_k.weight");
        const auto dv = on_metrics.at("blk.3.attn_v.weight");
        const Buf& qb = ybuf.at("blk.3.attn_q.weight");
        const Buf& kb = ybuf.at("blk.3.attn_k.weight");
        const Buf& vb = ybuf.at("blk.3.attn_v.weight");
        bool same_kv = true;
        for (int64_t tok = 0; tok < T; ++tok)
            for (int64_t i = 0; i < kb.out; ++i) {
                const float kd = kb.host[(size_t) (tok * kb.ld + i)] -
                                 before.at("blk.3.attn_k.weight")[(size_t) (tok * kb.ld + i)];
                const float vd = vb.host[(size_t) (tok * vb.ld + i)] -
                                 before.at("blk.3.attn_v.weight")[(size_t) (tok * vb.ld + i)];
                same_kv = same_kv && kd == vd;
            }
        check(!same_kv && dq.biggest > 1e-3 && dk.biggest > 1e-3 && dv.biggest > 1e-3 && qb.out == 30 &&
                  kb.out == 10 && vb.out == 10,
              "q, k and v wrote three different results into three different buffers (k and v same width)");
    }

    // ------------------------------------------------------------------ (c) the flag moves on the device
    std::printf("the flag\n");
    k::lora_set_enabled(false);
    check(!k::lora_enabled(), "off again");
    for (auto& kv : ybuf) {   // the very same bytes as the on-run started from
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    apply(3, io3, outs3);
    download();
    {
        bool same = true;
        for (const auto& kv : ybuf)
            same = same && std::memcmp(kv.second.host.data(), before.at(kv.first).data(),
                                       kv.second.host.size() * sizeof(float)) == 0;
        check(same, "after the switch, off still writes not one byte");
    }
    k::lora_set_enabled(true);
    check(k::lora_enabled(), "on again");
    for (auto& kv : ybuf) {
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    apply(3, io3, outs3);
    download();
    {
        bool same = true;
        for (const auto& kv : ybuf) {
            if (kv.first.rfind("blk.3.", 0) != 0) continue;   // this call only touches layer 3
            same = same && std::memcmp(kv.second.host.data(), on_result.at(kv.first).data(),
                                       kv.second.host.size() * sizeof(float)) == 0;
        }
        check(same, "on again: the same call reproduces the first on-run BIT FOR BIT (fixed accumulation order)");
    }

    // ------------------------------------------------------------------ (d) one call, every entry of the layer
    std::printf("one call per layer\n");
    for (auto& kv : ybuf) {
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    check(apply(3, io3, outs3), "layer 3's call returns true");
    download();
    {
        bool layer3_ok = true, layer0_untouched = true;
        for (const auto& kv : ybuf) {
            if (kv.first.rfind("blk.3.", 0) == 0)
                layer3_ok = layer3_ok && std::memcmp(kv.second.host.data(), on_result.at(kv.first).data(),
                                                     kv.second.host.size() * sizeof(float)) == 0;
            else
                layer0_untouched = layer0_untouched && std::memcmp(kv.second.host.data(), before.at(kv.first).data(),
                                                                   kv.second.host.size() * sizeof(float)) == 0;
        }
        check(layer3_ok, "one call applied ALL FOUR of layer 3's entries, in one launch");
        check(layer0_untouched, "and it wrote nothing in layer 0's tensors");
    }
    for (auto& kv : ybuf) {
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    check(apply(0, io0, outs0), "layer 0's call returns true");
    download();
    {
        bool layer0_ok = true, layer3_untouched = true;
        for (const auto& kv : ybuf) {
            if (kv.first.rfind("blk.0.", 0) == 0)
                layer0_ok = layer0_ok && std::memcmp(kv.second.host.data(), on_result.at(kv.first).data(),
                                                     kv.second.host.size() * sizeof(float)) == 0;
            else
                layer3_untouched = layer3_untouched && std::memcmp(kv.second.host.data(), before.at(kv.first).data(),
                                                                   kv.second.host.size() * sizeof(float)) == 0;
        }
        check(layer0_ok, "one call applied BOTH of layer 0's entries (input and SSM output)");
        check(layer3_untouched, "and it wrote nothing in layer 3's tensors");
    }

    // ------------------------------------------------------------------ (e) a missing activation or output
    std::printf("a missing activation or output\n");
    auto missing_x = io3;
    missing_x[k::kLoraActAttn].x = nullptr;   // attn_output's activation was not passed
    for (auto& kv : ybuf) {
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    check(!apply(3, missing_x, outs3), "an entry whose activation was not passed returns false");
    download();
    {
        bool same = true;
        for (const auto& kv : ybuf)
            same = same && std::memcmp(kv.second.host.data(), before.at(kv.first).data(),
                                       kv.second.host.size() * sizeof(float)) == 0;
        check(same, "and it wrote nothing - not even the entries whose activation WAS passed");
    }
    auto missing_y = outs3;
    missing_y[(size_t) (i_o >= 0 ? i_o : 0)].y = nullptr;   // attn_output's destination was not passed
    for (auto& kv : ybuf) {
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    check(!apply(3, io3, missing_y), "an entry whose output was not passed returns false");
    download();
    {
        bool same = true;
        for (const auto& kv : ybuf)
            same = same && std::memcmp(kv.second.host.data(), before.at(kv.first).data(),
                                       kv.second.host.size() * sizeof(float)) == 0;
        check(same, "and it wrote nothing either");
    }
    check(!k::lora_apply_layer(3, T, io3.data(), nullptr, nullptr) &&
              !k::lora_apply_layer(3, T, nullptr, outs3.data(), nullptr),
          "a null output array and a null activation array are both refused");

    // ------------------------------------------------------------------ the upload's refusal
    std::printf("refusals\n");
    {
        // the shared expert: the loader accepts it, the device table does not serve it yet (step 2.4)
        TempDir d2;
        const std::vector<Spec> shexp = {{"blk.0.ffn_gate_shexp.weight", 2, 2.0, 8, 3}};
        const Adapter ad = make_adapter(shexp, rng);
        write_adapter(d2.path, ad.rows, ad.blob);
        core::LoraAdapter a2;
        const bool loads = core::lora_load(d2.str(), g, a2, err);
        std::string up_err;
        const bool uploaded = k::lora_upload(a2, d2.str(), up_err);
        check(loads && !uploaded && contains(up_err, "ffn_gate_shexp") && contains(up_err, "step 2.4"),
              "a shared-expert target is refused by name: " + up_err);
        check(k::lora().loaded(), "a refused upload leaves the loaded adapter in place");
    }
    // and the adapter that survived the refusal is still the one on the device, bit for bit
    for (auto& kv : ybuf) {
        kv.second.host = before.at(kv.first);
        up(kv.second.dev, kv.second.host);
    }
    check(apply(3, io3, outs3), "the original adapter still applies after the refusal");
    download();
    {
        bool same = true;
        for (const auto& kv : ybuf) {
            if (kv.first.rfind("blk.3.", 0) != 0) continue;
            same = same && std::memcmp(kv.second.host.data(), on_result.at(kv.first).data(),
                                       kv.second.host.size() * sizeof(float)) == 0;
        }
        check(same, "and its arithmetic is unchanged");
    }

    // ------------------------------------------------------------------ replacing the adapter
    std::printf("replacing the adapter\n");
    {
        TempDir d4;
        const std::vector<Spec> one = {{"blk.1.attn_qkv.weight", 2, 2.0, 8, 300}};
        const Adapter ad = make_adapter(one, rng);
        write_adapter(d4.path, ad.rows, ad.blob);
        core::LoraAdapter a4;
        std::string e4;
        const bool replaced = core::lora_load(d4.str(), g, a4, e4) && k::lora_upload(a4, d4.str(), e4);
        check(replaced, "a second adapter replaces the first: " + e4);
        check(k::lora().n_entries == 1 && k::lora().covers(1) && !k::lora().covers(0),
              "the table is the new adapter's (layer 1 only)");
        check(k::lora_find(1, "attn_qkv.weight") == 0 && k::lora_find(0, "attn_qkv.weight") == -1,
              "lora_find follows the replacement");
        check(k::lora_enabled(), "and the replacement starts ON");
        // one application against the reference, so the reallocated table is a working one
        Buf& b = ybuf.at("blk.0.attn_qkv.weight");   // same shape as layer 1's qkv (8 -> 300)
        for (int64_t tok = 0; tok < T; ++tok) {
            for (int64_t i = 0; i < b.out; ++i) b.host[(size_t) (tok * b.ld + i)] = ud(rng);
            for (int64_t i = b.out; i < b.ld; ++i) b.host[(size_t) (tok * b.ld + i)] = 1234.5f;
        }
        up(b.dev, b.host);
        const std::vector<float> pre4 = b.host;
        const auto io1 = io_for(true, false, false);
        const std::vector<k::LoraOut> outs1 = {k::LoraOut{b.dev, b.ld}};
        check(apply(1, io1, outs1), "layer 1 launches with the replacement");
        download();
        const core::LoraTarget* t = a4.find("blk.1.attn_qkv.weight");
        double worst = 0.0;
        for (int64_t tok = 0; tok < T; ++tok) {
            std::vector<float> xr((size_t) t->in);
            std::copy(hxin.begin() + tok * LD_IN, hxin.begin() + tok * LD_IN + t->in, xr.begin());
            std::vector<float> want(pre4.begin() + tok * b.ld, pre4.begin() + tok * b.ld + b.out);
            core::lora_delta_f32(d4.str(), *t, xr.data(), want.data(), e4);
            for (int64_t i = 0; i < b.out; ++i)
                worst = std::fmax(worst, std::fabs((double) b.host[(size_t) (tok * b.ld + i)] - want[(size_t) i]));
        }
        std::printf("    blk.1.attn_qkv.weight (replacement) max |err| %.3g\n", worst);
        check(worst < kTol, "the replacement's delta matches the reference");
    }

    // ------------------------------------------------------------------ the VRAM arithmetic (pure, no GPU)
    // The upload's query cannot be exercised on this machine - the card has several GB free while the live engine
    // grows its expert cache - so what is pinned is the arithmetic the decision and the message share, with the
    // spike's own numbers: 1061 MiB free against the 1024 MiB reserve, i.e. 37 MiB of headroom.
    std::printf("the VRAM arithmetic\n");
    {
        constexpr uint64_t MB = 1ull << 20;
        const int64_t R = k::kLoraVramReserveMib;
#ifdef _WIN32
        check(R == 1024, "the reserve is the engine's Windows value (kVramSafeMib)");
#else
        check(R == 512, "the reserve is the engine's non-Windows value (kVramSafeMib)");
#endif
        const auto at = [](uint64_t table, uint64_t free_bytes, int64_t reserve) {
            return k::lora_vram_fit(table, free_bytes, reserve);
        };
        const uint64_t free_spike = 1061 * MB;
        check(at(11'400'000ull, free_spike, 1024).fits, "rank-16 attention-only (11.4 MB) fits 1061 MiB less 1024");
        check(at(22'800'000ull, free_spike, 1024).fits, "rank-32 (22.8 MB) fits");
        check(!at(45'600'000ull, free_spike, 1024).fits, "rank-64 (45.6 MB) is refused");
        check(at(37 * MB, free_spike, 1024).fits && !at(37 * MB + 1, free_spike, 1024).fits,
              "the boundary is the 37 MiB of headroom exactly, and one byte over is refused");
        check(at(1, free_spike + MB - 1, 1024).free_mib == 1061, "the free figure rounds down");
        check(at(MB + 1, free_spike, 1024).table_mib == 2, "the table figure rounds up");
        {
            const k::LoraVramFit f = at(45'600'000ull, free_spike, 1024);
            check(f.table_mib == 44 && f.free_mib == 1061 && f.need_mib() == 1068 && !f.fits,
                  "the refusal's numbers: 44 MiB tables + 1024 MiB reserve against 1061 MiB free");
        }
        check(!at(1, 0, 0).fits && at(0, 0, 0).fits, "no room refuses even a byte; nothing to place always fits");
        check(at(1, MB, -5).fits, "a negative reserve is clamped to zero rather than subtracting room");
    }

    std::printf(g_fail ? "lora_parity: %d FAILED\n" : "lora_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
