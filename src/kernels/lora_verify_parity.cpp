// src/kernels/lora_verify_parity.cpp - the LoRA hook in the VERIFY window (`src/core/verify.cpp`): the sequence of
// `lora_apply_tensor` calls one verify layer makes, against the host reference, with the three activations and the
// seven destinations the window actually has.
//
// WHAT THIS CAN AND CANNOT SEE.  There is no model on this machine, so nothing here runs verify.cpp: the branch
// switches (STRATA_DF_BRANCH, STRATA_SIDE_MIXER, STRATA_DEC_BATCH, STRATA_DF_QB_Q4) select code in that file and are
// not reachable from a kernel test.  What IS pinned here is everything the hook itself decides:
//
//   (a) flag off: every destination buffer is BIT-IDENTICAL to its pre-call bytes, for the verify window's row
//       counts (T = 1, 6, 9 - one, less than and more than the kernel's 8-token tile) and its CONTIGUOUS strides
//       (`ld == width`, which is how verify's buffers are laid out; `lora_parity` covers padded rows).
//   (b) flag on: each targeted tensor's rows move by exactly `core::lora_delta_f32` of THAT tensor's activation,
//       and no other tensor moves.  The three activations differ at every element, so an entry reading the wrong one
//       - the one mistake `act` exists to prevent - cannot pass.
//   (c) the order the call sites depend on: a delta written into a destination the base path writes again after it
//       is LOST.  That is why `attn_qkv`'s delta sits before the conv and `attn_q`'s before the q/gate split rather
//       than at the end of the layer, and this is the executable form of that rule.
//   (d) the coverage check (`lora_layer_begin` / `lora_layer_end`): an entry of a targeted layer no site applied
//       makes the layer's capture FAIL, naming the tensor.  Without it a forgotten site is a projection answering
//       from the base weights while the rest of the model is adapted - the one silent failure of this step.
//   (e) `lora_apply_tensor`'s call-site errors throw, and a name the layer does not target is a quiet no-op.
//
// The adapter is a real directory read by `lora_load`, so the entry order, the `act` mapping and the shapes all come
// from the format the loader validates - a call site cannot disagree with them silently.
#include "strata/core/lora_pack.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/lora.hpp"

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
#include <initializer_list>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace core = strata::core;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;

/// The same tolerance `lora_parity` justifies: the reference sums in double, the kernel in fp32 with a fixed order,
/// and the widest entry here (rank 16 on 128 inputs) stays far below it.
constexpr double kTol = 1e-4;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

void check(bool ok, const std::string& what) {
    std::printf("  %-104s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

/// The geometry `src/core/lora_pack_test.cpp` and `lora_parity` use, so the shapes here are the layout table's:
/// layer 0 is a GDN layer (attn_qkv 600 wide, attn_gate 100, ssm_out 100 -> 128) and layer 3 a QSA one (attn_q 30,
/// attn_k and attn_v 10, attn_output 15 -> 128).  Every width differs from every other except k/v, which are equal
/// on purpose: a call site that handed one projection's delta another's buffer would still have to reproduce each
/// tensor's own values.
core::ModelGeometry tiny_geometry() {
    core::ModelGeometry g;
    g.n_embd = 128;
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
    g.ssm_conv_channels = 600;
    g.ssm_value_dim = 100;
    g.n_expert = 2;
    g.n_ff = 3;
    g.n_ff_shexp = 3;
    return g;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-lora-verify-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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

struct Spec {
    const char* base;
    int64_t rank;
    double alpha;
    int64_t in, out;   // the model's GGUF shape: the projection's input, then its output
};

/// The seven tensors the verify window can serve, with the widths verify's buffers have.  `attn_gate` is here and
/// `lora_parity` does not target it: its output is `z_`, which the recurrence reads, and it is one of the five
/// projections whose destination is consumed inside the layer.
const std::vector<Spec> kSpec = {
    {"blk.0.attn_qkv.weight", 16, 32.0, 128, 600},
    {"blk.0.attn_gate.weight", 4, 8.0, 128, 100},      // kLoraActInput, output = z_ (ssm_value_dim)
    {"blk.0.ssm_out.weight", 3, 6.0, 100, 128},        // kLoraActSsm
    {"blk.3.attn_q.weight", 4, 8.0, 128, 30},          // kLoraActInput, output = q|gate (2 * n_head * head_dim)
    {"blk.3.attn_k.weight", 2, 4.0, 128, 10},          // kLoraActInput, same width as v
    {"blk.3.attn_v.weight", 4, 8.0, 128, 10},          // kLoraActInput, same width as k
    {"blk.3.attn_output.weight", 6, 12.0, 15, 128},    // kLoraActAttn
};

bool write_adapter(const fs::path& dir, const std::vector<std::string>& rows, const std::vector<uint16_t>& blob) {
    std::ofstream idx(dir / "index.txt", std::ios::binary);
    idx << "# strata lora index v1\n# name rank alpha ne0 ne1 offset bytes\n";
    for (const std::string& r : rows) idx << r << "\n";
    idx.close();
    std::ofstream bin(dir / "lora.bin", std::ios::binary);
    bin.write((const char*) blob.data(), (std::streamsize) (blob.size() * 2));
    return idx.good() && bin.good();
}

/// A destination and its pre-call bytes: `out` wide, T rows, contiguous (`ld == out`), as verify's buffers are.
struct Buf {
    int64_t out = 0;
    std::vector<float> host;   // what the base path left there
    float* dev = nullptr;
    int64_t ld() const { return out; }
};

}  // namespace

int main() {
    const core::ModelGeometry g = tiny_geometry();
    const std::vector<int64_t> rows_to_run = {1, 6, 9};   // one row, under and over the kernel's 8-token tile
    std::mt19937 rng(20261023);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::string err;

    // ------------------------------------------------------------------ the adapter, through the real loader
    std::printf("fixture\n");
    TempDir dir;
    std::vector<std::string> rows;
    std::vector<uint16_t> blob;
    {
        uint64_t off = 0;
        for (const Spec& s : kSpec) {
            std::vector<uint16_t> ab, bb;
            for (int64_t j = 0; j < s.in; ++j)
                for (int64_t k = 0; k < s.rank; ++k) ab.push_back(k::f16_from_f32(ud(rng)));
            for (int64_t k = 0; k < s.rank; ++k)
                for (int64_t i = 0; i < s.out; ++i) bb.push_back(k::f16_from_f32(ud(rng)));
            rows.push_back(row(std::string(s.base) + ".lora_a", s.rank, s.alpha, s.rank, s.in, off));
            off += (uint64_t) ab.size() * 2;
            rows.push_back(row(std::string(s.base) + ".lora_b", s.rank, s.alpha, s.out, s.rank, off));
            off += (uint64_t) bb.size() * 2;
            blob.insert(blob.end(), ab.begin(), ab.end());
            blob.insert(blob.end(), bb.begin(), bb.end());
        }
        check(write_adapter(dir.path, rows, blob), "the index and the fp16 blob are written");
    }
    core::LoraAdapter adapter;
    if (!core::lora_load(dir.str(), g, adapter, err)) {
        std::fprintf(stderr, "lora_load: %s\n", err.c_str());
        return 2;
    }
    check(adapter.targets.size() == kSpec.size(), "all seven verified tensors load against the fixture geometry");
    if (!k::lora_upload(adapter, dir.str(), err)) {
        std::fprintf(stderr, "lora_upload: %s\n", err.c_str());
        return 2;
    }
    check(k::lora().covers(0) && k::lora().covers(3) && !k::lora().covers(1),
          "covers(): the two layers the adapter touches, and no other");
    k::lora_set_enabled(false);   // the engine's state after --lora: resident and off

    // The verify window's three activation buffers, at their real widths: `xm` (mixed_, N wide), `attn32_`
    // (n_head * head_dim) and `y_` (ssm_value_dim).  Every element differs between them, so an entry that read the
    // wrong one cannot match the reference.
    const int64_t N = g.n_embd, W_attn = g.n_head * g.head_dim, W_ssm = g.ssm_value_dim;
    check(W_attn == 15 && W_ssm == 100 && N == 128, "the three verify activations have their own widths");
    std::vector<float> hx_in, hx_attn, hx_ssm, hx_alt;   // hx_alt: a second input, for the order check below
    auto fill_activation = [&](std::vector<float>& v, int64_t rows, float bias) {
        v.resize((size_t) (rows * N), 0.0f);
        for (int64_t t = 0; t < rows; ++t)
            for (int64_t i = 0; i < N; ++i) v[(size_t) (t * N + i)] = ud(rng) + bias;
    };
    fill_activation(hx_in, 9, 0.0f);
    fill_activation(hx_alt, 9, 3.0f);
    hx_attn.resize((size_t) (9 * W_attn));
    for (float& v : hx_attn) v = ud(rng) + 7.0f;
    hx_ssm.resize((size_t) (9 * W_ssm));
    for (float& v : hx_ssm) v = ud(rng) - 5.0f;
    float* d_in = dalloc<float>(hx_in.size());
    float* d_alt = dalloc<float>(hx_alt.size());
    float* d_attn = dalloc<float>(hx_attn.size());
    float* d_ssm = dalloc<float>(hx_ssm.size());
    up(d_in, hx_in);
    up(d_alt, hx_alt);
    up(d_attn, hx_attn);
    up(d_ssm, hx_ssm);

    // The destinations, one per targeted tensor, at verify's widths (`qkv` C wide, `z_` ZV, `qfull_` 2 * n_head *
    // head_dim, k/v n_head_kv * head_dim, `bo_` N).  Row t's output starts at y + t * width.
    std::map<std::string, Buf> buf;
    for (const Spec& s : kSpec) buf.emplace(s.base, Buf{s.out, {}, nullptr});
    for (auto& kv : buf) kv.second.dev = dalloc<float>((size_t) (9 * kv.second.out));
    // A destination no entry names: the test hands it to no call, so it can only move by a write that escaped the
    // buffer it was aimed at (a wrong `outs` index or a stride wider than the row).  Checked in (b).
    std::vector<float> h_other((size_t) (9 * N), 4321.0f);
    float* d_other = dalloc<float>(h_other.size());
    up(d_other, h_other);

    auto io_for = [&](const float* xin) {
        std::array<k::LoraIo, k::kLoraActCount> io{};
        io[k::kLoraActInput] = k::LoraIo{xin, N};
        io[k::kLoraActAttn] = k::LoraIo{d_attn, W_attn};
        io[k::kLoraActSsm] = k::LoraIo{d_ssm, W_ssm};
        return io;
    };

    // verify's own call sequence for `layer`, one site per projection, naming each tensor the way the model does.
    auto apply_layer = [&](int64_t layer, int64_t T, const std::array<k::LoraIo, k::kLoraActCount>& io) {
        auto one = [&](const char* base) {
            k::lora_apply_tensor(layer, T, io.data(), base, {buf.at(base).dev, buf.at(base).out}, nullptr);
        };
        if (layer == 0) {
            one("blk.0.attn_qkv.weight");
            one("blk.0.attn_gate.weight");
            one("blk.0.ssm_out.weight");
        } else {
            one("blk.3.attn_q.weight");
            one("blk.3.attn_k.weight");
            one("blk.3.attn_v.weight");
            one("blk.3.attn_output.weight");
        }
        ck(cudaDeviceSynchronize(), "apply");
    };

    auto fill_destinations = [&](int64_t T) {
        for (auto& kv : buf) {
            Buf& b = kv.second;
            b.host.assign((size_t) (T * b.out), 0.0f);
            for (float& v : b.host) v = ud(rng) * 4.0f;   // the "base projection's" values
            up(b.dev, b.host);
        }
    };
    auto download = [&](int64_t T) {
        for (auto& kv : buf) kv.second.host = down(kv.second.dev, (size_t) (T * kv.second.out));
    };
    auto snapshot = [&](int64_t T) {
        std::map<std::string, std::vector<float>> m;
        for (const auto& kv : buf) m.emplace(kv.first, kv.second.host);
        (void) T;
        return m;
    };
    auto restore = [&](const std::map<std::string, std::vector<float>>& m) {
        for (auto& kv : buf) {
            kv.second.host = m.at(kv.first);
            up(kv.second.dev, kv.second.host);
        }
    };

    for (const int64_t T : rows_to_run) {
        std::printf("T = %lld rows\n", (long long) T);
        const auto io = io_for(d_in);

        // ---------------------------------------------------------------- (a) flag off
        k::lora_set_enabled(false);
        fill_destinations(T);
        const auto base = snapshot(T);
        k::lora_layer_begin(0);
        apply_layer(0, T, io);
        k::lora_layer_end(0);
        k::lora_layer_begin(3);
        apply_layer(3, T, io);
        k::lora_layer_end(3);
        download(T);
        {
            bool same = true;
            for (const auto& kv : buf)
                same = same && std::memcmp(kv.second.host.data(), base.at(kv.first).data(),
                                           kv.second.host.size() * sizeof(float)) == 0;
            check(same, "off: every destination is BIT-IDENTICAL to its pre-call bytes");
        }

        // ---------------------------------------------------------------- (b) the delta, against the reference
        k::lora_set_enabled(true);
        restore(base);
        k::lora_layer_begin(0);
        apply_layer(0, T, io);
        k::lora_layer_end(0);
        k::lora_layer_begin(3);
        apply_layer(3, T, io);
        k::lora_layer_end(3);
        download(T);
        std::map<std::string, bool> moved;
        bool all_ok = true, all_moved = true;
        for (const Spec& s : kSpec) {
            const core::LoraTarget* t = adapter.find(s.base);
            const Buf& b = buf.at(s.base);
            // which buffer this entry reads, as the FORMAT says it does (not as the test hopes): the reference is
            // fed the same one the kernel reads.
            const bool attn = contains(s.base, "attn_output");
            const bool ssm = contains(s.base, "ssm_out");
            const std::vector<float>& x = attn ? hx_attn : (ssm ? hx_ssm : hx_in);
            const int64_t xw = attn ? W_attn : (ssm ? W_ssm : N);
            double worst = 0.0, biggest = 0.0;
            for (int64_t tok = 0; tok < T; ++tok) {
                std::vector<float> xr((size_t) t->in);
                std::copy(x.begin() + tok * xw, x.begin() + tok * xw + t->in, xr.begin());
                std::vector<float> want(base.at(s.base).begin() + tok * b.out,
                                        base.at(s.base).begin() + tok * b.out + b.out);
                if (!core::lora_delta_f32(dir.str(), *t, xr.data(), want.data(), err)) {
                    std::fprintf(stderr, "lora_delta_f32: %s\n", err.c_str());
                    return 2;
                }
                for (int64_t i = 0; i < b.out; ++i) {
                    const float got = b.host[(size_t) (tok * b.out + i)];
                    const float had = base.at(s.base)[(size_t) (tok * b.out + i)];
                    worst = std::fmax(worst, std::fabs((double) got - want[(size_t) i]));
                    biggest = std::fmax(biggest, std::fabs((double) want[(size_t) i] - had));
                }
            }
            std::printf("    %-30s max |err| %.3g (largest delta %.3g)\n", s.base, worst, biggest);
            all_ok = all_ok && worst < kTol;
            all_moved = all_moved && biggest > 1e-3;
            moved[s.base] = biggest > 1e-3;
        }
        check(all_ok, "on: every targeted tensor matches core::lora_delta_f32 on ITS activation");
        check(all_moved, "on: every targeted tensor moved (no entry applied nothing)");
        check(down(d_other, h_other.size()) == h_other,
              "on: a buffer no entry names is untouched - no delta escaped the tensor it was aimed at");
        // k and v share a width: their deltas must differ, or one entry overwrote the other's tensor
        {
            bool same_kv = true;
            for (int64_t tok = 0; tok < T; ++tok)
                for (int64_t i = 0; i < 10; ++i) {
                    const auto& kb = buf.at("blk.3.attn_k.weight");
                    const auto& vb = buf.at("blk.3.attn_v.weight");
                    same_kv = same_kv && kb.host[(size_t) (tok * 10 + i)] - base.at("blk.3.attn_k.weight")[(size_t) (tok * 10 + i)] ==
                                              vb.host[(size_t) (tok * 10 + i)] - base.at("blk.3.attn_v.weight")[(size_t) (tok * 10 + i)];
                }
            check(!same_kv, "k and v (same width) carry their own deltas in their own buffers");
        }
        // The activation really is the one the entry names.  The same entries run against a DIFFERENT `xm`: the
        // input-reading projections must move with it, and `attn_output` (which reads `attn32_`) must not move at all.
        // A call site that handed `attn_output` the layer input, or `attn_q` the attention output, cannot pass both.
        {
            std::vector<float> q_with_in, q_with_alt, o_with_in, o_with_alt;
            for (int run = 0; run < 2; ++run) {
                restore(base);
                const auto run_io = io_for(run == 0 ? d_in : d_alt);
                k::lora_layer_begin(3);
                apply_layer(3, T, run_io);
                k::lora_layer_end(3);
                download(T);
                (run == 0 ? q_with_in : q_with_alt) = buf.at("blk.3.attn_q.weight").host;
                (run == 0 ? o_with_in : o_with_alt) = buf.at("blk.3.attn_output.weight").host;
            }
            bool q_differs = false, o_same = true;
            for (size_t i = 0; i < q_with_in.size(); ++i) q_differs = q_differs || q_with_in[i] != q_with_alt[i];
            for (size_t i = 0; i < o_with_in.size(); ++i) o_same = o_same && o_with_in[i] == o_with_alt[i];
            check(q_differs, "an input-reading entry follows `xm`: another layer input gives another delta");
            check(o_same, "attn_output follows `attn32_`, not `xm`: the other input leaves it bit-identical");
        }

        // ---------------------------------------------------------------- (c) the order the sites depend on
        // A destination the base path writes AFTER the delta loses it.  This is not a property of the kernel; it is
        // the reason `attn_qkv`'s site sits before the conv and `attn_q`'s before the q/gate split, and it is the one
        // placement mistake that cannot be seen from this test's other checks.
        {
            restore(base);
            k::lora_layer_begin(0);
            apply_layer(0, T, io);
            k::lora_layer_end(0);
            // the conv would now read qkv; simulate a variant that writes the base values again afterwards
            Buf& qkv = buf.at("blk.0.attn_qkv.weight");
            up(qkv.dev, base.at("blk.0.attn_qkv.weight"));
            ck(cudaDeviceSynchronize(), "rewrite");
            download(T);
            bool lost = std::memcmp(qkv.host.data(), base.at("blk.0.attn_qkv.weight").data(),
                                    qkv.host.size() * sizeof(float)) == 0;
            check(lost && moved.at("blk.0.attn_qkv.weight"),
                  "a destination written after the delta keeps no delta: the sites must sit before the last write");
        }
    }

    // ------------------------------------------------------------------ (d) the coverage check
    std::printf("the coverage check\n");
    const int64_t T = 6;
    const auto io = io_for(d_in);
    fill_destinations(T);
    auto expect_end_throw = [&](const char* what, const std::vector<std::string>& fragments, auto&& call) {
        std::string why;
        try {
            call();
        } catch (const std::exception& e) {
            why = e.what();
        }
        bool ok = !why.empty();
        for (const std::string& f : fragments) ok = ok && contains(why, f);
        check(ok, std::string(what) + ": " + (why.empty() ? "NO THROW" : why));
    };
    auto expect_silent = [&](const char* what, auto&& call) {
        std::string why;
        try {
            call();
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(why.empty(), std::string(what) + (why.empty() ? ": silent" : ": " + why));
    };
    {
        // five of layer 3's six... one of its four projections skipped: the end of the layer must say which
        k::lora_layer_begin(3);
        k::lora_apply_tensor(3, T, io.data(), "attn_q.weight", {buf.at("blk.3.attn_q.weight").dev, 30}, nullptr);
        k::lora_apply_tensor(3, T, io.data(), "attn_k.weight", {buf.at("blk.3.attn_k.weight").dev, 10}, nullptr);
        k::lora_apply_tensor(3, T, io.data(), "attn_v.weight", {buf.at("blk.3.attn_v.weight").dev, 10}, nullptr);
        expect_end_throw("a layer with a projection no site applied", {"blk.3.attn_output.weight", "never applied"},
                         [&] { k::lora_layer_end(3); });
    }
    {
        // the whole layer: silent
        k::lora_layer_begin(0);
        k::lora_apply_tensor(0, T, io.data(), "attn_qkv.weight", {buf.at("blk.0.attn_qkv.weight").dev, 600}, nullptr);
        k::lora_apply_tensor(0, T, io.data(), "attn_gate.weight", {buf.at("blk.0.attn_gate.weight").dev, 100}, nullptr);
        k::lora_apply_tensor(0, T, io.data(), "ssm_out.weight", {buf.at("blk.0.ssm_out.weight").dev, 128}, nullptr);
        expect_silent("a layer whose every entry was applied", [&] { k::lora_layer_end(0); });
    }
    {
        // a layer the adapter does not touch: nothing to begin, nothing to miss
        k::lora_layer_begin(1);
        k::lora_apply_tensor(1, T, io.data(), "attn_qkv.weight", {buf.at("blk.0.attn_qkv.weight").dev, 600}, nullptr);
        expect_silent("a layer the adapter does not target", [&] { k::lora_layer_end(1); });
    }
    {
        // a name the layer does not target is a quiet no-op (a GDN tensor asked of a QSA layer, and the reverse), and
        // it must not mark anything as applied
        std::string why;
        try {
            k::lora_layer_begin(3);
            k::lora_apply_tensor(3, T, io.data(), "attn_qkv.weight", {buf.at("blk.0.attn_qkv.weight").dev, 600},
                                 nullptr);   // layer 3 has no attn_qkv
            k::lora_apply_tensor(3, T, io.data(), "ssm_out.weight", {buf.at("blk.0.ssm_out.weight").dev, 128}, nullptr);
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(why.empty(), "a tensor this layer does not have is not an error: nothing to apply");
        expect_end_throw("... and it applied nothing", {"blk.3.attn_q.weight", "never applied"},
                         [&] { k::lora_layer_end(3); });
    }

    // ------------------------------------------------------------------ (e) call-site errors
    std::printf("call-site errors throw\n");
    auto expect_throw = [&](const char* what, const std::vector<std::string>& fragments, auto&& call) {
        std::string why;
        try {
            call();
        } catch (const std::exception& e) {
            why = e.what();
        }
        bool ok = !why.empty();
        for (const std::string& f : fragments) ok = ok && contains(why, f);
        check(ok, std::string(what) + ": " + (why.empty() ? "NO THROW" : why));
    };
    {
        auto missing = io;
        missing[k::kLoraActAttn].x = nullptr;
        expect_throw("an entry whose activation was not passed", {"blk.3.attn_output.weight", "activation", "null"},
                     [&] { k::lora_apply_tensor(3, T, missing.data(), "attn_output.weight",
                                                {buf.at("blk.3.attn_output.weight").dev, 128}, nullptr); });
    }
    expect_throw("a null output", {"blk.3.attn_output.weight", "output is null"},
                 [&] { k::lora_apply_tensor(3, T, io.data(), "attn_output.weight", {nullptr, 128}, nullptr); });
    expect_throw("a null activation array", {"activation array is null"},
                 [&] { k::lora_apply_tensor(3, T, nullptr, "attn_q.weight",
                                            {buf.at("blk.3.attn_q.weight").dev, 30}, nullptr); });
    {
        auto bad = io;
        bad[k::kLoraActInput].ld_x = N - 1;
        expect_throw("an activation stride smaller than the row", {"row stride", std::to_string(N - 1)},
                     [&] { k::lora_apply_tensor(3, T, bad.data(), "attn_q.weight",
                                                {buf.at("blk.3.attn_q.weight").dev, 30}, nullptr); });
    }
    expect_throw("an output stride smaller than the row", {"output's row stride", "29"},
                 [&] { k::lora_apply_tensor(3, T, io.data(), "attn_q.weight",
                                            {buf.at("blk.3.attn_q.weight").dev, 29}, nullptr); });

    std::printf(g_fail ? "lora_verify_parity: %d FAILED\n" : "lora_verify_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
