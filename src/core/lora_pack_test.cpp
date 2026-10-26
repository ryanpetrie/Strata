// src/core/lora_pack_test.cpp - the LoRA adapter's file format and its checks, on files the test writes itself
// (no GPU, no model, no engine): `conversation_cache_test`'s shape, not `cuda/`'s.
//
//   * the byte layout: one small adapter (rank 2, A [2, 3], B [3, 2]) whose TRANSFORMED bytes are pinned as fp16
//     bits, so an orientation change fails here instead of producing a plausible wrong model.  The same six A bits
//     and six B bits are pinned in tools/test_lora_pack.py, which is what keeps the converter and this reader on
//     one format;
//   * the arithmetic: `lora_delta_f32` against a hand-computed `scale * B @ (A @ x)`;
//   * the refusals of the plan's 3.5, each with the offending name in the message: a wrong shape, a tensor the
//     model does not have, a QSA projection on a GDN layer, a routed-expert tensor (with both of its sizes), a
//     rank above the engine's maximum, a missing half of a pair, a stale blob, overlapping rows, and a pack's
//     index given as an adapter's;
//   * the digest: stable across reads, changed by one byte of weight or one digit of alpha.
#include "strata/core/lora_pack.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// A model small enough to write adapters for by hand.  The layout table reads every dimension from here, so
/// `blk.3.attn_k.weight` is {in 3, out 3} and `blk.0.attn_qkv.weight` is {in 3, out 4} - the same code path as the
/// real geometry, with numbers a person can check.  The qkv tensor is the ASYMMETRIC case: a symmetric fixture
/// cannot tell [in, out] from [out, in].
ModelGeometry tiny_geometry() {
    ModelGeometry g;
    g.n_embd = 3;
    g.n_layers = 4;
    g.qsa_interval = 4;
    g.n_head = 1;
    g.head_dim = 3;
    g.n_head_kv = 1;
    g.n_vocab = 16;
    g.ssm_state_size = 2;
    g.ssm_k_heads = 1;
    g.ssm_v_heads = 2;
    g.ssm_d_conv = 2;
    g.ssm_conv_channels = 4;
    g.ssm_value_dim = 4;
    g.n_expert = 2;
    g.n_ff = 3;
    g.n_ff_shexp = 3;
    return g;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("strata-lora-test-" +
                                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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

/// Writes the index and the blob exactly as `tools/lora_pack.py` does: header lines, then one row per tensor, then
/// the fp16 bytes in row order.
bool write_adapter(const fs::path& dir, const std::vector<std::string>& rows, const std::vector<uint16_t>& blob,
                   bool header = true) {
    std::ofstream idx(dir / "index.txt", std::ios::binary);
    if (header) idx << "# strata lora index v1\n# name rank alpha ne0 ne1 offset bytes\n";
    for (const std::string& r : rows) idx << r << "\n";
    idx.close();
    std::ofstream bin(dir / "lora.bin", std::ios::binary);
    bin.write((const char*) blob.data(), (std::streamsize) (blob.size() * 2));
    return idx.good() && bin.good();
}

/// The one small adapter both this test and tools/test_lora_pack.py pin: PEFT's A is [rank 2, in 3] and B is
/// [out 3, rank 2], and the FILE holds A^T ([in][rank]) and B^T ([rank][out]).
const double kA[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
const double kB[3][2] = {{0.5, -0.5}, {1.5, 1.0}, {-2.0, 0.25}};

std::vector<uint16_t> golden_blob() {
    std::vector<uint16_t> v;
    for (int j = 0; j < 3; ++j)  // A, [in][rank]
        for (int k = 0; k < 2; ++k) v.push_back(strata::kernels::f16_from_f32((float) kA[k][j]));
    for (int k = 0; k < 2; ++k)  // B, [rank][out]
        for (int i = 0; i < 3; ++i) v.push_back(strata::kernels::f16_from_f32((float) kB[i][k]));
    return v;
}

std::vector<std::string> golden_rows() {
    return {row("blk.3.attn_k.weight.lora_a", 2, 2.0, 2, 3, 0),
            row("blk.3.attn_k.weight.lora_b", 2, 2.0, 3, 2, 12)};
}

std::string refusal(const ModelGeometry& g, const std::vector<std::string>& rows,
                    const std::vector<uint16_t>& blob, bool header = true) {
    TempDir d;
    write_adapter(d.path, rows, blob, header);
    LoraAdapter a;
    std::string err;
    if (lora_load(d.str(), g, a, err)) return std::string();  // no refusal: the caller's check fails on the empty string
    return err;
}

bool contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

}  // namespace

int main() {
    const ModelGeometry g = tiny_geometry();

    // ------------------------------------------------------------------ the byte layout and the arithmetic
    {
        TempDir d;
        check(write_adapter(d.path, golden_rows(), golden_blob()), "the fixture is written");
        LoraAdapter a;
        std::string err;
        check(lora_load(d.str(), g, a, err), "one adapter for blk.3.attn_k.weight loads: " + err);
        check(a.targets.size() == 1, "one target");
        const LoraTarget& t = a.targets[0];
        check(t.base == "blk.3.attn_k.weight" && t.out == 3 && t.in == 3 && t.rank == 2,
              "the base tensor's shape comes from the layout table, not from the row");
        check(t.alpha == 2.0 && t.scale() == 1.0, "the scaling is alpha / rank");
        check(a.blob_bytes == 24 && a.weight_bytes() == 24, "24 bytes: 2 x (2 x 3) matrices of fp16");

        // THE ORIENTATION.  Read through the format's own rule (element (i, j) at i + ne0 * j) the file must give
        // back PEFT's matrices exactly - A [2, 3] and B [3, 2] - not their transposes.
        std::vector<uint16_t> got;
        check(lora_read_f16(d.str(), t.a, got, err), "A reads back");
        bool a_ok = got.size() == 6;
        for (int k = 0; a_ok && k < 2; ++k)
            for (int j = 0; a_ok && j < 3; ++j)
                a_ok = (double) strata::kernels::f32_from_f16(got[(size_t) (k + 2 * j)]) == kA[k][j];
        check(a_ok, "A is PEFT's [rank, in] through the index's ne0 rule (the file holds A^T)");
        check(lora_read_f16(d.str(), t.b, got, err) && got.size() == 6, "B reads back");
        bool b_ok = got.size() == 6;
        for (int i = 0; b_ok && i < 3; ++i)
            for (int k = 0; b_ok && k < 2; ++k)
                b_ok = (double) strata::kernels::f32_from_f16(got[(size_t) (i + 3 * k)]) == kB[i][k];
        check(b_ok, "B is PEFT's [out, rank] through the index's ne0 rule (the file holds B^T)");

        // The six pinned bits per matrix: A^T is [1, 4, 2, 5, 3, 6] and B^T is [.5, 1.5, -2, -.5, 1, .25].
        const std::vector<uint16_t> want = golden_blob();
        check(want.size() == 12 && want[0] == 0x3C00 && want[1] == 0x4400 && want[2] == 0x4000 &&
                  want[5] == 0x4600 && want[6] == 0x3800 && want[8] == 0xC000 && want[11] == 0x3400,
              "the pinned fp16 bits: A^T then B^T, rank contiguous inside each matrix's own axis");

        // scale * B @ (A @ x) at x = [1, -2, 0.5]: A x = [-1.5, -3], B (A x) = [0.75, -5.25, 2.25].
        const float x[3] = {1.0f, -2.0f, 0.5f};
        float dy[3] = {0.0f, 0.0f, 0.0f};
        check(lora_delta_f32(d.str(), t, x, dy, err), "the host reference runs: " + err);
        check(dy[0] == 0.75f && dy[1] == -5.25f && dy[2] == 2.25f,
              "dy = scale * B @ (A @ x) = [0.75, -5.25, 2.25]");
        dy[0] = dy[1] = dy[2] = 1.0f;
        check(lora_delta_f32(d.str(), t, x, dy, err) && dy[0] == 1.75f && dy[1] == -4.25f && dy[2] == 3.25f,
              "the reference ADDS into dy (the base output is already there)");
        check(a.find("blk.3.attn_k.weight") == &a.targets[0] && a.find("blk.3.attn_q.weight") == nullptr,
              "find() names a target and refuses one the adapter does not have");

        // The converted values are fp16-exact, so this is an identity, not a tolerance: a transposed or truncated
        // file cannot pass it.
        std::vector<uint16_t> again;
        check(lora_read_f16(d.str(), t.a, again, err) && again == std::vector<uint16_t>(want.begin(), want.begin() + 6),
              "A's bytes are the fixture's first six, unchanged by the read");
    }

    // ------------------------------------------------------------------ the digest
    {
        TempDir d;
        write_adapter(d.path, golden_rows(), golden_blob());
        LoraAdapter a;
        std::string err;
        uint64_t first = 0, second = 0;
        check(lora_load(d.str(), g, a, err) && lora_digest(d.str(), a, first, err), "the digest is computed");
        lora_digest(d.str(), a, second, err);
        check(first == second, "the digest is stable across reads");
        std::vector<uint16_t> changed = golden_blob();
        changed[3] = strata::kernels::f16_from_f32(5.5f);  // one element of B
        write_adapter(d.path, golden_rows(), changed);
        lora_digest(d.str(), a, second, err);
        check(first != second, "one byte of weight changes the digest");
        write_adapter(d.path, golden_rows(), golden_blob());
        lora_digest(d.str(), a, second, err);
        auto rows = golden_rows();
        rows[0] = row("blk.3.attn_k.weight.lora_a", 2, 3.0, 2, 3, 0);  // alpha 2 -> 3, on BOTH halves
        rows[1] = row("blk.3.attn_k.weight.lora_b", 2, 3.0, 3, 2, 12);
        write_adapter(d.path, rows, golden_blob());
        LoraAdapter b;
        check(lora_load(d.str(), g, b, err), "the alpha-3 adapter still loads: " + err);
        lora_digest(d.str(), b, second, err);
        check(first != second, "one digit of alpha changes the digest");
        // The scaling is a property of the PAIR: a half written with another alpha is refused, not averaged.
        rows[1] = row("blk.3.attn_k.weight.lora_b", 2, 2.0, 3, 2, 12);
        write_adapter(d.path, rows, golden_blob());
        LoraAdapter c;
        std::string mismatch;
        check(!lora_load(d.str(), g, c, mismatch) && contains(mismatch, "disagree on rank/alpha"),
              "an A and B with different alpha are refused: " + mismatch);
    }

    // ------------------------------------------------------------------ the refusals (plan 3.5)
    {
        // The wrong shape: B is [out 3, rank 2] for this tensor; hand it [2, 3].
        auto rows = golden_rows();
        rows[1] = row("blk.3.attn_k.weight.lora_b", 2, 2.0, 2, 3, 12);
        const std::string err = refusal(g, rows, golden_blob());
        check(contains(err, "blk.3.attn_k.weight") && contains(err, "[2, 3]") && contains(err, "[3, 3]"),
              "a wrong-shaped pair is refused with the adapter's A, its B, and the model's shape: " + err);
    }
    {
        // AN ASYMMETRIC TENSOR, the case the symmetric fixture above cannot see.  The layout table's dim 0 is the
        // projection's INPUT (`blk.0.attn_qkv.weight` is {in 3, out 4}), so a correct adapter is A {rank 2, in 3}
        // and B {out 4, rank 2}.  Reading the table as [out, in] refuses exactly this pair.
        const std::vector<std::string> rows = {
            row("blk.0.attn_qkv.weight.lora_a", 2, 2.0, 2, 3, 0),
            row("blk.0.attn_qkv.weight.lora_b", 2, 2.0, 4, 2, 12)};
        const std::vector<uint16_t> blob(14, 0);          // 2 x 3 + 2 x 4 fp16 values
        TempDir d;
        check(write_adapter(d.path, rows, blob), "the asymmetric fixture is written");
        LoraAdapter a;
        std::string err;
        check(lora_load(d.str(), g, a, err), "an adapter on a non-square tensor loads: " + err);
        check(a.targets.size() == 1 && a.targets[0].in == 3 && a.targets[0].out == 4,
              "the non-square tensor's in and out are the layout table's, not swapped");

        // The mirrored pair (A {rank, out}, B {in, rank}) is the one the wrong reading accepted.
        const std::vector<std::string> swapped = {
            row("blk.0.attn_qkv.weight.lora_a", 2, 2.0, 2, 4, 0),
            row("blk.0.attn_qkv.weight.lora_b", 2, 2.0, 3, 2, 16)};
        const std::vector<uint16_t> blob2(14, 0);
        const std::string e2 = refusal(g, swapped, blob2);
        check(contains(e2, "[3, 4]"), "the refusal names the model's tensor as [in, out]: " + e2);
    }
    {
        // A tensor the model does not have.
        auto rows = golden_rows();
        rows[0] = row("blk.3.foo.weight.lora_a", 2, 2.0, 2, 3, 0);
        rows[1] = row("blk.3.foo.weight.lora_b", 2, 2.0, 3, 2, 12);
        const std::string err = refusal(g, rows, golden_blob());
        check(contains(err, "blk.3.foo.weight") && contains(err, "does not have"),
              "a tensor the model does not have is refused by name: " + err);
    }
    {
        // A QSA projection on a GDN layer - the mistake the layout table exists to catch.
        auto rows = golden_rows();
        rows[0] = row("blk.0.attn_q.weight.lora_a", 2, 2.0, 2, 6, 0);
        rows[1] = row("blk.0.attn_q.weight.lora_b", 2, 2.0, 3, 2, 12);
        const std::string err = refusal(g, rows, golden_blob());
        check(contains(err, "layer 0 is a GDN layer") && contains(err, "full-attention layers are 3"),
              "attn_q on a GDN layer is refused, naming the layers that have one: " + err);
    }
    {
        // A routed-expert tensor, refused with BOTH sizes.  The tensor is {in 2560, out 640, 512 experts}: its
        // dim 0 is the projection's INPUT, so a correct adapter is A {rank 16, in 2560} and B {out 640, rank 16}.
        // The plan's arithmetic, re-run with the right labels: 42.0 MB fused for one tensor (2.01 GB over the 48
        // layers), 52.4 MB per expert (2.52 GB per tensor, 7.55 GB for the three); the three tensors' fused
        // total is 4.54 GB, because the down projection is {in 640, out 2560} and smaller.
        const ModelGeometry real;  // Qwen3.8-Flash-Next's compiled defaults
        auto rows = golden_rows();
        rows[0] = row("blk.3.ffn_gate_exps.weight.lora_a", 16, 16.0, 16, 2560, 0);
        rows[1] = row("blk.3.ffn_gate_exps.weight.lora_b", 16, 16.0, 640, 16, 81920);
        std::vector<uint16_t> blob(16 * 2560 + 640 * 16, 0);
        const std::string err = refusal(real, rows, blob);
        check(contains(err, "routed-expert") && contains(err, "42.0 MB") && contains(err, "2.01 GB") &&
                  contains(err, "52.4 MB") && contains(err, "2.52 GB"),
              "a routed-expert tensor is refused with its two sizes (per-expert x3 = the plan's 7.55 GB): " + err);
    }
    {
        // A rank above the engine's maximum.
        auto rows = golden_rows();
        rows[0] = row("blk.3.attn_k.weight.lora_a", 128, 128.0, 128, 3, 0);
        rows[1] = row("blk.3.attn_k.weight.lora_b", 128, 128.0, 3, 128, 768);
        std::vector<uint16_t> blob(128 * 3 + 3 * 128, 0);
        const std::string err = refusal(g, rows, blob);
        check(contains(err, "rank 128") && contains(err, "maximum (64)"),
              "a rank above the engine's maximum is refused: " + err);
    }
    {
        // Half a pair.
        auto rows = golden_rows();
        rows.pop_back();
        const std::string err = refusal(g, rows, golden_blob());
        check(contains(err, "blk.3.attn_k.weight.lora_b is missing"),
              "an A without its B is refused, naming the half: " + err);
    }
    {
        // A stale blob: one more byte than the rows describe.
        auto blob = golden_blob();
        blob.push_back(0);
        const std::string err = refusal(g, golden_rows(), blob);
        check(contains(err, "26 bytes") && contains(err, "describes 24"),
              "a blob longer than the index describes is refused: " + err);
    }
    {
        // Two rows on the same bytes.
        auto rows = golden_rows();
        rows[1] = row("blk.3.attn_k.weight.lora_b", 2, 2.0, 3, 2, 0);
        const std::string err = refusal(g, rows, golden_blob());
        check(contains(err, "overlap"), "overlapping rows in the blob are refused: " + err);
    }
    {
        // A file with no version header: a row before it, or a pack's own header.
        const std::string err = refusal(g, golden_rows(), golden_blob(), /*header=*/false);
        check(contains(err, "strata lora index v1") && contains(err, "before its first row"),
              "a file whose first line is a row is refused, not parsed: " + err);
        TempDir d;
        write_adapter(d.path, golden_rows(), golden_blob(), /*header=*/false);
        {
            std::ofstream idx(d.path / "index.txt", std::ios::binary);
            idx << "# align 256 pool 1234 tensors 9\n";
            for (const std::string& r : golden_rows()) idx << r << "\n";
        }
        LoraAdapter a;
        std::string pack_err;
        check(!lora_load(d.str(), g, a, pack_err) && contains(pack_err, "not a LoRA index"),
              "a pack's index given as an adapter's is refused by its header: " + pack_err);
    }
    {
        // A 1-D tensor: a norm vector has no projection to adapt.
        auto rows = golden_rows();
        rows[0] = row("blk.3.attn_q_norm.weight.lora_a", 2, 2.0, 2, 3, 0);
        rows[1] = row("blk.3.attn_q_norm.weight.lora_b", 2, 2.0, 3, 2, 12);
        const std::string err = refusal(g, rows, golden_blob());
        check(contains(err, "1-dimensional") && contains(err, "projection"),
              "a 1-D tensor is refused as not a projection: " + err);
    }

    std::printf("\n%s\n", g_fail == 0 ? "all checks passed" : "FAILED");
    if (g_fail != 0) std::printf("%d check(s) failed\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
