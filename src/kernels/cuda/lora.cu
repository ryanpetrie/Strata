// src/kernels/cuda/lora.cu - see include/strata/kernels/lora.hpp.
//
// WHAT THIS FILE OWNS, and why each piece is shaped the way it is:
//
//   * the tables.  `lora_upload` turns the loader's validated description (`core::LoraAdapter`) into ONE device
//     allocation holding every A table, a 256-byte boundary, then every B table.  One allocation because a
//     second cudaMalloc that fails half-way has no state worth unwinding, and because the adapter is read once
//     at start-up and never grows - the same reason the control vector keeps one host copy and one table per
//     device (`src/kernels/cuda/cvec.cu`).  It refuses an adapter that would eat the engine's VRAM reserve, with
//     both numbers (`lora_vram.cuh`), and it refuses a SECOND adapter while one is resident: the adapter is a
//     start-up decision whose kernel node lives in every graph captured after the upload, so one process keeps
//     one adapter and changing it means restarting the engine.
//   * the flag.  A device `int` that `lora_set_enabled` writes between requests, synchronizing first, exactly
//     as `cvec_set_enabled` does.  The kernel reads it on the device, so the captured graphs are the same with
//     the adapter on and off, and "off" writes not one byte.
//   * the delta.  One launch per layer, one block per (entry, token tile): a layer's projections are one node in
//     the graph, which is what keeps the launch tax at 48 per token instead of 156.
//
// THE ACCUMULATION ORDER.  The host reference (`core::lora_delta_f32`) sums in double; this kernel sums in fp32
// in an order that depends only on the shapes: j is walked by lane and reduced by a fixed 32-lane tree, k is
// walked serially.  Two runs of the same request therefore agree bit for bit, which is what makes the parity
// test's "off" and "on" runs comparable at all.
//
// THE INPUT IS SHARED, THE OUTPUT IS NOT.  A QSA layer's `attn_q`, `attn_k` and `attn_v` all read the layer's
// input, but write three different tensors of different widths (12288 / 1024 / 1024 on the UD-IQ4_XS pack), so
// `LoraIo` is the activation (one per `act`) and `LoraOut` is the destination, one per ENTRY of the layer in the
// adapter's own order (`lora_find` gives a call site that index).  Without that split the second input entry
// would write its delta over the first one's tensor: a plausible, silently wrong model, which is the failure
// mode the whole path exists to avoid.
//
// WHY THE OUTPUTS ARE KERNEL ARGUMENTS, NOT A DEVICE TABLE.  The output buffers change with every request, and a
// captured graph replays the pointers it was captured with; a device array filled by a host memcpy between
// requests would either be baked in at capture time or need an upload node per layer per token.  They are passed
// by value - a layer of this family has at most four servable tensors, so the struct is 128 bytes - exactly as
// the control vector's pointers are passed.  `kMaxEntriesPerLayer` is the cap `lora_upload` checks.
#include "strata/kernels/lora.hpp"

#include "strata/core/lora_pack.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "lora_vram.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRank = 64;          // core::kMaxLoraRank; repeated because `mid` below is sized at compile time
constexpr int kMaxEntriesPerLayer = 8;  // an attention/GDN layer has at most 4 servable tensors; checked at upload
constexpr int kTokensPerBlock = 8;    // a 128k-token prompt stays inside gridDim.y's 65,535
constexpr int kDevices = 64;          // as cvec.cu: a layer split applies the adapter on several cards
constexpr uint64_t kTableAlign = 256; // bytes between the A table and the B table
static_assert(kMaxRank == core::kMaxLoraRank, "the shared mid[] must hold every rank the loader accepts");

/// The per-call outputs of one layer, by value: `LoraOut` is two machine words, and a layer's entry count is
/// known only at run time while a kernel parameter's size is fixed, so this is the fixed-size carrier.
struct LoraOuts {
    LoraOut e[kMaxEntriesPerLayer];
};

struct DevTables {
    uint16_t* tables = nullptr;    // the whole adapter: every A, a 256-byte boundary, every B
    LoraEntry* entries = nullptr;  // flattened per layer, in `g_layer_first`'s order
    LoraLayer* layers = nullptr;   // n_layers entries; the device copy of `Lora::layers`
    int* on = nullptr;
};

DevTables g_dev[kDevices];

// `Lora::layers` points at the HOST mirror, because `Lora::covers()` is an inline host predicate that
// dereferences it (`l >= 0 && l < n_layers && layers[l].n > 0`).  A cudaMalloc'd array would make that read
// undefined.  The kernel never follows `Lora::layers`: it is launched with this device's `DevTables::entries`
// slice and the layer's entry count, so the forward path pays no host round trip and the predicate works.
Lora g_lora;
bool g_on_host = false;

struct PlanEntry {
    int layer = 0;
    const core::LoraTarget* target = nullptr;
    LoraEntry desc;           // sizes, scale, act, name; `a`/`b` are re-pointed per device
    uint64_t a_off = 0;       // in fp16 elements into the tables buffer
    uint64_t b_off = 0;
};

std::vector<PlanEntry> g_plan;                              // sorted by layer
std::vector<std::vector<LoraEntry>> g_entries_host;         // per layer, the host mirror
std::vector<LoraLayer> g_layers_host;                       // what `g_lora.layers` points at
std::vector<size_t> g_layer_first;                          // where a layer's slice starts in g_plan
std::vector<uint16_t> g_tables_host;                        // the table image every device gets
std::vector<std::unique_ptr<std::string>> g_names;          // lifetimed copies of `LoraEntry::name`

int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}

std::string mb(uint64_t bytes) {
    char b[48];
    std::snprintf(b, sizeof b, "%.1f MB", (double) bytes / 1e6);
    return b;
}

/// False with `err` when the adapter's tables plus the engine's reserve do not fit in the free VRAM of the
/// CURRENT device.  A failed query is neither evidence of room nor of the opposite, so it does not refuse: the
/// allocation that follows still reports its own failure.  When the numbers ARE visible the message names the
/// device, both numbers and the way out, and never reads like an adapter defect - the adapter may be perfect and
/// the card simply full.
bool vram_fits_here(uint64_t table_bytes, std::string& err) {
    size_t free_b = 0, total_b = 0;
    const int dev = cur_device();
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return true;
    const LoraVramFit f = lora_vram_fit(table_bytes, (uint64_t) free_b, kLoraVramReserveMib);
    if (f.fits) return true;
    err = "lora: the adapter needs " + std::to_string(f.table_mib) + " MiB (its fp16 tables) + " +
          std::to_string(f.reserve_mib) + " MiB reserve, but only " + std::to_string(f.free_mib) +
          " MiB of VRAM is free on CUDA" + std::to_string(dev) + "; free the card or lower --resident-budget-gib";
    return false;
}

void free_dev(DevTables& t) {
    if (t.tables != nullptr) cudaFree(t.tables);
    if (t.entries != nullptr) cudaFree(t.entries);
    if (t.layers != nullptr) cudaFree(t.layers);
    if (t.on != nullptr) cudaFree(t.on);
    t = DevTables{};
}

/// Builds this device's copy of the tables, the entry array, the layer array and the flag.  False with `err`
/// naming the allocation that did not fit; the device is left with nothing (no half table to apply).
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.tables != nullptr) return true;
    const uint64_t table_bytes = (uint64_t) g_tables_host.size() * sizeof(uint16_t);
    if (!vram_fits_here(table_bytes, err)) return false;
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.tables, table_bytes) != cudaSuccess) {
        err = "lora: the adapter's " + mb(table_bytes) + " of fp16 tables do not fit on this device";
        free_dev(t);
        return false;
    }
    if (cudaMemcpy(t.tables, g_tables_host.data(), table_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "lora: the adapter's fp16 tables could not be copied to this device";
        free_dev(t);
        return false;
    }

    // The device entries are a copy of the plan with this device's pointers; `name` is dropped because the
    // kernel never reads it and a host pointer in device memory is a trap for the next reader of this struct.
    std::vector<LoraEntry> entries(g_plan.size());
    for (size_t i = 0; i < g_plan.size(); ++i) {
        entries[i] = g_plan[i].desc;
        entries[i].a = t.tables + g_plan[i].a_off;
        entries[i].b = t.tables + g_plan[i].b_off;
        entries[i].name = nullptr;
    }
    if (cudaMalloc(&t.entries, entries.size() * sizeof(LoraEntry)) != cudaSuccess) {
        err = "lora: the adapter's " + std::to_string(entries.size()) + " entries do not fit on this device";
        free_dev(t);
        return false;
    }
    if (cudaMemcpy(t.entries, entries.data(), entries.size() * sizeof(LoraEntry), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        err = "lora: the adapter's entry table could not be copied to this device";
        free_dev(t);
        return false;
    }

    std::vector<LoraLayer> layers(g_layers_host.size());
    for (size_t l = 0; l < layers.size(); ++l) {
        layers[l].n = g_layers_host[l].n;
        layers[l].entries = layers[l].n > 0 ? t.entries + g_layer_first[l] : nullptr;
    }
    if (cudaMalloc(&t.layers, layers.size() * sizeof(LoraLayer)) != cudaSuccess ||
        cudaMalloc(&t.on, sizeof(int)) != cudaSuccess) {
        err = "lora: the adapter's per-layer table or its flag do not fit on this device";
        free_dev(t);
        return false;
    }
    if (cudaMemcpy(t.layers, layers.data(), layers.size() * sizeof(LoraLayer), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "lora: the adapter's per-layer table or its flag could not be copied to this device";
        free_dev(t);
        return false;
    }
    return true;
}

/// `blk.<L>.<tensor>.weight` -> the layer, and which activation the projection reads.
bool parse_target(const std::string& base, int& layer, int& act, std::string& err) {
    const std::string bw = "blk.";
    const std::string kw = ".weight";
    if (base.compare(0, bw.size(), bw) != 0) {
        err = "lora: `" + base + "` is not a `blk.<layer>.<projection>.weight` name";
        return false;
    }
    const size_t dot = base.find('.', bw.size());
    if (dot == std::string::npos) {
        err = "lora: `" + base + "` is not a `blk.<layer>.<projection>.weight` name";
        return false;
    }
    const std::string num = base.substr(bw.size(), dot - bw.size());
    if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos) {
        err = "lora: `" + base + "`: `" + num + "` is not a layer index";
        return false;
    }
    const long v = std::strtol(num.c_str(), nullptr, 10);
    if (v < 0 || v > 4096) {
        err = "lora: `" + base + "`: layer " + num + " is outside the range this table can hold";
        return false;
    }
    layer = (int) v;

    const std::string tail = base.substr(dot + 1);
    if (tail.size() <= kw.size() || tail.compare(tail.size() - kw.size(), kw.size(), kw) != 0) {
        err = "lora: `" + base + "` does not end in `.weight`";
        return false;
    }
    const std::string tensor = tail.substr(0, tail.size() - kw.size());
    if (tensor == "attn_qkv" || tensor == "attn_gate" || tensor == "attn_q" || tensor == "attn_k" ||
        tensor == "attn_v") {
        act = kLoraActInput;
    } else if (tensor == "attn_output") {
        act = kLoraActAttn;
    } else if (tensor == "ssm_out") {
        act = kLoraActSsm;
    } else {
        err = "lora: " + base + " targets `" + tensor + "`, which this build cannot serve: the device table holds "
              "the attention and GDN projections only (attn_qkv, attn_gate, attn_q, attn_k, attn_v, attn_output, "
              "ssm_out); the shared expert (ffn_*_shexp) lands in step 2.4";
        return false;
    }
    return true;
}

// The flag, read on the device: a request that does not ask for the adapter returns before touching anything,
// so its output is bit-identical to the base engine's (`docs/research/phase2_engine_lora_plan.md` 5.4).
__global__ void lora_kernel(const LoraEntry* __restrict__ entries, const int* __restrict__ on, int64_t T,
                            LoraIo act0, LoraIo act1, LoraIo act2, LoraOuts outs) {
    if (*on == 0) return;   // the first statement: this request runs the base model
    const LoraEntry en = entries[blockIdx.x];
    const LoraIo io = en.act == kLoraActAttn ? act1 : (en.act == kLoraActSsm ? act2 : act0);
    const LoraOut out = outs.e[blockIdx.x];

    __shared__ float mid[kMaxRank];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t t0 = (int64_t) blockIdx.y * kTokensPerBlock;
    const int64_t t1 = t0 + kTokensPerBlock < T ? t0 + kTokensPerBlock : T;
    for (int64_t t = t0; t < t1; ++t) {
        const float* x = io.x + t * io.ld_x;
        float* y = out.y + t * out.ld_y;
        __syncthreads();   // `mid` from the previous token is no longer needed
        // A @ x: one warp per k, j walked by lane, reduced by a fixed tree.  A is A^T, element (k, j) at
        // k + rank * j (`core::LoraMatrix`'s rule), so a fixed k is a strided read - the table is small and
        // stays in L1 for the token tile.
        for (int k = warp; k < en.rank; k += kWarps) {
            float s = 0.0f;
            for (int j = lane; j < en.n_in; j += 32)
                s = fmaf(f32_from_f16(en.a[(size_t) k + (size_t) en.rank * j]), x[j], s);
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
            if (lane == 0) mid[k] = s;
        }
        __syncthreads();
        // B @ mid, serial over k (the fixed order the reference's comment promises), then one add into the
        // base projection's output.
        for (int i = threadIdx.x; i < en.n_out; i += kThreads) {
            const uint16_t* b = en.b + i;
            float s = 0.0f;
            for (int k = 0; k < en.rank; ++k) s = fmaf(f32_from_f16(b[(size_t) en.n_out * k]), mid[k], s);
            y[i] += en.scale * s;
        }
    }
}

}  // namespace

const Lora& lora() { return g_lora; }

bool lora_upload(const core::LoraAdapter& a, const std::string& dir, std::string& err) {
    // One adapter per process (the header's rule).  A second upload would have to free tables that graphs already
    // captured still read, and nothing re-captures them: the result would be a plausible answer from freed memory.
    if (g_lora.loaded()) {
        err = "lora: an adapter is already resident (" + std::to_string(g_lora.n_entries) + " entries, " +
              mb(g_lora.bytes) + "); this build loads one adapter at start-up, because its kernel node is inside "
              "every graph captured after the upload, so replacing it would leave those graphs reading freed tables "
              "- restart the engine to change adapters";
        return false;
    }
    if (a.targets.empty()) {
        err = "lora: the adapter has no targets";
        return false;
    }

    // ---- 1. what the tables will hold.  Nothing below touches the resident adapter, so a refusal leaves it exactly
    //         as it was.
    std::vector<PlanEntry> plan;
    plan.reserve(a.targets.size());
    std::vector<std::unique_ptr<std::string>> names;
    names.reserve(a.targets.size());
    for (const core::LoraTarget& t : a.targets) {
        PlanEntry p;
        p.target = &t;
        if (!parse_target(t.base, p.layer, p.desc.act, err)) return false;
        if (t.rank < 1 || t.rank > kMaxRank) {
            err = "lora: " + t.base + ": rank " + std::to_string(t.rank) + " is outside 1.." +
                  std::to_string(kMaxRank);
            return false;
        }
        if (t.in < 1 || t.out < 1 || t.in > INT_MAX || t.out > INT_MAX) {
            err = "lora: " + t.base + ": the projection's shape " + std::to_string(t.in) + " x " +
                  std::to_string(t.out) + " is outside what the kernel holds";
            return false;
        }
        // The loader built these; a mismatch here would be a bug in the reader, and the kernel's indexing
        // depends on it, so it is refused rather than trusted.
        if (t.a.ne0 != t.rank || t.a.ne1 != t.in || t.b.ne0 != t.out || t.b.ne1 != t.rank) {
            err = "lora: " + t.base + ": the loader's A/B shape disagrees with its own in/out";
            return false;
        }
        p.desc.n_in = (int) t.in;
        p.desc.n_out = (int) t.out;
        p.desc.rank = (int) t.rank;
        p.desc.scale = (float) t.scale();
        names.push_back(std::make_unique<std::string>(t.base));
        p.desc.name = names.back()->c_str();
        plan.push_back(p);
    }
    // Keep the layers contiguous so one device array and one slice offset per layer describe the whole table.
    std::stable_sort(plan.begin(), plan.end(),
                     [](const PlanEntry& x, const PlanEntry& y) { return x.layer < y.layer; });

    // `LoraOuts`, the kernel's output carrier, has a fixed size, so a layer's entry count has a ceiling.  Four
    // is the most this family has on one layer (q/k/v/o, or qkv/gate/ssm_out); the check is here so a
    // hand-written index cannot overrun the carrier.
    for (size_t i = 0; i < plan.size();) {
        size_t j = i;
        while (j < plan.size() && plan[j].layer == plan[i].layer) ++j;
        if (j - i > (size_t) kMaxEntriesPerLayer) {
            err = "lora: layer " + std::to_string(plan[i].layer) + " has " + std::to_string(j - i) +
                  " adapted projections; the kernel's output carrier holds " + std::to_string(kMaxEntriesPerLayer);
            return false;
        }
        i = j;
    }

    // ---- 2. the fp16 bytes, into the layout the kernel indexes: every A, 256-byte boundary, every B.
    uint64_t a_elems = 0, b_elems = 0;
    for (const PlanEntry& p : plan) {
        a_elems += (uint64_t) p.desc.rank * p.desc.n_in;
        b_elems += (uint64_t) p.desc.n_out * p.desc.rank;
    }
    const uint64_t a_bytes = a_elems * sizeof(uint16_t);
    const uint64_t b_start = (a_bytes + kTableAlign - 1) / kTableAlign * kTableAlign;
    const uint64_t table_bytes = b_start + b_elems * sizeof(uint16_t);
    // The loader checks shapes and ranks, never VRAM (step-2 checklist 2.2 assumed it did), so this is where an
    // adapter that cannot sit under the engine's reserve is refused - with both numbers, before a byte is read.
    // `upload_here` checks again for a layer split, where there is no earlier point to refuse at.
    if (!vram_fits_here(table_bytes, err)) return false;
    std::vector<uint16_t> tables((size_t) (table_bytes / sizeof(uint16_t)), 0);
    uint64_t ao = 0, bo = b_start / sizeof(uint16_t);
    for (PlanEntry& p : plan) {
        std::vector<uint16_t> buf;
        if (!core::lora_read_f16(dir, p.target->a, buf, err)) return false;
        if (buf.size() != (size_t) p.desc.rank * (size_t) p.desc.n_in) {
            err = std::string("lora: ") + p.desc.name + ": A read " + std::to_string(buf.size()) + " values, not " +
                  std::to_string(p.desc.rank * p.desc.n_in);
            return false;
        }
        std::copy(buf.begin(), buf.end(), tables.begin() + (ptrdiff_t) ao);
        p.a_off = ao;
        ao += buf.size();
        if (!core::lora_read_f16(dir, p.target->b, buf, err)) return false;
        if (buf.size() != (size_t) p.desc.n_out * (size_t) p.desc.rank) {
            err = std::string("lora: ") + p.desc.name + ": B read " + std::to_string(buf.size()) + " values, not " +
                  std::to_string(p.desc.n_out * p.desc.rank);
            return false;
        }
        std::copy(buf.begin(), buf.end(), tables.begin() + (ptrdiff_t) bo);
        p.b_off = bo;
        bo += buf.size();
    }
    // The matrices are read; `LoraTarget` belongs to the caller and may not outlive this call.
    for (PlanEntry& p : plan) p.target = nullptr;

    // ---- 3. the per-layer view.  A layer the adapter does not touch keeps `n == 0`, so `covers()` and the
    //         launcher both see it as "nothing to do" without a second description of the adapter.
    const int64_t n_layers = (int64_t) plan.back().layer + 1;
    std::vector<std::vector<LoraEntry>> entries_host((size_t) n_layers);
    std::vector<LoraLayer> layers_host((size_t) n_layers);
    std::vector<size_t> layer_first((size_t) n_layers, 0);
    size_t idx = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        layer_first[(size_t) l] = idx;
        while (idx < plan.size() && plan[idx].layer == l) {
            LoraEntry e = plan[idx].desc;
            e.a = tables.data() + plan[idx].a_off;
            e.b = tables.data() + plan[idx].b_off;
            entries_host[(size_t) l].push_back(e);
            ++idx;
        }
        layers_host[(size_t) l].entries =
            entries_host[(size_t) l].empty() ? nullptr : entries_host[(size_t) l].data();
        layers_host[(size_t) l].n = (int) entries_host[(size_t) l].size();
    }

    // ---- 4. the host state, then this device.  There is no replace path: an adapter is resident for the life of
    //         the process, so every device that never got one (a layer split) is served by `lora_replicate`.
    g_plan = std::move(plan);
    g_names = std::move(names);
    g_tables_host = std::move(tables);
    g_entries_host = std::move(entries_host);
    g_layers_host = std::move(layers_host);
    g_layer_first = std::move(layer_first);

    g_lora = Lora{};
    g_lora.layers = g_layers_host.data();
    g_lora.n_layers = (int64_t) g_layers_host.size();
    g_lora.n_entries = (int64_t) g_plan.size();
    g_lora.bytes = (uint64_t) g_tables_host.size() * sizeof(uint16_t);
    g_on_host = true;   // a new adapter starts ON, like a new control vector
    if (!upload_here(err)) {
        g_lora = Lora{};
        g_on_host = false;
        g_plan.clear();
        g_names.clear();
        g_tables_host.clear();
        g_entries_host.clear();
        g_layers_host.clear();
        g_layer_first.clear();
        return false;
    }
    g_lora.on = g_dev[cur_device()].on;
    return true;
}

bool lora_replicate(std::string& err) { return !g_lora.loaded() || upload_here(err); }

void lora_set_enabled(bool on) {
    if (!g_lora.loaded() || on == g_on_host) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

bool lora_enabled() { return g_lora.loaded() && g_on_host; }

int lora_find(int64_t layer, const char* base) {
    if (!g_lora.loaded() || base == nullptr) return -1;
    if (layer < 0 || layer >= (int64_t) g_entries_host.size()) return -1;
    const std::vector<LoraEntry>& es = g_entries_host[(size_t) layer];
    // The adapter stores `blk.<layer>.<tensor>.weight`; a call site names the tensor the way the model's layout
    // does (`attn_k.weight`).  Both spellings are accepted so a caller that already holds the full name is not
    // forced to strip it.  A name the layer does not target is -1 - never a guess, because a guessed index would
    // aim one projection's delta at another projection's tensor.
    for (size_t i = 0; i < es.size(); ++i) {
        const char* n = es[i].name;
        if (n == nullptr) continue;
        if (std::strcmp(n, base) == 0) return (int) i;
        const char* dot = std::strchr(n, '.');
        if (dot != nullptr) dot = std::strchr(dot + 1, '.');
        if (dot != nullptr && std::strcmp(dot + 1, base) == 0) return (int) i;
    }
    return -1;
}

void lora_apply_layer(int64_t layer, int64_t T, const LoraIo io[kLoraActCount], const LoraOut* outs, void* stream) {
    if (!g_lora.loaded() || T < 1) return;   // no adapter: the base path, and nothing to check
    if (layer < 0 || layer >= (int64_t) g_entries_host.size()) return;   // a layer the adapter does not touch
    const std::vector<LoraEntry>& es = g_entries_host[(size_t) layer];
    if (es.empty()) return;
    const DevTables& t = g_dev[cur_device()];
    // Everything below is a programming error at the call site, not a data condition: a silent return would be a
    // base-model answer wearing the adapter's name, which is the one outcome this path must not produce.  They
    // throw exactly where `cvec_apply` throws (`src/kernels/cuda/cvec.cu:162-172`).
    if (t.layers == nullptr)
        throw std::runtime_error("lora_apply_layer: the adapter is not on this device (call lora_replicate for the "
                                 "layer split)");
    if (io == nullptr)
        throw std::runtime_error("lora_apply_layer: layer " + std::to_string(layer) +
                                 " has entries but the activation array is null");
    if (outs == nullptr)
        throw std::runtime_error("lora_apply_layer: layer " + std::to_string(layer) + " has " +
                                 std::to_string(es.size()) + " entries but the output array is null");
    if (es.size() > (size_t) kMaxEntriesPerLayer)
        throw std::runtime_error("lora_apply_layer: layer " + std::to_string(layer) + " has " +
                                 std::to_string(es.size()) + " entries; the output carrier holds " +
                                 std::to_string(kMaxEntriesPerLayer));   // refused at upload; belt and braces
    LoraOuts arg{};
    for (size_t i = 0; i < es.size(); ++i) {
        const LoraEntry& e = es[i];
        const std::string name = e.name != nullptr ? e.name : ("entry " + std::to_string(i));
        if (e.act < 0 || e.act >= kLoraActCount)
            throw std::runtime_error("lora_apply_layer: " + name + " names activation " + std::to_string(e.act) +
                                     ", which is not one of the three");   // cannot happen; never index io[] with it
        if (io[e.act].x == nullptr)
            throw std::runtime_error("lora_apply_layer: " + name + " reads activation " + std::to_string(e.act) +
                                     " but that pointer was not passed (io[" + std::to_string(e.act) + "].x is null)");
        if (outs[i].y == nullptr)
            throw std::runtime_error("lora_apply_layer: " + name + "'s output is null (outs[" + std::to_string(i) +
                                     "].y)");
        // A stride smaller than the row's width would make token t read or write token t-1's bytes.  A non-positive
        // stride is caught by the same comparison, because both widths are at least 1.
        if (io[e.act].ld_x < (int64_t) e.n_in)
            throw std::runtime_error("lora_apply_layer: " + name + ": the activation's row stride " +
                                     std::to_string(io[e.act].ld_x) + " is smaller than its width " +
                                     std::to_string(e.n_in));
        if (outs[i].ld_y < (int64_t) e.n_out)
            throw std::runtime_error("lora_apply_layer: " + name + ": the output's row stride " +
                                     std::to_string(outs[i].ld_y) + " is smaller than its width " +
                                     std::to_string(e.n_out));
        arg.e[i] = outs[i];
    }
    const dim3 grid((unsigned) es.size(), (unsigned) ((T + kTokensPerBlock - 1) / kTokensPerBlock));
    lora_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(t.entries + g_layer_first[(size_t) layer], t.on, T,
                                                             io[kLoraActInput], io[kLoraActAttn], io[kLoraActSsm],
                                                             arg);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("lora_apply_layer: launch failed");
}

}  // namespace strata::kernels
