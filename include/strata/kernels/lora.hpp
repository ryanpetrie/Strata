// include/strata/kernels/lora.hpp - a LoRA adapter on the device: its fp16 tables, the per-request switch, and the
// one kernel that adds `scale * B @ (A @ x)` to a projection's output.
//
// WHAT THIS IS FOR.  The engine can load one adapter at start-up (`--lora <dir>`, step 2's plumbing) and then answer
// any request either with it or without it, on the same captured graphs: the switch is a device `int` that the
// kernel reads, exactly as the control vector's is (`include/strata/kernels/cvec.hpp:25-63`).  With the flag off the
// kernel returns before writing one byte, so the base path is untouched - that is the mechanical form of the exit
// criterion's "chat output is byte-identical" (`docs/research/phase2_engine_lora_plan.md` 5.4).
//
// THE INPUT IS PER ACTIVATION, THE OUTPUT IS PER ENTRY.  An activation is shared: a QSA layer's `attn_q`, `attn_k`
// and `attn_v` all read the layer's input, and a GDN layer's `attn_qkv` and `attn_gate` do too.  Their outputs are
// not shared: the same layer writes three tensors of different widths (`attn_q` 12288, `attn_k` 1024, `attn_v` 1024
// on the UD-IQ4_XS pack), so one output per activation could not serve a normal q/k/v adapter - the second entry
// would overwrite the first entry's tensor.  Hence `LoraIo` is indexed by `act` (one activation pointer, one row
// stride) and `LoraOut` is indexed by the layer's ENTRY, in the adapter's own order (`lora_find` maps a tensor's
// name to that index).  Both carry a row stride, so the prompt path (T tokens, step 4) and the single-token path
// (step 3) call the same function.
//
// WHY `act` EXISTS.  Two tensors in one layer do not read the same vector: `attn_output` reads the attention output
// (`src/core/verify.cpp:1828` reads `attn32_`; `src/prefill/prefill.cpp:3450` reads the gated attention image), not
// the layer's input, and `ssm_out` reads the SSM's `y` (`verify.cpp:1457-1459`).  A kernel that assumed "the layer's
// input" would apply `attn_output`'s adapter to the wrong vector and produce a plausible, silently wrong delta.
// Hence one `act` per entry and one `LoraIo` per activation; the plan's hook table is `phase2_engine_lora_plan.md` 2.
// Because the input is selected BY THE ENTRY'S `act`, a call site cannot hand `attn_output`'s delta the layer's
// input by mistake: there is one pointer for that activation and no other.
//
// THE ARITHMETIC, and the one order it is summed in.  For a target with mathematical A [rank][in] and B [out][rank],
//
//     y_i += scale * sum_k B[i][k] * sum_j A[k][j] * x_j
//
// with `scale = alpha / rank` and the sums emitted in a fixed order, so two runs of the same request agree bit for
// bit.  The file stores both matrices transposed (`include/strata/core/lora_pack.hpp:17-25`), so `a` is A^T with
// element (k, j) at `k + rank * j` and `b` is B^T with element (i, k) at `i + out * k`.  The host reference the
// parity test compares against is `core::lora_delta_f32` (`src/core/lora_pack.cpp`), which sums in double.
//
// NOT BUILT HERE: the forward-path hooks (steps 3 and 4) and the cache/session key (step 6).  This file is the
// device side only.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace strata::core {
struct LoraAdapter;  ///< `include/strata/core/lora_pack.hpp`; not included, so this header stays device-layer only
}  // namespace strata::core

namespace strata::kernels {

/// Which buffer a targeted projection reads.  The index is the position in `lora_apply_layer`'s `io` array.
enum LoraAct : int {
    kLoraActInput = 0,  ///< the layer's input (fp32 `xm`): `attn_qkv`, `attn_gate`, `attn_q`, `attn_k`, `attn_v`
    kLoraActAttn = 1,   ///< the attention output, after the heads are combined: `attn_output`
    kLoraActSsm = 2,    ///< the SSM/GDN output `y`: `ssm_out`
};
inline constexpr int kLoraActCount = 3;

/// One targeted projection of ONE layer, as the device needs it.
struct LoraEntry {
    const uint16_t* a = nullptr;  ///< device, file bytes: A^T, element (k, j) at `k + rank * j`
    const uint16_t* b = nullptr;  ///< device, file bytes: B^T, element (i, k) at `i + out * k`
    int n_in = 0;
    int n_out = 0;
    int rank = 0;
    float scale = 0.0f;  ///< alpha / rank (the plan's 2.4 `s`)
    int act = kLoraActInput;  ///< which `LoraIo` this entry reads and writes
    const char* name = nullptr;  ///< host-owned copy of the base tensor's name, for a refusal that names it
};

/// A layer's entries: a device array and its length.  A layer the adapter does not touch has `n == 0`.
struct LoraLayer {
    const LoraEntry* entries = nullptr;
    int n = 0;
};

/// One activation, shared by every entry of the layer whose `act` names it.  Token `t`'s row starts at
/// `x + t * ld_x`.  A null `x` means the call site does not have this activation; an entry needing it is NOT
/// skipped silently - `lora_apply_layer` returns false, and the call site treats that as a bug (step 3/4 pin the
/// mapping per branch).
struct LoraIo {
    const float* x = nullptr;
    int64_t ld_x = 0;
};

/// One entry's base output buffer: where its delta is added.  Entry `i` of `layer` has `outs[i]`, in the adapter's
/// own order - `lora_find` turns a base tensor's name into that index.  Token `t`'s row starts at `y + t * ld_y`.
struct LoraOut {
    float* y = nullptr;
    int64_t ld_y = 0;
};

/// The loaded adapter (empty until `lora_upload`).
struct Lora {
    /// HOST mirror, `n_layers` entries indexed by the trunk layer: what a call site reads to fill `LoraOut` and what
    /// `covers`/`lora_find` answer from.  The device's own copy is private to `lora.cu` (`cudaMalloc`'d memory may
    /// not be dereferenced here).  Entry order within a layer is the adapter's own.
    const LoraLayer* layers = nullptr;
    const int* on = nullptr;            ///< device int of the UPLOADING device: 0 = this request runs the base model.
                                        ///< `lora_apply_layer` reads the current device's own copy, so a caller that
                                        ///< launches its own kernel must not reuse this pointer after a layer split.
    int64_t n_layers = 0;
    int64_t n_entries = 0;
    uint64_t bytes = 0;  ///< the fp16 A and B tables, as allocated
    bool loaded() const { return layers != nullptr && n_entries > 0; }
    /// Whether layer `l` has anything to apply, decided at load time (it does not change per request).
    bool covers(int64_t l) const { return loaded() && l >= 0 && l < n_layers && layers[l].n > 0; }
};

/// The loaded adapter (empty until `lora_upload`).
const Lora& lora();

/// Uploads `a`'s tables, read from `dir` (the same directory `core::lora_load` validated).  Allocates one fp16
/// buffer for the whole adapter plus the per-layer entry tables and the flag, on the CURRENT device.  Starts ON,
/// like `cvec_upload`; the engine turns it off again before its first capture (`--lora`'s path in generate.cpp).
/// REFUSES a second adapter while one is resident: the adapter is a start-up decision whose kernel node is inside
/// every graph captured after the upload, so replacing it would silently leave those graphs pointing at freed
/// tables.  Changing adapters means restarting the engine.  False with `err` naming what failed (an adapter already
/// resident, an activation this build cannot serve, a name this build cannot place, the allocation, the VRAM).
bool lora_upload(const core::LoraAdapter& a, const std::string& dir, std::string& err);

/// A layer split: put the loaded adapter's tables on the CURRENT device too (`lora_apply_layer` uses the tables of
/// the device it runs on).  No-op without an adapter or when this device has them already.
bool lora_replicate(std::string& err);

/// The per-request switch, on every device that holds the adapter.  Synchronizes them when it changes, so call it
/// between requests - never inside a captured graph.
void lora_set_enabled(bool on);
bool lora_enabled();

/// The index of base tensor `base` (e.g. "attn_k.weight") among `layer`'s entries, or -1 when the adapter does not
/// target it.  A call site uses it to fill `outs` for `lora_apply_layer` without assuming the adapter's order.
int lora_find(int64_t layer, const char* base);

/// Layer `layer`'s delta on `T` tokens, one launch for every entry of the layer.  `io` is indexed by `act` (the
/// activation, shared by the entries that read it); `outs` has one element per entry of the layer, in the adapter's
/// own order (`lora_find`).  Nothing is launched, and nothing is an error, for a layer the adapter does not touch -
/// the forward path calls it unconditionally.
///
/// IT THROWS where `cvec_apply` throws, because every one of these is a programming error at the call site and a
/// silent return would be a base-model answer wearing the adapter's name: an entry whose activation was not passed,
/// an entry whose output is null, a stride too small for the entry's shape (`ld_x < n_in` or `ld_y < n_out`), the
/// tables absent from the current device (a missing `lora_replicate`), or a launch failure.
void lora_apply_layer(int64_t layer, int64_t T, const LoraIo io[kLoraActCount], const LoraOut* outs, void* stream);

}  // namespace strata::kernels
