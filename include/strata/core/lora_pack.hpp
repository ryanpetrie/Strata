// include/strata/core/lora_pack.hpp - a LoRA adapter as the ENGINE reads it: one flat text index and one fp16 blob.
//
// The base weights' rule is the adapter's rule: the engine does not parse JSON, because a parser here is a new,
// unaudited component whose failure mode is a wrong byte offset - which decodes to a plausible weight and
// produces plausible logits (`include/strata/core/weights.hpp:25-30`).  The conversion stays in Python
// (`tools/lora_pack.py`), where the safetensors header is already being written and read, and this file only
// checks what the index CLAIMS against the model's own layout table.
//
//     <dir>/index.txt      a flat, fixed-width manifest (space separated, no quoting)
//     <dir>/lora.bin       fp16, little-endian, in the index's own order
//
//     # strata lora index v1
//     # name rank alpha ne0 ne1 offset bytes
//     blk.3.attn_q.weight.lora_a 16 16.0    16  2560      0  81920
//     blk.3.attn_q.weight.lora_b 16 16.0 12288    16  81920 393216
//
// THE ORIENTATION, which is the trap this format exists to survive.  `ne0` is the CONTIGUOUS axis, the pack's
// own convention (`include/strata/core/family_layout.hpp:12`), so an element (i, j) sits at `i + ne0 * j` and
// the file holds each matrix TRANSPOSED against its row-major source: for a base tensor of GGUF shape
// [out, in], the adapter's A is ne0=rank ne1=in (the file holds [in][rank]) and B is ne0=out ne1=rank (the file
// holds [rank][out]).  Hugging Face / PEFT keep the LAST axis contiguous, so BOTH A and B are transposed on the
// way in - the plan's first draft said only B (`docs/research/phase2_engine_lora_plan.md` 2.1, corrected by
// `docs/research/phase2_lora_spike.md` 3).  Getting either one wrong produces a delta of the same magnitude as
// the right answer and uncorrelated with it (measured: 27175 and 24515 against a norm of 24344, cosines -0.103
// and 0.015), with no error anywhere.  `lora_delta_f32` below is the host reference that catches it.
//
// WHAT THE LOADER REFUSES.  Never a partial load, never a warning, never a silent skip
// (`docs/research/phase2_engine_lora_plan.md` 3.5): a name the model does not have, a name that a layer of the
// other kind would have (a QSA projection on a GDN layer), the wrong shape, a routed-expert tensor (with its
// size), a rank above `kMaxLoraRank`, a row whose bytes disagree with its own shape, or a pair whose rank and
// alpha disagree.  Each refusal names the tensor, because the alternative is a plausible wrong adapter.
#pragma once

#include "strata/core/geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The rank this engine accepts.  An adapter above it is refused at load rather than truncated: a truncated
/// adapter is a different function, and its error would be silent.
inline constexpr int64_t kMaxLoraRank = 64;

/// One matrix of an adapter, exactly as one index row describes it.  `ne0` is the contiguous axis.
struct LoraMatrix {
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    uint64_t offset = 0;  ///< into lora.bin
    uint64_t bytes = 0;   ///< as the row claims; `lora_load` checks it against ne0 * ne1 * 2
};

/// One base tensor's adapter.  A is [rank, in] and B is [out, rank] in the mathematical sense, and both are
/// stored transposed (see the orientation note above), so `delta = scale() * B @ (A @ x)`.
struct LoraTarget {
    std::string base;  ///< the engine's name for the base tensor, e.g. "blk.3.attn_q.weight"
    int64_t out = 0;   ///< the base tensor's ne0
    int64_t in = 0;    ///< the base tensor's ne1
    int64_t rank = 0;
    double alpha = 0.0;
    LoraMatrix a;
    LoraMatrix b;
    /// PEFT's scaling, `alpha / rank` (the plan's 2.4).
    double scale() const { return rank > 0 ? alpha / static_cast<double>(rank) : 0.0; }
};

/// A whole adapter, validated against one model's geometry.
struct LoraAdapter {
    std::vector<LoraTarget> targets;
    uint64_t blob_bytes = 0;  ///< the size of lora.bin, so the caller can size the device allocation

    /// The target for a base tensor's engine name, or null when the adapter does not touch it.
    const LoraTarget* find(const std::string& base) const {
        for (const LoraTarget& t : targets)
            if (t.base == base) return &t;
        return nullptr;
    }
    /// What the adapter's weights cost in fp16: the A and B of every target.
    uint64_t weight_bytes() const {
        uint64_t n = 0;
        for (const LoraTarget& t : targets) n += static_cast<uint64_t>(t.a.ne0 * t.a.ne1 + t.b.ne0 * t.b.ne1) * 2;
        return n;
    }
};

/// Reads and checks `<dir>/index.txt` (+ the presence and size of `lora.bin`) against `g`'s layout table.  False
/// with `err` naming the offending tensor and what the model has instead.  `out` is untouched on failure.
bool lora_load(const std::string& dir, const ModelGeometry& g, LoraAdapter& out, std::string& err);

/// The fp16 bytes of one matrix, as the file holds them (transposed against the source's row-major order).
bool lora_read_f16(const std::string& dir, const LoraMatrix& m, std::vector<uint16_t>& out, std::string& err);

/// THE HOST REFERENCE: `dy += scale() * B @ (A @ x)`, in double precision internally and written as float, with
/// the matrices read through the index's own rule.  This is what step 2's kernel parity test compares against
/// and what step 1's round-trip test proves the orientation with; it is not the run-time path.
bool lora_delta_f32(const std::string& dir, const LoraTarget& t, const float* x, float* dy, std::string& err);

/// A digest over the adapter's bytes, rank, alpha, target set and scaling - what a saved conversation is bound
/// to, exactly as the control vector's digest binds one (`include/strata/core/conversation_file.hpp:71-83`).
/// Changes when any of those change, so a session cannot resume state read with a different adapter.
bool lora_digest(const std::string& dir, const LoraAdapter& a, uint64_t& out, std::string& err);

}  // namespace strata::core
