// src/kernels/cuda/lora_vram.cuh - the part of the adapter upload's VRAM check that can be tested without a GPU.
//
// `lora_upload` refuses an adapter whose fp16 tables would eat into the engine's own reserve, because the loader
// checks shapes and ranks but never VRAM (the step-2 checklist's 2.2 assumed it did; `src/core/lora_pack.cpp`
// never calls a memory query).  The query needs a card; the decision does not, so the arithmetic lives here and
// `src/kernels/lora_parity.cpp` pins it with synthetic numbers - the same split as a request rule tested without
// the engine.  Host-only: no CUDA types, so the test needs no device to include it.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The engine's reserve, in MiB, that a request may not take VRAM from.  DUPLICATED from `kVramSafeMib` in
/// `src/program/generate.cpp` (line 1422 in this tree): it is a file-local constant there (`constexpr` at
/// namespace scope has internal linkage), so `src/kernels/` cannot link to it and this is the same number for
/// the same reason (#2065: when the card is this full, WDDM pages GPU memory to system RAM instead of failing,
/// decode falls to a few tok/s and a window may never finish).  1024 MiB on Windows, 512 elsewhere, exactly as
/// that constant splits.  A change there has to be made here too; no test can see the other copy.
inline constexpr int64_t kLoraVramReserveMib =
#ifdef _WIN32
    1024;
#else
    512;
#endif

/// What the upload's check decided, with the numbers its refusal prints, so a test can pin both the decision and
/// the message's arithmetic.
struct LoraVramFit {
    bool fits = false;
    uint64_t table_mib = 0;   ///< the adapter's fp16 tables, rounded UP to whole MiB - never claim less than allocated
    uint64_t free_mib = 0;    ///< the free VRAM the query reported, rounded DOWN to whole MiB
    int64_t reserve_mib = 0;  ///< the reserve the caller passed (`kLoraVramReserveMib` at the upload)
    /// The whole MiB the adapter needs: its tables plus the reserve.
    uint64_t need_mib() const { return table_mib + (uint64_t) (reserve_mib > 0 ? reserve_mib : 0); }
};

/// True when `table_bytes` of tables leave `reserve_mib` free with `free_bytes` free on the card.  Both roundings
/// lean toward refusing - the tables up, the free figure down - so the numbers the refusal prints are the numbers
/// the decision used, and a caller never has to explain a "did not fit" whose printed sum appears to fit.
inline LoraVramFit lora_vram_fit(uint64_t table_bytes, uint64_t free_bytes, int64_t reserve_mib) {
    LoraVramFit f;
    f.table_mib = (table_bytes + (1ull << 20) - 1) >> 20;
    f.free_mib = free_bytes >> 20;
    f.reserve_mib = reserve_mib;
    f.fits = f.free_mib >= f.need_mib();
    return f;
}

}  // namespace strata::kernels
