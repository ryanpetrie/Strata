// Phase 3 fast path: on-device verify and scatter for pages DirectStorage placed in the staging ring.
//
// One job is one page: its bytes are already in a device staging buffer; `expect_hash` is the content
// hash the page pool's manifest names for it (session_hash64(data, hash_bytes, 0)).  The kernel hashes
// each staging page and copies it to `dst` only on a match, so a corrupt or stale page never reaches
// the K/V pools; any mismatch raises the `bad` flag and the restore fails closed like the portable path.
// CUDA/HIP-portable code (the kernels library compiles this file for both); the DirectStorage engine
// that feeds it is Windows + CUDA only.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

struct DsPageJob {
    const uint8_t* src = nullptr;  // staging: the page's bytes, in place
    uint8_t* dst = nullptr;        // the live K/V pool's device address for this page
    uint64_t expect_hash = 0;      // session_hash64(src, hash_bytes, 0)
    uint32_t hash_bytes = 0;       // the page's saved length (verified)
    uint32_t copy_bytes = 0;       // bytes to scatter: hash_bytes, or less for a borrow prefix
};

// One block per job (256 threads), `stream`-ordered. Sets *bad (atomicOr 1) if any page's hash does not
// match; a mismatched job copies nothing. `bad` must be device memory the caller zeroes and reads back.
void ds_verify_scatter(const DsPageJob* jobs, uint32_t njobs, uint32_t* bad, cudaStream_t stream);

} // namespace strata::kernels
