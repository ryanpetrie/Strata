// ds_verify_scatter: hash each staged page exactly as the host SessionHasher does (a copy of the
// streaming rounds in src/core/conversation_file.cpp; conversation_ds_test compares them) and scatter
// it to the live pool only when the hash matches.  Thread 0 of the block hashes (the rounds are a
// serial lane chain); after the shared verdict the block copies the page with 16-byte vectors.
#include "strata/kernels/ds_pages.hpp"

namespace {
constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull,
                   P4 = 0x85EBCA77C2B2AE63ull, P5 = 0x27D4EB2F165667C5ull;
__device__ __forceinline__ uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
__device__ __forceinline__ uint64_t round1(uint64_t acc, uint64_t w) { return rotl(acc + w * P2, 31) * P1; }
__device__ __forceinline__ uint64_t load64(const uint8_t* p) {
    return *reinterpret_cast<const uint64_t*>(p);   // callers keep every page base 64-byte aligned
}

// session_hash64(data, n, 0): the SessionHasher streaming order in one pass, no pending buffer needed.
__device__ uint64_t session_hash64_d(const uint8_t* p, uint32_t n) {
    uint64_t lane[4] = {P1 + P2, P2, 0ull, 0ull - P1};
    uint32_t i = 0;
    for (; i + 32 <= n; i += 32)
#pragma unroll
        for (int k = 0; k < 4; ++k) lane[k] = round1(lane[k], load64(p + i + 8 * k));
    uint64_t h = rotl(lane[0], 1) + rotl(lane[1], 7) + rotl(lane[2], 12) + rotl(lane[3], 18);
#pragma unroll
    for (int k = 0; k < 4; ++k) { h ^= round1(0, lane[k]); h = h * P1 + P4; }
    h += n;
    const uint32_t rem = n - i;
    uint32_t j = 0;
    for (; j + 8 <= rem; j += 8) { h ^= round1(0, load64(p + i + j)); h = rotl(h, 27) * P1 + P4; }
    for (; j < rem; ++j) { h ^= (uint64_t)__ldg(p + i + j) * P5; h = rotl(h, 11) * P1; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

__global__ void ds_verify_scatter_kernel(const strata::kernels::DsPageJob* __restrict__ jobs, uint32_t njobs,
                                         uint32_t* __restrict__ bad) {
    const uint32_t job_i = blockIdx.x;
    if (job_i >= njobs) return;
    const strata::kernels::DsPageJob job = jobs[job_i];
    __shared__ int ok;
    if (threadIdx.x == 0) ok = session_hash64_d(job.src, job.hash_bytes) == job.expect_hash ? 1 : 0;
    __syncthreads();
    if (!ok) {
        if (threadIdx.x == 0) atomicOr(bad, 1u);
        return;
    }
    const uint4* s = reinterpret_cast<const uint4*>(job.src);
    uint4* d = reinterpret_cast<uint4*>(job.dst);
    const uint32_t n16 = job.copy_bytes / 16u;
    for (uint32_t i = threadIdx.x; i < n16; i += blockDim.x) d[i] = s[i];
    for (uint32_t i = n16 * 16u + threadIdx.x; i < job.copy_bytes; i += blockDim.x) job.dst[i] = job.src[i];
}
} // namespace

namespace strata::kernels {
void ds_verify_scatter(const DsPageJob* jobs, uint32_t njobs, uint32_t* bad, cudaStream_t stream) {
    if (njobs == 0) return;
    ds_verify_scatter_kernel<<<njobs, 256, 0, stream>>>(jobs, njobs, bad);
}
} // namespace strata::kernels
