// Phase 3 fast path: the verify/scatter kernel (ds_verify_scatter) against the host content hash.
//
// The DirectStorage engine stages page bytes in VRAM and calls this kernel to hash every page with
// session_hash64 (the page pool's own content address) and scatter it to the live K/V pool only on a
// match.  Two things must hold for the fast path to be as trustworthy as the portable read, and this
// test checks both on the device:
//   1. the kernel's hash equals the host session_hash64(data, n, 0) for every page length the pool
//      writes - full 64 KiB pages, short tails, and odd byte counts that exercise the 8-byte and
//      single-byte tail rounds and the final avalanche;
//   2. a single flipped byte in one staged page (a) raises the `bad` flag, (b) scatters NOTHING from
//      that page (its destination is left untouched), and (c) does not disturb a good page staged
//      alongside it.  That is the fail-closed contract: a corrupt page never reaches the K/V.
//
// Real page geometry; synthetic bytes. GPU only - it exits 77 (a ctest skip) where no device is present.
#include "strata/kernels/ds_pages.hpp"
#include "strata/core/conversation_file.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA (%s): %s\n", where, cudaGetErrorString(e)); std::exit(1); }
}

constexpr uint32_t kPage = 65536;   // kConversationPageBytes

// One page's worth of device staging plus a page's worth of destination, both filled on demand.
struct Device {
    uint8_t* staging = nullptr;   // where the kernel reads each page from
    uint8_t* dst = nullptr;       // where a verified page is scattered to
    strata::kernels::DsPageJob* jobs = nullptr;
    uint32_t* bad = nullptr;
    void alloc(size_t n_pages) {
        cuda_check(cudaMalloc(&staging, n_pages * kPage), "staging");
        cuda_check(cudaMalloc(&dst, n_pages * kPage), "dst");
        cuda_check(cudaMalloc((void**)&jobs, n_pages * sizeof(strata::kernels::DsPageJob)), "jobs");
        cuda_check(cudaMalloc((void**)&bad, sizeof(uint32_t)), "bad");
    }
    void free() {
        cudaFree(staging); cudaFree(dst); cudaFree(jobs); cudaFree(bad);
    }
};

// Runs the kernel over `n` pages described by `pages`, and reads back (bad, dst).
struct Outcome {
    uint32_t bad = 0;
    std::vector<uint8_t> dst;
};

Outcome run(Device& d, const std::vector<std::vector<uint8_t>>& page_bytes,
            const std::vector<uint64_t>& expect, const std::vector<uint32_t>& copy_bytes, size_t total) {
    std::vector<strata::kernels::DsPageJob> jobs(total);
    std::vector<uint8_t> flat_staging(total * kPage, 0);
    for (size_t i = 0; i < total; ++i)
        std::memcpy(flat_staging.data() + i * kPage, page_bytes[i].data(), page_bytes[i].size());
    for (size_t i = 0; i < total; ++i) {
        jobs[i].src = d.staging + i * kPage;
        jobs[i].dst = d.dst + i * kPage;
        jobs[i].expect_hash = expect[i];
        jobs[i].hash_bytes = (uint32_t) page_bytes[i].size();
        jobs[i].copy_bytes = copy_bytes[i];
    }
    cuda_check(cudaDeviceSynchronize(), "pre-sync");
    cuda_check(cudaMemcpy(d.staging, flat_staging.data(), total * kPage, cudaMemcpyHostToDevice), "upload");
    cuda_check(cudaMemcpy(d.jobs, jobs.data(), total * sizeof(strata::kernels::DsPageJob), cudaMemcpyHostToDevice), "jobs");
    cuda_check(cudaMemset(d.dst, 0xAB, total * kPage), "poison dst");   // any untouched byte reads as 0xAB
    cuda_check(cudaMemset(d.bad, 0, sizeof(uint32_t)), "clear bad");
    strata::kernels::ds_verify_scatter(d.jobs, (uint32_t) total, d.bad, nullptr);
    cuda_check(cudaGetLastError(), "launch");
    cuda_check(cudaDeviceSynchronize(), "sync");
    Outcome o;
    o.dst.assign(total * kPage, 0);
    cuda_check(cudaMemcpy(&o.bad, d.bad, sizeof(uint32_t), cudaMemcpyDeviceToHost), "bad back");
    cuda_check(cudaMemcpy(o.dst.data(), d.dst, total * kPage, cudaMemcpyDeviceToHost), "dst back");
    return o;
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "no CUDA device: ds_pages_test skipped\n");
        return 77;
    }

    Device d;
    d.alloc(2);   // reused for the per-length hash checks and the two-page corruption check

    std::mt19937 rng(0x51A9);
    auto random_bytes = [&](size_t n) {
        std::vector<uint8_t> v(n);
        for (auto& b : v) b = (uint8_t) rng();
        return v;
    };

    // 1. every length the pool can write: full pages, tails, and small/odd counts that hit each
    //    tail round (0..7 leftover bytes) and the digest avalanche. The kernel must agree with the host.
    const std::vector<size_t> lengths = {
        1, 2, 7, 8, 9, 16, 31, 32, 33, 40, 63, 64, 65, 127, 128, 1000, 4096, 4095, kPage - 1, kPage,
    };
    for (size_t n : lengths) {
        const std::vector<uint8_t> page = random_bytes(n);
        const uint64_t want = strata::core::session_hash64(page.data(), n, 0);
        const auto o = run(d, {page}, {want}, {(uint32_t) n}, 1);
        check(o.bad == 0, "clean page raises no bad flag");
        // the scattered bytes must be exactly the page (copy_bytes == the full page length)
        check(std::memcmp(o.dst.data(), page.data(), n) == 0, "scattered bytes equal the page");
    }

    // 2. a one-byte corruption in a staged page is refused (bad flag set) and scatters nothing.
    {
        std::vector<uint8_t> good = random_bytes(kPage);
        std::vector<uint8_t> bad  = random_bytes(kPage);
        // expect hashes as if both pages were intact
        const uint64_t good_hash = strata::core::session_hash64(good.data(), kPage, 0);
        const uint64_t bad_hash  = strata::core::session_hash64(bad.data(), kPage, 0);
        // corrupt page 1 by one byte, but still send its original hash: the staged bytes no longer match.
        std::vector<uint8_t> corrupt = bad;
        corrupt[12345] ^= 0x01;
        check(corrupt != bad, "corruption changed a byte");
        const auto o = run(d, {good, corrupt}, {good_hash, bad_hash}, {kPage, kPage}, 2);
        check(o.bad == 1, "a corrupt page sets the bad flag");
        // the good page scattered normally
        check(std::memcmp(o.dst.data(), good.data(), kPage) == 0, "the good page scattered intact");
        // the corrupt page's destination is untouched: every byte still the 0xAB poison
        bool untouched = true;
        for (uint32_t i = 0; i < kPage; ++i)
            if (o.dst[kPage + i] != 0xAB) { untouched = false; break; }
        check(untouched, "a corrupt page scatters nothing (destination left untouched)");
    }

    // 3. a copy_bytes SHORTER than hash_bytes scatters only the prefix (a borrow's clipped last page)
    //    but still verifies the whole page; the bytes past the copy stay poison.
    {
        std::vector<uint8_t> page = random_bytes(kPage);
        const uint64_t want = strata::core::session_hash64(page.data(), kPage, 0);
        const uint32_t copy = 4096 + 3;   // not 16-aligned: exercises the kernel's byte tail too
        const auto o = run(d, {page}, {want}, {copy}, 1);
        check(o.bad == 0, "prefix copy: hash matches the full page");
        check(std::memcmp(o.dst.data(), page.data(), copy) == 0, "prefix copy: the copied prefix scattered");
        bool clean = true;
        for (uint32_t i = copy; i < kPage; ++i)
            if (o.dst[i] != 0xAB) { clean = false; break; }
        check(clean, "prefix copy: bytes past the copy are untouched");
    }

    // 4. zero copy_bytes still verifies (a page past a borrow prefix) and writes nothing.
    {
        std::vector<uint8_t> page = random_bytes(1000);
        const uint64_t want = strata::core::session_hash64(page.data(), 1000, 0);
        const auto o = run(d, {page}, {want}, {0}, 1);
        check(o.bad == 0, "zero copy: verified");
        bool clean = true;
        for (uint32_t i = 0; i < 1000; ++i)
            if (o.dst[i] != 0xAB) { clean = false; break; }
        check(clean, "zero copy: nothing scattered");
    }

    // 5. one bad page among many: the whole batch is flagged, every good page still scattered, only
    //    the bad one left untouched.
    {
        constexpr size_t N = 5;
        Device many;
        many.alloc(N);
        std::vector<std::vector<uint8_t>> pages(N);
        std::vector<uint64_t> hashes(N);
        for (size_t i = 0; i < N; ++i) {
            pages[i] = random_bytes(kPage);
            hashes[i] = strata::core::session_hash64(pages[i].data(), kPage, 0);
        }
        size_t poison_page = 2;
        std::vector<uint8_t> tampered = pages[poison_page];
        tampered[kPage - 1] ^= 0x80;                 // corrupt the final byte (hits the byte tail round)
        auto page_args = pages;
        page_args[poison_page] = tampered;
        const auto o = run(many, page_args, hashes, std::vector<uint32_t>(N, kPage), N);
        check(o.bad == 1, "a batch with one bad page is flagged");
        for (size_t i = 0; i < N; ++i) {
            if (i == poison_page) {
                bool untouched = true;
                for (uint32_t b = 0; b < kPage; ++b)
                    if (o.dst[i * kPage + b] != 0xAB) { untouched = false; break; }
                check(untouched, "the flagged page scattered nothing");
            } else {
                check(std::memcmp(o.dst.data() + i * kPage, pages[i].data(), kPage) == 0,
                      "a good page scattered intact beside a bad one");
            }
        }
        many.free();
    }

    d.free();
    std::printf("ds_pages_test: %d checks passed\n", checks);
    return 0;
}
