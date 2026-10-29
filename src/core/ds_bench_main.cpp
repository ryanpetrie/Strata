// ds_bench - the Phase 3 fast path's restore measured against the portable page read, on the SAME pool
// file, the SAME pages and the SAME device destination, in one process (Windows + CUDA; manual run, never
// a ctest test: it needs a card and writes a few hundred MiB of pool under --dir).
//
// The engine's disk restore has two I/O shapes for a paged (--conversation-cache-disk-cow) entry:
//   portable: read each page's bytes file -> host staging (the OS's read path), then copy to the K/V pool;
//   fast:     DirectStorage DMAs the same bytes file -> a shared VRAM ring, a kernel hashes every page
//             against the manifest and scatters it to the K/V pool (ConversationDs::write).
// This tool builds a real pool with the engine's own writer (pages of 64 KiB, 4 KiB-aligned records,
// content hashes), then runs both shapes into one device buffer and prints MiB/s for each, plus a
// byte-exact match check between the two paths and against the source bytes.  Nothing here goes through
// a session or a model: it isolates the bytes moved.
//
//   ds_bench [--pages N] [--dir PATH]
//
// --pages defaults to 8192 (512 MiB); --dir defaults to a temp folder and is cleaned up.  The pool file
// is written once, dropped from the OS cache is NOT attempted (both arms then read the same cached file
// - the portable restore's steady state, and the harsher fairness for the fast path is noted in the docs).
#include "strata/core/conversation_ds.hpp"
#include "strata/core/conversation_pages.hpp"
#include "strata/core/conversation_file.hpp"

#include <cuda_runtime.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA (%s): %s\n", where, cudaGetErrorString(e)); std::exit(1); }
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace

int main(int argc, char** argv) {
    uint32_t pages = 8192;
    std::string dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--pages") && i + 1 < argc) pages = (uint32_t) std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else { std::fprintf(stderr, "usage: ds_bench [--pages N] [--dir PATH]\n"); return 2; }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::fprintf(stderr, "no CUDA device\n"); return 77; }
    if (dir.empty()) dir = (fs::temp_directory_path() / "strata_ds_bench_pool").string();
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) { std::fprintf(stderr, "cannot create %s: %s\n", dir.c_str(), ec.message().c_str()); return 1; }

    // 1. a real pool with the engine's writer: pages of 64 KiB, real hashes, 4 KiB-aligned records.
    const uint64_t page_bytes = kConversationPageBytes;
    const uint64_t total = (uint64_t) pages * page_bytes;
    std::mt19937_64 rng(0xD5B0);
    std::vector<uint8_t> source(total);
    for (uint64_t i = 0; i < total; i += 64) {   // 64-byte granularity: fast to fill, distinct pages
        uint64_t a = rng(), b = rng(), c = rng(), d = rng();
        for (unsigned k = 0; k < 64 && i + k < total; k += 8) {
            std::memcpy(source.data() + i + k, &a, 8); std::memcpy(source.data() + i + k + 8, &b, 8);
            std::memcpy(source.data() + i + k + 16, &c, 8); std::memcpy(source.data() + i + k + 24, &d, 8);
        }
    }
    std::string perr;
    ConversationPagePool pool(dir + "/pool");
    if (!pool.open(true, perr)) { std::fprintf(stderr, "pool open: %s\n", perr.c_str()); return 1; }
    std::vector<ConversationPageRef> refs(pages);
    for (uint32_t i = 0; i < pages; ++i) {
        ConversationPageRef ref;
        bool wrote = false;
        if (!pool.append(session_hash64(source.data() + (uint64_t) i * page_bytes, page_bytes, 0),
                          source.data() + (uint64_t) i * page_bytes, (uint32_t) page_bytes, ref, wrote, perr)) {
            std::fprintf(stderr, "append %u: %s\n", i, perr.c_str()); return 1;
        }
        refs[i] = ref;
    }
    const std::string pool_path = pool.path();
    std::printf("ds_bench: pool %s, %u pages, %llu MiB\n", pool_path.c_str(), pages, (unsigned long long) (total >> 20));

#ifdef _WIN32   // diagnosis: which open combinations does the live pool handle permit?
    {
        const DWORD accs[] = {GENERIC_READ, GENERIC_READ | DELETE};
        const DWORD shares[] = {FILE_SHARE_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE};
        for (int a = 0; a < 2; ++a)
            for (int s = 0; s < 3; ++s) {
                std::wstring wv(pool_path.size(), L'\0');
                for (size_t i = 0; i < pool_path.size(); ++i) wv[i] = (wchar_t) (unsigned char) pool_path[i];
                HANDLE h = CreateFileW(wv.c_str(), accs[a], shares[s], nullptr, OPEN_EXISTING,
                                       FILE_FLAG_OVERLAPPED, nullptr);
                std::printf("  diag acc=%s share=%d: %s (%lu)\n", a ? "R|DELETE" : "R", s,
                            h != INVALID_HANDLE_VALUE ? "ok" : "FAIL",
                            h != INVALID_HANDLE_VALUE ? 0ul : GetLastError());
                if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
            }
    }
#endif

    // 2. the destination: one device buffer, as a K/V pool part would be.
    uint8_t* dst = nullptr;
    cuda_check(cudaMalloc(&dst, total), "dst alloc");

    // 3. portable shape: the reader's own reads (pool.read verifies each hash), into host staging, then H2D.
    auto portable = [&](std::vector<uint8_t>* out_host) {
        static std::vector<uint8_t> stage;
        if (stage.empty()) stage.resize(page_bytes);
        cuda_check(cudaMemset(dst, 0, total), "clear dst");
        uint8_t* pin = nullptr;
        cuda_check(cudaHostAlloc((void**)&pin, total, cudaHostAllocDefault), "pin");
        for (uint32_t i = 0; i < pages; ++i) {
            ConversationPagePool::Page pg;
            if (!pool.lookup(refs[i].hash, refs[i].bytes, pg) ||
                !pool.read(refs[i].hash, refs[i].bytes, pg.offset, stage.data(), perr)) {
                std::fprintf(stderr, "portable read %u: %s\n", i, perr.c_str()); std::exit(1);
            }
            std::memcpy(pin + (uint64_t) i * page_bytes, stage.data(), page_bytes);
        }
        cuda_check(cudaMemcpy(dst, pin, total, cudaMemcpyHostToDevice), "h2d");
        if (out_host) out_host->assign(pin, pin + total);
        cudaFreeHost(pin);
    };
    // warm the OS cache and prove the shape once
    std::vector<uint8_t> host_copy;
    portable(&host_copy);
    check(std::memcmp(host_copy.data(), source.data(), total) == 0, "portable read back equals source");

    // 4. the fast shape: the same pages as ConversationDs jobs, staged and scattered by the queue.
    std::string why;
    auto ds = ConversationDs::create(why);
    if (!ds) { std::fprintf(stderr, "no fast path here: %s\n", why.c_str()); return 77; }
    std::vector<ConversationDsPage> jobs(pages);
    for (uint32_t i = 0; i < pages; ++i) {
        ConversationPagePool::Page pg;
        if (!pool.lookup(refs[i].hash, refs[i].bytes, pg)) { std::fprintf(stderr, "lookup %u\n", i); return 1; }
        jobs[i].dst = dst + (uint64_t) i * page_bytes;
        jobs[i].file_offset = pg.offset;
        jobs[i].bytes = refs[i].bytes;
        jobs[i].copy_bytes = refs[i].bytes;
        jobs[i].expect_hash = refs[i].hash;
    }
    std::string werr;
    cuda_check(cudaMemset(dst, 0, total), "clear dst");
    std::string berr;
    if (!pool.borrow_for_fast_read(berr)) { std::fprintf(stderr, "borrow: %s\n", berr.c_str()); return 1; }
    const bool wrote = ds->write(pool_path, jobs, werr);
    pool.return_from_fast_read();
    if (!wrote) { std::fprintf(stderr, "fast write failed: %s\n", werr.c_str()); return 1; }
    std::vector<uint8_t> fast_back(total);
    cuda_check(cudaMemcpy(fast_back.data(), dst, total, cudaMemcpyDeviceToHost), "fast back");
    check(std::memcmp(fast_back.data(), source.data(), total) == 0, "fast write bytes equal source");
    std::printf("ds_bench: fast path: %s\n", ds->describe().c_str());

    // 5. timing: three runs of each, alternating, and report the best and every run (a contended box
    //    shows in the spread; the report quotes all of it, not a cherry-pick).
    auto gib = [&](double ms) { return (double) total / (1ull << 30) / (ms / 1000.0); };
    double p_best = 1e30, f_best = 1e30;
    for (int round = 0; round < 3; ++round) {
        cuda_check(cudaMemset(dst, 0, total), "clear");
        auto t0 = std::chrono::steady_clock::now();
        portable(nullptr);
        double pm = ms_since(t0);
        cuda_check(cudaMemset(dst, 0, total), "clear");
        auto t1 = std::chrono::steady_clock::now();
        if (!pool.borrow_for_fast_read(berr)) { std::fprintf(stderr, "borrow: %s\n", berr.c_str()); return 1; }
        const bool okw = ds->write(pool_path, jobs, werr);
        pool.return_from_fast_read();
        if (!okw) { std::fprintf(stderr, "fast round: %s\n", werr.c_str()); return 1; }
        double fm = ms_since(t1);
        cudaDeviceSynchronize();
        p_best = pm < p_best ? pm : p_best;
        f_best = fm < f_best ? fm : f_best;
        std::printf("round %d: portable %8.1f ms (%6.0f MiB/s)   fast %8.1f ms (%6.0f MiB/s)\n",
                    round + 1, pm, (double) total / (1ull << 20) / (pm / 1000.0),
                    fm, (double) total / (1ull << 20) / (fm / 1000.0));
    }
    std::printf("best: portable %.1f ms (%.0f MiB/s)   fast %.1f ms (%.0f MiB/s)   ratio %.2fx\n",
                p_best, (double) total / (1ull << 20) / (p_best / 1000.0),
                f_best, (double) total / (1ull << 20) / (f_best / 1000.0), p_best / f_best);
    (void) gib;
    cudaFree(dst);
    fs::remove_all(dir, ec);
    std::printf("ds_bench: %d checks passed\n", checks);
    return 0;
}
