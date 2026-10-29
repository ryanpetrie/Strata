// Phase 3 fast path (opt-in): DirectStorage restore for the disk tier's page pool.  Windows + CUDA only.
//
// The portable disk restore reads pages through the OS into host staging and copies them to the K/V pools.
// When --conversation-cache-disk-fast-path is given (and the disk tier is in --conversation-cache-disk-cow
// mode), this engine instead DMAs the page bytes from pool/pages.bin straight into a small shared VRAM
// staging ring with DirectStorage (dstorage.dll), signals a shared D3D12 fence, waits on that fence inside
// a CUDA stream through an imported external semaphore, and lets a kernel verify every page's hash
// (session_hash64, the pool's own content address) and scatter it to the live K/V pool only on a match.
// A corrupt or stale page therefore never reaches the K/V, and a failed write keeps the fail-closed
// contract of the streaming restore: the session must be discarded and the prompt read from scratch.
//
// The runtime is loaded dynamically: an engine build with no DirectStorage present (no DLL, an AMD or Mac
// build, a machine where the D3D12 adapter cannot be matched to the CUDA device) simply has no fast path
// and the portable path serves every restore.  The ring is two 32 MiB slots: save and restore paths never
// allocate more than that for the fast path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

// One page to DMA, verify and scatter.  `file_offset`..+`bytes` are read from the pool file passed to
// write(); the kernel hashes exactly those bytes and compares with `expect_hash` (the manifest's page
// hash); on a match it copies the first `copy_bytes` (== bytes, or less for a borrowed prefix) to `dst`.
struct ConversationDsPage {
    void* dst = nullptr;        // device address inside a live K/V pool part
    uint64_t file_offset = 0;   // inside the pool file (a 4 KiB-aligned page record's data)
    uint32_t bytes = 0;         // the page's saved length: the verified extent
    uint32_t copy_bytes = 0;    // the scattered extent: <= bytes
    uint64_t expect_hash = 0;   // session_hash64(page, bytes, 0)
};

class ConversationDs {
public:
    // Probes and initializes everything once (library, factory, D3D12 device on the CUDA adapter, the
    // shared staging ring, fence + CUDA semaphore, queue).  Returns null with the reason in `why` when
    // the fast path is not available here; that is never an error, only a refusal.
    static std::unique_ptr<ConversationDs> create(std::string& why);
    ~ConversationDs();
    ConversationDs(const ConversationDs&) = delete;
    ConversationDs& operator=(const ConversationDs&) = delete;

    // Reads, verifies and scatters `pages` in order.  True only when every page's hash matched and all
    // copies completed.  On false the session may hold partial K/V: the caller must discard it (the
    // same contract as a failed streaming restore) and may fall back to the portable path.
    bool write(const std::string& pool_path, const std::vector<ConversationDsPage>& pages, std::string& error);

    uint64_t staging_bytes() const;          // the VRAM staging ring (two slots)
    const std::string& describe() const;     // for the engine INFO line and the log

private:
    struct Impl;
    explicit ConversationDs(std::unique_ptr<Impl> impl);   // defined in the .cpp, where Impl is complete
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
