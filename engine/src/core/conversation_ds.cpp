// ConversationDs: the DirectStorage fast path.  See the header for the contract.
//
// Windows + CUDA only.  On every other platform, and on Windows machines without the dstorage runtime or
// without a D3D12 adapter that matches the CUDA device, create() refuses with a reason and the engine
// keeps its portable disk restore - nothing else in this file is ever reached.
#include "strata/core/conversation_ds.hpp"

// The fast path is compiled only in Windows CUDA builds (the CMake sets STRATA_DS_FASTPATH=1 there).
// Everywhere else this file compiles to a refusal, so the engine links the same on every backend.
#if defined(_WIN32) && defined(STRATA_DS_FASTPATH)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda_runtime.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>

#include "strata/kernels/ds_pages.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::core {
namespace {

constexpr uint64_t kSlotBytes = 32ull << 20;   // ninfer's 32-page lesson: keep staging bounded
constexpr uint32_t kSlots = 2;
constexpr uint32_t kMaxPageBytes = 64u << 10;  // kConversationPageBytes; the pool never writes a longer page
constexpr uint32_t kPagesPerSlot = (uint32_t)(kSlotBytes / kMaxPageBytes);

using FnGetFactory = HRESULT(WINAPI*)(REFIID, void**);
using FnSetConfig = HRESULT(WINAPI*)(DSTORAGE_CONFIGURATION const*);
using FnCreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
using FnCreateFactory = HRESULT(WINAPI*)(REFIID, void**);

bool ce(cudaError_t e, const char* what, std::string& error) {
    if (e == cudaSuccess) return true;
    error = std::string("disk fast path: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

std::wstring to_wide(const std::string& utf8) {
    if (utf8.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), w.data(), n);
    return w;
}

HMODULE load_side_by_side(const wchar_t* name) {
    wchar_t exe[MAX_PATH] = L"";
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    wchar_t* slash = wcsrchr(exe, L'\\');
    if (slash) {
        wcscpy_s(slash + 1, MAX_PATH - (size_t)(slash + 1 - exe), name);
        if (HMODULE m = LoadLibraryExW(exe, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) return m;   // the redist beside the engine
    }
    return LoadLibraryExW(name, nullptr, 0);
}

} // namespace

struct ConversationDs::Impl {
    HMODULE ds = nullptr, d3d = nullptr, dxgi = nullptr;
    IDStorageFactory* factory = nullptr;
    IDStorageQueue1* queue = nullptr;
    ID3D12Device* device = nullptr;
    ID3D12Resource* staging = nullptr;
    ID3D12Fence* fence = nullptr;
    cudaExternalMemory_t ext_mem = nullptr;
    cudaExternalSemaphore_t fence_sem = nullptr;
    void* d_staging = nullptr;                        // the ring, as a CUDA device pointer
    kernels::DsPageJob* d_jobs[kSlots] = {};          // per-slot job arrays (device)
    kernels::DsPageJob* h_jobs[kSlots] = {};          // per-slot job staging (pinned host)
    cudaEvent_t ev[kSlots] = {};
    cudaEvent_t ev_done = nullptr;
    cudaStream_t stream = nullptr;
    uint32_t* d_bad = nullptr;                        // device: any page mismatched
    uint32_t* h_bad = nullptr;                        // pinned read-back
    uint64_t fence_value = 0;
    uint64_t slot_fence[kSlots] = {};
    std::string describe = "directstorage staging=2x32MiB";

    ~Impl() {
        if (queue) { queue->Close(); queue->Release(); }
        if (fence_sem) cudaDestroyExternalSemaphore(fence_sem);
        if (ext_mem) cudaDestroyExternalMemory(ext_mem);
        for (uint32_t i = 0; i < kSlots; ++i) {
            if (d_jobs[i]) cudaFree(d_jobs[i]);
            if (h_jobs[i]) cudaFreeHost(h_jobs[i]);
            if (ev[i]) cudaEventDestroy(ev[i]);
        }
        if (ev_done) cudaEventDestroy(ev_done);
        if (stream) cudaStreamDestroy(stream);
        if (d_bad) cudaFree(d_bad);
        if (h_bad) cudaFreeHost(h_bad);
        if (staging) staging->Release();
        if (fence) fence->Release();
        if (device) device->Release();
        if (factory) factory->Release();
        // The DLLs outlive the COM objects they made nothing of: they stay loaded for the process, which
        // is a singleton fast path.  FreeLibrary during teardown can race the driver's worker threads.
        (void)ds; (void)d3d; (void)dxgi;
    }
};

std::unique_ptr<ConversationDs> ConversationDs::create(std::string& why) {
    why.clear();
    auto impl = std::make_unique<Impl>();
    auto refuse = [&](const std::string& r) -> std::unique_ptr<ConversationDs> { why = r; return nullptr; };

    // 1. the DirectStorage runtime (beside the engine first: the app ships the redist)
    impl->ds = load_side_by_side(L"dstorage.dll");
    if (!impl->ds) return refuse("dstorage.dll not found");
    auto getFactory = (FnGetFactory)GetProcAddress(impl->ds, "DStorageGetFactory");
    auto setConfig = (FnSetConfig)GetProcAddress(impl->ds, "DStorageSetConfiguration");
    if (!getFactory) return refuse("dstorage.dll has no DStorageGetFactory");
    if (setConfig) {
        DSTORAGE_CONFIGURATION cfg{};
        cfg.DisableTelemetry = TRUE;   // Strata sends no telemetry
        setConfig(&cfg);
    }
    if (FAILED(getFactory(__uuidof(IDStorageFactory), (void**)&impl->factory)) || !impl->factory)
        return refuse("DStorageGetFactory failed");

    // 2. CUDA's current device, and its LUID (CUDA 12.8+ reports it)
    int device_index = 0;
    if (cudaGetDevice(&device_index) != cudaSuccess) return refuse("no CUDA device");
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_index) != cudaSuccess) return refuse("CUDA device properties");
    const bool have_luid = prop.luid[0] || prop.luid[1] || prop.luid[2] || prop.luid[3] ||
                           prop.luid[4] || prop.luid[5] || prop.luid[6] || prop.luid[7];

    // 3. D3D12 on the SAME adapter (the reference host is single-adapter: no LUID match is needed there)
    impl->d3d = LoadLibraryW(L"d3d12.dll");
    impl->dxgi = LoadLibraryW(L"dxgi.dll");
    if (!impl->d3d || !impl->dxgi) return refuse("d3d12.dll or dxgi.dll unavailable");
    auto createDevice = (FnCreateDevice)GetProcAddress(impl->d3d, "D3D12CreateDevice");
    auto createDxgi = (FnCreateFactory)GetProcAddress(impl->dxgi, "CreateDXGIFactory1");
    if (!createDevice || !createDxgi) return refuse("D3D12CreateDevice or CreateDXGIFactory1 missing");
    IDXGIFactory1* dxgi_factory = nullptr;
    if (FAILED(createDxgi(__uuidof(IDXGIFactory1), (void**)&dxgi_factory))) return refuse("CreateDXGIFactory1");
    IDXGIAdapter1* adapter = nullptr, *solo = nullptr;
    for (UINT i = 0; dxgi_factory->EnumAdapters1(i, &adapter) == S_OK;) {
        DXGI_ADAPTER_DESC1 d1{};
        adapter->GetDesc1(&d1);
        const bool software = d1.VendorId == 0x1414 || (d1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE);
        const bool match = have_luid && memcmp(prop.luid, &d1.AdapterLuid, sizeof(LUID)) == 0;
        if (match) break;
        IDXGIAdapter1* cur = adapter;
        if (!software && !solo) solo = cur; else cur->Release();
        adapter = nullptr;
        ++i;
    }
    if (!adapter) adapter = solo; else if (solo) solo->Release();
    dxgi_factory->Release();
    if (!adapter) return refuse("no D3D12 adapter matches the CUDA device");
    const HRESULT dev_hr = createDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&impl->device);
    adapter->Release();
    if (FAILED(dev_hr) || !impl->device) return refuse("D3D12CreateDevice on the CUDA adapter failed");

    // 4. the staging ring: one shared default-heap buffer, two slots of kSlotBytes
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        rd.Width = kSlotBytes * kSlots;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(impl->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                         nullptr, __uuidof(ID3D12Resource), (void**)&impl->staging)))
            return refuse("staging buffer allocation failed");
        HANDLE shared = nullptr;
        if (FAILED(impl->device->CreateSharedHandle(impl->staging, nullptr, GENERIC_ALL, nullptr, &shared)))
            return refuse("staging shared handle failed");
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeD3D12Resource;
        hd.handle.win32.handle = shared;
        hd.size = kSlotBytes * kSlots;
        hd.flags = cudaExternalMemoryDedicated;
        const cudaError_t import = cudaImportExternalMemory(&impl->ext_mem, &hd);
        CloseHandle(shared);
        if (import != cudaSuccess) return refuse("CUDA could not import the staging buffer");
        cudaExternalMemoryBufferDesc bd{};
        bd.size = kSlotBytes * kSlots;
        if (cudaExternalMemoryGetMappedBuffer(&impl->d_staging, impl->ext_mem, &bd) != cudaSuccess)
            return refuse("CUDA could not map the staging buffer");
    }

    // 5. the shared fence, imported as a CUDA external semaphore; the queue that writes the ring
    {
        if (FAILED(impl->device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&impl->fence)))
            return refuse("fence allocation failed");
        HANDLE fh = nullptr;
        if (FAILED(impl->device->CreateSharedHandle(impl->fence, nullptr, GENERIC_ALL, nullptr, &fh)))
            return refuse("fence shared handle failed");
        cudaExternalSemaphoreHandleDesc sd{};
        sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        sd.handle.win32.handle = fh;
        const cudaError_t import = cudaImportExternalSemaphore(&impl->fence_sem, &sd);
        CloseHandle(fh);
        if (import != cudaSuccess) return refuse("CUDA could not import the fence");
        DSTORAGE_QUEUE_DESC qd{};
        qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        qd.Priority = DSTORAGE_PRIORITY_HIGH;
        qd.Name = "strata-conversation-disk";
        qd.Device = impl->device;
        if (FAILED(impl->factory->CreateQueue(&qd, __uuidof(IDStorageQueue1), (void**)&impl->queue)) || !impl->queue)
            return refuse("DirectStorage queue creation failed");
    }

    // 6. CUDA sidecar state: stream, per-slot job arrays and events, the mismatch flag
    if (!ce(cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking), "stream", why)) return nullptr;
    for (uint32_t i = 0; i < kSlots; ++i) {
        if (!ce(cudaMalloc((void**)&impl->d_jobs[i], kPagesPerSlot * sizeof(kernels::DsPageJob)), "jobs", why)) return nullptr;
        if (!ce(cudaHostAlloc((void**)&impl->h_jobs[i], kPagesPerSlot * sizeof(kernels::DsPageJob), 0), "jobs pinned", why)) return nullptr;
        if (!ce(cudaEventCreateWithFlags(&impl->ev[i], cudaEventDisableTiming), "event", why)) return nullptr;
    }
    if (!ce(cudaEventCreateWithFlags(&impl->ev_done, cudaEventDisableTiming), "done event", why)) return nullptr;
    if (!ce(cudaMalloc((void**)&impl->d_bad, sizeof(uint32_t)), "bad flag", why)) return nullptr;
    if (!ce(cudaHostAlloc((void**)&impl->h_bad, sizeof(uint32_t), 0), "bad pinned", why)) return nullptr;

    return std::unique_ptr<ConversationDs>(new ConversationDs(std::move(impl)));
}

ConversationDs::ConversationDs(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ConversationDs::~ConversationDs() = default;
uint64_t ConversationDs::staging_bytes() const { return kSlotBytes * kSlots; }
const std::string& ConversationDs::describe() const { return impl_->describe; }

bool ConversationDs::write(const std::string& pool_path, const std::vector<ConversationDsPage>& pages, std::string& error) {
    if (pages.empty()) return true;
    Impl& p = *impl_;
    for (const auto& page : pages)
        if (page.bytes == 0 || page.bytes > kMaxPageBytes || page.copy_bytes > page.bytes) {
            error = "disk fast path: page larger than the staging slot";
            return false;
        }
    IDStorageFile* file = nullptr;
    const std::wstring wpath = to_wide(pool_path);
    HRESULT open_hr = E_FAIL;
    if (wpath.empty() ||
        FAILED(open_hr = p.factory->OpenFile(wpath.c_str(), __uuidof(IDStorageFile), (void**)&file)) || !file) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "disk fast path: OpenFile failed for the pool (hr=%08lx)", (unsigned long) open_hr);
        error = buf;
        return false;
    }
    *p.h_bad = 0;
    if (!ce(cudaMemsetAsync(p.d_bad, 0, sizeof(uint32_t), p.stream), "clear flag", error)) {
        file->Close();
        file->Release();
        return false;
    }

    // Two-slot software pipeline: batch b's DirectStorage reads overlap batch b-1's verify/scatter.  A slot
    // is refilled only after its kernels are done (its event), so a slot's bytes are never overwritten while
    // the kernel still reads them, and the fence wait orders every kernel after its own DMA.
    uint32_t done = 0;   // pages consumed
    uint32_t batch = 0;
    bool ok = true;
    std::string err;
    while (done < pages.size() && ok) {
        const uint32_t slot = batch % kSlots;
        if (batch >= kSlots && cudaEventSynchronize(p.ev[slot]) != cudaSuccess)   // this slot's kernels finished
            ok = ce(cudaGetLastError(), "slot wait", err);
        if (!ok) break;
        const uint32_t n = (uint32_t)std::min<size_t>(pages.size() - done, kPagesPerSlot);
        kernels::DsPageJob* jobs = p.h_jobs[slot];
        for (uint32_t i = 0; i < n; ++i) {
            const ConversationDsPage& pg = pages[done + i];
            DSTORAGE_REQUEST r{};
            r.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            r.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            r.Source.File.Source = file;
            r.Source.File.Offset = pg.file_offset;
            r.Source.File.Size = pg.bytes;
            r.UncompressedSize = pg.bytes;
            r.Destination.Buffer.Resource = p.staging;
            r.Destination.Buffer.Offset = (UINT64)slot * kSlotBytes + (uint64_t)i * kMaxPageBytes;
            r.Destination.Buffer.Size = pg.bytes;
            p.queue->EnqueueRequest(&r);
            jobs[i].src = (const uint8_t*)p.d_staging + (uint64_t)slot * kSlotBytes + (uint64_t)i * kMaxPageBytes;
            jobs[i].dst = (uint8_t*)pg.dst;
            jobs[i].expect_hash = pg.expect_hash;
            jobs[i].hash_bytes = pg.bytes;
            jobs[i].copy_bytes = pg.copy_bytes;
        }
        const uint64_t signal = ++p.fence_value;
        p.queue->EnqueueSignal(p.fence, signal);
        p.queue->Submit();
        p.slot_fence[slot] = signal;
        if (!ce(cudaMemcpyAsync(p.d_jobs[slot], p.h_jobs[slot], n * sizeof(kernels::DsPageJob), cudaMemcpyHostToDevice, p.stream),
                "jobs upload", err)) {
            ok = false;
            break;
        }
        cudaExternalSemaphoreWaitParams wp{};
        wp.params.fence.value = signal;
        if (!ce(cudaWaitExternalSemaphoresAsync(&p.fence_sem, &wp, 1, p.stream), "fence wait", err)) {
            ok = false;
            break;
        }
        kernels::ds_verify_scatter(p.d_jobs[slot], n, p.d_bad, p.stream);
        if (!ce(cudaGetLastError(), "verify scatter", err)) {
            ok = false;
            break;
        }
        cudaEventRecord(p.ev[slot], p.stream);
        done += n;
        ++batch;
    }
    if (ok) {
        if (!ce(cudaMemcpyAsync(p.h_bad, p.d_bad, sizeof(uint32_t), cudaMemcpyDeviceToHost, p.stream), "flag readback", err))
            ok = false;
        else {
            cudaEventRecord(p.ev_done, p.stream);
            ok = ce(cudaEventSynchronize(p.ev_done), "final wait", err);
        }
    }
    if (ok && *p.h_bad) {
        error = "disk fast path: page hash mismatch (staging refused)";
        ok = false;
    }
    file->Close();
    file->Release();
    // A failed batch may leave DirectStorage writes in flight.  Nothing must be reset on the stream that
    // they could still touch (no device objects are reused after a failure): the fence value keeps
    // growing, and every later wait uses a strictly larger value, so a late write lands in a slot no
    // later kernel reads before its own, larger fence value has passed.
    if (!ok) {
        if (error.empty()) error = "disk fast path: queue failure";
        return false;
    }
    return true;
}

} // namespace strata::core

#else  // no fast path on this platform/build: create() refuses, nothing else compiles

#include <stdexcept>

namespace strata::core {
struct ConversationDs::Impl {};
ConversationDs::ConversationDs(std::unique_ptr<Impl>) {}
std::unique_ptr<ConversationDs> ConversationDs::create(std::string& why) {
    why = "the disk fast path is Windows + CUDA only";
    return nullptr;
}
ConversationDs::~ConversationDs() = default;
uint64_t ConversationDs::staging_bytes() const { return 0; }
const std::string& ConversationDs::describe() const {
    static const std::string none = "none";
    return none;
}
bool ConversationDs::write(const std::string&, const std::vector<ConversationDsPage>&, std::string&) { return false; }
} // namespace strata::core

#endif
