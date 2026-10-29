// DirectStorage fast-path probe (Windows + CUDA), for measurement and capability checks only.
//
// It exercises the exact chain the disk tier's Phase 3 fast path needs, before any of it is wired into
// the engine: load dstorage.dll, create a D3D12 device on the SAME adapter as the CUDA device (LUID),
// read a file's bytes with a DirectStorage queue straight into a shared default-heap buffer, signal a
// shared fence, import the buffer into CUDA and read the bytes from CUDA, and wait on the fence inside
// a CUDA stream through the imported external semaphore.  It reports measured MB/s so a fast path can
// be accepted or rejected on numbers, not on a claim.  It changes nothing else; run it standalone:
//
//   ds_probe [file] [MiB] [chunk_MiB]
//
// With no file it writes a temp file of the given size filled with a deterministic pattern.
// Prints "PROBE PASS" or "PROBE FAIL: <reason>"; exit code matches.
#if !defined(_WIN32)
int main() { puts("PROBE FAIL: Windows only"); return 2; }
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool g_fail = false;

// loaded lazily so a missing DLL prints a clean PROBE FAIL line instead of a loader error
HMODULE g_ds = nullptr, g_d3d = nullptr, g_dxgi = nullptr;
using FnGetFactory = HRESULT(WINAPI*)(REFIID, void**);
using FnSetConfig = HRESULT(WINAPI*)(DSTORAGE_CONFIGURATION const*);
using FnCreateDevice = HRESULT(WINAPI*)(IUnknown*, int, REFIID, void**);
using FnCreateFactory = HRESULT(WINAPI*)(REFIID, void**);

void note(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
}
bool fail(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("PROBE FAIL: ");
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
    g_fail = true;
    return false;
}

bool load_dll(HMODULE& into, const wchar_t* name, const wchar_t* side_by_side) {
    if (side_by_side) into = LoadLibraryExW(side_by_side, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!into) into = LoadLibraryExW(name, nullptr, 0);
    return into != nullptr;
}
}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : nullptr;
    const uint64_t mib = argc > 2 ? std::atoll(argv[2]) : 256;
    const uint64_t chunk = (argc > 3 ? std::atoll(argv[3]) : 32) << 20;
    std::string temp;
    uint64_t size = 0;
    if (!path) {
        // a deterministic pattern: every 64 KiB page differs from its neighbours
        char dir[MAX_PATH];
        GetTempPathA(MAX_PATH, dir);
        temp = std::string(dir) + "strata_ds_probe.bin";
        path = temp.c_str();
        size = mib << 20;
        std::vector<uint8_t> page(64u << 10);
        FILE* f = fopen(path, "wb");
        if (!f) { fail("cannot write %s", path); return 1; }
        for (uint64_t off = 0; off < size; off += page.size()) {
            uint64_t p = off / 8 + 0x9E3779B97F4A7C15ull;
            for (size_t i = 0; i < page.size(); i += 8) { std::memcpy(page.data() + i, &p, 8); p = p * 6364136223846793005ull + 1442695040888963407ull; }
            fwrite(page.data(), 1, page.size(), f);
        }
        fclose(f);
        note("wrote %s (%llu MiB)\n", path, (unsigned long long)(size >> 20));
    } else {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) { fail("cannot open %s", path); return 1; }
        LARGE_INTEGER li;
        GetFileSizeEx(h, &li);
        size = (uint64_t)li.QuadPart;
        CloseHandle(h);
    }
    if (size < (64u << 20)) { fail("file too small for a meaningful probe"); return 1; }

    // --- CUDA up ---
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(0) != cudaSuccess) { fail("cudaSetDevice"); return 1; }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    note("cuda device 0: %s\n", prop.name);

    // --- DirectStorage runtime ---
    wchar_t exe_side[MAX_PATH] = L"";
    {   // look beside this exe first (the redist ships with the engine)
        wchar_t exe[MAX_PATH];
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        wchar_t* slash = wcsrchr(exe, L'\\');
        if (slash) { wcscpy_s(slash + 1, MAX_PATH - (slash + 1 - exe), L"dstorage.dll"); load_dll(g_ds, L"dstorage.dll", exe); }
    }
    if (!g_ds && !load_dll(g_ds, L"dstorage.dll", nullptr)) { fail("dstorage.dll not found (install the DirectStorage redist beside the exe)"); return 1; }
    (void)exe_side;
    auto getFactory = (FnGetFactory)GetProcAddress(g_ds, "DStorageGetFactory");
    auto setConfig = (FnSetConfig)GetProcAddress(g_ds, "DStorageSetConfiguration");
    if (!getFactory) { fail("DStorageGetFactory not exported"); return 1; }
    if (setConfig) {   // Strata does not send DirectStorage telemetry
        DSTORAGE_CONFIGURATION cfg{};
        cfg.DisableTelemetry = TRUE;
        setConfig(&cfg);
    }
    IDStorageFactory* factory = nullptr;
    if (FAILED(getFactory(__uuidof(IDStorageFactory), (void**)&factory)) || !factory) { fail("DStorageGetFactory"); return 1; }
    note("dstorage.dll loaded, factory ok\n");

    // --- D3D12 on the CUDA adapter (matched by the LUID cudaDeviceProp reports) ---
    if (!load_dll(g_d3d, L"d3d12.dll", nullptr)) { fail("d3d12.dll"); return 1; }
    auto createDevice = (FnCreateDevice)GetProcAddress(g_d3d, "D3D12CreateDevice");
    if (!createDevice) { fail("D3D12CreateDevice"); return 1; }
    if (!load_dll(g_dxgi, L"dxgi.dll", nullptr)) { fail("dxgi.dll"); return 1; }
    auto createFactory = (FnCreateFactory)GetProcAddress(g_dxgi, "CreateDXGIFactory1");
    if (!createFactory) { fail("CreateDXGIFactory1"); return 1; }
    IDXGIFactory1* dxgi = nullptr;
    if (FAILED(createFactory(__uuidof(IDXGIFactory1), (void**)&dxgi))) { fail("CreateDXGIFactory1"); return 1; }
    const bool have_luid = prop.luid[0] || prop.luid[1] || prop.luid[2] || prop.luid[3] ||
                           prop.luid[4] || prop.luid[5] || prop.luid[6] || prop.luid[7];
    IDXGIAdapter1* adapter = nullptr;
    IDXGIAdapter1* solo = nullptr;   // fallback: a machine with one hardware adapter
    for (UINT i = 0; dxgi->EnumAdapters1(i, &adapter) == S_OK;) {
        DXGI_ADAPTER_DESC1 d1{};
        adapter->GetDesc1(&d1);
        const bool software = d1.VendorId == 0x1414 || (d1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE);
        const bool match = have_luid && memcmp(prop.luid, &d1.AdapterLuid, sizeof(LUID)) == 0;
        if (match) { note("dxgi adapter %u: %S\n", i, d1.Description); break; }   // adapter: kept
        IDXGIAdapter1* cur = adapter;                                              // non-match below: released or kept as solo
        if (!software && !solo) solo = cur;                                        // keeps this object's reference
        else cur->Release();
        adapter = nullptr;
        ++i;
    }
    if (!adapter) adapter = solo;   // single-adapter machine (the reference host): no LUID match needed
    solo = nullptr;
    if (!adapter) { fail("no DXGI adapter matches the CUDA device"); dxgi->Release(); return 1; }
    ID3D12Device* device = nullptr;
    if (FAILED(createDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&device))) {
        adapter->Release(); dxgi->Release(); fail("D3D12CreateDevice"); return 1;
    }
    adapter->Release(); dxgi->Release();
    note("d3d12 device ok on the cuda adapter\n");

    // --- a shared default-heap committed buffer: D3D12 writes it, CUDA reads it (GENERIC_ALL handle) ---
    const uint64_t ring = 2, slot = chunk;
    const uint64_t bytes = slot * ring;
    ID3D12Resource* heap_res = nullptr;
    HANDLE shared = nullptr;
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        rd.Width = bytes;
        rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, __uuidof(ID3D12Resource), (void**)&heap_res);
        if (FAILED(hr) || !heap_res) { fail("CreateCommittedResource(%llu MiB): hr=%08lx", (unsigned long long)(bytes >> 20), (unsigned long)hr); return 1; }
        hr = device->CreateSharedHandle(heap_res, nullptr, GENERIC_ALL, nullptr, &shared);
        if (FAILED(hr) || !shared) { fail("CreateSharedHandle(resource): hr=%08lx", (unsigned long)hr); return 1; }
    }
    note("shared default-heap buffer: %llu MiB at GPU VA %llx\n", (unsigned long long)(bytes >> 20),
         (unsigned long long)heap_res->GetGPUVirtualAddress());

    // --- a shared fence, imported as a CUDA external semaphore (stream-ordered completion) ---
    ID3D12Fence* fence = nullptr;
    cudaExternalSemaphore_t fence_sem = nullptr;
    uint64_t fence_value = 0;
    {
        HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&fence);
        if (FAILED(hr) || !fence) { fail("CreateFence: hr=%08lx", (unsigned long)hr); return 1; }
        HANDLE fh = nullptr;
        hr = device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &fh);
        if (FAILED(hr) || !fh) { fail("CreateSharedHandle(fence): hr=%08lx", (unsigned long)hr); return 1; }
        cudaExternalSemaphoreHandleDesc sd{};
        sd.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        sd.handle.win32.handle = fh;
        if (cudaImportExternalSemaphore(&fence_sem, &sd) != cudaSuccess) { CloseHandle(fh); fail("cudaImportExternalSemaphore(fence)"); return 1; }
        CloseHandle(fh);
    }
    note("shared fence imported as a CUDA external semaphore\n");

    // --- import the buffer into CUDA ---
    void* dptr = nullptr;
    cudaExternalMemory_t ext = nullptr;
    {
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeD3D12Resource;
        hd.handle.win32.handle = shared;
        hd.size = bytes;
        hd.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&ext, &hd) != cudaSuccess) { fail("cudaImportExternalMemory"); return 1; }
        cudaExternalMemoryBufferDesc bd{};
        bd.offset = 0; bd.size = bytes;
        if (cudaExternalMemoryGetMappedBuffer(&dptr, ext, &bd) != cudaSuccess) { fail("cudaExternalMemoryGetMappedBuffer"); return 1; }
    }
    note("imported into CUDA at %p\n", dptr);

    // --- DirectStorage: file -> staging buffer, chunked through a two-slot ring ---
    wchar_t wpath[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, MAX_PATH);
    IDStorageFile* file = nullptr;
    if (FAILED(factory->OpenFile(wpath, __uuidof(IDStorageFile), (void**)&file)) || !file) { fail("OpenFile"); return 1; }
    IDStorageQueue1* queue = nullptr;
    {
        DSTORAGE_QUEUE_DESC qd{};
        qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        qd.Priority = DSTORAGE_PRIORITY_HIGH;
        qd.Name = "strata-ds-probe";
        qd.Device = device;
        if (FAILED(factory->CreateQueue(&qd, __uuidof(IDStorageQueue1), (void**)&queue)) || !queue) { fail("CreateQueue"); return 1; }
    }

    // timing: read the whole file in ring-slot-sized passes; pass 0 completes on the D3D12 fence via a CPU
    // event, later passes through cudaWaitExternalSemaphoresAsync on the CUDA stream (the production order)
    LARGE_INTEGER qpc{};
    QueryPerformanceFrequency(&qpc);
    auto now_ms = [&qpc]() { LARGE_INTEGER c; QueryPerformanceCounter(&c); return 1000.0 * (double)c.QuadPart / (double)qpc.QuadPart; };
    uint64_t total = 0;
    double best = 0, worst = 1e30, sum_ms = 0;
    uint32_t passes = 0;
    for (uint64_t base = 0; base < size; base += bytes) {
        uint64_t pass_bytes = bytes < size - base ? bytes : size - base;
        auto t0 = now_ms();
        for (uint64_t off = 0; off < pass_bytes; off += chunk) {
            uint64_t n = chunk < pass_bytes - off ? chunk : pass_bytes - off;
            DSTORAGE_REQUEST r{};
            r.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            r.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            r.Source.File.Source = file;
            r.Source.File.Offset = base + off;
            r.Source.File.Size = (UINT32)n;
            r.UncompressedSize = (UINT32)n;
            r.Destination.Buffer.Resource = heap_res;
            r.Destination.Buffer.Offset = off % bytes;
            r.Destination.Buffer.Size = (UINT32)n;
            queue->EnqueueRequest(&r);
        }
        const uint64_t signal = ++fence_value;
        queue->EnqueueSignal(fence, signal);
        queue->Submit();
        if (passes == 0) {
            HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!done || FAILED(fence->SetEventOnCompletion(signal, done))) { if (done) CloseHandle(done); fail("SetEventOnCompletion"); return 1; }
            const DWORD w = WaitForSingleObject(done, 60000);
            CloseHandle(done);
            if (w != WAIT_OBJECT_0) { fail("wait for fence: %lu", w); return 1; }
        } else {   // the production ordering: wait on the fence INSIDE the CUDA stream, then launch D2D work
            cudaExternalSemaphoreWaitParams wp{};
            wp.params.fence.value = signal;
            if (cudaWaitExternalSemaphoresAsync(&fence_sem, &wp, 1, nullptr) != cudaSuccess) { fail("cudaWaitExternalSemaphoresAsync"); return 1; }
            void* scratch = nullptr;
            if (cudaMalloc(&scratch, slot) != cudaSuccess) { fail("cudaMalloc"); return 1; }
            if (cudaMemcpyAsync(scratch, dptr, slot, cudaMemcpyDeviceToDevice, nullptr) != cudaSuccess) { fail("D2D after semaphore"); return 1; }
            if (cudaDeviceSynchronize() != cudaSuccess) { fail("sync after semaphore D2D"); return 1; }
            cudaFree(scratch);
        }
        double ms = now_ms() - t0;
        double mib_s = ms > 0 ? (double)pass_bytes / (1048576.0 * ms / 1000.0) : 0;
        total += pass_bytes; sum_ms += ms; ++passes;
        if (mib_s > best) best = mib_s;
        if (mib_s < worst) worst = mib_s;
    }
    note("dstorage dma: %.0f MiB in %.0f ms avg %.0f MiB/s (best %.0f, worst %.0f)\n",
         (double)total / 1048576.0, sum_ms / passes, (double)total / (1048576.0 * sum_ms / 1000.0), best, worst);

    // --- CUDA: read the imported buffer (what the verify/scatter kernel touches) ---
    {
        const uint64_t span = slot < size ? slot : size;
        void* scratch = nullptr;
        if (cudaMalloc(&scratch, span) != cudaSuccess) { fail("cudaMalloc"); return 1; }
        std::vector<double> rates;
        for (int pass = 0; pass < 3; ++pass) {
            auto p0 = now_ms();
            if (cudaMemcpyAsync(scratch, dptr, span, cudaMemcpyDeviceToDevice, nullptr) != cudaSuccess) { fail("D2D from imported"); return 1; }
            if (cudaDeviceSynchronize() != cudaSuccess) { fail("sync"); return 1; }
            double ms = now_ms() - p0;
            rates.push_back(ms > 0.001 ? (double)span / (1048576.0 * ms / 1000.0) : 0.0);
        }
        note("imported-buffer D2D: %.0f / %.0f / %.0f MiB/s\n", rates[0], rates[1], rates[2]);
        // correctness: the bytes CUDA sees must equal the file bytes.  After the last pass the ring holds
        // the FINAL pass's bytes: ring offset 0 is the file offset where that pass started.
        const uint64_t last_base = ((size - 1) / bytes) * bytes;
        std::vector<uint8_t> host(1u << 20);
        std::vector<uint8_t> want(1u << 20);
        FILE* f = fopen(path, "rb");
        if (!f) { fail("reopen probe file"); return 1; }
        if (fseek(f, (long)last_base, SEEK_SET) != 0 || fread(want.data(), 1, want.size(), f) != want.size()) {
            fclose(f); fail("read probe file at %llu", (unsigned long long)last_base); return 1;
        }
        fclose(f);
        if (cudaMemcpy(host.data(), dptr, host.size(), cudaMemcpyDeviceToHost) != cudaSuccess) { fail("D2H from imported"); return 1; }
        if (memcmp(host.data(), want.data(), want.size()) != 0) { fail("imported bytes differ from the file"); return 1; }
        note("first 1 MiB through the imported buffer: MATCHES the file\n");
        cudaFree(scratch);
    }

    queue->Close(); queue->Release();
    file->Close(); file->Release();
    factory->Release();
    if (cudaDestroyExternalMemory(ext) != cudaSuccess) note("warn: DestroyExternalMemory\n");
    if (cudaDestroyExternalSemaphore(fence_sem) != cudaSuccess) note("warn: DestroyExternalSemaphore\n");
    fence->Release();
    heap_res->Release();
    device->Release();
    printf(g_fail ? "PROBE FAIL\n" : "PROBE PASS\n");
    return g_fail ? 1 : 0;
}
#endif
