# Microsoft DirectStorage SDK (headers and x64 runtime)

Verbatim copies from the NuGet package `Microsoft.Direct3D.DirectStorage` version 1.3.0 (SDK version 300),
published by Microsoft under the MIT license (`LICENSE.txt` for the package, `LICENSE-CODE.txt` for the code,
`NOTICES.txt` for third-party notices; all kept unmodified).

- `include/dstorage.h`, `include/dstorageerr.h`: the public COM API. Included by
  `src/core/conversation_ds.cpp` only; the runtime is **not** link-bound: the DLLs are loaded with
  `LoadLibraryExW` at startup when `--conversation-cache-disk-fast-path` is given, so an engine build runs
  unchanged with no DirectStorage present (the disk tier stays on its portable path).
- `bin-x64/dstorage.dll`, `bin-x64/dstoragecore.dll`: the redist runtime (the two must be beside each
  other, and beside the engine, or on the `PATH`). `dstorage.dll` alone is the app-facing DLL;
  `dstoragecore.dll` does the queue work.

Nothing here is compiled into the default path. The fast path is opt-in, and the portable
`conversation_file.cpp` buffered reader remains the default and the fallback everywhere.
