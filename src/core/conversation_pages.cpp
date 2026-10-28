#include "strata/core/conversation_pages.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "the disk cache page pool is little-endian only");

namespace strata::core {
namespace {

namespace fs = std::filesystem;

constexpr char kPoolMagic[8] = {'S', 'T', 'R', 'P', 'A', 'G', 'E', '1'};
constexpr char kIndexMagic[8] = {'S', 'T', 'R', 'P', 'I', 'D', 'X', '1'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kPageHeaderMagic = 0x31454750;   // "PGE1"
constexpr uint64_t kPoolHeaderBytes = 4096;
constexpr uint64_t kPageHeaderBytes = 16;
constexpr uint32_t kMaxPages = 32u << 20;           // the index's count is bounded before it is read

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
uint64_t ms_since(const std::chrono::steady_clock::time_point& start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start).count();
}

#ifdef _WIN32
bool wide(const std::string& s, std::wstring& w) {
    w.clear();
    if (s.empty()) return true;
    if (s.size() > (size_t) INT_MAX) return false;
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return false;
    w.assign((size_t) n, L'\0');
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n) == n;
}
std::string last_error(DWORD code = GetLastError()) {
    char msg[256] = {};
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, msg,
                                   sizeof msg, nullptr);
    std::string s(msg, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}
std::FILE* open_file(const std::string& path, const wchar_t* mode) {
    std::wstring w;
    if (!wide(path, w)) return nullptr;
    return _wfopen(w.c_str(), mode);
}
bool file_size(const std::string& path, uint64_t& out) {
    std::wstring w;
    if (!wide(path, w)) return false;
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExW(w.c_str(), GetFileExInfoStandard, &a)) return false;
    if (a.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) return false;
    out = ((uint64_t) a.nFileSizeHigh << 32) | a.nFileSizeLow;
    return true;
}
bool erase_file(const std::string& path, std::string& error) {
    std::wstring w;
    if (!wide(path, w)) { error = "invalid path"; return false; }
    if (DeleteFileW(w.c_str())) return true;
    const DWORD code = GetLastError();
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) return true;
    error = last_error(code);
    return false;
}
bool replace_file(const std::string& from, const std::string& to, std::string& error) {
    std::wstring a, b;
    if (!wide(from, a) || !wide(to, b)) { error = "invalid path"; return false; }
    if (MoveFileExW(a.c_str(), b.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
    error = last_error();
    return false;
}
bool flush_file(std::FILE* f) { return _commit(_fileno(f)) == 0; }
int seek_file(std::FILE* f, uint64_t off) { return _fseeki64(f, (__int64) off, SEEK_SET); }
uint64_t tell_file(std::FILE* f) { return (uint64_t) _ftelli64(f); }
#else
std::FILE* open_file(const std::string& path, const wchar_t* mode) {
    // The modes used here are pure ASCII.
    std::string m;
    for (const wchar_t* p = mode; *p; ++p) m.push_back((char) *p);
    return std::fopen(path.c_str(), m.c_str());
}
bool file_size(const std::string& path, uint64_t& out) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
    out = (uint64_t) st.st_size;
    return true;
}
bool erase_file(const std::string& path, std::string& error) {
    if (::unlink(path.c_str()) == 0) return true;
    if (errno == ENOENT) return true;
    error = std::strerror(errno);
    return false;
}
bool replace_file(const std::string& from, const std::string& to, std::string& error) {
    if (::rename(from.c_str(), to.c_str()) != 0) { error = std::strerror(errno); return false; }
    fs::path dir = fs::path(to).parent_path();
    if (!dir.empty()) {
        const int fd = ::open(dir.string().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd >= 0) { ::fsync(fd); ::close(fd); }
    }
    return true;
}
bool flush_file(std::FILE* f) { return ::fsync(::fileno(f)) == 0; }
int seek_file(std::FILE* f, uint64_t off) { return ::fseeko(f, (off_t) off, SEEK_SET); }
uint64_t tell_file(std::FILE* f) { return (uint64_t) ::ftello(f); }
#endif

bool read_exact(std::FILE* f, void* p, size_t n) { return n == 0 || std::fread(p, 1, n, f) == n; }
bool write_exact(std::FILE* f, const void* p, size_t n) { return n == 0 || std::fwrite(p, 1, n, f) == n; }

bool truncate_file(std::FILE* f, uint64_t size) {
#ifdef _WIN32
    return _chsize_s(_fileno(f), (__int64) size) == 0;
#else
    return ::ftruncate(::fileno(f), (off_t) size) == 0;
#endif
}

void put_u32(std::string& b, uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); }
void put_u64(std::string& b, uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); }
void put_bytes(std::string& b, const void* p, size_t n) { b.append(static_cast<const char*>(p), n); }

// ---- pages.bin ---------------------------------------------------------------------------------------------------
// A small RAII file with the operations both platforms share.
class File {
public:
    File() = default;
    ~File() { close(); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    bool open(const std::string& path, const wchar_t* mode, std::string& error) {
        close();
        path_ = path;
        f_ = open_file(path, mode);
        if (!f_) { error = "cannot open " + path; return false; }
#ifndef _WIN32
        // A write creates or truncates the file: keep it private to the user, like the session files (0600).
        bool writes = false;
        for (const wchar_t* p = mode; *p; ++p)
            if (*p == L'w' || *p == L'+') writes = true;
        if (writes) ::fchmod(::fileno(f_), 0600);
#endif
        return true;
    }
    void close() {
        if (f_) { std::fclose(f_); f_ = nullptr; }
    }
    std::FILE* get() const { return f_; }
    bool flush(std::string& error) {
        if (!f_) { error = "the file is closed"; return false; }
        if (std::fflush(f_) != 0 || !flush_file(f_)) { error = "cannot flush " + path_; return false; }
        return true;
    }
    bool seek(uint64_t at, std::string& error) {
        if (!f_ || seek_file(f_, at) != 0) { error = "cannot seek in " + path_; return false; }
        return true;
    }
    bool read(void* p, size_t n, std::string& error) {
        if (!f_ || !read_exact(f_, p, n)) { error = "cannot read " + path_; return false; }
        return true;
    }
    bool write(const void* p, size_t n, std::string& error) {
        if (!f_ || !write_exact(f_, p, n)) { error = "cannot write " + path_; return false; }
        return true;
    }
    uint64_t tell() const { return f_ ? tell_file(f_) : 0; }
    const std::string& path() const { return path_; }

private:
    std::FILE* f_ = nullptr;
    std::string path_;
};

uint64_t pool_page_offset_after(uint64_t offset, uint32_t bytes) {
    return align_up(offset + bytes + kPageHeaderBytes, kConversationPageAlign);
}

// The pool header: magic, version, generation, reserved, checksum of the first 32 bytes.
std::string pool_header(uint64_t generation) {
    std::string b;
    b.append(kPoolMagic, sizeof kPoolMagic);
    put_u32(b, kFormatVersion);
    put_u64(b, generation);
    put_u32(b, 0);
    b.append(32 - b.size(), '\0');
    put_u64(b, session_hash64(b.data(), b.size(), 0));
    b.append(kPoolHeaderBytes - b.size(), '\0');
    return b;
}

bool parse_pool_header(const char* data, uint64_t size, uint64_t& generation, std::string& error) {
    if (size < kPoolHeaderBytes) { error = "the page pool is shorter than its header"; return false; }
    if (std::memcmp(data, kPoolMagic, sizeof kPoolMagic) != 0) { error = "not a page pool"; return false; }
    uint32_t version = 0;
    std::memcpy(&version, data + 8, 4);
    if (version != kFormatVersion) { error = "unknown page pool version " + std::to_string(version); return false; }
    std::memcpy(&generation, data + 12, 8);
    uint64_t checksum = 0;
    std::memcpy(&checksum, data + 32, 8);
    if (checksum != session_hash64(data, 32, 0)) { error = "page pool header checksum mismatch"; return false; }
    return true;
}

// ---- pages.idx ---------------------------------------------------------------------------------------------------
struct IndexEntry {
    uint64_t hash = 0;
    uint32_t bytes = 0;
    uint64_t offset = 0;
};

std::string encode_index(uint64_t generation, const std::vector<IndexEntry>& entries) {
    std::string b;
    b.append(kIndexMagic, sizeof kIndexMagic);
    put_u32(b, kFormatVersion);
    put_u64(b, generation);
    put_u64(b, entries.size());
    b.reserve(b.size() + entries.size() * 24 + 8);
    for (const auto& e : entries) {
        put_u64(b, e.hash);
        put_u32(b, e.bytes);
        put_u32(b, 0);
        put_u64(b, e.offset);
    }
    put_u64(b, session_hash64(b.data(), b.size(), 0));
    return b;
}

bool decode_index(const std::string& bytes, uint64_t& generation, std::vector<IndexEntry>& out, std::string& error) {
    if (bytes.size() < 8 + 4 + 8 + 8 + 8) { error = "the page index is truncated"; return false; }
    if (std::memcmp(bytes.data(), kIndexMagic, sizeof kIndexMagic) != 0) { error = "not a page index"; return false; }
    uint32_t version = 0;
    std::memcpy(&version, bytes.data() + 8, 4);
    if (version != kFormatVersion) { error = "unknown page index version"; return false; }
    std::memcpy(&generation, bytes.data() + 12, 8);
    uint64_t count = 0;
    std::memcpy(&count, bytes.data() + 20, 8);
    if (count > kMaxPages || count > (bytes.size() - 36) / 24 || bytes.size() != 36 + count * 24) {
        error = "the page index has an invalid length";
        return false;
    }
    uint64_t checksum = 0;
    std::memcpy(&checksum, bytes.data() + bytes.size() - 8, 8);
    if (checksum != session_hash64(bytes.data(), bytes.size() - 8, 0)) {
        error = "page index checksum mismatch";
        return false;
    }
    out.resize((size_t) count);
    const char* p = bytes.data() + 28;
    for (auto& e : out) {
        std::memcpy(&e.hash, p, 8);
        std::memcpy(&e.bytes, p + 8, 4);
        std::memcpy(&e.offset, p + 16, 8);
        p += 24;
    }
    return true;
}

bool read_small_file(const std::string& path, std::string& out, std::string& error) {
    uint64_t size = 0;
    if (!file_size(path, size)) { error = "the file does not exist"; return false; }
    if (size > (512u << 20)) { error = "the file is too large"; return false; }
    std::FILE* f = open_file(path, L"rb");
    if (!f) { error = "cannot open " + path; return false; }
    out.resize((size_t) size);
    const bool ok = size == 0 || std::fread(out.data(), 1, (size_t) size, f) == size;
    std::fclose(f);
    if (!ok) { error = "cannot read " + path; return false; }
    return true;
}

bool write_atomic(const std::string& path, const std::string& bytes, const std::string& tmp, std::string& error) {
    {
        File f;
        if (!f.open(tmp, L"wb", error)) return false;
        if (!f.write(bytes.data(), bytes.size(), error)) { f.close(); erase_file(tmp, error); return false; }
        if (!f.flush(error)) { f.close(); erase_file(tmp, error); return false; }
        f.close();
    }
    if (!replace_file(tmp, path, error)) { erase_file(tmp, error); return false; }
    return true;
}

} // namespace

// ---- ConversationPagePool::Impl ----------------------------------------------------------------------------------
struct ConversationPagePool::Impl {
    File pool;
    mutable File reader;
    std::string pool_path, index_path, tmp_path;
    uint64_t generation = 0, next_offset = kPoolHeaderBytes;
    uint64_t pool_bytes = 0;
    uint64_t written_pages = 0, written_bytes = 0, reused_pages = 0, reused_bytes = 0;
    uint64_t compactions = 0, compacted_bytes = 0, last_compaction_ms = 0, last_compacted_bytes = 0;
    std::vector<uint8_t> page_buffer;

    bool write_page(File& f, uint64_t hash, const void* data, uint32_t bytes, uint64_t offset, std::string& error) {
        // The next page's 16-byte header lives in this record's padding, so the gap must exist.
        const uint64_t header_at = offset >= kPageHeaderBytes ? offset - kPageHeaderBytes : 0;
        if (!f.seek(header_at, error)) return false;
        uint8_t h[kPageHeaderBytes] = {};
        std::memcpy(h, &kPageHeaderMagic, 4);
        std::memcpy(h + 4, &bytes, 4);
        std::memcpy(h + 8, &hash, 8);
        if (!f.write(h, sizeof h, error)) return false;
        if (!f.seek(offset, error)) return false;
        if (!f.write(data, bytes, error)) return false;
        return true;
    }
};

ConversationPagePool::ConversationPagePool(std::string root)
    : impl_(new Impl), pool_path_(root + "/pages.bin"), index_path_(root + "/pages.idx"),
      tmp_path_(root + "/tmp") {
    impl_->pool_path = pool_path_;
    impl_->index_path = index_path_;
    impl_->tmp_path = tmp_path_;
}

ConversationPagePool::~ConversationPagePool() = default;

bool ConversationPagePool::open(bool writable, std::string& error) {
    error.clear();
    writable_ = writable;
    std::error_code ec;
    fs::create_directories(fs::path(tmp_path_), ec);
    if (ec) { error = "cannot create the page pool's temporary folder: " + ec.message(); return false; }
#ifndef _WIN32
    {
        std::error_code pe;
        fs::permissions(fs::path(tmp_path_), fs::perms::owner_all, fs::perm_options::replace, pe);
    }
#endif

    uint64_t size = 0;
    const bool exists = file_size(pool_path_, size);
    if (!exists) {
        if (!writable) { return true; }   // no pool yet: nothing to read
        // Create it empty: the header is written now, so a crash before the first page still leaves a valid pool.
        File f;
        if (!f.open(pool_path_, L"w+b", error)) return false;
        const std::string h = pool_header(1);
        if (!f.write(h.data(), h.size(), error) || !f.flush(error)) {
            f.close();
            erase_file(pool_path_, error);
            return false;
        }
        f.close();
        size = kPoolHeaderBytes;
    }
    if (size < kPoolHeaderBytes) { error = "the page pool is shorter than its header"; return false; }

    File f;
    if (!f.open(pool_path_, writable ? L"r+b" : L"rb", error)) return false;
    char head[kPoolHeaderBytes] = {};
    if (!f.seek(0, error) || !f.read(head, sizeof head, error)) return false;
    if (!parse_pool_header(head, sizeof head, generation_, error)) return false;
    impl_->generation = generation_;
    impl_->pool_bytes = size;

    // The index is a cache.  Load it only when it names this generation and its offsets are inside the pool.
    bool loaded = false;
    std::string bytes, why;
    if (read_small_file(index_path_, bytes, why)) {
        uint64_t gen = 0;
        std::vector<IndexEntry> entries;
        if (decode_index(bytes, gen, entries, why) && gen == generation_) {
            bool ok = true;
            for (const auto& e : entries) {
                if (e.bytes == 0 || e.bytes > kConversationPageBytes || e.offset % kConversationPageAlign != 0 ||
                    e.offset < kPoolHeaderBytes || e.offset > size || e.bytes > size - e.offset) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                index_.clear();
                index_.reserve(entries.size());
                for (const auto& e : entries) index_[Key{e.hash, e.bytes}] = Page{e.offset, e.bytes, 0};
                loaded = true;
            }
        }
    }
    if (!loaded) {
        // Missing, stale or corrupt: rebuild by scanning the pool.  A torn record at the end is ignored.
        index_.clear();
        uint64_t at = kPoolHeaderBytes;
        while (at <= size && size - at >= kPageHeaderBytes) {
            uint8_t h[kPageHeaderBytes] = {};
            if (!f.seek(at - kPageHeaderBytes, error) || !f.read(h, sizeof h, error)) break;
            uint32_t magic = 0, n = 0;
            uint64_t hash = 0;
            std::memcpy(&magic, h, 4);
            std::memcpy(&n, h + 4, 4);
            std::memcpy(&hash, h + 8, 8);
            if (magic != kPageHeaderMagic || n == 0 || n > kConversationPageBytes || n > size - at) break;
            std::vector<uint8_t> data(n);
            if (!f.seek(at, error) || !f.read(data.data(), n, error)) break;
            if (session_hash64(data.data(), n, 0) != hash) break;
            index_[Key{hash, n}] = Page{at, n, 0};
            at = pool_page_offset_after(at, n);
        }
        if (writable) {
            std::vector<IndexEntry> entries;
            entries.reserve(index_.size());
            for (const auto& [k, p] : index_) entries.push_back({k.hash, k.bytes, p.offset});
            const std::string idx = encode_index(generation_, entries);
            if (!write_atomic(index_path_, idx, tmp_path_ + "/pages.idx.tmp", why))
                return false;   // the pool is readable, but a save could not trust its index
        }
    }
    // The append position is after the last indexed page (not the file size: the last 16 bytes of a full page are
    // the next record's header space).  Anything past it is an orphan from an interrupted append; the writable pool
    // drops it so a later scan cannot read a torn record as a page.
    uint64_t next = kPoolHeaderBytes;
    for (const auto& [k, p] : index_)
        next = std::max(next, pool_page_offset_after(p.offset, p.bytes));
    impl_->next_offset = next;
    if (writable && size > next) {
        if (!impl_->pool.get() && !impl_->pool.open(pool_path_, L"r+b", error)) return false;
        if (!truncate_file(impl_->pool.get(), next)) {
            error = "cannot trim the page pool's unfinished tail";
            return false;
        }
        impl_->pool_bytes = next;
    } else {
        impl_->pool_bytes = size;
    }
    return true;
}
bool ConversationPagePool::lookup(uint64_t hash, uint32_t bytes, Page& out) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(Key{hash, bytes});
    if (it == index_.end()) return false;
    out = it->second;
    return true;
}

bool ConversationPagePool::read(uint64_t hash, uint32_t bytes, uint64_t offset, void* dst, std::string& error) const {
    if (bytes == 0 || bytes > kConversationPageBytes || offset % kConversationPageAlign != 0) {
        error = "the page reference is invalid";
        return false;
    }
    if (!impl_->reader.get() && !impl_->reader.open(pool_path_, L"rb", error)) return false;
    if (!impl_->reader.seek(offset, error) || !impl_->reader.read(dst, bytes, error)) return false;
    if (session_hash64(dst, bytes, 0) != hash) { error = "page checksum mismatch"; return false; }
    return true;
}

bool ConversationPagePool::append(uint64_t hash, const void* data, uint32_t bytes, ConversationPageRef& ref,
                                  bool& wrote, std::string& error) {
    wrote = false;
    ref = ConversationPageRef{hash, bytes};
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(Key{hash, bytes});
    if (it != index_.end()) {
        ++impl_->reused_pages;
        impl_->reused_bytes += bytes;
        return true;
    }
    if (!impl_->pool.get()) {
        if (!impl_->pool.open(pool_path_, L"r+b", error)) return false;
    }
    const uint64_t at = impl_->next_offset;
    if (!impl_->write_page(impl_->pool, hash, data, bytes, at, error)) return false;
    index_[Key{hash, bytes}] = Page{at, bytes, 0};
    impl_->next_offset = pool_page_offset_after(at, bytes);
    impl_->pool_bytes = std::max(impl_->pool_bytes, at + bytes);
    ++impl_->written_pages;
    impl_->written_bytes += bytes;
    index_dirty_ = true;
    wrote = true;
    return true;
}

void ConversationPagePool::reference(uint64_t hash, uint32_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(Key{hash, bytes});
    if (it != index_.end()) ++it->second.refs;
}

void ConversationPagePool::release(uint64_t hash, uint32_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = index_.find(Key{hash, bytes});
    if (it == index_.end()) return;
    if (it->second.refs > 0) --it->second.refs;
}

bool ConversationPagePool::persist(std::string& error) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!writable_) { error = "the page pool is read-only"; return false; }
    if (!impl_->pool.get() && !impl_->pool.open(pool_path_, L"r+b", error)) return false;
    if (!impl_->pool.flush(error)) return false;
    std::vector<IndexEntry> entries;
    entries.reserve(index_.size());
    for (const auto& [k, p] : index_) entries.push_back({k.hash, k.bytes, p.offset});
    const std::string idx = encode_index(generation_, entries);
    if (!write_atomic(index_path_, idx, tmp_path_ + "/pages.idx.tmp", error)) return false;
    index_dirty_ = false;
    return true;
}

bool ConversationPagePool::compact_needed(uint64_t min_dead_bytes, uint64_t min_dead_pages,
                                          double min_fragmentation) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t dead_bytes = 0, dead_pages = 0;
    for (const auto& [k, p] : index_) {
        if (p.refs != 0) continue;
        dead_bytes += p.bytes;
        ++dead_pages;
    }
    const double share = impl_->pool_bytes ? (double) dead_bytes / (double) impl_->pool_bytes : 0.0;
    return dead_pages >= min_dead_pages && dead_bytes >= min_dead_bytes && share >= min_fragmentation;
}

bool ConversationPagePool::compact(uint64_t min_dead_bytes, uint64_t min_dead_pages, double min_fragmentation,
                                   std::string& error) {
    error.clear();
    std::lock_guard<std::mutex> lock(mu_);
    if (!writable_) { error = "the page pool is read-only"; return false; }
    uint64_t dead_bytes = 0, dead_pages = 0;
    for (const auto& [k, p] : index_) {
        if (p.refs != 0) continue;
        dead_bytes += p.bytes;
        ++dead_pages;
    }
    const double share = impl_->pool_bytes ? (double) dead_bytes / (double) impl_->pool_bytes : 0.0;
    if (dead_pages < min_dead_pages || dead_bytes < min_dead_bytes || share < min_fragmentation) return true;

    const auto started = std::chrono::steady_clock::now();
    // Live pages in file order, so the new pool is still written sequentially.
    struct Live {
        uint64_t hash;
        uint32_t bytes;
        uint64_t offset;
        uint32_t refs;
    };
    std::vector<Live> live;
    live.reserve(index_.size() - (size_t) dead_pages);
    for (const auto& [k, p] : index_)
        if (p.refs != 0) live.push_back({k.hash, k.bytes, p.offset, p.refs});
    std::sort(live.begin(), live.end(), [](const Live& a, const Live& b) { return a.offset < b.offset; });

    const uint64_t generation = generation_ + 1;
    const std::string tmp_pool = tmp_path_ + "/pages.bin.tmp";
    const std::string tmp_idx = tmp_path_ + "/pages.idx.tmp";
    File in, out;
    if (!in.open(pool_path_, L"rb", error)) return false;
    if (!out.open(tmp_pool, L"w+b", error)) return false;
    const std::string head = pool_header(generation);
    if (!out.write(head.data(), head.size(), error)) { out.close(); erase_file(tmp_pool, error); return false; }

    std::vector<IndexEntry> entries;
    entries.reserve(live.size());
    uint64_t at = kPoolHeaderBytes, final_size = kPoolHeaderBytes;
    std::vector<uint8_t> data;
    for (const auto& page : live) {
        data.resize(page.bytes);
        if (!in.seek(page.offset, error) || !in.read(data.data(), page.bytes, error) ||
            session_hash64(data.data(), page.bytes, 0) != page.hash) {
            out.close();
            erase_file(tmp_pool, error);
            error = "cannot compact the page pool: " + (error.empty() ? std::string("a page is corrupt") : error);
            return false;
        }
        if (!impl_->write_page(out, page.hash, data.data(), page.bytes, at, error)) {
            out.close();
            erase_file(tmp_pool, error);
            return false;
        }
        entries.push_back({page.hash, page.bytes, at});
        final_size = at + page.bytes;
        at = pool_page_offset_after(at, page.bytes);
    }
    if (!out.flush(error)) { out.close(); erase_file(tmp_pool, error); return false; }
    out.close();
    // The index is a cache: if it cannot be written, the new pool carries a new generation and the next open
    // rebuilds the index by scanning.  The in-memory index is correct either way.
    const std::string idx = encode_index(generation, entries);
    bool index_ready = true;
    {
        File idx_file;
        if (!idx_file.open(tmp_idx, L"wb", error) || !idx_file.write(idx.data(), idx.size(), error) ||
            !idx_file.flush(error)) {
            idx_file.close();
            erase_file(tmp_idx, error);
            index_ready = false;
        }
    }
    in.close();
    impl_->pool.close();
    impl_->reader.close();
    if (!replace_file(tmp_pool, pool_path_, error)) {
        erase_file(tmp_pool, error);
        std::string ignored;
        impl_->pool.open(pool_path_, L"r+b", ignored);   // the old pool is still in place
        return false;
    }
    if (index_ready) {
        std::string why;
        if (!replace_file(tmp_idx, index_path_, why)) {
            // The pool is new and the index names the old generation: a restart rebuilds it.  A later persist()
            // rewrites it, and the in-memory index below is already the new one.
            erase_file(tmp_idx, why);
        }
    }

    generation_ = generation;
    impl_->generation = generation;
    impl_->pool_bytes = at;
    impl_->next_offset = at;
    index_.clear();
    index_.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i)
        index_[Key{entries[i].hash, entries[i].bytes}] = Page{entries[i].offset, entries[i].bytes, live[i].refs};
    ++impl_->compactions;
    impl_->compacted_bytes += dead_bytes;
    impl_->last_compacted_bytes = dead_bytes;
    impl_->last_compaction_ms = ms_since(started);
    return true;
}

ConversationPageStats ConversationPagePool::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    ConversationPageStats s;
    s.pages = index_.size();
    for (const auto& [k, p] : index_) {
        if (p.refs != 0) { s.live_pages += 1; s.live_bytes += p.bytes; }
        else { s.dead_pages += 1; s.dead_bytes += p.bytes; }
    }
    s.pool_bytes = impl_->pool_bytes;
    s.dead_bytes = s.pool_bytes > s.live_bytes ? s.pool_bytes - s.live_bytes : 0;
    s.pages_written = impl_->written_pages;
    s.pages_reused = impl_->reused_pages;
    s.bytes_written = impl_->written_bytes;
    s.bytes_reused = impl_->reused_bytes;
    s.compactions = impl_->compactions;
    s.bytes_compacted = impl_->compacted_bytes;
    s.last_compaction_ms = impl_->last_compaction_ms;
    s.last_compacted_bytes = impl_->last_compacted_bytes;
    return s;
}

uint64_t ConversationPagePool::generation() const {
    std::lock_guard<std::mutex> lock(mu_);
    return generation_;
}

// ---- the writer --------------------------------------------------------------------------------------------------
bool conversation_pages_write(ConversationPagePool& pool, const std::vector<SessionKvSource>& sources,
                              ConversationPageWriteResult& out, std::string& error) {
    out = {};
    std::vector<uint8_t> page(kConversationPageBytes);
    size_t since_beat = 0;
    for (const auto& source : sources) {
        ConversationPageLayer layer;
        layer.format = source.format;
        layer.cells = source.cells;
        layer.heads = source.heads;
        layer.head_dim = source.head_dim;
        layer.page_size = source.page_size;
        layer.pooled_rows = source.pooled_rows;
        layer.idx_dim = source.idx_dim;
        layer.sizes = source.sizes;
        for (size_t part = 0; part < 5; ++part) {
            const uint64_t size = source.sizes[part];
            if (size == 0) continue;
            layer.parts[part].reserve((size_t) ((size + kConversationPageBytes - 1) / kConversationPageBytes));
            for (uint64_t offset = 0; offset < size;) {
                const uint32_t n = (uint32_t) std::min<uint64_t>(kConversationPageBytes, size - offset);
                if (!source.read || !source.read(part, (size_t) offset, page.data(), n)) {
                    error = "reading the K/V for the page pool failed";
                    if (source.error) {
                        const std::string why = source.error();
                        if (!why.empty()) error += ": " + why;
                    }
                    return false;
                }
                const uint64_t hash = session_hash64(page.data(), n, 0);
                ConversationPageRef ref;
                bool wrote = false;
                if (!pool.append(hash, page.data(), n, ref, wrote, error)) return false;
                layer.parts[part].push_back(ref);
                if (wrote) { out.bytes_written += n; ++out.pages_written; }
                else { out.bytes_reused += n; ++out.pages_reused; }
                offset += n;
                if (++since_beat >= 16) { since_beat = 0; progress_beat(); }
            }
        }
        out.layout.layers.push_back(std::move(layer));
    }
    return true;
}

// ---- the reader --------------------------------------------------------------------------------------------------
struct ConversationPageReader::Impl {
    SessionFileStream head;
    ConversationPageLayout layout;
    const ConversationPagePool* pool = nullptr;
    size_t next_layer = 0;
    bool finished = false;
    uint64_t bytes = 0;
    uint64_t peak = 0;
    std::vector<uint8_t> page;
};

ConversationPageReader::ConversationPageReader() : impl_(new Impl) {}
ConversationPageReader::~ConversationPageReader() = default;
ConversationPageReader::ConversationPageReader(ConversationPageReader&&) noexcept = default;
ConversationPageReader& ConversationPageReader::operator=(ConversationPageReader&&) noexcept = default;

bool ConversationPageReader::open(const std::string& head_path, const ConversationPageLayout& layout,
                                  const ConversationPagePool& pool, const SessionFileIdentity& identity,
                                  const SessionReadLimits& limits, std::string& error, SessionStatus* status,
                                  uint64_t head_bytes) {
    error.clear();
    impl_->layout = layout;
    impl_->pool = &pool;
    impl_->next_layer = 0;
    impl_->finished = false;
    uint64_t max_layer = 0;
    for (const auto& layer : layout.layers) max_layer = std::max(max_layer, layer.kv_bytes());
    SessionReadLimits head_limits = limits;
    head_limits.max_kv_layers = 0;   // the head is metadata only
    if (head_bytes != 0 && head_limits.max_file_bytes > head_bytes) head_limits.max_file_bytes = head_bytes;
    if (!impl_->head.open(head_path, identity, head_limits, error, status)) return false;
    if (impl_->head.layers() != 0) { error = "the disk cache head holds K/V layers"; return false; }
    if (head_bytes != 0 && impl_->head.bytes() != head_bytes) {
        error = "the disk cache head is a different size than its manifest";
        if (status) status->error = SessionError::invalid;
        return false;
    }
    if (!impl_->head.finish(error)) return false;
    // The caller's admission covers the head's metadata, one layer and one page.
    if (limits.admit) {
        const uint64_t need = impl_->head.peak_bytes() + max_layer + kConversationPageBytes;
        std::string why;
        if (!limits.admit(need, why)) {
            error = why.empty() ? "the restore was refused by the host staging limit" : why;
            if (status) status->error = SessionError::memory;
            return false;
        }
    }
    // Every page is resolved now: a missing page refuses the entry before the session is touched.
    for (const auto& layer : layout.layers)
        for (const auto& part : layer.parts)
            for (const auto& ref : part) {
                ConversationPagePool::Page page;
                if (!pool.lookup(ref.hash, ref.bytes, page)) {
                    error = "a K/V page of the disk cache is missing";
                    if (status) status->error = SessionError::invalid;
                    return false;
                }
            }
    impl_->bytes = impl_->head.bytes() + layout.kv_bytes();
    impl_->peak = impl_->head.peak_bytes() + kConversationPageBytes;
    return true;
}

const SavedConversation& ConversationPageReader::meta() const { return impl_->head.meta(); }
size_t ConversationPageReader::layers() const { return impl_->layout.layers.size(); }

bool ConversationPageReader::next(ConversationKv& layer, std::string& error) {
    try {
    if (impl_->finished) { error = "the paged conversation is already finished"; return false; }
    if (impl_->next_layer >= impl_->layout.layers.size()) { error = "no more K/V layers"; return false; }
    const ConversationPageLayer& src = impl_->layout.layers[impl_->next_layer];
    layer = {};
    layer.format = src.format;
    layer.cells = src.cells;
    layer.heads = src.heads;
    layer.head_dim = src.head_dim;
    layer.page_size = src.page_size;
    layer.pooled_rows = src.pooled_rows;
    layer.idx_dim = src.idx_dim;
    const std::array<ConversationBuffer*, 5> parts = {&layer.k, &layer.v, &layer.k_scale, &layer.v_scale,
                                                      &layer.pooled};
    impl_->page.resize(kConversationPageBytes);
    uint64_t layer_bytes = 0;
    for (size_t part = 0; part < 5; ++part) {
        const uint64_t size = src.sizes[part];
        if (size == 0) continue;
        uint64_t covered = 0;
        for (const auto& ref : src.parts[part]) covered += ref.bytes;
        if (covered != size) { error = "the K/V page list does not cover its part"; return false; }
        parts[part]->resize((size_t) size);
        uint64_t offset = 0;
        for (const auto& ref : src.parts[part]) {
            ConversationPagePool::Page page;
            if (!impl_->pool->lookup(ref.hash, ref.bytes, page) ||
                !impl_->pool->read(ref.hash, ref.bytes, page.offset, impl_->page.data(), error)) return false;
            const uint64_t at = offset;
            const uint64_t n = ref.bytes;
            if (!parts[part]->visit((size_t) at, (size_t) n,
                                    [&](uint8_t* p, size_t count, size_t where) {
                                        std::memcpy(p, impl_->page.data() + (where - at), count);
                                        return true;
                                    })) {
                error = "cannot fill the K/V layer from its pages";
                return false;
            }
            offset += n;
        }
        layer_bytes += size;
    }
    impl_->peak = std::max(impl_->peak, impl_->head.peak_bytes() + layer_bytes + kConversationPageBytes);
    ++impl_->next_layer;
    return true;
    } catch (const std::bad_alloc&) {
        error = "out of memory while restoring a K/V layer from the page pool";
        return false;
    }
}

bool ConversationPageReader::finish(std::string& error) {
    if (impl_->next_layer != impl_->layout.layers.size()) {
        error = "not every K/V layer was read";
        return false;
    }
    impl_->finished = true;
    return true;
}

uint64_t ConversationPageReader::bytes() const { return impl_->bytes; }
uint64_t ConversationPageReader::logical_bytes() const { return impl_->bytes; }
uint64_t ConversationPageReader::peak_bytes() const { return impl_->peak; }

} // namespace strata::core
