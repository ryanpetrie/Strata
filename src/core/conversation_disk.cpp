#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "the disk cache manifest is little-endian only");

namespace strata::core {
namespace {

namespace fs = std::filesystem;

constexpr char kMagic[8] = {'S', 'T', 'R', 'D', 'I', 'S', 'K', '\x01'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kVersionPaged = 2;
constexpr uint32_t kFlagCvec = 1, kFlagPinned = 2;
// A manifest is a few hundred KB for a 200K-token conversation; the file cap bounds what a corrupt or foreign file
// can make the start-up scan allocate.  The count caps bound one list before its array is sized.
constexpr uint64_t kMaxManifestBytes = 64u << 20;
constexpr uint32_t kMaxTokens = 64u << 20, kMaxImages = 1u << 20, kMaxCheckpoints = 4096;
constexpr uint32_t kMaxPayloadName = 64;
constexpr uint32_t kMaxLayers = 4096;
constexpr size_t kNameTag = 16;

uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t ms_since(const std::chrono::steady_clock::time_point& start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start).count();
}

#ifdef _WIN32
// Paths reach us as UTF-8 (the same convention as the session files); invalid UTF-8 is refused, not replaced.
bool wide(const std::string& s, std::wstring& w) {
    w.clear();
    if (s.empty()) return true;
    if (s.size() > (size_t) INT_MAX) return false;
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return false;
    w.assign((size_t) n, L'\0');
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n) == n;
}
bool utf8(const std::wstring& w, std::string& s) {
    s.clear();
    if (w.empty()) return true;
    if (w.size() > (size_t) INT_MAX) return false;
    const int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w.data(), (int) w.size(), nullptr, 0, nullptr,
                                      nullptr);
    if (n <= 0) return false;
    s.assign((size_t) n, '\0');
    return WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w.data(), (int) w.size(), s.data(), n, nullptr,
                               nullptr) == n;
}
fs::path fs_path(const std::string& s) {
    std::wstring w;
    return wide(s, w) ? fs::path(w) : fs::path();
}
std::string fs_string(const fs::path& p) {
    std::string s;
    return utf8(p.wstring(), s) ? s : std::string();
}
std::string last_error(DWORD code = GetLastError()) {
    char msg[256] = {};
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, msg,
                                   sizeof msg, nullptr);
    std::string s(msg, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}
bool link_or_reparse(const fs::path& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
}
#else
fs::path fs_path(const std::string& s) { return fs::path(s); }
std::string fs_string(const fs::path& p) { return p.string(); }
bool link_or_reparse(const fs::path& p) {
    struct stat st {};
    if (::lstat(p.c_str(), &st) != 0) return false;
    return S_ISLNK(st.st_mode);
}
#endif

uint64_t random_tag() {
    static uint64_t counter = 0;
    std::random_device rd;
    uint64_t r = (uint64_t(rd()) << 32) ^ rd();
    r ^= (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * 0x9E3779B185EBCA87ull;
    r ^= ++counter * 0xC2B2AE3D27D4EB4Full;
    return r;
}

std::string hex16(uint64_t v) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long) v);
    return s;
}

bool all_hex(const std::string& s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

// The names the store makes itself: "snap-<16 lowercase hex>.bin" and ".payload".  A name from the directory is
// accepted only in exactly that shape, so a manifest can never point the payload outside `conversations/`.
bool payload_name_of(const std::string& manifest_leaf, std::string& payload_name) {
    constexpr const char* prefix = "snap-";
    constexpr const char* suffix = ".bin";
    const size_t need = 5 + kNameTag + 4;
    if (manifest_leaf.size() != need || manifest_leaf.compare(0, 5, prefix) != 0 ||
        manifest_leaf.compare(need - 4, 4, suffix) != 0)
        return false;
    const std::string tag = manifest_leaf.substr(5, kNameTag);
    if (!all_hex(tag)) return false;
    payload_name = "snap-" + tag + ".payload";
    return true;
}

std::string head_name_of(const std::string& tag) { return "snap-" + tag + ".head"; }

bool valid_payload_name(const std::string& name) {
    const size_t need = 5 + kNameTag + 8;
    return name.size() == need && name.compare(0, 5, "snap-") == 0 && name.compare(need - 8, 8, ".payload") == 0 &&
           all_hex(name.substr(5, kNameTag));
}

bool valid_head_name(const std::string& name) {
    const size_t need = 5 + kNameTag + 5;
    return name.size() == need && name.compare(0, 5, "snap-") == 0 && name.compare(need - 5, 5, ".head") == 0 &&
           all_hex(name.substr(5, kNameTag));
}

// A data file of an entry: the Phase 1 payload or the Phase 2 metadata head.
bool valid_data_name(const std::string& name) { return valid_payload_name(name) || valid_head_name(name); }

bool plain_file(const fs::path& p, uint64_t& size) {
    if (link_or_reparse(p)) return false;
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(p, ec);
    if (ec || !fs::is_regular_file(st)) return false;
    size = fs::file_size(p, ec);
    return !ec;
}

bool path_taken(const fs::path& p) {
    std::error_code ec;
    return fs::symlink_status(p, ec).type() != fs::file_type::not_found;
}

// The free space the payload's folder reports.  session_free_space_dir() is the session writer's rule for the same
// question (a UNC folder and a drive root need their trailing separator).
bool free_space(const std::string& path, uint64_t& out) {
#ifdef _WIN32
    std::wstring w;
    if (!wide(session_free_space_dir(path, true), w)) return false;
    ULARGE_INTEGER avail{};
    if (!GetDiskFreeSpaceExW(w.empty() ? nullptr : w.c_str(), &avail, nullptr, nullptr)) return false;
    out = avail.QuadPart;
    return true;
#else
    struct statvfs s {};
    if (::statvfs(session_free_space_dir(path, false).c_str(), &s) != 0) return false;
    out = (uint64_t) s.f_bavail * (uint64_t) s.f_frsize;
    return true;
#endif
}

// Removes a file or a link at `path`; never a folder's contents and never through a link.  A name that is already
// gone is not an error.
bool remove_file(const fs::path& path, std::string& error) {
#ifdef _WIN32
    if (DeleteFileW(path.c_str())) return true;
    const DWORD code = GetLastError();
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) return true;
    error = last_error(code);
    return false;
#else
    if (::unlink(path.c_str()) == 0) return true;
    if (errno == ENOENT) return true;
    error = std::strerror(errno);
    return false;
#endif
}

// A new file created exclusively (never an existing file or link), mode 0600 on POSIX, flushed before it is closed.
bool write_exclusive(const fs::path& path, const void* data, size_t n, std::string& error) {
#ifdef _WIN32
    const HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = last_error(); return false; }
    bool ok = true;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t left = n;
    while (left) {
        DWORD written = 0;
        const DWORD want = (DWORD) std::min<size_t>(left, size_t(1) << 30);
        if (!WriteFile(h, p, want, &written, nullptr) || written == 0) { ok = false; break; }
        p += written;
        left -= written;
    }
    if (ok && !FlushFileBuffers(h)) ok = false;
    if (!CloseHandle(h)) ok = false;
    if (!ok) {
        const std::string why = last_error();
        DeleteFileW(path.c_str());
        error = why;
    }
    return ok;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { error = std::strerror(errno); return false; }
    bool ok = true;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t left = n;
    while (left) {
        const ssize_t w = ::write(fd, p, left);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { ok = false; break; }
        p += w;
        left -= (size_t) w;
    }
    if (ok && ::fsync(fd) != 0) ok = false;
    if (::close(fd) != 0) ok = false;
    if (!ok) {
        const std::string why = errno ? std::strerror(errno) : "short write";
        ::unlink(path.c_str());
        error = why;
    }
    return ok;
#endif
}

// The manifest's commit: the temporary name becomes the final one.  A directory cannot be flushed on Windows, and a
// move on one volume is a rename; POSIX also flushes the folder, like the session writer does.
bool publish(const fs::path& from, const fs::path& to, std::string& error) {
#ifdef _WIN32
    if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
    error = last_error();
    return false;
#else
    if (::rename(from.c_str(), to.c_str()) != 0) { error = std::strerror(errno); return false; }
    std::string dir = fs_string(fs::path(to).parent_path());
    const int fd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
    return true;
#endif
}

// ---- manifest bytes ---------------------------------------------------------------------------------------------

void put_u32(std::string& b, uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); }
void put_u64(std::string& b, uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); }
void put_ids(std::string& b, const std::vector<int32_t>& ids) {
    put_u32(b, (uint32_t) ids.size());
    if (!ids.empty()) b.append(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(int32_t));
}
void put_images(std::string& b, const std::vector<ConversationImageKey>& images) {
    put_u32(b, (uint32_t) images.size());
    for (const auto& i : images) {
        put_u64(b, (uint64_t) i.start);
        put_u64(b, i.hash);
    }
}

struct ManifestData {
    uint64_t model = 0, config = 0, created = 0, last_used = 0, payload_bytes = 0;
    std::string payload_name;
    ConversationDiskMeta meta;
    bool pinned = false;
    // v2 (paged): the head's name and size, and the K/V page list.
    bool paged = false;
    std::string head_name;
    uint64_t head_bytes = 0;
    ConversationPageLayout layout;
};

bool encode_manifest(const ManifestData& m, std::string& out, std::string& error) {
    out.clear();
    out.append(kMagic, sizeof kMagic);
    put_u32(out, m.paged ? kVersionPaged : kVersion);
    put_u32(out, (m.meta.cvec ? kFlagCvec : 0u) | (m.pinned ? kFlagPinned : 0u));
    put_u64(out, m.model);
    put_u64(out, m.config);
    put_u64(out, m.created);
    put_u64(out, m.last_used);
    put_u64(out, m.payload_bytes);
    if (m.paged) {
        put_u32(out, (uint32_t) m.head_name.size());
        out.append(m.head_name);
        put_u64(out, m.head_bytes);
    } else {
        put_u32(out, (uint32_t) m.payload_name.size());
        out.append(m.payload_name);
    }
    put_ids(out, m.meta.ids);
    put_images(out, m.meta.images);
    put_u32(out, (uint32_t) m.meta.checkpoints.size());
    for (const auto& c : m.meta.checkpoints) {
        put_ids(out, c.ids);
        put_images(out, c.imgs);
        put_u64(out, c.used);
        put_u32(out, c.pinned ? kFlagPinned : 0u);
    }
    if (m.paged) {
        put_u32(out, (uint32_t) m.layout.layers.size());
        for (const auto& layer : m.layout.layers) {
            put_u64(out, (uint64_t) (int64_t) layer.format);
            put_u64(out, (uint64_t) layer.cells);
            put_u64(out, (uint64_t) layer.heads);
            put_u64(out, (uint64_t) layer.head_dim);
            put_u64(out, (uint64_t) layer.page_size);
            put_u64(out, (uint64_t) layer.pooled_rows);
            put_u64(out, (uint64_t) layer.idx_dim);
            for (size_t part = 0; part < 5; ++part) {
                put_u64(out, layer.sizes[part]);
                put_u32(out, (uint32_t) layer.parts[part].size());
                for (const auto& ref : layer.parts[part]) {
                    put_u64(out, ref.hash);
                    put_u32(out, ref.bytes);
                }
            }
        }
        put_u64(out, m.layout.kv_bytes());
    }
    if (out.size() > kMaxManifestBytes) { error = "the manifest is too large"; return false; }
    put_u64(out, session_hash64(out.data(), out.size(), 0));
    return true;
}

// A cursor over the bytes before the trailing checksum.  Every read is checked against what is left, so a truncated
// or hostile file is refused before an array is sized.
class Reader {
public:
    Reader(const char* data, size_t n) : p_(reinterpret_cast<const uint8_t*>(data)), n_(n) {}
    bool u32(uint32_t& v) {
        if (left() < 4) return false;
        std::memcpy(&v, p_ + at_, 4);
        at_ += 4;
        return true;
    }
    bool u64(uint64_t& v) {
        if (left() < 8) return false;
        std::memcpy(&v, p_ + at_, 8);
        at_ += 8;
        return true;
    }
    bool take(size_t n, std::string& out) {
        if (n > left()) return false;
        out.assign(reinterpret_cast<const char*>(p_ + at_), n);
        at_ += n;
        return true;
    }
    bool ids(std::vector<int32_t>& out) {
        uint32_t n = 0;
        if (!u32(n) || n > kMaxTokens || (uint64_t) n * 4 > left()) return false;
        out.resize(n);
        if (n) std::memcpy(out.data(), p_ + at_, (size_t) n * 4);
        at_ += (size_t) n * 4;
        return true;
    }
    bool images(std::vector<ConversationImageKey>& out) {
        uint32_t n = 0;
        if (!u32(n) || n > kMaxImages || (uint64_t) n * 16 > left()) return false;
        out.resize(n);
        for (auto& i : out) {
            uint64_t start = 0, hash = 0;
            if (!u64(start) || !u64(hash)) return false;
            i.start = (int64_t) start;
            i.hash = hash;
        }
        return true;
    }
    size_t left() const { return n_ - at_; }
    bool done() const { return at_ == n_; }

private:
    const uint8_t* p_;
    size_t n_, at_ = 0;
};

bool decode_manifest(const std::string& bytes, ManifestData& m, std::string& error) {
    if (bytes.size() > kMaxManifestBytes) { error = "the manifest is too large"; return false; }
    if (bytes.size() < sizeof kMagic + 4 + 8) { error = "the manifest is truncated"; return false; }
    Reader r(bytes.data(), bytes.size() - 8);
    std::string magic;
    uint32_t version = 0;
    if (!r.take(sizeof kMagic, magic) || magic != std::string(kMagic, sizeof kMagic)) {
        error = "not a disk cache manifest";
        return false;
    }
    if (!r.u32(version) || (version != kVersion && version != kVersionPaged)) {
        error = "unknown manifest version";
        return false;
    }
    uint64_t checksum = 0;
    std::memcpy(&checksum, bytes.data() + bytes.size() - 8, 8);
    if (session_hash64(bytes.data(), bytes.size() - 8, 0) != checksum) {
        error = "manifest checksum mismatch";
        return false;
    }
    uint32_t flags = 0, name_length = 0, checkpoints = 0;
    if (!r.u32(flags) || (flags & ~(kFlagCvec | kFlagPinned)) != 0) { error = "unknown manifest flags"; return false; }
    if (!r.u64(m.model) || !r.u64(m.config) || !r.u64(m.created) || !r.u64(m.last_used) || !r.u64(m.payload_bytes)) {
        error = "the manifest is truncated";
        return false;
    }
    m.paged = version == kVersionPaged;
    if (!r.u32(name_length) || name_length == 0 || name_length > kMaxPayloadName || name_length > r.left()) {
        error = "the manifest's data name is invalid";
        return false;
    }
    if (!r.take(name_length, m.paged ? m.head_name : m.payload_name)) {
        error = "the manifest's data name is invalid";
        return false;
    }
    if (m.paged) {
        if (!valid_head_name(m.head_name) || !r.u64(m.head_bytes)) {
            error = "the manifest's head name is invalid";
            return false;
        }
    } else if (!valid_payload_name(m.payload_name)) {
        error = "the manifest's payload name is invalid";
        return false;
    }
    if (!r.ids(m.meta.ids) || m.meta.ids.empty()) { error = "the manifest has no tokens"; return false; }
    if (!r.images(m.meta.images)) { error = "the manifest is truncated"; return false; }
    if (!r.u32(checkpoints) || checkpoints > kMaxCheckpoints) {
        error = "the manifest has too many checkpoints";
        return false;
    }
    m.meta.checkpoints.resize(checkpoints);
    for (auto& c : m.meta.checkpoints) {
        uint32_t cp_flags = 0;
        if (!r.ids(c.ids) || !r.images(c.imgs) || !r.u64(c.used) || !r.u32(cp_flags) ||
            (cp_flags & ~kFlagPinned) != 0) {
            error = "the manifest is truncated";
            return false;
        }
        c.pinned = (cp_flags & kFlagPinned) != 0;
    }
    if (m.paged) {
        uint32_t layers = 0;
        if (!r.u32(layers) || layers == 0 || layers > kMaxLayers) {
            error = "the manifest's layer count is invalid";
            return false;
        }
        m.layout.layers.resize(layers);
        for (auto& layer : m.layout.layers) {
            uint64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
            if (!r.u64(format) || !r.u64(cells) || !r.u64(heads) || !r.u64(head_dim) || !r.u64(page_size) ||
                !r.u64(pooled_rows) || !r.u64(idx_dim)) {
                error = "the manifest's K/V layer is truncated";
                return false;
            }
            layer.format = (int) (int64_t) format;
            layer.cells = (int64_t) cells;
            layer.heads = (int64_t) heads;
            layer.head_dim = (int64_t) head_dim;
            layer.page_size = (int64_t) page_size;
            layer.pooled_rows = (int64_t) pooled_rows;
            layer.idx_dim = (int64_t) idx_dim;
            for (size_t part = 0; part < 5; ++part) {
                uint32_t pages = 0;
                if (!r.u64(layer.sizes[part]) || !r.u32(pages) || pages > kMaxTokens) {
                    error = "the manifest's K/V part is truncated";
                    return false;
                }
                if ((uint64_t) pages * 12 > r.left()) {
                    error = "the manifest's K/V part exceeds the file";
                    return false;
                }
                auto& refs = layer.parts[part];
                refs.resize(pages);
                uint64_t covered = 0;
                for (auto& ref : refs) {
                    uint32_t bytes = 0;
                    if (!r.u64(ref.hash) || !r.u32(bytes) || bytes == 0 || bytes > kConversationPageBytes) {
                        error = "the manifest's page list is invalid";
                        return false;
                    }
                    ref.bytes = bytes;
                    covered += bytes;
                }
                if (covered != layer.sizes[part]) {
                    error = "the manifest's page list does not cover its part";
                    return false;
                }
            }
        }
        uint64_t kv_bytes = 0;
        if (!r.u64(kv_bytes) || kv_bytes != m.layout.kv_bytes()) {
            error = "the manifest's K/V size is invalid";
            return false;
        }
        if (m.payload_bytes != sat_add(m.head_bytes, kv_bytes)) {
            error = "the manifest's total size is invalid";
            return false;
        }
    }
    if (!r.done()) { error = "trailing bytes after the manifest"; return false; }
    m.meta.cvec = (flags & kFlagCvec) != 0;
    m.pinned = (flags & kFlagPinned) != 0;
    return true;
}

// The store's own directory check.  All the generated names live below this path, so it must be absolute, must not
// carry a control character, and must not look like a traversal or a Windows device.
bool valid_directory(const std::string& dir, std::string& error) {
    if (dir.empty()) { error = "the disk cache directory is empty"; return false; }
    for (unsigned char c : dir) {
        if (c < 0x20 || c == 0x7f) { error = "the disk cache directory has a control character"; return false; }
    }
#ifdef _WIN32
    std::wstring w;
    if (!wide(dir, w)) { error = "the disk cache directory is not valid UTF-8"; return false; }
    const fs::path p(w);
#else
    const fs::path p(dir);
#endif
    if (!p.is_absolute()) { error = "the disk cache directory is not an absolute path: " + dir; return false; }
    for (const auto& part : p) {
        if (part.has_root_name() || part.has_root_directory()) continue;
        const std::string s = fs_string(part);
        if (s.empty()) continue;
        if (s == "." || s == "..") { error = "the disk cache directory contains a " + s + " component"; return false; }
#ifdef _WIN32
        // Windows strips a trailing dot or space from a name: the path would not mean what it says.
        if (s.back() == ' ' || s.back() == '.') {
            error = "the disk cache directory has a name ending in a dot or space";
            return false;
        }
        std::string base = s.substr(0, s.find('.'));
        while (!base.empty() && base.back() == ' ') base.pop_back();
        for (char& c : base) c = (char) std::toupper((unsigned char) c);
        static const char* const devices[] = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
                                              "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4",
                                              "LPT5", "LPT6", "LPT7", "LPT8", "LPT9", "CONIN$", "CONOUT$"};
        for (const char* device : devices)
            if (base == device) { error = "the disk cache directory names a device: " + s; return false; }
#endif
    }
    return true;
}

bool read_file(const fs::path& path, std::string& out, std::string& error) {
    uint64_t size = 0;
    if (!plain_file(path, size)) { error = "not a regular file"; return false; }
    if (size > kMaxManifestBytes) { error = "the manifest is too large"; return false; }
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "cannot open the manifest"; return false; }
    out.resize((size_t) size);
    if (size && !f.read(out.data(), (std::streamsize) size)) { error = "the manifest ends early"; return false; }
    if (f.peek() != std::ifstream::traits_type::eof()) { error = "the manifest grew while it was read"; return false; }
    return true;
}

} // namespace

ConversationDiskStore::ConversationDiskStore(const ConversationDiskOptions& options,
                                             const SessionFileIdentity& identity, std::string& error)
    : options_(options), identity_(identity) {
    error.clear();
    if (options.budget_bytes == 0) return;   // off: no directory, no file, no scan
    if (!valid_directory(options.dir, error)) return;
    std::lock_guard<std::mutex> lock(mu_);
    if (!setup(error)) { valid_ = false; return; }
    if (options_.cow) {
        const fs::path root = fs_path(options_.dir);
        pool_.reset(new ConversationPagePool(fs_string(root / "pool")));
        if (!pool_->open(true, error)) {
            pool_.reset();
            valid_ = false;
            return;
        }
    }
    if (!scan(error)) { valid_ = false; return; }
    valid_ = true;
}

bool ConversationDiskStore::setup(std::string& error) {
    const fs::path root = fs_path(options_.dir);
    auto ensure = [&](const fs::path& p, const char* what) {
        if (link_or_reparse(p)) {
            error = std::string("the disk cache ") + what + " is a symlink or reparse point";
            return false;
        }
        std::error_code ec;
        const bool made = fs::create_directories(p, ec);
        if (ec) {
            error = std::string("the disk cache cannot create ") + what + ": " + ec.message();
            return false;
        }
        const fs::file_status st = fs::symlink_status(p, ec);
        if (ec || !fs::is_directory(st) || link_or_reparse(p)) {
            error = std::string("the disk cache ") + what + " is not a folder";
            return false;
        }
#ifndef _WIN32
        if (made) {
            std::error_code pe;
            fs::permissions(p, fs::perms::owner_all, fs::perm_options::replace, pe);
        }
#else
        (void) made;
#endif
        return true;
    };
    if (!ensure(root, "directory")) return false;
    const fs::path manifests = root / "manifests", conversations = root / "conversations", tmp = root / "tmp";
    if (!ensure(manifests, "manifests folder")) return false;
    if (!ensure(conversations, "conversations folder")) return false;
    if (!ensure(tmp, "temporary folder")) return false;
    if (options_.cow && !ensure(root / "pool", "page pool folder")) return false;
    manifests_ = fs_string(manifests);    conversations_ = fs_string(conversations);
    tmp_ = fs_string(tmp);
    return true;
}

bool ConversationDiskStore::scan(std::string& error) {
    std::error_code ec;
    // A short-lived file from an interrupted save is garbage; so is any .tmp beside a manifest or a payload.
    {
        std::vector<fs::path> junk;
        for (fs::directory_iterator it(fs_path(tmp_), ec), end; !ec && it != end; it.increment(ec))
            junk.push_back(it->path());
        for (const auto& p : junk) {
            std::error_code rec;
            fs::remove(p, rec);   // a folder stays: tmp holds files, and this never recurses through a link
        }
    }
    for (const std::string& folder : {manifests_, conversations_}) {
        std::vector<fs::path> junk;
        std::error_code fec;
        for (fs::directory_iterator it(fs_path(folder), fec), end; !fec && it != end; it.increment(fec))
            if (it->path().extension() == ".tmp") junk.push_back(it->path());
        for (const auto& p : junk) {
            std::error_code rec;
            fs::remove(p, rec);
        }
    }

    std::vector<Entry> loaded;
    std::vector<std::string> referenced;
    std::error_code lec;
    for (fs::directory_iterator it(fs_path(manifests_), lec), end; !lec && it != end; it.increment(lec)) {
        const fs::path path = it->path();
        const std::string leaf = fs_string(path.filename());
        std::string payload_leaf;
        if (!payload_name_of(leaf, payload_leaf)) continue;   // not one of ours: leave it alone
        // The data file's name is derived from the manifest's; decode picks the payload or head field.
        const fs::path payload = fs_path(conversations_) / fs_path(payload_leaf);
        const fs::path head = fs_path(conversations_) /
                              fs_path(head_name_of(leaf.substr(5, kNameTag)));
        auto drop = [&] {
            std::string ignored;
            remove_file(path, ignored);
            remove_file(payload, ignored);
            remove_file(head, ignored);
        };
        if (link_or_reparse(path)) {
            drop();
            ++stats_.errors;
            continue;
        }
        std::string content, why;
        ManifestData md;
        if (!read_file(path, content, why) || !decode_manifest(content, md, why)) {
            drop();
            ++stats_.errors;
            continue;
        }
        if (md.paged && md.head_name != head_name_of(leaf.substr(5, kNameTag))) {
            drop();   // a manifest may name only its own head, never another entry's
            ++stats_.errors;
            continue;
        }
        if (md.model != identity_.model || md.config != identity_.config) {
            drop();
            continue;
        }
        Entry e;
        if (md.paged) {
            if (!pool_) { drop(); continue; }   // a paged manifest under a whole-file store: it cannot be read
            uint64_t size = 0;
            const fs::path head_path = fs_path(conversations_) / fs_path(md.head_name);
            if (!plain_file(head_path, size) || size != md.head_bytes) {
                drop();
                ++stats_.errors;
                continue;
            }
            // Every page must be in the pool; a missing one makes the entry unreadable (fail closed).
            bool pages_ok = true;
            for (const auto& layer : md.layout.layers) {
                if (!pages_ok) break;
                for (const auto& part : layer.parts) {
                    if (!pages_ok) break;
                    for (const auto& ref : part) {
                        ConversationPagePool::Page page;
                        if (!pool_->lookup(ref.hash, ref.bytes, page)) { pages_ok = false; break; }
                    }
                }
            }
            if (!pages_ok) {
                drop();
                ++stats_.errors;
                continue;
            }
            e.paged = true;
            e.head = fs_string(head_path);
            e.head_bytes = md.head_bytes;
            e.kv_bytes = md.layout.kv_bytes();
            e.bytes = md.head_bytes + e.kv_bytes;
            e.layout = std::move(md.layout);
            referenced.push_back(md.head_name);
        } else {
            uint64_t size = 0;
            if (!plain_file(payload, size) || size != md.payload_bytes) {
                drop();
                ++stats_.errors;
                continue;
            }
            e.bytes = md.payload_bytes;
            e.payload = fs_string(payload);
            referenced.push_back(std::move(payload_leaf));
        }
        e.meta = std::move(md.meta);
        e.last_used = md.last_used;
        e.pinned = md.pinned;
        e.manifest = fs_string(path);
        loaded.push_back(std::move(e));
    }
    if (lec) { error = "the disk cache cannot read its manifests folder: " + lec.message(); return false; }
    // A data file whose manifest never arrived (a crash between the two writes) is garbage; the manifest is the
    // commit.
    {
        std::error_code pec;
        for (fs::directory_iterator it(fs_path(conversations_), pec), end; !pec && it != end; it.increment(pec)) {
            const std::string leaf = fs_string(it->path().filename());
            if (!valid_data_name(leaf)) continue;
            if (std::find(referenced.begin(), referenced.end(), leaf) != referenced.end()) continue;
            std::string ignored;
            remove_file(it->path(), ignored);
        }
    }

    // The index runs least recently used first, as it does inside a run.
    std::stable_sort(loaded.begin(), loaded.end(),
                     [](const Entry& a, const Entry& b) { return a.last_used < b.last_used; });
    for (auto& e : loaded) {
        e.serial = next_serial_++;
        bytes_ += e.bytes;
        clock_ = std::max(clock_, e.last_used);
        if (e.paged && pool_)
            for (const auto& layer : e.layout.layers)
                for (const auto& part : layer.parts)
                    for (const auto& ref : part) pool_->reference(ref.hash, ref.bytes);
        entries_.push_back(std::move(e));
    }
    enforce_limits();   // the budget or the slot count may have shrunk since the files were written
    note_pool_stats();
    return true;
}

bool ConversationDiskStore::enabled() const { return valid_ && options_.budget_bytes != 0; }

bool ConversationDiskStore::wants(int64_t tokens) const {
    std::lock_guard<std::mutex> lock(mu_);
    return wants_locked(tokens);
}

size_t ConversationDiskStore::bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (size_t) bytes_;
}

size_t ConversationDiskStore::entries() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

size_t ConversationDiskStore::slots() const { return options_.slots; }

ConversationDiskStore::Stats ConversationDiskStore::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

bool ConversationDiskStore::borrows(const Match& m) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (m.tokens <= 0 || m.live || m.index >= entries_.size() || entries_[m.index].serial != m.serial) return false;
    size_t deepest = 0;
    for (const auto& c : entries_[m.index].meta.checkpoints) {
        if (c.pinned && (int64_t) c.ids.size() <= m.tokens) return false;
        deepest = std::max(deepest, c.ids.size());
    }
    return (size_t) m.tokens < deepest;
}

const ConversationDiskStore::Entry* ConversationDiskStore::find(uint64_t serial) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& e : entries_)
        if (e.serial == serial) return &e;
    return nullptr;
}

bool ConversationDiskStore::save(const ConversationDiskMeta& meta, uint64_t estimate_bytes, const Writer& writer,
                                 uint64_t& bytes, std::string& error) {
    bytes = 0;
    error.clear();
    const auto started = std::chrono::steady_clock::now();
    if (!writer) {
        std::lock_guard<std::mutex> lock(mu_);
        ++stats_.refusals;
        error = "the disk cache save has no writer";
        return false;
    }
    fs::path payload_path, manifest_path;
    uint64_t tag = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!enabled()) { ++stats_.refusals; error = "the disk cache is off"; return false; }
        if (saving_) { ++stats_.refusals; error = "another disk cache save is running"; return false; }
        if (meta.ids.empty()) { ++stats_.refusals; error = "the disk cache save has no tokens"; return false; }
        if (meta.ids.size() > kMaxTokens || meta.images.size() > kMaxImages ||
            meta.checkpoints.size() > kMaxCheckpoints) {
            ++stats_.refusals;
            error = "the conversation is over the disk cache's manifest limits";
            return false;
        }
        for (const auto& c : meta.checkpoints)
            if (c.ids.size() > kMaxTokens) {
                ++stats_.refusals;
                error = "a checkpoint is over the disk cache's limits";
                return false;
            }
        if (!wants_locked((int64_t) meta.ids.size())) {
            ++stats_.refusals;
            error = "the conversation is shorter than the disk cache's minimum";
            return false;
        }
        if (estimate_bytes > options_.budget_bytes) {
            ++stats_.refusals;
            error = "the estimate " + std::to_string(estimate_bytes) + " bytes is over the disk cache budget " +
                    std::to_string(options_.budget_bytes);
            return false;
        }
        for (int attempt = 0; attempt < 8; ++attempt) {
            tag = random_tag();
            const std::string leaf = "snap-" + hex16(tag);
            payload_path = fs_path(conversations_) / fs_path(leaf + ".payload");
            manifest_path = fs_path(manifests_) / fs_path(leaf + ".bin");
            if (!path_taken(payload_path) && !path_taken(manifest_path)) break;
            tag = 0;
        }
        if (tag == 0) { ++stats_.errors; error = "the disk cache cannot make a file name"; return false; }
        // The free-space preflight comes before any eviction; a save that cannot fit leaves every entry alone.
        const uint64_t need = sat_add(estimate_bytes, options_.min_free_bytes);
        uint64_t free = 0;
        if (!free_space(fs_string(payload_path), free)) {
            ++stats_.refusals;
            error = "the disk cache cannot read the free space of " + conversations_;
            return false;
        }
        if (need == UINT64_MAX || free < need) {
            ++stats_.refusals;
            error = "the disk cache needs " + std::to_string(estimate_bytes) + " bytes plus " +
                    std::to_string(options_.min_free_bytes) + " free, has " + std::to_string(free);
            return false;
        }
        std::string why;
        if (!evict_until_fits(estimate_bytes, why)) { ++stats_.refusals; error = why; return false; }
        saving_ = true;
    }

    uint64_t written = 0;
    std::string writer_error;
    bool wrote = false;
    try {
        wrote = writer(fs_string(payload_path), written, writer_error);
    } catch (...) {
        writer_error = "the disk cache writer threw";
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        saving_ = false;
        if (!wrote) {
            std::string ignored;
            remove_file(payload_path, ignored);
            ++stats_.errors;
            error = writer_error.empty() ? "the disk cache writer failed" : writer_error;
            return false;
        }
        uint64_t actual = 0;
        if (!plain_file(payload_path, actual) || actual == 0 || actual != written) {
            std::string ignored;
            remove_file(payload_path, ignored);
            ++stats_.errors;
            error = "the disk cache writer reported a different size than the payload has";
            return false;
        }
        // The writer's own size is the real one; if the estimate was short, make room now (nothing else has failed).
        std::string why;
        if (!evict_until_fits(actual, why)) {
            std::string ignored;
            remove_file(payload_path, ignored);
            ++stats_.refusals;
            error = why;
            return false;
        }

        ManifestData md;
        md.model = identity_.model;
        md.config = identity_.config;
        md.created = (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch()).count();
        md.last_used = ++clock_;
        md.payload_bytes = actual;
        md.payload_name = fs_string(payload_path.filename());
        md.meta.ids = meta.ids;
        md.meta.images = meta.images;
        md.meta.cvec = meta.cvec;
        md.meta.checkpoints.reserve(meta.checkpoints.size());
        for (const auto& c : meta.checkpoints) {
            ConversationCheckpoint slim;
            slim.ids = c.ids;
            slim.imgs = c.imgs;
            slim.used = c.used;
            slim.pinned = c.pinned;
            md.meta.checkpoints.push_back(std::move(slim));
            md.pinned = md.pinned || c.pinned;
        }
        std::string manifest, why_manifest;
        if (!encode_manifest(md, manifest, why_manifest)) {
            std::string ignored;
            remove_file(payload_path, ignored);
            ++stats_.errors;
            error = why_manifest;
            return false;
        }
        const fs::path tmp_path = fs_path(tmp_) / fs_path("manifest-" + hex16(random_tag()) + ".tmp");
        if (!write_exclusive(tmp_path, manifest.data(), manifest.size(), why_manifest)) {
            std::string ignored;
            remove_file(tmp_path, ignored);
            remove_file(payload_path, ignored);
            ++stats_.errors;
            error = "the disk cache cannot write its manifest: " + why_manifest;
            return false;
        }
        if (!publish(tmp_path, manifest_path, why_manifest)) {
            std::string ignored;
            remove_file(tmp_path, ignored);
            remove_file(payload_path, ignored);
            ++stats_.errors;
            error = "the disk cache cannot publish its manifest: " + why_manifest;
            return false;
        }
        Entry e;
        e.serial = next_serial_++;
        e.meta = std::move(md.meta);
        e.bytes = actual;
        e.last_used = md.last_used;
        e.pinned = md.pinned;
        e.payload = fs_string(payload_path);
        e.manifest = fs_string(manifest_path);
        entries_.push_back(std::move(e));
        bytes_ += actual;
        bytes = actual;
        ++stats_.saves;
        stats_.last_save_ms = ms_since(started);
        return true;
    }
}

bool ConversationDiskStore::save_paged(const ConversationDiskMeta& meta, const SavedConversation& head,
                                       const std::vector<SessionKvSource>& layers, uint64_t& bytes,
                                       uint64_t& written, uint64_t& reused, std::string& error) {
    bytes = written = reused = 0;
    error.clear();
    const auto started = std::chrono::steady_clock::now();
    if (!pool_) { ++stats_.refusals; error = "the disk cache's page pool is off"; return false; }
    if (!head.kv.empty()) { ++stats_.refusals; error = "a paged save's head must not hold K/V"; return false; }
    uint64_t kv_estimate = 0;
    for (const auto& source : layers)
        for (const uint64_t part : source.sizes) kv_estimate = sat_add(kv_estimate, part);

    // What a copy-on-write save can actually write: the pages of the parts that grew, plus the last page of every
    // part (its bytes past the previous copy may have changed).  A previous paged copy that holds a prefix of this
    // conversation tells us where its pages ended; without one, the whole K/V is new.  Computed before `saving_` is
    // set: it allocates a probe list.
    std::vector<std::array<uint64_t, 5>> previous;
    {
        // `best` needs a prompt whose last token starts the next window; the sentinel lets the saved live state match
        // exactly (a copy of this conversation from a previous turn, or the same one saved again).
        std::vector<int32_t> probe = meta.ids;
        probe.push_back(-1);
        const Match m = best(probe, meta.images, meta.cvec);
        if (m.live && m.tokens > 0) {
            if (const Entry* e = find(m.serial); e != nullptr && e->paged)
                for (const auto& layer : e->layout.layers) previous.push_back(layer.sizes);
        }
    }

    std::string leaf;
    fs::path head_path, manifest_path;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!enabled()) { ++stats_.refusals; error = "the disk cache is off"; return false; }
        if (saving_) { ++stats_.refusals; error = "another disk cache save is running"; return false; }
        if (meta.ids.empty()) { ++stats_.refusals; error = "the disk cache save has no tokens"; return false; }
        if (meta.ids.size() > kMaxTokens || meta.images.size() > kMaxImages ||
            meta.checkpoints.size() > kMaxCheckpoints) {
            ++stats_.refusals;
            error = "the conversation is over the disk cache's manifest limits";
            return false;
        }
        if (!wants_locked((int64_t) meta.ids.size())) {
            ++stats_.refusals;
            error = "the conversation is shorter than the disk cache's minimum";
            return false;
        }
        uint64_t tag = 0;
        for (int attempt = 0; attempt < 8; ++attempt) {
            tag = random_tag();
            leaf = "snap-" + hex16(tag);
            head_path = fs_path(conversations_) / fs_path(leaf + ".head");
            manifest_path = fs_path(manifests_) / fs_path(leaf + ".bin");
            if (!path_taken(head_path) && !path_taken(manifest_path)) break;
            tag = 0;
        }
        if (tag == 0) { ++stats_.errors; error = "the disk cache cannot make a file name"; return false; }
        saving_ = true;
    }

    // The head is small (metadata and the running checkpoint): write it first.  It is an orphan until the manifest
    // names it, so a failure below only has to remove it.
    uint64_t head_bytes = 0;
    {
        strata::core::SessionWriteOptions wo;
        wo.min_free_bytes = 0;   // this call preflights the folder's free space below, before any eviction
        wo.durable = true;
        size_t n = 0;
        if (!session_file_write(fs_string(head_path), head, identity_, n, error, wo)) {
            std::lock_guard<std::mutex> lock(mu_);
            saving_ = false;
            std::string ignored;
            remove_file(head_path, ignored);
            ++stats_.errors;
            return false;
        }
        head_bytes = (uint64_t) n;
    }
    const uint64_t logical = sat_add(head_bytes, kv_estimate);
    uint64_t new_upper = head_bytes;
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t part = 0; part < 5; ++part) {
            const uint64_t size = layers[i].sizes[part];
            if (size == 0) continue;
            const uint64_t old = i < previous.size() ? previous[i][part] : 0;
            const uint64_t add = size > old ? sat_add(size - old, kConversationPageBytes) : size;
            new_upper = sat_add(new_upper, add);
        }
    }
    if (new_upper > logical) new_upper = logical;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (logical > options_.budget_bytes) {
            saving_ = false;
            std::string ignored;
            remove_file(head_path, ignored);
            ++stats_.refusals;
            error = "the conversation (" + std::to_string(logical) + " bytes) is over the disk cache budget " +
                    std::to_string(options_.budget_bytes);
            return false;
        }
        // Free space first; a save that cannot fit evicts nothing.  A pool with dead pages may be compacted to
        // reclaim them before the preflight gives up.
        uint64_t free = 0;
        auto enough = [&] {
            return free_space(fs_string(fs_path(conversations_)), free) &&
                   free >= sat_add(new_upper, options_.min_free_bytes);
        };
        if (!enough()) {
            std::string why;
            if (pool_->compact_needed(options_.compact_min_bytes, options_.compact_min_pages,
                                      options_.compact_fragmentation) &&
                pool_->compact(options_.compact_min_bytes, options_.compact_min_pages,
                               options_.compact_fragmentation, why)) {
                note_pool_stats();
            }
            if (!enough()) {
                saving_ = false;
                std::string ignored;
                remove_file(head_path, ignored);
                ++stats_.refusals;
                error = "the disk cache needs " + std::to_string(new_upper) + " bytes plus " +
                        std::to_string(options_.min_free_bytes) + " free, has " + std::to_string(free);
                return false;
            }
        }
        std::string why;
        if (!evict_until_fits(logical, why)) {
            saving_ = false;
            std::string ignored;
            remove_file(head_path, ignored);
            ++stats_.refusals;
            error = why;
            return false;
        }
    }

    ConversationPageWriteResult result;
    std::string page_error;
    bool wrote = false;
    try {
        wrote = conversation_pages_write(*pool_, layers, result, page_error) && pool_->persist(page_error);
    } catch (...) {
        page_error = "the page writer threw";
        wrote = false;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        saving_ = false;
        if (!wrote) {
            std::string ignored;
            remove_file(head_path, ignored);
            ++stats_.errors;
            error = page_error.empty() ? "the disk cache page writer failed" : page_error;
            return false;
        }
        const uint64_t actual = sat_add(head_bytes, result.layout.kv_bytes());
        // A writer that read fewer bytes than its sources promised would leave a manifest that cannot restore.
        if (actual != logical) {
            std::string ignored;
            remove_file(head_path, ignored);
            ++stats_.errors;
            error = "the disk cache wrote " + std::to_string(actual) + " bytes for an estimate of " +
                    std::to_string(logical);
            return false;
        }
        ManifestData md;
        md.paged = true;
        md.model = identity_.model;
        md.config = identity_.config;
        md.created = (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch()).count();
        md.last_used = ++clock_;
        md.payload_bytes = actual;
        md.head_name = leaf + ".head";
        md.head_bytes = head_bytes;
        md.layout = result.layout;
        md.meta.ids = meta.ids;
        md.meta.images = meta.images;
        md.meta.cvec = meta.cvec;
        md.meta.checkpoints.reserve(meta.checkpoints.size());
        for (const auto& c : meta.checkpoints) {
            ConversationCheckpoint slim;
            slim.ids = c.ids;
            slim.imgs = c.imgs;
            slim.used = c.used;
            slim.pinned = c.pinned;
            md.meta.checkpoints.push_back(std::move(slim));
            md.pinned = md.pinned || c.pinned;
        }
        std::string manifest, why_manifest;
        if (!encode_manifest(md, manifest, why_manifest)) {
            std::string ignored;
            remove_file(head_path, ignored);
            ++stats_.errors;
            error = why_manifest;
            return false;
        }
        const fs::path tmp_path = fs_path(tmp_) / fs_path("manifest-" + hex16(random_tag()) + ".tmp");
        if (!write_exclusive(tmp_path, manifest.data(), manifest.size(), why_manifest) ||
            !publish(tmp_path, manifest_path, why_manifest)) {
            std::string ignored;
            remove_file(tmp_path, ignored);
            remove_file(head_path, ignored);
            ++stats_.errors;
            error = "the disk cache cannot publish its manifest: " + why_manifest;
            return false;
        }
        Entry e;
        e.serial = next_serial_++;
        e.meta = std::move(md.meta);
        e.bytes = actual;
        e.last_used = md.last_used;
        e.pinned = md.pinned;
        e.paged = true;
        e.head = fs_string(head_path);
        e.head_bytes = head_bytes;
        e.kv_bytes = result.layout.kv_bytes();
        e.layout = std::move(md.layout);
        e.manifest = fs_string(manifest_path);
        for (const auto& layer : e.layout.layers)
            for (const auto& part : layer.parts)
                for (const auto& ref : part) pool_->reference(ref.hash, ref.bytes);
        entries_.push_back(std::move(e));
        bytes_ += actual;
        bytes = actual;
        written = result.bytes_written;
        reused = result.bytes_reused;
        stats_.last_save_written_bytes = written;
        stats_.last_save_reused_bytes = reused;
        stats_.last_save_pages_written = result.pages_written;
        stats_.last_save_pages_reused = result.pages_reused;
        stats_.pages_written += result.pages_written;
        stats_.pages_reused += result.pages_reused;
        stats_.bytes_written += written;
        stats_.bytes_reused += reused;
        ++stats_.saves;
        stats_.last_save_ms = ms_since(started);
        note_pool_stats();
    }
    // Reclaim dead pages when the pool passed its thresholds.  Best-effort: a failed compaction does not fail a save.
    if (pool_->compact_needed(options_.compact_min_bytes, options_.compact_min_pages,
                              options_.compact_fragmentation)) {
        std::string why;
        pool_->compact(options_.compact_min_bytes, options_.compact_min_pages, options_.compact_fragmentation, why);
        std::lock_guard<std::mutex> lock(mu_);
        note_pool_stats();
    }
    return true;
}

bool ConversationDiskStore::touch(uint64_t serial) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->serial != serial) continue;
        it->last_used = ++clock_;
        // The index runs oldest first, so a used entry moves to the back: eviction and matching both read that order.
        if (it + 1 != entries_.end()) {
            Entry used = std::move(*it);
            entries_.erase(it);
            entries_.push_back(std::move(used));
        }
        return true;
    }
    return false;
}

bool ConversationDiskStore::open_paged(uint64_t serial, ConversationPageReader& reader,
                                       const SessionReadLimits& limits, std::string& error,
                                       SessionStatus* status) const {
    error.clear();
    if (!pool_) { error = "the disk cache's page pool is off"; return false; }
    std::string head;
    ConversationPageLayout layout;
    uint64_t head_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.serial == serial; });
        if (it == entries_.end()) { error = "no such disk cache entry"; return false; }
        if (!it->paged) { error = "the disk cache entry is not paged"; return false; }
        head = it->head;
        head_bytes = it->head_bytes;
        layout = it->layout;   // copies the page list; the reader outlives this call's lock
    }
    return reader.open(head, layout, *pool_, identity_, limits, error, status, head_bytes);
}

bool ConversationDiskStore::compact(std::string& error) {
    error.clear();
    if (!pool_) return true;
    if (!pool_->compact(options_.compact_min_bytes, options_.compact_min_pages, options_.compact_fragmentation,
                        error)) return false;
    std::lock_guard<std::mutex> lock(mu_);
    note_pool_stats();
    return true;
}

bool ConversationDiskStore::compact_needed() const {
    return pool_ && pool_->compact_needed(options_.compact_min_bytes, options_.compact_min_pages,
                                          options_.compact_fragmentation);
}

uint64_t ConversationDiskStore::pool_bytes() const {
    return pool_ ? pool_->stats().pool_bytes : 0;
}

uint64_t ConversationDiskStore::pool_dead_bytes() const {
    return pool_ ? pool_->stats().dead_bytes : 0;
}

void ConversationDiskStore::note_pool_stats() {
    if (!pool_) return;
    const ConversationPageStats s = pool_->stats();
    stats_.pool_bytes = s.pool_bytes;
    stats_.pool_live_bytes = s.live_bytes;
    stats_.pool_dead_bytes = s.dead_bytes;
    stats_.pool_dead_pages = s.dead_pages;
    stats_.compactions = s.compactions;
    stats_.bytes_compacted = s.bytes_compacted;
    stats_.last_compaction_ms = s.last_compaction_ms;
    stats_.last_compacted_bytes = s.last_compacted_bytes;
}

bool ConversationDiskStore::remove(uint64_t serial, std::string& error) {
    error.clear();
    std::lock_guard<std::mutex> lock(mu_);
    auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.serial == serial; });
    if (it == entries_.end()) { error = "no such disk cache entry"; return false; }
    // The manifest is the entry; once it is gone the payload is garbage.  A failed manifest delete leaves the entry
    // usable in the index, so the caller may retry.
    std::string why;
    if (!remove_file(fs_path(it->manifest), why)) {
        ++stats_.errors;
        error = "the disk cache cannot remove the manifest: " + why;
        return false;
    }
    const Entry gone = std::move(*it);
    entries_.erase(it);
    bytes_ -= gone.bytes;
    release_pages(gone);
    const std::string data = gone.paged ? gone.head : gone.payload;
    if (!data.empty() && !remove_file(fs_path(data), why)) {
        ++stats_.errors;
        error = std::string("the disk cache cannot remove the ") + (gone.paged ? "head: " : "payload: ") + why;
        return false;
    }
    return true;
}

void ConversationDiskStore::note_restore(uint64_t ms, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.restores;
    stats_.last_restore_ms = ms;
    stats_.last_restore_bytes = bytes;
}

void ConversationDiskStore::note_restore_failure() {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.restore_failures;
}

size_t ConversationDiskStore::clear() {
    std::deque<Entry> gone;
    {
        std::lock_guard<std::mutex> lock(mu_);
        gone.swap(entries_);
        bytes_ = 0;
    }
    for (const auto& e : gone) erase_files(e);
    return gone.size();
}

bool ConversationDiskStore::evict_until_fits(uint64_t need, std::string& why) {
    auto fits = [&] {
        if (options_.slots != 0 && entries_.size() >= options_.slots) return false;
        return bytes_ <= options_.budget_bytes && need <= options_.budget_bytes - bytes_;
    };
    while (!fits()) {
        bool evicted = false;
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->pinned) continue;
            erase_files(*it);
            bytes_ -= it->bytes;
            entries_.erase(it);
            ++stats_.evictions;
            evicted = true;
            break;
        }
        if (!evicted) {
            why = "the disk cache has no unpinned entry left to evict";
            return false;
        }
    }
    return true;
}

void ConversationDiskStore::enforce_limits() {
    auto over = [&] {
        return (options_.slots != 0 && entries_.size() > options_.slots) || bytes_ > options_.budget_bytes;
    };
    while (over()) {
        bool evicted = false;
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->pinned) continue;
            erase_files(*it);
            bytes_ -= it->bytes;
            entries_.erase(it);
            ++stats_.evictions;
            evicted = true;
            break;
        }
        if (!evicted) return;   // only pinned entries left: they stay, and the next save refuses until they can go
    }
}

void ConversationDiskStore::erase_files(const Entry& entry) {
    std::string ignored;
    remove_file(fs_path(entry.manifest), ignored);   // the manifest first: a crash cannot bring the entry back
    release_pages(entry);
    remove_file(fs_path(entry.paged ? entry.head : entry.payload), ignored);
}

void ConversationDiskStore::release_pages(const Entry& entry) {
    if (!entry.paged || !pool_) return;
    for (const auto& layer : entry.layout.layers)
        for (const auto& part : layer.parts)
            for (const auto& ref : part) pool_->release(ref.hash, ref.bytes);
}

} // namespace strata::core
