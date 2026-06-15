/**
 * @file torch_pickle.cpp
 * @brief Native PyTorch checkpoint (.pth / .pt) loader.
 *
 * Audit H2-followup: implements the three layers needed to turn a PyTorch
 * `.pth` file into a Tenzor `state_dict` map without spawning a Python
 * interpreter:
 *
 *   1. Minimal ZIP archive reader (PyTorch's `_use_new_zipfile_serialization`
 *      writes STORED entries — no DEFLATE for tensor data; the pickle entry
 *      itself is small and also STORED in modern torch versions).
 *   2. Python pickle protocol 2-5 decoder, restricted to the opcodes
 *      `torch.save(state_dict, ...)` actually emits.
 *   3. `_rebuild_tensor_v2(storage, offset, size, stride, ...)` reconstruction
 *      that pulls bytes from the `data/N` archive entries and wraps them in
 *      a fresh CPU `Tensor` of the requested shape/dtype.
 *
 * Out of scope (callers get an actionable runtime_error):
 *   - Compressed entries (DEFLATE / DEFLATE64 / LZMA / etc.).
 *   - Tensors with non-default strides that don't match contiguous row-major.
 *   - Legacy (pre-1.6) non-zipfile pickles.
 */

#include "tenzor/io/torch_pickle.hpp"

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/transform.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tenzor::io {

namespace {

// =========================================================================
// ZIP archive reader (minimal: STORED entries only, no compression).
// =========================================================================

struct ZipEntry {
    std::string name;
    uint64_t local_header_offset = 0;
    uint64_t compressed_size     = 0;
    uint64_t uncompressed_size   = 0;
    uint16_t compression_method  = 0;  // 0 = STORED, 8 = DEFLATE
};

class ZipReader {
public:
    explicit ZipReader(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            throw std::runtime_error("torch_pickle: cannot open file: " + path);
        }
        std::streamsize size = f.tellg();
        if (size < 22) {
            throw std::runtime_error(
                "torch_pickle: file too small to be a ZIP archive: " + path);
        }
        f.seekg(0, std::ios::beg);
        data_.resize(static_cast<size_t>(size));
        f.read(reinterpret_cast<char*>(data_.data()), size);
        if (!f) {
            throw std::runtime_error(
                "torch_pickle: failed to read file contents: " + path);
        }
        parse_central_directory(path);
    }

    auto list_entries() const -> std::vector<std::string> {
        std::vector<std::string> names;
        names.reserve(entries_.size());
        for (const auto& [n, _] : entries_) names.push_back(n);
        return names;
    }

    auto has_entry(const std::string& name) const -> bool {
        return entries_.find(name) != entries_.end();
    }

    /// Read the bytes of an archive entry by name. Throws on missing
    /// entry or compressed payload.
    auto read(const std::string& name) const -> std::vector<uint8_t> {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            throw std::runtime_error(
                "torch_pickle: archive entry not found: " + name);
        }
        return read_entry(it->second);
    }

    /// Find an entry whose name ends with `suffix` (used to locate
    /// `data.pkl` since modern torch.save wraps everything in a top-level
    /// `{archive}/` directory whose name depends on the writer).
    auto find_suffix(const std::string& suffix) const -> std::string {
        for (const auto& [n, _] : entries_) {
            if (n.size() >= suffix.size() &&
                n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
                return n;
            }
        }
        return {};
    }

    /// Borrow a view into archive bytes WITHOUT copying — fast path for the
    /// large `data/N` tensor storage entries (each can be 100s of MB).
    auto read_view(const std::string& name) const -> std::string_view {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            throw std::runtime_error(
                "torch_pickle: archive entry not found: " + name);
        }
        const auto& e = it->second;
        if (e.compression_method != 0) {
            throw std::runtime_error(
                "torch_pickle: entry " + name + " uses compression method " +
                std::to_string(e.compression_method) +
                "; only STORED (0) is supported");
        }
        size_t data_off = local_data_offset(e);
        // Overflow-safe bound: data_off and uncompressed_size derive from
        // untrusted ZIP fields, so `data_off + uncompressed_size` can wrap and
        // pass a naive check, after which data_.data()+data_off is a wild ptr.
        if (data_off > data_.size() ||
            e.uncompressed_size > data_.size() - data_off) {
            throw std::runtime_error(
                "torch_pickle: entry " + name + " data extends past EOF");
        }
        return std::string_view(
            reinterpret_cast<const char*>(data_.data() + data_off),
            e.uncompressed_size);
    }

private:
    static auto read_u16(const uint8_t* p) -> uint16_t {
        return static_cast<uint16_t>(p[0]) |
               (static_cast<uint16_t>(p[1]) << 8);
    }
    static auto read_u32(const uint8_t* p) -> uint32_t {
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
    }
    static auto read_u64(const uint8_t* p) -> uint64_t {
        return static_cast<uint64_t>(read_u32(p)) |
               (static_cast<uint64_t>(read_u32(p + 4)) << 32);
    }

    // Parse the ZIP64 extended-information extra field (header id 0x0001) for a
    // central-directory entry. The 8-byte 64-bit values appear only for the
    // fields whose 32-bit slot holds the 0xFFFFFFFF sentinel, in this fixed
    // order: uncompressed size, compressed size, local header offset, disk
    // start number. Overwrites e.* with the real 64-bit values in place.
    static void apply_zip64_extra(ZipEntry& e, const uint8_t* extra,
                                  uint16_t extra_len, bool need_uncomp,
                                  bool need_comp, bool need_offset) {
        size_t pos = 0;
        while (pos + 4 <= extra_len) {
            uint16_t id   = read_u16(extra + pos);
            uint16_t size = read_u16(extra + pos + 2);
            if (pos + 4 + size > extra_len) {
                break;  // malformed extra field; leave 32-bit values as-is
            }
            if (id == 0x0001) {  // ZIP64 extended information
                const uint8_t* z = extra + pos + 4;
                size_t zpos = 0;
                auto take = [&](bool needed, uint64_t& dst) {
                    if (needed && zpos + 8 <= size) {
                        dst = read_u64(z + zpos);
                        zpos += 8;
                    }
                };
                take(need_uncomp, e.uncompressed_size);
                take(need_comp,   e.compressed_size);
                take(need_offset, e.local_header_offset);
                return;
            }
            pos += 4 + size;
        }
        if (need_uncomp || need_comp || need_offset) {
            throw std::runtime_error(
                "torch_pickle: ZIP entry '" + e.name +
                "' uses a ZIP64 sentinel but has no ZIP64 extra field "
                "(unsupported / corrupt >4GB archive)");
        }
    }

    auto local_data_offset(const ZipEntry& e) const -> size_t {
        // Local file header is 30 bytes + variable name/extra fields.
        // Overflow-safe: local_header_offset is an untrusted uint64; bound it
        // before adding 30 so a near-UINT64_MAX value can't wrap past the guard.
        if (e.local_header_offset > data_.size() ||
            30 > data_.size() - e.local_header_offset) {
            throw std::runtime_error("torch_pickle: truncated local header");
        }
        const uint8_t* p = data_.data() + e.local_header_offset;
        if (read_u32(p) != 0x04034b50u) {
            throw std::runtime_error(
                "torch_pickle: bad local header signature for " + e.name);
        }
        uint16_t name_len  = read_u16(p + 26);
        uint16_t extra_len = read_u16(p + 28);
        return e.local_header_offset + 30 + name_len + extra_len;
    }

    auto read_entry(const ZipEntry& e) const -> std::vector<uint8_t> {
        if (e.compression_method != 0) {
            throw std::runtime_error(
                "torch_pickle: entry " + e.name + " uses compression method " +
                std::to_string(e.compression_method) +
                "; only STORED (0) is supported");
        }
        size_t off = local_data_offset(e);
        // Overflow-safe bound (see read_view): untrusted offset/size.
        if (off > data_.size() ||
            e.uncompressed_size > data_.size() - off) {
            throw std::runtime_error(
                "torch_pickle: entry " + e.name + " data past EOF");
        }
        return std::vector<uint8_t>(data_.begin() + off,
                                    data_.begin() + off + e.uncompressed_size);
    }

    void parse_central_directory(const std::string& path) {
        // Locate End-of-Central-Directory (EOCD): scan backwards for signature
        // 0x06054b50. EOCD can have a comment of up to 65535 bytes after it.
        if (data_.size() < 22) {
            throw std::runtime_error(
                "torch_pickle: file too short for ZIP EOCD: " + path);
        }
        size_t scan_start = data_.size() < (size_t(0xFFFF) + 22)
                              ? 0 : data_.size() - (0xFFFF + 22);
        size_t eocd_pos = SIZE_MAX;
        for (size_t i = data_.size() - 22 + 1; i-- > scan_start; ) {
            if (read_u32(data_.data() + i) == 0x06054b50u) {
                eocd_pos = i;
                break;
            }
        }
        if (eocd_pos == SIZE_MAX) {
            throw std::runtime_error(
                "torch_pickle: ZIP EOCD signature not found — not a torch.save "
                "ZIP archive (legacy pre-1.6 PyTorch checkpoints aren't "
                "supported; see audit H2).  If you have a .pth from PyTorch "
                ">= 1.6 that Tenzor still can't load, re-export it to "
                ".safetensors via `safetensors.torch.save_file` — Tenzor's "
                "safetensors loader has full coverage.  File: " + path);
        }
        const uint8_t* eocd = data_.data() + eocd_pos;
        uint64_t total_entries  = read_u16(eocd + 10);
        uint64_t cd_size        = read_u32(eocd + 12);
        uint64_t cd_offset      = read_u32(eocd + 16);

        // ZIP64: when the archive exceeds the 16/32-bit limits, the EOCD fields
        // hold 0xFFFF / 0xFFFFFFFF sentinels and the real 64-bit values live in
        // the ZIP64 EOCD record, located via the ZIP64 EOCD locator that sits
        // 20 bytes before the EOCD (signature 0x07064b50).
        if (total_entries == 0xFFFFu || cd_size == 0xFFFFFFFFu ||
            cd_offset == 0xFFFFFFFFu) {
            if (eocd_pos < 20) {
                throw std::runtime_error(
                    "torch_pickle: ZIP64 sentinel present but no ZIP64 EOCD "
                    "locator (corrupt >4GB archive): " + path);
            }
            const uint8_t* loc = data_.data() + (eocd_pos - 20);
            if (read_u32(loc) != 0x07064b50u) {
                throw std::runtime_error(
                    "torch_pickle: ZIP64 EOCD locator signature not found "
                    "(corrupt >4GB archive): " + path);
            }
            uint64_t zip64_eocd_off = read_u64(loc + 8);
            // Overflow-safe: zip64_eocd_off comes from the untrusted ZIP64 EOCD
            // locator; bound it before adding 56 to avoid a wraparound bypass.
            if (zip64_eocd_off > data_.size() ||
                56 > data_.size() - zip64_eocd_off) {
                throw std::runtime_error(
                    "torch_pickle: ZIP64 EOCD record past EOF: " + path);
            }
            const uint8_t* z = data_.data() + zip64_eocd_off;
            if (read_u32(z) != 0x06064b50u) {
                throw std::runtime_error(
                    "torch_pickle: bad ZIP64 EOCD record signature: " + path);
            }
            total_entries = read_u64(z + 32);
            cd_size       = read_u64(z + 40);
            cd_offset     = read_u64(z + 48);
        }

        if (cd_offset + cd_size > data_.size()) {
            throw std::runtime_error(
                "torch_pickle: ZIP central directory past EOF");
        }
        size_t cursor = cd_offset;
        for (uint64_t i = 0; i < total_entries; ++i) {
            if (cursor + 46 > cd_offset + cd_size) {
                throw std::runtime_error(
                    "torch_pickle: ZIP central directory entry truncated");
            }
            const uint8_t* cd = data_.data() + cursor;
            if (read_u32(cd) != 0x02014b50u) {
                throw std::runtime_error(
                    "torch_pickle: bad central directory signature at entry " +
                    std::to_string(i));
            }
            uint16_t method = read_u16(cd + 10);
            uint32_t comp_size = read_u32(cd + 20);
            uint32_t uncomp_size = read_u32(cd + 24);
            uint16_t name_len = read_u16(cd + 28);
            uint16_t extra_len = read_u16(cd + 30);
            uint16_t comment_len = read_u16(cd + 32);
            uint32_t lh_off = read_u32(cd + 42);

            if (cursor + 46 + name_len + extra_len > cd_offset + cd_size) {
                throw std::runtime_error(
                    "torch_pickle: ZIP central directory entry truncated");
            }
            std::string name(reinterpret_cast<const char*>(cd + 46), name_len);
            ZipEntry e;
            e.name = name;
            e.local_header_offset = lh_off;
            e.compressed_size = comp_size;
            e.uncompressed_size = uncomp_size;
            e.compression_method = method;

            // ZIP64: any 32-bit field set to 0xFFFFFFFF means the real 64-bit
            // value lives in the ZIP64 extended-information extra block.
            bool need_uncomp = (uncomp_size == 0xFFFFFFFFu);
            bool need_comp   = (comp_size == 0xFFFFFFFFu);
            bool need_offset = (lh_off == 0xFFFFFFFFu);
            if (need_uncomp || need_comp || need_offset) {
                apply_zip64_extra(e, cd + 46 + name_len, extra_len,
                                  need_uncomp, need_comp, need_offset);
            }
            entries_.emplace(std::move(name), std::move(e));

            cursor += 46 + name_len + extra_len + comment_len;
        }
    }

    std::vector<uint8_t> data_;
    std::unordered_map<std::string, ZipEntry> entries_;
};

// =========================================================================
// Pickle protocol 2-5 decoder.
//
// Reference: CPython's `Lib/pickle.py` for the opcode definitions, plus
// PyTorch's `_use_new_zipfile_serialization` codepath for the persistent
// ID format and the `_rebuild_tensor_v2` reduce protocol.
// =========================================================================

struct GlobalRef {
    std::string module;
    std::string name;
};

struct PValue;
using PValuePtr = std::shared_ptr<PValue>;

// Persistent ID payload — torch storage references look like:
//   ('storage', dtype_global, 'data/N', device_str, numel_int).
struct PersistentId {
    std::vector<PValuePtr> items;
};

// A "reduce" call captured at parse time — the result of REDUCE on
// (callable, args). For state_dict pickles the callable is always one of
// torch's _rebuild_* helpers and we materialise tensors immediately.
struct ReduceCall {
    GlobalRef func;
    std::vector<PValuePtr> args;
};

struct PValue {
    enum class Kind {
        None,
        Bool,
        Int,
        Float,
        String,
        Bytes,
        Tuple,
        List,
        Dict,
        Global,
        PersistentId,
        ReduceCall,
        TensorObj,   // resolved tensor (from _rebuild_tensor_v2)
        Mark,        // pickle stack marker
    } kind = Kind::None;

    bool        b = false;
    int64_t     i = 0;
    double      f = 0.0;
    std::string s;             // for String / Bytes
    std::vector<PValuePtr> seq;             // tuple/list
    std::vector<std::pair<PValuePtr, PValuePtr>> dict_entries;
    GlobalRef           g;
    PersistentId        pid;
    ReduceCall          rc;
    Tensor              tensor;
};

inline auto make(PValue::Kind k) -> PValuePtr {
    auto v = std::make_shared<PValue>();
    v->kind = k;
    return v;
}

// Map a torch dtype global like ("torch", "FloatStorage") → tenzor DType
// (and element size in bytes for the raw storage). PyTorch's torch.save
// uses both the legacy `torch.FloatStorage`/`torch.HalfStorage`/... names
// and the modern `torch.storage._TypedStorage` form — the dtype is in the
// first element of the persistent ID tuple in either case.
struct TorchStorageInfo {
    DType dtype;
    size_t elem_size;
};

auto torch_storage_name_to_dtype(const std::string& mod, const std::string& name)
    -> TorchStorageInfo {
    // Legacy names: torch.{Float,Double,Half,BFloat16,Long,Int,Short,Char,Byte,Bool}Storage.
    if (mod != "torch") {
        throw std::runtime_error(
            "torch_pickle: persistent ID dtype must be torch.*Storage, got " +
            mod + "." + name);
    }
    if (name == "FloatStorage")          return {DType::Float32, 4};
    if (name == "DoubleStorage")         return {DType::Float64, 8};
    if (name == "HalfStorage")           return {DType::Float16, 2};
    if (name == "BFloat16Storage")       return {DType::BFloat16, 2};
    if (name == "LongStorage")           return {DType::Int64, 8};
    if (name == "IntStorage")            return {DType::Int32, 4};
    if (name == "ShortStorage")          return {DType::Int16, 2};
    if (name == "CharStorage")           return {DType::Int8, 1};
    if (name == "ByteStorage")           return {DType::UInt8, 1};
    if (name == "BoolStorage")           return {DType::Bool, 1};
    // Audit I.13: extend coverage to complex storages. torch saves
    // Complex64 as `ComplexFloatStorage` (two float32 interleaved) and
    // Complex128 as `ComplexDoubleStorage` (two float64 interleaved).
    // Our DType::Complex64/Complex128 use the same byte layout so the
    // raw storage copy is a 1:1 bit copy with no per-element conversion.
    if (name == "ComplexFloatStorage")   return {DType::Complex64, 8};
    if (name == "ComplexDoubleStorage")  return {DType::Complex128, 16};
    throw std::runtime_error(
        "torch_pickle: unsupported torch storage dtype: torch." + name +
        ". Supported storage types: FloatStorage (Float32), "
        "DoubleStorage (Float64), HalfStorage (Float16), "
        "BFloat16Storage (BFloat16), LongStorage (Int64), "
        "IntStorage (Int32), ShortStorage (Int16), CharStorage (Int8), "
        "ByteStorage (UInt8), BoolStorage (Bool), "
        "ComplexFloatStorage (Complex64), ComplexDoubleStorage (Complex128). "
        "Quantized storages (QInt8Storage / QUInt8Storage) are not yet "
        "implemented in the native pickle reader; re-save those tensors as "
        "safetensors (`model.safetensors`) and load through "
        "ModelHub::load_pretrained_weights instead.");
}

// Read raw bytes from the ZIP `data/N` entry into a freshly-allocated CPU
// Tensor of the requested shape/dtype. Validates that `(offset + numel) *
// elem_size <= entry_size` and that `stride` matches contiguous row-major
// (the only layout torch.save typically emits for state_dicts).
auto build_tensor_from_storage(const ZipReader& zip,
                                const std::string& archive_prefix,
                                const std::string& storage_key,
                                DType dtype,
                                size_t elem_size,
                                int64_t storage_offset,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& stride) -> Tensor {
    // Try both the prefixed and unprefixed entry names. Modern torch.save
    // wraps everything under `{archive_name}/data/N`; some writers omit
    // the prefix.
    std::string full = archive_prefix + storage_key;
    if (!zip.has_entry(full)) {
        if (zip.has_entry(storage_key)) {
            full = storage_key;
        } else {
            // Try suffix match.
            full = zip.find_suffix("/" + storage_key);
            if (full.empty()) {
                throw std::runtime_error(
                    "torch_pickle: storage entry not found: " + storage_key);
            }
        }
    }
    auto view = zip.read_view(full);

    // Reject negative offset/dims and use checked arithmetic. A negative
    // storage_offset or an overflowed (negative) numel could make
    // (storage_offset+numel) small enough to pass the bounds check while
    // copy_bytes/src_byte below point far outside the view (wild pointer /
    // huge copy). .pth/.pt are untrusted by design.
    if (storage_offset < 0) {
        throw std::runtime_error("torch_pickle: negative storage_offset");
    }
    int64_t numel = 1;
    for (auto d : shape) {
        if (static_cast<int64_t>(d) < 0) {
            throw std::runtime_error("torch_pickle: negative tensor dimension");
        }
        if (__builtin_mul_overflow(numel, static_cast<int64_t>(d), &numel)) {
            throw std::runtime_error("torch_pickle: tensor element count overflows int64");
        }
    }
    size_t copy_bytes;
    size_t offset_bytes;
    size_t required_bytes;
    if (__builtin_mul_overflow(static_cast<size_t>(numel), elem_size, &copy_bytes) ||
        __builtin_mul_overflow(static_cast<size_t>(storage_offset), elem_size, &offset_bytes) ||
        __builtin_add_overflow(offset_bytes, copy_bytes, &required_bytes)) {
        throw std::runtime_error("torch_pickle: storage byte size overflow");
    }
    if (required_bytes > view.size()) {
        throw std::runtime_error(
            "torch_pickle: storage " + storage_key + " too small for tensor "
            "(needs " + std::to_string(required_bytes) + " bytes, has " +
            std::to_string(view.size()) + ")");
    }

    // Verify stride matches row-major contiguous; throw with a clear
    // pointer to the future "support arbitrary strides" followup.
    // shape and stride are parsed by independent passes that impose no length
    // relationship, so a crafted/corrupt .pth can carry a stride vector shorter
    // than shape — indexing stride[i] up to shape.size()-1 would be an OOB read.
    if (!stride.empty() && stride.size() != shape.size()) {
        throw std::runtime_error(
            "torch_pickle: stride rank (" + std::to_string(stride.size()) +
            ") does not match shape rank (" + std::to_string(shape.size()) +
            ") in saved tensor");
    }
    if (!stride.empty()) {
        int64_t expected = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
            if (stride[i] != expected) {
                throw std::runtime_error(
                    "torch_pickle: non-contiguous stride in saved tensor "
                    "(stride[" + std::to_string(i) + "] = " +
                    std::to_string(stride[i]) + ", expected " +
                    std::to_string(expected) + "). Only row-major contiguous "
                    "tensors are supported in this loader.");
            }
            expected *= shape[i];
        }
    }

    // Allocate a fresh CPU tensor and memcpy the bytes (sizes validated above).
    Tensor t(std::vector<int64_t>(shape.begin(), shape.end()), dtype, Device::cpu());
    const char* src_byte = view.data() + offset_bytes;
    std::memcpy(t.data_ptr(), src_byte, copy_bytes);
    return t;
}

// Recursive helper: walk a PValue tree, materialising any nested
// `_rebuild_tensor_v2` ReduceCall results into actual Tensors and flattening
// the dict_entries spine into the output state_dict.
class Unpickler {
public:
    Unpickler(const ZipReader& zip, std::string archive_prefix)
        : zip_(zip), archive_prefix_(std::move(archive_prefix)) {}

    auto load(std::string_view bytes) -> PValuePtr {
        cursor_ = 0;
        data_ = bytes;
        stack_.clear();
        memo_.clear();
        while (cursor_ < data_.size()) {
            uint8_t op = read_byte();
            if (handle_op(op)) {
                // STOP encountered.
                if (stack_.empty()) {
                    throw std::runtime_error("torch_pickle: STOP with empty stack");
                }
                return stack_.back();
            }
        }
        throw std::runtime_error("torch_pickle: pickle stream ended without STOP");
    }

private:
    const ZipReader& zip_;
    std::string archive_prefix_;
    std::string_view data_;
    size_t cursor_ = 0;
    std::vector<PValuePtr> stack_;
    std::unordered_map<int64_t, PValuePtr> memo_;

    auto read_byte() -> uint8_t {
        if (cursor_ >= data_.size()) {
            throw std::runtime_error("torch_pickle: unexpected EOF");
        }
        return static_cast<uint8_t>(data_[cursor_++]);
    }
    void read_into(void* dst, size_t n) {
        if (cursor_ + n > data_.size()) {
            throw std::runtime_error("torch_pickle: read past EOF");
        }
        std::memcpy(dst, data_.data() + cursor_, n);
        cursor_ += n;
    }
    auto read_u8() -> uint8_t {
        return read_byte();
    }
    auto read_u16le() -> uint16_t {
        uint16_t v;
        read_into(&v, 2);
        return v;
    }
    auto read_u32le() -> uint32_t {
        uint32_t v;
        read_into(&v, 4);
        return v;
    }
    auto read_i32le() -> int32_t {
        int32_t v;
        read_into(&v, 4);
        return v;
    }
    auto read_u64le() -> uint64_t {
        uint64_t v;
        read_into(&v, 8);
        return v;
    }
    auto read_line() -> std::string {
        // Pickle GLOBAL uses \n-terminated module + name.
        std::string s;
        while (cursor_ < data_.size()) {
            char c = data_[cursor_++];
            if (c == '\n') return s;
            s.push_back(c);
        }
        throw std::runtime_error("torch_pickle: unterminated line");
    }
    auto read_string(size_t n) -> std::string {
        if (cursor_ + n > data_.size()) {
            throw std::runtime_error("torch_pickle: string read past EOF");
        }
        std::string s(data_.data() + cursor_, n);
        cursor_ += n;
        return s;
    }
    void push(PValuePtr v) { stack_.push_back(std::move(v)); }
    auto pop() -> PValuePtr {
        if (stack_.empty()) throw std::runtime_error("torch_pickle: stack underflow");
        auto v = stack_.back();
        stack_.pop_back();
        return v;
    }
    auto top() -> PValuePtr& {
        if (stack_.empty()) throw std::runtime_error("torch_pickle: stack empty");
        return stack_.back();
    }
    /// Pop entries down to the topmost MARK, return them in order.
    auto pop_to_mark() -> std::vector<PValuePtr> {
        std::vector<PValuePtr> items;
        while (!stack_.empty() && stack_.back()->kind != PValue::Kind::Mark) {
            items.push_back(stack_.back());
            stack_.pop_back();
        }
        if (stack_.empty()) {
            throw std::runtime_error("torch_pickle: pop_to_mark with no MARK");
        }
        stack_.pop_back();  // consume MARK
        std::reverse(items.begin(), items.end());
        return items;
    }

    auto resolve_persistent_id(const std::vector<PValuePtr>& tup) -> PValuePtr {
        // PyTorch persistent ID for a storage:
        //   ('storage', <torch.FloatStorage>, 'data/N', <device_str>, <numel>)
        // We capture this as a PersistentId; later, when REDUCE on
        // _rebuild_tensor_v2 fires, we use the captured storage info to
        // materialise the actual tensor.
        if (tup.size() < 5 || tup[0]->kind != PValue::Kind::String ||
            tup[0]->s != "storage") {
            throw std::runtime_error(
                "torch_pickle: unrecognised persistent ID shape (expected "
                "('storage', dtype, key, device, numel))");
        }
        auto v = make(PValue::Kind::PersistentId);
        v->pid.items = tup;
        return v;
    }

    /// Handle the REDUCE opcode: stack = [..., callable, args_tuple].
    /// For state_dict loading, the only callable we need to actually
    /// execute is `torch._utils._rebuild_tensor_v2` (and a few aliases);
    /// all others are captured as deferred ReduceCall values.
    void do_reduce() {
        auto args = pop();
        auto callable = pop();
        if (callable->kind != PValue::Kind::Global) {
            throw std::runtime_error(
                "torch_pickle: REDUCE expects a GLOBAL callable on the stack");
        }
        const auto& g = callable->g;
        // _rebuild_from_type_v2(func, new_type, args, state): the real tensor
        // rebuild arguments are NOT this call's top-level args — args[0] is the
        // underlying callable (e.g. _rebuild_tensor_v2), and the (storage,
        // offset, size, stride, ...) tuple is nested at args[2]. Unwrap it and
        // dispatch to the underlying rebuild path; treating the top-level args
        // as a tensor-rebuild list would fail ("arg 0 must be a persistent
        // storage ID"), breaking any checkpoint using tensor subclasses /
        // parameter wrapping.
        if (g.module == "torch._tensor" && g.name == "_rebuild_from_type_v2" &&
            args->kind == PValue::Kind::Tuple) {
            const auto& outer = args->seq;
            if (outer.size() < 3) {
                throw std::runtime_error(
                    "torch_pickle: _rebuild_from_type_v2 needs (func, new_type, "
                    "args, [state]), got " + std::to_string(outer.size()));
            }
            if (outer[0]->kind != PValue::Kind::Global) {
                throw std::runtime_error(
                    "torch_pickle: _rebuild_from_type_v2 arg 0 must be a GLOBAL "
                    "callable");
            }
            const auto& inner_fn = outer[0]->g;
            if (outer[2]->kind != PValue::Kind::Tuple &&
                outer[2]->kind != PValue::Kind::List) {
                throw std::runtime_error(
                    "torch_pickle: _rebuild_from_type_v2 arg 2 must be the nested "
                    "rebuild-args tuple");
            }
            auto t = materialise_rebuild_tensor(inner_fn.name, outer[2]->seq);
            auto v = make(PValue::Kind::TensorObj);
            v->tensor = std::move(t);
            push(v);
            return;
        }
        bool is_rebuild_tensor =
            (g.module == "torch._utils" && g.name == "_rebuild_tensor_v2") ||
            (g.module == "torch._utils" && g.name == "_rebuild_tensor");
        if (is_rebuild_tensor && args->kind == PValue::Kind::Tuple) {
            auto t = materialise_rebuild_tensor(g.name, args->seq);
            auto v = make(PValue::Kind::TensorObj);
            v->tensor = std::move(t);
            push(v);
            return;
        }
        // OrderedDict / collections.OrderedDict constructors — torch.save
        // wraps the top-level state_dict in OrderedDict. The reduce
        // returns an empty dict that's then filled via BUILD with the
        // actual entries.
        if ((g.module == "collections" && g.name == "OrderedDict") ||
            (g.module == "__builtin__" && g.name == "dict") ||
            (g.module == "builtins" && g.name == "dict")) {
            push(make(PValue::Kind::Dict));
            return;
        }
        // Unknown callable — capture as a deferred reduce so the outer
        // walk can still see its arg structure.
        auto v = make(PValue::Kind::ReduceCall);
        v->rc.func = g;
        if (args->kind == PValue::Kind::Tuple) v->rc.args = args->seq;
        else v->rc.args = {args};
        push(v);
    }

    auto materialise_rebuild_tensor(const std::string& fn_name,
                                     const std::vector<PValuePtr>& args) -> Tensor {
        // _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)
        // _rebuild_tensor   (storage, storage_offset, size, stride)
        if (args.size() < 4) {
            throw std::runtime_error(
                "torch_pickle: " + fn_name + " needs at least 4 args, got " +
                std::to_string(args.size()));
        }
        auto& storage = args[0];
        if (storage->kind != PValue::Kind::PersistentId) {
            throw std::runtime_error(
                "torch_pickle: " + fn_name + " arg 0 must be a persistent storage ID");
        }
        const auto& pid = storage->pid.items;
        // pid: ('storage', <Global FloatStorage>, 'data/N', <device_str>, <numel>)
        if (pid[1]->kind != PValue::Kind::Global) {
            throw std::runtime_error(
                "torch_pickle: storage dtype slot must be a GLOBAL ref");
        }
        TorchStorageInfo info = torch_storage_name_to_dtype(pid[1]->g.module, pid[1]->g.name);
        if (pid[2]->kind != PValue::Kind::String) {
            throw std::runtime_error("torch_pickle: storage key must be a string");
        }
        const std::string& storage_key = pid[2]->s;

        if (args[1]->kind != PValue::Kind::Int) {
            throw std::runtime_error("torch_pickle: storage_offset must be int");
        }
        int64_t storage_offset = args[1]->i;

        auto to_int_vec = [](const PValuePtr& v) -> std::vector<int64_t> {
            if (v->kind != PValue::Kind::Tuple && v->kind != PValue::Kind::List) {
                throw std::runtime_error(
                    "torch_pickle: expected tuple/list for size or stride");
            }
            std::vector<int64_t> out;
            out.reserve(v->seq.size());
            for (auto& it : v->seq) {
                if (it->kind != PValue::Kind::Int) {
                    throw std::runtime_error(
                        "torch_pickle: size/stride entry must be int");
                }
                out.push_back(it->i);
            }
            return out;
        };
        std::vector<int64_t> shape = to_int_vec(args[2]);
        std::vector<int64_t> stride = to_int_vec(args[3]);

        return build_tensor_from_storage(
            zip_, archive_prefix_, storage_key,
            info.dtype, info.elem_size, storage_offset,
            shape, stride);
    }

    /// Return value: true if STOP was hit.
    auto handle_op(uint8_t op) -> bool {
        switch (op) {
            case 0x80: {  // PROTO
                (void)read_byte();
                return false;
            }
            case 0x95: {  // FRAME
                (void)read_u64le();
                return false;
            }
            case '.': return true;  // STOP
            case '(': {              // MARK
                push(make(PValue::Kind::Mark));
                return false;
            }
            case 'N': push(make(PValue::Kind::None));  return false;
            case 0x88: {              // NEWTRUE
                auto v = make(PValue::Kind::Bool); v->b = true;  push(v); return false;
            }
            case 0x89: {              // NEWFALSE
                auto v = make(PValue::Kind::Bool); v->b = false; push(v); return false;
            }
            case 'K': {              // BININT1
                auto v = make(PValue::Kind::Int); v->i = read_byte();        push(v); return false;
            }
            case 'M': {              // BININT2
                auto v = make(PValue::Kind::Int); v->i = read_u16le();       push(v); return false;
            }
            case 'J': {              // BININT (signed 32-bit)
                auto v = make(PValue::Kind::Int); v->i = read_i32le();       push(v); return false;
            }
            case 0x8a: {              // LONG1 (n-byte LE signed)
                uint8_t n = read_byte();
                int64_t val = 0;
                std::vector<uint8_t> buf(n);
                if (n > 0) read_into(buf.data(), n);
                for (size_t i = 0; i < n; ++i) {
                    val |= static_cast<int64_t>(buf[i]) << (8 * i);
                }
                // Sign-extend if MSB set.
                if (n > 0 && (buf[n - 1] & 0x80)) {
                    for (size_t i = n; i < 8; ++i) {
                        val |= static_cast<int64_t>(0xff) << (8 * i);
                    }
                }
                auto v = make(PValue::Kind::Int); v->i = val; push(v);
                return false;
            }
            case 0x8b: {              // LONG4
                uint32_t n = read_u32le();
                int64_t val = 0;
                std::vector<uint8_t> buf(n);
                if (n > 0) read_into(buf.data(), n);
                // The true sign is the MSB of the highest byte (buf[n-1]), not
                // buf[7]; a wider-than-64-bit value cannot be represented in an
                // int64_t.
                bool negative = (n > 0) && (buf[n - 1] & 0x80);
                if (n > 8) {
                    // Bytes above the low 8 must be pure sign-fill (0x00 when
                    // positive, 0xff when negative); otherwise the value does
                    // not fit in int64_t and silently truncating would corrupt
                    // it. Also require bit 63 to already carry the sign so the
                    // low 8 bytes represent the same signed value.
                    uint8_t fill = negative ? 0xff : 0x00;
                    for (size_t i = 8; i < n; ++i) {
                        if (buf[i] != fill) {
                            throw std::runtime_error(
                                "torch_pickle: LONG4 integer too large for int64_t");
                        }
                    }
                    if (negative != ((buf[7] & 0x80) != 0)) {
                        throw std::runtime_error(
                            "torch_pickle: LONG4 integer too large for int64_t");
                    }
                }
                for (size_t i = 0; i < std::min<size_t>(n, 8); ++i) {
                    val |= static_cast<int64_t>(buf[i]) << (8 * i);
                }
                if (negative) {
                    for (size_t i = n; i < 8; ++i) {
                        val |= static_cast<int64_t>(0xff) << (8 * i);
                    }
                }
                auto v = make(PValue::Kind::Int); v->i = val; push(v);
                return false;
            }
            case 'G': {              // BINFLOAT (8 bytes BIG-endian IEEE-754 double)
                uint8_t b[8]; read_into(b, 8);
                uint8_t rev[8];
                for (int i = 0; i < 8; ++i) rev[i] = b[7 - i];
                double d;
                std::memcpy(&d, rev, 8);
                auto v = make(PValue::Kind::Float); v->f = d; push(v); return false;
            }
            case 'X': {              // BINUNICODE
                uint32_t n = read_u32le();
                auto v = make(PValue::Kind::String); v->s = read_string(n);
                push(v); return false;
            }
            case 0x8c: {              // SHORT_BINUNICODE
                uint8_t n = read_byte();
                auto v = make(PValue::Kind::String); v->s = read_string(n);
                push(v); return false;
            }
            case 0x8d: {              // BINUNICODE8
                uint64_t n = read_u64le();
                auto v = make(PValue::Kind::String); v->s = read_string(static_cast<size_t>(n));
                push(v); return false;
            }
            case 'T': {              // BINSTRING
                uint32_t n = read_u32le();
                auto v = make(PValue::Kind::Bytes); v->s = read_string(n);
                push(v); return false;
            }
            case 'U': {              // SHORT_BINSTRING
                uint8_t n = read_byte();
                auto v = make(PValue::Kind::Bytes); v->s = read_string(n);
                push(v); return false;
            }
            case 'B': {              // BINBYTES
                uint32_t n = read_u32le();
                auto v = make(PValue::Kind::Bytes); v->s = read_string(n);
                push(v); return false;
            }
            case 'C': {              // SHORT_BINBYTES
                uint8_t n = read_byte();
                auto v = make(PValue::Kind::Bytes); v->s = read_string(n);
                push(v); return false;
            }
            case 0x8e: {              // BINBYTES8
                uint64_t n = read_u64le();
                auto v = make(PValue::Kind::Bytes); v->s = read_string(static_cast<size_t>(n));
                push(v); return false;
            }
            case '}': push(make(PValue::Kind::Dict)); return false;
            case ']': push(make(PValue::Kind::List)); return false;
            case ')': {              // EMPTY_TUPLE
                push(make(PValue::Kind::Tuple)); return false;
            }
            case 0x85: {              // TUPLE1
                auto v = make(PValue::Kind::Tuple);
                v->seq.push_back(pop());
                push(v); return false;
            }
            case 0x86: {              // TUPLE2
                auto v = make(PValue::Kind::Tuple);
                auto b = pop(); auto a = pop();
                v->seq.push_back(a); v->seq.push_back(b);
                push(v); return false;
            }
            case 0x87: {              // TUPLE3
                auto v = make(PValue::Kind::Tuple);
                auto c = pop(); auto b = pop(); auto a = pop();
                v->seq = {a, b, c};
                push(v); return false;
            }
            case 't': {              // TUPLE
                auto items = pop_to_mark();
                auto v = make(PValue::Kind::Tuple);
                v->seq = std::move(items);
                push(v); return false;
            }
            case 'l': {              // LIST
                auto items = pop_to_mark();
                auto v = make(PValue::Kind::List);
                v->seq = std::move(items);
                push(v); return false;
            }
            case 'q': {              // BINPUT
                uint8_t k = read_byte(); memo_[k] = top(); return false;
            }
            case 'r': {              // LONG_BINPUT
                uint32_t k = read_u32le(); memo_[k] = top(); return false;
            }
            case 'h': {              // BINGET
                uint8_t k = read_byte();
                auto it = memo_.find(k);
                if (it == memo_.end()) {
                    throw std::runtime_error(
                        "torch_pickle: BINGET missing memo key " + std::to_string(k));
                }
                push(it->second); return false;
            }
            case 'j': {              // LONG_BINGET
                uint32_t k = read_u32le();
                auto it = memo_.find(k);
                if (it == memo_.end()) {
                    throw std::runtime_error(
                        "torch_pickle: LONG_BINGET missing memo key " + std::to_string(k));
                }
                push(it->second); return false;
            }
            case 0x94: {              // MEMOIZE
                memo_[static_cast<int64_t>(memo_.size())] = top();
                return false;
            }
            case 's': {              // SETITEM
                auto value = pop(); auto key = pop();
                auto& d = top();
                if (d->kind != PValue::Kind::Dict) {
                    throw std::runtime_error(
                        "torch_pickle: SETITEM target is not a dict");
                }
                d->dict_entries.emplace_back(std::move(key), std::move(value));
                return false;
            }
            case 'u': {              // SETITEMS
                auto items = pop_to_mark();
                if (items.size() % 2 != 0) {
                    throw std::runtime_error(
                        "torch_pickle: SETITEMS expects even number of items");
                }
                auto& d = top();
                if (d->kind != PValue::Kind::Dict) {
                    throw std::runtime_error(
                        "torch_pickle: SETITEMS target is not a dict");
                }
                for (size_t i = 0; i < items.size(); i += 2) {
                    d->dict_entries.emplace_back(items[i], items[i + 1]);
                }
                return false;
            }
            case 'a': {              // APPEND
                auto v = pop();
                auto& lst = top();
                if (lst->kind != PValue::Kind::List) {
                    throw std::runtime_error(
                        "torch_pickle: APPEND target is not a list");
                }
                lst->seq.push_back(std::move(v));
                return false;
            }
            case 'e': {              // APPENDS
                auto items = pop_to_mark();
                auto& lst = top();
                if (lst->kind != PValue::Kind::List) {
                    throw std::runtime_error(
                        "torch_pickle: APPENDS target is not a list");
                }
                for (auto& v : items) lst->seq.push_back(std::move(v));
                return false;
            }
            case 'c': {              // GLOBAL (module \n name \n)
                auto v = make(PValue::Kind::Global);
                v->g.module = read_line();
                v->g.name   = read_line();
                push(v); return false;
            }
            case 0x93: {              // STACK_GLOBAL (pops name then module)
                auto name = pop();
                auto mod = pop();
                if (name->kind != PValue::Kind::String ||
                    mod->kind  != PValue::Kind::String) {
                    throw std::runtime_error(
                        "torch_pickle: STACK_GLOBAL expects two strings");
                }
                auto v = make(PValue::Kind::Global);
                v->g.module = mod->s;
                v->g.name   = name->s;
                push(v); return false;
            }
            case 'Q': {              // BINPERSID
                auto tup = pop();
                if (tup->kind != PValue::Kind::Tuple) {
                    throw std::runtime_error(
                        "torch_pickle: BINPERSID expects a tuple on stack");
                }
                push(resolve_persistent_id(tup->seq));
                return false;
            }
            case 'R': {              // REDUCE
                do_reduce(); return false;
            }
            case 'b': {              // BUILD — apply __setstate__ to top
                // For our restricted scope, just discard the state and keep
                // the object. Torch state_dict pickles only use BUILD on
                // the OrderedDict (state = None) which is a no-op anyway.
                (void)pop();         // state
                return false;
            }
            case 0x8f: {              // EMPTY_SET
                // Treat as empty list — state_dicts never embed sets but the
                // protocol lib emits this for empty tuples occasionally.
                push(make(PValue::Kind::List));
                return false;
            }
            default:
                throw std::runtime_error(
                    "torch_pickle: unsupported pickle opcode 0x" +
                    [op]() {
                        char buf[3];
                        std::snprintf(buf, sizeof(buf), "%02x", op);
                        return std::string(buf);
                    }());
        }
    }
};

// =========================================================================
// State-dict extraction: walk the top-level dict and flatten any
// `_rebuild_*` reduce results into a flat name → Tensor map.
// =========================================================================

void flatten_state_dict(const PValuePtr& node,
                         const std::string& prefix,
                         std::unordered_map<std::string, Tensor>& out) {
    switch (node->kind) {
        case PValue::Kind::Dict: {
            for (auto& [k, v] : node->dict_entries) {
                // A state-dict spine key must be a string/bytes name. Coercing a
                // non-string key to "" would collapse distinct subtrees onto the
                // same prefix and make tensors silently overwrite each other —
                // fail loudly instead.
                if (k->kind != PValue::Kind::String &&
                    k->kind != PValue::Kind::Bytes) {
                    throw std::runtime_error(
                        "torch_pickle: state-dict key is not a string/bytes name "
                        "(cannot build a flattened parameter name)");
                }
                const std::string& name = k->s;
                std::string sub = prefix.empty() ? name : prefix + "." + name;
                flatten_state_dict(v, sub, out);
            }
            break;
        }
        case PValue::Kind::TensorObj: {
            // insert (not emplace) and detect collisions: two tensors flattening
            // to the same name must not silently drop the second, which would
            // return a partial state_dict that looks complete.
            auto [it, inserted] = out.insert({prefix, node->tensor});
            if (!inserted) {
                throw std::runtime_error(
                    "torch_pickle: duplicate flattened tensor name '" + prefix +
                    "' in state dict");
            }
            break;
        }
        default:
            // Skip non-tensor values silently (PyTorch state_dicts sometimes
            // include scalars or version markers).
            break;
    }
}

}  // namespace

auto load_torch_pickle(const std::string& path)
    -> std::unordered_map<std::string, Tensor> {
    ZipReader zip(path);

    // Find the `data.pkl` entry. Modern torch.save wraps everything under
    // `{archive_name}/data.pkl`; the archive name depends on the save
    // call's writer.
    std::string pkl_name = zip.find_suffix("/data.pkl");
    if (pkl_name.empty() && zip.has_entry("data.pkl")) {
        pkl_name = "data.pkl";
    }
    if (pkl_name.empty()) {
        throw std::runtime_error(
            "torch_pickle: no `data.pkl` entry in archive — not a recognised "
            "torch.save layout. File: " + path);
    }

    // archive prefix = everything before /data.pkl (used to find
    // `data/N` entries).
    std::string archive_prefix;
    if (auto slash = pkl_name.rfind('/'); slash != std::string::npos) {
        archive_prefix = pkl_name.substr(0, slash + 1);
    }

    auto pkl_bytes = zip.read(pkl_name);
    Unpickler up(zip, archive_prefix);
    auto root = up.load(std::string_view(
        reinterpret_cast<const char*>(pkl_bytes.data()), pkl_bytes.size()));

    std::unordered_map<std::string, Tensor> state;
    // Some torch.save outputs wrap the state_dict in an outer dict like
    // {"model": <dict>, "optimizer": ..., ...} — checkpoint files for full
    // training state. The audit's scope is state_dict only; if the top
    // level looks like a flat name→Tensor map we use it directly, else
    // walk one level deeper looking for a dict whose values are tensors.
    flatten_state_dict(root, "", state);
    if (state.empty() && root->kind == PValue::Kind::Dict) {
        // Try one level deeper — pick the first sub-dict that contains tensors.
        for (auto& [k, v] : root->dict_entries) {
            if (v->kind == PValue::Kind::Dict) {
                std::unordered_map<std::string, Tensor> trial;
                flatten_state_dict(v, "", trial);
                if (!trial.empty()) {
                    state = std::move(trial);
                    break;
                }
            }
        }
    }
    return state;
}

}  // namespace tenzor::io
