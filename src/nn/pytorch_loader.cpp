/**
 * @file pytorch_loader.cpp
 * @brief PyTorch .pth/.pt file loader implementation
 *
 * Implements a minimal ZIP reader and restricted pickle interpreter
 * to load PyTorch state dictionaries. Only supports tensor loading
 * (not arbitrary Python objects) for safety.
 *
 * PyTorch file format:
 * - ZIP archive containing:
 *   - archive/data.pkl: Pickle stream with state dict metadata
 *   - archive/data/0, archive/data/1, ...: Raw tensor storage data
 */

#include "tenzor/nn/pytorch_loader.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <map>

namespace tenzor {
namespace nn {

namespace {

// ============================================================================
// Minimal ZIP reader (read-only, no compression)
// ============================================================================

// ZIP local file header signature
constexpr uint32_t ZIP_LOCAL_MAGIC = 0x04034b50;  // "PK\x03\x04"
constexpr uint32_t ZIP_CENTRAL_MAGIC = 0x02014b50; // "PK\x01\x02"
constexpr uint32_t ZIP_END_MAGIC = 0x06054b50;     // "PK\x05\x06"

struct ZipEntry {
    std::string name;
    uint64_t offset;       // Offset of data in file
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint16_t compression;  // 0 = stored, 8 = deflate
};

auto read_u16(const uint8_t* p) -> uint16_t {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

auto read_u32(const uint8_t* p) -> uint32_t {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

class ZipReader {
public:
    explicit ZipReader(const std::string& path) : path_(path) {
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        // Read entire file into memory for simplicity
        file_.seekg(0, std::ios::end);
        size_ = static_cast<size_t>(file_.tellg());
        file_.seekg(0, std::ios::beg);
        data_.resize(size_);
        file_.read(reinterpret_cast<char*>(data_.data()), size_);

        parse_entries();
    }

    auto has_entry(const std::string& name) const -> bool {
        return entries_.count(name) > 0;
    }

    auto read_entry(const std::string& name) const -> std::vector<uint8_t> {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            throw std::runtime_error("ZIP entry not found: " + name);
        }

        const auto& entry = it->second;
        if (entry.compression != 0) {
            throw std::runtime_error("Compressed ZIP entries not supported (need stored/uncompressed). "
                                     "Re-save with torch.save(..., _use_new_zipfile_serialization=True)");
        }

        std::vector<uint8_t> result(entry.uncompressed_size);
        std::memcpy(result.data(), data_.data() + entry.offset, entry.uncompressed_size);
        return result;
    }

    auto get_raw_ptr(const std::string& name) const -> std::pair<const uint8_t*, size_t> {
        auto it = entries_.find(name);
        if (it == entries_.end()) {
            return {nullptr, 0};
        }
        const auto& entry = it->second;
        if (entry.compression != 0) {
            return {nullptr, 0};
        }
        return {data_.data() + entry.offset, static_cast<size_t>(entry.uncompressed_size)};
    }

    auto list_entries() const -> std::vector<std::string> {
        std::vector<std::string> names;
        names.reserve(entries_.size());
        for (const auto& [name, _] : entries_) {
            names.push_back(name);
        }
        return names;
    }

private:
    void parse_entries() {
        // Scan for local file headers
        size_t pos = 0;
        while (pos + 30 <= size_) {
            uint32_t sig = read_u32(data_.data() + pos);
            if (sig != ZIP_LOCAL_MAGIC) break;

            uint16_t compression = read_u16(data_.data() + pos + 8);
            uint32_t compressed = read_u32(data_.data() + pos + 18);
            uint32_t uncompressed = read_u32(data_.data() + pos + 22);
            uint16_t name_len = read_u16(data_.data() + pos + 26);
            uint16_t extra_len = read_u16(data_.data() + pos + 28);

            std::string name(reinterpret_cast<const char*>(data_.data() + pos + 30), name_len);
            uint64_t data_offset = pos + 30 + name_len + extra_len;

            ZipEntry entry;
            entry.name = name;
            entry.offset = data_offset;
            entry.compressed_size = compressed;
            entry.uncompressed_size = uncompressed;
            entry.compression = compression;

            entries_[name] = entry;

            pos = data_offset + compressed;
        }
    }

    std::string path_;
    std::ifstream file_;
    std::vector<uint8_t> data_;
    size_t size_{0};
    std::map<std::string, ZipEntry> entries_;
};

// ============================================================================
// Minimal pickle interpreter (restricted to tensor state dicts)
// ============================================================================

// Pickle opcodes we care about
constexpr uint8_t PICKLE_PROTO = 0x80;
constexpr uint8_t PICKLE_STOP = '.';
constexpr uint8_t PICKLE_EMPTY_DICT = '}';
constexpr uint8_t PICKLE_MARK = '(';
constexpr uint8_t PICKLE_SETITEMS = 'u';
constexpr uint8_t PICKLE_SETITEM = 's';
constexpr uint8_t PICKLE_SHORT_BINUNICODE = 0x8C;
constexpr uint8_t PICKLE_BINUNICODE = 'X';
constexpr uint8_t PICKLE_GLOBAL = 'c';
constexpr uint8_t PICKLE_STACK_GLOBAL = 0x93;
constexpr uint8_t PICKLE_REDUCE = 'R';
constexpr uint8_t PICKLE_BUILD = 'b';
constexpr uint8_t PICKLE_BINPUT = 'q';
constexpr uint8_t PICKLE_LONG_BINPUT = 'r';
constexpr uint8_t PICKLE_BINGET = 'h';
constexpr uint8_t PICKLE_LONG_BINGET = 'j';
constexpr uint8_t PICKLE_TUPLE = 't';
constexpr uint8_t PICKLE_TUPLE1 = 0x85;
constexpr uint8_t PICKLE_TUPLE2 = 0x86;
constexpr uint8_t PICKLE_TUPLE3 = 0x87;
constexpr uint8_t PICKLE_EMPTY_TUPLE = ')';
constexpr uint8_t PICKLE_EMPTY_LIST = ']';
constexpr uint8_t PICKLE_APPEND = 'a';
constexpr uint8_t PICKLE_APPENDS = 'e';
constexpr uint8_t PICKLE_BININT1 = 'K';
constexpr uint8_t PICKLE_BININT2 = 'M';
constexpr uint8_t PICKLE_BININT = 'J';
constexpr uint8_t PICKLE_LONG1 = 0x8A;
constexpr uint8_t PICKLE_NONE = 'N';
constexpr uint8_t PICKLE_NEWOBJ = 0x81;
constexpr uint8_t PICKLE_NEWTRUE = 0x88;
constexpr uint8_t PICKLE_NEWFALSE = 0x89;
constexpr uint8_t PICKLE_BINPERSID = 'Q';
constexpr uint8_t PICKLE_SHORT_BINSTRING = 'U';
constexpr uint8_t PICKLE_BINFLOAT = 'G';
constexpr uint8_t PICKLE_BINBYTES = 'B';
constexpr uint8_t PICKLE_SHORT_BINBYTES = 'C';
constexpr uint8_t PICKLE_FRAME = 0x95;

// Simple tagged value for pickle stack
enum class PVal { None, Int, Float, String, List, Dict, Tuple, Class, Persistent, Bool };

struct PickleValue {
    PVal type{PVal::None};
    int64_t int_val{0};
    double float_val{0};
    std::string str_val;
    std::vector<PickleValue> list_val;
    // For dict: alternating key, value pairs in list_val
    // For persistent id: str_val = storage key, list_val = [type_str, data_key, device, numel]
};

// Map PyTorch storage type to Tenzor DType
auto pytorch_type_to_dtype(const std::string& type_str) -> DType {
    if (type_str.find("Float") != std::string::npos ||
        type_str.find("float32") != std::string::npos) return DType::Float32;
    if (type_str.find("Double") != std::string::npos ||
        type_str.find("float64") != std::string::npos) return DType::Float64;
    if (type_str.find("Half") != std::string::npos ||
        type_str.find("float16") != std::string::npos) return DType::Float16;
    if (type_str.find("BFloat16") != std::string::npos ||
        type_str.find("bfloat16") != std::string::npos) return DType::BFloat16;
    if (type_str.find("Long") != std::string::npos ||
        type_str.find("int64") != std::string::npos) return DType::Int64;
    if (type_str.find("Int") != std::string::npos ||
        type_str.find("int32") != std::string::npos) return DType::Int32;
    if (type_str.find("Short") != std::string::npos ||
        type_str.find("int16") != std::string::npos) return DType::Int16;
    if (type_str.find("Char") != std::string::npos ||
        type_str.find("int8") != std::string::npos) return DType::Int8;
    if (type_str.find("Byte") != std::string::npos ||
        type_str.find("uint8") != std::string::npos) return DType::UInt8;
    if (type_str.find("Bool") != std::string::npos ||
        type_str.find("bool") != std::string::npos) return DType::Bool;

    throw std::runtime_error("Unknown PyTorch storage type: " + type_str);
}

// Storage descriptor extracted from pickle persistent IDs
struct StorageDesc {
    std::string data_key;  // e.g., "0", "1" — maps to archive/data/0
    DType dtype;
    int64_t numel;
};

// Simplified pickle parser that extracts the state dict structure
class PickleParser {
public:
    PickleParser(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0) {}

    // Parse and return the top-level dict mapping name → StorageDesc + shape
    auto parse() -> std::map<std::string, std::pair<StorageDesc, std::vector<int64_t>>> {
        // Skip protocol header
        if (pos_ < size_ && data_[pos_] == PICKLE_PROTO) {
            pos_ += 2;  // Skip opcode + protocol version
        }
        // Skip frame opcode if present
        if (pos_ < size_ && data_[pos_] == PICKLE_FRAME) {
            pos_ += 9;  // opcode + 8-byte frame length
        }

        run();
        return result_;
    }

private:
    void run() {
        while (pos_ < size_) {
            uint8_t op = data_[pos_++];

            switch (op) {
                case PICKLE_STOP: return;

                case PICKLE_EMPTY_DICT: {
                    PickleValue v;
                    v.type = PVal::Dict;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_MARK:
                    marks_.push_back(stack_.size());
                    break;

                case PICKLE_SETITEMS: {
                    if (marks_.empty()) break;
                    size_t mark = marks_.back();
                    marks_.pop_back();
                    // Items between mark and top are key-value pairs
                    if (mark > 0 && stack_[mark - 1].type == PVal::Dict) {
                        for (size_t i = mark; i + 1 < stack_.size(); i += 2) {
                            process_dict_item(stack_[i], stack_[i + 1]);
                        }
                    }
                    stack_.resize(mark);
                    break;
                }

                case PICKLE_SETITEM: {
                    if (stack_.size() >= 3) {
                        auto val = std::move(stack_.back()); stack_.pop_back();
                        auto key = std::move(stack_.back()); stack_.pop_back();
                        process_dict_item(key, val);
                    }
                    break;
                }

                case PICKLE_SHORT_BINUNICODE: {
                    uint8_t len = data_[pos_++];
                    PickleValue v;
                    v.type = PVal::String;
                    v.str_val = std::string(reinterpret_cast<const char*>(data_ + pos_), len);
                    pos_ += len;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_BINUNICODE: {
                    uint32_t len = read_u32(data_ + pos_);
                    pos_ += 4;
                    PickleValue v;
                    v.type = PVal::String;
                    v.str_val = std::string(reinterpret_cast<const char*>(data_ + pos_), len);
                    pos_ += len;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_SHORT_BINSTRING: {
                    uint8_t len = data_[pos_++];
                    PickleValue v;
                    v.type = PVal::String;
                    v.str_val = std::string(reinterpret_cast<const char*>(data_ + pos_), len);
                    pos_ += len;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_GLOBAL: {
                    // Read module\nname\n
                    std::string module, name;
                    while (pos_ < size_ && data_[pos_] != '\n') {
                        module += static_cast<char>(data_[pos_++]);
                    }
                    pos_++; // skip \n
                    while (pos_ < size_ && data_[pos_] != '\n') {
                        name += static_cast<char>(data_[pos_++]);
                    }
                    pos_++; // skip \n
                    PickleValue v;
                    v.type = PVal::Class;
                    v.str_val = module + "." + name;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_STACK_GLOBAL: {
                    if (stack_.size() >= 2) {
                        auto name = std::move(stack_.back()); stack_.pop_back();
                        auto module = std::move(stack_.back()); stack_.pop_back();
                        PickleValue v;
                        v.type = PVal::Class;
                        v.str_val = module.str_val + "." + name.str_val;
                        stack_.push_back(std::move(v));
                    }
                    break;
                }

                case PICKLE_REDUCE: {
                    // Pop args and callable, push result
                    if (stack_.size() >= 2) {
                        auto args = std::move(stack_.back()); stack_.pop_back();
                        auto callable = std::move(stack_.back()); stack_.pop_back();
                        // For _rebuild_tensor_v2, keep the info
                        PickleValue v;
                        v.type = PVal::Tuple;
                        v.str_val = callable.str_val;
                        v.list_val = std::move(args.list_val);
                        stack_.push_back(std::move(v));
                    }
                    break;
                }

                case PICKLE_NEWOBJ: {
                    if (stack_.size() >= 2) {
                        stack_.pop_back(); // args
                        // Keep the class
                    }
                    break;
                }

                case PICKLE_BUILD: {
                    if (stack_.size() >= 2) {
                        stack_.pop_back(); // discard state
                    }
                    break;
                }

                case PICKLE_BINPUT: {
                    uint8_t idx = data_[pos_++];
                    if (!stack_.empty()) {
                        memo_[idx] = stack_.back();
                    }
                    break;
                }

                case PICKLE_LONG_BINPUT: {
                    uint32_t idx = read_u32(data_ + pos_);
                    pos_ += 4;
                    if (!stack_.empty()) {
                        memo_[idx] = stack_.back();
                    }
                    break;
                }

                case PICKLE_BINGET: {
                    uint8_t idx = data_[pos_++];
                    auto it = memo_.find(idx);
                    if (it != memo_.end()) {
                        stack_.push_back(it->second);
                    } else {
                        stack_.push_back(PickleValue{});
                    }
                    break;
                }

                case PICKLE_LONG_BINGET: {
                    uint32_t idx = read_u32(data_ + pos_);
                    pos_ += 4;
                    auto it = memo_.find(idx);
                    if (it != memo_.end()) {
                        stack_.push_back(it->second);
                    } else {
                        stack_.push_back(PickleValue{});
                    }
                    break;
                }

                case PICKLE_TUPLE: {
                    if (marks_.empty()) break;
                    size_t mark = marks_.back();
                    marks_.pop_back();
                    PickleValue v;
                    v.type = PVal::Tuple;
                    for (size_t i = mark; i < stack_.size(); ++i) {
                        v.list_val.push_back(std::move(stack_[i]));
                    }
                    stack_.resize(mark);
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_TUPLE1: {
                    if (stack_.size() >= 1) {
                        PickleValue v;
                        v.type = PVal::Tuple;
                        v.list_val.push_back(std::move(stack_.back()));
                        stack_.pop_back();
                        stack_.push_back(std::move(v));
                    }
                    break;
                }

                case PICKLE_TUPLE2: {
                    if (stack_.size() >= 2) {
                        PickleValue v;
                        v.type = PVal::Tuple;
                        auto b = std::move(stack_.back()); stack_.pop_back();
                        auto a = std::move(stack_.back()); stack_.pop_back();
                        v.list_val.push_back(std::move(a));
                        v.list_val.push_back(std::move(b));
                        stack_.push_back(std::move(v));
                    }
                    break;
                }

                case PICKLE_TUPLE3: {
                    if (stack_.size() >= 3) {
                        PickleValue v;
                        v.type = PVal::Tuple;
                        auto c = std::move(stack_.back()); stack_.pop_back();
                        auto b = std::move(stack_.back()); stack_.pop_back();
                        auto a = std::move(stack_.back()); stack_.pop_back();
                        v.list_val.push_back(std::move(a));
                        v.list_val.push_back(std::move(b));
                        v.list_val.push_back(std::move(c));
                        stack_.push_back(std::move(v));
                    }
                    break;
                }

                case PICKLE_EMPTY_TUPLE:
                case PICKLE_EMPTY_LIST: {
                    PickleValue v;
                    v.type = (op == PICKLE_EMPTY_LIST) ? PVal::List : PVal::Tuple;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_APPEND: {
                    if (stack_.size() >= 2) {
                        auto item = std::move(stack_.back()); stack_.pop_back();
                        if (!stack_.empty() && stack_.back().type == PVal::List) {
                            stack_.back().list_val.push_back(std::move(item));
                        }
                    }
                    break;
                }

                case PICKLE_APPENDS: {
                    if (!marks_.empty()) {
                        size_t mark = marks_.back();
                        marks_.pop_back();
                        if (mark > 0 && stack_[mark - 1].type == PVal::List) {
                            for (size_t i = mark; i < stack_.size(); ++i) {
                                stack_[mark - 1].list_val.push_back(std::move(stack_[i]));
                            }
                            stack_.resize(mark);
                        }
                    }
                    break;
                }

                case PICKLE_BININT1: {
                    PickleValue v;
                    v.type = PVal::Int;
                    v.int_val = data_[pos_++];
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_BININT2: {
                    PickleValue v;
                    v.type = PVal::Int;
                    v.int_val = read_u16(data_ + pos_);
                    pos_ += 2;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_BININT: {
                    PickleValue v;
                    v.type = PVal::Int;
                    v.int_val = static_cast<int32_t>(read_u32(data_ + pos_));
                    pos_ += 4;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_LONG1: {
                    uint8_t nbytes = data_[pos_++];
                    int64_t val = 0;
                    for (uint8_t i = 0; i < nbytes && i < 8; ++i) {
                        val |= static_cast<int64_t>(data_[pos_++]) << (i * 8);
                    }
                    // Sign-extend if needed
                    if (nbytes > 0 && nbytes < 8 && (data_[pos_ - 1] & 0x80)) {
                        for (uint8_t i = nbytes; i < 8; ++i) {
                            val |= static_cast<int64_t>(0xFF) << (i * 8);
                        }
                    }
                    PickleValue v;
                    v.type = PVal::Int;
                    v.int_val = val;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_BINFLOAT: {
                    double val;
                    // Big-endian double
                    uint8_t buf[8];
                    for (int i = 0; i < 8; ++i) buf[7 - i] = data_[pos_++];
                    std::memcpy(&val, buf, 8);
                    PickleValue v;
                    v.type = PVal::Float;
                    v.float_val = val;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_NONE: {
                    stack_.push_back(PickleValue{});
                    break;
                }

                case PICKLE_NEWTRUE: {
                    PickleValue v;
                    v.type = PVal::Bool;
                    v.int_val = 1;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_NEWFALSE: {
                    PickleValue v;
                    v.type = PVal::Bool;
                    v.int_val = 0;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_BINPERSID: {
                    // Persistent ID: top of stack is the ID tuple
                    if (!stack_.empty()) {
                        auto id = std::move(stack_.back());
                        stack_.pop_back();
                        PickleValue v;
                        v.type = PVal::Persistent;
                        v.list_val = std::move(id.list_val);
                        stack_.push_back(std::move(v));
                    }
                    break;
                }

                case PICKLE_BINBYTES: {
                    uint32_t len = read_u32(data_ + pos_);
                    pos_ += 4;
                    PickleValue v;
                    v.type = PVal::String;
                    v.str_val = std::string(reinterpret_cast<const char*>(data_ + pos_), len);
                    pos_ += len;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_SHORT_BINBYTES: {
                    uint8_t len = data_[pos_++];
                    PickleValue v;
                    v.type = PVal::String;
                    v.str_val = std::string(reinterpret_cast<const char*>(data_ + pos_), len);
                    pos_ += len;
                    stack_.push_back(std::move(v));
                    break;
                }

                case PICKLE_FRAME: {
                    pos_ += 8;  // Skip 8-byte frame length
                    break;
                }

                default:
                    // Skip unknown opcodes gracefully
                    break;
            }
        }
    }

    void process_dict_item(const PickleValue& key, const PickleValue& val) {
        if (key.type != PVal::String) return;

        // Check if value is a _rebuild_tensor_v2 result with persistent storage
        if (val.type == PVal::Tuple && val.list_val.size() >= 4) {
            // _rebuild_tensor_v2 args: (storage, offset, shape, stride, ...)
            // storage is a persistent ID
            const auto& storage = val.list_val[0];
            if (storage.type == PVal::Persistent && storage.list_val.size() >= 4) {
                // Persistent ID tuple: (storage_type_str, data_key, device, numel)
                StorageDesc desc;
                desc.dtype = pytorch_type_to_dtype(storage.list_val[0].str_val);
                desc.data_key = storage.list_val[1].str_val;
                desc.numel = storage.list_val[3].int_val;

                // Extract shape from the third arg
                std::vector<int64_t> shape;
                if (val.list_val.size() > 2 && val.list_val[2].type == PVal::Tuple) {
                    for (const auto& dim : val.list_val[2].list_val) {
                        shape.push_back(dim.int_val);
                    }
                }

                result_[key.str_val] = {desc, shape};
            }
        }
    }

    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    std::vector<PickleValue> stack_;
    std::vector<size_t> marks_;
    std::map<uint32_t, PickleValue> memo_;

    // Result: tensor_name -> (storage, shape)
    std::map<std::string, std::pair<StorageDesc, std::vector<int64_t>>> result_;
};

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

auto is_pytorch_file(const std::string& path) -> bool {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);
    if (!file.good()) return false;

    // ZIP magic: PK\x03\x04
    return magic[0] == 'P' && magic[1] == 'K' && magic[2] == 3 && magic[3] == 4;
}

auto list_pytorch_tensors(const std::string& path) -> std::vector<std::string> {
    ZipReader zip(path);

    // Find the pickle file
    std::string pkl_name;
    for (const auto& name : zip.list_entries()) {
        if (name.find("data.pkl") != std::string::npos) {
            pkl_name = name;
            break;
        }
    }
    if (pkl_name.empty()) {
        throw std::runtime_error("No data.pkl found in PyTorch checkpoint");
    }

    auto pkl_data = zip.read_entry(pkl_name);
    PickleParser parser(pkl_data.data(), pkl_data.size());
    auto tensors = parser.parse();

    std::vector<std::string> names;
    names.reserve(tensors.size());
    for (const auto& [name, _] : tensors) {
        names.push_back(name);
    }
    return names;
}

auto load_pytorch_state_dict(const std::string& path)
    -> std::unordered_map<std::string, Tensor> {
    ZipReader zip(path);

    // Find the pickle file
    std::string pkl_name;
    std::string data_prefix;
    for (const auto& name : zip.list_entries()) {
        if (name.find("data.pkl") != std::string::npos) {
            pkl_name = name;
            // Derive data prefix: "archive/data.pkl" → "archive/data/"
            auto slash = name.rfind('/');
            if (slash != std::string::npos) {
                data_prefix = name.substr(0, slash) + "/data/";
            } else {
                data_prefix = "data/";
            }
            break;
        }
    }
    if (pkl_name.empty()) {
        throw std::runtime_error("No data.pkl found in PyTorch checkpoint");
    }

    auto pkl_data = zip.read_entry(pkl_name);
    PickleParser parser(pkl_data.data(), pkl_data.size());
    auto tensors = parser.parse();

    std::unordered_map<std::string, Tensor> result;

    for (const auto& [name, info] : tensors) {
        const auto& [desc, shape] = info;

        // Find the data file in the ZIP
        std::string data_name = data_prefix + desc.data_key;

        auto [raw_ptr, raw_size] = zip.get_raw_ptr(data_name);
        if (!raw_ptr) {
            // Try reading as entry
            auto data = zip.read_entry(data_name);
            raw_ptr = data.data();
            raw_size = data.size();
        }

        // Create tensor and copy data
        Tensor tensor(shape, desc.dtype, Device::cpu());
        size_t expected_bytes = tensor.numel() * dtype_size(desc.dtype);

        if (raw_size >= expected_bytes) {
            std::memcpy(tensor.data_ptr(), raw_ptr, expected_bytes);
        } else {
            throw std::runtime_error("Data file too small for tensor '" + name +
                                     "': expected " + std::to_string(expected_bytes) +
                                     " bytes, got " + std::to_string(raw_size));
        }

        result[name] = std::move(tensor);
    }

    return result;
}

} // namespace nn
} // namespace tenzor
