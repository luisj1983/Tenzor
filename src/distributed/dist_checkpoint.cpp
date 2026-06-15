/**
 * @file dist_checkpoint.cpp
 * @brief Implementation of distributed checkpointing
 *
 * Each rank serializes its local state to a binary file. On load, if the
 * world size has changed, all shard files are read and tensors are
 * redistributed across the new set of ranks.
 *
 * Binary format is documented in dist_checkpoint.hpp.
 */

#include "tenzor/distributed/dist_checkpoint.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <limits>

namespace tenzor::distributed {

// ============================================================================
// Helper: byte-level serialization primitives
// ============================================================================

namespace {

template <typename T>
void write_val(std::vector<uint8_t>& buf, T val) {
    auto offset = buf.size();
    buf.resize(offset + sizeof(T));
    std::memcpy(buf.data() + offset, &val, sizeof(T));
}

template <typename T>
auto read_val(const uint8_t*& ptr, const uint8_t* end) -> T {
    if (ptr > end || static_cast<size_t>(end - ptr) < sizeof(T)) {
        throw std::runtime_error(
            "DistributedCheckpoint: truncated checkpoint (unexpected end of data)");
    }
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    ptr += sizeof(T);
    return val;
}

void write_bytes(std::vector<uint8_t>& buf, const void* data, size_t len) {
    auto offset = buf.size();
    buf.resize(offset + len);
    std::memcpy(buf.data() + offset, data, len);
}

// Validate an untrusted dtype field read off disk. DType enumerators are
// contiguous in [0, QInt4x2]; an out-of-range value would make dtype_size()
// return 0 and then divide-by-zero inside empty(). Throw before that happens.
auto checked_dtype(uint32_t raw, const std::string& name) -> DType {
    if (raw > static_cast<uint32_t>(DType::QInt4x2)) {
        throw std::runtime_error(
            "DistributedCheckpoint: invalid dtype value " + std::to_string(raw) +
            " for tensor '" + name + "'");
    }
    auto dt = static_cast<DType>(raw);
    if (dtype_size(dt) == 0) {
        throw std::runtime_error(
            "DistributedCheckpoint: unsupported (zero-size) dtype for tensor '" +
            name + "'");
    }
    return dt;
}

} // anonymous namespace

// ============================================================================
// DistributedCheckpoint Implementation
// ============================================================================

DistributedCheckpoint::DistributedCheckpoint(CheckpointConfig config)
    : config_(std::move(config)) {}

auto DistributedCheckpoint::save_async(
    const std::string& path,
    const std::unordered_map<std::string, Tensor>& state_dict,
    int64_t rank,
    int64_t world_size) -> std::future<void> {

    // Snapshot tensor data into the serialization buffer now,
    // so the caller can modify tensors immediately after returning
    auto data = serialize_state(state_dict, world_size);
    auto file_path = shard_path(path, rank);

    return std::async(std::launch::async,
        [data = std::move(data), file_path = std::move(file_path)]() {
            // Create parent directories
            std::filesystem::create_directories(file_path.parent_path());

            std::ofstream ofs(file_path, std::ios::binary);
            if (!ofs.is_open()) {
                throw std::runtime_error(
                    "DistributedCheckpoint: failed to open file for writing: " +
                    file_path.string());
            }
            ofs.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));
            if (!ofs.good()) {
                throw std::runtime_error(
                    "DistributedCheckpoint: write error to: " +
                    file_path.string());
            }
        });
}

auto DistributedCheckpoint::save(
    const std::string& path,
    const std::unordered_map<std::string, Tensor>& state_dict,
    int64_t rank,
    int64_t world_size) -> void {

    auto future = save_async(path, state_dict, rank, world_size);
    future.get();  // Block until complete
}

auto DistributedCheckpoint::load(
    const std::string& path,
    int64_t rank,
    int64_t world_size) -> std::unordered_map<std::string, Tensor> {

    // Validate rank/world_size before any resharding arithmetic. An
    // uninitialized process group (world_size==0) would otherwise divide by
    // zero, and an out-of-range rank would produce a degenerate/OOB slice.
    if (world_size < 1) {
        throw std::invalid_argument(
            "DistributedCheckpoint::load: world_size must be >= 1, got " +
            std::to_string(world_size));
    }
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "DistributedCheckpoint::load: rank " + std::to_string(rank) +
            " out of range [0, " + std::to_string(world_size) + ")");
    }

    // First, try to read this rank's own shard file
    auto own_shard = shard_path(path, rank);

    if (std::filesystem::exists(own_shard)) {
        // Read the shard file to check the original world size
        std::ifstream ifs(own_shard, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            throw std::runtime_error(
                "DistributedCheckpoint: failed to open: " + own_shard.string());
        }

        auto file_size = ifs.tellg();
        ifs.seekg(0);
        std::vector<uint8_t> data(static_cast<size_t>(file_size));
        ifs.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(file_size));

        auto [state, saved_world_size] = deserialize_state(data);

        // If world size matches, return directly — no resharding needed
        if (saved_world_size == world_size) {
            return state;
        }

        // World size changed: need to load all shards and reshard
        // Fall through to the resharding path below
    }

    // Resharding path: load all shard files from the original save,
    // concatenate tensors, then slice for the new rank/world_size.
    //
    // Discover how many shard files exist
    auto dir = std::filesystem::path(config_.storage_path) / path;
    if (!std::filesystem::exists(dir)) {
        throw std::runtime_error(
            "DistributedCheckpoint: checkpoint directory not found: " +
            dir.string());
    }

    // Find all rank_*.ckpt files
    std::vector<int64_t> saved_ranks;
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        auto fname = entry.path().filename().string();
        if (fname.starts_with("rank_") && fname.ends_with(".ckpt")) {
            auto rank_str = fname.substr(5, fname.size() - 10);  // strip "rank_" and ".ckpt"
            saved_ranks.push_back(std::stoll(rank_str));
        }
    }
    std::sort(saved_ranks.begin(), saved_ranks.end());

    if (saved_ranks.empty()) {
        throw std::runtime_error(
            "DistributedCheckpoint: no shard files found in: " + dir.string());
    }

    int64_t saved_world_size = static_cast<int64_t>(saved_ranks.size());

    // Load all shards
    std::vector<std::unordered_map<std::string, Tensor>> all_shards;
    all_shards.reserve(saved_ranks.size());

    for (auto sr : saved_ranks) {
        auto sp = shard_path(path, sr);
        std::ifstream ifs(sp, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            throw std::runtime_error(
                "DistributedCheckpoint: failed to open: " + sp.string());
        }
        auto file_size = ifs.tellg();
        ifs.seekg(0);
        std::vector<uint8_t> data(static_cast<size_t>(file_size));
        ifs.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(file_size));

        auto [state, ws] = deserialize_state(data);
        all_shards.push_back(std::move(state));
    }

    // For each parameter name, concatenate shards along dim 0, then
    // re-slice for the new world_size
    std::unordered_map<std::string, Tensor> result;

    // Gather all parameter names from the first shard
    for (auto& [name, tensor] : all_shards[0]) {
        // Check if this is a sharded parameter (present in all shards
        // with matching shapes except dim 0)
        bool is_sharded = true;
        for (size_t i = 1; i < all_shards.size(); ++i) {
            auto it = all_shards[i].find(name);
            if (it == all_shards[i].end()) {
                is_sharded = false;
                break;
            }
        }

        if (!is_sharded || all_shards.size() == 1) {
            // Not sharded or only one shard — this rank takes it as-is
            // For replicated tensors, just copy from shard 0
            result[name] = tensor;
            continue;
        }

        // Concatenate all shards along dim 0 to reconstruct the full tensor
        std::vector<Tensor> chunks;
        chunks.reserve(all_shards.size());
        for (auto& shard : all_shards) {
            chunks.push_back(shard.at(name));
        }
        auto full = cat(chunks, 0);

        // Now slice for this rank's portion in the new world_size.
        //
        // Use a balanced partition: the first (total % world_size) ranks get
        // one extra row each, instead of the old floor-division scheme that
        // dumped every remainder row onto the last rank (and handed empty
        // slices to ranks 0..ws-2 whenever world_size > total). A tensor whose
        // dim-0 is smaller than world_size cannot be resharded without empty
        // shards, so reject that case explicitly rather than silently
        // concentrating data on one rank.
        auto total = full.shape()[0];
        if (world_size > total) {
            throw std::runtime_error(
                "DistributedCheckpoint: cannot reshard '" + name + "' (dim-0 = " +
                std::to_string(total) + ") across world_size " +
                std::to_string(world_size) + " without empty shards");
        }
        auto base = total / world_size;
        auto rem = total % world_size;
        // start = rank*base + min(rank, rem); each of the first `rem` ranks
        // carries one extra row.
        auto start = rank * base + std::min<int64_t>(rank, rem);
        auto count = base + (rank < rem ? 1 : 0);
        auto end = start + count;

        result[name] = full.slice(0, start, end);
    }

    return result;
}

auto DistributedCheckpoint::serialize_state(
    const std::unordered_map<std::string, Tensor>& state,
    int64_t world_size) const -> std::vector<uint8_t> {

    std::vector<uint8_t> buf;
    // Reserve a reasonable initial size
    buf.reserve(1024 * 1024);

    // Header
    write_val(buf, MAGIC);
    write_val(buf, VERSION);
    write_val(buf, world_size);
    write_val(buf, static_cast<int64_t>(state.size()));

    // Entries
    for (auto& [name, tensor] : state) {
        // Ensure tensor is contiguous and on CPU for serialization
        auto cpu_tensor = tensor.contiguous();
        if (cpu_tensor.device().type != Device::Type::CPU) {
            cpu_tensor = cpu_tensor.to(Device::cpu());
        }

        // Name
        auto name_len = static_cast<int64_t>(name.size());
        write_val(buf, name_len);
        write_bytes(buf, name.data(), name.size());

        // Shape
        auto ndim = static_cast<int64_t>(cpu_tensor.ndim());
        write_val(buf, ndim);
        for (int64_t d = 0; d < ndim; ++d) {
            write_val(buf, cpu_tensor.shape()[d]);
        }

        // DType
        write_val(buf, static_cast<uint32_t>(cpu_tensor.dtype()));

        // Data
        auto elem_size = dtype_size(cpu_tensor.dtype());
        auto data_bytes = static_cast<int64_t>(cpu_tensor.numel() * elem_size);
        write_val(buf, data_bytes);
        write_bytes(buf, cpu_tensor.data_ptr(), static_cast<size_t>(data_bytes));
    }

    return buf;
}

auto DistributedCheckpoint::deserialize_state(const std::vector<uint8_t>& data) const
    -> std::pair<std::unordered_map<std::string, Tensor>, int64_t> {

    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    // Header
    auto magic = read_val<uint32_t>(ptr, end);
    if (magic != MAGIC) {
        throw std::runtime_error(
            "DistributedCheckpoint: invalid checkpoint magic number");
    }

    auto version = read_val<uint32_t>(ptr, end);
    if (version != VERSION) {
        throw std::runtime_error(
            "DistributedCheckpoint: unsupported checkpoint version " +
            std::to_string(version));
    }

    auto world_size = read_val<int64_t>(ptr, end);
    auto num_entries = read_val<int64_t>(ptr, end);
    if (num_entries < 0) {
        throw std::runtime_error(
            "DistributedCheckpoint: negative entry count in checkpoint");
    }

    std::unordered_map<std::string, Tensor> state;
    // Each entry needs at least name_len + ndim + dtype + data_bytes headers, so
    // num_entries cannot exceed the remaining byte count. Cap the reservation to
    // avoid an attacker-controlled count triggering a huge allocation.
    state.reserve(static_cast<size_t>(
        std::min<int64_t>(num_entries, static_cast<int64_t>(data.size() / 8))));

    for (int64_t i = 0; i < num_entries; ++i) {
        // Name
        auto name_len = read_val<int64_t>(ptr, end);
        if (name_len < 0 || static_cast<size_t>(end - ptr) < static_cast<size_t>(name_len)) {
            throw std::runtime_error(
                "DistributedCheckpoint: invalid or truncated tensor name");
        }
        std::string name(reinterpret_cast<const char*>(ptr),
                         static_cast<size_t>(name_len));
        ptr += name_len;

        // Shape
        auto ndim = read_val<int64_t>(ptr, end);
        // Each dimension consumes 8 bytes; bound ndim by the remaining data so a
        // bogus value cannot drive a huge std::vector allocation.
        if (ndim < 0 || static_cast<size_t>(end - ptr) < static_cast<size_t>(ndim) * sizeof(int64_t)) {
            throw std::runtime_error(
                "DistributedCheckpoint: invalid or truncated tensor shape for '" +
                name + "'");
        }
        std::vector<int64_t> shape(static_cast<size_t>(ndim));
        int64_t numel = 1;
        for (int64_t d = 0; d < ndim; ++d) {
            shape[d] = read_val<int64_t>(ptr, end);
            if (shape[d] < 0) {
                throw std::runtime_error(
                    "DistributedCheckpoint: negative dimension in shape for '" +
                    name + "'");
            }
            // Checked multiply to detect numel overflow.
            if (shape[d] != 0 && numel > std::numeric_limits<int64_t>::max() / shape[d]) {
                throw std::runtime_error(
                    "DistributedCheckpoint: tensor element count overflow for '" +
                    name + "'");
            }
            numel *= shape[d];
        }

        // DType — validated against the known enumerator set so an untrusted
        // out-of-range value cannot reach empty()/dtype_size() (which would
        // divide by a zero element size → SIGFPE).
        auto dtype = checked_dtype(read_val<uint32_t>(ptr, end), name);

        // Data
        auto data_bytes = read_val<int64_t>(ptr, end);

        // Create tensor and copy data into it. The serialized byte count must
        // exactly match the destination tensor's capacity; otherwise the file is
        // malformed and copying could overflow the heap allocation.
        auto tensor = empty(shape, dtype, Device::cpu());
        auto expected_bytes = static_cast<int64_t>(
            tensor.numel() * dtype_size(tensor.dtype()));
        if (data_bytes != expected_bytes) {
            throw std::runtime_error(
                "DistributedCheckpoint: data size mismatch for '" + name +
                "' (expected " + std::to_string(expected_bytes) + ", got " +
                std::to_string(data_bytes) + ")");
        }
        if (data_bytes > 0) {
            if (static_cast<size_t>(end - ptr) < static_cast<size_t>(data_bytes)) {
                throw std::runtime_error(
                    "DistributedCheckpoint: truncated checkpoint data for '" +
                    name + "'");
            }
            std::memcpy(tensor.data_ptr(), ptr, static_cast<size_t>(data_bytes));
            ptr += data_bytes;
        }

        state[name] = std::move(tensor);
    }

    return {std::move(state), world_size};
}

auto DistributedCheckpoint::shard_path(const std::string& path, int64_t rank) const
    -> std::filesystem::path {
    return std::filesystem::path(config_.storage_path) / path /
           ("rank_" + std::to_string(rank) + ".ckpt");
}

} // namespace tenzor::distributed
