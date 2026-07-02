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
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#endif

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
    if (raw > static_cast<uint32_t>(DType::FP8_E5M2FNUZ)) {  // last enumerator; QInt4x2 was stale and rejected valid FP8 FNUZ tensors
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

// Two checkpoint tensors are "byte identical" when they share dtype, shape,
// and raw contiguous bytes. Used to distinguish a replicated parameter (FSDP/
// DDP saves the same tensor into every rank's shard — e.g. norm weights,
// biases, buffers) from a genuinely dim-0-sharded parameter. Tensors come
// straight out of deserialize_state(), which always builds contiguous CPU
// tensors, so a flat memcmp over numel*elem_size bytes is valid.
auto tensors_byte_identical(const Tensor& a, const Tensor& b) -> bool {
    if (a.dtype() != b.dtype()) {
        return false;
    }
    if (a.ndim() != b.ndim()) {
        return false;
    }
    for (int64_t d = 0; d < a.ndim(); ++d) {
        if (a.shape()[d] != b.shape()[d]) {
            return false;
        }
    }
    auto nbytes = static_cast<size_t>(a.numel()) * dtype_size(a.dtype());
    if (nbytes == 0) {
        return true;  // both empty with matching shape/dtype
    }
    return std::memcmp(a.data_ptr(), b.data_ptr(), nbytes) == 0;
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
    int64_t world_size,
    const std::unordered_map<std::string, ShardSpec>& shard_specs)
    -> std::future<void> {

    // Snapshot tensor data into the serialization buffer now,
    // so the caller can modify tensors immediately after returning
    auto data = serialize_state(state_dict, world_size, shard_specs);
    auto file_path = shard_path(path, rank);

    return std::async(std::launch::async,
        [data = std::move(data), file_path = std::move(file_path)]() {
            // Create parent directories
            std::filesystem::create_directories(file_path.parent_path());

            // Write to a temporary sibling first, flush + fsync it to durable
            // storage, then atomically rename it over the final path. A crash or
            // concurrent reader therefore never observes a half-written shard:
            // either the old file (or nothing) or the complete new one.
            auto tmp_path = file_path;
            tmp_path += ".tmp";

            {
                std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
                if (!ofs.is_open()) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: failed to open file for writing: " +
                        tmp_path.string());
                }
                ofs.write(reinterpret_cast<const char*>(data.data()),
                          static_cast<std::streamsize>(data.size()));
                if (!ofs.good()) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: write error to: " +
                        tmp_path.string());
                }
                ofs.flush();
                if (!ofs.good()) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: flush error to: " +
                        tmp_path.string());
                }
            }  // stream closed here (bytes handed to the OS)

#if defined(__unix__) || defined(__APPLE__)
            // Force the temp file's data to durable storage before the rename,
            // so a power loss immediately after rename cannot leave a named-but-
            // empty checkpoint. fsync() flushes the underlying file regardless of
            // the fd's open mode, so a read-only fd is sufficient.
            {
                int fd = ::open(tmp_path.c_str(), O_RDONLY);
                if (fd < 0) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: failed to reopen for fsync: " +
                        tmp_path.string());
                }
                if (::fsync(fd) != 0) {
                    ::close(fd);
                    throw std::runtime_error(
                        "DistributedCheckpoint: fsync failed for: " +
                        tmp_path.string());
                }
                ::close(fd);
            }
#endif

            std::filesystem::rename(tmp_path, file_path);
        });
}

auto DistributedCheckpoint::save(
    const std::string& path,
    const std::unordered_map<std::string, Tensor>& state_dict,
    int64_t rank,
    int64_t world_size,
    const std::unordered_map<std::string, ShardSpec>& shard_specs) -> void {

    auto future = save_async(path, state_dict, rank, world_size, shard_specs);
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
        if (file_size < 0) {
            throw std::runtime_error(
                "DistributedCheckpoint: failed to size shard: " +
                own_shard.string());
        }
        ifs.seekg(0);
        std::vector<uint8_t> data(static_cast<size_t>(file_size));
        ifs.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(file_size));
        if (!ifs || ifs.gcount() != static_cast<std::streamsize>(file_size)) {
            throw std::runtime_error(
                "DistributedCheckpoint: short read on shard: " +
                own_shard.string());
        }

        auto contents = deserialize_state(data);

        // If world size matches, return directly — no resharding needed
        if (contents.world_size == world_size) {
            return std::move(contents.tensors);
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
            // The rank component may be empty ("rank_.ckpt") or non-numeric
            // ("rank_abc.ckpt"); std::stoll would throw and abort the whole
            // load. Parse defensively and skip any filename whose rank does
            // not fully parse to an integer, rather than propagating.
            try {
                size_t consumed = 0;
                int64_t parsed = std::stoll(rank_str, &consumed);
                if (consumed == rank_str.size()) {
                    saved_ranks.push_back(parsed);
                }
            } catch (...) {
                // malformed rank component — skip this file
            }
        }
    }
    std::sort(saved_ranks.begin(), saved_ranks.end());
    saved_ranks.erase(std::unique(saved_ranks.begin(), saved_ranks.end()),
                      saved_ranks.end());

    if (saved_ranks.empty()) {
        throw std::runtime_error(
            "DistributedCheckpoint: no shard files found in: " + dir.string());
    }

    // The authoritative shard count is the world_size recorded in the shard
    // headers at save time — NOT the number of files that happen to be on disk.
    // Read rank 0's header to learn the saved world_size, then verify that the
    // full contiguous set of shards [0, saved_world_size) is present and that
    // every shard agrees on that world_size. Inferring the count from
    // discovered files would silently truncate the reconstructed tensor (and
    // mis-distribute every per-rank slice) whenever a shard is missing or a
    // filename failed to parse.
    auto read_shard = [&](int64_t sr) -> ShardContents {
        auto sp = shard_path(path, sr);
        std::ifstream ifs(sp, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            throw std::runtime_error(
                "DistributedCheckpoint: failed to open: " + sp.string());
        }
        auto file_size = ifs.tellg();
        if (file_size < 0) {
            throw std::runtime_error(
                "DistributedCheckpoint: failed to size: " + sp.string());
        }
        ifs.seekg(0);
        std::vector<uint8_t> data(static_cast<size_t>(file_size));
        ifs.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(file_size));
        if (!ifs || ifs.gcount() != static_cast<std::streamsize>(file_size)) {
            throw std::runtime_error(
                "DistributedCheckpoint: short read on: " + sp.string());
        }
        return deserialize_state(data);
    };

    if (saved_ranks.front() != 0) {
        throw std::runtime_error(
            "DistributedCheckpoint: missing shard rank_0.ckpt in: " +
            dir.string());
    }
    auto shard0 = read_shard(0);
    auto saved_world_size = shard0.world_size;
    if (saved_world_size < 1) {
        throw std::runtime_error(
            "DistributedCheckpoint: shard header reports invalid world_size " +
            std::to_string(saved_world_size) + " in: " + dir.string());
    }

    // Load all shards in rank order [0, saved_world_size), requiring every
    // expected shard file to exist and to report the same world_size. A missing
    // or extra shard, or a header disagreement, means the checkpoint set is
    // inconsistent and resharding would operate on a truncated/garbled tensor.
    const uint32_t saved_version = shard0.version;
    auto shard0_specs = std::move(shard0.shard_specs);

    std::vector<std::unordered_map<std::string, Tensor>> all_shards;
    all_shards.reserve(static_cast<size_t>(saved_world_size));
    all_shards.push_back(std::move(shard0.tensors));

    for (int64_t sr = 1; sr < saved_world_size; ++sr) {
        auto sp = shard_path(path, sr);
        if (!std::filesystem::exists(sp)) {
            throw std::runtime_error(
                "DistributedCheckpoint: incomplete checkpoint — expected shard "
                "rank_" + std::to_string(sr) + ".ckpt (saved world_size " +
                std::to_string(saved_world_size) + ") not found in: " +
                dir.string());
        }
        auto contents = read_shard(sr);
        if (contents.world_size != saved_world_size) {
            throw std::runtime_error(
                "DistributedCheckpoint: shard rank_" + std::to_string(sr) +
                ".ckpt reports world_size " + std::to_string(contents.world_size) +
                " but rank_0 reports " + std::to_string(saved_world_size) +
                " — inconsistent checkpoint in: " + dir.string());
        }
        all_shards.push_back(std::move(contents.tensors));
    }

    // Any discovered shard with rank >= saved_world_size (e.g. a stale file
    // left over from a larger previous run) makes the on-disk set ambiguous;
    // reject rather than silently ignoring it.
    if (saved_ranks.back() >= saved_world_size) {
        throw std::runtime_error(
            "DistributedCheckpoint: found stale shard rank_" +
            std::to_string(saved_ranks.back()) + ".ckpt beyond saved world_size "
            + std::to_string(saved_world_size) + " in: " + dir.string());
    }

    // For each parameter name, decide whether it is sharded and along which
    // dim, then concatenate the shards along that dim and re-slice for the new
    // world_size. Replicated params are taken as-is from shard 0.
    std::unordered_map<std::string, Tensor> result;

    for (auto& [name, tensor] : all_shards[0]) {
        // A parameter can only be reconstructed by concatenation if it is
        // present in EVERY shard file. If it is missing from any shard it
        // cannot have been sharded across all ranks, so this rank just takes
        // shard 0's copy.
        bool present_in_all = true;
        for (size_t i = 1; i < all_shards.size(); ++i) {
            if (!all_shards[i].contains(name)) {
                present_in_all = false;
                break;
            }
        }

        // Determine whether this parameter is sharded, and along which dim.
        //
        // v2+ checkpoints carry an explicit, per-tensor sharded flag + shard
        // dim written at save time. This is authoritative: byte-identical
        // row-sharded params (e.g. two ranks that happen to hold the same rows)
        // are no longer mis-classified as replicated. Legacy v1 checkpoints
        // have no such metadata, so we fall back to the historical
        // byte-identity heuristic (replicated == byte-identical across shards,
        // always along dim 0) purely for backward compatibility.
        bool is_sharded = false;
        int64_t shard_dim = 0;

        if (saved_version >= VERSION) {
            auto it = shard0_specs.find(name);
            if (it != shard0_specs.end() && it->second.sharded &&
                present_in_all && all_shards.size() > 1) {
                is_sharded = true;
                shard_dim = it->second.dim;
            }
        } else {
            // Legacy v1 heuristic.
            is_sharded = present_in_all && all_shards.size() > 1;
            if (is_sharded) {
                for (size_t i = 1; i < all_shards.size(); ++i) {
                    if (!tensors_byte_identical(tensor, all_shards[i].at(name))) {
                        break;  // genuinely sharded: contents/shape differ
                    }
                    if (i + 1 == all_shards.size()) {
                        // Every comparison identical → replicated, not sharded.
                        is_sharded = false;
                    }
                }
            }
            shard_dim = 0;
        }

        if (!is_sharded) {
            // Replicated, single-shard, or not-present-in-all — this rank
            // takes the parameter as-is from shard 0.
            result[name] = tensor;
            continue;
        }

        // Concatenate all shards along shard_dim to reconstruct the full tensor.
        std::vector<Tensor> chunks;
        chunks.reserve(all_shards.size());
        for (auto& shard : all_shards) {
            chunks.push_back(shard.at(name));
        }
        auto full = cat(chunks, shard_dim);

        // Now slice for this rank's portion in the new world_size.
        //
        // Use a balanced partition: the first (total % world_size) entries get
        // one extra slice each. A tensor whose shard_dim is smaller than
        // world_size cannot be resharded without empty shards, so reject that
        // case explicitly rather than silently concentrating data on one rank.
        auto total = full.shape()[shard_dim];
        if (world_size > total) {
            throw std::runtime_error(
                "DistributedCheckpoint: cannot reshard '" + name + "' (dim-" +
                std::to_string(shard_dim) + " = " + std::to_string(total) +
                ") across world_size " + std::to_string(world_size) +
                " without empty shards");
        }
        auto base = total / world_size;
        auto rem = total % world_size;
        // start = rank*base + min(rank, rem); each of the first `rem` ranks
        // carries one extra slice.
        auto start = rank * base + std::min<int64_t>(rank, rem);
        auto count = base + (rank < rem ? 1 : 0);
        auto end = start + count;

        result[name] = full.slice(shard_dim, start, end);
    }

    return result;
}

auto DistributedCheckpoint::serialize_state(
    const std::unordered_map<std::string, Tensor>& state,
    int64_t world_size,
    const std::unordered_map<std::string, ShardSpec>& shard_specs) const
    -> std::vector<uint8_t> {

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

        // Resolve sharding metadata for this tensor. Anything not explicitly
        // declared sharded is treated as replicated — the safe default that
        // never over-concatenates on load.
        ShardSpec spec{};
        if (auto it = shard_specs.find(name); it != shard_specs.end()) {
            spec = it->second;
            if (spec.sharded) {
                auto ndim_t = cpu_tensor.ndim();
                if (ndim_t == 0) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: tensor '" + name +
                        "' is marked sharded but is a 0-dim scalar");
                }
                // Normalize negative shard dim and bounds-check it now, so a
                // bad spec fails at save time rather than corrupting load.
                if (spec.dim < 0) {
                    spec.dim += ndim_t;
                }
                if (spec.dim < 0 || spec.dim >= ndim_t) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: shard dim " +
                        std::to_string(spec.dim) + " out of range for tensor '" +
                        name + "' with " + std::to_string(ndim_t) + " dims");
                }
            }
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

        // Sharding metadata (v2+)
        write_val(buf, static_cast<uint8_t>(spec.sharded ? 1 : 0));
        write_val(buf, spec.sharded ? spec.dim : int64_t{0});

        // Data
        auto elem_size = dtype_size(cpu_tensor.dtype());
        auto data_bytes = static_cast<int64_t>(cpu_tensor.numel() * elem_size);
        write_val(buf, data_bytes);
        write_bytes(buf, cpu_tensor.data_ptr(), static_cast<size_t>(data_bytes));
    }

    return buf;
}

auto DistributedCheckpoint::deserialize_state(const std::vector<uint8_t>& data) const
    -> ShardContents {

    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    // Header
    auto magic = read_val<uint32_t>(ptr, end);
    if (magic != MAGIC) {
        throw std::runtime_error(
            "DistributedCheckpoint: invalid checkpoint magic number");
    }

    auto version = read_val<uint32_t>(ptr, end);
    if (version != VERSION && version != VERSION_LEGACY_NO_SHARD_META) {
        throw std::runtime_error(
            "DistributedCheckpoint: unsupported checkpoint version " +
            std::to_string(version));
    }
    // v1 files carry no per-tensor sharding metadata; v2+ do.
    const bool has_shard_meta = (version >= VERSION);

    auto world_size = read_val<int64_t>(ptr, end);
    auto num_entries = read_val<int64_t>(ptr, end);
    if (num_entries < 0) {
        throw std::runtime_error(
            "DistributedCheckpoint: negative entry count in checkpoint");
    }

    std::unordered_map<std::string, ShardSpec> shard_specs;
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

        // Sharding metadata (v2+ only). The raw flag is an untrusted byte: treat
        // any non-zero value as "sharded", but require the recorded shard_dim to
        // be a valid axis of this tensor so a malformed file cannot drive an
        // out-of-range cat()/slice() during resharding.
        ShardSpec spec{};
        if (has_shard_meta) {
            auto raw_flag = read_val<uint8_t>(ptr, end);
            auto raw_dim = read_val<int64_t>(ptr, end);
            spec.sharded = (raw_flag != 0);
            spec.dim = raw_dim;
            if (spec.sharded) {
                if (ndim == 0) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: tensor '" + name +
                        "' marked sharded but has 0 dims");
                }
                if (spec.dim < 0 || spec.dim >= ndim) {
                    throw std::runtime_error(
                        "DistributedCheckpoint: invalid shard dim " +
                        std::to_string(spec.dim) + " for tensor '" + name +
                        "' with " + std::to_string(ndim) + " dims");
                }
            }
        }
        shard_specs[name] = spec;

        // Data
        auto data_bytes = read_val<int64_t>(ptr, end);

        // Create tensor and copy data into it. The serialized byte count must
        // exactly match the destination tensor's capacity; otherwise the file is
        // malformed and copying could overflow the heap allocation.
        auto tensor = empty(shape, dtype, Device::cpu());
        // Checked multiply: numel * elem_size can overflow int64 for a crafted
        // shape/dtype combination, wrapping to a small value that would
        // spuriously match a small data_bytes. Mirror the per-dim numel guard
        // above and reject before comparing against the on-disk byte count.
        const int64_t elem_size = static_cast<int64_t>(dtype_size(tensor.dtype()));
        const int64_t numel_bytes = tensor.numel();
        if (elem_size != 0 &&
            numel_bytes > std::numeric_limits<int64_t>::max() / elem_size) {
            throw std::runtime_error(
                "DistributedCheckpoint: tensor byte size overflow for '" +
                name + "'");
        }
        auto expected_bytes = numel_bytes * elem_size;
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

    return ShardContents{std::move(state), world_size, version,
                         std::move(shard_specs)};
}

auto DistributedCheckpoint::shard_path(const std::string& path, int64_t rank) const
    -> std::filesystem::path {
    return std::filesystem::path(config_.storage_path) / path /
           ("rank_" + std::to_string(rank) + ".ckpt");
}

} // namespace tenzor::distributed
