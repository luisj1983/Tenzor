/**
 * @file tensorboard.cpp
 * @brief Implementation of TensorBoard SummaryWriter
 *
 * Implements TensorBoard event file format with protobuf-like serialization.
 * Uses a simplified binary format compatible with TensorBoard's event reader.
 */

#include "tenzor/utils/tensorboard.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/logging.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/io/image.hpp"
#include <filesystem>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <ctime>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <sys/types.h>
#include <limits.h>
#include <thread>
#include <atomic>

namespace tenzor {

namespace {

// TensorBoard event format helpers
constexpr uint32_t kMaskedCrc32c = 0xA282EAD8;

// CRC-32C (Castagnoli, reflected polynomial 0x82F63B78). The TFRecord framing
// REQUIRES this exact CRC; the previous hand-rolled formula was neither CRC32
// nor CRC32C, so real TensorBoard rejected every record and showed no data.
uint32_t crc32c(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc & 1u) ? (crc >> 1) ^ 0x82F63B78u : (crc >> 1);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

// TensorFlow's masked CRC used in the TFRecord frame:
//   masked = ((crc >> 15) | (crc << 17)) + 0xa282ead8
uint32_t masked_crc32c(const uint8_t* data, size_t length) {
    uint32_t crc = crc32c(data, length);
    return ((crc >> 15) | (crc << 17)) + kMaskedCrc32c;
}

// Write little-endian uint64
void write_uint64_le(std::vector<uint8_t>& buffer, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        buffer.push_back(static_cast<uint8_t>(value & 0xFF));
        value >>= 8;
    }
}

// Write little-endian uint32
void write_uint32_le(std::vector<uint8_t>& buffer, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        buffer.push_back(static_cast<uint8_t>(value & 0xFF));
        value >>= 8;
    }
}

// Write string with length prefix
void write_string(std::vector<uint8_t>& buffer, std::string_view str) {
    // Varint encoding for length
    size_t len = str.length();
    while (len >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((len & 0x7F) | 0x80));
        len >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(len));

    // String data
    buffer.insert(buffer.end(), str.begin(), str.end());
}

// Protobuf field encoding
void write_field_header(std::vector<uint8_t>& buffer, uint32_t field_number, uint32_t wire_type) {
    uint32_t tag = (field_number << 3) | wire_type;
    // Tags above 15 need multi-byte varint encoding (field_number > 15 -> tag >= 128).
    while (tag >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((tag & 0x7F) | 0x80));
        tag >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(tag));
}

// Write a base-128 varint.
void write_varint(std::vector<uint8_t>& buffer, uint64_t value) {
    while (value >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(value));
}

// Write a length-delimited field: tag + varint(length) + payload.
void write_length_delimited(std::vector<uint8_t>& buffer,
                            uint32_t field_number,
                            const std::vector<uint8_t>& payload) {
    write_field_header(buffer, field_number, 2);
    write_varint(buffer, payload.size());
    buffer.insert(buffer.end(), payload.begin(), payload.end());
}

// Write a length-delimited string field: tag + varint(len) + bytes.
void write_string_field(std::vector<uint8_t>& buffer,
                        uint32_t field_number,
                        std::string_view str) {
    write_field_header(buffer, field_number, 2);
    write_varint(buffer, str.size());
    buffer.insert(buffer.end(), str.begin(), str.end());
}

} // anonymous namespace

// Implementation struct (Pimpl idiom)
struct SummaryWriter::Impl {
    std::string log_dir;
    std::string event_file_path;
    std::ofstream event_file;
    std::mutex mutex;
    int max_queue;
    int flush_secs;
    int event_count{0};
    std::chrono::steady_clock::time_point last_flush;
    // Atomic so the lock-free public is_open() accessor and the destructor's
    // pre-close check can observe it without holding `mutex` while the writers
    // (which hold `mutex`) toggle it. All mutations occur under `mutex`.
    std::atomic<bool> is_open{false};
    double wall_time_offset{0.0};

    Impl(std::string_view dir, int queue, int flush)
        : log_dir(dir), max_queue(queue), flush_secs(flush) {
        last_flush = std::chrono::steady_clock::now();
        wall_time_offset = get_current_wall_time();
    }

    double get_current_wall_time() const {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration - seconds);
        return seconds.count() + microseconds.count() / 1e6;
    }

    // A flush is due when either the in-memory event queue has reached max_queue,
    // OR (for a positive flush_secs) at least flush_secs of wall time has elapsed
    // since the last flush. The latter bounds how long buffered events can sit in
    // the ofstream buffer and be lost on a crash, honoring the documented
    // periodic-flush contract.
    bool maybe_flush_due() const {
        if (event_count >= max_queue) {
            return true;
        }
        if (flush_secs > 0) {
            auto elapsed = std::chrono::steady_clock::now() - last_flush;
            if (elapsed >= std::chrono::seconds(flush_secs)) {
                return true;
            }
        }
        return false;
    }

    // Flush the underlying ofstream. MUST be called with `mutex` already held;
    // it touches `event_file` and `last_flush`, both of which are otherwise
    // mutated only under the lock. The public SummaryWriter::flush() acquires
    // the lock and delegates here, while the already-locked write paths
    // (add_*/close) call this directly to avoid self-deadlock.
    void flush_locked() {
        if (!is_open) {
            return;
        }
        event_file.flush();
        last_flush = std::chrono::steady_clock::now();
    }
};

SummaryWriter::SummaryWriter(std::string_view log_dir, int max_queue, int flush_secs)
    : impl_(std::make_unique<Impl>(log_dir, max_queue, flush_secs)) {

    // Create log directory if it doesn't exist
    ensure_directory_exists(log_dir);

    // Create event file
    impl_->event_file_path = impl_->log_dir + "/" + create_event_filename();
    impl_->event_file.open(impl_->event_file_path, std::ios::binary | std::ios::out);

    if (!impl_->event_file.is_open()) {
        throw TensorBoardException(
            std::format("Failed to open event file: {}", impl_->event_file_path).c_str());
    }

    impl_->is_open = true;

    TENZOR_LOG_INFO(std::format("TensorBoard: Writing to {}", impl_->event_file_path));

    // Write initial event with version info
    std::vector<uint8_t> version_event;
    write_field_header(version_event, 1, 1); // wall_time (double, wire_type=1)

    double wall_time = impl_->get_current_wall_time();
    uint64_t wall_time_bits;
    std::memcpy(&wall_time_bits, &wall_time, sizeof(double));
    write_uint64_le(version_event, wall_time_bits);

    write_field_header(version_event, 2, 0); // step (int64, wire_type=0)
    version_event.push_back(0); // step = 0

    write_field_header(version_event, 5, 2); // file_version (string, wire_type=2)
    write_string(version_event, "brain.Event:2");

    write_event("__version__", version_event, 0);
}

SummaryWriter::~SummaryWriter() {
    if (impl_ && impl_->is_open) {
        try {
            close();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

auto SummaryWriter::add_scalar(std::string_view tag, float value, int64_t step) -> void {
    // Early closed-writer check: throw before any work so a caller that uses
    // the writer after close() gets the exception (matches add_histogram /
    // add_image, which throw on the same condition). This unlocked check is
    // racy with a concurrent close(); the locked re-check below handles that
    // race by silently no-op-ing instead of writing to a closed stream.
    if (!impl_->is_open) {
        throw TensorBoardException("SummaryWriter is closed");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Re-check under the lock: the early is_open check above is racy with a
    // concurrent close() that resets the stream/impl state.
    if (!impl_->is_open) return;

    auto data = serialize_scalar(value);
    write_event(tag, data, step, /*is_scalar=*/true);

    // Auto-flush if needed
    impl_->event_count++;
    if (impl_->maybe_flush_due()) {
        impl_->flush_locked();
        impl_->event_count = 0;
    }
}

auto SummaryWriter::add_histogram(std::string_view tag,
                                  const Tensor& tensor,
                                  int64_t step,
                                  int bins) -> void {
    if (!impl_->is_open) {
        throw TensorBoardException("SummaryWriter is closed");
    }

    // Validate the caller-controlled bin count before any allocation/copy:
    // bins <= 0 leads to OOB writes / huge allocations in serialize_histogram.
    if (bins < 1 || bins > 1'000'000) {
        throw TensorBoardException("Histogram bins must be in [1, 1000000]");
    }

    // Ensure tensor is on CPU
    Tensor cpu_tensor = tensor.device().type == Device::Type::CPU ?
                        tensor : tensor.cpu();

    // serialize_histogram reads data<float>() linearly over [0, numel), so the
    // tensor must be Float32 and contiguous (otherwise data<float>() throws on a
    // dtype mismatch or reads non-contiguous storage in the wrong order).
    if (cpu_tensor.dtype() != DType::Float32) {
        cpu_tensor = cpu_tensor.to(DType::Float32);
    }
    cpu_tensor = cpu_tensor.contiguous();

    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Re-check under the lock: the early is_open fast-path above is racy with a
    // concurrent close() that resets the stream/impl state.
    if (!impl_->is_open) return;

    auto data = serialize_histogram(cpu_tensor, bins);
    write_event(tag, data, step);

    impl_->event_count++;
    if (impl_->maybe_flush_due()) {
        impl_->flush_locked();
        impl_->event_count = 0;
    }
}

auto SummaryWriter::add_image(std::string_view tag,
                             const Tensor& tensor,
                             int64_t step,
                             [[maybe_unused]] std::string_view dataformats) -> void {
    if (!impl_->is_open) {
        throw TensorBoardException("SummaryWriter is closed");
    }

    // Validate image shape
    auto shape = tensor.shape();
    if (shape.size() != 3) {
        throw TensorBoardException(
            std::format("Image tensor must be 3D [C, H, W], got {}D", shape.size()).c_str());
    }

    int64_t channels = shape[0];
    if (channels != 1 && channels != 3 && channels != 4) {
        throw TensorBoardException(
            std::format("Image channels must be 1, 3, or 4, got {}", channels).c_str());
    }

    // Ensure tensor is on CPU
    Tensor cpu_tensor = tensor.device().type == Device::Type::CPU ?
                        tensor : tensor.cpu();

    // serialize_image reads data<float>() linearly over [0, numel), so the
    // tensor must be Float32 and contiguous (otherwise data<float>() throws on a
    // dtype mismatch or scrambles pixels from a non-contiguous view).
    if (cpu_tensor.dtype() != DType::Float32) {
        cpu_tensor = cpu_tensor.to(DType::Float32);
    }
    cpu_tensor = cpu_tensor.contiguous();

    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Re-check under the lock: the early is_open fast-path above is racy with a
    // concurrent close() that resets the stream/impl state.
    if (!impl_->is_open) return;

    auto data = serialize_image(cpu_tensor);
    write_event(tag, data, step);

    impl_->event_count++;
    if (impl_->maybe_flush_due()) {
        impl_->flush_locked();
        impl_->event_count = 0;
    }
}

namespace {

// Encode a single NodeDef message body (without outer tag/length).
//   field 1: name   (string)
//   field 2: op     (string)
//   field 3: input  (repeated string)
auto encode_node_def(std::string_view name,
                     std::string_view op,
                     const std::vector<std::string>& inputs) -> std::vector<uint8_t> {
    std::vector<uint8_t> node;
    write_string_field(node, 1, name);
    write_string_field(node, 2, op);
    for (const auto& in : inputs) {
        write_string_field(node, 3, in);
    }
    return node;
}

// BFS the autograd graph reachable from `root` via Function::next_functions().
//
// We assign a sequential id to every Function we visit (in discovery order),
// then emit one NodeDef per Function with op = Function::name() and inputs =
// the names assigned to its `next_functions()` (i.e. upstream gradient
// producers — equivalently, the forward-pass inputs to this op).
struct GraphDefResult {
    std::vector<uint8_t> bytes;
    size_t node_count{0};
};

auto build_graph_def(const std::shared_ptr<Function>& root) -> GraphDefResult {
    GraphDefResult out;
    if (!root) {
        return out;  // Empty GraphDef — leaf-only Variable.
    }

    // Discover all reachable Functions and assign stable ids.
    std::unordered_map<Function*, std::string> name_for;
    std::vector<std::shared_ptr<Function>> order;
    std::queue<std::shared_ptr<Function>> q;
    q.push(root);
    name_for[root.get()] = "node_0";
    order.push_back(root);

    while (!q.empty()) {
        auto fn = q.front();
        q.pop();
        for (const auto& nxt : fn->next_functions()) {
            if (!nxt) continue;
            if (name_for.find(nxt.get()) != name_for.end()) continue;
            std::string id = "node_" + std::to_string(name_for.size());
            name_for[nxt.get()] = id;
            order.push_back(nxt);
            q.push(nxt);
        }
    }

    // Emit nodes. GraphDef field 1 = repeated NodeDef.
    for (const auto& fn : order) {
        std::vector<std::string> input_names;
        input_names.reserve(fn->next_functions().size());
        for (const auto& nxt : fn->next_functions()) {
            if (!nxt) continue;
            auto it = name_for.find(nxt.get());
            if (it != name_for.end()) {
                input_names.push_back(it->second);
            }
        }
        auto node_bytes = encode_node_def(name_for[fn.get()], fn->name(), input_names);
        write_length_delimited(out.bytes, /*field_number=*/1, node_bytes);
    }
    out.node_count = order.size();
    return out;
}

} // anonymous namespace

auto SummaryWriter::add_graph(std::string_view model_name,
                             const Variable& output) -> void {
    // Early closed-writer check: throw before any work so a caller that uses
    // the writer after close() gets the exception (matches add_histogram /
    // add_image, which throw on the same condition). This unlocked check is
    // racy with a concurrent close(); the locked re-check below handles that
    // race by silently no-op-ing instead of writing to a closed stream.
    if (!impl_->is_open) {
        throw TensorBoardException("SummaryWriter is closed");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Re-check under the lock: the early is_open check above is racy with a
    // concurrent close() that resets the stream/impl state.
    if (!impl_->is_open) return;

    // Walk the autograd graph rooted at `output.grad_fn()` and build a real
    // GraphDef protobuf message.
    auto graph_def_result = build_graph_def(output.grad_fn());
    const auto& graph_def_bytes = graph_def_result.bytes;

    // Wrap the GraphDef in an event's `graph_def` field. The Event proto
    // (tensorflow/core/util/event.proto) has:
    //   field 1: wall_time (double)
    //   field 2: step      (int64)
    //   field 4: summary   (Summary)
    //   field 5: file_version (string)
    //   field 6: graph_def (bytes)  <-- serialized GraphDef
    //
    // We emit a *standalone* event whose only payload is graph_def, which is
    // how TensorBoard's reader recognises and routes it to the graph plugin.
    std::vector<uint8_t> event;

    // wall_time
    write_field_header(event, 1, 1);
    double wall_time = impl_->get_current_wall_time();
    uint64_t wall_time_bits;
    std::memcpy(&wall_time_bits, &wall_time, sizeof(double));
    write_uint64_le(event, wall_time_bits);

    // step = 0
    write_field_header(event, 2, 0);
    event.push_back(0);

    // graph_def (bytes, field 6)
    write_length_delimited(event, /*field_number=*/6, graph_def_bytes);

    // TFRecord-frame the event and append.
    uint64_t length = event.size();
    std::vector<uint8_t> length_bytes;
    write_uint64_le(length_bytes, length);
    uint32_t length_crc = masked_crc32c(length_bytes.data(), length_bytes.size());
    uint32_t data_crc = masked_crc32c(event.data(), event.size());

    // CRC words must be little-endian per the TFRecord spec; serialize through
    // write_uint32_le rather than dumping host-order bytes of the uint32.
    std::vector<uint8_t> length_crc_bytes;
    write_uint32_le(length_crc_bytes, length_crc);
    std::vector<uint8_t> data_crc_bytes;
    write_uint32_le(data_crc_bytes, data_crc);

    impl_->event_file.write(reinterpret_cast<const char*>(length_bytes.data()), 8);
    impl_->event_file.write(reinterpret_cast<const char*>(length_crc_bytes.data()), 4);
    impl_->event_file.write(reinterpret_cast<const char*>(event.data()), event.size());
    impl_->event_file.write(reinterpret_cast<const char*>(data_crc_bytes.data()), 4);

    TENZOR_LOG_INFO(std::format("TensorBoard: Added graph for {} ({} nodes, {} bytes)",
                                model_name,
                                graph_def_result.node_count,
                                graph_def_bytes.size()));
}

auto SummaryWriter::flush() -> void {
    // Take the same mutex the write paths use so a public flush() can never race
    // an in-progress event_file.write() from add_*/add_graph. The locked write
    // paths must NOT call this (they would self-deadlock); they call
    // impl_->flush_locked() directly.
    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Re-check under the lock: the early is_open fast-path above is racy with a
    // concurrent close() that resets the stream/impl state.
    if (!impl_->is_open) return;
    impl_->flush_locked();
}

auto SummaryWriter::close() -> void {
    if (!impl_->is_open) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    // Re-check under the lock: the early is_open fast-path above is racy with a
    // concurrent close() that resets the stream/impl state.
    if (!impl_->is_open) return;

    impl_->flush_locked();
    impl_->event_file.close();
    impl_->is_open = false;

    TENZOR_LOG_INFO("TensorBoard: Writer closed");
}

auto SummaryWriter::is_open() const -> bool {
    return impl_->is_open;
}

// Private helper methods

auto SummaryWriter::write_event(std::string_view tag,
                                const std::vector<uint8_t>& data,
                                int64_t step,
                                bool is_scalar) -> void {
    // Build complete event
    std::vector<uint8_t> event;

    // Field 1: wall_time (double)
    write_field_header(event, 1, 1);
    double wall_time = impl_->get_current_wall_time();
    uint64_t wall_time_bits;
    std::memcpy(&wall_time_bits, &wall_time, sizeof(double));
    write_uint64_le(event, wall_time_bits);

    // Field 2: step (int64)
    write_field_header(event, 2, 0);
    // Varint encoding for step
    uint64_t s = static_cast<uint64_t>(step);
    while (s >= 0x80) {
        event.push_back(static_cast<uint8_t>((s & 0x7F) | 0x80));
        s >>= 7;
    }
    event.push_back(static_cast<uint8_t>(s));

    // Field 4: summary
    write_field_header(event, 4, 2);

    // Build summary
    std::vector<uint8_t> summary;

    // Summary.value (repeated)
    write_field_header(summary, 1, 2);

    // Summary.Value message
    std::vector<uint8_t> value;

    // tag (string)
    write_field_header(value, 1, 2);
    write_string(value, tag);

    // simple_value (float) or other data. The encoding is selected by the
    // explicit is_scalar flag passed from the caller (add_scalar vs.
    // add_histogram/add_image), never inferred from data.size(): a scalar
    // payload is the raw 4-byte float to wrap as simple_value, while any other
    // payload is a pre-wrapped field body appended verbatim.
    if (is_scalar) {
        // Scalar value
        write_field_header(value, 2, 5); // simple_value (float, wire_type=5)
        value.insert(value.end(), data.begin(), data.end());
    } else {
        // Complex value (histogram, image, etc.)
        value.insert(value.end(), data.begin(), data.end());
    }

    // Write value length
    size_t value_len = value.size();
    while (value_len >= 0x80) {
        summary.push_back(static_cast<uint8_t>((value_len & 0x7F) | 0x80));
        value_len >>= 7;
    }
    summary.push_back(static_cast<uint8_t>(value_len));
    summary.insert(summary.end(), value.begin(), value.end());

    // Write summary length
    size_t summary_len = summary.size();
    while (summary_len >= 0x80) {
        event.push_back(static_cast<uint8_t>((summary_len & 0x7F) | 0x80));
        summary_len >>= 7;
    }
    event.push_back(static_cast<uint8_t>(summary_len));
    event.insert(event.end(), summary.begin(), summary.end());

    // Write to file with TFRecord format
    // Format: uint64_le(length) | uint32_le(masked_crc32c of length) | data | uint32_le(masked_crc32c of data)

    uint64_t length = event.size();
    std::vector<uint8_t> length_bytes;
    write_uint64_le(length_bytes, length);

    uint32_t length_crc = masked_crc32c(length_bytes.data(), length_bytes.size());

    uint32_t data_crc = masked_crc32c(event.data(), event.size());

    // CRC words must be little-endian per the TFRecord spec; serialize through
    // write_uint32_le rather than dumping host-order bytes of the uint32.
    std::vector<uint8_t> length_crc_bytes;
    write_uint32_le(length_crc_bytes, length_crc);
    std::vector<uint8_t> data_crc_bytes;
    write_uint32_le(data_crc_bytes, data_crc);

    // Write record
    impl_->event_file.write(reinterpret_cast<const char*>(length_bytes.data()), 8);
    impl_->event_file.write(reinterpret_cast<const char*>(length_crc_bytes.data()), 4);
    impl_->event_file.write(reinterpret_cast<const char*>(event.data()), event.size());
    impl_->event_file.write(reinterpret_cast<const char*>(data_crc_bytes.data()), 4);
}

auto SummaryWriter::serialize_scalar(float value) -> std::vector<uint8_t> {
    std::vector<uint8_t> buffer;

    // Just return the float as bytes (will be wrapped as simple_value)
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    write_uint32_le(buffer, bits);

    return buffer;
}

auto SummaryWriter::serialize_histogram(const Tensor& tensor, int bins) -> std::vector<uint8_t> {
    // Validate bins before allocating: bins <= 0 would make counts[] zero-length
    // (the all-equal branch then writes counts[0] OOB, and the clamp
    // min(max(bin,0), bins-1) == -1 writes counts[-1]), while a negative bins
    // requests a huge std::vector<double> allocation.
    if (bins < 1 || bins > 1'000'000) {
        throw TensorBoardException("Histogram bins must be in [1, 1000000]");
    }

    // Calculate histogram statistics
    const float* data = tensor.data<float>();
    size_t numel = static_cast<size_t>(tensor.numel());

    if (numel == 0) {
        throw TensorBoardException("Cannot create histogram from empty tensor");
    }

    // Find min/max
    float min_val = data[0];
    float max_val = data[0];
    double sum = 0.0;
    double sum_squares = 0.0;

    for (size_t i = 0; i < numel; ++i) {
        float val = data[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        sum_squares += val * val;
    }

    // Calculate histogram bins
    std::vector<int64_t> counts(bins, 0);
    std::vector<double> limits(bins + 1);

    if (max_val > min_val) {
        double range = max_val - min_val;
        for (int i = 0; i <= bins; ++i) {
            limits[i] = min_val + (i * range / bins);
        }

        for (size_t i = 0; i < numel; ++i) {
            float val = data[i];
            int bin = static_cast<int>((val - min_val) / range * bins);
            bin = std::min(std::max(bin, 0), bins - 1);
            counts[bin]++;
        }
    } else {
        // All values are the same
        limits[0] = min_val;
        for (int i = 1; i <= bins; ++i) {
            limits[i] = max_val;
        }
        counts[0] = numel;
    }

    // Create histogram protobuf message
    std::vector<uint8_t> buffer;

    // Field 7: histo (HistogramProto, wire_type=2)
    write_field_header(buffer, 7, 2);

    std::vector<uint8_t> histo;

    // min
    write_field_header(histo, 1, 1);
    uint64_t min_bits;
    double min_d = static_cast<double>(min_val);
    std::memcpy(&min_bits, &min_d, sizeof(double));
    write_uint64_le(histo, min_bits);

    // max
    write_field_header(histo, 2, 1);
    uint64_t max_bits;
    double max_d = static_cast<double>(max_val);
    std::memcpy(&max_bits, &max_d, sizeof(double));
    write_uint64_le(histo, max_bits);

    // num
    write_field_header(histo, 3, 1);
    uint64_t num_bits;
    double num_d = static_cast<double>(numel);
    std::memcpy(&num_bits, &num_d, sizeof(double));
    write_uint64_le(histo, num_bits);

    // sum
    write_field_header(histo, 4, 1);
    uint64_t sum_bits;
    std::memcpy(&sum_bits, &sum, sizeof(double));
    write_uint64_le(histo, sum_bits);

    // sum_squares
    write_field_header(histo, 5, 1);
    uint64_t sum_sq_bits;
    std::memcpy(&sum_sq_bits, &sum_squares, sizeof(double));
    write_uint64_le(histo, sum_sq_bits);

    // bucket_limit (repeated double)
    for (int i = 0; i < bins; ++i) {
        write_field_header(histo, 6, 1);
        uint64_t limit_bits;
        std::memcpy(&limit_bits, &limits[i + 1], sizeof(double));
        write_uint64_le(histo, limit_bits);
    }

    // bucket (repeated int64)
    for (int i = 0; i < bins; ++i) {
        write_field_header(histo, 7, 0);
        uint64_t count = static_cast<uint64_t>(counts[i]);
        while (count >= 0x80) {
            histo.push_back(static_cast<uint8_t>((count & 0x7F) | 0x80));
            count >>= 7;
        }
        histo.push_back(static_cast<uint8_t>(count));
    }

    // Write histo length
    size_t histo_len = histo.size();
    while (histo_len >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((histo_len & 0x7F) | 0x80));
        histo_len >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(histo_len));
    buffer.insert(buffer.end(), histo.begin(), histo.end());

    return buffer;
}

auto SummaryWriter::serialize_image(const Tensor& tensor) -> std::vector<uint8_t> {
    // Get image dimensions
    auto shape = tensor.shape();
    int64_t channels = shape[0];
    int64_t height = shape[1];
    int64_t width = shape[2];

    // Create image protobuf message
    std::vector<uint8_t> buffer;

    // Field 8: image (Image, wire_type=2)
    write_field_header(buffer, 8, 2);

    std::vector<uint8_t> image;

    // height
    write_field_header(image, 1, 0);
    uint64_t h = static_cast<uint64_t>(height);
    while (h >= 0x80) {
        image.push_back(static_cast<uint8_t>((h & 0x7F) | 0x80));
        h >>= 7;
    }
    image.push_back(static_cast<uint8_t>(h));

    // width
    write_field_header(image, 2, 0);
    uint64_t w = static_cast<uint64_t>(width);
    while (w >= 0x80) {
        image.push_back(static_cast<uint8_t>((w & 0x7F) | 0x80));
        w >>= 7;
    }
    image.push_back(static_cast<uint8_t>(w));

    // colorspace (1=grayscale, 3=RGB, 4=RGBA)
    write_field_header(image, 3, 0);
    image.push_back(static_cast<uint8_t>(channels));

    // encoded_image_string: TensorBoard's image plugin requires actual encoded
    // image bytes (PNG/JPEG) in field 4 — raw pixel bytes do not decode and
    // render nothing.
    write_field_header(image, 4, 2);

    const float* data = tensor.data<float>();
    size_t numel = static_cast<size_t>(tensor.numel());

    // Decide normalization from the tensor's GLOBAL value range, not per pixel
    // (per-pixel scaling mixes scales within one image). Track BOTH the min and
    // the max: tracking only the max black-clamps negative-valued images (e.g.
    // tanh/GAN outputs in [-1, 1]) because the negative half clamps to 0.
    float global_min = numel > 0 ? data[0] : 0.0f;
    float global_max = numel > 0 ? data[0] : 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        global_min = std::min(global_min, data[i]);
        global_max = std::max(global_max, data[i]);
    }

    // Affine map x -> x * gain + bias chosen so the source range lands in
    // [0, 255]:
    //   [0, 1]    normalized        -> x * 255
    //   [-1, 1]   signed normalized -> (x + 1) / 2 * 255
    //   [0, 255]  already byte-range -> x
    //   otherwise general [min, max] -> [0, 255] rescale
    float gain;
    float bias;
    if (global_min >= 0.0f && global_max <= 1.0f) {
        gain = 255.0f;
        bias = 0.0f;
    } else if (global_min >= -1.0f && global_max <= 1.0f) {
        gain = 127.5f;
        bias = 127.5f;
    } else if (global_min >= 0.0f && global_max <= 255.0f) {
        gain = 1.0f;
        bias = 0.0f;
    } else {
        const float range = global_max - global_min;
        gain = range > 0.0f ? 255.0f / range : 0.0f;
        bias = -global_min * gain;
    }

    // Convert to a contiguous uint8 CHW buffer, then encode as PNG.
    std::vector<uint8_t> pixels(numel);
    for (size_t i = 0; i < numel; ++i) {
        pixels[i] = static_cast<uint8_t>(std::clamp(data[i] * gain + bias, 0.0f, 255.0f));
    }

    // Wrap the uint8 buffer as a CHW UInt8 tensor (non-owning view) and encode.
    Tensor u8 = Tensor::from_blob(
        pixels.data(),
        {channels, height, width},
        DType::UInt8,
        Device::cpu(),
        /*deleter=*/[](void*) noexcept {});
    std::vector<uint8_t> img_data = tenzor::io::encode_png(u8);

    // Write image data length and data
    size_t img_len = img_data.size();
    while (img_len >= 0x80) {
        image.push_back(static_cast<uint8_t>((img_len & 0x7F) | 0x80));
        img_len >>= 7;
    }
    image.push_back(static_cast<uint8_t>(img_len));
    image.insert(image.end(), img_data.begin(), img_data.end());

    // Write image length
    size_t image_len = image.size();
    while (image_len >= 0x80) {
        buffer.push_back(static_cast<uint8_t>((image_len & 0x7F) | 0x80));
        image_len >>= 7;
    }
    buffer.push_back(static_cast<uint8_t>(image_len));
    buffer.insert(buffer.end(), image.begin(), image.end());

    return buffer;
}

auto SummaryWriter::get_current_time() -> double {
    return impl_->get_current_wall_time();
}

auto SummaryWriter::get_hostname() -> std::string {
    char hostname[HOST_NAME_MAX + 1];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[HOST_NAME_MAX] = '\0';
        return std::string(hostname);
    }
    return "unknown";
}

auto SummaryWriter::create_event_filename() -> std::string {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm;
    localtime_r(&now_time_t, &tm);

    // Format: events.out.tfevents.{timestamp}.{hostname}
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer),
                 "events.out.tfevents.%04d%02d%02d-%02d%02d%02d.%s",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                 tm.tm_hour, tm.tm_min, tm.tm_sec,
                 get_hostname().c_str());

    return std::string(buffer);
}

auto SummaryWriter::ensure_directory_exists(std::string_view path) -> void {
    try {
        std::filesystem::create_directories(path);
    } catch (const std::filesystem::filesystem_error& e) {
        throw TensorBoardException(
            std::format("Failed to create directory {}: {}", path, e.what()).c_str());
    }
}

} // namespace tenzor
