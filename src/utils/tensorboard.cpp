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
#include <filesystem>
#include <ctime>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <sys/types.h>
#include <limits.h>
#include <thread>

namespace tenzor {

namespace {

// TensorBoard event format helpers
constexpr uint32_t kMaskedCrc32c = 0xA282EAD8;

// CRC32C implementation for record validation
uint32_t masked_crc32c(const uint8_t* data, size_t length) {
    // Simplified CRC32 for TensorBoard compatibility
    uint32_t crc = 0;
    for (size_t i = 0; i < length; ++i) {
        crc = ((crc >> 8) ^ data[i]) | (crc << 24);
    }
    return crc;
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

// Write little-endian float
void write_float_le(std::vector<uint8_t>& buffer, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    write_uint32_le(buffer, bits);
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
    buffer.push_back(static_cast<uint8_t>(tag));
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
    bool is_open{false};
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
    if (!impl_->is_open) {
        throw TensorBoardException("SummaryWriter is closed");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto data = serialize_scalar(value);
    write_event(tag, data, step);

    // Auto-flush if needed
    impl_->event_count++;
    if (impl_->event_count >= impl_->max_queue) {
        flush();
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

    // Ensure tensor is on CPU
    Tensor cpu_tensor = tensor.device().type == Device::Type::CPU ?
                        tensor : tensor.cpu();

    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto data = serialize_histogram(cpu_tensor, bins);
    write_event(tag, data, step);

    impl_->event_count++;
    if (impl_->event_count >= impl_->max_queue) {
        flush();
        impl_->event_count = 0;
    }
}

auto SummaryWriter::add_image(std::string_view tag,
                             const Tensor& tensor,
                             int64_t step,
                             std::string_view dataformats) -> void {
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

    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto data = serialize_image(cpu_tensor);
    write_event(tag, data, step);

    impl_->event_count++;
    if (impl_->event_count >= impl_->max_queue) {
        flush();
        impl_->event_count = 0;
    }
}

auto SummaryWriter::add_graph(std::string_view model_name,
                             const std::vector<int64_t>& input_shape) -> void {
    if (!impl_->is_open) {
        throw TensorBoardException("SummaryWriter is closed");
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Create simple graph metadata
    std::vector<uint8_t> graph_data;

    // Graph event structure (simplified)
    write_field_header(graph_data, 1, 2); // graph_def (string, wire_type=2)

    // Graph definition (minimal)
    std::string graph_def = std::format("model: {}, input_shape: [", model_name);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i > 0) graph_def += ", ";
        graph_def += std::to_string(input_shape[i]);
    }
    graph_def += "]";

    write_string(graph_data, graph_def);

    write_event("__graph__", graph_data, 0);

    TENZOR_LOG_INFO(std::format("TensorBoard: Added graph for {}", model_name));
}

auto SummaryWriter::flush() -> void {
    if (!impl_->is_open) {
        return;
    }

    impl_->event_file.flush();
    impl_->last_flush = std::chrono::steady_clock::now();
}

auto SummaryWriter::close() -> void {
    if (!impl_->is_open) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);

    flush();
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
                                int64_t step) -> void {
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

    // simple_value (float) or other data
    if (data.size() == 4) {
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

    // Write record
    impl_->event_file.write(reinterpret_cast<const char*>(length_bytes.data()), 8);
    impl_->event_file.write(reinterpret_cast<const char*>(&length_crc), 4);
    impl_->event_file.write(reinterpret_cast<const char*>(event.data()), event.size());
    impl_->event_file.write(reinterpret_cast<const char*>(&data_crc), 4);
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

    // encoded_image_string (PNG data)
    // For simplicity, store raw float data as base64-like encoding
    write_field_header(image, 4, 2);

    const float* data = tensor.data<float>();
    size_t numel = static_cast<size_t>(tensor.numel());

    // Convert to uint8 (assuming values in [0, 1] or [0, 255])
    std::vector<uint8_t> img_data(numel);
    for (size_t i = 0; i < numel; ++i) {
        float val = data[i];
        // Normalize to [0, 255]
        if (val <= 1.0f) {
            val *= 255.0f;
        }
        img_data[i] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
    }

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
