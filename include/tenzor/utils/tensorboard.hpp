/**
 * @file tensorboard.hpp
 * @brief TensorBoard integration for real-time training visualization
 *
 * Provides SummaryWriter class for logging scalars, histograms, images,
 * and computation graphs to TensorBoard-compatible event files.
 *
 * Supports the TensorBoard event format with protobuf serialization
 * for visualization in TensorBoard.
 *
 * @par Example Usage
 * @code
 * // Create writer
 * SummaryWriter writer("runs/experiment1");
 *
 * // Log scalars
 * for (int step = 0; step < 1000; ++step) {
 *     float loss = train_step();
 *     writer.add_scalar("loss", loss, step);
 * }
 *
 * // Log histograms
 * Tensor weights = model.get_weights();
 * writer.add_histogram("weights/layer1", weights, step);
 *
 * // Log images
 * Tensor img({3, 64, 64}, DType::Float32, Device::cpu());
 * writer.add_image("generated/sample", img, step);
 *
 * writer.close();
 * @endcode
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <mutex>
#include <fstream>
#include <chrono>
#include "../core/tensor.hpp"
#include "error.hpp"

namespace tenzor {

/**
 * @brief Writer for TensorBoard event files
 *
 * Creates TensorBoard-compatible event files for real-time visualization
 * of training metrics, model parameters, and images.
 *
 * **Features:**
 * - Scalar logging (loss, accuracy, learning rate)
 * - Histogram logging (weights, gradients, activations)
 * - Image logging (generated images, input samples)
 * - Graph visualization (computation graph structure)
 * - Thread-safe concurrent writes
 * - Automatic file flushing
 *
 * **File Format:**
 * - Binary event files with protobuf serialization
 * - Stored in: `log_dir/events.out.tfevents.{timestamp}.{hostname}`
 * - Compatible with TensorBoard 1.x and 2.x
 *
 * @par Thread Safety
 * All methods are thread-safe and can be called from multiple threads.
 *
 * @par Performance
 * - Writes are buffered and flushed periodically
 * - Configurable queue size and flush interval
 * - Minimal overhead on training loop
 *
 * @code
 * SummaryWriter writer("runs/exp1", 10, 120);
 *
 * // Training loop
 * for (int epoch = 0; epoch < 100; ++epoch) {
 *     float loss = train_epoch();
 *     writer.add_scalar("train/loss", loss, epoch);
 *
 *     if (epoch % 10 == 0) {
 *         Tensor weights = model.get_weights();
 *         writer.add_histogram("weights", weights, epoch);
 *     }
 * }
 *
 * writer.flush();  // Ensure all data is written
 * writer.close();
 * @endcode
 */
class SummaryWriter {
public:
    /**
     * @brief Construct SummaryWriter with specified log directory
     *
     * Creates the log directory if it doesn't exist and opens a new
     * TensorBoard event file with timestamped filename.
     *
     * @param log_dir Directory to store event files
     * @param max_queue Maximum number of events to buffer before flush (default: 10)
     * @param flush_secs Seconds between automatic flushes (default: 120)
     *
     * @throws std::runtime_error if directory creation fails or file cannot be opened
     *
     * @code
     * // Basic usage
     * SummaryWriter writer("runs/experiment1");
     *
     * // Custom buffering
     * SummaryWriter writer("logs", 50, 60);  // Larger buffer, flush every minute
     * @endcode
     */
    explicit SummaryWriter(std::string_view log_dir,
                          int max_queue = 10,
                          int flush_secs = 120);

    /**
     * @brief Destructor - ensures all data is flushed
     *
     * Automatically flushes any pending events and closes the file.
     */
    ~SummaryWriter();

    // Prevent copying
    SummaryWriter(const SummaryWriter&) = delete;
    SummaryWriter& operator=(const SummaryWriter&) = delete;

    // Allow moving
    SummaryWriter(SummaryWriter&&) noexcept = default;
    SummaryWriter& operator=(SummaryWriter&&) noexcept = default;

    /**
     * @brief Log scalar value
     *
     * Records a single scalar value (e.g., loss, accuracy, learning rate)
     * at the specified training step.
     *
     * @param tag Unique identifier for this metric (e.g., "train/loss", "accuracy")
     * @param value Scalar value to log
     * @param step Training step or iteration number
     *
     * @code
     * writer.add_scalar("loss", 0.523, 100);
     * writer.add_scalar("accuracy", 0.95, 100);
     * writer.add_scalar("lr", 0.001, 100);
     * @endcode
     */
    auto add_scalar(std::string_view tag, float value, int64_t step) -> void;

    /**
     * @brief Log histogram of tensor values
     *
     * Records the distribution of values in a tensor (e.g., weight distributions,
     * gradient magnitudes, activation patterns).
     *
     * Creates histogram with configurable number of bins and min/max range.
     *
     * @param tag Unique identifier (e.g., "weights/layer1", "gradients/conv1")
     * @param tensor Tensor to create histogram from
     * @param step Training step
     * @param bins Number of histogram bins (default: 30)
     *
     * @throws std::runtime_error if tensor is not on CPU or dtype unsupported
     *
     * @code
     * Tensor weights = model.get_parameter("layer1.weight");
     * writer.add_histogram("weights/layer1", weights, 100);
     *
     * Tensor gradients = weights.grad();
     * writer.add_histogram("gradients/layer1", gradients, 100);
     * @endcode
     */
    auto add_histogram(std::string_view tag,
                      const Tensor& tensor,
                      int64_t step,
                      int bins = 30) -> void;

    /**
     * @brief Log image tensor
     *
     * Records an image for visualization. Tensor should be in format:
     * - Shape: [C, H, W] (channels, height, width)
     * - C = 1 (grayscale), 3 (RGB), or 4 (RGBA)
     * - Values in range [0, 1] or [0, 255]
     *
     * @param tag Unique identifier (e.g., "generated/sample", "input/batch0")
     * @param tensor Image tensor [C, H, W]
     * @param step Training step
     * @param dataformats Format string (default: "CHW")
     *
     * @throws std::runtime_error if tensor shape invalid or not on CPU
     *
     * @code
     * // Log generated image
     * Tensor img({3, 64, 64}, DType::Float32, Device::cpu());
     * // ... fill img with data ...
     * writer.add_image("generated/sample", img, 100);
     *
     * // Log grayscale
     * Tensor gray({1, 28, 28}, DType::Float32, Device::cpu());
     * writer.add_image("input/mnist", gray, 100);
     * @endcode
     */
    auto add_image(std::string_view tag,
                  const Tensor& tensor,
                  int64_t step,
                  std::string_view dataformats = "CHW") -> void;

    /**
     * @brief Log computation graph
     *
     * Records the structure of a neural network model for visualization
     * in TensorBoard's graph view.
     *
     * @param model_name Name of the model
     * @param input_shape Shape of input tensor (e.g., {1, 3, 224, 224})
     *
     * @note This is a simplified implementation. Full graph tracing
     *       requires integration with autograd system.
     *
     * @code
     * writer.add_graph("ResNet50", {1, 3, 224, 224});
     * @endcode
     */
    auto add_graph(std::string_view model_name,
                  const std::vector<int64_t>& input_shape) -> void;

    /**
     * @brief Flush all pending events to disk
     *
     * Forces immediate write of all buffered events. Normally called
     * automatically based on flush_secs, but can be called manually
     * for critical checkpoints.
     *
     * @code
     * writer.add_scalar("loss", loss, step);
     * writer.flush();  // Ensure data is written immediately
     * @endcode
     */
    auto flush() -> void;

    /**
     * @brief Close the writer and release resources
     *
     * Flushes all pending events and closes the file handle.
     * Writer cannot be used after closing.
     *
     * @note Automatically called by destructor
     *
     * @code
     * writer.close();
     * // writer is now closed and cannot be used
     * @endcode
     */
    auto close() -> void;

    /**
     * @brief Check if writer is currently open
     *
     * @return true if writer is open and ready to accept events
     */
    auto is_open() const -> bool;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Helper methods
    auto write_event(std::string_view tag, const std::vector<uint8_t>& data, int64_t step) -> void;
    auto serialize_scalar(float value) -> std::vector<uint8_t>;
    auto serialize_histogram(const Tensor& tensor, int bins) -> std::vector<uint8_t>;
    auto serialize_image(const Tensor& tensor) -> std::vector<uint8_t>;
    auto get_current_time() -> double;
    auto get_hostname() -> std::string;
    auto create_event_filename() -> std::string;
    auto ensure_directory_exists(std::string_view path) -> void;
};

/**
 * @brief Exception for TensorBoard-related errors
 */
class TensorBoardException : public TenzorException {
public:
    using TenzorException::TenzorException;
};

} // namespace tenzor
