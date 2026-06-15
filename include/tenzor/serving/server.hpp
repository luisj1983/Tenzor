/**
 * @file server.hpp
 * @brief Inference serving infrastructure
 *
 * Provides a high-performance inference server with:
 * - REST API for predictions (POST /v1/models/{name}/predict)
 * - Dynamic batching to maximize GPU throughput
 * - Multi-model serving with independent worker pools
 * - Health checks, metrics, model lifecycle management
 *
 * Build with -DTENZOR_BUILD_SERVING=ON. Optionally with gRPC
 * for high-performance binary protocol.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <array>
#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include "../jit/compiler.hpp"
#include "traffic_router.hpp"

namespace tenzor {
namespace serving {

// ============================================================================
// Path sandboxing
// ============================================================================

/**
 * @brief Resolve a client-supplied model path and confine it to a repository
 *        root, defeating absolute-path and `..` traversal (including symlink
 *        escapes resolved by weakly_canonical).
 *
 * Single source of truth shared by the HTTP and gRPC LoadModel handlers so the
 * two transports cannot apply inconsistent policy. Rejects absolute paths and
 * any path whose canonicalised form escapes `root_dir`. A relative path
 * component that merely *contains* the substring ".." (e.g. "model..v2.tz") is
 * accepted — only a real ".." path component is treated as traversal.
 *
 * @param requested  Client-supplied, untrusted relative model path.
 * @param root_dir   Repository root to confine to (empty → current directory).
 * @return The resolved, contained absolute path (as a string).
 * @throws std::invalid_argument if the path is absolute or escapes the root.
 */
auto sanitize_repository_path(const std::string& requested,
                              const std::string& root_dir) -> std::string;

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Server configuration.
 */
struct ServerConfig {
    int32_t http_port{8080};              ///< REST API port
    int32_t grpc_port{8081};              ///< gRPC port (if built with gRPC)
    int32_t num_workers{4};               ///< Worker threads per model
    std::string model_repository_path;    ///< Path to model directory
    bool enable_metrics{true};            ///< Enable /metrics endpoint
    bool enable_health_check{true};       ///< Enable /health endpoint
    Device default_device{Device::cpu()}; ///< Default device for model loading

    // Authentication
    bool enable_auth{false};              ///< Enable API key authentication
    std::vector<std::string> api_keys;    ///< Valid API keys (Bearer tokens)
    std::string auth_header{"Authorization"}; ///< Header name for auth token

    // Rate limiting
    bool enable_rate_limit{false};        ///< Enable per-client rate limiting
    double rate_limit_rps{100.0};         ///< Max requests per second per client
    int32_t rate_limit_burst{200};        ///< Burst capacity
};

// ============================================================================
// Inference request/response
// ============================================================================

/**
 * @brief A single inference request.
 */
struct InferRequest {
    Tensor input;
    std::promise<Tensor> result;
    std::chrono::steady_clock::time_point arrival;

    InferRequest(Tensor in)
        : input(std::move(in)), arrival(std::chrono::steady_clock::now()) {}
};

// ============================================================================
// Dynamic batcher
// ============================================================================

/**
 * @brief Configuration for dynamic batching.
 */
struct BatchConfig {
    int32_t max_batch_size{32};           ///< Maximum batch size
    int32_t max_latency_us{10000};        ///< Maximum waiting time (microseconds)
};

/**
 * @brief Collects individual requests into batches for efficient execution.
 *
 * Waits until batch is full or deadline expires, then concatenates inputs
 * along dim=0 and executes a single forward pass.
 */
class DynamicBatcher {
public:
    DynamicBatcher(std::shared_ptr<jit::CompiledModule> model,
                   BatchConfig config = {});
    ~DynamicBatcher();

    /**
     * @brief Submit a request for batched inference.
     *
     * @param input Input tensor
     * @return Future that will hold the result
     */
    auto submit(Tensor input) -> std::future<Tensor>;

    /**
     * @brief Start the batching loop.
     */
    auto start() -> void;

    /**
     * @brief Stop the batching loop.
     */
    auto stop() -> void;

private:
    std::shared_ptr<jit::CompiledModule> model_;
    BatchConfig config_;
    std::atomic<bool> running_{false};
    std::thread batch_thread_;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::shared_ptr<InferRequest>> queue_;

    auto batch_loop() -> void;
    auto execute_batch(std::vector<std::shared_ptr<InferRequest>>& batch) -> void;
};

// ============================================================================
// Model repository
// ============================================================================

/**
 * @brief State of a loaded model.
 */
enum class ModelState : uint8_t {
    LOADING,
    READY,
    UNLOADING,
    FAILED,
};

/**
 * @brief Entry for a loaded model.
 */
struct ModelEntry {
    std::string name;
    int32_t version{1};
    std::shared_ptr<jit::CompiledModule> module;
    Device device;
    std::unique_ptr<DynamicBatcher> batcher;
    std::atomic<ModelState> state{ModelState::LOADING};
};

/**
 * @brief Thread-safe model registry with load/unload/versioning.
 */
class ModelRepository {
public:
    /**
     * @brief Load a model from file.
     *
     * Supports TZJT (native JIT), ONNX, and TNZR formats.
     * Runs optimize_for_inference() after loading.
     *
     * @param name Model name
     * @param path Path to model file
     * @param device Target device
     * @param batch_config Batching configuration
     */
    auto load_model(const std::string& name, const std::string& path,
                    Device device, BatchConfig batch_config = {}) -> void;

    /**
     * @brief Unload a model.
     */
    auto unload_model(const std::string& name) -> void;

    /**
     * @brief Get a loaded model by name.
     *
     * @return Model entry, or nullptr if not found
     */
    auto get_model(const std::string& name) -> std::shared_ptr<ModelEntry>;

    /**
     * @brief List all loaded models.
     */
    auto list_models() const -> std::vector<std::string>;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ModelEntry>> models_;
};

// ============================================================================
// Metrics
// ============================================================================

/**
 * @brief Lock-free metrics per model.
 */
struct ModelMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> total_latency_us{0};
    std::atomic<uint64_t> total_batch_count{0};
    std::atomic<uint64_t> total_batch_size{0};
    std::atomic<uint64_t> error_count{0};

    // Latency percentile tracking (ring buffer of recent latencies)
    static constexpr size_t kLatencyWindowSize = 1000;
    // Atomic elements: record_latency() writes from request-handler threads
    // while format_prometheus() reads concurrently during a /metrics scrape.
    // A plain array was a data race (UB); per-element atomics make both sides
    // well-defined (relaxed ordering is sufficient for a best-effort window).
    std::array<std::atomic<uint64_t>, kLatencyWindowSize> latency_window{};
    std::atomic<size_t> latency_idx{0};

    auto record_latency(uint64_t latency_us) -> void {
        auto idx = latency_idx.fetch_add(1, std::memory_order_relaxed) % kLatencyWindowSize;
        latency_window[idx].store(latency_us, std::memory_order_relaxed);
    }
};

/**
 * @brief Global metrics registry.
 */
class MetricsRegistry {
public:
    static auto instance() -> MetricsRegistry&;

    auto get_metrics(const std::string& model_name) -> ModelMetrics&;
    auto format_prometheus() const -> std::string;

private:
    MetricsRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ModelMetrics>> metrics_;
};

// ============================================================================
// Inference server
// ============================================================================

/**
 * @brief Main inference server.
 *
 * Owns the HTTP server, model repository, and metrics.
 * Start/stop lifecycle with blocking wait().
 */
class InferenceServer {
public:
    explicit InferenceServer(ServerConfig config);
    ~InferenceServer();

    /**
     * @brief Start the server.
     */
    auto start() -> void;

    /**
     * @brief Stop the server.
     */
    auto stop() -> void;

    /**
     * @brief Block until the server is stopped.
     */
    auto wait() -> void;

    /**
     * @brief Block until the server is stopped or the timeout expires.
     *
     * Audit-11 QQ.19: bounded wait so Python callers can release the GIL
     * for the timeout window and re-acquire to service signals
     * (KeyboardInterrupt) without leaking the server thread.
     *
     * @param timeout Maximum time to block before returning.
     * @return true if the server stopped within the timeout, false on timeout.
     */
    auto wait_for(std::chrono::milliseconds timeout) -> bool;

    /**
     * @brief Whether this build can actually serve HTTP requests.
     *
     * False when compiled without httplib (TENZOR_BUILD_SERVING=OFF): the
     * lifecycle API (start/stop/wait) still works, but serve_loop() only
     * idles. Standalone entry points (tenzor_serve) should check this and
     * exit non-zero instead of wait()-ing on a loop that never serves and
     * never terminates.
     */
    static auto has_http_transport() -> bool;

    /**
     * @brief Get the model repository.
     */
    auto repository() -> ModelRepository& { return repository_; }

    /// Get the traffic router for A/B experiments
    auto traffic_router() -> TrafficRouter& { return traffic_router_; }

private:
    ServerConfig config_;
    ModelRepository repository_;
    TrafficRouter traffic_router_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;

    auto serve_loop() -> void;
};

} // namespace serving
} // namespace tenzor
