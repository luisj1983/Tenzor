#pragma once

#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include <cstddef>

namespace tenzor {

// Forward declarations
namespace nn {
class Module;
}

namespace models {

/**
 * @brief Progress callback function type
 * @param downloaded Bytes downloaded so far
 * @param total Total bytes to download (0 if unknown)
 * @param speed Current download speed in bytes/sec
 * @param eta Estimated time remaining in seconds
 */
using ProgressCallback = std::function<void(size_t downloaded, size_t total, double speed, double eta)>;

/**
 * @brief Configuration options for ModelHub
 */
struct HubConfig {
    std::string cache_dir;          // Cache directory path
    size_t max_cache_size;          // Maximum cache size in bytes (0 = unlimited)
    bool verify_checksums;          // Whether to verify SHA256 checksums
    bool resume_downloads;          // Whether to resume interrupted downloads
    int connection_timeout;         // Connection timeout in seconds
    int max_retries;                // Maximum number of download retries
    bool show_progress;             // Whether to show progress by default

    HubConfig();
};

/**
 * @brief Model weight information
 */
struct ModelWeightInfo {
    std::string name;               // Model name
    std::string url;                // Download URL (legacy .pth or .safetensors)
    std::string sha256;             // Expected SHA256 checksum
    size_t size;                    // File size in bytes (0 if unknown)
    std::string description;        // Model description
    // H3-followup: optional SafeTensors mirror URL. When non-empty,
    // `download_pretrained(name, prefer_safetensors=true)` resolves to this
    // URL instead of `url`. SafeTensors mirrors are typically hosted on
    // HuggingFace (`https://huggingface.co/{org}/{model}/resolve/main/model.safetensors`)
    // and parse cleanly through H2's format-aware loader. The legacy `url`
    // (PyTorch .pth) currently throws an actionable error because the
    // pickle parser is H2-followup.
    std::string safetensors_url;    // Optional SafeTensors mirror (HuggingFace)
};

/**
 * @brief Download statistics
 */
struct DownloadStats {
    size_t total_bytes;             // Total bytes downloaded
    size_t bytes_downloaded;        // Bytes downloaded in this session
    double download_time;           // Time taken in seconds
    double average_speed;           // Average speed in bytes/sec
    bool resumed;                   // Whether download was resumed
    bool verified;                  // Whether checksum was verified
};

/**
 * @brief ModelHub - Pretrained weight management system
 *
 * Features:
 * - Download weights from URLs with resume support
 * - Automatic caching and checksum verification
 * - Progress tracking with speed and ETA
 * - Thread-safe operations
 * - Cache management with size limits
 */
class ModelHub {
public:
    /**
     * @brief Download pretrained weights
     * @param model_name Unique model identifier
     * @param url Download URL (HTTP/HTTPS)
     * @param expected_sha256 Expected SHA256 checksum (empty to skip verification)
     * @param show_progress Whether to show download progress
     * @param progress_callback Custom progress callback
     * @return Path to downloaded/cached weights file
     * @throws std::runtime_error on download or verification failure
     */
    static std::string download_weights(
        const std::string& model_name,
        const std::string& url,
        const std::string& expected_sha256 = "",
        bool show_progress = true,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Download weights using registered model info
     * @param model_name Registered model name
     * @param show_progress Whether to show download progress
     * @param progress_callback Custom progress callback
     * @return Path to downloaded/cached weights file
     * @throws std::runtime_error if model not registered or download fails
     */
    static std::string download_pretrained(
        const std::string& model_name,
        bool show_progress = true,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Download pretrained weights, preferring a SafeTensors mirror
     *        when the model's registry entry provides one.
     *
     * H3-followup: SafeTensors-mirror-aware variant. When the model's
     * `safetensors_url` field is non-empty, this routes the download to
     * the safetensors URL instead of the legacy `.pth` URL. The downloaded
     * file is then parsed by H2's format dispatcher, which handles
     * `.safetensors` natively. Falls back to the legacy URL if the model
     * has no safetensors mirror registered.
     *
     * @param model_name Unique model identifier
     * @param prefer_safetensors If true and a safetensors mirror is registered,
     *                            download from the mirror; else use legacy URL.
     * @param show_progress Whether to show download progress
     * @param progress_callback Custom progress callback
     * @return Path to downloaded/cached weights file
     */
    static std::string download_pretrained_safetensors(
        const std::string& model_name,
        bool prefer_safetensors = true,
        bool show_progress = true,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Load pretrained weights into model
     * @param model Target model to load weights into
     * @param weights_path Path to weights file
     * @param strict If true, raise error on architecture mismatch
     * @throws std::runtime_error on loading failure (if strict=true)
     */
    static void load_pretrained_weights(
        nn::Module& model,
        const std::string& weights_path,
        bool strict = true
    );

    /**
     * @brief Configure cache directory
     * @param path Cache directory path
     */
    static void set_cache_dir(const std::string& path);

    /**
     * @brief Get current cache directory
     * @return Cache directory path
     */
    static std::string get_cache_dir();

    /**
     * @brief Set ModelHub configuration
     * @param config Configuration options
     */
    static void set_config(const HubConfig& config);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    static HubConfig get_config();

    /**
     * @brief Clear all cached weights
     */
    static void clear_cache();

    /**
     * @brief Get total cache size
     * @return Cache size in bytes
     */
    static size_t cache_size();

    /**
     * @brief List cached models
     * @return Vector of cached model names
     */
    static std::vector<std::string> list_cached_models();

    /**
     * @brief Check if model weights are cached
     * @param model_name Model name
     * @return true if cached, false otherwise
     */
    static bool is_cached(const std::string& model_name);

    /**
     * @brief Get cached weights path
     * @param model_name Model name
     * @return Path to cached weights (empty if not cached)
     */
    static std::string get_cached_path(const std::string& model_name);

    /**
     * @brief Register a model in the hub
     * @param info Model weight information
     */
    static void register_model(const ModelWeightInfo& info);

    /**
     * @brief Register multiple models
     * @param models Vector of model weight information
     */
    static void register_models(const std::vector<ModelWeightInfo>& models);

    /**
     * @brief Get registered model info
     * @param model_name Model name
     * @return Model weight information
     * @throws std::runtime_error if model not registered
     */
    static ModelWeightInfo get_model_info(const std::string& model_name);

    /**
     * @brief List all registered models
     * @return Vector of registered model names
     */
    static std::vector<std::string> list_registered_models();

    /**
     * @brief Check if model is registered
     * @param model_name Model name
     * @return true if registered, false otherwise
     */
    static bool is_registered(const std::string& model_name);

    /**
     * @brief Remove model from cache
     * @param model_name Model name
     * @return true if removed, false if not cached
     */
    static bool remove_from_cache(const std::string& model_name);

    /**
     * @brief Get download statistics for last download
     * @return Download statistics
     */
    static DownloadStats get_last_download_stats();

    /**
     * @brief Verify file checksum
     * @param file_path Path to file
     * @param expected_sha256 Expected SHA256 checksum
     * @return true if checksum matches, false otherwise
     */
    static bool verify_checksum(const std::string& file_path, const std::string& expected_sha256);

    /**
     * @brief Compute SHA256 checksum of file
     * @param file_path Path to file
     * @return SHA256 checksum as hex string
     */
    static std::string compute_checksum(const std::string& file_path);

    /**
     * @brief Clean cache to fit within size limit
     * @param max_size Maximum cache size in bytes
     * @return Number of files removed
     */
    static size_t clean_cache(size_t max_size);

private:
    // Internal implementation
    class Impl;
    static std::unique_ptr<Impl> impl_;
    static std::mutex mutex_;

    // Initialize implementation
    static void ensure_initialized();

    // Default progress callback
    static void default_progress_callback(size_t downloaded, size_t total, double speed, double eta);
};

/**
 * @brief Default model registry
 *
 * Pre-registered popular models with download URLs
 */
namespace registry {

// Internal registry initialization helpers (get_pytorch_model_url,
// initialize_default_registry) live in hub.cpp — they are implementation
// details and intentionally not part of the public API.

/**
 * @brief Look up the reason a previously-registered model was removed.
 *
 * Audit C.7: certain pretrained entries (vgg*, alexnet, googlenet,
 * inception_v3, fcn/deeplab/faster_rcnn/mask_rcnn/retinanet/yolo*) were
 * dropped from the default registry because they had no verifiable
 * safetensors mirror — every download would either hit a dead URL or feed
 * a .pth file into a parser that throws.  `ModelHub::download_pretrained*`
 * consults this function before raising the generic "not registered"
 * error so that users get an actionable diagnostic.
 *
 * @param model_name name passed to `download_pretrained()`.
 * @return Non-empty reason string if the name is on the removal list;
 *         empty string otherwise.
 */
std::string removed_pretrained_reason(const std::string& model_name);

} // namespace registry

} // namespace models
} // namespace tenzor
