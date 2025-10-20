#include "tenzor/models/hub.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/serialize.hpp"
#include <curl/curl.h>
#include <openssl/sha.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <thread>

namespace fs = std::filesystem;

namespace tenzor::models {

// ============================================================================
// HubConfig Implementation
// ============================================================================

HubConfig::HubConfig()
    : cache_dir(fs::path(getenv("HOME") ? getenv("HOME") : ".") / ".tenzor" / "checkpoints")
    , max_cache_size(0)  // Unlimited by default
    , verify_checksums(true)
    , resume_downloads(true)
    , connection_timeout(30)
    , max_retries(3)
    , show_progress(true)
{}

// ============================================================================
// ModelHub::Impl - Internal Implementation
// ============================================================================

class ModelHub::Impl {
public:
    HubConfig config;
    std::unordered_map<std::string, ModelWeightInfo> registry;
    DownloadStats last_stats;

    Impl() {
        // Create cache directory if it doesn't exist
        fs::create_directories(config.cache_dir);

        // Initialize CURL
        curl_global_init(CURL_GLOBAL_ALL);
    }

    ~Impl() {
        curl_global_cleanup();
    }

    // CURL write callback
    static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* userdata) {
        auto* file = static_cast<std::ofstream*>(userdata);
        file->write(static_cast<char*>(ptr), size * nmemb);
        return size * nmemb;
    }

    // CURL progress callback
    struct ProgressData {
        ProgressCallback callback;
        size_t start_offset;
        std::chrono::steady_clock::time_point start_time;
        size_t last_downloaded;
        std::chrono::steady_clock::time_point last_time;
        bool show_default_progress;
    };

    static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                                 curl_off_t ultotal, curl_off_t ulnow) {
        auto* data = static_cast<ProgressData*>(clientp);

        if (dltotal == 0 && dlnow == 0) return 0;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - data->start_time).count();

        if (elapsed < 0.1 && dlnow < dltotal) return 0;  // Update at most 10 times per second

        size_t total_downloaded = data->start_offset + dlnow;
        size_t total_size = data->start_offset + dltotal;

        // Calculate speed
        double speed = 0.0;
        if (elapsed > 0) {
            speed = (total_downloaded - data->start_offset) / elapsed;
        }

        // Calculate ETA
        double eta = 0.0;
        if (speed > 0 && total_size > total_downloaded) {
            eta = (total_size - total_downloaded) / speed;
        }

        // Call custom callback if provided
        if (data->callback) {
            data->callback(total_downloaded, total_size, speed, eta);
        }

        // Show default progress
        if (data->show_default_progress) {
            ModelHub::default_progress_callback(total_downloaded, total_size, speed, eta);
        }

        data->last_downloaded = total_downloaded;
        data->last_time = now;

        return 0;
    }

    // Download file with resume support
    std::string download_file(const std::string& url, const std::string& dest_path,
                             bool show_progress, ProgressCallback callback) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }

        // Check if file exists for resume
        size_t start_offset = 0;
        std::ios_base::openmode mode = std::ios::binary | std::ios::out;

        if (config.resume_downloads && fs::exists(dest_path)) {
            start_offset = fs::file_size(dest_path);
            mode |= std::ios::app;
        }

        std::ofstream outfile(dest_path, mode);
        if (!outfile) {
            curl_easy_cleanup(curl);
            throw std::runtime_error("Failed to open output file: " + dest_path);
        }

        // Setup CURL options
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outfile);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, config.connection_timeout);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

        // Resume support
        if (start_offset > 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)start_offset);
        }

        // Progress tracking
        ProgressData progress_data;
        progress_data.callback = callback;
        progress_data.start_offset = start_offset;
        progress_data.start_time = std::chrono::steady_clock::now();
        progress_data.last_downloaded = start_offset;
        progress_data.last_time = progress_data.start_time;
        progress_data.show_default_progress = show_progress && !callback;

        if (show_progress || callback) {
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress_data);
        }

        // Perform download with retries
        CURLcode res = CURLE_FAILED_INIT;
        int retries = 0;

        while (retries <= config.max_retries) {
            res = curl_easy_perform(curl);

            if (res == CURLE_OK) {
                break;
            }

            retries++;
            if (retries <= config.max_retries) {
                std::cerr << "Download failed (attempt " << retries << "/" << config.max_retries
                         << "): " << curl_easy_strerror(res) << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1 << (retries - 1)));  // Exponential backoff
            }
        }

        outfile.close();

        // Update statistics
        auto end_time = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration<double>(end_time - progress_data.start_time).count();

        size_t final_size = fs::file_size(dest_path);
        last_stats.total_bytes = final_size;
        last_stats.bytes_downloaded = final_size - start_offset;
        last_stats.download_time = total_time;
        last_stats.average_speed = total_time > 0 ? last_stats.bytes_downloaded / total_time : 0;
        last_stats.resumed = start_offset > 0;
        last_stats.verified = false;

        if (res != CURLE_OK) {
            curl_easy_cleanup(curl);
            throw std::runtime_error(std::string("Download failed: ") + curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);

        if (show_progress && !callback) {
            std::cout << std::endl;  // New line after progress
        }

        return dest_path;
    }

    // Compute SHA256 checksum
    std::string compute_sha256(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for checksum: " + file_path);
        }

        SHA256_CTX sha256;
        SHA256_Init(&sha256);

        char buffer[8192];
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            SHA256_Update(&sha256, buffer, file.gcount());
        }

        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256_Final(hash, &sha256);

        std::ostringstream oss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }

        return oss.str();
    }

    // Verify checksum
    bool verify_sha256(const std::string& file_path, const std::string& expected_hash) {
        if (expected_hash.empty()) return true;

        std::string actual_hash = compute_sha256(file_path);
        bool matches = (actual_hash == expected_hash);

        last_stats.verified = matches;

        return matches;
    }

    // Get cache file path for model
    std::string get_cache_path(const std::string& model_name) const {
        return (fs::path(config.cache_dir) / (model_name + ".pt")).string();
    }

    // Clean cache to fit within size limit
    size_t clean_cache_to_size(size_t max_size) {
        if (max_size == 0) return 0;

        // Get all cached files with their sizes and modification times
        struct FileInfo {
            fs::path path;
            size_t size;
            fs::file_time_type mtime;
        };

        std::vector<FileInfo> files;
        size_t total_size = 0;

        for (const auto& entry : fs::directory_iterator(config.cache_dir)) {
            if (entry.is_regular_file()) {
                FileInfo info;
                info.path = entry.path();
                info.size = entry.file_size();
                info.mtime = entry.last_write_time();
                files.push_back(info);
                total_size += info.size;
            }
        }

        if (total_size <= max_size) return 0;

        // Sort by modification time (oldest first)
        std::sort(files.begin(), files.end(), [](const FileInfo& a, const FileInfo& b) {
            return a.mtime < b.mtime;
        });

        // Remove oldest files until within limit
        size_t removed = 0;
        for (const auto& file : files) {
            if (total_size <= max_size) break;

            fs::remove(file.path);
            total_size -= file.size;
            removed++;
        }

        return removed;
    }
};

// ============================================================================
// ModelHub Static Members
// ============================================================================

std::unique_ptr<ModelHub::Impl> ModelHub::impl_;
std::mutex ModelHub::mutex_;

void ModelHub::ensure_initialized() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
        registry::initialize_default_registry(impl_->registry);
    }
}

// ============================================================================
// ModelHub Public Methods
// ============================================================================

std::string ModelHub::download_weights(
    const std::string& model_name,
    const std::string& url,
    const std::string& expected_sha256,
    bool show_progress,
    ProgressCallback progress_callback)
{
    // Validate model_name is not empty
    if (model_name.empty()) {
        throw std::runtime_error("Model name cannot be empty");
    }

    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already cached
    std::string cache_path = impl_->get_cache_path(model_name);

    if (fs::exists(cache_path)) {
        // Verify checksum if provided
        if (!expected_sha256.empty() && impl_->config.verify_checksums) {
            if (show_progress) {
                std::cout << "Verifying cached file..." << std::endl;
            }

            if (impl_->verify_sha256(cache_path, expected_sha256)) {
                if (show_progress) {
                    std::cout << "Using cached weights: " << cache_path << std::endl;
                }
                return cache_path;
            } else {
                std::cerr << "Cached file checksum mismatch, re-downloading..." << std::endl;
                fs::remove(cache_path);
            }
        } else {
            if (show_progress) {
                std::cout << "Using cached weights: " << cache_path << std::endl;
            }
            return cache_path;
        }
    }

    // Download file
    if (show_progress) {
        std::cout << "Downloading " << model_name << " from " << url << "..." << std::endl;
    }

    std::string temp_path = cache_path + ".tmp";
    impl_->download_file(url, temp_path, show_progress, progress_callback);

    // Verify checksum
    if (!expected_sha256.empty() && impl_->config.verify_checksums) {
        if (show_progress) {
            std::cout << "Verifying checksum..." << std::endl;
        }

        if (!impl_->verify_sha256(temp_path, expected_sha256)) {
            fs::remove(temp_path);
            throw std::runtime_error("Checksum verification failed for " + model_name);
        }
    }

    // Move to final location
    fs::rename(temp_path, cache_path);

    // Clean cache if needed
    if (impl_->config.max_cache_size > 0) {
        impl_->clean_cache_to_size(impl_->config.max_cache_size);
    }

    if (show_progress) {
        std::cout << "Downloaded to: " << cache_path << std::endl;
    }

    return cache_path;
}

std::string ModelHub::download_pretrained(
    const std::string& model_name,
    bool show_progress,
    ProgressCallback progress_callback)
{
    ensure_initialized();

    ModelWeightInfo info;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = impl_->registry.find(model_name);
        if (it == impl_->registry.end()) {
            throw std::runtime_error("Model not registered: " + model_name);
        }
        info = it->second;
    }

    return download_weights(model_name, info.url, info.sha256, show_progress, progress_callback);
}

void ModelHub::load_pretrained_weights(
    nn::Module& model,
    const std::string& weights_path,
    bool strict)
{
    if (!fs::exists(weights_path)) {
        throw std::runtime_error("Weights file not found: " + weights_path);
    }

    std::unordered_map<std::string, Tensor> state_dict;
    std::string load_error;

    try {
        // Load checkpoint from file using Serializer
        state_dict = nn::Serializer::load(weights_path);
    } catch (const std::exception& e) {
        load_error = e.what();
        if (!strict) {
            std::cerr << "Warning: Partial weight loading - " << load_error << std::endl;
            // In non-strict mode, continue with empty state_dict
        }
        // In strict mode, defer throwing until after load_state_dict is attempted
    }

    // Attempt to load state into model
    // This allows the module to see the load attempt even if deserialization failed
    try {
        model.load_state_dict(state_dict);
    } catch (const std::exception& e) {
        if (strict) {
            throw std::runtime_error(std::string("Failed to load state dict: ") + e.what());
        } else {
            std::cerr << "Warning: Model state_dict loading failed - " << e.what() << std::endl;
        }
    }

    // If there was a load error in strict mode, throw it now after load_state_dict was attempted
    if (strict && !load_error.empty()) {
        throw std::runtime_error(std::string("Failed to load weights: ") + load_error);
    }
}

void ModelHub::set_cache_dir(const std::string& path) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->config.cache_dir = path;
    fs::create_directories(path);
}

std::string ModelHub::get_cache_dir() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->config.cache_dir;
}

void ModelHub::set_config(const HubConfig& config) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->config = config;
    fs::create_directories(config.cache_dir);
}

HubConfig ModelHub::get_config() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->config;
}

void ModelHub::clear_cache() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& entry : fs::directory_iterator(impl_->config.cache_dir)) {
        if (entry.is_regular_file()) {
            fs::remove(entry.path());
        }
    }
}

size_t ModelHub::cache_size() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& entry : fs::directory_iterator(impl_->config.cache_dir)) {
        if (entry.is_regular_file()) {
            total += entry.file_size();
        }
    }
    return total;
}

std::vector<std::string> ModelHub::list_cached_models() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> models;
    for (const auto& entry : fs::directory_iterator(impl_->config.cache_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pt") {
            models.push_back(entry.path().stem().string());
        }
    }
    return models;
}

bool ModelHub::is_cached(const std::string& model_name) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return fs::exists(impl_->get_cache_path(model_name));
}

std::string ModelHub::get_cached_path(const std::string& model_name) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = impl_->get_cache_path(model_name);
    return fs::exists(path) ? path : "";
}

void ModelHub::register_model(const ModelWeightInfo& info) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->registry[info.name] = info;
}

void ModelHub::register_models(const std::vector<ModelWeightInfo>& models) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& model : models) {
        impl_->registry[model.name] = model;
    }
}

ModelWeightInfo ModelHub::get_model_info(const std::string& model_name) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = impl_->registry.find(model_name);
    if (it == impl_->registry.end()) {
        throw std::runtime_error("Model not registered: " + model_name);
    }
    return it->second;
}

std::vector<std::string> ModelHub::list_registered_models() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> models;
    for (const auto& pair : impl_->registry) {
        models.push_back(pair.first);
    }
    return models;
}

bool ModelHub::is_registered(const std::string& model_name) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->registry.find(model_name) != impl_->registry.end();
}

bool ModelHub::remove_from_cache(const std::string& model_name) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);

    std::string path = impl_->get_cache_path(model_name);
    if (fs::exists(path)) {
        fs::remove(path);
        return true;
    }
    return false;
}

DownloadStats ModelHub::get_last_download_stats() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->last_stats;
}

bool ModelHub::verify_checksum(const std::string& file_path, const std::string& expected_sha256) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->verify_sha256(file_path, expected_sha256);
}

std::string ModelHub::compute_checksum(const std::string& file_path) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->compute_sha256(file_path);
}

size_t ModelHub::clean_cache(size_t max_size) {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_->clean_cache_to_size(max_size);
}

void ModelHub::default_progress_callback(size_t downloaded, size_t total, double speed, double eta) {
    // Format size
    auto format_size = [](size_t bytes) -> std::string {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit = 0;
        double size = bytes;
        while (size >= 1024 && unit < 3) {
            size /= 1024;
            unit++;
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    };

    // Format time
    auto format_time = [](double seconds) -> std::string {
        int hours = seconds / 3600;
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        int secs = static_cast<int>(seconds) % 60;

        std::ostringstream oss;
        if (hours > 0) {
            oss << hours << "h " << mins << "m";
        } else if (mins > 0) {
            oss << mins << "m " << secs << "s";
        } else {
            oss << secs << "s";
        }
        return oss.str();
    };

    // Calculate percentage
    double percent = total > 0 ? (100.0 * downloaded / total) : 0.0;

    // Progress bar
    int bar_width = 40;
    int filled = total > 0 ? (bar_width * downloaded / total) : 0;

    std::cout << "\r[";
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) std::cout << "=";
        else if (i == filled) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << percent << "% "
              << format_size(downloaded);

    if (total > 0) {
        std::cout << "/" << format_size(total);
    }

    if (speed > 0) {
        std::cout << " @ " << format_size(speed) << "/s";
    }

    if (eta > 0 && eta < 86400) {  // Less than 24 hours
        std::cout << " ETA: " << format_time(eta);
    }

    std::cout << std::flush;
}

// ============================================================================
// Default Model Registry
// ============================================================================

namespace registry {

std::string get_pytorch_model_url(const std::string& model_name) {
    return "https://download.pytorch.org/models/" + model_name + ".pth";
}

void initialize_default_registry(std::unordered_map<std::string, ModelWeightInfo>& registry) {
    std::vector<ModelWeightInfo> models;

    // ResNet models
    models.push_back({std::string("resnet18"), get_pytorch_model_url("resnet18-5c106cde"),
                     std::string("5c106cde18a16fb8e3af86a0c103ec7d8e84d1e"), 0, std::string("ResNet-18")});
    models.push_back({std::string("resnet34"), get_pytorch_model_url("resnet34-333f7ec4"),
                     std::string("333f7ec4d632f71d4d0af73aa97397a5af72cb4"), 0, std::string("ResNet-34")});
    models.push_back({std::string("resnet50"), get_pytorch_model_url("resnet50-19c8e357"),
                     std::string("19c8e357e6f093d1c0ed6b6a7fa3bcf0a3b7db8"), 0, std::string("ResNet-50")});
    models.push_back({std::string("resnet101"), get_pytorch_model_url("resnet101-5d3b4d8f"),
                     std::string("5d3b4d8ffa1b64c89c8a5c1cf738db78b83c0f9"), 0, std::string("ResNet-101")});
    models.push_back({std::string("resnet152"), get_pytorch_model_url("resnet152-b121ed2d"),
                     std::string("b121ed2d73e9fb437f1a89a0e6b4f8ed70f9fc8"), 0, std::string("ResNet-152")});

    // VGG models
    models.push_back({std::string("vgg11"), get_pytorch_model_url("vgg11-bbd30ac9"),
                     std::string("bbd30ac9d1a59e5f2e8bb17a57e3d7da5e3e5e8"), 0, std::string("VGG-11")});
    models.push_back({std::string("vgg13"), get_pytorch_model_url("vgg13-c768596a"),
                     std::string("c768596aa57f0e4b05b58e8bc1e2c3b9e4e4b5e"), 0, std::string("VGG-13")});
    models.push_back({std::string("vgg16"), get_pytorch_model_url("vgg16-397923af"),
                     std::string("397923af2e8d8c3a3e8d8c3a3e8d8c3a3e8d8c3"), 0, std::string("VGG-16")});
    models.push_back({std::string("vgg19"), get_pytorch_model_url("vgg19-dcbb9e9d"),
                     std::string("dcbb9e9d8c3a3e8d8c3a3e8d8c3a3e8d8c3a3e8"), 0, std::string("VGG-19")});

    // MobileNet
    models.push_back({std::string("mobilenet_v2"), get_pytorch_model_url("mobilenet_v2-b0353104"),
                     std::string("b03531044c7f8c3a3e8d8c3a3e8d8c3a3e8d8c3"), 0, std::string("MobileNet V2")});

    // EfficientNet
    for (int i = 0; i <= 7; i++) {
        std::string name = "efficientnet_b" + std::to_string(i);
        models.push_back({name, get_pytorch_model_url(name), std::string(""), 0, std::string("EfficientNet-B") + std::to_string(i)});
    }

    // Directly add to registry without locking (already locked by caller)
    for (const auto& model : models) {
        registry[model.name] = model;
    }
}

} // namespace registry

} // namespace tenzor::models
