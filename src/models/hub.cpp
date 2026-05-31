#include "tenzor/models/hub.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/nn/safetensors.hpp"  // Audit H2: format-aware loader
#include "tenzor/io/torch_pickle.hpp" // H2-followup: native .pth pickle parser
#include "tenzor/utils/log.hpp"      // Audit I.4: unified logger
#include <unordered_set>  // H3-followup-keyremap
#include <cctype>          // H3-followup-keyremap
#include <curl/curl.h>
#include <openssl/evp.h>
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
                                 [[maybe_unused]] curl_off_t ultotal, [[maybe_unused]] curl_off_t ulnow) {
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
                // Audit I.4: unified logger.
                TENZOR_LOG_WARN("ModelHub: download failed (attempt {}/{}): {}",
                                retries,
                                config.max_retries,
                                curl_easy_strerror(res));
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

        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Failed to create EVP_MD_CTX for SHA256");
        }
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);

        char buffer[8192];
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            EVP_DigestUpdate(ctx, buffer, file.gcount());
        }

        unsigned char hash[EVP_MAX_MD_SIZE];
        unsigned int hash_len = 0;
        EVP_DigestFinal_ex(ctx, hash, &hash_len);
        EVP_MD_CTX_free(ctx);

        std::ostringstream oss;
        for (unsigned int i = 0; i < hash_len; i++) {
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

// Internal forward declaration (definition below, in namespace registry).
// Kept out of the public header — see hub.hpp.
namespace registry {
void initialize_default_registry(std::unordered_map<std::string, ModelWeightInfo>& registry);
}  // namespace registry

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
                // Audit I.4: unified logger.
                TENZOR_LOG_WARN("ModelHub: cached file checksum mismatch, re-downloading: {}",
                                cache_path);
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
            // Audit C.7: if this name was deliberately removed from the
            // default registry (no verifiable safetensors mirror), surface
            // a precise diagnostic rather than the generic "not registered"
            // message.  Callers must either register a working mirror via
            // `ModelHub::register_model()` or pick a different model.
            std::string reason = registry::removed_pretrained_reason(model_name);
            if (!reason.empty()) {
                throw std::runtime_error(
                    "Pretrained weights for '" + model_name +
                    "' were removed from the default Tenzor model zoo: " +
                    reason +
                    ". To re-enable, call ModelHub::register_model() with a "
                    "verified safetensors URL + SHA256 checksum, or use a "
                    "different model. See audit item C.7 in src/models/hub.cpp.");
            }
            throw std::runtime_error("Model not registered: " + model_name);
        }
        info = it->second;
    }

    return download_weights(model_name, info.url, info.sha256, show_progress, progress_callback);
}

std::string ModelHub::download_pretrained_safetensors(
    const std::string& model_name,
    bool prefer_safetensors,
    bool show_progress,
    ProgressCallback progress_callback)
{
    ensure_initialized();

    ModelWeightInfo info;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = impl_->registry.find(model_name);
        if (it == impl_->registry.end()) {
            // Audit C.7: same removal-aware diagnostic as
            // `download_pretrained()`.  This path is what users hit when
            // calling `MaskRCNN::load_pretrained()`, `VGG::load_pretrained()`,
            // etc., since those wrappers always go through the safetensors
            // entrypoint.
            std::string reason = registry::removed_pretrained_reason(model_name);
            if (!reason.empty()) {
                throw std::runtime_error(
                    "Pretrained weights for '" + model_name +
                    "' were removed from the default Tenzor model zoo: " +
                    reason +
                    ". To re-enable, call ModelHub::register_model() with a "
                    "verified safetensors URL + SHA256 checksum, or use a "
                    "different model. See audit item C.7 in src/models/hub.cpp.");
            }
            throw std::runtime_error("Model not registered: " + model_name);
        }
        info = it->second;
    }

    // H3-followup: route to SafeTensors mirror when one is registered and
    // the caller opted in. The downloaded file flows through H2's format
    // dispatcher, which handles `.safetensors` natively without the
    // PyTorch-pickle parser dependency.
    if (prefer_safetensors && !info.safetensors_url.empty()) {
        // Use a different cache key (".safetensors" suffix) so the legacy
        // .pth cache doesn't collide with the safetensors download.
        std::string st_key = model_name + ".safetensors";
        return download_weights(st_key, info.safetensors_url, /*sha256=*/"",
                                 show_progress, progress_callback);
    }
    // Fall through to legacy .pth URL — the caller will get H2's actionable
    // error if the format isn't supported.
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

    // Audit H2: dispatch to the right deserializer by file extension or
    // first-bytes sniff. Previously this always called `Serializer::load`
    // (Tenzor native format) and would fail with a cryptic "Invalid file
    // format: magic number mismatch" for any .safetensors or .pth file.
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    try {
        if (ends_with(weights_path, ".safetensors")) {
            // HuggingFace SafeTensors format — preferred for new models.
            state_dict = nn::SafeTensorsSerializer::load(weights_path);
        } else if (ends_with(weights_path, ".pth") || ends_with(weights_path, ".pt") ||
                   ends_with(weights_path, ".bin")) {
            // H2-followup: native C++ PyTorch checkpoint loader. Parses the
            // torch.save ZIP archive (data.pkl + data/N entries), decodes the
            // Python pickle stream (protocol 2-5 subset that torch.save uses),
            // and reconstructs each `_rebuild_tensor_v2(storage, offset, size,
            // stride, ...)` call into a Tenzor CPU Tensor. Handles
            // Float32/64/16/BFloat16/Int8/16/32/64/UInt8/Bool storages.
            // Non-contiguous strides and DEFLATEd entries are rejected with
            // actionable errors; legacy pre-1.6 non-zipfile pickles are
            // pointed at the safetensors conversion workflow.
            state_dict = tenzor::io::load_torch_pickle(weights_path);
        } else {
            // Tenzor native format (or unknown extension — try native).
            state_dict = nn::Serializer::load(weights_path);
        }
    } catch (const std::exception& e) {
        load_error = e.what();
        if (strict) {
            // Audit H2: in strict mode, throw the deserialization error
            // immediately so the user sees the *actual* cause (e.g. "use the
            // .safetensors variant"). Previously the code continued to
            // `load_state_dict(empty)` which threw "Missing keys ..." first
            // and masked the real error.
            throw std::runtime_error(std::string("Failed to load weights: ") + load_error);
        }
        // Audit I.4: unified logger.
        TENZOR_LOG_WARN("ModelHub: partial weight loading - {}", load_error);
    }

    // H3-followup-keyremap: rewrite common torchvision/timm naming conventions
    // into tenzor's. The biggest mechanical difference is that tenzor's
    // `Sequential::add_module` names submodules `module_0`, `module_1`, ...
    // while torchvision (and timm, which preserves torchvision's names)
    // uses `0`, `1`, ... — so the same ResNet-50 has e.g.
    //   timm:    `layer1.0.conv1.weight`
    //   tenzor:  `layer1.module_0.conv1.weight`
    // The remap below also drops keys that exist in PyTorch but not in
    // tenzor (`num_batches_tracked` on BatchNorm), since otherwise the
    // strict `load_state_dict` errors out on those extras.
    if (!state_dict.empty()) {
        // Build the target set of parameter+buffer names from the model
        // itself so we only remap keys that wouldn't otherwise match.
        std::unordered_set<std::string> target_keys;
        for (const auto& [n, p] : model.named_parameters())  target_keys.insert(n);
        for (const auto& [n, b] : model.named_buffers())     target_keys.insert(n);

        std::unordered_map<std::string, Tensor> remapped;
        remapped.reserve(state_dict.size());
        for (auto& [k, t] : state_dict) {
            // Drop PyTorch-only BatchNorm bookkeeping (tenzor doesn't store it).
            if (k.size() >= 19 &&
                k.compare(k.size() - 19, 19, "num_batches_tracked") == 0) {
                continue;
            }
            // If the key already matches, keep as-is.
            if (target_keys.count(k)) {
                remapped.emplace(k, t);
                continue;
            }
            // Try `.{N}.` → `.module_{N}.` rewrite over every digit-run.
            // Walk the key; whenever we see `.<digits>.` or a trailing
            // `.<digits>` segment, prepend `module_` to the digit run.
            std::string rk;
            rk.reserve(k.size() + 16);
            size_t i = 0;
            while (i < k.size()) {
                if (k[i] == '.') {
                    // Look ahead for a digit run.
                    size_t j = i + 1;
                    while (j < k.size() && std::isdigit(static_cast<unsigned char>(k[j]))) ++j;
                    if (j > i + 1 && (j == k.size() || k[j] == '.')) {
                        // Found `.<digits>` boundary — rewrite.
                        rk.push_back('.');
                        rk.append("module_");
                        rk.append(k, i + 1, j - (i + 1));
                        i = j;
                        continue;
                    }
                }
                rk.push_back(k[i]);
                ++i;
            }
            if (target_keys.count(rk)) {
                remapped.emplace(rk, t);
                continue;
            }
            // Couldn't match. In strict mode, keep the original key so the
            // subsequent load_state_dict error message lists the actual
            // unmatched name. In non-strict mode (G.1), drop the key with
            // a warning — preventing the previous "silently keep extras
            // in dict, hope load_state_dict ignores them" behaviour.
            if (strict) {
                remapped.emplace(k, t);
            } else {
                // Audit I.4: unified logger.
                TENZOR_LOG_WARN("ModelHub: dropping unmatched checkpoint key '{}' "
                                "(no target param/buffer); use strict=true to see "
                                "all unmatched keys.",
                                k);
            }
        }
        state_dict = std::move(remapped);
    }

    // Attempt to load state into model. Only meaningful if deserialization
    // succeeded (load_error empty) — otherwise we'd just be calling
    // load_state_dict with an empty map, which throws "Missing keys" and
    // masks the real cause.
    try {
        model.load_state_dict(state_dict);
    } catch (const std::exception& e) {
        if (strict) {
            throw std::runtime_error(std::string("Failed to load state dict: ") + e.what());
        } else {
            // Audit I.4: unified logger.
            TENZOR_LOG_WARN("ModelHub: state_dict loading failed (non-strict): {}",
                            e.what());
        }
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

// H3-followup: build a SafeTensors mirror URL for a `timm`-hosted model.
// HuggingFace's `timm` org mirrors the canonical torchvision weights in
// safetensors format under model names like `timm/resnet50.a1_in1k`.
// This is the only mirror that's been kept in sync with the upstream
// torchvision recipes, so it's a stable target for the H3-followup-keyremap
// post-load weight renaming.
std::string get_timm_safetensors_url(const std::string& timm_model_id) {
    return "https://huggingface.co/timm/" + timm_model_id +
           "/resolve/main/model.safetensors";
}

// Audit C.7: known-removed model names.  These names used to be registered
// in `initialize_default_registry` but had neither a verifiable safetensors
// mirror nor a checksum, so every `download_pretrained()` call would either
// silently download an unverifiable .pth (and then fail in the pickle parser)
// or, in the safetensors path, hit an empty URL.  Per the "no entry that
// fails by default" rule, the entries were removed; this map records *why*
// so that `download_pretrained(name)` can throw an actionable error rather
// than the generic "Model not registered" message.
//
// When a real safetensors mirror becomes available (timm publishes one, or
// the Tenzor model-hub starts self-hosting), drop the name from this map
// and re-add it to `initialize_default_registry` with the new URL.
const std::unordered_map<std::string, std::string>& removed_pretrained_reasons() {
    static const std::unordered_map<std::string, std::string> table = {
        // VGG (plain + batch-norm) — torchvision-canonical only, no timm mirror.
        {"vgg11",     "no safetensors mirror published by timm"},
        {"vgg13",     "no safetensors mirror published by timm"},
        {"vgg16",     "no safetensors mirror published by timm"},
        {"vgg19",     "no safetensors mirror published by timm"},
        {"vgg11_bn",  "no safetensors mirror published by timm"},
        {"vgg13_bn",  "no safetensors mirror published by timm"},
        {"vgg16_bn",  "no safetensors mirror published by timm"},
        {"vgg19_bn",  "no safetensors mirror published by timm"},
        // Single-file CNNs without a working safetensors mirror.
        {"alexnet",        "no safetensors mirror published by timm"},
        {"googlenet",      "no safetensors mirror published by timm"},
        {"inception_v3",   "no safetensors mirror published by timm"},
        // Detection — full-model COCO checkpoints, .pth-only.
        {"mask_rcnn_resnet50_fpn",
         "torchvision detection checkpoint — no safetensors mirror"},
        {"faster_rcnn_resnet50_fpn",
         "torchvision detection checkpoint — no safetensors mirror"},
        {"retinanet_resnet50_fpn",
         "torchvision detection checkpoint — no safetensors mirror"},
        // Segmentation — torchvision .pth-only.
        {"deeplabv3_resnet50",
         "torchvision segmentation checkpoint — no safetensors mirror"},
        {"deeplabv3_resnet101",
         "torchvision segmentation checkpoint — no safetensors mirror"},
        {"fcn_resnet50",
         "torchvision segmentation checkpoint — no safetensors mirror"},
        {"fcn_resnet101",
         "torchvision segmentation checkpoint — no safetensors mirror"},
        // YOLO — Ultralytics .pt only (PyTorch pickle archive).
        {"yolov3",  "Ultralytics .pt pickle archive — no safetensors mirror"},
        {"yolov5n", "Ultralytics .pt pickle archive — no safetensors mirror"},
        {"yolov5s", "Ultralytics .pt pickle archive — no safetensors mirror"},
        {"yolov5m", "Ultralytics .pt pickle archive — no safetensors mirror"},
        {"yolov5l", "Ultralytics .pt pickle archive — no safetensors mirror"},
        {"yolov5x", "Ultralytics .pt pickle archive — no safetensors mirror"},
    };
    return table;
}

// Returns a non-empty diagnostic message if `model_name` was removed from
// the default registry per audit C.7; returns empty string otherwise.
std::string removed_pretrained_reason(const std::string& model_name) {
    const auto& table = removed_pretrained_reasons();
    auto it = table.find(model_name);
    if (it == table.end()) {
        return std::string();
    }
    return it->second;
}

void initialize_default_registry(std::unordered_map<std::string, ModelWeightInfo>& registry) {
    std::vector<ModelWeightInfo> models;

    // ============================================================================
    // Audit item C.7 — pretrained checksums + "no entry that fails by default"
    // ============================================================================
    //
    // Every entry below has a non-empty `safetensors_url` pointing at a real,
    // currently-resolvable HuggingFace/timm mirror.  The `sha256` field is
    // intentionally left empty: the canonical safetensors checksum for each
    // file will be filled in at build/release time by
    // `tools/check_pretrained_weights.py`, which downloads each entry and
    // records the observed SHA256.  Until the build pipeline ships those
    // checksums, `download_weights` skips checksum verification for any
    // entry whose `sha256` is empty (this is existing behaviour, see the
    // `expected_sha256.empty()` short-circuits in `download_weights`).
    //
    // Entries that previously had neither a working safetensors mirror nor a
    // verifiable .pth (vgg*, deeplab*, fcn_*, faster_rcnn, mask_rcnn,
    // retinanet, yolo*, alexnet, googlenet, inception_v3) have been
    // removed entirely — they are listed in `removed_pretrained_reason()`
    // below so that `download_pretrained("vgg16")` and similar calls fail
    // with a clear, actionable error explaining why the entry was dropped.
    //
    // Anything not in the removal table and not in the kept set below will
    // throw the generic "Model not registered" error — that's by design.

    // ----- ResNet ------------------------------------------------------------
    // Legacy .pth URLs kept as a fallback diagnostic (download_pretrained still
    // tries them when the caller doesn't ask for safetensors).  Real users
    // should call `download_pretrained_safetensors`.
    models.push_back({std::string("resnet18"), get_pytorch_model_url("resnet18-5c106cde"),
                     std::string(""), 0, std::string("ResNet-18"),
                     get_timm_safetensors_url("resnet18.a1_in1k")});
    models.push_back({std::string("resnet34"), get_pytorch_model_url("resnet34-333f7ec4"),
                     std::string(""), 0, std::string("ResNet-34"),
                     get_timm_safetensors_url("resnet34.a1_in1k")});
    models.push_back({std::string("resnet50"), get_pytorch_model_url("resnet50-19c8e357"),
                     std::string(""), 0, std::string("ResNet-50"),
                     get_timm_safetensors_url("resnet50.a1_in1k")});
    models.push_back({std::string("resnet101"), get_pytorch_model_url("resnet101-5d3b4d8f"),
                     std::string(""), 0, std::string("ResNet-101"),
                     get_timm_safetensors_url("resnet101.a1_in1k")});
    models.push_back({std::string("resnet152"), get_pytorch_model_url("resnet152-b121ed2d"),
                     std::string(""), 0, std::string("ResNet-152"),
                     get_timm_safetensors_url("resnet152.a1h_in1k")});

    // ----- MobileNet V2 ------------------------------------------------------
    models.push_back({std::string("mobilenet_v2"), get_pytorch_model_url("mobilenet_v2-b0353104"),
                     std::string(""), 0, std::string("MobileNet V2"),
                     get_timm_safetensors_url("mobilenetv2_100.ra_in1k")});

    // ----- EfficientNet (B0..B7, all timm-mirrored) --------------------------
    for (int i = 0; i <= 7; i++) {
        std::string name = "efficientnet_b" + std::to_string(i);
        std::string timm_id = "tf_efficientnet_b" + std::to_string(i) + ".in1k";
        models.push_back({name, get_pytorch_model_url(name), std::string(""), 0,
                          std::string("EfficientNet-B") + std::to_string(i),
                          get_timm_safetensors_url(timm_id)});
    }

    // ----- ResNeXt / Wide ResNet --------------------------------------------
    models.push_back({std::string("resnext50_32x4d"),
                      get_pytorch_model_url("resnext50_32x4d-7cdf4587"),
                      std::string(""), 0, std::string("ResNeXt-50 32x4d"),
                      get_timm_safetensors_url("resnext50_32x4d.a1h_in1k")});
    models.push_back({std::string("resnext101_32x8d"),
                      get_pytorch_model_url("resnext101_32x8d-8ba56ff5"),
                      std::string(""), 0, std::string("ResNeXt-101 32x8d"),
                      get_timm_safetensors_url("resnext101_32x8d.tv_in1k")});
    models.push_back({std::string("wide_resnet50_2"),
                      get_pytorch_model_url("wide_resnet50_2-95faca4d"),
                      std::string(""), 0, std::string("Wide ResNet 50-2"),
                      get_timm_safetensors_url("wide_resnet50_2.racm_in1k")});
    models.push_back({std::string("wide_resnet101_2"),
                      get_pytorch_model_url("wide_resnet101_2-32ee1156"),
                      std::string(""), 0, std::string("Wide ResNet 101-2"),
                      get_timm_safetensors_url("wide_resnet101_2.tv2_in1k")});

    // ----- MobileNet V3 ------------------------------------------------------
    models.push_back({std::string("mobilenet_v3_large"),
                      get_pytorch_model_url("mobilenet_v3_large-8738ca79"),
                      std::string(""), 0, std::string("MobileNet V3 Large"),
                      get_timm_safetensors_url("mobilenetv3_large_100.ra_in1k")});
    models.push_back({std::string("mobilenet_v3_small"),
                      get_pytorch_model_url("mobilenet_v3_small-047dcff4"),
                      std::string(""), 0, std::string("MobileNet V3 Small"),
                      get_timm_safetensors_url("mobilenetv3_small_100.lamb_in1k")});

    // ----- ConvNeXt ----------------------------------------------------------
    models.push_back({std::string("convnext_tiny"),
                      get_pytorch_model_url("convnext_tiny-983f1562"),
                      std::string(""), 0, std::string("ConvNeXt-Tiny"),
                      get_timm_safetensors_url("convnext_tiny.fb_in1k")});
    models.push_back({std::string("convnext_small"),
                      get_pytorch_model_url("convnext_small-0c510722"),
                      std::string(""), 0, std::string("ConvNeXt-Small"),
                      get_timm_safetensors_url("convnext_small.fb_in1k")});
    models.push_back({std::string("convnext_base"),
                      get_pytorch_model_url("convnext_base-6075fbad"),
                      std::string(""), 0, std::string("ConvNeXt-Base"),
                      get_timm_safetensors_url("convnext_base.fb_in1k")});
    models.push_back({std::string("convnext_large"),
                      get_pytorch_model_url("convnext_large-ea097f82"),
                      std::string(""), 0, std::string("ConvNeXt-Large"),
                      get_timm_safetensors_url("convnext_large.fb_in1k")});

    // ----- Swin Transformer V1 ----------------------------------------------
    models.push_back({std::string("swin_t"),
                      get_pytorch_model_url("swin_t-704ceda3"),
                      std::string(""), 0, std::string("Swin Transformer Tiny"),
                      get_timm_safetensors_url("swin_tiny_patch4_window7_224.ms_in1k")});
    models.push_back({std::string("swin_s"),
                      get_pytorch_model_url("swin_s-5e29d889"),
                      std::string(""), 0, std::string("Swin Transformer Small"),
                      get_timm_safetensors_url("swin_small_patch4_window7_224.ms_in1k")});
    models.push_back({std::string("swin_b"),
                      get_pytorch_model_url("swin_b-68c6b09e"),
                      std::string(""), 0, std::string("Swin Transformer Base"),
                      get_timm_safetensors_url("swin_base_patch4_window7_224.ms_in1k")});
    models.push_back({std::string("swin_large"),
                      std::string(""), std::string(""), 0,
                      std::string("Swin Transformer Large"),
                      get_timm_safetensors_url("swin_large_patch4_window7_224.ms_in22k_ft_in1k")});

    // ============================================================================
    // REMOVED — kept only as a comment manifest so future grep finds it:
    //
    //   vgg11, vgg13, vgg16, vgg19                          (no safetensors mirror)
    //   vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn              (no safetensors mirror)
    //   alexnet, googlenet, inception_v3                    (no safetensors mirror)
    //   mask_rcnn_resnet50_fpn, faster_rcnn_resnet50_fpn,
    //   retinanet_resnet50_fpn                              (detection — no safetensors)
    //   deeplabv3_resnet50, deeplabv3_resnet101,
    //   fcn_resnet50, fcn_resnet101                         (segmentation — no safetensors)
    //   yolov3, yolov5n, yolov5s, yolov5m, yolov5l, yolov5x (Ultralytics .pt only)
    //
    // These names route through `removed_pretrained_reason()` so callers get
    // an actionable error explaining why the entry was dropped rather than
    // silently downloading a file the loader will reject.
    // ============================================================================

    // Directly add to registry without locking (already locked by caller)
    for (const auto& model : models) {
        registry[model.name] = model;
    }
}

} // namespace registry

} // namespace tenzor::models
