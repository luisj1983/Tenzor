// Fail-loud ModelHub stub, compiled when TENZOR_BUILD_MODEL_HUB=OFF.
//
// The pretrained-model factories (alexnet.cpp, bert.cpp, ...) reference
// ModelHub unconditionally; without this TU a hub-less build links cleanly
// but libtenzor_core.so carries undefined ModelHub symbols and fails at
// load time (dlopen/import). Every entry point throws instead.

#include <tenzor/models/hub.hpp>

#include <stdexcept>
#include <string>

namespace tenzor::models {

namespace {

[[noreturn]] void hub_unavailable(const char* fn) {
    throw std::runtime_error(
        std::string("ModelHub::") + fn +
        " is unavailable: Tenzor was built without the model hub "
        "(TENZOR_BUILD_MODEL_HUB=OFF). Rebuild with "
        "-DTENZOR_BUILD_MODEL_HUB=ON (requires CURL + OpenSSL), or "
        "construct the model with pretrained=false and load weights "
        "manually.");
}

}  // namespace

// HubConfig's constructor is declared out-of-line in hub.hpp and is also
// referenced by the Python bindings; keep the same defaults as hub.cpp
// (constructing a config is harmless — only ModelHub operations throw).
HubConfig::HubConfig()
    : cache_dir(""),
      max_cache_size(0),
      verify_checksums(true),
      resume_downloads(true),
      connection_timeout(30),
      max_retries(3),
      show_progress(true) {}

std::string ModelHub::download_weights(const std::string&, const std::string&,
                                       const std::string&, bool, ProgressCallback) {
    hub_unavailable("download_weights");
}

std::string ModelHub::download_pretrained(const std::string&, bool, ProgressCallback) {
    hub_unavailable("download_pretrained");
}

std::string ModelHub::download_pretrained_safetensors(const std::string&, bool, bool,
                                                      ProgressCallback) {
    hub_unavailable("download_pretrained_safetensors");
}

void ModelHub::load_pretrained_weights(nn::Module&, const std::string&, bool) {
    hub_unavailable("load_pretrained_weights");
}

void ModelHub::set_cache_dir(const std::string&) { hub_unavailable("set_cache_dir"); }

std::string ModelHub::get_cache_dir() { hub_unavailable("get_cache_dir"); }

void ModelHub::set_config(const HubConfig&) { hub_unavailable("set_config"); }

HubConfig ModelHub::get_config() { hub_unavailable("get_config"); }

void ModelHub::clear_cache() { hub_unavailable("clear_cache"); }

size_t ModelHub::cache_size() { hub_unavailable("cache_size"); }

std::vector<std::string> ModelHub::list_cached_models() {
    hub_unavailable("list_cached_models");
}

bool ModelHub::is_cached(const std::string&) { hub_unavailable("is_cached"); }

std::string ModelHub::get_cached_path(const std::string&) {
    hub_unavailable("get_cached_path");
}

void ModelHub::register_model(const ModelWeightInfo&) { hub_unavailable("register_model"); }

void ModelHub::register_models(const std::vector<ModelWeightInfo>&) {
    hub_unavailable("register_models");
}

ModelWeightInfo ModelHub::get_model_info(const std::string&) {
    hub_unavailable("get_model_info");
}

std::vector<std::string> ModelHub::list_registered_models() {
    hub_unavailable("list_registered_models");
}

bool ModelHub::is_registered(const std::string&) { hub_unavailable("is_registered"); }

bool ModelHub::remove_from_cache(const std::string&) { hub_unavailable("remove_from_cache"); }

DownloadStats ModelHub::get_last_download_stats() {
    hub_unavailable("get_last_download_stats");
}

bool ModelHub::verify_checksum(const std::string&, const std::string&) {
    hub_unavailable("verify_checksum");
}

std::string ModelHub::compute_checksum(const std::string&) {
    hub_unavailable("compute_checksum");
}

size_t ModelHub::clean_cache(size_t) { hub_unavailable("clean_cache"); }

}  // namespace tenzor::models
