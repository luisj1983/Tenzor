#include "tenzor/data/datasets/imagenet.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <stdexcept>
#include <fstream>

namespace tenzor::data::datasets {

namespace fs = std::filesystem;

ImageFolder::ImageFolder(const std::string& root_dir,
                         int64_t image_size,
                         std::vector<std::string> extensions)
    : image_size_(image_size) {

    if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
        throw std::runtime_error("ImageFolder: root directory does not exist: " + root_dir);
    }

    // Collect class directories sorted alphabetically
    for (const auto& entry : fs::directory_iterator(root_dir)) {
        if (entry.is_directory()) {
            class_names_.push_back(entry.path().filename().string());
        }
    }
    std::sort(class_names_.begin(), class_names_.end());

    if (class_names_.empty()) {
        throw std::runtime_error("ImageFolder: no class subdirectories found in " + root_dir);
    }

    // Build sample list
    for (int64_t label = 0; label < static_cast<int64_t>(class_names_.size()); ++label) {
        fs::path class_dir = fs::path(root_dir) / class_names_[label];
        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            for (const auto& allowed : extensions) {
                if (ext == allowed) {
                    samples_.push_back({entry.path(), label});
                    break;
                }
            }
        }
    }

    if (samples_.empty()) {
        throw std::runtime_error("ImageFolder: no image files found in " + root_dir);
    }
}

auto ImageFolder::size() const -> size_t {
    return samples_.size();
}

auto ImageFolder::get(size_t index) -> std::pair<Tensor, Tensor> {
    if (index >= samples_.size()) {
        throw std::out_of_range("ImageFolder: index out of range");
    }

    const auto& sample = samples_[index];

    // Load raw file bytes and convert to tensor
    // For a real implementation, this would use stb_image or similar.
    // Here we load raw bytes and create a placeholder tensor.
    std::ifstream file(sample.path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("ImageFolder: cannot open " + sample.path.string());
    }

    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw(file_size);
    file.read(reinterpret_cast<char*>(raw.data()), file_size);

    // Create a tensor from raw image bytes
    // In production, decode JPEG/PNG here. For now, return raw bytes as flat tensor.
    auto image = zeros({3, image_size_, image_size_}, DType::Float32, Device::cpu());
    // Fill with raw data scaled to [0,1] up to available bytes
    auto* dst = image.data<float>();
    size_t pixels = std::min(raw.size(),
                             static_cast<size_t>(3 * image_size_ * image_size_));
    for (size_t i = 0; i < pixels; ++i) {
        dst[i] = static_cast<float>(raw[i]) / 255.0f;
    }

    auto label = zeros({}, DType::Int64, Device::cpu());
    label.data<int64_t>()[0] = sample.label;

    return {image, label};
}

auto ImageFolder::class_names() const -> const std::vector<std::string>& {
    return class_names_;
}

auto ImageFolder::num_classes() const -> int64_t {
    return static_cast<int64_t>(class_names_.size());
}

} // namespace tenzor::data::datasets
