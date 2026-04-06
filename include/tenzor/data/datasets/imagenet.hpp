#pragma once

#include "tenzor/data/dataset.hpp"
#include "tenzor/core/tensor.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace tenzor::data::datasets {

/**
 * @brief ImageNet-style folder dataset
 *
 * Loads images from a directory structure:
 *   root/class_name_1/xxx.png
 *   root/class_name_2/yyy.jpg
 *   ...
 *
 * Classes are assigned indices in sorted order of directory names.
 * Images are loaded on-the-fly (lazy) to avoid excessive memory use.
 */
class ImageFolder : public MapDataset {
public:
    /**
     * @brief Construct ImageFolder dataset
     * @param root_dir Root directory containing class subdirectories
     * @param image_size Resize images to this size (square crop)
     * @param extensions Allowed file extensions (default: common image formats)
     */
    ImageFolder(const std::string& root_dir,
                int64_t image_size = 224,
                std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"});

    auto size() const -> size_t override;
    auto get(size_t index) -> std::pair<Tensor, Tensor> override;

    /// Get the class names in order
    auto class_names() const -> const std::vector<std::string>&;

    /// Number of classes found
    auto num_classes() const -> int64_t;

private:
    struct Sample {
        std::filesystem::path path;
        int64_t label;
    };

    std::vector<Sample> samples_;
    std::vector<std::string> class_names_;
    int64_t image_size_;
};

} // namespace tenzor::data::datasets
