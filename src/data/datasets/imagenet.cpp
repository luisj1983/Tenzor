#include "tenzor/data/datasets/imagenet.hpp"
#include "tenzor/io/image.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/vision.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace tenzor::data::datasets {

namespace fs = std::filesystem;

ImageFolder::ImageFolder(const std::string& root_dir,
                         int64_t image_size,
                         std::vector<std::string> extensions)
    : image_size_(image_size) {

    if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
        throw std::runtime_error("ImageFolder: root directory does not exist: " + root_dir);
    }

    // Resolve the canonical root once so every collected path can be verified
    // to stay under it. On an attacker-controlled dataset (e.g. an extracted
    // tarball) a symlink could otherwise point a "class dir" or "image file"
    // outside the tree (../../etc/..., a device node, or a FIFO) and the
    // is_directory()/is_regular_file() checks — which resolve THROUGH symlinks
    // — would happily follow it. We skip symlinked entries and also confirm
    // containment defensively.
    const fs::path canonical_root = fs::weakly_canonical(fs::path(root_dir));

    auto stays_under_root = [&](const fs::path& p) -> bool {
        const fs::path rel = fs::weakly_canonical(p).lexically_relative(canonical_root);
        return !rel.empty() && *rel.begin() != "..";
    };

    // Collect class directories sorted alphabetically. Skip symlinks so a
    // symlinked subdirectory cannot redirect enumeration outside the root.
    for (const auto& entry : fs::directory_iterator(root_dir)) {
        if (fs::is_symlink(entry.symlink_status())) continue;
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
            // Skip symlinks: a symlink-to-regular-file would otherwise pass the
            // is_regular_file() filter (which resolves through the link) and be
            // decoded, enabling an out-of-tree read via the image decoder.
            if (fs::is_symlink(entry.symlink_status())) continue;
            if (!entry.is_regular_file()) continue;
            // Defense in depth: reject any path that escapes the dataset root.
            if (!stays_under_root(entry.path())) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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

    // Decode the image file (JPEG/PNG/BMP/TGA/GIF/PSD/HDR/PIC/PNM via stb_image)
    // into a UInt8 (3, H, W) tensor.
    Tensor decoded = io::read_image(sample.path.string(), io::ImageMode::RGB);

    // Promote to Float32 in [0, 1].
    Tensor as_float = decoded.to(DType::Float32);
    {
        auto* p = as_float.data<float>();
        int64_t n = as_float.numel();
        for (int64_t i = 0; i < n; ++i) p[i] *= (1.0f / 255.0f);
    }

    // Resize to (3, image_size_, image_size_) via the bilinear interpolate op.
    // interpolate() expects a 4D (N, C, H, W) input; unsqueeze + squeeze around it.
    Tensor batched = as_float.unsqueeze(0); // (1, 3, H, W)
    Tensor resized = ops::interpolate(batched,
                                      {image_size_, image_size_},
                                      /*mode=*/"bilinear",
                                      /*align_corners=*/false);
    Tensor image = resized.squeeze(0); // (3, image_size_, image_size_)

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
