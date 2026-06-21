#include "tenzor/data/datasets/mnist.hpp"
#include "tenzor/ops/creation.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace tenzor::data::datasets {

namespace {

auto read_be_uint32(std::ifstream& f) -> uint32_t {
    uint8_t buf[4] = {0, 0, 0, 0};
    f.read(reinterpret_cast<char*>(buf), 4);
    // On a truncated/empty file read() short-reads and leaves buf partly
    // indeterminate (here zero-initialized). Reject the short read here rather
    // than assembling a header value from incomplete bytes downstream.
    if (f.gcount() != 4) {
        throw std::runtime_error("MNIST: truncated header (expected 4 bytes)");
    }
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8) |
           static_cast<uint32_t>(buf[3]);
}

auto read_idx_images(const std::string& path) -> std::pair<std::vector<uint8_t>, std::vector<int64_t>> {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("MNIST: cannot open " + path);
    }

    uint32_t magic = read_be_uint32(file);
    if (magic != 2051) {
        throw std::runtime_error("MNIST: invalid image file magic " + std::to_string(magic));
    }

    uint32_t num_images = read_be_uint32(file);
    uint32_t rows = read_be_uint32(file);
    uint32_t cols = read_be_uint32(file);
    if (!file) {
        throw std::runtime_error("MNIST: truncated header in " + path);
    }

    size_t total;
    if (__builtin_mul_overflow(static_cast<size_t>(num_images),
                               static_cast<size_t>(rows), &total) ||
        __builtin_mul_overflow(total, static_cast<size_t>(cols), &total)) {
        throw std::runtime_error("MNIST: image dimensions overflow in " + path);
    }

    // Sanity-bound the declared element count against the bytes actually
    // remaining in the file before allocating, so a crafted header cannot
    // force an enormous allocation.
    std::streampos cur = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos end = file.tellg();
    file.seekg(cur);
    if (cur < 0 || end < 0 ||
        total > static_cast<size_t>(end - cur)) {
        throw std::runtime_error("MNIST: declared image data exceeds file size in " + path);
    }

    std::vector<uint8_t> data(total);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total));
    // A truncated/crafted body would leave the tail silently zero (the output
    // tensor is pre-zeroed), yielding valid-looking black images. Reject it.
    if (file.gcount() != static_cast<std::streamsize>(total)) {
        throw std::runtime_error("MNIST: truncated image data in " + path);
    }

    return {std::move(data), {static_cast<int64_t>(num_images),
                              1, static_cast<int64_t>(rows), static_cast<int64_t>(cols)}};
}

auto read_idx_labels(const std::string& path) -> std::pair<std::vector<uint8_t>, int64_t> {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("MNIST: cannot open " + path);
    }

    uint32_t magic = read_be_uint32(file);
    if (magic != 2049) {
        throw std::runtime_error("MNIST: invalid label file magic " + std::to_string(magic));
    }

    uint32_t num_labels = read_be_uint32(file);
    if (!file) {
        throw std::runtime_error("MNIST: truncated header in " + path);
    }

    // Sanity-bound the declared count against the bytes actually remaining in
    // the file before allocating, so a crafted header cannot force an enormous
    // allocation.
    std::streampos cur = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos end = file.tellg();
    file.seekg(cur);
    if (cur < 0 || end < 0 ||
        static_cast<size_t>(num_labels) > static_cast<size_t>(end - cur)) {
        throw std::runtime_error("MNIST: declared label data exceeds file size in " + path);
    }

    std::vector<uint8_t> data(num_labels);
    file.read(reinterpret_cast<char*>(data.data()), num_labels);
    // A truncated/crafted body would leave the tail silently zero (labels_ is
    // pre-zeroed), turning missing labels into class 0. Reject it.
    if (file.gcount() != static_cast<std::streamsize>(num_labels)) {
        throw std::runtime_error("MNIST: truncated label data in " + path);
    }

    return {std::move(data), static_cast<int64_t>(num_labels)};
}

} // anonymous namespace

MNIST::MNIST(const std::string& root_dir, bool train, bool normalize) {
    std::string img_file = root_dir + "/" + (train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte");
    std::string lbl_file = root_dir + "/" + (train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte");

    auto [img_data, img_shape] = read_idx_images(img_file);
    auto [lbl_data, num_labels] = read_idx_labels(lbl_file);

    num_samples_ = static_cast<size_t>(img_shape[0]);
    if (static_cast<int64_t>(num_samples_) != num_labels) {
        throw std::runtime_error("MNIST: image/label count mismatch");
    }

    // Convert images to float32 tensor [N, 1, 28, 28]
    images_ = zeros(img_shape, DType::Float32, Device::cpu());
    auto* dst = images_.data<float>();
    float scale = normalize ? (1.0f / 255.0f) : 1.0f;
    for (size_t i = 0; i < img_data.size(); ++i) {
        dst[i] = static_cast<float>(img_data[i]) * scale;
    }

    // Convert labels to int64 tensor [N]
    labels_ = zeros({static_cast<int64_t>(num_samples_)}, DType::Int64, Device::cpu());
    auto* lbl_dst = labels_.data<int64_t>();
    for (size_t i = 0; i < num_samples_; ++i) {
        lbl_dst[i] = static_cast<int64_t>(lbl_data[i]);
    }
}

auto MNIST::size() const -> size_t {
    return num_samples_;
}

auto MNIST::get(size_t index) -> std::pair<Tensor, Tensor> {
    if (index >= num_samples_) {
        throw std::out_of_range("MNIST: index " + std::to_string(index) + " out of range");
    }
    // Slice single sample: images_[index], labels_[index]
    auto image = images_.slice(0, static_cast<int64_t>(index), static_cast<int64_t>(index) + 1).squeeze(0);
    auto label = labels_.slice(0, static_cast<int64_t>(index), static_cast<int64_t>(index) + 1).squeeze(0);
    return {image, label};
}

} // namespace tenzor::data::datasets
