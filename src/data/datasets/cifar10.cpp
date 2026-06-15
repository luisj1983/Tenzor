#include "tenzor/data/datasets/cifar10.hpp"
#include "tenzor/ops/creation.hpp"
#include <fstream>
#include <stdexcept>

namespace tenzor::data::datasets {

namespace {

// CIFAR-10 binary format: each record is 1 byte label + 3072 bytes image (3*32*32)
constexpr int64_t kImageBytes = 3 * 32 * 32;
constexpr int64_t kRecordBytes = 1 + kImageBytes;
constexpr int64_t kSamplesPerBatch = 10000;

auto load_cifar_batch(const std::string& path, std::vector<uint8_t>& images,
                      std::vector<uint8_t>& labels, [[maybe_unused]] int label_offset = 0) -> void {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("CIFAR: cannot open " + path);
    }

    for (int64_t i = 0; i < kSamplesPerBatch; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (file.gcount() != 1) {
            throw std::runtime_error(
                "CIFAR: truncated file " + path + " (expected " +
                std::to_string(kSamplesPerBatch) + " records, got " +
                std::to_string(i) + ")");
        }
        labels.push_back(label);

        size_t offset = images.size();
        images.resize(offset + kImageBytes);
        file.read(reinterpret_cast<char*>(images.data() + offset), kImageBytes);
        if (file.gcount() != static_cast<std::streamsize>(kImageBytes)) {
            throw std::runtime_error(
                "CIFAR: truncated image data in " + path + " at record " +
                std::to_string(i));
        }
    }
}

auto build_tensors(const std::vector<uint8_t>& img_data,
                   const std::vector<uint8_t>& lbl_data,
                   int64_t n, bool normalize, int64_t num_classes)
    -> std::pair<Tensor, Tensor> {
    // Images: stored as R plane, G plane, B plane (channel-first already)
    auto images = zeros({n, 3, 32, 32}, DType::Float32, Device::cpu());
    auto* dst = images.data<float>();
    float scale = normalize ? (1.0f / 255.0f) : 1.0f;
    for (size_t i = 0; i < img_data.size(); ++i) {
        dst[i] = static_cast<float>(img_data[i]) * scale;
    }

    auto labels = zeros({n}, DType::Int64, Device::cpu());
    auto* lbl_dst = labels.data<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        auto label = static_cast<int64_t>(lbl_data[i]);
        if (label < 0 || label >= num_classes) {
            throw std::runtime_error(
                "CIFAR: label " + std::to_string(label) + " at index " +
                std::to_string(i) + " is out of range [0, " +
                std::to_string(num_classes) + ")");
        }
        lbl_dst[i] = label;
    }

    return {images, labels};
}

} // anonymous namespace

CIFAR10::CIFAR10(const std::string& root_dir, bool train, bool normalize) {
    std::vector<uint8_t> img_data;
    std::vector<uint8_t> lbl_data;

    if (train) {
        for (int i = 1; i <= 5; ++i) {
            load_cifar_batch(root_dir + "/data_batch_" + std::to_string(i) + ".bin",
                             img_data, lbl_data);
        }
        num_samples_ = 5 * kSamplesPerBatch;
    } else {
        load_cifar_batch(root_dir + "/test_batch.bin", img_data, lbl_data);
        num_samples_ = kSamplesPerBatch;
    }

    auto [imgs, lbls] = build_tensors(img_data, lbl_data,
                                       static_cast<int64_t>(num_samples_), normalize,
                                       /*num_classes=*/10);
    images_ = std::move(imgs);
    labels_ = std::move(lbls);
}

auto CIFAR10::size() const -> size_t {
    return num_samples_;
}

auto CIFAR10::get(size_t index) -> std::pair<Tensor, Tensor> {
    if (index >= num_samples_) {
        throw std::out_of_range("CIFAR10: index out of range");
    }
    auto i = static_cast<int64_t>(index);
    auto image = images_.slice(0, i, i + 1).squeeze(0);
    auto label = labels_.slice(0, i, i + 1).squeeze(0);
    return {image, label};
}

// CIFAR-100: same binary format but with fine label (byte 1) after coarse label (byte 0)
CIFAR100::CIFAR100(const std::string& root_dir, bool train, bool normalize) {
    std::string path = root_dir + "/" + (train ? "train.bin" : "test.bin");
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("CIFAR100: cannot open " + path);
    }

    // CIFAR-100 format: 1 byte coarse label + 1 byte fine label + 3072 bytes image
    num_samples_ = train ? 50000 : 10000;

    std::vector<uint8_t> img_data;
    std::vector<uint8_t> lbl_data;
    img_data.reserve(num_samples_ * kImageBytes);
    lbl_data.reserve(num_samples_);

    for (size_t i = 0; i < num_samples_; ++i) {
        uint8_t coarse_label, fine_label;
        file.read(reinterpret_cast<char*>(&coarse_label), 1);
        file.read(reinterpret_cast<char*>(&fine_label), 1);
        if (file.gcount() != 1) {
            throw std::runtime_error(
                "CIFAR100: truncated file at record " + std::to_string(i));
        }
        lbl_data.push_back(fine_label);  // Use fine label (100 classes)

        size_t offset = img_data.size();
        img_data.resize(offset + kImageBytes);
        file.read(reinterpret_cast<char*>(img_data.data() + offset), kImageBytes);
        if (file.gcount() != static_cast<std::streamsize>(kImageBytes)) {
            throw std::runtime_error(
                "CIFAR100: truncated image data at record " + std::to_string(i));
        }
    }

    auto [imgs, lbls] = build_tensors(img_data, lbl_data,
                                       static_cast<int64_t>(num_samples_), normalize,
                                       /*num_classes=*/100);
    images_ = std::move(imgs);
    labels_ = std::move(lbls);
}

auto CIFAR100::size() const -> size_t {
    return num_samples_;
}

auto CIFAR100::get(size_t index) -> std::pair<Tensor, Tensor> {
    if (index >= num_samples_) {
        throw std::out_of_range("CIFAR100: index out of range");
    }
    auto i = static_cast<int64_t>(index);
    auto image = images_.slice(0, i, i + 1).squeeze(0);
    auto label = labels_.slice(0, i, i + 1).squeeze(0);
    return {image, label};
}

} // namespace tenzor::data::datasets
