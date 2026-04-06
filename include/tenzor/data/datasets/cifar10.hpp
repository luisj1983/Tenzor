#pragma once

#include "tenzor/data/dataset.hpp"
#include "tenzor/core/tensor.hpp"
#include <string>
#include <vector>

namespace tenzor::data::datasets {

/**
 * @brief CIFAR-10 image classification dataset
 *
 * Loads the CIFAR-10 dataset from binary batch files.
 * Images are 32x32 RGB, 10 classes.
 *
 * Expected file layout in root_dir:
 *   data_batch_1.bin ... data_batch_5.bin  (training)
 *   test_batch.bin                          (test)
 */
class CIFAR10 : public MapDataset {
public:
    /**
     * @brief Construct CIFAR-10 dataset
     * @param root_dir Directory containing CIFAR-10 binary files
     * @param train If true, load training set (50K); else test set (10K)
     * @param normalize If true, scale pixel values to [0, 1]
     */
    CIFAR10(const std::string& root_dir, bool train = true, bool normalize = true);

    auto size() const -> size_t override;
    auto get(size_t index) -> std::pair<Tensor, Tensor> override;

    /// Number of classes
    static constexpr int64_t num_classes = 10;
    /// Image height and width
    static constexpr int64_t image_size = 32;
    /// Number of channels (RGB)
    static constexpr int64_t num_channels = 3;

private:
    Tensor images_;   ///< [N, 3, 32, 32] float32
    Tensor labels_;   ///< [N] int64
    size_t num_samples_;
};

/**
 * @brief CIFAR-100 image classification dataset
 *
 * Same binary format as CIFAR-10 but with 100 classes.
 * Expected files: train.bin, test.bin
 */
class CIFAR100 : public MapDataset {
public:
    CIFAR100(const std::string& root_dir, bool train = true, bool normalize = true);

    auto size() const -> size_t override;
    auto get(size_t index) -> std::pair<Tensor, Tensor> override;

    static constexpr int64_t num_classes = 100;

private:
    Tensor images_;
    Tensor labels_;
    size_t num_samples_;
};

} // namespace tenzor::data::datasets
