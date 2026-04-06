#pragma once

#include "tenzor/data/dataset.hpp"
#include "tenzor/core/tensor.hpp"
#include <string>
#include <vector>

namespace tenzor::data::datasets {

/**
 * @brief MNIST handwritten digit dataset
 *
 * Loads the MNIST dataset from IDX binary files.
 * Images are 28x28 grayscale, labels are 0-9.
 *
 * Expected file layout in root_dir:
 *   train-images-idx3-ubyte (or .gz)
 *   train-labels-idx1-ubyte (or .gz)
 *   t10k-images-idx3-ubyte  (or .gz)
 *   t10k-labels-idx1-ubyte  (or .gz)
 */
class MNIST : public MapDataset {
public:
    /**
     * @brief Construct MNIST dataset
     * @param root_dir Directory containing MNIST binary files
     * @param train If true, load training set (60K); else test set (10K)
     * @param normalize If true, scale pixel values to [0, 1]
     */
    MNIST(const std::string& root_dir, bool train = true, bool normalize = true);

    auto size() const -> size_t override;
    auto get(size_t index) -> std::pair<Tensor, Tensor> override;

private:
    Tensor images_;   ///< [N, 1, 28, 28] float32
    Tensor labels_;   ///< [N] int64
    size_t num_samples_;
};

/**
 * @brief Fashion-MNIST dataset (same format as MNIST)
 *
 * Expected file layout is identical to MNIST.
 */
class FashionMNIST : public MNIST {
public:
    FashionMNIST(const std::string& root_dir, bool train = true, bool normalize = true)
        : MNIST(root_dir, train, normalize) {}
};

} // namespace tenzor::data::datasets
