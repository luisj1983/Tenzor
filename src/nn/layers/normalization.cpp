#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

// ============================================================================
// LayerNorm Implementation
// ============================================================================

// LayerNorm autograd function
class LayerNormBackward : public Function {
public:
    LayerNormBackward(bool elementwise_affine, double eps,
                     int64_t normalized_size, std::vector<Tensor> tensors_to_save)
        : elementwise_affine_(elementwise_affine), eps_(eps),
          normalized_size_(normalized_size) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("LayerNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_orig = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input_orig = saved[0];
        auto& mean_orig = saved[1];
        auto& rstd_orig = saved[2];  // reciprocal std (1 / sqrt(var + eps))
        auto& weight_orig = saved[3];

        // Save original device before transferring to CPU
        Device original_device = input_orig.device();

        // Make all tensors contiguous AND transfer to CPU for pointer-based access
        auto grad_output = (grad_output_orig.device() == Device::cpu())
                          ? grad_output_orig.contiguous()
                          : grad_output_orig.contiguous().to(Device::cpu());
        auto input = (input_orig.device() == Device::cpu())
                    ? input_orig.contiguous()
                    : input_orig.contiguous().to(Device::cpu());
        auto mean = (mean_orig.device() == Device::cpu())
                   ? mean_orig.contiguous()
                   : mean_orig.contiguous().to(Device::cpu());
        auto rstd = (rstd_orig.device() == Device::cpu())
                   ? rstd_orig.contiguous()
                   : rstd_orig.contiguous().to(Device::cpu());
        auto weight = (weight_orig.device() == Device::cpu())
                     ? weight_orig.contiguous()
                     : weight_orig.contiguous().to(Device::cpu());

        auto shape = input.shape();
        int64_t batch_size = 1;
        for (size_t i = 0; i < shape.size() - 1; i++) {
            batch_size *= shape[i];
        }

        int64_t N = normalized_size_;

        // Compute normalized input: (x - mean) * rstd
        auto* input_data = input.data<float>();
        auto* mean_data = mean.data<float>();
        auto* rstd_data = rstd.data<float>();
        auto* grad_out_data = grad_output.data<float>();
        auto* weight_data = weight.data<float>();

        // Allocate gradient tensors on same device as input
        // Use the contiguous tensor's device to ensure consistency
        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({N}, grad_output.dtype(), grad_output.device());
        auto grad_bias = zeros({N}, grad_output.dtype(), grad_output.device());

        auto* grad_in_data = grad_input.data<float>();
        auto* grad_weight_data = grad_weight.data<float>();
        auto* grad_bias_data = grad_bias.data<float>();

        // Compute gradients for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            float mu = mean_data[b];
            float inv_std = rstd_data[b];

            // Compute intermediate sums for this batch element
            float sum_grad_out = 0.0f;
            float sum_grad_out_normalized = 0.0f;

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float x_normalized = (input_data[idx] - mu) * inv_std;
                float grad_out = grad_out_data[idx] * weight_data[i];

                sum_grad_out += grad_out;
                sum_grad_out_normalized += grad_out * x_normalized;

                // Accumulate weight and bias gradients
                grad_weight_data[i] += grad_out_data[idx] * x_normalized;
                grad_bias_data[i] += grad_out_data[idx];
            }

            // Compute input gradients using the formula:
            // grad_input = (grad_out - mean(grad_out) - normalized * mean(grad_out * normalized)) * rstd * weight
            float mean_grad_out = sum_grad_out / N;
            float mean_grad_out_normalized = sum_grad_out_normalized / N;

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float x_normalized = (input_data[idx] - mu) * inv_std;
                float grad_out = grad_out_data[idx] * weight_data[i];

                grad_in_data[idx] = (grad_out - mean_grad_out -
                                    x_normalized * mean_grad_out_normalized) * inv_std;
            }
        }

        // Transfer gradients back to original device if needed
        Tensor grad_input_final = (original_device == Device::cpu())
                                 ? grad_input.contiguous()
                                 : grad_input.contiguous().to(original_device);
        Tensor grad_weight_final = (original_device == Device::cpu())
                                  ? grad_weight.contiguous()
                                  : grad_weight.contiguous().to(original_device);
        Tensor grad_bias_final = (original_device == Device::cpu())
                                ? grad_bias.contiguous()
                                : grad_bias.contiguous().to(original_device);

        return {grad_input_final, grad_weight_final, grad_bias_final};
    }

private:
    bool elementwise_affine_;
    double eps_;
    int64_t normalized_size_;
};

LayerNorm::LayerNorm(std::vector<int64_t> normalized_shape,
                     double eps,
                     bool elementwise_affine)
    : normalized_shape_(normalized_shape), eps_(eps),
      elementwise_affine_(elementwise_affine) {

    // Compute total number of features to normalize
    num_features_ = 1;
    for (auto dim : normalized_shape_) {
        num_features_ *= dim;
    }

    if (elementwise_affine) {
        weight_ = Variable(ones(normalized_shape_), true);
        bias_ = Variable(zeros(normalized_shape_), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones(normalized_shape_), false);
        bias_ = Variable(zeros(normalized_shape_), false);
    }

    reset_parameters();
}

auto LayerNorm::forward(const Variable& input) -> Variable {
    auto shape = input.shape();

    // Verify that input shape matches normalized_shape at the end
    if (shape.size() < normalized_shape_.size()) {
        throw std::runtime_error("Input dimensions must be >= normalized_shape dimensions");
    }

    size_t norm_start = shape.size() - normalized_shape_.size();
    for (size_t i = 0; i < normalized_shape_.size(); i++) {
        if (shape[norm_start + i] != normalized_shape_[i]) {
            throw std::runtime_error("Input shape doesn't match normalized_shape");
        }
    }

    // Calculate batch size (all dimensions before normalized dimensions)
    int64_t batch_size = 1;
    for (size_t i = 0; i < norm_start; i++) {
        batch_size *= shape[i];
    }

    int64_t N = num_features_;

    // Save original device and move input to CPU for pointer-based computation
    Device original_device = input.tensor().device();
    Tensor input_cpu = (original_device == Device::cpu()) ? input.tensor() : input.tensor().to(Device::cpu());

    // Move weight and bias to CPU if needed
    Tensor weight_cpu = (elementwise_affine_ && weight_.tensor().device() != Device::cpu())
                        ? weight_.tensor().to(Device::cpu()) : weight_.tensor();
    Tensor bias_cpu = (elementwise_affine_ && bias_.tensor().device() != Device::cpu())
                      ? bias_.tensor().to(Device::cpu()) : bias_.tensor();

    // Compute mean and variance for each batch element on CPU
    // Using manual computation to ensure correct memory layout
    auto batch_mean = zeros({batch_size}, DType::Float32, Device::cpu());
    auto batch_var = zeros({batch_size}, DType::Float32, Device::cpu());

    auto* input_data = input_cpu.data<float>();
    auto* mean_data = batch_mean.data<float>();
    auto* var_data = batch_var.data<float>();

    // Compute mean for each batch element
    for (int64_t b = 0; b < batch_size; b++) {
        double sum = 0.0;
        for (int64_t i = 0; i < N; i++) {
            sum += input_data[b * N + i];
        }
        mean_data[b] = static_cast<float>(sum / N);
    }

    // Compute variance for each batch element
    for (int64_t b = 0; b < batch_size; b++) {
        double sum_sq = 0.0;
        float mu = mean_data[b];
        for (int64_t i = 0; i < N; i++) {
            float diff = input_data[b * N + i] - mu;
            sum_sq += diff * diff;
        }
        var_data[b] = static_cast<float>(sum_sq / N);
    }

    // Compute reciprocal std (1 / sqrt(var + eps))
    auto rstd = zeros({batch_size}, DType::Float32, Device::cpu());
    auto* rstd_data = rstd.data<float>();
    for (int64_t b = 0; b < batch_size; b++) {
        rstd_data[b] = 1.0f / std::sqrt(var_data[b] + static_cast<float>(eps_));
    }

    // Normalize: (x - mean) * rstd on CPU
    auto output_cpu = zeros_like(input_cpu);
    auto* output_data = output_cpu.data<float>();
    auto* weight_data = weight_cpu.data<float>();
    auto* bias_data = bias_cpu.data<float>();

    for (int64_t b = 0; b < batch_size; b++) {
        float mu = mean_data[b];
        float inv_std = rstd_data[b];

        for (int64_t i = 0; i < N; i++) {
            int64_t idx = b * N + i;
            float normalized = (input_data[idx] - mu) * inv_std;

            if (elementwise_affine_) {
                output_data[idx] = normalized * weight_data[i] + bias_data[i];
            } else {
                output_data[idx] = normalized;
            }
        }
    }

    // Move output back to original device if needed
    Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

    // Set up autograd if needed
    if (input.requires_grad() || (elementwise_affine_ && weight_.requires_grad())) {
        auto result = Variable(output, true);

        // Prepare tensors to save for backward
        std::vector<Tensor> tensors_to_save = {
            input.tensor(),
            batch_mean,
            rstd,
            elementwise_affine_ ? weight_.tensor() : ones({N})
        };

        auto grad_fn = std::make_shared<LayerNormBackward>(
            elementwise_affine_, eps_, N, std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        // Set next functions to chain backward pass
        std::vector<std::shared_ptr<Function>> next_funcs;
        // Only add grad_fn if it exists (not null)
        if (auto input_grad_fn = input.grad_fn()) {
            next_funcs.push_back(input_grad_fn);
        }
        if (elementwise_affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                if (auto weight_grad_fn = weight_it->second->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                if (auto bias_grad_fn = bias_it->second->grad_fn()) {
                    next_funcs.push_back(bias_grad_fn);
                }
            }
        }
        grad_fn->set_next_functions(next_funcs);

        // Track input variables
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (elementwise_affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                input_vars.push_back(*weight_it->second);
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                input_vars.push_back(*bias_it->second);
            }
        }
        grad_fn->set_input_variables(input_vars);

        return result;
    } else {
        return Variable(output, false);
    }
}

auto LayerNorm::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

// ============================================================================
// GroupNorm Implementation
// ============================================================================

// GroupNorm autograd function
class GroupNormBackward : public Function {
public:
    GroupNormBackward(bool affine, double eps, int64_t num_groups,
                     int64_t num_channels, int64_t group_size,
                     std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), num_groups_(num_groups),
          num_channels_(num_channels), group_size_(group_size) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("GroupNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output_orig = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input_orig = saved[0];
        auto& mean_orig = saved[1];
        auto& rstd_orig = saved[2];
        auto& weight_orig = saved[3];

        // Make all tensors contiguous for pointer-based access
        auto grad_output = grad_output_orig.contiguous();
        auto input = input_orig.contiguous();
        auto mean = mean_orig.contiguous();
        auto rstd = rstd_orig.contiguous();
        auto weight = weight_orig.contiguous();

        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t spatial_size = H * W;

        auto* input_data = input.data<float>();
        auto* mean_data = mean.data<float>();
        auto* rstd_data = rstd.data<float>();
        auto* grad_out_data = grad_output.data<float>();
        auto* weight_data = weight.data<float>();

        auto grad_input = zeros_like(input);
        auto grad_weight = zeros({C}, grad_output.dtype(), grad_output.device());
        auto grad_bias = zeros({C}, grad_output.dtype(), grad_output.device());

        auto* grad_in_data = grad_input.data<float>();
        auto* grad_weight_data = grad_weight.data<float>();
        auto* grad_bias_data = grad_bias.data<float>();

        // Process each batch and group
        for (int64_t n = 0; n < N; n++) {
            for (int64_t g = 0; g < num_groups_; g++) {
                int64_t group_idx = n * num_groups_ + g;
                float mu = mean_data[group_idx];
                float inv_std = rstd_data[group_idx];

                int64_t c_start = g * group_size_;
                int64_t c_end = c_start + group_size_;

                // Compute intermediate sums for this group
                float sum_grad_out = 0.0f;
                float sum_grad_out_normalized = 0.0f;
                int64_t group_numel = group_size_ * spatial_size;

                for (int64_t c = c_start; c < c_end; c++) {
                    for (int64_t h = 0; h < H; h++) {
                        for (int64_t w = 0; w < W; w++) {
                            int64_t idx = ((n * C + c) * H + h) * W + w;
                            float x_normalized = (input_data[idx] - mu) * inv_std;
                            float grad_out = grad_out_data[idx] * weight_data[c];

                            sum_grad_out += grad_out;
                            sum_grad_out_normalized += grad_out * x_normalized;

                            // Accumulate weight and bias gradients
                            grad_weight_data[c] += grad_out_data[idx] * x_normalized;
                            grad_bias_data[c] += grad_out_data[idx];
                        }
                    }
                }

                // Compute input gradients
                float mean_grad_out = sum_grad_out / group_numel;
                float mean_grad_out_normalized = sum_grad_out_normalized / group_numel;

                for (int64_t c = c_start; c < c_end; c++) {
                    for (int64_t h = 0; h < H; h++) {
                        for (int64_t w = 0; w < W; w++) {
                            int64_t idx = ((n * C + c) * H + h) * W + w;
                            float x_normalized = (input_data[idx] - mu) * inv_std;
                            float grad_out = grad_out_data[idx] * weight_data[c];

                            grad_in_data[idx] = (grad_out - mean_grad_out -
                                                x_normalized * mean_grad_out_normalized) * inv_std;
                        }
                    }
                }
            }
        }

        return {grad_input.contiguous(), grad_weight.contiguous(), grad_bias.contiguous()};
    }

private:
    bool affine_;
    double eps_;
    int64_t num_groups_;
    int64_t num_channels_;
    int64_t group_size_;
};

GroupNorm::GroupNorm(int64_t num_groups,
                     int64_t num_channels,
                     double eps,
                     bool affine)
    : num_groups_(num_groups), num_channels_(num_channels),
      eps_(eps), affine_(affine) {

    if (num_channels_ % num_groups_ != 0) {
        throw std::runtime_error("num_channels must be divisible by num_groups");
    }

    if (affine) {
        weight_ = Variable(ones({num_channels_}), true);
        bias_ = Variable(zeros({num_channels_}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_channels_}), false);
        bias_ = Variable(zeros({num_channels_}), false);
    }

    reset_parameters();
}

auto GroupNorm::forward(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("GroupNorm expects 4D input (got " +
                               std::to_string(shape.size()) + "D)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t spatial_size = H * W;

    if (C != num_channels_) {
        throw std::runtime_error("Expected " + std::to_string(num_channels_) +
                               " channels, got " + std::to_string(C));
    }

    int64_t group_size = num_channels_ / num_groups_;

    // Compute mean and variance for each group
    // Shape: [N, num_groups]
    auto group_mean = zeros({N * num_groups_});
    auto group_var = zeros({N * num_groups_});

    auto* input_data = input.tensor().data<float>();
    auto* mean_data = group_mean.data<float>();
    auto* var_data = group_var.data<float>();

    int64_t group_numel = group_size * spatial_size;

    // Compute mean for each group (using manual NCHW indexing)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups_; g++) {
            double sum = 0.0;
            int64_t c_start = g * group_size;
            int64_t c_end = c_start + group_size;

            for (int64_t c = c_start; c < c_end; c++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += input_data[idx];
                    }
                }
            }

            mean_data[n * num_groups_ + g] = static_cast<float>(sum / group_numel);
        }
    }

    // Compute variance for each group
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups_; g++) {
            double sum_sq = 0.0;
            float mu = mean_data[n * num_groups_ + g];
            int64_t c_start = g * group_size;
            int64_t c_end = c_start + group_size;

            for (int64_t c = c_start; c < c_end; c++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        float diff = input_data[idx] - mu;
                        sum_sq += diff * diff;
                    }
                }
            }

            var_data[n * num_groups_ + g] = static_cast<float>(sum_sq / group_numel);
        }
    }

    // Compute reciprocal std
    auto rstd = zeros({N * num_groups_});
    auto* rstd_data = rstd.data<float>();
    for (int64_t i = 0; i < N * num_groups_; i++) {
        rstd_data[i] = 1.0f / std::sqrt(var_data[i] + static_cast<float>(eps_));
    }

    // Normalize and apply affine transformation
    auto output = zeros_like(input.tensor());
    auto* output_data = output.data<float>();
    auto* weight_data = weight_.tensor().data<float>();
    auto* bias_data = bias_.tensor().data<float>();

    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups_; g++) {
            float mu = mean_data[n * num_groups_ + g];
            float inv_std = rstd_data[n * num_groups_ + g];
            int64_t c_start = g * group_size;
            int64_t c_end = c_start + group_size;

            for (int64_t c = c_start; c < c_end; c++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        float normalized = (input_data[idx] - mu) * inv_std;

                        if (affine_) {
                            output_data[idx] = normalized * weight_data[c] + bias_data[c];
                        } else {
                            output_data[idx] = normalized;
                        }
                    }
                }
            }
        }
    }

    // Set up autograd if needed
    if (input.requires_grad() || (affine_ && weight_.requires_grad())) {
        auto result = Variable(output, true);

        std::vector<Tensor> tensors_to_save = {
            input.tensor(),
            group_mean,
            rstd,
            affine_ ? weight_.tensor() : ones({C})
        };

        auto grad_fn = std::make_shared<GroupNormBackward>(
            affine_, eps_, num_groups_, num_channels_, group_size,
            std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        // Set next functions to chain backward pass
        std::vector<std::shared_ptr<Function>> next_funcs;
        // Only add grad_fn if it exists (not null)
        if (auto input_grad_fn = input.grad_fn()) {
            next_funcs.push_back(input_grad_fn);
        }
        if (affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                if (auto weight_grad_fn = weight_it->second->grad_fn()) {
                    next_funcs.push_back(weight_grad_fn);
                }
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                if (auto bias_grad_fn = bias_it->second->grad_fn()) {
                    next_funcs.push_back(bias_grad_fn);
                }
            }
        }
        grad_fn->set_next_functions(next_funcs);

        // Track input variables
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (affine_) {
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second->requires_grad()) {
                input_vars.push_back(*weight_it->second);
            }
            if (bias_it != parameters_.end() && bias_it->second->requires_grad()) {
                input_vars.push_back(*bias_it->second);
            }
        }
        grad_fn->set_input_variables(input_vars);

        return result;
    } else {
        return Variable(output, false);
    }
}

auto GroupNorm::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

} // namespace tenzor::nn
