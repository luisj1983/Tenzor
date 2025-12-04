#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

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

        // Debug for Float64
        if (grad_output_orig.dtype() == DType::Float64) {
            std::cerr << "[LAYERNORM_BACKWARD_F64] grad_output shape: [";
            for (size_t i = 0; i < grad_output_orig.shape().size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << grad_output_orig.shape()[i];
            }
            std::cerr << "], input shape: [";
            for (size_t i = 0; i < input_orig.shape().size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << input_orig.shape()[i];
            }
            std::cerr << "], normalized_size=" << normalized_size_ << std::endl;
        }

        // Save original device before transferring to CPU
        Device original_device = input_orig.device();

        // Make all tensors contiguous AND transfer to CPU for pointer-based access
        // Save original dtype for conversion back
        DType original_dtype = grad_output_orig.dtype();

        // Move to CPU and convert to Float32 for computation
        auto grad_output = (grad_output_orig.device() == Device::cpu())
                          ? grad_output_orig.contiguous().to(DType::Float32)
                          : grad_output_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto input = (input_orig.device() == Device::cpu())
                    ? input_orig.contiguous().to(DType::Float32)
                    : input_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto mean = (mean_orig.device() == Device::cpu())
                   ? mean_orig.contiguous().to(DType::Float32)
                   : mean_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto rstd = (rstd_orig.device() == Device::cpu())
                   ? rstd_orig.contiguous().to(DType::Float32)
                   : rstd_orig.contiguous().to(Device::cpu()).to(DType::Float32);
        auto weight = (weight_orig.device() == Device::cpu())
                     ? weight_orig.contiguous().to(DType::Float32)
                     : weight_orig.contiguous().to(Device::cpu()).to(DType::Float32);

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

        // Debug: Check input values for Float64
        if (original_dtype == DType::Float64 && batch_size > 0) {
            std::cerr << "[LAYERNORM_BACKWARD_F64] batch_size=" << batch_size << ", N=" << N << std::endl;
            std::cerr << "[LAYERNORM_BACKWARD_F64] mean[0]=" << mean_data[0]
                      << ", rstd[0]=" << rstd_data[0] << std::endl;
            std::cerr << "[LAYERNORM_BACKWARD_F64] input first5=[";
            for (int i = 0; i < std::min(5, static_cast<int>(N)); i++) {
                if (i > 0) std::cerr << ",";
                std::cerr << input_data[i];
            }
            std::cerr << "]" << std::endl;
            std::cerr << "[LAYERNORM_BACKWARD_F64] grad_out first5=[";
            for (int i = 0; i < std::min(5, static_cast<int>(N)); i++) {
                if (i > 0) std::cerr << ",";
                std::cerr << grad_out_data[i];
            }
            std::cerr << "]" << std::endl;
            std::cerr << "[LAYERNORM_BACKWARD_F64] weight first5=[";
            for (int i = 0; i < std::min(5, static_cast<int>(N)); i++) {
                if (i > 0) std::cerr << ",";
                std::cerr << weight_data[i];
            }
            std::cerr << "]" << std::endl;
        }

        // Compute gradients for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            float mu = mean_data[b];
            float inv_std = rstd_data[b];

            // Check if variance is essentially zero (which causes numerical instability)
            // When all inputs are identical (variance ~0), gradients should be zero
            // because small changes to input don't change the normalized output
            bool zero_variance = (inv_std > 100.0f);  // inv_std = 1/sqrt(var+eps), large value means var≈0

            if (zero_variance) {
                // With zero variance, input gradients should be zero
                for (int64_t i = 0; i < N; i++) {
                    grad_in_data[b * N + i] = 0.0f;
                }
                // Weight and bias gradients are still computed normally from output gradients
                for (int64_t i = 0; i < N; i++) {
                    int64_t idx = b * N + i;
                    float x_normalized = 0.0f;  // (input - mean) * large_inv_std, but input≈mean
                    grad_weight_data[i] += grad_out_data[idx] * x_normalized;  // Will be ~0
                    grad_bias_data[i] += grad_out_data[idx];
                }
                continue;
            }

            // Normal case: compute gradients with stable variance
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

                grad_in_data[idx] = (grad_out_data[idx] * weight_data[i] - mean_grad_out -
                                    x_normalized * mean_grad_out_normalized) * inv_std;
            }
        }

        // Debug: Check if grad_input has NaN before final conversion
        if (original_dtype == DType::Float64) {
            bool has_nan = false;
            for (int64_t i = 0; i < std::min(static_cast<int64_t>(100), grad_input.numel()); i++) {
                if (std::isnan(grad_in_data[i])) {
                    has_nan = true;
                    break;
                }
            }
            std::cerr << "[LAYERNORM_BACKWARD_F64] grad_input (Float32) has_nan=" << has_nan
                      << ", first5=[";
            for (int i = 0; i < std::min(5, static_cast<int>(grad_input.numel())); i++) {
                if (i > 0) std::cerr << ",";
                std::cerr << grad_in_data[i];
            }
            std::cerr << "]" << std::endl;
        }

        // Transfer gradients back to original device and dtype if needed
        Tensor grad_input_final = (original_device == Device::cpu())
                                 ? grad_input.contiguous().to(original_dtype)
                                 : grad_input.contiguous().to(original_device).to(original_dtype);
        Tensor grad_weight_final = (original_device == Device::cpu())
                                  ? grad_weight.contiguous().to(original_dtype)
                                  : grad_weight.contiguous().to(original_device).to(original_dtype);
        Tensor grad_bias_final = (original_device == Device::cpu())
                                ? grad_bias.contiguous().to(original_dtype)
                                : grad_bias.contiguous().to(original_device).to(original_dtype);

        // Debug: Check if grad_input_final has NaN after conversion back to Float64
        if (original_dtype == DType::Float64) {
            auto grad_check = grad_input_final.to(Device::cpu());
            auto* data = grad_check.data<double>();
            bool has_nan = false;
            for (int64_t i = 0; i < std::min(static_cast<int64_t>(100), grad_check.numel()); i++) {
                if (std::isnan(data[i])) {
                    has_nan = true;
                    break;
                }
            }
            std::cerr << "[LAYERNORM_BACKWARD_F64] grad_input_final (Float64) has_nan=" << has_nan
                      << ", first5=[";
            for (int i = 0; i < std::min(5, static_cast<int>(grad_check.numel())); i++) {
                if (i > 0) std::cerr << ",";
                std::cerr << data[i];
            }
            std::cerr << "]" << std::endl;
        }

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

auto LayerNorm::forward_impl(const Variable& input) -> Variable {
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

    // Get weight/bias from parameters_ to respect offload hooks
    // Note: hooks modify parameters_["weight"]->tensor(), not the member weight_
    Tensor weight_cpu = elementwise_affine_ ? parameters_["weight"]->tensor() : weight_.tensor();
    Tensor bias_cpu = elementwise_affine_ ? parameters_["bias"]->tensor() : bias_.tensor();
    if (elementwise_affine_) {
        if (weight_cpu.device() != Device::cpu()) {
            weight_cpu = weight_cpu.to(Device::cpu());
        }
        if (weight_cpu.dtype() != input_cpu.dtype()) {
            weight_cpu = weight_cpu.to(input_cpu.dtype());
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        if (bias_cpu.dtype() != input_cpu.dtype()) {
            bias_cpu = bias_cpu.to(input_cpu.dtype());
        }
    }

    // Compute mean and variance for each batch element on CPU
    // Using manual computation to ensure correct memory layout
    auto batch_mean = zeros({batch_size}, DType::Float32, Device::cpu());
    auto batch_var = zeros({batch_size}, DType::Float32, Device::cpu());

    auto* mean_data = batch_mean.data<float>();
    auto* var_data = batch_var.data<float>();

    // Dtype-aware computation
    DType input_dtype = input_cpu.dtype();

    if (input_dtype == DType::Float16) {
        auto* input_data = input_cpu.data<Float16>();

        // Compute mean for each batch element (use float accumulation)
        for (int64_t b = 0; b < batch_size; b++) {
            double sum = 0.0;
            for (int64_t i = 0; i < N; i++) {
                sum += static_cast<float>(input_data[b * N + i]);
            }
            mean_data[b] = static_cast<float>(sum / N);
        }

        // Compute variance for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            float mu = mean_data[b];
            for (int64_t i = 0; i < N; i++) {
                float diff = static_cast<float>(input_data[b * N + i]) - mu;
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
        auto* output_data = output_cpu.data<Float16>();
        auto* weight_data = elementwise_affine_ ? weight_cpu.data<Float16>() : nullptr;
        auto* bias_data = elementwise_affine_ ? bias_cpu.data<Float16>() : nullptr;

        for (int64_t b = 0; b < batch_size; b++) {
            float mu = mean_data[b];
            float inv_std = rstd_data[b];

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                float normalized = (static_cast<float>(input_data[idx]) - mu) * inv_std;

                if (elementwise_affine_) {
                    output_data[idx] = Float16(normalized * static_cast<float>(weight_data[i]) + static_cast<float>(bias_data[i]));
                } else {
                    output_data[idx] = Float16(normalized);
                }
            }
        }

        // Move output back to original device if needed
        Tensor output = (original_device == Device::cpu()) ? output_cpu : output_cpu.to(original_device);

        // Set up autograd if needed
        if (input.requires_grad() || (elementwise_affine_ && parameters_["weight"]->requires_grad())) {
            auto result = Variable(output, true);

            // Prepare tensors to save for backward
            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                batch_mean,
                rstd,
                elementwise_affine_ ? parameters_["weight"]->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
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
            if (elementwise_affine_ && parameters_["weight"]->grad_fn()) {
                next_funcs.push_back(parameters_["weight"]->grad_fn());
            }

            grad_fn->set_next_functions(std::move(next_funcs));
            return result;
        }

        return Variable(output, false);

    } else if (input_dtype == DType::Float32) {
        auto* input_data = input_cpu.data<float>();

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
        if (input.requires_grad() || (elementwise_affine_ && parameters_["weight"]->requires_grad())) {
            auto result = Variable(output, true);

            // Prepare tensors to save for backward
            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                batch_mean,
                rstd,
                elementwise_affine_ ? parameters_["weight"]->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
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

            // Removed set_inputs call - method doesn't exist
            return result;
        }

        return Variable(output, false);

    } else if (input_dtype == DType::Float64) {
        auto* input_data = input_cpu.data<double>();

        // Use double precision for mean and variance computation
        auto batch_mean_f64 = zeros({batch_size}, DType::Float64, Device::cpu());
        auto batch_var_f64 = zeros({batch_size}, DType::Float64, Device::cpu());
        auto* mean_data_f64 = batch_mean_f64.data<double>();
        auto* var_data_f64 = batch_var_f64.data<double>();

        // Compute mean for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            double sum = 0.0;
            for (int64_t i = 0; i < N; i++) {
                sum += input_data[b * N + i];
            }
            mean_data_f64[b] = sum / N;
        }

        // Compute variance for each batch element
        for (int64_t b = 0; b < batch_size; b++) {
            double sum_sq = 0.0;
            double mu = mean_data_f64[b];
            for (int64_t i = 0; i < N; i++) {
                double diff = input_data[b * N + i] - mu;
                sum_sq += diff * diff;
            }
            var_data_f64[b] = sum_sq / N;
        }

        // Compute reciprocal std (1 / sqrt(var + eps))
        auto rstd_f64 = zeros({batch_size}, DType::Float64, Device::cpu());
        auto* rstd_data_f64 = rstd_f64.data<double>();
        for (int64_t b = 0; b < batch_size; b++) {
            rstd_data_f64[b] = 1.0 / std::sqrt(var_data_f64[b] + eps_);
        }

        // Normalize: (x - mean) * rstd on CPU
        auto output_cpu = zeros_like(input_cpu);
        auto* output_data = output_cpu.data<double>();
        auto* weight_data = elementwise_affine_ ? weight_cpu.data<double>() : nullptr;
        auto* bias_data = elementwise_affine_ ? bias_cpu.data<double>() : nullptr;

        for (int64_t b = 0; b < batch_size; b++) {
            double mu = mean_data_f64[b];
            double inv_std = rstd_data_f64[b];

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = b * N + i;
                double normalized = (input_data[idx] - mu) * inv_std;

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
        if (input.requires_grad() || (elementwise_affine_ && parameters_["weight"]->requires_grad())) {
            auto result = Variable(output, true);

            // Convert mean and rstd to Float32 for backward compatibility
            auto batch_mean_f32 = zeros({batch_size}, DType::Float32, Device::cpu());
            auto rstd_f32 = zeros({batch_size}, DType::Float32, Device::cpu());
            auto* mean_f32_data = batch_mean_f32.data<float>();
            auto* rstd_f32_data = rstd_f32.data<float>();

            for (int64_t b = 0; b < batch_size; b++) {
                mean_f32_data[b] = static_cast<float>(mean_data_f64[b]);
                rstd_f32_data[b] = static_cast<float>(rstd_data_f64[b]);
            }

            // Prepare tensors to save for backward
            std::vector<Tensor> tensors_to_save = {
                input.tensor(),
                batch_mean_f32,
                rstd_f32,
                elementwise_affine_ ? parameters_["weight"]->tensor() : ones({N}, input.tensor().dtype(), input.tensor().device())
            };

            auto grad_fn = std::make_shared<LayerNormBackward>(
                elementwise_affine_, eps_, N, std::move(tensors_to_save)
            );

            result.set_grad_fn(grad_fn);

            // Set next functions to chain backward pass
            std::vector<std::shared_ptr<Function>> next_funcs;
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

            return result;
        }

        return Variable(output, false);

    } else {
        throw std::runtime_error("LayerNorm only supports Float16, Float32, and Float64 dtypes");
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

auto GroupNorm::forward_impl(const Variable& input) -> Variable {
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

    // Save original device and move to CPU for computation
    auto original_device = input.tensor().device();
    Variable input_cpu = (original_device.type == Device::Type::CPU) ?
                         input : Variable(input.tensor().cpu(), input.requires_grad());
    // Get weight/bias from parameters_ to respect offload hooks
    Tensor weight_tensor = affine_ ? parameters_["weight"]->tensor() : weight_.tensor();
    Tensor bias_tensor = affine_ ? parameters_["bias"]->tensor() : bias_.tensor();
    Variable weight_cpu = (weight_tensor.device().type == Device::Type::CPU) ?
                          Variable(weight_tensor, parameters_["weight"]->requires_grad()) : Variable(weight_tensor.cpu(), parameters_["weight"]->requires_grad());
    Variable bias_cpu = (bias_tensor.device().type == Device::Type::CPU) ?
                        Variable(bias_tensor, parameters_["bias"]->requires_grad()) : Variable(bias_tensor.cpu(), parameters_["bias"]->requires_grad());

    // Compute mean and variance for each group
    // Shape: [N, num_groups]
    auto group_mean = zeros({N * num_groups_});
    auto group_var = zeros({N * num_groups_});

    auto* input_data = input_cpu.tensor().data<float>();
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
    auto output = zeros_like(input_cpu.tensor());
    auto* output_data = output.data<float>();
    auto* weight_data = weight_cpu.tensor().data<float>();
    auto* bias_data = bias_cpu.tensor().data<float>();

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

    // Move output back to original device if needed
    Tensor output_final = (original_device.type == Device::Type::CPU) ?
                          output : output.to(original_device);

    // Set up autograd if needed
    if (input.requires_grad() || (affine_ && parameters_["weight"]->requires_grad())) {
        auto result = Variable(output_final, true);

        std::vector<Tensor> tensors_to_save = {
            input.tensor(),
            group_mean,
            rstd,
            affine_ ? parameters_["weight"]->tensor() : ones({C}, input.tensor().dtype(), input.tensor().device())
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
        return Variable(output_final, false);
    }
}

auto GroupNorm::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
}

} // namespace tenzor::nn
