#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace tenzor::nn {

// BatchNorm2d autograd function
class BatchNorm2dBackward : public Function {
public:
    BatchNorm2dBackward(bool affine, double eps, std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps) {
        // Save tensors in constructor (protected member access is allowed here)
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // Not used - forward is handled by BatchNorm2d::forward
        throw std::runtime_error("BatchNorm2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Ensure grad_output is contiguous
        auto grad_output = grad_outputs[0].contiguous();
        auto saved = saved_tensors();
        // Ensure all saved tensors are contiguous
        auto input = saved[0].contiguous();
        auto mean = saved[1].contiguous();
        auto invstd = saved[2].contiguous();
        auto weight = saved[3].contiguous();

        // grad_output: [N, C, H, W]
        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t spatial_size = H * W;
        int64_t batch_size = N * spatial_size;

        // Compute normalized input for gradient computation
        auto mean_broadcast = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto invstd_broadcast = invstd.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto normalized = ((input - mean_broadcast) * invstd_broadcast).contiguous();

        // Gradient with respect to weight: sum(grad_output * normalized, dim=[0,2,3])
        // After first sum over dim 0: [C, spatial_size], then sum over dim 1 → [C]
        auto grad_weight = sum(sum((grad_output * normalized)
            .reshape({N, C, spatial_size}).contiguous(), 0, false), 1, false);

        // Gradient with respect to bias: sum(grad_output, dim=[0,2,3])
        // After first sum over dim 0: [C, spatial_size], then sum over dim 1 → [C]
        auto grad_bias = sum(sum(grad_output
            .reshape({N, C, spatial_size}).contiguous(), 0, false), 1, false);

        // Gradient with respect to normalized input
        auto weight_broadcast = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto grad_normalized = (grad_output * weight_broadcast).contiguous();

        // Gradient with respect to input (using efficient batch norm backward formulation)
        auto grad_input_normalized = grad_normalized.reshape({N, C, spatial_size}).contiguous();
        auto normalized_reshaped = normalized.reshape({N, C, spatial_size}).contiguous();

        // After first sum over dim 0: [1, C, spatial_size], then sum over dim 2 → [1, C, 1]
        auto sum_grad = sum(sum(grad_input_normalized, 0, true), 2, true).contiguous();
        auto sum_grad_x_norm = sum(sum((grad_input_normalized * normalized_reshaped),
                                0, true), 2, true).contiguous();

        auto invstd_expanded = invstd.unsqueeze(0).unsqueeze(-1).contiguous();

        // Break down complex expression to ensure contiguity
        auto term1 = (sum_grad / static_cast<float>(batch_size)).contiguous();
        auto term2 = (normalized_reshaped * sum_grad_x_norm / static_cast<float>(batch_size)).contiguous();
        auto grad_input = ((grad_input_normalized - term1 - term2) * invstd_expanded).contiguous();

        grad_input = grad_input.reshape({N, C, H, W}).contiguous();

        // Return gradients in the order of input_vars: [input, weight, bias]
        return {grad_input, grad_weight, grad_bias};
    }

private:
    bool affine_;
    double eps_;
};

BatchNorm2d::BatchNorm2d(int64_t num_features, double eps, double momentum,
                        bool affine, bool track_running_stats)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      affine_(affine), track_running_stats_(track_running_stats) {

    if (affine) {
        weight_ = Variable(ones({num_features}), true);
        bias_ = Variable(zeros({num_features}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
    }

    reset_parameters();
}

auto BatchNorm2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H, W]
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("BatchNorm2d expects 4D input (got " +
                               std::to_string(shape.size()) + "D)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t spatial_size = H * W;

    if (C != num_features_) {
        throw std::runtime_error("Expected " + std::to_string(num_features_) +
                               " channels, got " + std::to_string(C));
    }

    // Validate to prevent division by zero
    int64_t batch_size = N * spatial_size;
    if (training_ && batch_size == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute statistics for empty batch (N * H * W = 0)");
    }

    // Track original device for final output
    Device original_device = input.tensor().device();
    bool use_gpu = (original_device.type == Device::Type::CUDA);

    // Keep data on original device throughout (no CPU fallbacks)
    Tensor input_work = input.tensor();

    Tensor batch_mean, batch_var;

    if (training_) {
        // Training mode: compute batch statistics using backend dispatch
        OpAttributes mean_var_attrs;
        std::vector<Tensor> mean_var_inputs = {input_work};
        std::vector<Tensor> mean_var_results = dispatch(OpId::BatchNorm2dMeanVar, mean_var_inputs, mean_var_attrs);
        batch_mean = mean_var_results[0];
        batch_var = mean_var_results[1];

        // Update running statistics using CUDA kernel (no CPU fallback)
        if (track_running_stats_) {
            // Use unbiased variance estimate for running statistics
            int64_t batch_size = N * spatial_size;
            auto unbiased_var = batch_var * (static_cast<float>(batch_size) /
                                            static_cast<float>(batch_size - 1));

            // Get running stats (stay on original device)
            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            // Use CUDA kernel for running stats update
            OpAttributes update_attrs;
            std::ostringstream momentum_ss;
            momentum_ss << std::scientific << std::setprecision(9) << static_cast<float>(momentum_);
            update_attrs["momentum"] = momentum_ss.str();
            std::vector<Tensor> update_inputs = {rm_var_ptr->tensor(), rv_var_ptr->tensor(), batch_mean, unbiased_var};
            std::vector<Tensor> updated_stats = dispatch(OpId::BatchNorm2dUpdateRunningStats, update_inputs, update_attrs);

            rm_var_ptr->tensor() = updated_stats[0];
            rv_var_ptr->tensor() = updated_stats[1];

            num_batches_tracked_++;
        }
    } else {
        // Inference mode: use running statistics (stay on original device)
        if (track_running_stats_) {
            batch_mean = buffers_["running_mean"]->tensor();
            batch_var = buffers_["running_var"]->tensor();
        } else {
            throw std::runtime_error("BatchNorm2d in eval mode requires track_running_stats=true");
        }
    }

    // Normalize using CUDA kernel (no CPU operations)
    Tensor output;
    OpAttributes forward_attrs;
    std::ostringstream epsilon_ss;
    epsilon_ss << std::scientific << std::setprecision(9) << static_cast<float>(eps_);
    forward_attrs["epsilon"] = epsilon_ss.str();

    if (affine_) {
        // Use affine forward kernel: output = gamma * (x - mean) / sqrt(var + eps) + beta
        auto& weight_ptr = parameters_["weight"];
        auto& bias_ptr = parameters_["bias"];

        std::vector<Tensor> forward_inputs = {input_work, batch_mean, batch_var, weight_ptr->tensor(), bias_ptr->tensor()};
        std::vector<Tensor> forward_results = dispatch(OpId::BatchNorm2dForwardAffine, forward_inputs, forward_attrs);
        output = forward_results[0];
    } else {
        // Use non-affine forward kernel: output = (x - mean) / sqrt(var + eps)
        std::vector<Tensor> forward_inputs = {input_work, batch_mean, batch_var};
        std::vector<Tensor> forward_results = dispatch(OpId::BatchNorm2dForward, forward_inputs, forward_attrs);
        output = forward_results[0];
    }

    // Set up autograd if needed
    bool requires_grad = input.requires_grad();
    if (affine_ && parameters_.find("weight") != parameters_.end()) {
        requires_grad = requires_grad || parameters_["weight"]->requires_grad();
    }
    if (requires_grad) {
        // Create result variable from output
        auto result = Variable(output, true);

        // Prepare tensors to save for backward (already on correct device, no transfers needed)
        // Compute invstd from variance for backward pass
        // Create epsilon tensor with same dtype as batch_var to avoid dtype mismatch
        auto eps_tensor = full({}, eps_, batch_var.dtype(), batch_var.device());
        auto invstd = pow(batch_var + eps_tensor, -0.5f);

        // Ensure contiguous for backward
        Tensor batch_mean_final = batch_mean.contiguous();
        Tensor invstd_final = invstd.contiguous();

        // CRITICAL: Access weight from parameters_ map
        Tensor weight_tensor = affine_ ? parameters_["weight"]->tensor() : ones({C}, input.tensor().dtype(), original_device);
        // Ensure all tensors are contiguous before saving
        std::vector<Tensor> tensors_to_save = {
            input.tensor().contiguous(),  // input (original device, made contiguous)
            batch_mean_final,             // mean (transferred to original device, already contiguous)
            invstd_final,                 // invstd (transferred to original device, already contiguous)
            weight_tensor.contiguous()    // weight (or ones on original device, made contiguous)
        };

        // Create backward function with saved tensors
        auto grad_fn = std::make_shared<BatchNorm2dBackward>(
            affine_, eps_, std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        // Track input variables for gradient accumulation
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        if (affine_) {
            // Use pointers to the Variables in the parameters_ map
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

        // CRITICAL FIX: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        return result;
    } else {
        return Variable(output, false);
    }
}

auto BatchNorm2d::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
    if (track_running_stats_) {
        // CRITICAL: Access from buffers_ map
        buffers_["running_mean"]->tensor().zero_();
        buffers_["running_var"]->tensor().fill_(1.0f);
        num_batches_tracked_ = 0;
    }
}

// ============================================================================
// BatchNorm1d Implementation
// ============================================================================

// BatchNorm1d autograd function
class BatchNorm1dBackward : public Function {
public:
    BatchNorm1dBackward(bool affine, double eps, std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("BatchNorm1dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto grad_output = grad_outputs[0].contiguous();
        auto saved = saved_tensors();
        auto input = saved[0].contiguous();
        auto mean = saved[1].contiguous();
        auto invstd = saved[2].contiguous();
        auto weight = saved[3].contiguous();

        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t L = shape.size() == 3 ? shape[2] : 1;
        int64_t batch_size = N * L;

        // Compute normalized input
        Tensor mean_broadcast, invstd_broadcast;
        if (shape.size() == 3) {
            mean_broadcast = mean.unsqueeze(0).unsqueeze(2).contiguous();
            invstd_broadcast = invstd.unsqueeze(0).unsqueeze(2).contiguous();
        } else {
            mean_broadcast = mean.unsqueeze(0).contiguous();
            invstd_broadcast = invstd.unsqueeze(0).contiguous();
        }

        auto normalized = ((input - mean_broadcast) * invstd_broadcast).contiguous();

        // Gradient with respect to weight: sum(grad_output * normalized, dim=[0,2])
        Tensor grad_weight;
        if (shape.size() == 3) {
            grad_weight = sum(sum((grad_output * normalized)
                .reshape({N, C, L}).contiguous(), 0, false), 1, false);
        } else {
            grad_weight = sum((grad_output * normalized).contiguous(), 0, false);
        }

        // Gradient with respect to bias: sum(grad_output, dim=[0,2])
        Tensor grad_bias;
        if (shape.size() == 3) {
            grad_bias = sum(sum(grad_output
                .reshape({N, C, L}).contiguous(), 0, false), 1, false);
        } else {
            grad_bias = sum(grad_output.contiguous(), 0, false);
        }

        // Gradient with respect to normalized input
        Tensor weight_broadcast;
        if (shape.size() == 3) {
            weight_broadcast = weight.unsqueeze(0).unsqueeze(2).contiguous();
        } else {
            weight_broadcast = weight.unsqueeze(0).contiguous();
        }
        auto grad_normalized = (grad_output * weight_broadcast).contiguous();

        // Gradient with respect to input
        Tensor grad_input_normalized;
        Tensor normalized_reshaped;
        if (shape.size() == 3) {
            grad_input_normalized = grad_normalized.reshape({N, C, L}).contiguous();
            normalized_reshaped = normalized.reshape({N, C, L}).contiguous();
        } else {
            grad_input_normalized = grad_normalized.contiguous();
            normalized_reshaped = normalized.contiguous();
        }

        Tensor sum_grad, sum_grad_x_norm;
        if (shape.size() == 3) {
            sum_grad = sum(sum(grad_input_normalized, 0, true), 2, true).contiguous();
            sum_grad_x_norm = sum(sum((grad_input_normalized * normalized_reshaped),
                                0, true), 2, true).contiguous();
        } else {
            sum_grad = sum(grad_input_normalized, 0, true).contiguous();
            sum_grad_x_norm = sum((grad_input_normalized * normalized_reshaped),
                                0, true).contiguous();
        }

        Tensor invstd_expanded;
        if (shape.size() == 3) {
            invstd_expanded = invstd.unsqueeze(0).unsqueeze(-1).contiguous();
        } else {
            invstd_expanded = invstd.unsqueeze(0).contiguous();
        }

        auto term1 = (sum_grad / static_cast<float>(batch_size)).contiguous();
        auto term2 = (normalized_reshaped * sum_grad_x_norm / static_cast<float>(batch_size)).contiguous();
        auto grad_input = ((grad_input_normalized - term1 - term2) * invstd_expanded).contiguous();

        if (shape.size() == 3) {
            grad_input = grad_input.reshape({N, C, L}).contiguous();
        }

        return {grad_input, grad_weight, grad_bias};
    }

private:
    bool affine_;
    double eps_;
};

BatchNorm1d::BatchNorm1d(int64_t num_features, double eps, double momentum,
                        bool affine, bool track_running_stats)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      affine_(affine), track_running_stats_(track_running_stats) {

    if (affine) {
        weight_ = Variable(ones({num_features}), true);
        bias_ = Variable(zeros({num_features}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
    }

    reset_parameters();
}

auto BatchNorm1d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C] or [N, C, L]
    auto shape = input.shape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("BatchNorm1d expects 2D or 3D input (got " +
                               std::to_string(shape.size()) + "D)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape.size() == 3 ? shape[2] : 1;

    if (C != num_features_) {
        throw std::runtime_error("Expected " + std::to_string(num_features_) +
                               " features, got " + std::to_string(C));
    }

    int64_t batch_size = N * L;
    if (training_ && batch_size == 0) {
        throw std::runtime_error("BatchNorm1d: Cannot compute statistics for empty batch");
    }

    Device original_device = input.tensor().device();
    Tensor input_work = input.tensor();

    Tensor batch_mean, batch_var;

    if (training_) {
        // Compute mean and variance over N and L dimensions
        // Reshape to [N*L, C] for easier computation
        Tensor reshaped_input = shape.size() == 3 ?
            input_work.reshape({N * L, C}).contiguous() : input_work.contiguous();

        // Compute mean: average over batch dimension (N*L)
        batch_mean = mean(reshaped_input, 0, false);

        // Compute variance
        auto mean_broadcast = batch_mean.unsqueeze(0).contiguous();
        auto centered = (reshaped_input - mean_broadcast).contiguous();
        batch_var = mean(centered * centered, 0, false);

        // Update running statistics
        if (track_running_stats_) {
            auto unbiased_var = batch_var * (static_cast<float>(batch_size) /
                                            static_cast<float>(batch_size - 1));

            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            // Exponential moving average update using tensor operations
            // This properly handles both CPU and CUDA/GPU tensors
            auto& rm_tensor = rm_var_ptr->tensor();
            auto& rv_tensor = rv_var_ptr->tensor();
            rm_tensor = rm_tensor * (1.0f - momentum_) + batch_mean * momentum_;
            rv_tensor = rv_tensor * (1.0f - momentum_) + unbiased_var * momentum_;

            num_batches_tracked_++;
        }
    } else {
        if (track_running_stats_) {
            batch_mean = buffers_["running_mean"]->tensor();
            batch_var = buffers_["running_var"]->tensor();
        } else {
            throw std::runtime_error("BatchNorm1d in eval mode requires track_running_stats=true");
        }
    }

    // Normalize
    Tensor output;
    if (shape.size() == 3) {
        auto mean_broadcast = batch_mean.unsqueeze(0).unsqueeze(2).contiguous();
        auto var_broadcast = batch_var.unsqueeze(0).unsqueeze(2).contiguous();
        auto eps_tensor = full({}, eps_, var_broadcast.dtype(), var_broadcast.device());
        auto invstd = pow(var_broadcast + eps_tensor, -0.5f).contiguous();
        auto normalized = ((input_work - mean_broadcast) * invstd).contiguous();

        if (affine_) {
            auto& weight_ptr = parameters_["weight"];
            auto& bias_ptr = parameters_["bias"];
            auto weight_broadcast = weight_ptr->tensor().unsqueeze(0).unsqueeze(2).contiguous();
            auto bias_broadcast = bias_ptr->tensor().unsqueeze(0).unsqueeze(2).contiguous();
            output = (normalized * weight_broadcast + bias_broadcast).contiguous();
        } else {
            output = normalized;
        }
    } else {
        auto mean_broadcast = batch_mean.unsqueeze(0).contiguous();
        auto var_broadcast = batch_var.unsqueeze(0).contiguous();
        auto eps_tensor = full({}, eps_, var_broadcast.dtype(), var_broadcast.device());
        auto invstd = pow(var_broadcast + eps_tensor, -0.5f).contiguous();
        auto normalized = ((input_work - mean_broadcast) * invstd).contiguous();

        if (affine_) {
            auto& weight_ptr = parameters_["weight"];
            auto& bias_ptr = parameters_["bias"];
            auto weight_broadcast = weight_ptr->tensor().unsqueeze(0).contiguous();
            auto bias_broadcast = bias_ptr->tensor().unsqueeze(0).contiguous();
            output = (normalized * weight_broadcast + bias_broadcast).contiguous();
        } else {
            output = normalized;
        }
    }

    // Set up autograd if needed
    bool requires_grad = input.requires_grad();
    if (affine_ && parameters_.find("weight") != parameters_.end()) {
        requires_grad = requires_grad || parameters_["weight"]->requires_grad();
    }

    if (requires_grad) {
        auto result = Variable(output, true);

        // Compute invstd for backward
        auto eps_tensor = full({}, eps_, batch_var.dtype(), batch_var.device());
        auto invstd = pow(batch_var + eps_tensor, -0.5f);

        Tensor batch_mean_final = batch_mean.contiguous();
        Tensor invstd_final = invstd.contiguous();

        Tensor weight_tensor = affine_ ? parameters_["weight"]->tensor() : ones({C}, input.tensor().dtype(), original_device);

        std::vector<Tensor> tensors_to_save = {
            input.tensor().contiguous(),
            batch_mean_final,
            invstd_final,
            weight_tensor.contiguous()
        };

        auto grad_fn = std::make_shared<BatchNorm1dBackward>(
            affine_, eps_, std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

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

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        return result;
    } else {
        return Variable(output, false);
    }
}

auto BatchNorm1d::reset_parameters() -> void {
    if (track_running_stats_) {
        buffers_["running_mean"]->tensor().zero_();
        buffers_["running_var"]->tensor().fill_(1.0f);
        num_batches_tracked_ = 0;
    }
}

} // namespace tenzor::nn
