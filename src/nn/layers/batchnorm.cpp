#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>

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

auto BatchNorm2d::forward(const Variable& input) -> Variable {
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

    // Track original device for final output
    Device original_device = input.tensor().device();
    bool use_gpu = (original_device.type == Device::Type::CUDA);

    // Work on CPU for now (TODO: implement GPU batch norm kernels)
    Tensor input_work = use_gpu ? input.tensor().to(Device::cpu()) : input.tensor();

    Tensor batch_mean, batch_var;

    if (training_) {
        // Training mode: compute batch statistics
        // We need to compute mean and variance over dimensions [0, 2, 3], keeping dimension 1 (channels)

        // Manual computation to correctly handle NCHW layout
        // Allocate tensors for mean and variance (per channel) on CPU
        batch_mean = zeros({C});
        batch_var = zeros({C});

        auto* input_data = input_work.data<float>();
        auto* mean_data = batch_mean.data<float>();
        auto* var_data = batch_var.data<float>();

        int64_t batch_size = N * spatial_size;

        // Compute mean for each channel
        for (int64_t c = 0; c < C; c++) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        sum += input_data[idx];
                    }
                }
            }
            mean_data[c] = static_cast<float>(sum / batch_size);
        }

        // Compute variance for each channel
        for (int64_t c = 0; c < C; c++) {
            double sum_sq = 0.0;
            float mu = mean_data[c];
            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        int64_t idx = ((n * C + c) * H + h) * W + w;
                        float diff = input_data[idx] - mu;
                        sum_sq += diff * diff;
                    }
                }
            }
            var_data[c] = static_cast<float>(sum_sq / batch_size);
        }

        // Update running statistics (on CPU, then transfer back if needed)
        if (track_running_stats_) {
            // Use unbiased variance estimate for running statistics
            int64_t batch_size = N * spatial_size;
            auto unbiased_var = batch_var * (static_cast<float>(batch_size) /
                                            static_cast<float>(batch_size - 1));

            // Get running stats on CPU for computation
            auto& rm_var = buffers_["running_mean"];
            auto& rv_var = buffers_["running_var"];
            Tensor rm_cpu = use_gpu ? rm_var.tensor().to(Device::cpu()) : rm_var.tensor();
            Tensor rv_cpu = use_gpu ? rv_var.tensor().to(Device::cpu()) : rv_var.tensor();

            rm_cpu = rm_cpu * (1.0f - static_cast<float>(momentum_)) +
                     batch_mean * static_cast<float>(momentum_);
            rv_cpu = rv_cpu * (1.0f - static_cast<float>(momentum_)) +
                     unbiased_var * static_cast<float>(momentum_);

            // Transfer back to original device if needed
            rm_var.tensor() = use_gpu ? rm_cpu.to(original_device) : rm_cpu;
            rv_var.tensor() = use_gpu ? rv_cpu.to(original_device) : rv_cpu;

            num_batches_tracked_++;
        }
    } else {
        // Inference mode: use running statistics (transfer to CPU if needed)
        if (track_running_stats_) {
            Tensor rm_tensor = buffers_["running_mean"].tensor();
            Tensor rv_tensor = buffers_["running_var"].tensor();
            batch_mean = use_gpu ? rm_tensor.to(Device::cpu()) : rm_tensor;
            batch_var = use_gpu ? rv_tensor.to(Device::cpu()) : rv_tensor;
        } else {
            throw std::runtime_error("BatchNorm2d in eval mode requires track_running_stats=true");
        }
    }

    // Normalize: (x - mean) / sqrt(var + eps) (all on CPU)
    // Reshape statistics to [1, C, 1, 1] for broadcasting
    auto mean_broadcast = batch_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
    auto var_broadcast = batch_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();

    auto invstd = pow(var_broadcast + static_cast<float>(eps_), -0.5f);
    auto normalized = (input_work - mean_broadcast) * invstd;

    // Determine output tensor (all operations on CPU)
    Tensor output;
    if (affine_) {
        // Get weight and bias on CPU
        auto& weight = parameters_["weight"];
        auto& bias = parameters_["bias"];
        Tensor weight_work = use_gpu ? weight.tensor().to(Device::cpu()) : weight.tensor();
        Tensor bias_work = use_gpu ? bias.tensor().to(Device::cpu()) : bias.tensor();

        // Apply affine transformation: output = weight * normalized + bias
        auto weight_broadcast = weight_work.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto bias_broadcast = bias_work.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        output = normalized * weight_broadcast + bias_broadcast;
    } else {
        output = normalized;
    }

    // Transfer back to original device if needed
    if (use_gpu) {
        output = output.to(original_device);
    }

    // Set up autograd if needed
    bool requires_grad = input.requires_grad();
    if (affine_ && parameters_.find("weight") != parameters_.end()) {
        requires_grad = requires_grad || parameters_["weight"].requires_grad();
    }
    if (requires_grad) {
        // Create result variable from output
        auto result = Variable(output, true);

        // Prepare tensors to save for backward
        auto invstd_squeezed = invstd.reshape({C});

        // Transfer statistics to original device if needed for backward
        // Make sure they're contiguous first, then transfer
        // CRITICAL: Always make contiguous, even for CPU-only execution
        Tensor batch_mean_final = use_gpu ? batch_mean.contiguous().to(original_device) : batch_mean.contiguous();
        Tensor invstd_final = use_gpu ? invstd_squeezed.contiguous().to(original_device) : invstd_squeezed.contiguous();

        // CRITICAL: Access weight from parameters_ map
        Tensor weight_tensor = affine_ ? parameters_["weight"].tensor() : ones({C}, DType::Float32, original_device);
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
        std::vector<Variable*> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(const_cast<Variable*>(&input));
        }
        if (affine_) {
            // Use pointers to the Variables in the parameters_ map, not member variables
            auto weight_it = parameters_.find("weight");
            auto bias_it = parameters_.find("bias");
            if (weight_it != parameters_.end() && weight_it->second.requires_grad()) {
                input_vars.push_back(&(weight_it->second));
            }
            if (bias_it != parameters_.end() && bias_it->second.requires_grad()) {
                input_vars.push_back(&(bias_it->second));
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
        buffers_["running_mean"].tensor().zero_();
        buffers_["running_var"].tensor().fill_(1.0f);
        num_batches_tracked_ = 0;
    }
}

} // namespace tenzor::nn
