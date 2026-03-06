#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cmath>

// SIMD headers for optimized BatchNorm
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// OpenMP for parallel execution
#ifdef _OPENMP
#include <omp.h>
#include <thread>
#endif

#include <iostream>

// Get optimal thread count for compute-bound operations
static inline int get_optimal_threads() {
#ifdef _OPENMP
    static int optimal = []() {
        // Use all hardware threads for compute-bound work
        int hw_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        // Ensure OpenMP uses this count
        omp_set_num_threads(hw_threads);
        return hw_threads;
    }();
    return optimal;
#else
    return 1;
#endif
}

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
        auto invstd = saved[2].contiguous();  // saved_inv_var from cuDNN or computed invstd
        auto weight = saved[3].contiguous();

        // grad_output: [N, C, H, W]
        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t spatial_size = H * W;
        int64_t batch_size = N * spatial_size;

        // ================================================================
        // FAST CUDA PATH: Use cuDNN backward kernel (single kernel launch)
        // ================================================================
        if (input.device().type == Device::Type::CUDA &&
            (input.dtype() == DType::Float32 || input.dtype() == DType::Float16 || input.dtype() == DType::Float64)) {
            // Dispatch to cuDNN/CUDA kernel for BatchNorm backward
            // inputs: [grad_output, input, gamma, saved_mean, saved_inv_var]
            OpAttributes backward_attrs;
            backward_attrs.set(AttrKey::Eps, static_cast<float>(eps_));

            std::vector<Tensor> backward_inputs = {grad_output, input, weight, mean, invstd};
            std::vector<Tensor> backward_results = dispatch(OpId::BatchNorm2dBackward, backward_inputs, backward_attrs);

            // Return gradients in the order of input_vars: [input, weight, bias]
            return {backward_results[0], backward_results[1], backward_results[2]};
        }

        // ================================================================
        // FALLBACK: Use tensor operations (CPU, Vulkan, etc.)
        // ================================================================

        // For Float16, upcast to Float32 to avoid overflow/precision loss
        // (matches CUDA/cuDNN which uses FP32 internally for batchnorm backward)
        DType orig_dtype = input.dtype();
        bool needs_upcast = (orig_dtype == DType::Float16);
        if (needs_upcast) {
            grad_output = grad_output.to(DType::Float32);
            input = input.to(DType::Float32);
            mean = mean.to(DType::Float32);
            invstd = invstd.to(DType::Float32);
            weight = weight.to(DType::Float32);
        }

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

        // Downcast back to original dtype if we upcasted
        if (needs_upcast) {
            grad_input = grad_input.to(orig_dtype);
            grad_weight = grad_weight.to(orig_dtype);
            grad_bias = grad_bias.to(orig_dtype);
        }

        // Return gradients in the order of input_vars: [input, weight, bias]
        return {grad_input, grad_weight, grad_bias};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Complex batch normalization backward -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
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
        // Cache pointers to avoid hash map lookups in forward pass (~2-3ms savings)
        cached_weight_ = parameters_["weight"];
        cached_bias_ = parameters_["bias"];
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        num_batches_tracked_ = Variable(zeros({}, DType::Int64), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
        register_buffer("num_batches_tracked", num_batches_tracked_);
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

    // ============================================================================
    // FAST INFERENCE PATH: CPU Float32 eval mode with running stats
    // ============================================================================
    const bool needs_grad = is_grad_enabled() &&
        (input.requires_grad() || (affine_ && cached_weight_ && cached_weight_->requires_grad()));

    if (!needs_grad && !training_ && track_running_stats_ &&
        input.tensor().device().type == Device::Type::CPU && input.tensor().dtype() == DType::Float32) {

        const Tensor& input_tensor = input.tensor();
        const auto* input_data = input_tensor.data<float>();

        // Get running stats and affine parameters directly
        const Tensor& running_mean = buffers_["running_mean"]->tensor();
        const Tensor& running_var = buffers_["running_var"]->tensor();
        const auto* mean_data = running_mean.data<float>();
        const auto* var_data = running_var.data<float>();

        const float* gamma_data = nullptr;
        const float* beta_data = nullptr;
        if (affine_ && cached_weight_ && cached_bias_) {
            gamma_data = cached_weight_->tensor().data<float>();
            beta_data = cached_bias_->tensor().data<float>();
        }

        // Allocate output
        auto output = Tensor::empty_uninitialized(
            {N, C, H, W}, DType::Float32, Device::cpu());
        auto* output_data = output.data<float>();

        const float eps = static_cast<float>(eps_);

        // Use thread-local storage to avoid allocation per forward call
        // This gives significant speedup for repeated inference
        thread_local std::vector<float> tls_scale, tls_shift;
        if (static_cast<int64_t>(tls_scale.size()) < C) {
            tls_scale.resize(C);
            tls_shift.resize(C);
        }
        float* scale = tls_scale.data();
        float* shift = tls_shift.data();

        // Precompute scale and shift for each channel
        for (int64_t c = 0; c < C; c++) {
            float inv_std = 1.0f / std::sqrt(var_data[c] + eps);
            if (affine_) {
                scale[c] = gamma_data[c] * inv_std;
                shift[c] = beta_data[c] - mean_data[c] * scale[c];
            } else {
                scale[c] = inv_std;
                shift[c] = -mean_data[c] * inv_std;
            }
        }

        // Apply normalization with SIMD
        for (int64_t n = 0; n < N; n++) {
            for (int64_t c = 0; c < C; c++) {
                const float s = scale[c];
                const float b = shift[c];
                const int64_t idx = (n * C + c) * spatial_size;

                #if defined(__AVX512F__)
                __m512 v_scale = _mm512_set1_ps(s);
                __m512 v_shift = _mm512_set1_ps(b);
                int64_t i = 0;
                for (; i + 16 <= spatial_size; i += 16) {
                    __m512 v_in = _mm512_loadu_ps(input_data + idx + i);
                    __m512 v_out = _mm512_fmadd_ps(v_in, v_scale, v_shift);
                    _mm512_storeu_ps(output_data + idx + i, v_out);
                }
                for (; i < spatial_size; i++) {
                    output_data[idx + i] = input_data[idx + i] * s + b;
                }
                #else
                for (int64_t i = 0; i < spatial_size; i++) {
                    output_data[idx + i] = input_data[idx + i] * s + b;
                }
                #endif
            }
        }

        return Variable(output, false);
    }

    // ============================================================================
    // STANDARD PATH: Full dispatch with autograd support
    // ============================================================================

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

    // FP16/BF16 forward upcast: compute in FP32 to prevent overflow/underflow
    // in mean/variance reductions. Matches cuDNN behavior and the backward pass pattern.
    DType orig_dtype = input_work.dtype();
    bool needs_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    if (needs_upcast) {
        input_work = input_work.to(DType::Float32);
    }

    Tensor batch_mean, batch_var;

    if (training_) {
        // Check if we can use fused cuDNN training path (CUDA + affine + track_running_stats)
        bool use_fused_training = use_gpu && affine_ && track_running_stats_ &&
                                  cached_weight_ && cached_bias_;

        if (use_fused_training) {
            // ================================================================
            // FUSED CUDNN PATH: Single kernel for mean/var + normalize + update
            // ================================================================
            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            Tensor running_mean = rm_var_ptr->tensor();
            Tensor running_var = rv_var_ptr->tensor();
            if (running_mean.device() != original_device) {
                running_mean = running_mean.to(original_device);
            }
            if (running_var.device() != original_device) {
                running_var = running_var.to(original_device);
            }
            // Convert running stats to input dtype for kernel compatibility
            DType input_dtype = input_work.dtype();
            if (running_mean.dtype() != input_dtype) {
                running_mean = running_mean.to(input_dtype);
            }
            if (running_var.dtype() != input_dtype) {
                running_var = running_var.to(input_dtype);
            }

            Tensor weight = cached_weight_->tensor();
            Tensor bias = cached_bias_->tensor();
            if (weight.device() != original_device) {
                weight = weight.to(original_device);
            }
            if (bias.device() != original_device) {
                bias = bias.to(original_device);
            }
            // Convert weight/bias to input dtype for kernel compatibility
            if (weight.dtype() != input_dtype) {
                weight = weight.to(input_dtype);
            }
            if (bias.dtype() != input_dtype) {
                bias = bias.to(input_dtype);
            }

            OpAttributes fused_attrs;
            fused_attrs.set(AttrKey::Eps, static_cast<float>(eps_));
            fused_attrs.set(AttrKey::Momentum, static_cast<float>(momentum_));

            std::vector<Tensor> fused_inputs = {input_work, running_mean, running_var, weight, bias};
            std::vector<Tensor> fused_results = dispatch(OpId::BatchNorm2dFusedTraining, fused_inputs, fused_attrs);

            // Update running stats from cuDNN (store back as Float32 for storage efficiency)
            rm_var_ptr->tensor() = fused_results[1].to(DType::Float32);
            rv_var_ptr->tensor() = fused_results[2].to(DType::Float32);
            batch_mean = fused_results[3];  // saved_mean for backward
            batch_var = fused_results[4];   // saved_inv_var for backward
            buffers_["num_batches_tracked"]->tensor().data<int64_t>()[0]++;

            // Return output directly (autograd handled below if needed)
            Tensor output = fused_results[0];

            // Handle autograd for fused path
            bool requires_grad = input.requires_grad();
            if (affine_ && cached_weight_) {
                requires_grad = requires_grad || cached_weight_->requires_grad();
            }
            if (is_grad_enabled() && requires_grad) {
                auto result = Variable(output, true);
                // Note: For fused path, batch_var is actually saved_inv_var from cuDNN
                std::vector<Tensor> tensors_to_save = {
                    input.tensor().contiguous(),
                    batch_mean.contiguous(),
                    batch_var.contiguous(),  // This is inv_std from cuDNN
                    weight.contiguous()
                };
                auto grad_fn = std::make_shared<BatchNorm2dBackward>(
                    affine_, eps_, std::move(tensors_to_save)
                );
                result.set_grad_fn(grad_fn);

                // Track input variables for gradient accumulation
                // MUST include all inputs to maintain 1:1 index correspondence with gradients
                // The engine correctly skips variables that don't require grad
                std::vector<Variable> input_vars = {input};
                if (affine_ && cached_weight_ && cached_bias_) {
                    input_vars.push_back(*cached_weight_);
                    input_vars.push_back(*cached_bias_);
                }
                grad_fn->set_input_variables(input_vars);

                // Connect to input's grad_fn to continue the backward chain
                std::vector<std::shared_ptr<Function>> next_funcs;
                if (input.grad_fn()) {
                    next_funcs.push_back(input.grad_fn());
                }
                grad_fn->set_next_functions(next_funcs);

                return result;
            }
            return Variable(output, false);
        }

        // ================================================================
        // STANDARD TRAINING PATH: Separate kernels for backward compat
        // ================================================================
        // Training mode: compute batch statistics using backend dispatch
        OpAttributes mean_var_attrs;
        std::vector<Tensor> mean_var_inputs = {input_work};
        std::vector<Tensor> mean_var_results = dispatch(OpId::BatchNorm2dMeanVar, mean_var_inputs, mean_var_attrs);
        batch_mean = mean_var_results[0];
        batch_var = mean_var_results[1];

        // Update running statistics using backend kernel
        if (track_running_stats_) {
            // Use unbiased variance estimate for running statistics
            int64_t batch_size = N * spatial_size;
            auto unbiased_var = batch_var * (static_cast<float>(batch_size) /
                                            static_cast<float>(batch_size - 1));

            // Get running stats and transfer to input device if needed
            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            Tensor running_mean_on_device = rm_var_ptr->tensor();
            Tensor running_var_on_device = rv_var_ptr->tensor();
            if (running_mean_on_device.device() != original_device) {
                running_mean_on_device = running_mean_on_device.to(original_device);
            }
            if (running_var_on_device.device() != original_device) {
                running_var_on_device = running_var_on_device.to(original_device);
            }
            // Convert running stats to input dtype for kernel compatibility
            DType input_dtype = input_work.dtype();
            if (running_mean_on_device.dtype() != input_dtype) {
                running_mean_on_device = running_mean_on_device.to(input_dtype);
            }
            if (running_var_on_device.dtype() != input_dtype) {
                running_var_on_device = running_var_on_device.to(input_dtype);
            }

            // Use backend kernel for running stats update
            NewOpAttributes update_attrs;
            update_attrs.set(AttrKey::Momentum, static_cast<double>(momentum_));
            std::vector<Tensor> update_inputs = {running_mean_on_device, running_var_on_device, batch_mean, unbiased_var};
            std::vector<Tensor> updated_stats = dispatch(OpId::BatchNorm2dUpdateRunningStats, update_inputs, update_attrs);

            // Store updated stats back in Float32 for storage efficiency
            rm_var_ptr->tensor() = updated_stats[0].to(DType::Float32);
            rv_var_ptr->tensor() = updated_stats[1].to(DType::Float32);

            buffers_["num_batches_tracked"]->tensor().data<int64_t>()[0]++;
        }
    } else {
        // Inference mode: use running statistics (transfer to input device if needed)
        if (track_running_stats_) {
            batch_mean = buffers_["running_mean"]->tensor();
            batch_var = buffers_["running_var"]->tensor();
            // Transfer to input device if they're on different devices
            if (batch_mean.device() != original_device) {
                batch_mean = batch_mean.to(original_device);
            }
            if (batch_var.device() != original_device) {
                batch_var = batch_var.to(original_device);
            }
            // Convert to input dtype for kernel compatibility
            DType input_dtype = input_work.dtype();
            if (batch_mean.dtype() != input_dtype) {
                batch_mean = batch_mean.to(input_dtype);
            }
            if (batch_var.dtype() != input_dtype) {
                batch_var = batch_var.to(input_dtype);
            }
        } else {
            throw std::runtime_error("BatchNorm2d in eval mode requires track_running_stats=true");
        }
    }

    // Normalize using backend kernel
    Tensor output;
    OpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Eps, static_cast<float>(eps_));

    if (affine_ && cached_weight_ && cached_bias_) {
        // Use affine forward kernel: output = gamma * (x - mean) / sqrt(var + eps) + beta
        // Transfer weight and bias to input device if needed
        Tensor weight_on_device = cached_weight_->tensor();
        Tensor bias_on_device = cached_bias_->tensor();
        if (weight_on_device.device() != original_device) {
            weight_on_device = weight_on_device.to(original_device);
        }
        if (bias_on_device.device() != original_device) {
            bias_on_device = bias_on_device.to(original_device);
        }
        // Convert weight and bias to input dtype for kernel compatibility
        DType input_dtype = input_work.dtype();
        if (weight_on_device.dtype() != input_dtype) {
            weight_on_device = weight_on_device.to(input_dtype);
        }
        if (bias_on_device.dtype() != input_dtype) {
            bias_on_device = bias_on_device.to(input_dtype);
        }
        std::vector<Tensor> forward_inputs = {input_work, batch_mean, batch_var, weight_on_device, bias_on_device};
        std::vector<Tensor> forward_results = dispatch(OpId::BatchNorm2dForwardAffine, forward_inputs, forward_attrs);
        output = forward_results[0];
    } else {
        // Use non-affine forward kernel: output = (x - mean) / sqrt(var + eps)
        std::vector<Tensor> forward_inputs = {input_work, batch_mean, batch_var};
        std::vector<Tensor> forward_results = dispatch(OpId::BatchNorm2dForward, forward_inputs, forward_attrs);
        output = forward_results[0];
    }

    // Downcast output back to original dtype if we upcasted
    if (needs_upcast) {
        output = output.to(orig_dtype);
    }

    // Set up autograd if needed - check is_grad_enabled() first for fast inference path
    bool requires_grad = input.requires_grad();
    if (affine_ && cached_weight_) {
        requires_grad = requires_grad || cached_weight_->requires_grad();
    }
    if (is_grad_enabled() && requires_grad) {
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

        // Use cached weight pointer for efficiency, transfer to input device and dtype if needed
        Tensor weight_tensor;
        DType input_dtype = input.tensor().dtype();
        if (affine_ && cached_weight_) {
            weight_tensor = cached_weight_->tensor();
            if (weight_tensor.device() != original_device) {
                weight_tensor = weight_tensor.to(original_device);
            }
            if (weight_tensor.dtype() != input_dtype) {
                weight_tensor = weight_tensor.to(input_dtype);
            }
        } else {
            weight_tensor = ones({C}, input_dtype, original_device);
        }
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
        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input};
        if (affine_ && cached_weight_ && cached_bias_) {
            input_vars.push_back(*cached_weight_);
            input_vars.push_back(*cached_bias_);
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
        buffers_["num_batches_tracked"]->tensor().zero_();
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

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Complex batch normalization backward -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
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
        // Cache pointers to avoid hash map lookups in forward pass (~2-3ms savings)
        cached_weight_ = parameters_["weight"];
        cached_bias_ = parameters_["bias"];
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        num_batches_tracked_ = Variable(zeros({}, DType::Int64), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
        register_buffer("num_batches_tracked", num_batches_tracked_);
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

            buffers_["num_batches_tracked"]->tensor().data<int64_t>()[0]++;
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

        if (affine_ && cached_weight_ && cached_bias_) {
            auto weight_broadcast = cached_weight_->tensor().unsqueeze(0).unsqueeze(2).contiguous();
            auto bias_broadcast = cached_bias_->tensor().unsqueeze(0).unsqueeze(2).contiguous();
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

        if (affine_ && cached_weight_ && cached_bias_) {
            auto weight_broadcast = cached_weight_->tensor().unsqueeze(0).contiguous();
            auto bias_broadcast = cached_bias_->tensor().unsqueeze(0).contiguous();
            output = (normalized * weight_broadcast + bias_broadcast).contiguous();
        } else {
            output = normalized;
        }
    }

    // Set up autograd if needed
    bool requires_grad = input.requires_grad();
    if (affine_ && cached_weight_) {
        requires_grad = requires_grad || cached_weight_->requires_grad();
    }

    if (is_grad_enabled() && requires_grad) {
        auto result = Variable(output, true);

        // Compute invstd for backward
        auto eps_tensor = full({}, eps_, batch_var.dtype(), batch_var.device());
        auto invstd = pow(batch_var + eps_tensor, -0.5f);

        Tensor batch_mean_final = batch_mean.contiguous();
        Tensor invstd_final = invstd.contiguous();

        Tensor weight_tensor = (affine_ && cached_weight_) ? cached_weight_->tensor() : ones({C}, input.tensor().dtype(), original_device);

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

        // Track input variables for gradient accumulation
        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input};
        if (affine_ && cached_weight_ && cached_bias_) {
            input_vars.push_back(*cached_weight_);
            input_vars.push_back(*cached_bias_);
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
        buffers_["num_batches_tracked"]->tensor().zero_();
    }
}

} // namespace tenzor::nn
