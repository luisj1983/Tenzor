/**
 * @file pooling.cpp
 * @brief CPU pooling kernel implementations
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tenzor {
namespace cpu {

auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, H_out, W_out}, DType::Int64, input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();
    int64_t* idx_data = indices.data<int64_t>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;

                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = 0;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                                if (in_data[in_idx] > max_val) {
                                    max_val = in_data[in_idx];
                                    max_idx = h * W + w;
                                }
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    out_data[out_idx] = max_val;
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }

    return {output, indices};
}

auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    int64_t max_idx = idx_data[out_idx];
                    int64_t h = max_idx / W;
                    int64_t w = max_idx % W;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    grad_in_data[in_idx] += grad_out_data[out_idx];
                }
            }
        }
    }

    return grad_input;
}

auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;

                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                sum += in_data[((n * C + c) * H + h) * W + w];
                                count++;
                            }
                        }
                    }

                    out_data[((n * C + c) * H_out + oh) * W_out + ow] = sum / count;
                }
            }
        }
    }

    return output;
}

auto avgpool2d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride,
                                int64_t padding) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;

                    int64_t count = 0;
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;
                            if (h >= 0 && h < H && w >= 0 && w < W) count++;
                        }
                    }

                    float grad_val = grad_out_data[((n * C + c) * H_out + oh) * W_out + ow] / count;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                grad_in_data[((n * C + c) * H + h) * W + w] += grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

auto adaptive_avgpool2d_kernel(const Tensor& input, int64_t output_h,
                                int64_t output_w) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, output_h, output_w}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            sum += in_data[((n * C + c) * H + h) * W + w];
                            count++;
                        }
                    }

                    out_data[((n * C + c) * output_h + oh) * output_w + ow] = sum / count;
                }
            }
        }
    }

    return output;
}

auto adaptive_avgpool2d_backward_kernel(const Tensor& grad_output,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    int64_t count = (h_end - h_start) * (w_end - w_start);
                    float grad_val = grad_out_data[((n * C + c) * output_h + oh) * output_w + ow] / count;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            grad_in_data[((n * C + c) * H + h) * W + w] += grad_val;
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

auto adaptive_maxpool2d_kernel(const Tensor& input, int64_t output_h,
                                int64_t output_w) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, output_h, output_w}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, output_h, output_w}, DType::Int64, input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();
    int64_t* idx_data = indices.data<int64_t>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            int64_t in_idx = ((n * C + c) * H + h) * W + w;
                            if (in_data[in_idx] > max_val) {
                                max_val = in_data[in_idx];
                                max_idx = h * W + w;
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * output_h + oh) * output_w + ow;
                    out_data[out_idx] = max_val;
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }

    return {output, indices};
}

auto adaptive_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t out_idx = ((n * C + c) * output_h + oh) * output_w + ow;
                    int64_t max_idx = idx_data[out_idx];
                    int64_t h = max_idx / W;
                    int64_t w = max_idx % W;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    grad_in_data[in_idx] += grad_out_data[out_idx];
                }
            }
        }
    }

    return grad_input;
}

} // namespace cpu
} // namespace tenzor
