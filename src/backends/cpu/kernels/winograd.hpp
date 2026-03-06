#pragma once
/// @file winograd.hpp
/// @brief Winograd F(2x2, 3x3) convolution transform for CPU.
///
/// Implements the minimal filtering algorithm that reduces arithmetic complexity
/// from 9 multiplies per output element (direct) to 4 multiplies (transform domain).
///
/// Transform matrices for F(2, 3):
///   B^T (input transform, 4x4):
///     [[1,  0, -1,  0],
///      [0,  1,  1,  0],
///      [0, -1,  1,  0],
///      [0,  1,  0, -1]]
///
///   G (filter transform, 4x3):
///     [[1,      0,     0   ],
///      [0.5,    0.5,   0.5 ],
///      [0.5,   -0.5,   0.5 ],
///      [0,      0,     1   ]]
///
///   A^T (output transform, 2x4):
///     [[1, 1,  1,  0],
///      [0, 1, -1, -1]]
///
/// Gate: Only applicable when kH==3 && kW==3 && stride==1 && dilation==1.

#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/omp_thresholds.hpp"
#include <vector>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor::cpu {

/// Check whether the convolution parameters are eligible for Winograd F(2x2,3x3).
inline bool can_use_winograd_f2x3(int64_t kH, int64_t kW,
                                    int64_t stride, int64_t dilation,
                                    int64_t groups) {
    return kH == 3 && kW == 3 && stride == 1 && dilation == 1 && groups == 1;
}

/// Winograd F(2x2, 3x3) convolution.
///
/// @param input  Tensor of shape (N, C_in, H, W), must be contiguous and Float32.
/// @param weight Tensor of shape (C_out, C_in, 3, 3), must be contiguous and Float32.
/// @param output Tensor of shape (N, C_out, out_H, out_W), pre-allocated.
/// @param pad_h  Padding in the height dimension.
/// @param pad_w  Padding in the width dimension.
///
/// Requires: kH==3, kW==3, stride==1, dilation==1, groups==1.
inline void winograd_conv2d_f2x3(const float* input_data,
                                  const float* weight_data,
                                  float* output_data,
                                  int64_t batch,
                                  int64_t C_in, int64_t H, int64_t W,
                                  int64_t C_out,
                                  int64_t out_h, int64_t out_w,
                                  int64_t pad_h, int64_t pad_w) {
    // Number of 2x2 output tiles
    const int64_t tile_h = (out_h + 1) / 2;
    const int64_t tile_w = (out_w + 1) / 2;
    const int64_t num_tiles = tile_h * tile_w;

    // =========================================================================
    // Step 1: Pre-transform all filters: U = G * g * G^T
    // =========================================================================
    // G is the 4x3 filter transform matrix for F(2,3):
    //   G = [[1,    0,    0   ],
    //        [0.5,  0.5,  0.5 ],
    //        [0.5, -0.5,  0.5 ],
    //        [0,    0,    1   ]]
    //
    // Each 3x3 filter becomes a 4x4 element in the transform domain.
    std::vector<float> U(C_out * C_in * 16);

    #pragma omp parallel for collapse(2) if(C_out * C_in > OmpThresholds::medium())
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic = 0; ic < C_in; ++ic) {
            const float* g = weight_data + (oc * C_in + ic) * 9;
            float* u = U.data() + (oc * C_in + ic) * 16;

            // Compute temp = G * g  (4x3 * 3x3 -> 4x3)
            float tmp[4][3];
            for (int s = 0; s < 3; ++s) {
                tmp[0][s] = g[0 * 3 + s];
                tmp[1][s] = 0.5f * (g[0 * 3 + s] + g[1 * 3 + s] + g[2 * 3 + s]);
                tmp[2][s] = 0.5f * (g[0 * 3 + s] - g[1 * 3 + s] + g[2 * 3 + s]);
                tmp[3][s] = g[2 * 3 + s];
            }
            // Compute U = temp * G^T  (4x3 * 3x4 -> 4x4)
            for (int r = 0; r < 4; ++r) {
                u[r * 4 + 0] = tmp[r][0];
                u[r * 4 + 1] = 0.5f * (tmp[r][0] + tmp[r][1] + tmp[r][2]);
                u[r * 4 + 2] = 0.5f * (tmp[r][0] - tmp[r][1] + tmp[r][2]);
                u[r * 4 + 3] = tmp[r][2];
            }
        }
    }

    // =========================================================================
    // Step 2: For each batch/tile, transform input, multiply, inverse-transform
    // =========================================================================
    #pragma omp parallel if(batch * num_tiles > OmpThresholds::complex())
    {
        // Per-thread workspace
        std::vector<float> V(C_in * 16);    // Transformed input tiles
        std::vector<float> M(C_out * 16);   // Element-wise products

        #pragma omp for
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t th = 0; th < tile_h; ++th) {
                for (int64_t tw = 0; tw < tile_w; ++tw) {
                    const int64_t tile_start_h = th * 2 - pad_h;
                    const int64_t tile_start_w = tw * 2 - pad_w;

                    // ---- Input transform: V = B^T * d * B ----
                    for (int64_t ic = 0; ic < C_in; ++ic) {
                        // Load 4x4 input tile (zero-pad out-of-bounds)
                        float d[4][4];
                        for (int r = 0; r < 4; ++r) {
                            for (int s = 0; s < 4; ++s) {
                                const int64_t ih = tile_start_h + r;
                                const int64_t iw = tile_start_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    d[r][s] = input_data[b * (C_in * H * W) +
                                                         ic * (H * W) +
                                                         ih * W + iw];
                                } else {
                                    d[r][s] = 0.0f;
                                }
                            }
                        }

                        // B^T * d  (left multiply)
                        float temp[4][4];
                        for (int s = 0; s < 4; ++s) {
                            temp[0][s] = d[0][s] - d[2][s];
                            temp[1][s] = d[1][s] + d[2][s];
                            temp[2][s] = -d[1][s] + d[2][s];
                            temp[3][s] = d[1][s] - d[3][s];
                        }
                        // (B^T * d) * B  (right multiply)
                        float* v = V.data() + ic * 16;
                        for (int r = 0; r < 4; ++r) {
                            v[r * 4 + 0] = temp[r][0] - temp[r][2];
                            v[r * 4 + 1] = temp[r][1] + temp[r][2];
                            v[r * 4 + 2] = -temp[r][1] + temp[r][2];
                            v[r * 4 + 3] = temp[r][1] - temp[r][3];
                        }
                    }

                    // ---- Element-wise multiply: M[oc] = sum_ic U[oc,ic] * V[ic] ----
                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        float* m = M.data() + oc * 16;
                        std::memset(m, 0, 16 * sizeof(float));
                        for (int64_t ic = 0; ic < C_in; ++ic) {
                            const float* u = U.data() + (oc * C_in + ic) * 16;
                            const float* v = V.data() + ic * 16;
                            for (int i = 0; i < 16; ++i) {
                                m[i] += u[i] * v[i];
                            }
                        }
                    }

                    // ---- Output transform: out = A^T * M * A ----
                    // A^T = [[1, 1,  1,  0],
                    //        [0, 1, -1, -1]]
                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        const float* m = M.data() + oc * 16;

                        // A^T * M  (2x4 * 4x4 -> 2x4)
                        float tmp2[2][4];
                        for (int s = 0; s < 4; ++s) {
                            tmp2[0][s] = m[0 * 4 + s] + m[1 * 4 + s] + m[2 * 4 + s];
                            tmp2[1][s] = m[1 * 4 + s] - m[2 * 4 + s] - m[3 * 4 + s];
                        }
                        // (A^T * M) * A  (2x4 * 4x2 -> 2x2)
                        float out[2][2];
                        out[0][0] = tmp2[0][0] + tmp2[0][1] + tmp2[0][2];
                        out[0][1] = tmp2[0][1] - tmp2[0][2] - tmp2[0][3];
                        out[1][0] = tmp2[1][0] + tmp2[1][1] + tmp2[1][2];
                        out[1][1] = tmp2[1][1] - tmp2[1][2] - tmp2[1][3];

                        // Store 2x2 output tile (handle boundary for odd output sizes)
                        for (int r = 0; r < 2; ++r) {
                            for (int s = 0; s < 2; ++s) {
                                const int64_t oh = th * 2 + r;
                                const int64_t ow = tw * 2 + s;
                                if (oh < out_h && ow < out_w) {
                                    output_data[b * (C_out * out_h * out_w) +
                                                oc * (out_h * out_w) +
                                                oh * out_w + ow] = out[r][s];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Convenience wrapper that operates on Tensor objects.
/// Allocates the output tensor and returns it.
///
/// @param input  Tensor (N, C_in, H, W), Float32 on CPU.
/// @param weight Tensor (C_out, C_in, 3, 3), Float32 on CPU.
/// @param pad_h  Padding height.
/// @param pad_w  Padding width.
/// @return Output tensor (N, C_out, out_H, out_W).
inline auto winograd_conv2d_f2x3(const Tensor& input, const Tensor& weight,
                                  int pad_h, int pad_w) -> Tensor {
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    const int64_t batch = in_shape[0];
    const int64_t C_in = in_shape[1];
    const int64_t H = in_shape[2];
    const int64_t W = in_shape[3];
    const int64_t C_out = w_shape[0];

    const int64_t out_h = H + 2 * pad_h - 2;  // (H + 2*pad - 3 + 1) for 3x3 stride 1
    const int64_t out_w = W + 2 * pad_w - 2;

    Tensor output({batch, C_out, out_h, out_w}, DType::Float32, input.device());

    winograd_conv2d_f2x3(
        input.data<float>(), weight.data<float>(), output.data<float>(),
        batch, C_in, H, W, C_out, out_h, out_w, pad_h, pad_w);

    return output;
}

/// Check whether the convolution parameters are eligible for Winograd F(4x4,3x3).
/// Requires larger spatial dimensions to benefit from the bigger tile.
inline bool can_use_winograd_f4x3(int64_t kH, int64_t kW,
                                    int64_t stride, int64_t dilation,
                                    int64_t groups,
                                    int64_t out_h, int64_t out_w) {
    return kH == 3 && kW == 3 && stride == 1 && dilation == 1 && groups == 1
           && out_h >= 8 && out_w >= 8;
}

/// Winograd F(4x4, 3x3) convolution.
///
/// Uses 6x6 input tiles to produce 4x4 output tiles, reducing from 9 multiplies
/// per output element (direct) to 2.25 multiplies (36/16 per tile element).
///
/// Transform matrices for F(4, 3):
///   B^T (input transform, 6x6)
///   G   (filter transform, 6x3)
///   A^T (output transform, 4x6)
///
/// @param input_data  Pointer to input data (N, C_in, H, W)
/// @param weight_data Pointer to weight data (C_out, C_in, 3, 3)
/// @param output_data Pointer to output data (N, C_out, out_H, out_W)
inline void winograd_conv2d_f4x3(const float* input_data,
                                  const float* weight_data,
                                  float* output_data,
                                  int64_t batch,
                                  int64_t C_in, int64_t H, int64_t W,
                                  int64_t C_out,
                                  int64_t out_h, int64_t out_w,
                                  int64_t pad_h, int64_t pad_w) {
    // Number of 4x4 output tiles
    const int64_t tile_h = (out_h + 3) / 4;
    const int64_t tile_w = (out_w + 3) / 4;
    const int64_t num_tiles = tile_h * tile_w;

    // =========================================================================
    // Step 1: Pre-transform all filters: U = G * g * G^T  (3x3 -> 6x6)
    // =========================================================================
    // G (6x3) for F(4,3):
    //   [[ 1/4,     0,       0     ],
    //    [-1/6,    -1/6,    -1/6   ],
    //    [-1/6,     1/6,    -1/6   ],
    //    [ 1/24,    1/12,    1/6   ],
    //    [ 1/24,   -1/12,    1/6   ],
    //    [ 0,       0,       1     ]]
    std::vector<float> U(C_out * C_in * 36);

    #pragma omp parallel for collapse(2) if(C_out * C_in > OmpThresholds::medium())
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic = 0; ic < C_in; ++ic) {
            const float* g = weight_data + (oc * C_in + ic) * 9;
            float* u = U.data() + (oc * C_in + ic) * 36;

            // Compute temp = G * g  (6x3 * 3x3 -> 6x3)
            float tmp[6][3];
            for (int s = 0; s < 3; ++s) {
                float g0 = g[0 * 3 + s], g1 = g[1 * 3 + s], g2 = g[2 * 3 + s];
                tmp[0][s] = g0 * 0.25f;
                tmp[1][s] = -(g0 + g1 + g2) / 6.0f;
                tmp[2][s] = (-g0 + g1 - g2) / 6.0f;
                tmp[3][s] = g0 / 24.0f + g1 / 12.0f + g2 / 6.0f;
                tmp[4][s] = g0 / 24.0f - g1 / 12.0f + g2 / 6.0f;
                tmp[5][s] = g2;
            }
            // Compute U = temp * G^T  (6x3 * 3x6 -> 6x6)
            for (int r = 0; r < 6; ++r) {
                float t0 = tmp[r][0], t1 = tmp[r][1], t2 = tmp[r][2];
                u[r * 6 + 0] = t0 * 0.25f;
                u[r * 6 + 1] = -(t0 + t1 + t2) / 6.0f;
                u[r * 6 + 2] = (-t0 + t1 - t2) / 6.0f;
                u[r * 6 + 3] = t0 / 24.0f + t1 / 12.0f + t2 / 6.0f;
                u[r * 6 + 4] = t0 / 24.0f - t1 / 12.0f + t2 / 6.0f;
                u[r * 6 + 5] = t2;
            }
        }
    }

    // =========================================================================
    // Step 2: For each batch/tile, transform input, multiply, inverse-transform
    // =========================================================================
    #pragma omp parallel if(batch * num_tiles > OmpThresholds::complex())
    {
        // Per-thread workspace
        std::vector<float> V(C_in * 36);    // Transformed input tiles (6x6 each)
        std::vector<float> M(C_out * 36);   // Element-wise products

        #pragma omp for
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t th = 0; th < tile_h; ++th) {
                for (int64_t tw = 0; tw < tile_w; ++tw) {
                    const int64_t tile_start_h = th * 4 - pad_h;
                    const int64_t tile_start_w = tw * 4 - pad_w;

                    // ---- Input transform: V = B^T * d * B ----
                    // B^T (6x6) for F(4,3):
                    //   [[ 4,  0, -5,  0,  1,  0],
                    //    [ 0, -4, -4,  1,  1,  0],
                    //    [ 0,  4, -4, -1,  1,  0],
                    //    [ 0, -2, -1,  2,  1,  0],
                    //    [ 0,  2, -1, -2,  1,  0],
                    //    [ 0,  4,  0, -5,  0,  1]]
                    for (int64_t ic = 0; ic < C_in; ++ic) {
                        // Load 6x6 input tile (zero-pad out-of-bounds)
                        float d[6][6];
                        for (int r = 0; r < 6; ++r) {
                            for (int s = 0; s < 6; ++s) {
                                const int64_t ih = tile_start_h + r;
                                const int64_t iw = tile_start_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    d[r][s] = input_data[b * (C_in * H * W) +
                                                         ic * (H * W) +
                                                         ih * W + iw];
                                } else {
                                    d[r][s] = 0.0f;
                                }
                            }
                        }

                        // B^T * d  (6x6 * 6x6 -> 6x6)
                        float temp[6][6];
                        for (int s = 0; s < 6; ++s) {
                            float d0 = d[0][s], d1 = d[1][s], d2 = d[2][s];
                            float d3 = d[3][s], d4 = d[4][s], d5 = d[5][s];
                            temp[0][s] = 4*d0 - 5*d2 + d4;
                            temp[1][s] = -4*d1 - 4*d2 + d3 + d4;
                            temp[2][s] = 4*d1 - 4*d2 - d3 + d4;
                            temp[3][s] = -2*d1 - d2 + 2*d3 + d4;
                            temp[4][s] = 2*d1 - d2 - 2*d3 + d4;
                            temp[5][s] = 4*d1 - 5*d3 + d5;
                        }
                        // (B^T * d) * B  (6x6 * 6x6 -> 6x6)
                        float* v = V.data() + ic * 36;
                        for (int r = 0; r < 6; ++r) {
                            float t0 = temp[r][0], t1 = temp[r][1], t2 = temp[r][2];
                            float t3 = temp[r][3], t4 = temp[r][4], t5 = temp[r][5];
                            v[r * 6 + 0] = 4*t0 - 5*t2 + t4;
                            v[r * 6 + 1] = -4*t1 - 4*t2 + t3 + t4;
                            v[r * 6 + 2] = 4*t1 - 4*t2 - t3 + t4;
                            v[r * 6 + 3] = -2*t1 - t2 + 2*t3 + t4;
                            v[r * 6 + 4] = 2*t1 - t2 - 2*t3 + t4;
                            v[r * 6 + 5] = 4*t1 - 5*t3 + t5;
                        }
                    }

                    // ---- Element-wise multiply: M[oc] = sum_ic U[oc,ic] * V[ic] ----
                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        float* m = M.data() + oc * 36;
                        std::memset(m, 0, 36 * sizeof(float));
                        for (int64_t ic = 0; ic < C_in; ++ic) {
                            const float* u = U.data() + (oc * C_in + ic) * 36;
                            const float* v = V.data() + ic * 36;
                            for (int i = 0; i < 36; ++i) {
                                m[i] += u[i] * v[i];
                            }
                        }
                    }

                    // ---- Output transform: out = A^T * M * A ----
                    // A^T (4x6) for F(4,3):
                    //   [[ 1,  1,  1,  1,  1,  0],
                    //    [ 0,  1, -1,  2, -2,  0],
                    //    [ 0,  1,  1,  4,  4,  0],
                    //    [ 0,  1, -1,  8, -8,  1]]
                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        const float* m = M.data() + oc * 36;

                        // A^T * M  (4x6 * 6x6 -> 4x6)
                        float tmp2[4][6];
                        for (int s = 0; s < 6; ++s) {
                            float m0 = m[0*6+s], m1 = m[1*6+s], m2 = m[2*6+s];
                            float m3 = m[3*6+s], m4 = m[4*6+s], m5 = m[5*6+s];
                            tmp2[0][s] = m0 + m1 + m2 + m3 + m4;
                            tmp2[1][s] = m1 - m2 + 2*m3 - 2*m4;
                            tmp2[2][s] = m1 + m2 + 4*m3 + 4*m4;
                            tmp2[3][s] = m1 - m2 + 8*m3 - 8*m4 + m5;
                        }
                        // (A^T * M) * A  (4x6 * 6x4 -> 4x4)
                        float out[4][4];
                        for (int r = 0; r < 4; ++r) {
                            float t0 = tmp2[r][0], t1 = tmp2[r][1], t2 = tmp2[r][2];
                            float t3 = tmp2[r][3], t4 = tmp2[r][4], t5 = tmp2[r][5];
                            out[r][0] = t0 + t1 + t2 + t3 + t4;
                            out[r][1] = t1 - t2 + 2*t3 - 2*t4;
                            out[r][2] = t1 + t2 + 4*t3 + 4*t4;
                            out[r][3] = t1 - t2 + 8*t3 - 8*t4 + t5;
                        }

                        // Store 4x4 output tile (handle boundary for non-multiple-of-4 output sizes)
                        for (int r = 0; r < 4; ++r) {
                            for (int s = 0; s < 4; ++s) {
                                const int64_t oh = th * 4 + r;
                                const int64_t ow = tw * 4 + s;
                                if (oh < out_h && ow < out_w) {
                                    output_data[b * (C_out * out_h * out_w) +
                                                oc * (out_h * out_w) +
                                                oh * out_w + ow] = out[r][s];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Convenience wrapper for F(4x4, 3x3) Winograd.
inline auto winograd_conv2d_f4x3(const Tensor& input, const Tensor& weight,
                                  int pad_h, int pad_w) -> Tensor {
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    const int64_t batch = in_shape[0];
    const int64_t C_in = in_shape[1];
    const int64_t H = in_shape[2];
    const int64_t W = in_shape[3];
    const int64_t C_out = w_shape[0];

    const int64_t out_h = H + 2 * pad_h - 2;
    const int64_t out_w = W + 2 * pad_w - 2;

    Tensor output({batch, C_out, out_h, out_w}, DType::Float32, input.device());

    winograd_conv2d_f4x3(
        input.data<float>(), weight.data<float>(), output.data<float>(),
        batch, C_in, H, W, C_out, out_h, out_w, pad_h, pad_w);

    return output;
}

} // namespace tenzor::cpu
