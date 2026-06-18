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

#ifdef TENZOR_USE_MKL
#include <mkl.h>
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
/// Templated on the element/accumulator type T (Float32 or Float64) so the same
/// canonical F(2,3) transforms are the single source of truth for every CPU
/// dtype (Float64 dispatches here directly; Float16/BFloat16 widen to Float32
/// and dispatch here as well). All arithmetic is performed in T.
template <typename T>
inline void winograd_conv2d_f2x3(const T* input_data,
                                  const T* weight_data,
                                  T* output_data,
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
    std::vector<T> U(C_out * C_in * 16);

    #pragma omp parallel for collapse(2) if(C_out * C_in > OmpThresholds::medium())
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic = 0; ic < C_in; ++ic) {
            const T* g = weight_data + (oc * C_in + ic) * 9;
            T* u = U.data() + (oc * C_in + ic) * 16;

            // Compute temp = G * g  (4x3 * 3x3 -> 4x3)
            T tmp[4][3];
            for (int s = 0; s < 3; ++s) {
                tmp[0][s] = g[0 * 3 + s];
                tmp[1][s] = static_cast<T>(0.5) * (g[0 * 3 + s] + g[1 * 3 + s] + g[2 * 3 + s]);
                tmp[2][s] = static_cast<T>(0.5) * (g[0 * 3 + s] - g[1 * 3 + s] + g[2 * 3 + s]);
                tmp[3][s] = g[2 * 3 + s];
            }
            // Compute U = temp * G^T  (4x3 * 3x4 -> 4x4)
            for (int r = 0; r < 4; ++r) {
                u[r * 4 + 0] = tmp[r][0];
                u[r * 4 + 1] = static_cast<T>(0.5) * (tmp[r][0] + tmp[r][1] + tmp[r][2]);
                u[r * 4 + 2] = static_cast<T>(0.5) * (tmp[r][0] - tmp[r][1] + tmp[r][2]);
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
        std::vector<T> V(C_in * 16);    // Transformed input tiles
        std::vector<T> M(C_out * 16);   // Element-wise products

        #pragma omp for
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t th = 0; th < tile_h; ++th) {
                for (int64_t tw = 0; tw < tile_w; ++tw) {
                    const int64_t tile_start_h = th * 2 - pad_h;
                    const int64_t tile_start_w = tw * 2 - pad_w;

                    // ---- Input transform: V = B^T * d * B ----
                    for (int64_t ic = 0; ic < C_in; ++ic) {
                        // Load 4x4 input tile (zero-pad out-of-bounds)
                        T d[4][4];
                        for (int r = 0; r < 4; ++r) {
                            for (int s = 0; s < 4; ++s) {
                                const int64_t ih = tile_start_h + r;
                                const int64_t iw = tile_start_w + s;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    d[r][s] = input_data[b * (C_in * H * W) +
                                                         ic * (H * W) +
                                                         ih * W + iw];
                                } else {
                                    d[r][s] = T{};
                                }
                            }
                        }

                        // B^T * d  (left multiply)
                        T temp[4][4];
                        for (int s = 0; s < 4; ++s) {
                            temp[0][s] = d[0][s] - d[2][s];
                            temp[1][s] = d[1][s] + d[2][s];
                            temp[2][s] = -d[1][s] + d[2][s];
                            temp[3][s] = d[1][s] - d[3][s];
                        }
                        // (B^T * d) * B  (right multiply)
                        T* v = V.data() + ic * 16;
                        for (int r = 0; r < 4; ++r) {
                            v[r * 4 + 0] = temp[r][0] - temp[r][2];
                            v[r * 4 + 1] = temp[r][1] + temp[r][2];
                            v[r * 4 + 2] = -temp[r][1] + temp[r][2];
                            v[r * 4 + 3] = temp[r][1] - temp[r][3];
                        }
                    }

                    // ---- Element-wise multiply: M[oc] = sum_ic U[oc,ic] * V[ic] ----
                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        T* m = M.data() + oc * 16;
                        for (int i = 0; i < 16; ++i) {
                            m[i] = T{};
                        }
                        for (int64_t ic = 0; ic < C_in; ++ic) {
                            const T* u = U.data() + (oc * C_in + ic) * 16;
                            const T* v = V.data() + ic * 16;
                            for (int i = 0; i < 16; ++i) {
                                m[i] += u[i] * v[i];
                            }
                        }
                    }

                    // ---- Output transform: out = A^T * M * A ----
                    // A^T = [[1, 1,  1,  0],
                    //        [0, 1, -1, -1]]
                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        const T* m = M.data() + oc * 16;

                        // A^T * M  (2x4 * 4x4 -> 2x4)
                        T tmp2[2][4];
                        for (int s = 0; s < 4; ++s) {
                            tmp2[0][s] = m[0 * 4 + s] + m[1 * 4 + s] + m[2 * 4 + s];
                            tmp2[1][s] = m[1 * 4 + s] - m[2 * 4 + s] - m[3 * 4 + s];
                        }
                        // (A^T * M) * A  (2x4 * 4x2 -> 2x2)
                        T out[2][2];
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

/// Winograd F(4x4, 3x3) convolution with MKL SGEMM acceleration.
///
/// Uses 6x6 input tiles to produce 4x4 output tiles, reducing from 9 multiplies
/// per output element (direct) to 2.25 multiplies (36/16 per tile element).
///
/// The key optimization over the naive per-tile approach: all input tiles for a
/// batch element are transformed first, then the element-wise multiply in the
/// transform domain is expressed as 36 independent GEMM operations:
///   For each position p in [0, 36):
///     M_p[C_out, num_tiles] = U_p[C_out, C_in] * V_p[C_in, num_tiles]
/// This leverages MKL's highly optimized SGEMM for the compute-bound step.
///
/// Data layout for batched GEMM:
///   U_scatter[p][oc * C_in + ic] = U[(oc * C_in + ic) * 36 + p]  (scattered by position)
///   V_scatter[p][ic * num_tiles + tile] = V[ic][tile * 36 + p]
///   M_scatter[p][oc * num_tiles + tile] = result
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
    // Then scatter into position-major layout: U_pos[36][C_out * C_in]
    // so that U_pos[p] is a contiguous (C_out x C_in) matrix for GEMM.
    std::vector<float> U_pos(36 * C_out * C_in);

    #pragma omp parallel for collapse(2) if(C_out * C_in > OmpThresholds::medium())
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t ic = 0; ic < C_in; ++ic) {
            const float* g = weight_data + (oc * C_in + ic) * 9;

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
            // Compute U = temp * G^T  (6x3 * 3x6 -> 6x6) and scatter to position-major
            for (int r = 0; r < 6; ++r) {
                float t0 = tmp[r][0], t1 = tmp[r][1], t2 = tmp[r][2];
                float u[6];
                u[0] = t0 * 0.25f;
                u[1] = -(t0 + t1 + t2) / 6.0f;
                u[2] = (-t0 + t1 - t2) / 6.0f;
                u[3] = t0 / 24.0f + t1 / 12.0f + t2 / 6.0f;
                u[4] = t0 / 24.0f - t1 / 12.0f + t2 / 6.0f;
                u[5] = t2;
                for (int s = 0; s < 6; ++s) {
                    int p = r * 6 + s;
                    // Row-major (C_out x C_in) for each position p
                    U_pos[p * (C_out * C_in) + oc * C_in + ic] = u[s];
                }
            }
        }
    }

    // =========================================================================
    // Step 2: For each batch element, transform all tiles, batched GEMM, inverse transform
    // =========================================================================
    // V_pos layout: [36][C_in * num_tiles] — position-major, then (C_in x num_tiles)
    // M_pos layout: [36][C_out * num_tiles] — position-major, then (C_out x num_tiles)

    #pragma omp parallel if(batch > 1 || num_tiles * C_in > OmpThresholds::complex())
    {
        std::vector<float> V_pos(36 * C_in * num_tiles);
        std::vector<float> M_pos(36 * C_out * num_tiles);

        #pragma omp for
        for (int64_t b = 0; b < batch; ++b) {
            // ---- Transform all input tiles for this batch element ----
            for (int64_t th = 0; th < tile_h; ++th) {
                for (int64_t tw = 0; tw < tile_w; ++tw) {
                    const int64_t tile_idx = th * tile_w + tw;
                    const int64_t tile_start_h = th * 4 - pad_h;
                    const int64_t tile_start_w = tw * 4 - pad_w;

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
                        // (B^T * d) * B  and scatter to position-major layout
                        for (int r = 0; r < 6; ++r) {
                            float t0 = temp[r][0], t1 = temp[r][1], t2 = temp[r][2];
                            float t3 = temp[r][3], t4 = temp[r][4], t5 = temp[r][5];
                            float v[6];
                            v[0] = 4*t0 - 5*t2 + t4;
                            v[1] = -4*t1 - 4*t2 + t3 + t4;
                            v[2] = 4*t1 - 4*t2 - t3 + t4;
                            v[3] = -2*t1 - t2 + 2*t3 + t4;
                            v[4] = 2*t1 - t2 - 2*t3 + t4;
                            v[5] = 4*t1 - 5*t3 + t5;
                            for (int s = 0; s < 6; ++s) {
                                int p = r * 6 + s;
                                // (C_in x num_tiles) row-major
                                V_pos[p * (C_in * num_tiles) + ic * num_tiles + tile_idx] = v[s];
                            }
                        }
                    }
                }
            }

            // ---- Batched GEMM: For each position p, M_p = U_p * V_p ----
            // M_p[C_out, num_tiles] = U_p[C_out, C_in] * V_p[C_in, num_tiles]
#ifdef TENZOR_USE_MKL
            for (int p = 0; p < 36; ++p) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(C_out),      // M
                            static_cast<int>(num_tiles),   // N
                            static_cast<int>(C_in),        // K
                            1.0f,                          // alpha
                            U_pos.data() + p * (C_out * C_in),  // A
                            static_cast<int>(C_in),        // lda
                            V_pos.data() + p * (C_in * num_tiles),  // B
                            static_cast<int>(num_tiles),   // ldb
                            0.0f,                          // beta
                            M_pos.data() + p * (C_out * num_tiles),  // C
                            static_cast<int>(num_tiles));  // ldc
            }
#else
            // Fallback: naive GEMM for each position
            for (int p = 0; p < 36; ++p) {
                const float* u_p = U_pos.data() + p * (C_out * C_in);
                const float* v_p = V_pos.data() + p * (C_in * num_tiles);
                float* m_p = M_pos.data() + p * (C_out * num_tiles);
                std::memset(m_p, 0, C_out * num_tiles * sizeof(float));
                for (int64_t oc = 0; oc < C_out; ++oc) {
                    for (int64_t ic = 0; ic < C_in; ++ic) {
                        float u_val = u_p[oc * C_in + ic];
                        for (int64_t t = 0; t < num_tiles; ++t) {
                            m_p[oc * num_tiles + t] += u_val * v_p[ic * num_tiles + t];
                        }
                    }
                }
            }
#endif

            // ---- Inverse transform: gather from position-major and apply A^T * M * A ----
            for (int64_t th = 0; th < tile_h; ++th) {
                for (int64_t tw = 0; tw < tile_w; ++tw) {
                    const int64_t tile_idx = th * tile_w + tw;

                    for (int64_t oc = 0; oc < C_out; ++oc) {
                        // Gather 6x6 M values for this (oc, tile) from position-major layout
                        float m[6][6];
                        for (int r = 0; r < 6; ++r) {
                            for (int s = 0; s < 6; ++s) {
                                int p = r * 6 + s;
                                m[r][s] = M_pos[p * (C_out * num_tiles) + oc * num_tiles + tile_idx];
                            }
                        }

                        // A^T * M  (4x6 * 6x6 -> 4x6)
                        float tmp2[4][6];
                        for (int s = 0; s < 6; ++s) {
                            float m0 = m[0][s], m1 = m[1][s], m2 = m[2][s];
                            float m3 = m[3][s], m4 = m[4][s], m5 = m[5][s];
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

                        // Store 4x4 output tile (handle boundary)
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
