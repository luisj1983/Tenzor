#include "vulkan_ops_common.hpp"
#include "tenzor/sparse/sparse_ops.hpp"   // complex sparse CPU fallback (F104)
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include <limits>

namespace tenzor {

static constexpr int64_t MAX_SMALL_LINALG_SIZE = 32;  // single-workgroup (shared-mem [32][32]) shader limit; larger N uses the tiled path
static constexpr int64_t TILED_BLOCK_SIZE = 32;        // panel width for blocked algorithms
// The blocked LU / Cholesky / bidiag panel shaders declare a fixed-size shared
// panel (panel[256][32] / col_data[256]) and index it by row up to n-1, so n
// must not exceed 256 or the load/store loops read/write out of bounds (UB).
static constexpr int64_t MAX_BLOCKED_LINALG_SIZE = 256;

// AUTOGRAD-R044: Vulkan linalg supports Float32/Float64/Float16 natively
// (BFloat16 is upcast to Float32 by the caller before reaching these
// dispatch functions). Every dtype branch below is a plain
// `is_f64 ? ... : is_f16 ? ... : "..._f32_shader"` ternary, so anything
// that is NOT f64/f16 — including Complex64/Complex128 — silently falls
// through to the Float32 shader instead of being rejected, unlike ROCm's
// validate_linalg_dtype (src/backends/rocm/kernels/linalg.hip.cpp) which
// throws a named error. Currently unreachable in practice (complex inputs
// are shuttled to CPU by try_gpu_dispatch before any GPU backend is
// reached, src/ops/linalg.cpp), but a defense-in-depth gap if that routing
// ever changes. Mirror ROCm's guard here.
static void validate_linalg_dtype(const Tensor& t, const std::string& op_name) {
    auto dt = t.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 && dt != DType::Float16) {
        throw std::invalid_argument(
            "linalg::" + op_name + ": unsupported dtype " +
            std::string(dtype_name(dt)) +
            " on Vulkan. Supported: Float32, Float64, Float16.");
    }
}

// =============================================================================
// Linear Algebra Operations — Native Vulkan shaders for small matrices,
// tiled blocked algorithms for medium matrices (33-256)
// =============================================================================

// ---------------------------------------------------------------------------
// Blocked LU decomposition: panel factorization + trailing GEMM update
// Modifies A in-place to contain L (below diagonal) and U (on/above diagonal).
// pivots is [batch_size, n] Int32 tensor storing row pivot indices.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedLU(Tensor& A, Tensor& pivots, int64_t n,
                                  int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    if (n > MAX_BLOCKED_LINALG_SIZE) {
        throw std::runtime_error(
            "Vulkan blocked LU: matrix dimension " + std::to_string(n) +
            " exceeds the supported maximum of " + std::to_string(MAX_BLOCKED_LINALG_SIZE) +
            " (linalg_lu_panel uses a fixed shared panel[256][32]).");
    }
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t piv_size = static_cast<size_t>(batch_size) * n * sizeof(int32_t);

    for (int64_t col_start = 0; col_start < n; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - col_start);

        // --- Panel factorization ---
        // One workgroup (256 threads) per batch element, folded into a
        // single dispatch with the batch as the X grid dimension (batch
        // index read from gl_WorkGroupID.x in the shader) instead of one
        // host-side command-buffer submission per batch element per
        // column-panel. That previously made this O(batch_size * n/32)
        // separate submissions; now it is O(n/32).
        {
            std::string shader = is_f64 ? "linalg_lu_panel_f64" : is_f16 ? "linalg_lu_panel_f16" : "linalg_lu_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, pivots.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, piv_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);  // batch_size workgroups (256 threads each)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Trailing matrix update ---
        // Fold the batch dimension into the Z grid axis (batch index read
        // from gl_WorkGroupID.z in the shader) — one dispatch instead of one
        // submission per batch element per column-panel.
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;
            uint32_t tile_count = static_cast<uint32_t>((trail_size + 31) / 32);

            std::string shader = is_f64 ? "linalg_lu_update_f64" : is_f16 ? "linalg_lu_update_f16" : "linalg_lu_update";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t block_size;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.block_size = static_cast<uint32_t>(panel_cols);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, tile_count, tile_count, static_cast<uint32_t>(batch_size));  // 16x16 threads per tile
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked Cholesky: panel factorization + trailing SYRK update
// Modifies A in-place to contain L (lower triangle).
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedCholesky(Tensor& A, int64_t n, int64_t batch_size, int32_t device_id,
                                        bool is_f64, bool is_f16, Tensor& error_flag) {
    if (n > MAX_BLOCKED_LINALG_SIZE) {
        throw std::runtime_error(
            "Vulkan blocked Cholesky: matrix dimension " + std::to_string(n) +
            " exceeds the supported maximum of " + std::to_string(MAX_BLOCKED_LINALG_SIZE) +
            " (linalg_cholesky_tiled uses a fixed shared panel[256][32]).");
    }
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t flag_size = sizeof(uint32_t);

    for (int64_t col_start = 0; col_start < n; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - col_start);

        // --- Panel factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_cholesky_tiled_f64" : is_f16 ? "linalg_cholesky_tiled_f16" : "linalg_cholesky_tiled";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, error_flag.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, flag_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Trailing SYRK update ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;
            uint32_t tile_count = static_cast<uint32_t>((trail_size + 31) / 32);

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_cholesky_update_f64" : is_f16 ? "linalg_cholesky_update_f16" : "linalg_cholesky_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t block_size;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.block_size = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                // Dispatch tile_count x tile_count grid; shader skips upper triangle tiles
                vkCmdDispatch(cmd, tile_count, tile_count, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked QR: panel Householder factorization + apply reflections to trailing
// Modifies A in-place (Householder vectors below diagonal, R on/above diagonal).
// tau is [batch_size, n] storing Householder scalars.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedQR(Tensor& A, Tensor& tau, int64_t m, int64_t n,
                                  int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * m * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;
    int64_t k = std::min(m, n);

    for (int64_t col_start = 0; col_start < k; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, k - col_start);

        // --- Panel factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_qr_panel_f64" : is_f16 ? "linalg_qr_panel_f16" : "linalg_qr_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t m;
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.m = static_cast<uint32_t>(m);
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply reflections to trailing columns ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_cols = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_qr_update_f64" : is_f16 ? "linalg_qr_update_f16" : "linalg_qr_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t m;
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.m = static_cast<uint32_t>(m);
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                // One workgroup per trailing column
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_cols), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

void VulkanBackend::runBlockedBidiag(Tensor& A, Tensor& tau_l, Tensor& tau_r, int64_t n,
                                      int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    if (n > MAX_BLOCKED_LINALG_SIZE) {
        throw std::runtime_error(
            "Vulkan blocked bidiagonalization: matrix dimension " + std::to_string(n) +
            " exceeds the supported maximum of " + std::to_string(MAX_BLOCKED_LINALG_SIZE) +
            " (linalg_bidiag_panel uses a fixed shared col_data[256]).");
    }
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;

    for (int64_t col_start = 0; col_start < n - 1; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - 1 - col_start);

        // --- Panel bidiagonalization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_bidiag_panel_f64"
                               : is_f16 ? "linalg_bidiag_panel_f16"
                                         : "linalg_bidiag_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size, tau_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply reflections to trailing submatrix ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_bidiag_update_f64"
                                   : is_f16 ? "linalg_bidiag_update_f16"
                                             : "linalg_bidiag_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, tau_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                // One workgroup per trailing row/column
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_size), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side singularity check for already-factorized LU factors consumed by
// the linalg_trsm shader (dispatchLinalgInv / dispatchLinalgSolve tiled
// paths and dispatchLinalgLUSolve). linalg_trsm.comp (kernels/linalg_trsm.comp)
// divides by the LU diagonal with no epsilon guard and no error-reporting
// path back to the host, so a singular matrix would otherwise silently
// produce Inf/NaN instead of throwing. Matches the CPU/LAPACK convention
// (getrf/getrs info != 0 for an exactly-zero pivot). `lu_cont` must already
// be a contiguous [batch_size, n, n] buffer on-device.
// ---------------------------------------------------------------------------
void VulkanBackend::checkLuNonSingular(const Tensor& lu_cont, int64_t n, int64_t batch_size,
                                        bool is_f64, const std::string& op_name) {
    synchronize(lu_cont.device().index);
    Tensor lu_host = lu_cont.to(Device::cpu());
    auto check_batch = [&](auto* lu_data) {
        for (int64_t b = 0; b < batch_size; ++b) {
            auto* mat = lu_data + b * n * n;
            for (int64_t i = 0; i < n; ++i) {
                if (mat[i * n + i] == 0) {
                    throw std::runtime_error(
                        "linalg::" + op_name + ": singular matrix (zero pivot in LU factorization)");
                }
            }
        }
    };
    if (is_f64) {
        check_batch(lu_host.data<double>());
    } else {
        check_batch(lu_host.data<float>());
    }
}

auto VulkanBackend::dispatchLinalgDet(const Tensor& input) -> Tensor {
    // No native BFloat16 shader: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgDet(dispatchCast(input.contiguous(), DType::Float32));
        return dispatchCast(res, DType::BFloat16);
    }
    validate_linalg_dtype(input, "det");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) {
        throw std::runtime_error("linalg.det: input must be at least 2D");
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        throw std::runtime_error("linalg.det: input must be square");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);

    // Compute batch size
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Output shape: batch dims (no matrix dims)
    std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
    if (out_shape.empty()) out_shape = {1};  // Scalar output
    Tensor output(out_shape, input.dtype(), input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        std::string shader = is_f64 ? "linalg_det_f64" : is_f16 ? "linalg_det_f16" : "linalg_det";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
        size_t in_numel = batch_size * n * n;
        size_t out_numel = batch_size;
        size_t in_size = is_f16 ? f16_buf(in_numel) : in_numel * elem_size;
        size_t out_size = is_f16 ? f16_buf(out_numel) : out_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        // Tiled path (n > MAX_SMALL_LINALG_SIZE, i.e. n > 32): blocked LU, then compute det from diagonal
        Tensor A = dispatchClone(input.contiguous());
        Tensor pivots({batch_size, n}, DType::Int32, input.device());

        runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);

        // Compute determinant from LU diagonal + pivot sign entirely on GPU
        std::string det_shader = is_f64 ? "linalg_det_from_lu_f64" : is_f16 ? "linalg_det_from_lu_f16" : "linalg_det_from_lu";
        auto* det_pipeline = getPipeline(det_shader, device_id);

        struct DetPC {
            uint32_t n_dim;
            uint32_t batch_cnt;
            uint32_t lda;
        } det_pc;
        det_pc.n_dim = static_cast<uint32_t>(n);
        det_pc.batch_cnt = static_cast<uint32_t>(batch_size);
        det_pc.lda = static_cast<uint32_t>(n);

        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
        size_t lu_numel = batch_size * n * n;
        size_t det_numel = batch_size;
        size_t lu_buf_size = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;
        size_t piv_buf_size = batch_size * n * sizeof(int32_t);
        size_t det_buf_size = is_f16 ? f16_buf(det_numel) : det_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> det_bindings = {
            {0, A.data_ptr()}, {1, pivots.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> det_sizes = {lu_buf_size, piv_buf_size, det_buf_size};
        VkDescriptorSet det_ds = allocateAndWriteDescriptorSet(device_id, det_pipeline, det_bindings, det_sizes);

        VkCommandBuffer det_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(det_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, det_pipeline->pipeline());
        vkCmdBindDescriptorSets(det_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               det_pipeline->layout(), 0, 1, &det_ds, 0, nullptr);
        vkCmdPushConstants(det_cmd, det_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(det_pc), &det_pc);
        uint32_t det_groups = (static_cast<uint32_t>(batch_size) + 255) / 256;
        vkCmdDispatch(det_cmd, det_groups, 1, 1);
        insertComputeOnlyBarrier(det_cmd);
        endSingleTimeCommands(det_cmd, device_id);
    }

    // If input was truly scalar (0-d batch), squeeze
    if (ndim == 2) {
        return output.reshape({});
    }
    return output;
}

auto VulkanBackend::dispatchLinalgInv(const Tensor& input) -> Tensor {
    // No native BFloat16 shader: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgInv(dispatchCast(input.contiguous(), DType::Float32));
        return dispatchCast(res, DType::BFloat16);
    }
    validate_linalg_dtype(input, "inv");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) {
        throw std::runtime_error("linalg.inv: input must be at least 2D");
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        throw std::runtime_error("linalg.inv: input must be square");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

        std::string shader = is_f64 ? "linalg_inv_f64" : is_f16 ? "linalg_inv_f16" : "linalg_inv";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t mat_numel = batch_size * n * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, mat_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    // Tiled path (n > MAX_SMALL_LINALG_SIZE, i.e. n > 32): LU factorize on GPU, then backsolve on GPU via TRSM shader
    Tensor A = dispatchClone(input.contiguous());
    Tensor pivots({batch_size, n}, DType::Int32, input.device());

    runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);
    checkLuNonSingular(A, n, batch_size, is_f64, "inv");

    // Create identity matrix as RHS: solve LU * X = P * I => X = A^{-1}
    // dispatchEye creates a single n x n identity on GPU; expand for batches
    Tensor identity = dispatchEye(n, n, input.dtype(), input.device());
    if (batch_size > 1) {
        // Repeat identity for each batch element
        std::vector<Tensor> eyes(batch_size, identity);
        identity = dispatchStack(eyes, 0);
    } else {
        identity = identity.unsqueeze(0);
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    // Dispatch TRSM shader: solve LU * X = P * I
    std::string trsm_shader = is_f64 ? "linalg_trsm_f64" : is_f16 ? "linalg_trsm_f16" : "linalg_trsm";
    auto* trsm_pipeline = getPipeline(trsm_shader, device_id);

    struct TrsmPC {
        uint32_t n_dim;
        uint32_t nrhs;
        uint32_t lda;
        uint32_t ldb;
        uint32_t batch_cnt;
    } trsm_pc;
    trsm_pc.n_dim = static_cast<uint32_t>(n);
    trsm_pc.nrhs = static_cast<uint32_t>(n);
    trsm_pc.lda = static_cast<uint32_t>(n);
    trsm_pc.ldb = static_cast<uint32_t>(n);
    trsm_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    size_t lu_numel = batch_size * n * n;
    size_t lu_sz = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;
    size_t piv_sz = batch_size * n * sizeof(int32_t);
    size_t mat_sz = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;

    // Note: identity is read-only input, output receives solution
    auto identity_cont = identity.contiguous();
    std::vector<std::pair<uint32_t, const void*>> trsm_bindings = {
        {0, A.data_ptr()}, {1, pivots.data_ptr()},
        {2, identity_cont.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> trsm_sizes = {lu_sz, piv_sz, mat_sz, mat_sz};
    VkDescriptorSet trsm_ds = allocateAndWriteDescriptorSet(device_id, trsm_pipeline, trsm_bindings, trsm_sizes);

    VkCommandBuffer trsm_cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trsm_pipeline->pipeline());
    vkCmdBindDescriptorSets(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           trsm_pipeline->layout(), 0, 1, &trsm_ds, 0, nullptr);
    vkCmdPushConstants(trsm_cmd, trsm_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(trsm_pc), &trsm_pc);
    vkCmdDispatch(trsm_cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(trsm_cmd);
    endSingleTimeCommands(trsm_cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchLinalgSolve(const Tensor& a, const Tensor& b) -> Tensor {
    // No native BFloat16 shader: widen to Float32, compute, narrow back.
    if (a.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgSolve(dispatchCast(a.contiguous(), DType::Float32),
                                        dispatchCast(b.contiguous(), DType::Float32));
        return dispatchCast(res, DType::BFloat16);
    }
    validate_linalg_dtype(a, "solve");
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    int64_t a_ndim = static_cast<int64_t>(a_shape.size());

    if (a_ndim < 2) {
        throw std::runtime_error("linalg.solve: A must be at least 2D");
    }

    int64_t n = a_shape[a_ndim - 1];
    if (a_shape[a_ndim - 2] != n) {
        throw std::runtime_error("linalg.solve: A must be square");
    }

    int32_t device_id = a.device().index;
    bool is_f64 = (a.dtype() == DType::Float64);
    bool is_f16 = (a.dtype() == DType::Float16);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < a_ndim - 2; ++i) batch_size *= a_shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), a.dtype(), a.device());

        std::string shader = is_f64 ? "linalg_solve_f64" : is_f16 ? "linalg_solve_f16" : "linalg_solve";
        auto* pipeline = getPipeline(shader, device_id);

        // Extract nrhs from B's trailing dim (B shape is (..., N, nrhs) or
        // (..., N) for the nrhs=1 case).
        int64_t b_ndim_small = static_cast<int64_t>(b_shape.size());
        int64_t nrhs_small = (b_ndim_small >= 2 &&
                              b_shape[b_ndim_small - 2] == n)
                                 ? b_shape[b_ndim_small - 1]
                                 : 1;

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
            uint32_t nrhs;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.nrhs = static_cast<uint32_t>(nrhs_small);

        auto a_cont = a.contiguous();
        auto b_cont = b.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t a_numel = batch_size * n * n;
        size_t b_numel = batch_size * n * nrhs_small;
        size_t a_size = is_f16 ? f16_buf(a_numel) : a_numel * elem_size;
        size_t b_size = is_f16 ? f16_buf(b_numel) : b_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, a_cont.data_ptr()}, {1, b_cont.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes = {a_size, b_size, b_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    // Tiled path (n > MAX_SMALL_LINALG_SIZE, i.e. n > 32): LU factorize on GPU, backsolve on GPU via TRSM shader
    Tensor A = dispatchClone(a.contiguous());
    Tensor pivots({batch_size, n}, DType::Int32, a.device());

    runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);
    checkLuNonSingular(A, n, batch_size, is_f64, "solve");

    // Determine nrhs from b shape. Mirror the small-matrix path's
    // disambiguation: a 2D-or-higher RHS is treated as a matrix (..., N, nrhs)
    // only when its second-to-last dim equals N; otherwise it is a batched
    // vector RHS (..., N) with nrhs == 1. Without this guard a batched vector
    // RHS shaped (batch, N) with batch != N would wrongly yield nrhs == N and
    // make the TRSM shader over-read the b buffer.
    int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2 && b_shape[b_ndim - 2] == n)
                       ? b_shape[b_ndim - 1]
                       : 1;
    int64_t ldb = nrhs;

    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), a.dtype(), a.device());

    // Dispatch TRSM shader: solve LU * X = P * B
    std::string trsm_shader = is_f64 ? "linalg_trsm_f64" : is_f16 ? "linalg_trsm_f16" : "linalg_trsm";
    auto* trsm_pipeline = getPipeline(trsm_shader, device_id);

    struct TrsmPC {
        uint32_t n_dim;
        uint32_t nrhs_cnt;
        uint32_t lda;
        uint32_t ldb_val;
        uint32_t batch_cnt;
    } trsm_pc;
    trsm_pc.n_dim = static_cast<uint32_t>(n);
    trsm_pc.nrhs_cnt = static_cast<uint32_t>(nrhs);
    trsm_pc.lda = static_cast<uint32_t>(n);
    trsm_pc.ldb_val = static_cast<uint32_t>(ldb);
    trsm_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    size_t lu_numel = batch_size * n * n;
    size_t b_numel = batch_size * n * nrhs;
    size_t lu_sz = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;
    size_t piv_sz = batch_size * n * sizeof(int32_t);
    size_t b_sz = is_f16 ? f16_buf(b_numel) : b_numel * elem_size;

    auto b_cont = b.contiguous();
    std::vector<std::pair<uint32_t, const void*>> trsm_bindings = {
        {0, A.data_ptr()}, {1, pivots.data_ptr()},
        {2, b_cont.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> trsm_sizes = {lu_sz, piv_sz, b_sz, b_sz};
    VkDescriptorSet trsm_ds = allocateAndWriteDescriptorSet(device_id, trsm_pipeline, trsm_bindings, trsm_sizes);

    VkCommandBuffer trsm_cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trsm_pipeline->pipeline());
    vkCmdBindDescriptorSets(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           trsm_pipeline->layout(), 0, 1, &trsm_ds, 0, nullptr);
    vkCmdPushConstants(trsm_cmd, trsm_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(trsm_pc), &trsm_pc);
    vkCmdDispatch(trsm_cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(trsm_cmd);
    endSingleTimeCommands(trsm_cmd, device_id);

    return output;
}

// ============================================================================
// Cholesky Decomposition (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgCholesky(const Tensor& input, bool upper) -> Tensor {
    // No native BFloat16 shader: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgCholesky(dispatchCast(input.contiguous(), DType::Float32), upper);
        return dispatchCast(res, DType::BFloat16);
    }
    validate_linalg_dtype(input, "cholesky");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.cholesky: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.cholesky: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Set to 1 (via atomicOr in the shader) when the input is not positive
    // definite. Both the small and tiled paths below wire into this same
    // flag/throw mechanism so they behave identically on non-PD input,
    // matching CPU/CUDA/ROCm's LAPACK/rocSOLVER info-code check.
    Tensor error_flag = dispatchZeros({1}, DType::Int32, input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

        std::string shader = is_f64 ? "linalg_cholesky_f64" : is_f16 ? "linalg_cholesky_f16" : "linalg_cholesky";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; uint32_t upper; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.upper = upper ? 1 : 0;

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
        size_t mat_numel = batch_size * n * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
        size_t flag_size = sizeof(uint32_t);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}, {2, error_flag.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, mat_size, flag_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        synchronize(device_id);
        if (error_flag.to(Device::cpu()).data<int32_t>()[0] != 0) {
            throw std::runtime_error("linalg::cholesky: factorization failed (not positive definite)");
        }

        return output;
    }

    // Tiled path (n > MAX_SMALL_LINALG_SIZE, i.e. n > 32): blocked Cholesky factorization
    Tensor A = dispatchClone(input.contiguous());

    runBlockedCholesky(A, n, batch_size, device_id, is_f64, is_f16, error_flag);

    synchronize(device_id);
    if (error_flag.to(Device::cpu()).data<int32_t>()[0] != 0) {
        throw std::runtime_error("linalg::cholesky: factorization failed (not positive definite)");
    }

    // Zero the upper triangle (Cholesky produces L in lower triangle)
    // Use dispatchTriuTril to extract lower triangle
    Tensor L = dispatchTriuTril("tril", A, 0);

    if (upper) {
        // If upper requested, transpose L -> U
        return dispatchTranspose(L, ndim - 2, ndim - 1);
    }
    return L;
}

// ============================================================================
// QR Decomposition (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgQR(const Tensor& input) -> std::vector<Tensor> {
    // No native BFloat16 shader: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgQR(dispatchCast(input.contiguous(), DType::Float32));
        return { dispatchCast(res[0], DType::BFloat16), dispatchCast(res[1], DType::BFloat16) };
    }
    validate_linalg_dtype(input, "qr");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.qr: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Float16: compute in Float32 and narrow (the global QR shader is f32/f64 only).
    if (is_f16) {
        auto res = dispatchLinalgQR(dispatchCast(input.contiguous(), DType::Float32));
        return { dispatchCast(res[0], DType::Float16), dispatchCast(res[1], DType::Float16) };
    }

    // Householder QR via a global-memory shader correct for any m,n. The old
    // shared-memory shader was capped at 32x32 but dispatched up to 128 (corrupting
    // 32<N<=128), and the blocked path produced NaNs. Returns full Q (m×m), R (m×n),
    // then reduces to (m,k)/(k,n) to match CPU/LAPACK reduced-QR semantics.
    {
        int64_t k = std::min(m, n);
        const size_t esz = is_f64 ? 8u : 4u;

        std::vector<int64_t> q_full_shape(shape.begin(), shape.end() - 2);
        q_full_shape.push_back(m); q_full_shape.push_back(m);
        std::vector<int64_t> r_full_shape(shape.begin(), shape.end());
        std::vector<int64_t> v_shape(shape.begin(), shape.end() - 2);
        v_shape.push_back(m);

        Tensor Q_full(q_full_shape, input.dtype(), input.device());
        Tensor R_full(r_full_shape, input.dtype(), input.device());
        Tensor Vscratch(v_shape, input.dtype(), input.device());

        auto cont = input.contiguous();
        std::string shader = is_f64 ? "linalg_qr_global_f64" : "linalg_qr_global";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t m; uint32_t n_cols; uint32_t batch; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n_cols = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        size_t in_bytes = static_cast<size_t>(batch_size) * m * n * esz;
        size_t q_bytes  = static_cast<size_t>(batch_size) * m * m * esz;
        size_t v_bytes  = static_cast<size_t>(batch_size) * m * esz;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, Q_full.data_ptr()}, {2, R_full.data_ptr()}, {3, Vscratch.data_ptr()}
        };
        std::vector<size_t> sizes = {in_bytes, q_bytes, in_bytes, v_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        Tensor Q = (k == m) ? Q_full : dispatchContiguous(Q_full.slice(ndim - 1, 0, k, 1));
        Tensor R = (k == m) ? R_full : dispatchContiguous(R_full.slice(ndim - 2, 0, k, 1));
        return {Q, R};
    }
}

// ============================================================================
// SVD (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgSVD(const Tensor& input, bool full_matrices) -> std::vector<Tensor> {
    // No native BFloat16 shader at any matrix size: widen to Float32, narrow back.
    if (input.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgSVD(dispatchCast(input.contiguous(), DType::Float32), full_matrices);
        return { dispatchCast(res[0], DType::BFloat16),
                 dispatchCast(res[1], DType::BFloat16),
                 dispatchCast(res[2], DType::BFloat16) };
    }
    validate_linalg_dtype(input, "svd");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.svd: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];
    int64_t k = std::min(m, n);

    // Float16 above the small-matrix limit: compute in Float32, narrow back.
    if (is_f16 && (m > MAX_SMALL_LINALG_SIZE || n > MAX_SMALL_LINALG_SIZE)) {
        auto res = dispatchLinalgSVD(dispatchCast(input.contiguous(), DType::Float32), full_matrices);
        return { dispatchCast(res[0], DType::Float16),
                 dispatchCast(res[1], DType::Float16),
                 dispatchCast(res[2], DType::Float16) };
    }

    // Wide matrices (m < n) above the small single-workgroup limit: the
    // global-memory one-sided Jacobi shader (linalg_svd_global, used below)
    // only handles m >= n (tall/square) — it orthonormally completes columns
    // of U past n, which only makes sense when U is the "wide" factor.
    // Rather than special-casing a second shader variant, use the standard
    // SVD(A^T) transform: if A^T = U' S' V'^T (A^T is n x m, and n >= m so
    // the tall path applies directly), then A = (A^T)^T = V' S' U'^T, i.e.
    //   U_A = V'  (= transpose of Vt'),  S_A = S',  Vt_A = U'^T.
    // This recursive call operates on a tall/square matrix (rows = n > m =
    // cols, since m < n here), so it lands in the m >= n branch (either the
    // small path or the global shader) and can never re-enter this branch —
    // no infinite recursion. full_matrices carries straight through: for a
    // tall matrix, full_matrices only affects U' shape (m x m vs n x n) not
    // Vt' (always m x m since k' = m = cols(A^T)); transposing swaps that
    // onto Vt_A, so U_A is always (m x m) and Vt_A is (k x n) reduced / (n x
    // n) full — exactly the contract linalg.svd promises for a wide input.
    // The recursive call already sorts singular values descending and
    // permutes U'/Vt' to match, so no further sort is needed here.
    if (m < n && (m > MAX_SMALL_LINALG_SIZE || n > MAX_SMALL_LINALG_SIZE)) {
        Tensor At = dispatchTranspose(input.contiguous(), ndim - 2, ndim - 1).contiguous();
        auto res = dispatchLinalgSVD(At, full_matrices);
        Tensor U_a  = dispatchTranspose(res[2], ndim - 2, ndim - 1).contiguous();  // Vt'^T
        Tensor Vt_a = dispatchTranspose(res[0], ndim - 2, ndim - 1).contiguous();  // U'^T
        return {U_a, res[1], Vt_a};
    }

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    // S shape is identical in both paths (batch, k).
    std::vector<int64_t> s_shape(shape.begin(), shape.end() - 2);
    s_shape.push_back(k);

    // For small matrices, all outputs match input dtype.
    // For large (tiled) matrices, S is always f32/f64 for numerical stability.
    DType s_dtype = (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE)
                  ? input.dtype()
                  : (is_f64 ? DType::Float64 : DType::Float32);

    Tensor S(s_shape, s_dtype, input.device());

    // U / Vt shapes depend on full_matrices and on which path runs (the small
    // single-workgroup shader honours full_matrices; the large global-memory
    // shader currently only produces the reduced factors). Allocate them inside
    // each branch so the host buffers match exactly what the dispatched shader
    // writes — otherwise full mode (m > k) overflows a reduced U buffer and
    // reduced mode leaves the trailing n - k rows of an n×n Vt uninitialised.
    Tensor U;
    Tensor Vt;

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;

    if (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE) {
        // ---- Small matrix path: single-workgroup Jacobi SVD ----
        std::string shader = is_f64 ? "linalg_svd_f64" : is_f16 ? "linalg_svd_f16" : "linalg_svd";
        auto* pipeline = getPipeline(shader, device_id);

        // Match the fixed shader's output layout (LAPACK jobz 'A' vs 'S'):
        //   full mode    -> U (m × m), Vt (n × n)
        //   reduced mode -> U (m × k), Vt (k × n)
        int64_t u_cols  = full_matrices ? m : k;
        int64_t vt_rows = full_matrices ? n : k;

        std::vector<int64_t> u_shape(shape.begin(), shape.end() - 2);
        u_shape.push_back(m); u_shape.push_back(u_cols);
        std::vector<int64_t> vt_shape(shape.begin(), shape.end() - 2);
        vt_shape.push_back(vt_rows); vt_shape.push_back(n);
        U  = Tensor(u_shape, input.dtype(), input.device());
        Vt = Tensor(vt_shape, input.dtype(), input.device());

        struct PushConstants { uint32_t m; uint32_t n_cols; uint32_t batch; uint32_t full_matrices; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n_cols = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.full_matrices = full_matrices ? 1 : 0;

        auto cont = input.contiguous();
        size_t in_numel = batch_size * m * n;
        size_t u_numel = batch_size * m * u_cols;
        size_t s_numel = batch_size * k;
        size_t vt_numel = batch_size * vt_rows * n;
        size_t in_size = is_f16 ? f16_buf(in_numel) : in_numel * elem_size;
        size_t u_size = is_f16 ? f16_buf(u_numel) : u_numel * elem_size;
        size_t s_size = is_f16 ? f16_buf(s_numel) : s_numel * elem_size;
        size_t vt_size = is_f16 ? f16_buf(vt_numel) : vt_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, U.data_ptr()}, {2, S.data_ptr()}, {3, Vt.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, u_size, s_size, vt_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        // Larger matrices: one-sided Jacobi SVD (global-memory, correct for any size).
        // Handles m >= n (square + tall); the m < n case is transformed away above.
        auto cont = input.contiguous();
        std::string shader = is_f64 ? "linalg_svd_global_f64" : "linalg_svd_global";
        auto* pipeline = getPipeline(shader, device_id);

        // The global shader honours full_matrices. With m >= n we have k == n,
        // so V is already the full n × n basis (Vt = V^T is n × n in both
        // modes). U is written m × u_cols where u_cols == m in full mode (the
        // shader orthonormally completes the trailing m - n columns) or n in
        // reduced mode (== m × k). Allocate U to match exactly what the shader
        // writes so full mode does not overflow a reduced-sized buffer.
        int64_t u_cols = full_matrices ? m : n;
        std::vector<int64_t> u_shape(shape.begin(), shape.end() - 2);
        u_shape.push_back(m); u_shape.push_back(u_cols);
        U = Tensor(u_shape, input.dtype(), input.device());

        struct PushConstants { uint32_t m; uint32_t n; uint32_t batch; uint32_t full_matrices; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.full_matrices = full_matrices ? 1u : 0u;

        size_t esz = is_f64 ? 8u : 4u;
        size_t in_bytes = static_cast<size_t>(batch_size) * m * n * esz;
        size_t u_bytes  = static_cast<size_t>(batch_size) * m * u_cols * esz;
        size_t s_bytes  = static_cast<size_t>(batch_size) * n * esz;
        size_t v_bytes  = static_cast<size_t>(batch_size) * n * n * esz;

        // V (n x n) accumulates the right rotations; Vt is its transpose.
        std::vector<int64_t> v_shape(shape.begin(), shape.end() - 2);
        v_shape.push_back(n); v_shape.push_back(n);
        Tensor Vmat(v_shape, s_dtype, input.device());

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, U.data_ptr()}, {2, S.data_ptr()}, {3, Vmat.data_ptr()}
        };
        std::vector<size_t> sizes = {in_bytes, u_bytes, s_bytes, v_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        // Vt = V^T  (n x n)
        Vt = dispatchTranspose(Vmat, ndim - 2, ndim - 1).contiguous();
    }

    // Sort singular values descending and permute the corresponding columns
    // of U and rows of Vt to match. LAPACK/CPU return S in non-increasing
    // order — the Jacobi SVD shader emits them in an arbitrary order so
    // consumers that assume sorted output (SVD3x3 test, low-rank
    // approximations) would see wrong values.
    if (k > 1) {
        auto [sorted_S, sort_indices] = dispatchSort(S, /*dim=*/-1, /*descending=*/true);

        // Helper: extend a length-k permutation with the fixed identity tail
        // [k, k+1, ..., axis_len-1] when an axis is wider than k (full_matrices
        // mode). The first k singular vectors are permuted to match the sorted
        // singular values; the trailing null-space vectors stay in place.
        auto extend_indices = [&](int64_t axis_len) -> Tensor {
            if (axis_len == k) return sort_indices;
            auto tail_cpu = Tensor({axis_len - k}, DType::Int64, Device::cpu());
            auto* tail_data = tail_cpu.data<int64_t>();
            for (int64_t i = 0; i < axis_len - k; ++i) tail_data[i] = k + i;
            Tensor tail = tail_cpu.to(input.device());
            return dispatchCat({sort_indices, tail}, 0);
        };

        // U: (..., m, u_cols) where u_cols == m (full) or k (reduced). Gather
        // columns (last axis) by the (extended) permutation so full-mode U
        // keeps all m columns — its trailing m-k columns are the orthonormal
        // completion the shader wrote and must survive the sort untouched.
        Tensor U_t = dispatchTranspose(U, ndim - 2, ndim - 1);  // (..., u_cols, m)
        Tensor u_idx = extend_indices(U.shape()[ndim - 1]);
        Tensor sorted_U_t = dispatchIndexSelect(U_t.contiguous(), ndim - 2, u_idx);
        Tensor sorted_U = dispatchTranspose(sorted_U_t, ndim - 2, ndim - 1).contiguous();

        // Vt: (..., vt_rows, n) where vt_rows == n (full) or k (reduced). The
        // row axis (ndim - 2) holds the right singular vectors; gather rows by
        // the same (extended) permutation.
        Tensor vt_idx = extend_indices(Vt.shape()[ndim - 2]);
        Tensor sorted_Vt = dispatchIndexSelect(Vt.contiguous(), ndim - 2, vt_idx);

        return {sorted_U, sorted_S, sorted_Vt};
    }

    return {U, S, Vt};
}

// ---------------------------------------------------------------------------
// runBlockedTridiag — Blocked tridiagonalization of symmetric matrices
// Reduces A to tridiagonal form in-place using panel Householder reflections.
// tau stores Householder scalars, one per column reduced.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedTridiag(Tensor& A, Tensor& tau, int64_t n,
                                       int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t work_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;
    size_t work_size = is_f16 ? f16_buf(work_numel) : work_numel * elem_size;

    // Allocate workspace for panel computation
    Tensor work({batch_size, n}, A.dtype(), A.device());

    int64_t k = n - 1;  // number of columns to reduce (tridiag reduces n-1 columns)

    for (int64_t col_start = 0; col_start < k; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, k - col_start);

        // --- Panel factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_tridiag_panel_f64" : is_f16 ? "linalg_tridiag_panel_f16" : "linalg_tridiag_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()}, {2, work.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size, work_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- NO separate trailing-submatrix update ---
        // The panel shader (linalg_tridiag_panel{,_f64,_f16}.comp) already applies
        // each Householder reflection as a full two-sided symmetric rank-2 update
        // A <- H*A*H^T over the ENTIRE remaining submatrix [col+1, n) x [col+1, n)
        // (see the rank-2 update loop, which ranges r,c over [col+1, n) with no
        // panel-column restriction). After the panel completes columns
        // [col_start, col_start+panel_cols), the trailing block [trail_start, n)
        // has therefore ALREADY had all panel reflections applied on both sides.
        //
        // Dispatching linalg_tridiag_update here would re-apply every reflection a
        // second time, destroying symmetry and corrupting later panels (the
        // symmetric eigendecomposition was wrong for n > one panel). The correct,
        // self-consistent design is: the panel is an unblocked reduction over the
        // full trailing region, and there is no separate trailing update. The next
        // panel iteration simply continues from the (already fully updated) matrix.
    }
}

// ---------------------------------------------------------------------------
// runBlockedHessenberg — Blocked Hessenberg reduction for general matrices
// Reduces A to upper Hessenberg form in-place using panel Householder reflections.
// tau stores Householder scalars.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedHessenberg(Tensor& A, Tensor& tau, int64_t n,
                                          int64_t batch_size, int32_t device_id, bool is_f64) {
    size_t elem_size = is_f64 ? 8 : 4;
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = mat_numel * elem_size;
    size_t tau_size = tau_numel * elem_size;

    int64_t k = n - 2;  // Hessenberg reduces n-2 columns

    for (int64_t col_start = 0; col_start < k; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, k - col_start);

        // --- Panel Hessenberg factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_hessenberg_panel_f64" : "linalg_hessenberg_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply accumulated reflections to trailing columns ---
        int64_t trail_start = col_start + panel_cols;
        // Apply whenever ANY trailing column remains (< n), matching the blocked
        // QR/Bidiag/tridiag routines above. The previous `< k` (k = n-2) bound
        // skipped the update for panels ending at/past column n-2, leaving the
        // last one or two columns un-reflected and corrupting the Hessenberg form
        // (and thus the eigenvalues) for matrices spanning more than one block.
        if (trail_start < n) {
            int64_t trail_cols = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_hessenberg_update_f64" : "linalg_hessenberg_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_cols), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ============================================================================
// Eigh — Symmetric Eigenvalue Decomposition (tiled for arbitrary-sized matrices)
// ============================================================================

auto VulkanBackend::dispatchLinalgEigh(const Tensor& input) -> std::vector<Tensor> {
    // No native BFloat16 shader: widen to Float32, compute, narrow back.
    if (input.dtype() == DType::BFloat16) {
        auto res = dispatchLinalgEigh(dispatchCast(input.contiguous(), DType::Float32));
        return { dispatchCast(res[0], DType::BFloat16), dispatchCast(res[1], DType::BFloat16) };
    }
    validate_linalg_dtype(input, "eigh");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.eigh: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.eigh: input must be square");

    // Float16: compute in Float32 (the proven path) and narrow the results. Handling the
    // f16 widen/narrow at the function boundary — rather than with extra in-flight scratch
    // tensors inside the dispatch — keeps the allocation pattern identical to the f32 path
    // and avoids a buffer-recycling corner case that corrupted eigenvalues for some n.
    if (input.dtype() == DType::Float16) {
        auto res = dispatchLinalgEigh(dispatchCast(input.contiguous(), DType::Float32));
        return { dispatchCast(res[0], DType::Float16), dispatchCast(res[1], DType::Float16) };
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Output: eigenvalues (batch, n), eigenvectors (batch, n, n)
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 2);
    w_shape.push_back(n);

    Tensor W(w_shape, input.dtype(), input.device());
    Tensor V(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    // Symmetric eigendecomposition via global-memory cyclic Jacobi. Unlike the old
    // shared-memory shader (hard-capped at 32x32 but dispatched up to n=128, which
    // corrupted results for 32 < n <= 128) and the tiled tridiagonal path (which threw
    // on batched 3D diagonal extraction), this single path is correct for ANY n and is
    // batched (one workgroup per batch element). Float16 widens to Float32 then narrows.
    // Eigenvalues/eigenvectors come back unordered; the sort step below restores order.
    {
        const bool d64 = is_f64;
        const size_t esz = d64 ? 8u : 4u;

        Tensor src = input.contiguous();
        Tensor Dscratch(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

        const std::string shader = d64 ? "linalg_eigh_global_f64" : "linalg_eigh_global";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        const size_t mat_bytes = static_cast<size_t>(batch_size) * n * n * esz;
        const size_t w_bytes   = static_cast<size_t>(batch_size) * n * esz;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, src.data_ptr()}, {1, W.data_ptr()}, {2, V.data_ptr()}, {3, Dscratch.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_bytes, w_bytes, mat_bytes, mat_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Sort eigenvalues ascending and permute the eigenvector columns to match.
    // Matches LAPACK / CPU eigh semantics — LOBPCG / Eigh tests assume
    // W[0] <= W[1] <= ... <= W[n-1].
    //
    // The Jacobi/tridiagonal-QR shaders return W/V in an arbitrary order (the
    // order in which the sweep converges), which defeats any downstream
    // algorithm that expects a specific eigenvalue ordering.
    if (n > 1) {
        auto [sorted_W, sort_indices] = dispatchSort(W, /*dim=*/-1, /*descending=*/false);

        // Permute eigenvector columns to match the sorted eigenvalues, batch-aware:
        //   sorted_V[..., i, j] = V[..., i, sort_indices[..., j]].
        // Broadcast the per-row index (..., n) over the row axis to (..., n, n) and
        // gather along the last dim. The previous transpose + index_select approach
        // silently produced wrong shapes for batched inputs (index_select takes a
        // 1-D index), so batched eigh returned mis-shaped eigenvectors.
        Tensor idx_full = dispatchExpand(
            dispatchUnsqueeze(sort_indices, ndim - 2),
            std::vector<int64_t>(shape.begin(), shape.end()));
        Tensor sorted_V = dispatchGather(V, ndim - 1, idx_full.contiguous());

        return {sorted_W, sorted_V};
    }

    return {W, V};
}

// ============================================================================
// Eig — General Eigenvalue Decomposition (tiled for arbitrary-sized matrices)
// ============================================================================

auto VulkanBackend::dispatchLinalgEig(const Tensor& input) -> std::vector<Tensor> {
    validate_linalg_dtype(input, "eig");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.eig: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.eig: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Approximate-symmetry check — runs entirely on-device. Two parallel
    // max-reductions (max|A| and max|A − Aᵀ|) collapse the per-batch matrix
    // into two scalars; only those 8 bytes flow back to the host. gradcheck
    // perturbs SPD inputs by ε≈1e-6 so the symmetric path is taken whenever
    // diff_max / max(a_max, 1) < 1e-3 (Float64) or 1e-2 (Float32). The
    // Float64 probe reduces to Float32 maxes — the threshold is robust to
    // ~1 ULP loss and avoids the uint64-atomic extension.
    if (input.dtype() == DType::Float32 || input.dtype() == DType::Float64) {
        auto cont = input.contiguous();
        Tensor pair_buf = dispatchFull({2}, 0.0f, DType::Int32);

        std::string probe_shader = (input.dtype() == DType::Float64)
            ? "eig_symmetry_probe_f64" : "eig_symmetry_probe";
        auto* probe_pipeline = getPipeline(probe_shader, device_id);

        struct ProbePC {
            uint32_t nbatch;
            uint32_t n;
        } probe_pc{static_cast<uint32_t>(batch_size), static_cast<uint32_t>(n)};

        size_t mat_bytes = static_cast<size_t>(batch_size) * n * n
                         * (input.dtype() == DType::Float64 ? sizeof(double) : sizeof(float));
        size_t pair_bytes = 2 * sizeof(uint32_t);

        std::vector<std::pair<uint32_t, const void*>> probe_bindings = {
            {0, cont.data_ptr()},
            {1, pair_buf.data_ptr()},
        };
        std::vector<size_t> probe_sizes = {mat_bytes, pair_bytes};

        VkDescriptorSet probe_ds = allocateAndWriteDescriptorSet(
            device_id, probe_pipeline, probe_bindings, probe_sizes);

        VkCommandBuffer probe_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(probe_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                         probe_pipeline->pipeline());
        vkCmdBindDescriptorSets(probe_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               probe_pipeline->layout(), 0, 1, &probe_ds, 0, nullptr);
        vkCmdPushConstants(probe_cmd, probe_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(probe_pc), &probe_pc);
        vkCmdDispatch(probe_cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(probe_cmd);
        endSingleTimeCommands(probe_cmd, device_id);
        synchronize(device_id);

        Tensor pair_cpu = pair_buf.to(Device::cpu());
        const uint32_t* pair_bits = reinterpret_cast<const uint32_t*>(pair_cpu.data<int32_t>());
        float a_max_f, diff_max_f;
        std::memcpy(&a_max_f, &pair_bits[0], sizeof(float));
        std::memcpy(&diff_max_f, &pair_bits[1], sizeof(float));
        double a_max = static_cast<double>(a_max_f);
        double diff_max = static_cast<double>(diff_max_f);
        double tol = (input.dtype() == DType::Float32) ? 1e-2 : 1e-3;
        bool is_near_symmetric = (diff_max < tol * std::max(a_max, 1.0));

        if (is_near_symmetric) {
            auto At = ::tenzor::transpose(input, ndim - 2, ndim - 1).contiguous();
            auto A_sym = ::tenzor::mul(::tenzor::add(input, At), 0.5).contiguous();
            auto eigh_outs = dispatchLinalgEigh(A_sym);
            // dispatchLinalgEigh returns {W, V} (symmetric eigendecomposition).
            std::vector<int64_t> w_shape_v(shape.begin(), shape.end() - 2);
            w_shape_v.push_back(n);
            Tensor WI_zero = dispatchFull(w_shape_v, 0.0f, input.dtype());
            return {eigh_outs[0], WI_zero, eigh_outs[1]};
        }
    }

    // Non-symmetric eigendecomposition — native EISPACK-hqr2 shader. One
    // invocation per batch element over global-memory scratch; computes
    // eigenvalues (WR, WI) AND right eigenvectors (V) in one dispatch. Output
    // packing matches LAPACK geev: real eigenvalue k -> column k; complex pair
    // (WI[k]>0) -> column k = Re, column k+1 = Im (conjugate in column k+1).
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 2);
    w_shape.push_back(n);
    std::vector<int64_t> v_shape(shape.begin(), shape.end());

    Tensor WR(w_shape, input.dtype(), input.device());
    Tensor WI(w_shape, input.dtype(), input.device());
    Tensor V(v_shape, input.dtype(), input.device());

    // Working copy (overwritten with Schur form then eigenvectors) + Householder scratch.
    Tensor H = dispatchClone(input.contiguous());
    Tensor vscratch(w_shape, input.dtype(), input.device());

    size_t elem_size = is_f64 ? 8 : 4;
    size_t mat_size = static_cast<size_t>(batch_size) * n * n * elem_size;
    size_t w_size   = static_cast<size_t>(batch_size) * n * elem_size;

    std::string shader = is_f64 ? "linalg_eig_hqr2_f64" : "linalg_eig_hqr2";
    auto* pipeline = getPipeline(shader, device_id);

    struct PushConstants { uint32_t n; uint32_t nbatch; } pc;
    pc.n = static_cast<uint32_t>(n);
    pc.nbatch = static_cast<uint32_t>(batch_size);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, H.data_ptr()}, {1, V.data_ptr()}, {2, WR.data_ptr()},
        {3, WI.data_ptr()}, {4, vscratch.data_ptr()}
    };
    std::vector<size_t> sizes = {mat_size, mat_size, w_size, w_size, w_size};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t groups = (static_cast<uint32_t>(batch_size) + 63u) / 64u;
    vkCmdDispatch(cmd, groups, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return {WR, WI, V};
}

// ============================================================================
// LSTM Cell Forward (single timestep)
// ============================================================================

auto VulkanBackend::dispatchLSTMCellForward(const Tensor& input, const Tensor& hx, const Tensor& cx,
                                             const Tensor& weight_ih, const Tensor& weight_hh,
                                             const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // Phase 2.1: Float16 path runs natively via `lstm_cell_f16.comp`, which
    // accumulates in float32 internally. The internal dispatchMatmul that
    // computes the gates also keeps FP16 I/O with an FP32 accumulator, so
    // no host upcast is needed.

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();
    int64_t batch_size = in_shape[0];
    int64_t hidden_size = hx_shape[1];
    int32_t device_id = input.device().index;

    bool is_f64  = (input.dtype() == DType::Float64);
    bool is_f16  = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string cell_shader = is_f64  ? "lstm_cell_f64"
                            : is_f16  ? "lstm_cell_f16"
                            : is_bf16 ? "lstm_cell_bf16"
                                      : "lstm_cell";

    // Compute gates = input @ W_ih^T + hx @ W_hh^T + bias_ih + bias_hh
    Tensor W_ih_t = dispatchTranspose(weight_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(weight_hh, 0, 1);
    Tensor gates = dispatchMatmul(input, W_ih_t);
    Tensor h_gates = dispatchMatmul(hx, W_hh_t);
    gates = dispatchBinaryOp("add", gates, h_gates);
    if (bias_ih.numel() > 0) gates = dispatchBinaryOp("add", gates, bias_ih);
    if (bias_hh.numel() > 0) gates = dispatchBinaryOp("add", gates, bias_hh);

    // Allocate outputs
    Tensor hy({batch_size, hidden_size}, input.dtype(), input.device());
    Tensor cy({batch_size, hidden_size}, input.dtype(), input.device());

    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
    size_t elem_size = input.dtype_size();
    size_t gate_bytes = batch_size * 4 * hidden_size * elem_size;
    size_t state_bytes = batch_size * hidden_size * elem_size;

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(cell_shader, device_id);
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, gates.data_ptr()}, {1, cx.data_ptr()},
        {2, hy.data_ptr()}, {3, cy.data_ptr()}
    };
    std::vector<size_t> sizes = {gate_bytes, state_bytes, state_bytes, state_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg_checked(total, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch"), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return {hy, cy};
}

// ============================================================================
// GRU Cell Forward (single timestep)
// ============================================================================

auto VulkanBackend::dispatchGRUCellForward(const Tensor& input, const Tensor& hx,
                                            const Tensor& weight_ih, const Tensor& weight_hh,
                                            const Tensor& bias_ih, const Tensor& bias_hh)
    -> Tensor {
    // Phase 2.1: Float16 path runs natively via `gru_cell_f16.comp`, which
    // accumulates in float32 internally.

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();
    int64_t batch_size = in_shape[0];
    int64_t hidden_size = hx_shape[1];
    int32_t device_id = input.device().index;

    bool is_f64  = (input.dtype() == DType::Float64);
    bool is_f16  = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string cell_shader = is_f64  ? "gru_cell_f64"
                            : is_f16  ? "gru_cell_f16"
                            : is_bf16 ? "gru_cell_bf16"
                                      : "gru_cell";

    // Compute gate projections
    Tensor W_ih_t = dispatchTranspose(weight_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(weight_hh, 0, 1);
    Tensor gates_x = dispatchMatmul(input, W_ih_t);
    if (bias_ih.numel() > 0) gates_x = dispatchBinaryOp("add", gates_x, bias_ih);
    Tensor gates_h = dispatchMatmul(hx, W_hh_t);
    if (bias_hh.numel() > 0) gates_h = dispatchBinaryOp("add", gates_h, bias_hh);

    // Allocate output
    Tensor hy({batch_size, hidden_size}, input.dtype(), input.device());

    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
    size_t elem_size = input.dtype_size();
    size_t gate_bytes = batch_size * 3 * hidden_size * elem_size;
    size_t state_bytes = batch_size * hidden_size * elem_size;

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(cell_shader, device_id);
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, gates_x.data_ptr()}, {1, gates_h.data_ptr()},
        {2, hx.data_ptr()}, {3, hy.data_ptr()}
    };
    std::vector<size_t> sizes = {gate_bytes, gate_bytes, state_bytes, state_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg_checked(total, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch"), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return hy;
}

// ===========================================================================
// SearchSorted — native GPU binary search shader
// ===========================================================================

auto VulkanBackend::dispatchSearchSorted(const Tensor& sorted, const Tensor& values, bool right) -> Tensor {
    if (sorted.numel() == 0 || values.numel() == 0) {
        return Tensor(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                      DType::Int64, values.device());
    }

    // Float16/BFloat16: native packed shader path
    if (sorted.dtype() == DType::Float16 || sorted.dtype() == DType::BFloat16) {
        bool is_bf16_ss = (sorted.dtype() == DType::BFloat16);
        int32_t dev_id = sorted.device().index;

        auto sorted_cont = (sorted.is_contiguous() && sorted.offset() == 0) ? sorted : dispatchContiguous(sorted);
        auto values_cont = (values.is_contiguous() && values.offset() == 0) ? values : dispatchContiguous(values);

        std::vector<int64_t> out_shape_f16(values.shape().begin(), values.shape().end());
        Tensor output_f16(out_shape_f16, DType::Int32, values.device());

        auto* pipe = getPipeline(is_bf16_ss ? "searchsorted_bf16" : "searchsorted_f16", dev_id);

        struct { uint32_t array_size; uint32_t num_queries; uint32_t right; } pc;
        pc.array_size = static_cast<uint32_t>(sorted.numel());
        pc.num_queries = static_cast<uint32_t>(values.numel());
        pc.right = right ? 1u : 0u;

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t sorted_bsz = (static_cast<size_t>(sorted_cont.numel()) + 1) / 2 * 4;
        size_t values_bsz = (static_cast<size_t>(values_cont.numel()) + 1) / 2 * 4;
        size_t output_bsz = output_f16.numel() * output_f16.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> binds = {
            {0, sorted_cont.data_ptr()}, {1, values_cont.data_ptr()}, {2, output_f16.data_ptr()}
        };
        std::vector<size_t> szs = {sorted_bsz, values_bsz, output_bsz};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(dev_id, pipe, binds, szs);
        VkCommandBuffer cmd = beginSingleTimeCommands(dev_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipe->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipe->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg_checked(values.numel(), devices_[dev_id].workgroupSize, devices_[dev_id].maxComputeWorkGroupCount[0], "vk_dispatch"), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, dev_id);

        return output_f16.to(DType::Int64);
    }

    int32_t device_id = sorted.device().index;

    auto sorted_contig = (sorted.is_contiguous() && sorted.offset() == 0) ? sorted : dispatchContiguous(sorted);
    auto values_contig = (values.is_contiguous() && values.offset() == 0) ? values : dispatchContiguous(values);

    std::vector<int64_t> out_shape(values.shape().begin(), values.shape().end());
    Tensor output(out_shape, DType::Int32, values.device());

    bool is_f64 = (sorted.dtype() == DType::Float64);
    std::string ss_shader = is_f64 ? "searchsorted_f64" : "searchsorted";
    auto* pipeline = getPipeline(ss_shader, device_id);

    struct {
        uint32_t array_size;
        uint32_t num_queries;
        uint32_t right;
    } pc;
    pc.array_size = static_cast<uint32_t>(sorted.numel());
    pc.num_queries = static_cast<uint32_t>(values.numel());
    pc.right = right ? 1u : 0u;

    size_t sorted_bytes = sorted_contig.numel() * sorted_contig.dtype_size();
    size_t values_bytes = values_contig.numel() * values_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, sorted_contig.data_ptr()},
        {1, values_contig.data_ptr()},
        {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {sorted_bytes, values_bytes, output_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg_checked(values.numel(), devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch"), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Match the CPU/CUDA convention: searchsorted returns Int64 by default.
    // The shader writes Int32 for shader-side simplicity; widen before
    // returning so downstream consumers (and parity tests) see Int64.
    return output.to(DType::Int64);
}

// ===========================================================================
// Quantized Linear — native Int8 GEMM with Int32 accumulation
// ===========================================================================

auto VulkanBackend::dispatchQuantizedLinear(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp,
    const Tensor* per_channel_scales, const Tensor* per_channel_zps) -> Tensor
{
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t M = input_shape[0];   // batch_size
    int64_t K = input_shape[1];   // in_features
    int64_t N = weight_shape[0];  // out_features

    // F031/F072: QInt4x2 (packed INT4) weight. The int8 shader indexes each
    // weight row with stride K (one byte per input feature), but a QInt4x2
    // row is only ceil(K/2) packed bytes (two 4-bit values per byte) — every
    // row past the first would silently be read from the wrong offset, and
    // the last channels would read past the buffer end. Route to the
    // dedicated INT4-unpacking shader instead (mirrors CPU's
    // fused_qlinear_dequant / CUDA's quantized_linear_int4_cuda_kernel:
    // symmetric-only, K must be even since each packed byte holds a whole
    // pair of input columns).
    const bool weight_is_int4 = (weight.dtype() == DType::QInt4x2);
    if (weight_is_int4 && (K % 2 != 0)) {
        throw std::runtime_error(
            "Vulkan QuantizedLinear (INT4): in_features (K=" + std::to_string(K) +
            ") must be even for INT4 packing");
    }
    // Validate the weight dtype before binding it into either shader. CPU/CUDA
    // get this check "for free" via the typed `weight.data<int8_t>()` accessor
    // (which throws on a dtype mismatch); Vulkan instead binds the raw
    // `data_ptr()` with no dtype check at all, so a weight tensor of any other
    // dtype (Float32, Int32, QInt8-if-added-later, ...) would silently have its
    // bit pattern reinterpreted as packed-INT4 or plain INT8 by the shader --
    // wrong output with no diagnostic. Only QInt4x2 (routes to the INT4 shader)
    // and Int8 (routes to the plain INT8 shader) are valid.
    if (!weight_is_int4 && weight.dtype() != DType::Int8) {
        throw std::invalid_argument(
            "Vulkan QuantizedLinear: unsupported weight dtype '" +
            std::string(dtype_name(weight.dtype())) +
            "' -- expected Int8 (per-tensor/per-channel INT8) or QInt4x2 (packed INT4)");
    }

    int32_t device_id = input.device().index;

    auto input_contig = (input.is_contiguous() && input.offset() == 0) ? input : dispatchContiguous(input);
    auto weight_contig = (weight.is_contiguous() && weight.offset() == 0) ? weight : dispatchContiguous(weight);
    auto bias_contig = (bias.is_contiguous() && bias.offset() == 0) ? bias : dispatchContiguous(bias);

    Tensor output({M, N}, DType::Float32, input.device());

    auto* pipeline = getPipeline(weight_is_int4 ? "quantized_linear_int4" : "quantized_linear", device_id);

    // Per-channel (F045): bindings 4/5 must ALWAYS be bound (Vulkan descriptor
    // set completeness), so use placeholder 1-element buffers in the per-tensor
    // case. per_channel != 0 selects the buffer path in the shader.
    const bool per_channel = (per_channel_scales != nullptr) && (per_channel_scales->numel() > 1);
    Tensor wscale_buf, wzp_buf;
    if (per_channel) {
        wscale_buf = (per_channel_scales->is_contiguous() && per_channel_scales->offset() == 0)
            ? *per_channel_scales : dispatchContiguous(*per_channel_scales);
        wzp_buf = (per_channel_zps != nullptr && per_channel_zps->numel() > 0)
            ? ((per_channel_zps->is_contiguous() && per_channel_zps->offset() == 0)
                ? *per_channel_zps : dispatchContiguous(*per_channel_zps))
            : dispatchZeros({N}, DType::Int32, input.device());
    } else {
        wscale_buf = dispatchFull({1}, static_cast<double>(weight_scale), DType::Float32, input.device());
        wzp_buf = dispatchFull({1}, static_cast<double>(weight_zp), DType::Int32, input.device());
    }

    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        float input_scale;
        float weight_scale;
        int32_t input_zero_point;
        int32_t weight_zero_point;
        uint32_t per_channel;
    } pc;
    pc.M = static_cast<uint32_t>(M);
    pc.N = static_cast<uint32_t>(N);
    pc.K = static_cast<uint32_t>(K);
    pc.input_scale = input_scale;
    pc.weight_scale = weight_scale;
    pc.input_zero_point = input_zp;
    pc.weight_zero_point = weight_zp;
    pc.per_channel = per_channel ? 1u : 0u;

    size_t input_bytes = input_contig.numel() * input_contig.dtype_size();
    size_t weight_bytes = weight_contig.numel() * weight_contig.dtype_size();
    size_t bias_bytes = bias_contig.numel() * bias_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();
    size_t wscale_bytes = wscale_buf.numel() * wscale_buf.dtype_size();
    size_t wzp_bytes = wzp_buf.numel() * wzp_buf.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_contig.data_ptr()},
        {3, output.data_ptr()},
        {4, wscale_buf.data_ptr()},
        {5, wzp_buf.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, weight_bytes, bias_bytes, output_bytes,
                                 wscale_bytes, wzp_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    int64_t total = M * N;
    vkCmdDispatch(cmd, div_wg_checked(total, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch"), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Quantized Conv2d — native Int8 convolution with Int32 accumulation
// ===========================================================================

auto VulkanBackend::dispatchQuantizedConv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int64_t groups,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp,
    const Tensor* per_channel_scales, const Tensor* per_channel_zps) -> Tensor
{
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];
    int64_t out_channels = weight_shape[0];
    // Per-axis kernel size (F044: rectangular kernels).
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Validate convolution geometry before deriving output dims. Without this a
    // kernel larger than the padded input gives a negative numerator and the
    // integer division yields a non-positive h_out/w_out, which would reach the
    // Tensor constructor below as a negative dimension (and an invalid dispatch).
    if (stride_h <= 0 || stride_w <= 0) {
        throw std::runtime_error("QuantizedConv2d: stride must be positive, got (" +
                                 std::to_string(stride_h) + "," + std::to_string(stride_w) + ")");
    }
    if (pad_h < 0 || pad_w < 0) {
        throw std::runtime_error("QuantizedConv2d: padding must be non-negative, got (" +
                                 std::to_string(pad_h) + "," + std::to_string(pad_w) + ")");
    }
    if (groups < 1 || in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::runtime_error(
            "QuantizedConv2d: groups must be >= 1 and divide in/out channels; got groups=" +
            std::to_string(groups));
    }
    // Effective (dilated) kernel extent for the geometry check.
    int64_t eff_kh = dil_h * (kernel_h - 1) + 1;
    int64_t eff_kw = dil_w * (kernel_w - 1) + 1;
    if (eff_kh > h_in + 2 * pad_h || eff_kw > w_in + 2 * pad_w) {
        throw std::runtime_error(
            "QuantizedConv2d: dilated kernel (" + std::to_string(eff_kh) + "x" +
            std::to_string(eff_kw) + ") exceeds padded input (" +
            std::to_string(h_in + 2 * pad_h) + "x" + std::to_string(w_in + 2 * pad_w) + ")");
    }

    int64_t h_out = (h_in + 2 * pad_h - eff_kh) / stride_h + 1;
    int64_t w_out = (w_in + 2 * pad_w - eff_kw) / stride_w + 1;

    int32_t device_id = input.device().index;

    auto input_contig = (input.is_contiguous() && input.offset() == 0) ? input : dispatchContiguous(input);
    auto weight_contig = (weight.is_contiguous() && weight.offset() == 0) ? weight : dispatchContiguous(weight);
    auto bias_contig = (bias.is_contiguous() && bias.offset() == 0) ? bias : dispatchContiguous(bias);

    Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input.device());

    auto* pipeline = getPipeline("quantized_conv2d", device_id);

    // Per-channel (F045): bindings 4/5 always bound (placeholder in per-tensor case).
    const bool per_channel = (per_channel_scales != nullptr) && (per_channel_scales->numel() > 1);
    Tensor wscale_buf, wzp_buf;
    if (per_channel) {
        wscale_buf = (per_channel_scales->is_contiguous() && per_channel_scales->offset() == 0)
            ? *per_channel_scales : dispatchContiguous(*per_channel_scales);
        wzp_buf = (per_channel_zps != nullptr && per_channel_zps->numel() > 0)
            ? ((per_channel_zps->is_contiguous() && per_channel_zps->offset() == 0)
                ? *per_channel_zps : dispatchContiguous(*per_channel_zps))
            : dispatchZeros({out_channels}, DType::Int32, input.device());
    } else {
        wscale_buf = dispatchFull({1}, static_cast<double>(weight_scale), DType::Float32, input.device());
        wzp_buf = dispatchFull({1}, static_cast<double>(weight_zp), DType::Int32, input.device());
    }

    struct {
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t h_in;
        uint32_t w_in;
        uint32_t h_out;
        uint32_t w_out;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t pad_h;
        uint32_t pad_w;
        uint32_t dil_h;
        uint32_t dil_w;
        uint32_t groups;
        float input_scale;
        float weight_scale;
        int32_t input_zero_point;
        int32_t weight_zero_point;
        uint32_t per_channel;
    } pc;
    pc.batch = static_cast<uint32_t>(batch);
    pc.in_channels = static_cast<uint32_t>(in_channels);
    pc.out_channels = static_cast<uint32_t>(out_channels);
    pc.h_in = static_cast<uint32_t>(h_in);
    pc.w_in = static_cast<uint32_t>(w_in);
    pc.h_out = static_cast<uint32_t>(h_out);
    pc.w_out = static_cast<uint32_t>(w_out);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.pad_h = static_cast<uint32_t>(pad_h);
    pc.pad_w = static_cast<uint32_t>(pad_w);
    pc.dil_h = static_cast<uint32_t>(dil_h);
    pc.dil_w = static_cast<uint32_t>(dil_w);
    pc.groups = static_cast<uint32_t>(groups);
    pc.input_scale = input_scale;
    pc.weight_scale = weight_scale;
    pc.input_zero_point = input_zp;
    pc.weight_zero_point = weight_zp;
    pc.per_channel = per_channel ? 1u : 0u;

    size_t input_bytes = input_contig.numel() * input_contig.dtype_size();
    size_t weight_bytes = weight_contig.numel() * weight_contig.dtype_size();
    size_t bias_bytes = bias_contig.numel() * bias_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();
    size_t wscale_bytes = wscale_buf.numel() * wscale_buf.dtype_size();
    size_t wzp_bytes = wzp_buf.numel() * wzp_buf.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_contig.data_ptr()},
        {3, output.data_ptr()},
        {4, wscale_buf.data_ptr()},
        {5, wzp_buf.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, weight_bytes, bias_bytes, output_bytes,
                                 wscale_bytes, wzp_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    int64_t total = batch * out_channels * h_out * w_out;
    vkCmdDispatch(cmd, div_wg_checked(total, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch"), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Flash Attention — composed dispatch using existing matmul + softmax shaders
//
// This is NOT a single fused kernel. It composes existing Vulkan matmul and
// softmax shaders to avoid the CPU roundtrip of VULKAN_CPU_FALLBACK, keeping
// all intermediate data on GPU. A truly fused tiled flash attention kernel
// would require a dedicated shader with shared-memory tiling and online
// softmax, which is left as a future optimization.
// ===========================================================================

auto VulkanBackend::dispatchFlashAttention(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    float scale, bool causal) -> std::pair<Tensor, Tensor>
{
    // Q, K, V shapes: [batch, heads, seq_len, d_k] or [batch, seq_len, d_k]
    auto q_shape = Q.shape();
    bool has_head_dim = (q_shape.size() == 4);

    // For 4D tensors: [B, H, S, D] — use BMM on [B*H, S, D]
    // For 3D tensors: [B, S, D] — use BMM on [B, S, D]

    // Flatten batch and head dimensions for BMM
    Tensor q_flat = Q;
    Tensor k_flat = K;
    Tensor v_flat = V;
    int64_t batch_heads, seq_len_q, seq_len_k, d_k;

    if (has_head_dim) {
        // Per docs/internals/attention-contract.md (stride/contiguity rule):
        // .contiguous() MUST precede .reshape(). The previous code called
        // Q.reshape(...) on a possibly-permuted view (typical MHA pattern of
        // permute(0,2,1,3) on [B,S,H,D]), which silently throws or returns
        // a logically-wrong view (audit C4 Vulkan).
        Tensor Q_c = (Q.is_contiguous() && Q.offset() == 0) ? Q : dispatchContiguous(Q);
        Tensor K_c = (K.is_contiguous() && K.offset() == 0) ? K : dispatchContiguous(K);
        Tensor V_c = (V.is_contiguous() && V.offset() == 0) ? V : dispatchContiguous(V);

        // JIT-R160/JIT-R161: when H_kv != H_q (GQA/MQA), broadcast K/V along
        // the head dim BEFORE the batch_heads collapse below. Without this,
        // reshape({q_shape[0]*q_shape[1], seq_len_k, d_k}) on a
        // [B, H_kv, seq_len_k, d_k] tensor has a mismatched element count
        // and throws -- mirrors the identical fix in the CUDA/ROCm/CPU/
        // OneAPI OpId::FlashAttention registries.
        int64_t h_q = q_shape[1];
        int64_t h_kv = K_c.shape()[1];
        if (h_kv != h_q) {
            if (h_q % h_kv != 0) {
                throw std::invalid_argument(
                    "FlashAttention Vulkan: H_q must be a multiple of H_kv; got " +
                    std::to_string(h_q) + " and " + std::to_string(h_kv));
            }
            int64_t b = q_shape[0];
            int64_t sk = K_c.shape()[2];
            int64_t dk_kv = K_c.shape()[3];
            int64_t dv_kv = V_c.shape()[3];
            int64_t reps = h_q / h_kv;
            Tensor Ku = dispatchUnsqueeze(K_c, 2);
            Tensor Vu = dispatchUnsqueeze(V_c, 2);
            Tensor Ke = dispatchExpand(Ku, {b, h_kv, reps, sk, dk_kv});
            Tensor Ve = dispatchExpand(Vu, {b, h_kv, reps, sk, dv_kv});
            K_c = dispatchContiguous(Ke).reshape({b, h_q, sk, dk_kv});
            V_c = dispatchContiguous(Ve).reshape({b, h_q, sk, dv_kv});
        }

        batch_heads = q_shape[0] * q_shape[1];
        seq_len_q = q_shape[2];
        d_k = q_shape[3];
        seq_len_k = K_c.shape()[2];

        q_flat = Q_c.reshape({batch_heads, seq_len_q, d_k});
        k_flat = K_c.reshape({batch_heads, seq_len_k, d_k});
        v_flat = V_c.reshape({batch_heads, seq_len_k, V_c.shape()[3]});
    } else {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        d_k = q_shape[2];
        seq_len_k = K.shape()[1];
    }

    // Phase 5.4 fast path — tiled FlashAttention-2 shader. Per
    // docs/internals/attention-contract.md the shader now supports causal
    // masking inline (audit C2 Vulkan fix), so the !causal gate is removed.
    // Constraints remaining:
    //   - Float32 only (FP16/BF16/F64 variants land with M8)
    //   - head_dim / head_v <= 128
    //   - contiguous Q, K, V (forced via dispatchContiguous below)
    //
    // The tiled path avoids materializing the full seq_q × seq_k attention
    // matrix — it streams K/V blocks through shared memory with an online
    // softmax. Falls through to the composed path on any dtype/shape mismatch.
    int64_t head_v = v_flat.shape()[2];
    const int64_t kMaxHeadDimTiled = 128;

    // audit A.11 (Vulkan): native Float64 fused fast path. Uses the
    // dedicated `flash_attention_f64.comp` shader (double-precision
    // throughout, FP32 LSE narrowed at write per attention contract).
    // Buffer sizes are 8 bytes/elem for Q/K/V/O and 4 bytes/elem for LSE.
    //
    // Device gating: the Vulkan FP64 SPIR-V capability requires
    // VkPhysicalDeviceFeatures::shaderFloat64 = VK_TRUE. We bind it
    // unconditionally at logical-device creation, so if the host GPU
    // doesn't advertise it, device init already failed before reaching
    // this dispatch. We still query `supports_fp64` here to throw a clean
    // user-facing error (project rule: no CPU fallback, no Float32
    // upcast — devices without FP64 simply cannot run this op).
    if (Q.dtype() == DType::Float64
        && K.dtype() == DType::Float64
        && V.dtype() == DType::Float64
        && d_k <= kMaxHeadDimTiled
        && head_v <= kMaxHeadDimTiled
        && seq_len_q > 0 && seq_len_k > 0)
    {
        int32_t device_id = Q.device().index;
        DeviceInfo dev_info = get_device_info(device_id);
        if (!dev_info.supports_fp64) {
            throw std::runtime_error(
                "Vulkan FlashAttention: Float64 requested but the active "
                "Vulkan device does not advertise VkPhysicalDeviceFeatures::"
                "shaderFloat64. FP64 compute shaders are not supported on "
                "this GPU (common on mobile/integrated parts). The project "
                "rule forbids CPU fallback or Float32 upcast — either run "
                "on a discrete GPU with FP64 support or use a different "
                "backend (CPU / CUDA / ROCm / OneAPI all have native FP64 "
                "FlashAttention kernels).");
        }

        Tensor q_contig = (q_flat.is_contiguous() && q_flat.offset() == 0) ? q_flat : dispatchContiguous(q_flat);
        Tensor k_contig = (k_flat.is_contiguous() && k_flat.offset() == 0) ? k_flat : dispatchContiguous(k_flat);
        Tensor v_contig = (v_flat.is_contiguous() && v_flat.offset() == 0) ? v_flat : dispatchContiguous(v_flat);

        Tensor output_flat({batch_heads, seq_len_q, head_v},
                           DType::Float64, Q.device());

        // LSE buffer is ALWAYS Float32 per attention_contract.hpp kLseDType.
        // The FP64 shader narrows at the final store so this stays correct.
        Tensor lse_flat({batch_heads, seq_len_q},
                       DType::Float32, Q.device());

        auto* pipeline = getPipeline("flash_attention_f64", device_id);

        struct PushConstants {
            int32_t seq_q;
            int32_t seq_k;
            int32_t head_dim;
            int32_t head_v;
            float scale;        // shader widens to double internally
            int32_t causal;
        } pc;
        pc.seq_q    = static_cast<int32_t>(seq_len_q);
        pc.seq_k    = static_cast<int32_t>(seq_len_k);
        pc.head_dim = static_cast<int32_t>(d_k);
        pc.head_v   = static_cast<int32_t>(head_v);
        pc.scale    = scale;
        pc.causal   = causal ? 1 : 0;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, q_contig.data_ptr()},
            {1, k_contig.data_ptr()},
            {2, v_contig.data_ptr()},
            {3, output_flat.data_ptr()},
            {4, lse_flat.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(q_contig.numel()) * sizeof(double),
            static_cast<size_t>(k_contig.numel()) * sizeof(double),
            static_cast<size_t>(v_contig.numel()) * sizeof(double),
            static_cast<size_t>(output_flat.numel()) * sizeof(double),
            static_cast<size_t>(lse_flat.numel()) * sizeof(float),
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);

        const uint32_t Br = 16;
        uint32_t num_q_tiles = static_cast<uint32_t>((seq_len_q + Br - 1) / Br);
        vkCmdDispatch(cmd,
                      num_q_tiles,
                      static_cast<uint32_t>(batch_heads),
                      1);

        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);

        if (has_head_dim) {
            Tensor output = output_flat.reshape({q_shape[0], q_shape[1], seq_len_q, head_v});
            Tensor lse = lse_flat.reshape({q_shape[0], q_shape[1], seq_len_q});
            return {output, lse};
        }
        return {output_flat, lse_flat};
    }

    if (Q.dtype() == DType::Float32
        && K.dtype() == DType::Float32
        && V.dtype() == DType::Float32
        && d_k <= kMaxHeadDimTiled
        && head_v <= kMaxHeadDimTiled
        && seq_len_q > 0 && seq_len_k > 0)
    {
        Tensor q_contig = (q_flat.is_contiguous() && q_flat.offset() == 0) ? q_flat : dispatchContiguous(q_flat);
        Tensor k_contig = (k_flat.is_contiguous() && k_flat.offset() == 0) ? k_flat : dispatchContiguous(k_flat);
        Tensor v_contig = (v_flat.is_contiguous() && v_flat.offset() == 0) ? v_flat : dispatchContiguous(v_flat);

        Tensor output_flat({batch_heads, seq_len_q, head_v},
                           DType::Float32, Q.device());

        // Phase 1.5: LSE buffer (binding 4) — always Float32 per
        // `attention_contract.hpp` `kLseDType`. Shape (batch_heads, seq_q).
        // The shader writes `row_max + log(row_sum)` per row (-INFINITY
        // sentinel for fully-masked rows) in the SAME pass that writes the
        // output. No host-side `logsumexp(Q @ Kᵀ * scale)` recompute, no
        // materialisation of the (B,H,S_q,S_k) attention matrix.
        Tensor lse_flat({batch_heads, seq_len_q},
                       DType::Float32, Q.device());

        int32_t device_id = Q.device().index;
        auto* pipeline = getPipeline("flash_attention", device_id);

        struct PushConstants {
            int32_t seq_q;
            int32_t seq_k;
            int32_t head_dim;
            int32_t head_v;
            float scale;
            int32_t causal;     // M7: passed to shader (audit C2 Vulkan)
        } pc;
        pc.seq_q    = static_cast<int32_t>(seq_len_q);
        pc.seq_k    = static_cast<int32_t>(seq_len_k);
        pc.head_dim = static_cast<int32_t>(d_k);
        pc.head_v   = static_cast<int32_t>(head_v);
        pc.scale    = scale;
        pc.causal   = causal ? 1 : 0;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, q_contig.data_ptr()},
            {1, k_contig.data_ptr()},
            {2, v_contig.data_ptr()},
            {3, output_flat.data_ptr()},
            {4, lse_flat.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(q_contig.numel()) * sizeof(float),
            static_cast<size_t>(k_contig.numel()) * sizeof(float),
            static_cast<size_t>(v_contig.numel()) * sizeof(float),
            static_cast<size_t>(output_flat.numel()) * sizeof(float),
            static_cast<size_t>(lse_flat.numel()) * sizeof(float),
        };
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);

        // Dispatch: (num_q_tiles, batch_heads, 1). Br = 16 matches the
        // shader's local_size_x.
        const uint32_t Br = 16;
        uint32_t num_q_tiles = static_cast<uint32_t>((seq_len_q + Br - 1) / Br);
        vkCmdDispatch(cmd,
                      num_q_tiles,
                      static_cast<uint32_t>(batch_heads),
                      1);

        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);

        // Restore original shape. LSE is (B, H, S_q) for 4D inputs,
        // (B, S_q) for 3D — collapsed batch_heads dim depending on rank.
        if (has_head_dim) {
            Tensor output = output_flat.reshape({q_shape[0], q_shape[1], seq_len_q, head_v});
            Tensor lse = lse_flat.reshape({q_shape[0], q_shape[1], seq_len_q});
            return {output, lse};
        }
        return {output_flat, lse_flat};
    }

    // Step 1: Compute attention scores = Q @ K^T, scaled by 1/sqrt(d_k)
    // K^T is [batch_heads, d_k, seq_len_k]
    Tensor k_transposed = k_flat.transpose(-2, -1);
    Tensor scores = dispatchBmm(q_flat, k_transposed);  // [batch_heads, seq_len_q, seq_len_k]

    // Step 2: Scale by the provided scale factor (typically 1/sqrt(d_k))
    Tensor scale_tensor({1}, scores.dtype(), scores.device());
    scale_tensor.fill_(scale);
    scores = dispatchBinaryOp("mul", scores, scale_tensor);

    // Step 3: Apply causal mask if requested.
    // Per docs/internals/attention-contract.md: sentinel MUST be -INFINITY,
    // never -1e9 (saturates to -65504 in FP16 and leaks gradient mass
    // through softmax).
    if (causal) {
        Tensor mask_val({1}, scores.dtype(), scores.device());
        mask_val.fill_(-std::numeric_limits<float>::infinity());

        // Generate row indices [0..seq_len_q-1] and col indices [0..seq_len_k-1]
        // and mask where col > row
        Tensor row_idx = dispatchArange(0, seq_len_q, 1, DType::Int32, scores.device());
        Tensor col_idx = dispatchArange(0, seq_len_k, 1, DType::Int32, scores.device());

        // Reshape for broadcasting: row_idx [seq_len_q, 1], col_idx [1, seq_len_k]
        row_idx = row_idx.reshape({seq_len_q, 1});
        col_idx = col_idx.reshape({1, seq_len_k});

        // Expand to [seq_len_q, seq_len_k]
        Tensor row_expanded = dispatchExpand(row_idx, {seq_len_q, seq_len_k});
        Tensor col_expanded = dispatchExpand(col_idx, {seq_len_q, seq_len_k});

        // mask = col > row  (upper triangle = future tokens)
        Tensor causal_mask = dispatchComparisonOp("gt", col_expanded, row_expanded);

        // Expand mask to [batch_heads, seq_len_q, seq_len_k]
        causal_mask = causal_mask.unsqueeze(0);
        causal_mask = dispatchExpand(causal_mask, {batch_heads, seq_len_q, seq_len_k});

        // Apply mask: scores = where(mask, -INFINITY, scores)
        Tensor mask_expanded = dispatchExpand(mask_val, std::vector<int64_t>(scores.shape().begin(), scores.shape().end()));
        scores = dispatchWhere(causal_mask, mask_expanded, scores);
    }

    // Step 4: Apply softmax along last dimension
    Tensor attn_weights = dispatchSoftmax(scores, -1);  // [batch_heads, seq_len_q, seq_len_k]

    // Step 4b: LSE from the (masked) pre-softmax scores. Always Float32
    // per the contract (`attention_contract.hpp` `kLseDType`). The composed
    // slow path needs LSE just like the fused fast path so the registry
    // never has to recompute it from a host-side dispatch.
    Tensor lse = tenzor::logsumexp(scores, -1, /*keepdim=*/false);
    if (lse.dtype() != DType::Float32) {
        lse = lse.to(DType::Float32);
    }

    // Step 5: Compute output = attention_weights @ V
    Tensor output = dispatchBmm(attn_weights, v_flat);  // [batch_heads, seq_len_q, d_v]

    // Reshape back to original batch/head layout
    if (has_head_dim) {
        output = output.reshape({q_shape[0], q_shape[1], seq_len_q, V.shape()[3]});
        lse = lse.reshape({q_shape[0], q_shape[1], seq_len_q});
    }

    return {fp16_saturate_if_needed(*this, output), lse};
}

// ---------------------------------------------------------------------------
// Sparse tensor dispatch functions (CSR format via Vulkan compute shaders)
// ---------------------------------------------------------------------------

void VulkanBackend::checkSparseRowDispatch(int32_t device_id, const char* op_name, int64_t M) const {
    uint32_t limit = devices_[device_id].maxComputeWorkGroupCount[0];
    if (M < 0 || static_cast<uint64_t>(M) > static_cast<uint64_t>(limit)) {
        throw std::runtime_error(
            std::string("Vulkan ") + op_name + ": row count M=" + std::to_string(M) +
            " exceeds device maxComputeWorkGroupCount[0]=" + std::to_string(limit) +
            " (one workgroup is dispatched per row). This matrix is too large for the "
            "current Vulkan sparse kernels on this device.");
    }
}

void VulkanBackend::checkSparseIndexFitsI32(const char* op_name, int64_t num_cols,
                                             int64_t nnz) const {
    constexpr int64_t kMaxI32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if (num_cols > kMaxI32 || nnz > kMaxI32) {
        throw std::runtime_error(
            std::string("Vulkan ") + op_name + ": sparse tensor with num_cols=" +
            std::to_string(num_cols) + ", nnz=" + std::to_string(nnz) +
            " exceeds INT32_MAX (" + std::to_string(kMaxI32) + ") — the Vulkan sparse "
            "shaders only accept Int32 CSR indices, and narrowing Int64 col_indices/"
            "crow_indices here would silently wrap out-of-range values into wrong "
            "(aliased) column reads. Use a smaller matrix, or a backend with an "
            "int64-correct sparse path (CPU/ROCm).");
    }
}

auto VulkanBackend::dispatchSparseSpMM(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& dense,
                                        int64_t M, int64_t K, int64_t N) -> Tensor {
    // Wave G6 (deferred → landed): F16/BF16 via widen-narrow through F32.
    // Vulkan has no native half-type sparse shader; widen at the dispatch
    // boundary keeps correctness for downstream half-precision callers.
    if (values.dtype() == DType::Float16 || values.dtype() == DType::BFloat16) {
        DType orig = values.dtype();
        auto vals_f32 = values.to(DType::Float32);
        auto dense_f32 = dense.to(DType::Float32);
        auto result = dispatchSparseSpMM(crow_indices, col_indices, vals_f32,
                                          dense_f32, M, K, N);
        return result.to(orig);
    }
    // F104: no native complex sparse shader on Vulkan. Complex sparse routes
    // GPU->CPU uniformly (matching the codebase's complex-linalg convention):
    // compute on CPU with the native complex SpMM, then move the result back.
    if (values.dtype() == DType::Complex64 || values.dtype() == DType::Complex128) {
        auto sp_cpu = SparseTensor::sparse_csr(crow_indices.to(Device::cpu()),
                                               col_indices.to(Device::cpu()),
                                               values.to(Device::cpu()), {M, K});
        Tensor result = sparse::spmm(sp_cpu, dense.to(Device::cpu()));
        return result.to(values.device());
    }
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64 &&
        values.dtype() != DType::Int32 && values.dtype() != DType::Int64) {
        throw std::runtime_error("Vulkan SpMM only supports F32/F64/F16/BF16/Int32/Int64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    // Native shader per dtype (Int32/Int64 use plain integer arithmetic, no
    // atomics needed since each thread owns a distinct output column).
    bool is_wide = (values.dtype() == DType::Float64 || values.dtype() == DType::Int64);
    std::string shader_name;
    switch (values.dtype()) {
        case DType::Float64: shader_name = "sparse_spmm_f64"; break;
        case DType::Int32:   shader_name = "sparse_spmm_i32"; break;
        case DType::Int64:   shader_name = "sparse_spmm_i64"; break;
        default:              shader_name = "sparse_spmm";    break;
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // M31: guard the narrowing cast below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment).
    checkSparseIndexFitsI32("SpMM", K, col_indices.numel());

    // Convert Int64 indices to Int32 for shader compatibility
    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // The shader indexes the dense operand as flat row-major
    // (B[col_indices[i]*n_cols + col]); a non-default-strided view (e.g. a
    // transpose) with offset 0 passes the descriptor-write guard but would be
    // read with the wrong layout. Materialize a contiguous copy, mirroring
    // dispatchLinalgSolveTriangular.
    Tensor dense_c = (dense.is_contiguous() && dense.offset() == 0) ? dense : dispatchContiguous(dense);

    // The shader reads the CSR values buffer flat; a sliced/non-contiguous
    // values view would be read with the wrong layout. Normalize for
    // consistency with the dense/index operands.
    Tensor values_c = (values.is_contiguous() && values.offset() == 0) ? values : dispatchContiguous(values);

    // One workgroup per row: M can exceed the device X-dimension workgroup-count
    // limit for large sparse matrices, which is an invalid dispatch (validation
    // error / device-lost / silently truncated rows). Fail loudly instead.
    checkSparseRowDispatch(device_id, "SpMM", M);

    // Output: C of shape [M, N]
    Tensor output = dispatchZeros({M, N}, values.dtype(), values.device());

    size_t elem_size = is_wide ? 8 : 4;
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values_c.numel() * elem_size;
    size_t dense_size = dense_c.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values_c.data_ptr()},
        {3, dense_c.data_ptr()}, {4, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, dense_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t m; uint32_t k; uint32_t n_cols; } pc;
    pc.m = static_cast<uint32_t>(M);
    pc.k = static_cast<uint32_t>(K);
    pc.n_cols = static_cast<uint32_t>(N);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchSparseSpMV(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& vec,
                                        int64_t M, int64_t K) -> Tensor {
    // F16/BF16 via widen-narrow through F32 (mirrors dispatchSparseSpMM); Vulkan
    // has no native half-type sparse shader, so widen at the dispatch boundary.
    if (values.dtype() == DType::Float16 || values.dtype() == DType::BFloat16) {
        DType orig = values.dtype();
        auto vals_f32 = values.to(DType::Float32);
        auto vec_f32 = vec.to(DType::Float32);
        auto result = dispatchSparseSpMV(crow_indices, col_indices, vals_f32,
                                          vec_f32, M, K);
        return result.to(orig);
    }
    // F104: complex sparse SpMV -> CPU fallback (no native complex shader).
    if (values.dtype() == DType::Complex64 || values.dtype() == DType::Complex128) {
        auto sp_cpu = SparseTensor::sparse_csr(crow_indices.to(Device::cpu()),
                                               col_indices.to(Device::cpu()),
                                               values.to(Device::cpu()), {M, K});
        Tensor result = sparse::spmv(sp_cpu, vec.to(Device::cpu()));
        return result.to(values.device());
    }
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64 &&
        values.dtype() != DType::Int32 && values.dtype() != DType::Int64) {
        throw std::runtime_error("Vulkan SpMV only supports F32/F64/F16/BF16/Int32/Int64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    // Native shader per dtype (Int32/Int64 use plain integer arithmetic in the
    // shared-memory tree reduction, no atomics needed).
    bool is_wide = (values.dtype() == DType::Float64 || values.dtype() == DType::Int64);
    std::string shader_name;
    switch (values.dtype()) {
        case DType::Float64: shader_name = "sparse_spmv_f64"; break;
        case DType::Int32:   shader_name = "sparse_spmv_i32"; break;
        case DType::Int64:   shader_name = "sparse_spmv_i64"; break;
        default:              shader_name = "sparse_spmv";    break;
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // M31: guard the narrowing cast below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment).
    checkSparseIndexFitsI32("SpMV", K, col_indices.numel());

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // The shader indexes vec as flat (vec[col_indices[i]]); materialize a
    // contiguous copy so a strided/offset view is read with the right layout
    // (mirrors dispatchLinalgSolveTriangular and dispatchSparseSpMM).
    Tensor vec_c = (vec.is_contiguous() && vec.offset() == 0) ? vec : dispatchContiguous(vec);

    // Normalize the CSR values buffer too (read flat by the shader).
    Tensor values_c = (values.is_contiguous() && values.offset() == 0) ? values : dispatchContiguous(values);

    // One workgroup per row: guard M against the device limit (see SpMM).
    checkSparseRowDispatch(device_id, "SpMV", M);

    // Output: y of shape [M]
    Tensor output = dispatchZeros({M}, values.dtype(), values.device());

    size_t elem_size = is_wide ? 8 : 4;
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values_c.numel() * elem_size;
    size_t vec_size = vec_c.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values_c.data_ptr()},
        {3, vec_c.data_ptr()}, {4, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, vec_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchSparseToDense(const Tensor& crow_indices, const Tensor& col_indices,
                                           const Tensor& values, int64_t M, int64_t K,
                                           DType dtype) -> Tensor {
    // Float16/BFloat16: widen values to Float32, compute, narrow back (matches
    // OneAPI). The half front-ends widen before reaching here, but a raw OpId
    // dispatch can still land a half tensor — handle it instead of throwing,
    // consistent with Vulkan SpMM/SpMV which already widen half.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        Tensor out_f32 = dispatchSparseToDense(crow_indices, col_indices,
            values.to(DType::Float32), M, K, DType::Float32);
        return out_f32.to(dtype);
    }
    // F104: complex sparse -> dense on CPU (no native complex shader).
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        auto sp_cpu = SparseTensor::sparse_csr(crow_indices.to(Device::cpu()),
                                               col_indices.to(Device::cpu()),
                                               values.to(Device::cpu()), {M, K});
        return sp_cpu.to_dense().to(values.device());
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("Vulkan SparseToDense only supports Float32/Float64, got " +
            std::string(dtype_name(dtype)));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (dtype == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_to_dense_f64" : "sparse_to_dense";
    auto* pipeline = getPipeline(shader_name, device_id);

    // M31: guard the narrowing cast below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment).
    checkSparseIndexFitsI32("SparseToDense", K, col_indices.numel());

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // The descriptor bindings below require offset-0 (16-byte-aligned) buffers.
    // A sparse component that arrives as an unmaterialized offset view (e.g. from
    // COO->CSR slicing, or a densified Adam grad) would otherwise trip Vulkan's
    // descriptor-offset alignment guard. Materialize each input.
    crow_i32 = dispatchContiguous(crow_i32);
    col_i32 = dispatchContiguous(col_i32);
    Tensor values_mat = dispatchContiguous(values);

    // One workgroup per row: guard M against the device limit (see SpMM).
    checkSparseRowDispatch(device_id, "SparseToDense", M);

    // Output: dense matrix of shape [M, K], zero-initialized
    Tensor output = dispatchZeros({M, K}, dtype, values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values_mat.data_ptr()},
        {3, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchSparseAdd(const Tensor& crow_indices, const Tensor& col_indices,
                                       const Tensor& values, const Tensor& dense,
                                       int64_t M, int64_t K) -> Tensor {
    // Float16/BFloat16: widen values + dense to Float32, compute, narrow back
    // (matches OneAPI; consistent with Vulkan SpMM/SpMV half handling).
    if (values.dtype() == DType::Float16 || values.dtype() == DType::BFloat16) {
        DType orig = values.dtype();
        Tensor out_f32 = dispatchSparseAdd(crow_indices, col_indices,
            values.to(DType::Float32), dense.to(DType::Float32), M, K);
        return out_f32.to(orig);
    }
    // F104: complex sparse + dense on CPU (no native complex shader).
    if (values.dtype() == DType::Complex64 || values.dtype() == DType::Complex128) {
        auto sp_cpu = SparseTensor::sparse_csr(crow_indices.to(Device::cpu()),
                                               col_indices.to(Device::cpu()),
                                               values.to(Device::cpu()), {M, K});
        Tensor result = sparse::add(sp_cpu, dense.to(Device::cpu()));
        return result.to(values.device());
    }
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64 &&
        values.dtype() != DType::Int32 && values.dtype() != DType::Int64) {
        throw std::runtime_error("Vulkan SparseAdd only supports Float32/Float64/Int32/Int64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    // Duplicate CSR entries within a row require atomic accumulation into the
    // output buffer. Int32 uses core atomicAdd (no extension needed). Int64
    // needs a 64-bit CAS loop (sparse_add_i64.comp), which requires the
    // device to advertise VK_KHR_shader_atomic_int64 — mirrors
    // dispatchIndexAdd's Int64/UInt64 gate (clean throw, no CPU fallback).
    if (values.dtype() == DType::Int64 && !devices_[device_id].hasAtomicInt64) {
        throw std::runtime_error(
            "Vulkan SparseAdd with Int64 requires VK_KHR_shader_atomic_int64 "
            "support. Use CPU backend or Int32 for this device.");
    }
    bool is_wide = (values.dtype() == DType::Float64 || values.dtype() == DType::Int64);
    std::string shader_name;
    switch (values.dtype()) {
        case DType::Float64: shader_name = "sparse_add_f64"; break;
        case DType::Int32:   shader_name = "sparse_add_i32"; break;
        case DType::Int64:   shader_name = "sparse_add_i64"; break;
        default:              shader_name = "sparse_add";    break;
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // M31: guard the narrowing cast below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment).
    checkSparseIndexFitsI32("SparseAdd", K, col_indices.numel());

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // One workgroup per row: guard M against the device limit (see SpMM).
    checkSparseRowDispatch(device_id, "SparseAdd", M);

    // Output must be pre-filled with dense values; clone dense into output
    Tensor output = dispatchClone(dense);

    size_t elem_size = is_wide ? 8 : 4;
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchDenseToSparse(const Tensor& dense) -> std::vector<Tensor> {
    if (dense.dim() != 2) {
        throw std::runtime_error("Vulkan DenseToSparse requires a 2D tensor, got " +
            std::to_string(dense.dim()) + "D");
    }
    DType dtype = dense.dtype();
    // Float16/BFloat16: widen the dense input to Float32, compute the CSR, then
    // narrow the values component back (indices stay integer). Matches OneAPI.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto result = dispatchDenseToSparse(dense.to(DType::Float32));
        result[2] = result[2].to(dtype);  // {crow, col, values} — narrow values
        return result;
    }
    // F104: complex dense -> CSR on CPU (no native complex shader). Return the
    // CSR components on the original device with Int64 indices, matching the
    // Float32/Float64 return convention below.
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        auto sp_cpu = tenzor::to_sparse_csr(dense.to(Device::cpu()));
        return { sp_cpu.crow_indices().to(DType::Int64).to(dense.device()),
                 sp_cpu.col_indices().to(DType::Int64).to(dense.device()),
                 sp_cpu.values().to(dense.device()) };
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("Vulkan DenseToSparse only supports Float32/Float64, got " +
            std::string(dtype_name(dtype)));
    }

    int64_t M = dense.shape()[0];
    int64_t K = dense.shape()[1];
    int32_t device_id = dense.device().index;
    bool is_f64 = (dtype == DType::Float64);
    std::string shader_name = is_f64 ? "dense_to_sparse_f64" : "dense_to_sparse";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Passes 1 and 3 dispatch one workgroup per row: guard M (see SpMM).
    checkSparseRowDispatch(device_id, "DenseToSparse", M);

    // Pass 1: count nonzeros per row using the GPU shader
    Tensor row_counts = dispatchZeros({M}, DType::Int32, dense.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t dense_size = dense.numel() * elem_size;
    size_t row_counts_size = row_counts.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, dense.data_ptr()}, {1, row_counts.data_ptr()},
    };
    std::vector<size_t> sizes = {dense_size, row_counts_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Pass 2: GPU prefix sum on row_counts → crow_indices
    Tensor crow_indices = dispatchZeros({M + 1}, DType::Int32, dense.device());
    {
        auto* ps_pipeline = getPipeline("csr_prefix_sum", device_id);
        size_t rc_size = row_counts.numel() * sizeof(int32_t);
        size_t ci_size = crow_indices.numel() * sizeof(int32_t);
        std::vector<std::pair<uint32_t, const void*>> ps_bindings = {
            {0, row_counts.data_ptr()}, {1, crow_indices.data_ptr()},
        };
        std::vector<size_t> ps_sizes = {rc_size, ci_size};
        VkDescriptorSet ps_ds = allocateAndWriteDescriptorSet(device_id, ps_pipeline, ps_bindings, ps_sizes);

        struct { uint32_t n_rows; } ps_pc;
        ps_pc.n_rows = static_cast<uint32_t>(M);

        VkCommandBuffer ps_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(ps_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps_pipeline->pipeline());
        vkCmdBindDescriptorSets(ps_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ps_pipeline->layout(), 0, 1, &ps_ds, 0, nullptr);
        vkCmdPushConstants(ps_cmd, ps_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ps_pc), &ps_pc);
        vkCmdDispatch(ps_cmd, 1, 1, 1);
        insertComputeOnlyBarrier(ps_cmd);
        endSingleTimeCommands(ps_cmd, device_id);
    }
    synchronize(device_id);

    // Minimal scalar readback (single int32, 4 bytes) — NOT a CPU computation fallback.
    // This is the minimum GPU->CPU sync required for variable-size output allocation in Vulkan.
    Tensor nnz_scalar = crow_indices.slice(0, M, M + 1).to(Device::cpu());
    int64_t nnz = static_cast<int64_t>(nnz_scalar.data<int32_t>()[0]);

    if (nnz == 0) {
        // All zeros — return empty CSR
        Tensor crow_out = crow_indices.to(DType::Int64);
        Tensor col_out({0}, DType::Int64, dense.device());
        Tensor val_out({0}, dtype, dense.device());
        return {crow_out, col_out, val_out};
    }

    // Pass 3: GPU extract — scatter col_indices and values using crow_indices offsets
    Tensor col_indices_gpu({nnz}, DType::Int32, dense.device());
    Tensor values_gpu({nnz}, dtype, dense.device());
    {
        std::string extract_shader = is_f64 ? "csr_extract_f64" : "csr_extract";
        auto* ex_pipeline = getPipeline(extract_shader, device_id);

        size_t ci_size = crow_indices.numel() * sizeof(int32_t);
        size_t col_size = col_indices_gpu.numel() * sizeof(int32_t);
        size_t val_size = values_gpu.numel() * elem_size;

        std::vector<std::pair<uint32_t, const void*>> ex_bindings = {
            {0, dense.data_ptr()},
            {1, crow_indices.data_ptr()},
            {2, col_indices_gpu.data_ptr()},
            {3, values_gpu.data_ptr()},
        };
        std::vector<size_t> ex_sizes = {dense_size, ci_size, col_size, val_size};
        VkDescriptorSet ex_ds = allocateAndWriteDescriptorSet(device_id, ex_pipeline, ex_bindings, ex_sizes);

        struct { uint32_t n_rows; uint32_t n_cols; } ex_pc;
        ex_pc.n_rows = static_cast<uint32_t>(M);
        ex_pc.n_cols = static_cast<uint32_t>(K);

        VkCommandBuffer ex_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(ex_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ex_pipeline->pipeline());
        vkCmdBindDescriptorSets(ex_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ex_pipeline->layout(), 0, 1, &ex_ds, 0, nullptr);
        vkCmdPushConstants(ex_cmd, ex_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ex_pc), &ex_pc);
        vkCmdDispatch(ex_cmd, static_cast<uint32_t>(M), 1, 1);
        insertComputeOnlyBarrier(ex_cmd);
        endSingleTimeCommands(ex_cmd, device_id);
    }

    // Convert Int32 outputs to Int64 for CSR format compatibility
    Tensor crow_out = crow_indices.to(DType::Int64);
    Tensor col_out = col_indices_gpu.to(DType::Int64);

    return {crow_out, col_out, values_gpu};
}

// ============================================================================
// Cross Product
// ============================================================================

auto VulkanBackend::dispatchCross(const Tensor& a, const Tensor& b,
                                   int64_t dim) -> Tensor {
    auto shape = a.shape();
    int64_t ndim = shape.size();
    std::vector<int64_t> out_shape(shape.begin(), shape.end());

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t num_pairs = outer * inner;

    Tensor output(out_shape, a.dtype(), a.device());
    if (num_pairs == 0) return output;

    int32_t device_id = a.device().index;
    bool is_float64 = (a.dtype() == DType::Float64);
    bool is_float16 = (a.dtype() == DType::Float16);
    bool is_bfloat16 = (a.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "cross_f64" : is_float16 ? "cross_f16" : is_bfloat16 ? "cross_bf16" : "cross";
    auto* pipeline = getPipeline(shader_name, device_id);

    struct PushConstants {
        uint32_t num_pairs;
        uint32_t dim_stride;
    } push_constants;
    push_constants.num_pairs = static_cast<uint32_t>(num_pairs);
    push_constants.dim_stride = static_cast<uint32_t>(inner);

    size_t total_elems = a.numel();
    size_t elem_size = a.dtype_size();
    size_t buffer_size = total_elems * elem_size;
    if (is_float16 || is_bfloat16) {
        buffer_size = ((total_elems + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, a.data_ptr()}, {1, b.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg_checked(static_cast<uint32_t>(num_pairs), devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch");
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// LU Decomposition: factor A = P L U with partial pivoting.
// Output: (L, U, pivots) where L is unit lower-triangular, U is upper-triangular,
// pivots is Int32 tensor of shape (..., n) containing 1-based LAPACK pivot indices.
// ============================================================================
auto VulkanBackend::dispatchLinalgLU(const Tensor& input) -> std::vector<Tensor> {
    // Note: BFloat16 is valid here (promoted to Float32 below), unlike the other
    // linalg entry points which reject it outright — validate only Float64/Float16/
    // Float32/BFloat16.
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::BFloat16) {
        throw std::invalid_argument(
            "linalg::lu: unsupported dtype " + std::string(dtype_name(input.dtype())) +
            " on Vulkan. Supported: Float32, Float64, Float16, BFloat16.");
    }
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg.lu: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::invalid_argument("linalg.lu: input must be square");

    int32_t device_id = input.device().index;
    DType out_dtype = input.dtype();
    // LU is numerically unstable in Float16/BFloat16 — promote to Float32 for the factorization.
    bool needs_promote = (out_dtype == DType::Float16 || out_dtype == DType::BFloat16);
    DType work_dtype = needs_promote ? DType::Float32 : out_dtype;

    Tensor work = input;
    if (needs_promote) work = dispatchCast(work, DType::Float32);

    bool is_f64 = (work_dtype == DType::Float64);
    bool is_f16 = false;  // we always promote

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Clone so runBlockedLU can modify in place
    Tensor A = dispatchClone(work.contiguous());

    // Pivot buffer: flat [batch_size * n] Int32
    Tensor pivots_flat({batch_size, n}, DType::Int32, input.device());

    runBlockedLU(A, pivots_flat, n, batch_size, device_id, is_f64, is_f16);

    // Split packed LU into L (unit lower) and U (upper) via a dedicated shader.
    std::vector<int64_t> mat_shape(shape.begin(), shape.end());
    Tensor L(mat_shape, work_dtype, input.device());
    Tensor U(mat_shape, work_dtype, input.device());

    {
        std::string shader = is_f64 ? "linalg_lu_split_f64" : "linalg_lu_split";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch_size;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch_size = static_cast<uint32_t>(batch_size);

        size_t elem_size = is_f64 ? 8 : 4;
        size_t mat_size = static_cast<size_t>(batch_size) * n * n * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, A.data_ptr()}, {1, L.data_ptr()}, {2, U.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, mat_size, mat_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        uint32_t total = static_cast<uint32_t>(batch_size * n * n);
        uint32_t wg_size = devices_[device_id].workgroupSize;
        uint32_t groups = (total + wg_size - 1) / wg_size;
        vkCmdDispatch(cmd, groups, 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // Reshape pivots to match input batch shape + {n}. runBlockedLU emits 0-based
    // absolute row indices (its internal convention, consumed by linalg_trsm /
    // linalg_det_from_lu via their own runBlockedLU calls). The PUBLIC lu() API
    // contract — matching CPU/CUDA and documented in linalg.hpp — is 1-based
    // LAPACK ipiv. Convert the returned pivots to 1-based; dispatchLinalgLUSolve
    // converts them back to 0-based for the trsm shader.
    std::vector<int64_t> pivots_shape(shape.begin(), shape.end() - 2);
    pivots_shape.push_back(n);
    Tensor pivots_out = dispatchReshape(pivots_flat, pivots_shape);
    {
        // pivots_out += 1 (Int32). Build a ones tensor on-device and add.
        Tensor one = dispatchFull(pivots_shape, 1.0, DType::Int32).to(input.device());
        pivots_out = dispatchBinaryOp("add", pivots_out, one);
    }

    // Downcast L and U back to original dtype if we promoted
    if (needs_promote) {
        L = dispatchCast(L, out_dtype);
        U = dispatchCast(U, out_dtype);
    }

    return {L, U, pivots_out};
}

// ============================================================================
// LU Solve: given packed LU factors and pivots, solve AX = B.
// Reuses the existing linalg_trsm shader which consumes the packed LU format.
// The input pivots are 1-based LAPACK convention (from dispatchLinalgLU and the
// public lu() API), but linalg_trsm expects the 0-based internal convention used
// by runBlockedLU — convert by subtracting 1 (done below).
// ============================================================================
auto VulkanBackend::dispatchLinalgLUSolve(const Tensor& LU_data, const Tensor& pivots,
                                          const Tensor& B) -> Tensor {
    validate_linalg_dtype(LU_data, "lu_solve");
    auto lu_shape = LU_data.shape();
    auto b_shape = B.shape();
    int64_t lu_ndim = static_cast<int64_t>(lu_shape.size());
    int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    if (lu_ndim < 2) throw std::invalid_argument("linalg.lu_solve: LU must be at least 2D");
    if (b_ndim < 2) throw std::invalid_argument("linalg.lu_solve: B must be at least 2D");

    int64_t n = lu_shape[lu_ndim - 1];
    if (lu_shape[lu_ndim - 2] != n) throw std::invalid_argument("linalg.lu_solve: LU must be square");
    int64_t nrhs = b_shape[b_ndim - 1];

    int32_t device_id = LU_data.device().index;
    DType out_dtype = LU_data.dtype();
    bool needs_promote = (out_dtype == DType::Float16 || out_dtype == DType::BFloat16);
    DType work_dtype = needs_promote ? DType::Float32 : out_dtype;

    Tensor lu = LU_data;
    Tensor bmat = B;
    if (needs_promote) {
        lu = dispatchCast(lu, DType::Float32);
        bmat = dispatchCast(bmat, DType::Float32);
    }

    bool is_f64 = (work_dtype == DType::Float64);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < lu_ndim - 2; ++i) batch_size *= lu_shape[i];

    // The public lu() API returns 1-based LAPACK pivots (dispatchLinalgLU), but
    // linalg_trsm consumes runBlockedLU's 0-based convention — convert back by
    // subtracting 1.
    Tensor pivots_zero = pivots.contiguous();
    {
        auto ps = pivots_zero.shape();
        std::vector<int64_t> piv_shape(ps.begin(), ps.end());
        Tensor one = dispatchFull(piv_shape, 1.0, DType::Int32).to(pivots_zero.device());
        pivots_zero = dispatchBinaryOp("sub", pivots_zero, one);
    }

    // Dispatch TRSM shader (identical pattern to dispatchLinalgSolve tiled path)
    auto lu_cont = lu.contiguous();
    auto b_cont = bmat.contiguous();
    checkLuNonSingular(lu_cont, n, batch_size, is_f64, "lu_solve");
    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), work_dtype, LU_data.device());

    std::string trsm_shader = is_f64 ? "linalg_trsm_f64" : "linalg_trsm";
    auto* trsm_pipeline = getPipeline(trsm_shader, device_id);

    struct TrsmPC {
        uint32_t n_dim;
        uint32_t nrhs_cnt;
        uint32_t lda;
        uint32_t ldb_val;
        uint32_t batch_cnt;
    } trsm_pc;
    trsm_pc.n_dim = static_cast<uint32_t>(n);
    trsm_pc.nrhs_cnt = static_cast<uint32_t>(nrhs);
    trsm_pc.lda = static_cast<uint32_t>(n);
    trsm_pc.ldb_val = static_cast<uint32_t>(nrhs);
    trsm_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : 4;
    size_t lu_sz  = static_cast<size_t>(batch_size) * n * n * elem_size;
    size_t piv_sz = static_cast<size_t>(batch_size) * n * sizeof(int32_t);
    size_t b_sz   = static_cast<size_t>(batch_size) * n * nrhs * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, lu_cont.data_ptr()}, {1, pivots_zero.data_ptr()},
        {2, b_cont.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {lu_sz, piv_sz, b_sz, b_sz};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, trsm_pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trsm_pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           trsm_pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, trsm_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(trsm_pc), &trsm_pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    if (needs_promote) output = dispatchCast(output, out_dtype);
    return output;
}

// ============================================================================
// Sparse SpGEMM: C = A * B (CSR x CSR -> CSR)
// ============================================================================

auto VulkanBackend::dispatchSparseSpGEMM(const Tensor& a_crow, const Tensor& a_col,
                                          const Tensor& a_vals,
                                          const Tensor& b_crow, const Tensor& b_col,
                                          const Tensor& b_vals,
                                          int64_t M, int64_t K, int64_t N) -> std::vector<Tensor> {
    // F104: complex CSR x CSR on CPU (no native complex shader). Return the CSR
    // components on the original device with Int64 indices, matching the
    // Float32/Float64 return convention below.
    if (a_vals.dtype() == DType::Complex64 || a_vals.dtype() == DType::Complex128) {
        Device dev = a_vals.device();
        auto a_cpu = SparseTensor::sparse_csr(a_crow.to(Device::cpu()),
                                              a_col.to(Device::cpu()),
                                              a_vals.to(Device::cpu()), {M, K});
        auto b_cpu = SparseTensor::sparse_csr(b_crow.to(Device::cpu()),
                                              b_col.to(Device::cpu()),
                                              b_vals.to(Device::cpu()), {K, N});
        auto c_cpu = sparse::spgemm(a_cpu, b_cpu);
        return { c_cpu.crow_indices().to(DType::Int64).to(dev),
                 c_cpu.col_indices().to(DType::Int64).to(dev),
                 c_cpu.values().to(dev) };
    }
    if (a_vals.dtype() != DType::Float32 && a_vals.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SpGEMM only supports Float32/Float64, got " +
            std::string(dtype_name(a_vals.dtype())));
    }
    int32_t device_id = a_vals.device().index;
    bool is_f64 = (a_vals.dtype() == DType::Float64);
    DType dtype = a_vals.dtype();

    // The count (pass 1) and fill (pass 3) passes dispatch one workgroup per row
    // of A: guard M against the device limit (see SpMM).
    checkSparseRowDispatch(device_id, "SpGEMM", M);

    // M31: guard the narrowing casts below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment). A is (M,K),
    // B is (K,N): A's col_indices range over K, B's range over N.
    checkSparseIndexFitsI32("SpGEMM (A)", K, a_col.numel());
    checkSparseIndexFitsI32("SpGEMM (B)", N, b_col.numel());

    // Convert Int64 indices to Int32 for shader compatibility
    auto a_crow_i32 = (a_crow.dtype() == DType::Int32) ? a_crow : a_crow.to(DType::Int32);
    auto a_col_i32 = (a_col.dtype() == DType::Int32) ? a_col : a_col.to(DType::Int32);
    auto b_crow_i32 = (b_crow.dtype() == DType::Int32) ? b_crow : b_crow.to(DType::Int32);
    auto b_col_i32 = (b_col.dtype() == DType::Int32) ? b_col : b_col.to(DType::Int32);

    // The fill shader reads the CSR values buffers flat; normalize any
    // sliced/non-contiguous values view so it is read with the right layout.
    Tensor a_vals_c = (a_vals.is_contiguous() && a_vals.offset() == 0) ? a_vals : dispatchContiguous(a_vals);
    Tensor b_vals_c = (b_vals.is_contiguous() && b_vals.offset() == 0) ? b_vals : dispatchContiguous(b_vals);

    // --- Pass 1: Count nnz per row ---
    auto* count_pipeline = getPipeline("sparse_spgemm_count", device_id);
    Tensor row_nnz = dispatchZeros({M}, DType::Int32, a_vals.device());

    {
        size_t a_crow_size = a_crow_i32.numel() * sizeof(int32_t);
        size_t a_col_size = a_col_i32.numel() * sizeof(int32_t);
        size_t b_crow_size = b_crow_i32.numel() * sizeof(int32_t);
        size_t b_col_size = b_col_i32.numel() * sizeof(int32_t);
        size_t row_nnz_size = row_nnz.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, a_crow_i32.data_ptr()}, {1, a_col_i32.data_ptr()},
            {2, b_crow_i32.data_ptr()}, {3, b_col_i32.data_ptr()},
            {4, row_nnz.data_ptr()},
        };
        std::vector<size_t> sizes = {a_crow_size, a_col_size, b_crow_size, b_col_size, row_nnz_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, count_pipeline, bindings, sizes);

        struct { uint32_t m; uint32_t k; uint32_t n_cols; } pc;
        pc.m = static_cast<uint32_t>(M);
        pc.k = static_cast<uint32_t>(K);
        pc.n_cols = static_cast<uint32_t>(N);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, count_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               count_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, count_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    // --- Pass 2: Prefix sum on row_nnz -> c_crow_indices ---
    Tensor c_crow = dispatchZeros({M + 1}, DType::Int32, a_vals.device());
    {
        auto* ps_pipeline = getPipeline("csr_prefix_sum", device_id);
        size_t rc_size = row_nnz.numel() * sizeof(int32_t);
        size_t ci_size = c_crow.numel() * sizeof(int32_t);
        std::vector<std::pair<uint32_t, const void*>> ps_bindings = {
            {0, row_nnz.data_ptr()}, {1, c_crow.data_ptr()},
        };
        std::vector<size_t> ps_sizes = {rc_size, ci_size};
        VkDescriptorSet ps_ds = allocateAndWriteDescriptorSet(device_id, ps_pipeline, ps_bindings, ps_sizes);

        struct { uint32_t n_rows; } ps_pc;
        ps_pc.n_rows = static_cast<uint32_t>(M);

        VkCommandBuffer ps_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(ps_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps_pipeline->pipeline());
        vkCmdBindDescriptorSets(ps_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ps_pipeline->layout(), 0, 1, &ps_ds, 0, nullptr);
        vkCmdPushConstants(ps_cmd, ps_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ps_pc), &ps_pc);
        vkCmdDispatch(ps_cmd, 1, 1, 1);
        insertComputeOnlyBarrier(ps_cmd);
        endSingleTimeCommands(ps_cmd, device_id);
    }
    synchronize(device_id);

    // Read total nnz from c_crow[M]
    Tensor nnz_scalar = c_crow.slice(0, M, M + 1).to(Device::cpu());
    int64_t total_nnz = static_cast<int64_t>(nnz_scalar.data<int32_t>()[0]);

    if (total_nnz == 0) {
        Tensor crow_out = c_crow.to(DType::Int64);
        Tensor col_out({0}, DType::Int64, a_vals.device());
        Tensor val_out({0}, dtype, a_vals.device());
        return {crow_out, col_out, val_out};
    }

    // --- Pass 3: Fill col_indices and values ---
    std::string fill_shader = is_f64 ? "sparse_spgemm_fill_f64" : "sparse_spgemm_fill";
    auto* fill_pipeline = getPipeline(fill_shader, device_id);

    // Zero-initialize in case some rows produce fewer nnz than counted
    Tensor c_col_gpu = dispatchZeros({total_nnz}, DType::Int32, a_vals.device());
    Tensor c_vals_gpu = dispatchZeros({total_nnz}, dtype, a_vals.device());

    {
        size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
        size_t a_crow_size = a_crow_i32.numel() * sizeof(int32_t);
        size_t a_col_size = a_col_i32.numel() * sizeof(int32_t);
        size_t a_vals_size = a_vals_c.numel() * elem_size;
        size_t b_crow_size = b_crow_i32.numel() * sizeof(int32_t);
        size_t b_col_size = b_col_i32.numel() * sizeof(int32_t);
        size_t b_vals_size = b_vals_c.numel() * elem_size;
        size_t c_crow_size = c_crow.numel() * sizeof(int32_t);
        size_t c_col_size = c_col_gpu.numel() * sizeof(int32_t);
        size_t c_vals_size = c_vals_gpu.numel() * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, a_crow_i32.data_ptr()}, {1, a_col_i32.data_ptr()}, {2, a_vals_c.data_ptr()},
            {3, b_crow_i32.data_ptr()}, {4, b_col_i32.data_ptr()}, {5, b_vals_c.data_ptr()},
            {6, c_crow.data_ptr()}, {7, c_col_gpu.data_ptr()}, {8, c_vals_gpu.data_ptr()},
        };
        std::vector<size_t> sizes = {
            a_crow_size, a_col_size, a_vals_size,
            b_crow_size, b_col_size, b_vals_size,
            c_crow_size, c_col_size, c_vals_size,
        };

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, fill_pipeline, bindings, sizes);

        struct { uint32_t m; uint32_t k; uint32_t n_cols; } pc;
        pc.m = static_cast<uint32_t>(M);
        pc.k = static_cast<uint32_t>(K);
        pc.n_cols = static_cast<uint32_t>(N);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, fill_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }

    Tensor crow_out = c_crow.to(DType::Int64);
    Tensor col_out = c_col_gpu.to(DType::Int64);
    return {crow_out, col_out, c_vals_gpu};
}

// ============================================================================
// Sparse Triangular Solve: L*x = b (SparseTrsv)
// ============================================================================

auto VulkanBackend::dispatchSparseTrsv(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& b,
                                        int64_t N, bool upper) -> Tensor {
    // Complex64/Complex128 get genuine GPU compute via dedicated shader
    // variants (sparse_trsv_c64 / sparse_trsv_c128) below — same
    // single-workgroup sequential substitution as the real shaders, with
    // complex multiply-subtract and complex divide. Values/RHS/Output are
    // interleaved [re,im] float (Complex64) or float64_t (Complex128) pairs,
    // matching every other complex Vulkan kernel's layout (fft.comp,
    // complex_from_parts.comp, conj_f64.comp).
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64 &&
        values.dtype() != DType::Complex64 && values.dtype() != DType::Complex128) {
        throw std::runtime_error("Vulkan SparseTrsv only supports Float32/Float64/Complex64/Complex128, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_complex = (values.dtype() == DType::Complex64 || values.dtype() == DType::Complex128);
    bool is_f64 = (values.dtype() == DType::Float64 || values.dtype() == DType::Complex128);
    std::string shader_name;
    if (is_complex) {
        shader_name = (values.dtype() == DType::Complex128) ? "sparse_trsv_c128" : "sparse_trsv_c64";
    } else {
        shader_name = is_f64 ? "sparse_trsv_f64" : "sparse_trsv";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // M31: guard the narrowing cast below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment). Square N x N.
    checkSparseIndexFitsI32("SparseTrsv", N, col_indices.numel());

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // The shader reads the CSR values and the RHS b flat (b[row]); a
    // sliced/non-contiguous/offset view would be read with the wrong layout.
    // Materialize contiguous copies, mirroring SpMV.
    Tensor values_c = (values.is_contiguous() && values.offset() == 0) ? values : dispatchContiguous(values);
    Tensor b_c = (b.is_contiguous() && b.offset() == 0) ? b : dispatchContiguous(b);

    // Output: x of shape [N]
    Tensor output = dispatchZeros({N}, values.dtype(), values.device());

    // Solved flags: one int per row, zero-initialized
    Tensor solved = dispatchZeros({N}, DType::Int32, values.device());

    // Complex elements are 2 interleaved lanes (re,im) of the base float/double
    // width, matching the Complex64=8B / Complex128=16B convention used
    // throughout the Vulkan backend (e.g. vulkan_ops_fft.cpp's elem_size).
    size_t elem_size = (is_f64 ? sizeof(double) : sizeof(float)) * (is_complex ? 2 : 1);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values_c.numel() * elem_size;
    size_t b_size = b_c.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;
    size_t solved_size = solved.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values_c.data_ptr()},
        {3, b_c.data_ptr()}, {4, output.data_ptr()}, {5, solved.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, b_size, output_size, solved_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n; uint32_t upper; } pc;
    pc.n = static_cast<uint32_t>(N);
    pc.upper = upper ? 1u : 0u;

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // Single (1,1,1) workgroup: the shader now performs the entire triangular
    // solve sequentially from one invocation. The previous N-workgroup,
    // one-row-per-workgroup spin-wait could deadlock (no Vulkan cross-workgroup
    // forward-progress guarantee).
    vkCmdDispatch(cmd, 1, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Sparse Triangular Solve Multi-RHS: L*X = B (SparseTrsm)
// ============================================================================

auto VulkanBackend::dispatchSparseTrsm(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& B,
                                        int64_t N, int64_t K_rhs, bool upper) -> Tensor {
    // Complex64/Complex128 get genuine GPU compute via dedicated shader
    // variants (sparse_trsm_c64 / sparse_trsm_c128) below — same
    // single-workgroup, columns-parallel substitution as the real shaders,
    // with complex multiply-subtract and complex divide. Values/RHS/Output
    // are interleaved [re,im] float (Complex64) or float64_t (Complex128)
    // pairs, matching every other complex Vulkan kernel's layout (fft.comp,
    // complex_from_parts.comp, conj_f64.comp).
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64 &&
        values.dtype() != DType::Complex64 && values.dtype() != DType::Complex128) {
        throw std::runtime_error("Vulkan SparseTrsm only supports Float32/Float64/Complex64/Complex128, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_complex = (values.dtype() == DType::Complex64 || values.dtype() == DType::Complex128);
    bool is_f64 = (values.dtype() == DType::Float64 || values.dtype() == DType::Complex128);
    std::string shader_name;
    if (is_complex) {
        shader_name = (values.dtype() == DType::Complex128) ? "sparse_trsm_c128" : "sparse_trsm_c64";
    } else {
        shader_name = is_f64 ? "sparse_trsm_f64" : "sparse_trsm";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // M31: guard the narrowing cast below against silent Int64->Int32
    // wraparound (see checkSparseIndexFitsI32's doc comment). Square N x N.
    checkSparseIndexFitsI32("SparseTrsm", N, col_indices.numel());

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // The shader reads the CSR values and the RHS B flat (row-major); a
    // sliced/non-contiguous/offset view would be read with the wrong layout.
    // Materialize contiguous copies, mirroring SpMV / SparseTrsv.
    Tensor values_c = (values.is_contiguous() && values.offset() == 0) ? values : dispatchContiguous(values);
    Tensor B_c = (B.is_contiguous() && B.offset() == 0) ? B : dispatchContiguous(B);

    // Output: X of shape [N, K_rhs]
    Tensor output = dispatchZeros({N, K_rhs}, values.dtype(), values.device());

    // Solved flags: one int per row, zero-initialized
    Tensor solved = dispatchZeros({N}, DType::Int32, values.device());

    // Complex elements are 2 interleaved lanes (re,im) of the base float/double
    // width, matching the Complex64=8B / Complex128=16B convention used
    // throughout the Vulkan backend (e.g. vulkan_ops_fft.cpp's elem_size).
    size_t elem_size = (is_f64 ? sizeof(double) : sizeof(float)) * (is_complex ? 2 : 1);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values_c.numel() * elem_size;
    size_t B_size = B_c.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;
    size_t solved_size = solved.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values_c.data_ptr()},
        {3, B_c.data_ptr()}, {4, output.data_ptr()}, {5, solved.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, B_size, output_size, solved_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n; uint32_t k_rhs; uint32_t upper; } pc;
    pc.n = static_cast<uint32_t>(N);
    pc.k_rhs = static_cast<uint32_t>(K_rhs);
    pc.upper = upper ? 1u : 0u;

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // Single-workgroup sequential solve (matches sparse_trsv): the shader
    // processes rows in strict dependency order within one workgroup, so we
    // must dispatch exactly one workgroup. Dispatching N workgroups would
    // reintroduce the cross-workgroup spin-wait deadlock.
    vkCmdDispatch(cmd, 1, 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}


// ============================================================================
// Triangular Solve (AX = B, A triangular) — native Vulkan compute shader
// ============================================================================

auto VulkanBackend::dispatchLinalgSolveTriangular(const Tensor& A, const Tensor& B,
                                                   bool upper, bool unitriangular) -> Tensor {
    // AUTOGRAD-R044: reject unsupported dtypes (notably Complex64/128) up
    // front. Without this, the widen-and-recurse branch below would call
    // `A.to(DType::Float32)` on a complex tensor, which silently narrows to
    // the real component instead of failing loud.
    validate_linalg_dtype(A, "solve_triangular");
    // Half-precision dtypes widen to Float32; Float64 uses a dedicated shader
    // to keep double precision through cholesky_inverse backward.
    if (A.dtype() != DType::Float32 && A.dtype() != DType::Float64) {
        auto a_f32 = A.to(DType::Float32);
        auto b_f32 = B.to(DType::Float32);
        return dispatchLinalgSolveTriangular(a_f32, b_f32, upper, unitriangular).to(A.dtype());
    }

    // The shader indexes A/B as flat row-major (A_data[i*N+j]), so a
    // non-contiguous view (e.g. a transpose) would be read with the wrong
    // strides. cholesky_inverse passes transpose(L) directly, which caused
    // the second solve to read L instead of L^T and produce zeros in the
    // upper triangle of the result. Materialize contiguous copies up front.
    Tensor A_c = (A.is_contiguous() && A.offset() == 0) ? A : dispatchContiguous(A);
    Tensor B_c = (B.is_contiguous() && B.offset() == 0) ? B : dispatchContiguous(B);

    int32_t device_id = A_c.device().index;
    bool is_f64 = (A_c.dtype() == DType::Float64);
    auto* pipeline = getPipeline(is_f64 ? "solve_triangular_f64" : "solve_triangular", device_id);

    auto a_shape = A_c.shape();
    auto b_shape = B_c.shape();
    int64_t a_ndim = static_cast<int64_t>(a_shape.size());
    int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    uint32_t N = static_cast<uint32_t>(a_shape[a_shape.size() - 1]);
    // Disambiguate a matrix RHS (..., N, nrhs) from a batched vector RHS
    // (..., N) the same way dispatchLinalgSolve's small path does: only
    // treat the trailing dim as nrhs when the second-to-last dim equals N.
    // Without this guard a batched vector RHS shaped (batch, N) with
    // batch != N would wrongly yield M == N and make the shader over-read B.
    uint32_t M = (b_ndim >= 2 && b_shape[b_ndim - 2] == static_cast<int64_t>(N))
                     ? static_cast<uint32_t>(b_shape[b_ndim - 1])
                     : 1;

    // Batch dimension: product of all leading dims of A before the trailing
    // N x N matrix. The shader indexes A/B/X with a per-batch offset, so
    // this must match the batch layout B (and the output, which mirrors B's
    // shape) was flattened with above.
    int64_t batch_size = 1;
    for (int64_t i = 0; i < a_ndim - 2; ++i) batch_size *= a_shape[i];

    // The shader divides by the diagonal with no epsilon guard and no
    // error-reporting path back to the host, so a singular (zero-diagonal)
    // triangular matrix would otherwise silently produce Inf/NaN instead of
    // throwing. Matches the CPU backend's solve_triangular, which rejects a
    // zero diagonal up front with the same diagnostic (src/ops/linalg.cpp).
    if (!unitriangular) {
        Tensor a_host = A_c.to(Device::cpu());
        auto check_zero_diag = [&](auto* a_data) {
            for (int64_t b = 0; b < batch_size; ++b) {
                auto* A_mat = a_data + b * N * N;
                for (uint32_t i = 0; i < N; ++i) {
                    if (A_mat[i * N + i] == 0) {
                        throw std::runtime_error(
                            "linalg::solve_triangular: zero diagonal element at row " +
                            std::to_string(i));
                    }
                }
            }
        };
        if (is_f64) {
            check_zero_diag(a_host.data<double>());
        } else {
            check_zero_diag(a_host.data<float>());
        }
    }

    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), B_c.dtype(), B_c.device());

    struct { uint32_t N; uint32_t M; uint32_t upper; uint32_t unitriangular; uint32_t batch; } pc;
    pc.N = N;
    pc.M = M;
    pc.upper = upper ? 1 : 0;
    pc.unitriangular = unitriangular ? 1 : 0;
    pc.batch = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t a_buf = static_cast<size_t>(A_c.numel()) * elem_size;
    size_t b_buf = static_cast<size_t>(B_c.numel()) * elem_size;
    size_t x_buf = static_cast<size_t>(output.numel()) * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, A_c.data_ptr()}, {1, B_c.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {a_buf, b_buf, x_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg_checked(M, devices_[device_id].workgroupSize, devices_[device_id].maxComputeWorkGroupCount[0], "vk_dispatch"),
                  static_cast<uint32_t>(batch_size), 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Geqrf — raw QR factorization returning packed reflectors + tau (Vulkan)
// Uses the existing runBlockedQR which already produces exactly this form.
// ============================================================================

auto VulkanBackend::dispatchGeqrf(const Tensor& input) -> std::vector<Tensor> {
    validate_linalg_dtype(input, "geqrf");
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.geqrf: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Both small and large matrix paths use runBlockedQR which produces
    // packed Householder reflectors (below diagonal) + R (on/above diagonal) + tau.
    Tensor A = dispatchClone(input.contiguous());

    // runBlockedQR expects tau with n entries; we allocate for min(m,n) = k
    // and pass n to the shader (extra entries beyond k will be unused).
    // Actually, runBlockedQR iterates up to k = min(m,n) columns internally,
    // so we allocate n entries (the shader indexes by column) but only k are meaningful.
    std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
    tau_shape.push_back(n);
    Tensor tau(tau_shape, input.dtype(), input.device());

    runBlockedQR(A, tau, m, n, batch_size, device_id, is_f64, is_f16);

    // Return A (packed reflectors + R) and tau (first k entries are meaningful).
    // The caller (linalg::geqrf) expects tau of shape batch_dims + {min(m,n)}.
    // Since the extra entries are zero-initialized and unused, returning the
    // full n-sized tau is safe -- the dispatch wrapper can trim if needed.
    return {A, tau};
}

// ============================================================================
// Ormqr — multiply matrix by Q from QR factorization using tau (Vulkan)
// Uses Householder reflectors stored in packed form.
// For small matrices, applies reflectors one at a time via the qr_update shader.
// ============================================================================

auto VulkanBackend::dispatchOrmqr(const Tensor& reflectors, const Tensor& tau,
                                   const Tensor& C, bool left, bool transpose_q) -> Tensor {
    validate_linalg_dtype(C, "ormqr");
    auto c_shape = C.shape();
    auto r_shape = reflectors.shape();
    int64_t c_ndim = static_cast<int64_t>(c_shape.size());
    int64_t r_ndim = static_cast<int64_t>(r_shape.size());

    if (c_ndim < 2) throw std::runtime_error("linalg.ormqr: C must be at least 2D");
    if (r_ndim < 2) throw std::runtime_error("linalg.ormqr: reflectors must be at least 2D");

    int64_t c_m = c_shape[c_ndim - 2];
    int64_t c_n = c_shape[c_ndim - 1];
    int64_t r_m = r_shape[r_ndim - 2];
    int64_t r_n = r_shape[r_ndim - 1];
    int64_t k_refl = tau.shape()[static_cast<int64_t>(tau.shape().size()) - 1];

    int32_t device_id = C.device().index;
    bool is_f64 = (C.dtype() == DType::Float64);
    bool is_f16 = (C.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < c_ndim - 2; ++i) batch_size *= c_shape[i];

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    auto refl_cont = reflectors.contiguous();
    auto tau_cont = tau.contiguous();

    size_t refl_numel = static_cast<size_t>(batch_size) * r_m * r_n;
    size_t tau_numel = static_cast<size_t>(batch_size) * k_refl;
    size_t refl_size = is_f16 ? f16_buf(refl_numel) : refl_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;

    // Compose ormqr by reconstructing Q from packed reflectors + tau, then
    // performing the appropriate matrix multiply. This uses existing GPU shaders
    // (Q-reconstruction + matmul) without requiring a dedicated ormqr shader.
    {
        // Reconstruct Q from packed reflectors + tau
        int64_t q_dim = left ? c_m : c_n;
        std::vector<int64_t> q_shape(c_shape.begin(), c_shape.end() - 2);
        q_shape.push_back(q_dim);
        q_shape.push_back(q_dim);
        Tensor Q(q_shape, C.dtype(), C.device());

        std::string qr_recon_shader = is_f64 ? "linalg_q_reconstruct_f64" :
                                       is_f16 ? "linalg_q_reconstruct_f16" :
                                       "linalg_q_reconstruct";
        auto* qr_pipeline = getPipeline(qr_recon_shader, device_id);

        struct QReconPC {
            uint32_t m_rows;
            uint32_t n_cols;
            uint32_t k_refl;
            uint32_t ldq;
            uint32_t batch_cnt;
        } qr_pc;
        qr_pc.m_rows = static_cast<uint32_t>(r_m);
        qr_pc.n_cols = static_cast<uint32_t>(r_n);
        qr_pc.k_refl = static_cast<uint32_t>(k_refl);
        qr_pc.ldq = static_cast<uint32_t>(q_dim);
        qr_pc.batch_cnt = static_cast<uint32_t>(batch_size);

        size_t q_numel = static_cast<size_t>(batch_size) * q_dim * q_dim;
        size_t q_buf_size = is_f16 ? f16_buf(q_numel) : q_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> qr_bindings = {
            {0, refl_cont.data_ptr()}, {1, tau_cont.data_ptr()}, {2, Q.data_ptr()}
        };
        std::vector<size_t> qr_sizes = {refl_size, tau_size, q_buf_size};
        VkDescriptorSet qr_ds = allocateAndWriteDescriptorSet(device_id, qr_pipeline, qr_bindings, qr_sizes);

        VkCommandBuffer qr_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(qr_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, qr_pipeline->pipeline());
        vkCmdBindDescriptorSets(qr_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               qr_pipeline->layout(), 0, 1, &qr_ds, 0, nullptr);
        vkCmdPushConstants(qr_cmd, qr_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(qr_pc), &qr_pc);
        vkCmdDispatch(qr_cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(qr_cmd);
        endSingleTimeCommands(qr_cmd, device_id);

        // Now compute the matrix product using tenzor ops (dispatches through Vulkan)
        if (left && !transpose_q) {
            // Q * C
            return tenzor::matmul(Q, C.contiguous());
        } else if (left && transpose_q) {
            // Q^T * C
            return tenzor::matmul(tenzor::transpose(Q, -2, -1), C.contiguous());
        } else if (!left && !transpose_q) {
            // C * Q
            return tenzor::matmul(C.contiguous(), Q);
        } else {
            // C * Q^T
            return tenzor::matmul(C.contiguous(), tenzor::transpose(Q, -2, -1));
        }
    }
}

// =========================================================================
// Householder product — compose via dispatchOrmqr on identity
// =========================================================================
auto VulkanBackend::dispatchLinalgHouseholder(const Tensor& input,
                                               const Tensor& tau) -> Tensor {
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    int64_t m = shape[ndim - 2];

    auto I = tenzor::eye(m, std::nullopt, input.dtype(), input.device());

    if (ndim > 2) {
        std::vector<int64_t> eye_shape(shape.begin(), shape.end());
        eye_shape[ndim - 1] = m;
        I = tenzor::expand(I, std::move(eye_shape));
        I = I.contiguous();
    }

    return dispatchOrmqr(input, tau, I, /*left=*/true, /*transpose_q=*/false);
}

// =========================================================================
// LDL^T factorization — native Vulkan compute shader (Bunch-Kaufman)
// One workgroup per batch element, single invocation for the full factor.
// =========================================================================
auto VulkanBackend::dispatchLinalgLDLFactor(const Tensor& A)
    -> std::vector<Tensor> {
    validate_linalg_dtype(A, "ldl_factor");
    auto shape = A.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg.ldl_factor: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::invalid_argument("linalg.ldl_factor: input must be square");

    int32_t device_id = A.device().index;
    DType out_dtype = A.dtype();
    bool needs_promote = (out_dtype == DType::Float16 || out_dtype == DType::BFloat16);
    DType work_dtype = needs_promote ? DType::Float32 : out_dtype;

    Tensor work = A;
    if (needs_promote) work = dispatchCast(work, DType::Float32);

    bool is_f64 = (work_dtype == DType::Float64);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Clone input so shader can modify in-place
    Tensor LD = dispatchClone(work.contiguous());

    // Pivot buffer: flat [batch_size * n] Int32
    std::vector<int64_t> piv_shape(shape.begin(), shape.end() - 2);
    piv_shape.push_back(n);
    Tensor pivots_out(std::vector<int64_t>{batch_size, n}, DType::Int32, A.device());

    std::string shader = is_f64 ? "linalg_ldl_bk_factor_f64" : "linalg_ldl_bk_factor";
    auto* pipeline = getPipeline(shader, device_id);

    struct PushConstants {
        uint32_t n;
        uint32_t batch_size;
    } pc;
    pc.n = static_cast<uint32_t>(n);
    pc.batch_size = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : 4;
    size_t mat_size = static_cast<size_t>(batch_size) * n * n * elem_size;
    size_t piv_size = static_cast<size_t>(batch_size) * n * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, LD.data_ptr()}, {1, pivots_out.data_ptr()}
    };
    std::vector<size_t> sizes = {mat_size, piv_size};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Reshape pivots
    Tensor pivots_reshaped = dispatchReshape(pivots_out, piv_shape);

    if (needs_promote) LD = dispatchCast(LD, out_dtype);

    return {LD, pivots_reshaped};
}

// =========================================================================
// LDL^T solve — native Vulkan compute shader (Bunch-Kaufman solve)
// One workgroup per batch element.
// =========================================================================
auto VulkanBackend::dispatchLinalgLDLSolve(const Tensor& LD,
                                            const Tensor& pivots,
                                            const Tensor& B) -> Tensor {
    validate_linalg_dtype(LD, "ldl_solve");
    auto ld_shape = LD.shape();
    auto b_shape = B.shape();
    int64_t ld_ndim = static_cast<int64_t>(ld_shape.size());
    int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    if (ld_ndim < 2) throw std::invalid_argument("linalg.ldl_solve: LD must be at least 2D");
    if (b_ndim < 2) throw std::invalid_argument("linalg.ldl_solve: B must be at least 2D");

    int64_t n = ld_shape[ld_ndim - 1];
    if (ld_shape[ld_ndim - 2] != n) throw std::invalid_argument("linalg.ldl_solve: LD must be square");
    int64_t nrhs = b_shape[b_ndim - 1];

    int32_t device_id = LD.device().index;
    DType out_dtype = LD.dtype();
    bool needs_promote = (out_dtype == DType::Float16 || out_dtype == DType::BFloat16);
    DType work_dtype = needs_promote ? DType::Float32 : out_dtype;

    Tensor ld = LD;
    Tensor bmat = B;
    if (needs_promote) {
        ld = dispatchCast(ld, DType::Float32);
        bmat = dispatchCast(bmat, DType::Float32);
    }

    bool is_f64 = (work_dtype == DType::Float64);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ld_ndim - 2; ++i) batch_size *= ld_shape[i];

    Tensor pivots_cont = pivots.contiguous();
    auto ld_cont = ld.contiguous();
    auto b_cont = bmat.contiguous();

    // The LDL^T solve shader has no error-reporting path back to the host, so
    // an exactly-zero D-block pivot (a singular factor) would silently
    // produce Inf/NaN instead of throwing (matches CPU/CUDA/ROCm, which all
    // detect this via the same pivot-block decoding and throw). Validate
    // host-side before ever dispatching the shader.
    {
        Tensor ld_host = ld_cont.to(Device::cpu());
        Tensor piv_host = pivots_cont.to(Device::cpu());
        const int32_t* piv_ptr = piv_host.data<int32_t>();
        auto check_batch = [&](auto* ld_ptr) {
            for (int64_t bidx = 0; bidx < batch_size; ++bidx) {
                auto* ld_mat = ld_ptr + bidx * n * n;
                const int32_t* piv_mat = piv_ptr + bidx * n;
                for (int64_t k = 0; k < n; ) {
                    int32_t p = piv_mat[k];
                    if (p > 0) {
                        if (ld_mat[k * n + k] == 0) {
                            throw std::runtime_error(
                                "linalg.ldl_solve: singular LDL^T factor (zero pivot)");
                        }
                        k++;
                    } else {
                        auto d11 = ld_mat[k * n + k];
                        auto d21 = ld_mat[(k + 1) * n + k];
                        auto d22 = ld_mat[(k + 1) * n + (k + 1)];
                        if (d11 * d22 - d21 * d21 == 0) {
                            throw std::runtime_error(
                                "linalg.ldl_solve: singular LDL^T factor (zero pivot)");
                        }
                        k += 2;
                    }
                }
            }
        };
        if (is_f64) {
            check_batch(ld_host.data<double>());
        } else {
            check_batch(ld_host.data<float>());
        }
    }

    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), work_dtype, LD.device());

    std::string shader = is_f64 ? "linalg_ldl_bk_solve_f64" : "linalg_ldl_bk_solve";
    auto* pipeline = getPipeline(shader, device_id);

    struct PushConstants {
        uint32_t n;
        uint32_t nrhs;
        uint32_t batch_size;
    } pc;
    pc.n = static_cast<uint32_t>(n);
    pc.nrhs = static_cast<uint32_t>(nrhs);
    pc.batch_size = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : 4;
    size_t ld_sz  = static_cast<size_t>(batch_size) * n * n * elem_size;
    size_t piv_sz = static_cast<size_t>(batch_size) * n * sizeof(int32_t);
    size_t b_sz   = static_cast<size_t>(batch_size) * n * nrhs * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, ld_cont.data_ptr()}, {1, pivots_cont.data_ptr()},
        {2, b_cont.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {ld_sz, piv_sz, b_sz, b_sz};
    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    if (needs_promote) output = dispatchCast(output, out_dtype);
    return output;
}

} // namespace tenzor
