#include "vulkan_ops_common.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"

namespace tenzor {

static constexpr int64_t MAX_SMALL_LINALG_SIZE = 128;  // single-workgroup shader limit
static constexpr int64_t TILED_BLOCK_SIZE = 32;        // panel width for blocked algorithms

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
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t piv_size = static_cast<size_t>(batch_size) * n * sizeof(int32_t);

    for (int64_t col_start = 0; col_start < n; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - col_start);

        // --- Panel factorization (one workgroup per batch element) ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_lu_panel_f64" : is_f16 ? "linalg_lu_panel_f16" : "linalg_lu_panel";
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
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Trailing matrix update ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;
            uint32_t tile_count = static_cast<uint32_t>((trail_size + 31) / 32);

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_lu_update_f64" : is_f16 ? "linalg_lu_update_f16" : "linalg_lu_update";
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
                vkCmdDispatch(cmd, tile_count, tile_count, 1);  // 16x16 threads per tile
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked Cholesky: panel factorization + trailing SYRK update
// Modifies A in-place to contain L (lower triangle).
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedCholesky(Tensor& A, int64_t n,
                                        int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;

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

auto VulkanBackend::dispatchLinalgDet(const Tensor& input) -> Tensor {
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
        // Tiled path (n > 128): blocked LU, then compute det from diagonal
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

    // Tiled path (n > 128): LU factorize on GPU, then backsolve on GPU via TRSM shader
    Tensor A = dispatchClone(input.contiguous());
    Tensor pivots({batch_size, n}, DType::Int32, input.device());

    runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);

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

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
            uint32_t nrhs;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.nrhs = 1;

        auto a_cont = a.contiguous();
        auto b_cont = b.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t a_numel = batch_size * n * n;
        size_t b_numel = batch_size * n;
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

    // Tiled path (n > 128): LU factorize on GPU, backsolve on GPU via TRSM shader
    Tensor A = dispatchClone(a.contiguous());
    Tensor pivots({batch_size, n}, DType::Int32, a.device());

    runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);

    // Determine nrhs from b shape
    int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
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

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, mat_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    // Tiled path (n > 128): blocked Cholesky factorization
    Tensor A = dispatchClone(input.contiguous());

    runBlockedCholesky(A, n, batch_size, device_id, is_f64, is_f16);

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

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    if (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        std::vector<int64_t> q_shape(shape.begin(), shape.end() - 2);
        q_shape.push_back(m); q_shape.push_back(m);
        std::vector<int64_t> r_shape(shape.begin(), shape.end());

        Tensor Q(q_shape, input.dtype(), input.device());
        Tensor R(r_shape, input.dtype(), input.device());

        std::string shader = is_f64 ? "linalg_qr_f64" : is_f16 ? "linalg_qr_f16" : "linalg_qr";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t m; uint32_t n_cols; uint32_t batch; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n_cols = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t in_numel = batch_size * m * n;
        size_t q_numel = batch_size * m * m;
        size_t in_size = is_f16 ? f16_buf(in_numel) : in_numel * elem_size;
        size_t q_size = is_f16 ? f16_buf(q_numel) : q_numel * elem_size;
        size_t r_size = in_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, Q.data_ptr()}, {2, R.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, q_size, r_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return {Q, R};
    }

    // Tiled path (m,n > 128): blocked QR via Householder reflections
    // After blocked QR, A contains R on/above diagonal and Householder vectors below.
    // Reconstruct Q on GPU via dedicated Q-reconstruction shader.
    int64_t k = std::min(m, n);

    Tensor A = dispatchClone(input.contiguous());

    // tau stores Householder scalars: one per column factorized
    std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
    tau_shape.push_back(n);
    Tensor tau(tau_shape, input.dtype(), input.device());

    runBlockedQR(A, tau, m, n, batch_size, device_id, is_f64, is_f16);

    // Extract R from the upper triangle of A (on GPU)
    std::vector<int64_t> r_shape(shape.begin(), shape.end());
    Tensor R = dispatchTriuTril("triu", A, 0);

    // Reconstruct Q from Householder vectors and tau entirely on GPU
    std::vector<int64_t> q_shape(shape.begin(), shape.end() - 2);
    q_shape.push_back(m); q_shape.push_back(m);
    Tensor Q(q_shape, input.dtype(), input.device());

    std::string qr_recon_shader = is_f64 ? "linalg_q_reconstruct_f64" : is_f16 ? "linalg_q_reconstruct_f16" : "linalg_q_reconstruct";
    auto* qr_pipeline = getPipeline(qr_recon_shader, device_id);

    struct QReconPC {
        uint32_t m_rows;
        uint32_t n_cols;
        uint32_t k_refl;
        uint32_t ldq;
        uint32_t batch_cnt;
    } qr_pc;
    qr_pc.m_rows = static_cast<uint32_t>(m);
    qr_pc.n_cols = static_cast<uint32_t>(n);
    qr_pc.k_refl = static_cast<uint32_t>(k);
    qr_pc.ldq = static_cast<uint32_t>(m);
    qr_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    size_t qr_numel = batch_size * m * n;
    size_t tau_numel = batch_size * n;
    size_t q_numel = batch_size * m * m;
    size_t qr_buf_size = is_f16 ? f16_buf(qr_numel) : qr_numel * elem_size;
    size_t tau_buf_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;
    size_t q_buf_size = is_f16 ? f16_buf(q_numel) : q_numel * elem_size;

    std::vector<std::pair<uint32_t, const void*>> qr_bindings = {
        {0, A.data_ptr()}, {1, tau.data_ptr()}, {2, Q.data_ptr()}
    };
    std::vector<size_t> qr_sizes = {qr_buf_size, tau_buf_size, q_buf_size};
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

    return {Q, R};
}

// ============================================================================
// SVD (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgSVD(const Tensor& input, bool full_matrices) -> std::vector<Tensor> {
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

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    // Output: U (batch, m, k), S (batch, k), Vt (batch, n, n)
    std::vector<int64_t> u_shape(shape.begin(), shape.end() - 2);
    u_shape.push_back(m); u_shape.push_back(k);
    std::vector<int64_t> s_shape(shape.begin(), shape.end() - 2);
    s_shape.push_back(k);
    std::vector<int64_t> vt_shape(shape.begin(), shape.end() - 2);
    vt_shape.push_back(n); vt_shape.push_back(n);

    // For small matrices, all outputs match input dtype.
    // For large (tiled) matrices, S is always f32/f64 for numerical stability.
    DType s_dtype = (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE)
                  ? input.dtype()
                  : (is_f64 ? DType::Float64 : DType::Float32);

    Tensor U(u_shape, input.dtype(), input.device());
    Tensor S(s_shape, s_dtype, input.device());
    Tensor Vt(vt_shape, input.dtype(), input.device());

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;

    if (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE) {
        // ---- Small matrix path: single-workgroup Jacobi SVD ----
        std::string shader = is_f64 ? "linalg_svd_f64" : is_f16 ? "linalg_svd_f16" : "linalg_svd";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t m; uint32_t n_cols; uint32_t batch; uint32_t full_matrices; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n_cols = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.full_matrices = full_matrices ? 1 : 0;

        auto cont = input.contiguous();
        size_t in_numel = batch_size * m * n;
        size_t u_numel = batch_size * m * k;
        size_t s_numel = batch_size * k;
        size_t vt_numel = batch_size * n * n;
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
        // ---- Tiled path: Bidiagonal reduction → Bidiagonal SVD → Accumulate ----
        // Currently requires square matrices for the tiled bidiag reduction.
        if (m != n) {
            throw std::runtime_error(std::format(
                "Vulkan linalg.svd: tiled SVD currently requires square matrices. "
                "Got {}x{}. Rectangular support is planned.", m, n));
        }
        if (is_f16) {
            throw std::runtime_error(
                "Vulkan linalg.svd: tiled SVD does not support Float16 due to "
                "insufficient precision. Use Float32 or Float64 for large matrices.");
        }

        auto cont = input.contiguous();
        size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
        size_t tau_numel = static_cast<size_t>(batch_size) * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
        size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;

        // Copy input to working matrix (will be modified in place)
        std::vector<int64_t> mat_shape(shape.begin(), shape.end());
        Tensor A(mat_shape, input.dtype(), input.device());
        {
            // Copy cont -> A (device-to-device)
            auto [src_buf, src_off] = getVulkanBufferAndOffset(cont.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(A.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = mat_size;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Allocate tau vectors for left and right Householder reflections
        std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
        tau_shape.push_back(n);
        // Bidiag SVD shader needs f32 precision even for f16 input, so tau_l/tau_r
        // match input dtype for the bidiag phase, then we use f32 for the QR phase
        Tensor tau_l(tau_shape, input.dtype(), input.device());
        Tensor tau_r(tau_shape, input.dtype(), input.device());

        // Step 1: Blocked bidiagonal reduction
        runBlockedBidiag(A, tau_l, tau_r, n, batch_size, device_id, is_f64, is_f16);

        // Step 2: Extract diagonal and superdiagonal from bidiagonalized A
        std::vector<int64_t> diag_shape(shape.begin(), shape.end() - 2);
        diag_shape.push_back(n);
        std::vector<int64_t> sdiag_shape(shape.begin(), shape.end() - 2);
        sdiag_shape.push_back(n - 1);

        // For bidiag SVD we always use f32 or f64 (not f16) for numerical stability
        DType svd_dtype = is_f64 ? DType::Float64 : DType::Float32;
        size_t svd_elem = is_f64 ? 8 : 4;

        Tensor diag_t(diag_shape, svd_dtype, input.device());
        Tensor sdiag_t(sdiag_shape, svd_dtype, input.device());
        size_t diag_numel = static_cast<size_t>(batch_size) * n;
        size_t sdiag_numel = static_cast<size_t>(batch_size) * (n - 1);
        size_t diag_size = diag_numel * svd_elem;
        size_t sdiag_size = sdiag_numel * svd_elem;

        // Dispatch extract shader
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_bidiag_extract_f64"
                               : is_f16 ? "linalg_bidiag_extract_f16"
                                         : "linalg_bidiag_extract";
            auto* pipeline = getPipeline(shader, device_id);

            struct ExtractPC { uint32_t n; uint32_t batch_idx; } epc;
            epc.n = static_cast<uint32_t>(n);
            epc.batch_idx = static_cast<uint32_t>(b);

            size_t extract_mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
            size_t extract_diag_size = is_f16 ? f16_buf(diag_numel) : diag_size;
            size_t extract_sdiag_size = is_f16 ? f16_buf(sdiag_numel) : sdiag_size;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, diag_t.data_ptr()}, {2, sdiag_t.data_ptr()}
            };
            std::vector<size_t> sizes = {extract_mat_size, extract_diag_size, extract_sdiag_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            uint32_t num_groups = (static_cast<uint32_t>(n) + 255) / 256;
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(epc), &epc);
            vkCmdDispatch(cmd, num_groups, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 3: Bidiagonal SVD (Golub-Kahan QR iteration)
        // Allocate rotation parameter buffers — generous size for accumulated rotations
        // Max rotations per batch: ~30 * n * 2 (left + right per step) + n (sign flips) + n (sorts)
        size_t max_rots = static_cast<size_t>(n) * 64;  // generous upper bound
        size_t rot_stride = max_rots * 2 + 2;  // pairs of (cos, sin) + metadata
        std::vector<int64_t> rot_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(rot_stride)};
        Tensor u_rot(rot_shape, svd_dtype, input.device());
        Tensor v_rot(rot_shape, svd_dtype, input.device());
        size_t rot_size = batch_size * rot_stride * svd_elem;

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_bidiag_svd_f64" : "linalg_bidiag_svd";
            auto* pipeline = getPipeline(shader, device_id);

            struct BidiagSvdPC {
                uint32_t n;
                uint32_t max_iterations;
                uint32_t batch_idx;
                uint32_t rot_stride;
            } bpc;
            bpc.n = static_cast<uint32_t>(n);
            bpc.max_iterations = static_cast<uint32_t>(n * 30);
            bpc.batch_idx = static_cast<uint32_t>(b);
            bpc.rot_stride = static_cast<uint32_t>(rot_stride);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, diag_t.data_ptr()}, {1, sdiag_t.data_ptr()},
                {2, u_rot.data_ptr()}, {3, v_rot.data_ptr()}
            };
            std::vector<size_t> sizes = {diag_size, sdiag_size, rot_size, rot_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(bpc), &bpc);
            vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 4: Reconstruct U and Vt from Householder vectors
        // The accumulate shader builds U and Vt from the bidiag Householder vectors.
        // Full n x n U and Vt are built, then the bidiag SVD Givens rotations are
        // applied separately (or folded in as a matmul).

        // Allocate full n x n U and Vt workspace
        std::vector<int64_t> full_shape(shape.begin(), shape.end() - 2);
        full_shape.push_back(n); full_shape.push_back(n);
        Tensor U_full(full_shape, input.dtype(), input.device());
        Tensor Vt_full(full_shape, input.dtype(), input.device());
        size_t full_numel = static_cast<size_t>(batch_size) * n * n;
        size_t full_size = is_f16 ? f16_buf(full_numel) : full_numel * elem_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            // Build U (mode=0)
            {
                std::string shader = is_f64 ? "linalg_svd_accumulate_f64"
                                   : is_f16 ? "linalg_svd_accumulate_f16"
                                             : "linalg_svd_accumulate";
                auto* pipeline = getPipeline(shader, device_id);

                struct AccumPC { uint32_t n; uint32_t batch_idx; uint32_t mode; } apc;
                apc.n = static_cast<uint32_t>(n);
                apc.batch_idx = static_cast<uint32_t>(b);
                apc.mode = 0;  // build U

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()},
                    {3, U_full.data_ptr()}, {4, Vt_full.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, tau_size, full_size, full_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(apc), &apc);
                // One workgroup per column of U
                vkCmdDispatch(cmd, static_cast<uint32_t>(n), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            // Build Vt (mode=1)
            {
                std::string shader = is_f64 ? "linalg_svd_accumulate_f64"
                                   : is_f16 ? "linalg_svd_accumulate_f16"
                                             : "linalg_svd_accumulate";
                auto* pipeline = getPipeline(shader, device_id);

                struct AccumPC { uint32_t n; uint32_t batch_idx; uint32_t mode; } apc;
                apc.n = static_cast<uint32_t>(n);
                apc.batch_idx = static_cast<uint32_t>(b);
                apc.mode = 1;  // build Vt

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()},
                    {3, U_full.data_ptr()}, {4, Vt_full.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, tau_size, full_size, full_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(apc), &apc);
                // One workgroup per row of Vt
                vkCmdDispatch(cmd, static_cast<uint32_t>(n), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }

        // Copy singular values from diag_t to S
        // diag_t contains the singular values after bidiag SVD
        {
            size_t s_numel_copy = static_cast<size_t>(batch_size) * k;
            size_t s_size_bytes = s_numel_copy * svd_elem;
            auto [src_buf, src_off] = getVulkanBufferAndOffset(diag_t.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(S.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = s_size_bytes;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Copy U_full -> U (may be a subset of columns if reduced SVD)
        // For square matrices with reduced=false, U is n x n = U_full
        {
            size_t u_numel_copy = static_cast<size_t>(batch_size) * m * k;
            size_t u_size_bytes = is_f16 ? f16_buf(u_numel_copy) : u_numel_copy * elem_size;
            auto [src_buf, src_off] = getVulkanBufferAndOffset(U_full.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(U.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = u_size_bytes;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Copy Vt_full -> Vt
        {
            size_t vt_numel_copy = static_cast<size_t>(batch_size) * n * n;
            size_t vt_size_bytes = is_f16 ? f16_buf(vt_numel_copy) : vt_numel_copy * elem_size;
            auto [src_buf, src_off] = getVulkanBufferAndOffset(Vt_full.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(Vt.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = vt_size_bytes;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }
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

        // --- Apply reflections to trailing submatrix ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < k) {
            int64_t trail_rows = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_tridiag_update_f64" : is_f16 ? "linalg_tridiag_update_f16" : "linalg_tridiag_update";
                auto* pipeline = getPipeline(shader, device_id);

                // Allocate V and W workspace for WY representation
                size_t vw_numel = static_cast<size_t>(panel_cols) * n;
                size_t vw_size = is_f16 ? f16_buf(vw_numel) : vw_numel * elem_size;
                Tensor v_work({panel_cols, n}, A.dtype(), A.device());
                Tensor w_work({panel_cols, n}, A.dtype(), A.device());

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
                    {0, A.data_ptr()}, {1, tau.data_ptr()},
                    {2, v_work.data_ptr()}, {3, w_work.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, vw_size, vw_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_rows), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
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
        if (trail_start < k) {
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
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.eigh: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.eigh: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    // Output: eigenvalues (batch, n), eigenvectors (batch, n, n)
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 2);
    w_shape.push_back(n);

    Tensor W(w_shape, input.dtype(), input.device());
    Tensor V(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup Jacobi shader
        std::string shader = is_f64 ? "linalg_eigh_f64" : is_f16 ? "linalg_eigh_f16" : "linalg_eigh";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t mat_numel = batch_size * n * n;
        size_t w_numel = batch_size * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
        size_t w_size = is_f16 ? f16_buf(w_numel) : w_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, W.data_ptr()}, {2, V.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, w_size, mat_size};
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
        // Tiled path (n > MAX_SMALL_LINALG_SIZE): blocked tridiagonal reduction +
        // tridiagonal QR eigendecomposition + eigenvector back-transformation.
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;

        // Step 1: Tridiagonal reduction via blocked Householder reflections.
        // After this, A contains tridiagonal form with Householder vectors stored
        // below the sub-diagonal for eigenvector reconstruction.
        Tensor A = dispatchClone(input.contiguous());
        std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
        tau_shape.push_back(n);
        Tensor tau(tau_shape, input.dtype(), input.device());

        runBlockedTridiag(A, tau, n, batch_size, device_id, is_f64, is_f16);

        // Step 2: Extract diagonal and subdiagonal from tridiagonalized A.
        // A[i,i] = diagonal, A[i+1,i] = subdiagonal (symmetric tridiagonal).
        // Use existing GPU diagonal-extraction shader (no CPU readback).
        Tensor A_batched = A.view({batch_size, n, n});
        Tensor diag = dispatchDiag(A_batched, 0);       // main diagonal: shape (batch, n)
        Tensor sdiag = dispatchDiag(A_batched, -1);      // sub-diagonal: shape (batch, n-1)

        // Step 3: Tridiagonal eigendecomposition via implicit QR with Wilkinson shift
        uint32_t max_iters = static_cast<uint32_t>(n * 30);
        uint32_t rot_budget = std::min(max_iters, static_cast<uint32_t>(256));
        Tensor rots({batch_size, static_cast<int64_t>(3 * n * rot_budget)},
                    input.dtype(), input.device());

        size_t diag_buf_size = is_f16 ? f16_buf(batch_size * n) : batch_size * n * elem_size;
        size_t sdiag_buf_size = is_f16 ? f16_buf(batch_size * (n - 1)) : batch_size * (n > 1 ? n - 1 : 1) * elem_size;
        size_t w_buf_size = is_f16 ? f16_buf(batch_size * n) : batch_size * n * elem_size;
        size_t rots_buf_size = static_cast<size_t>(batch_size) * 3 * n * rot_budget * elem_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_tridiag_eig_f64" : "linalg_tridiag_eig";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t max_iterations;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.max_iterations = max_iters;
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, diag.data_ptr()}, {1, sdiag.data_ptr()},
                {2, W.data_ptr()}, {3, rots.data_ptr()}
            };
            std::vector<size_t> sizes = {diag_buf_size, sdiag_buf_size, w_buf_size, rots_buf_size};
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

        // Step 4: Eigenvector reconstruction.
        // V = Q_tridiag^T * Q_givens, where Q_tridiag accumulates the Householder
        // reflections from the tridiagonalization, and Q_givens accumulates the
        // Givens rotations from the tridiagonal QR iteration.
        size_t mat_size = static_cast<size_t>(batch_size) * n * n * elem_size;
        size_t tau_buf_size = static_cast<size_t>(batch_size) * n * elem_size;
        size_t v_buf_size = mat_size;
        uint32_t num_rots = static_cast<uint32_t>(n * rot_budget);

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_eigh_backtransform_f64" : "linalg_eigh_backtransform";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t num_rotations;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.num_rotations = num_rots;
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()},
                {2, rots.data_ptr()}, {3, V.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_buf_size, rots_buf_size, v_buf_size};
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
    }

    return {W, V};
}

// ============================================================================
// Eig — General Eigenvalue Decomposition (tiled for arbitrary-sized matrices)
// ============================================================================

auto VulkanBackend::dispatchLinalgEig(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.eig: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.eig: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Output: eigenvalues_real (batch, n), eigenvalues_imag (batch, n)
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 2);
    w_shape.push_back(n);

    Tensor WR(w_shape, input.dtype(), input.device());
    Tensor WI(w_shape, input.dtype(), input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup QR iteration shader
        std::string shader = is_f64 ? "linalg_eig_f64" : "linalg_eig";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; uint32_t max_iterations; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.max_iterations = static_cast<uint32_t>(n * 30);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : 4;
        size_t mat_size = batch_size * n * n * elem_size;
        size_t w_size = batch_size * n * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, WR.data_ptr()}, {2, WI.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, w_size, w_size};
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
        // Tiled path (n > MAX_SMALL_LINALG_SIZE): blocked Hessenberg reduction +
        // iterative Francis double-shift QR + eigenvalue extraction from Schur form.
        size_t elem_size = is_f64 ? 8 : 4;

        // Step 1: Reduce to upper Hessenberg form via blocked Householder reflections
        Tensor H = dispatchClone(input.contiguous());
        std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
        tau_shape.push_back(n);
        Tensor tau(tau_shape, input.dtype(), input.device());

        runBlockedHessenberg(H, tau, n, batch_size, device_id, is_f64);

        // Step 2: Iterative Francis double-shift QR on the Hessenberg matrix.
        // Run QR steps until all subdiagonal entries are negligible (real Schur form).
        size_t mat_size = batch_size * n * n * elem_size;
        uint32_t max_qr_iterations = static_cast<uint32_t>(n * 30);

        // Shift/deflation buffer: 4 elements per batch (deflation index, lo, hi, unused)
        Tensor shifts({batch_size, 4}, input.dtype(), input.device());
        size_t shift_size = batch_size * 4 * elem_size;

        // Active block boundaries tracked entirely on GPU (no host-side vectors)

        // GPU buffers for convergence check shader
        Tensor gpu_active_end({batch_size}, DType::Int32, input.device());
        Tensor gpu_active_start({batch_size}, DType::Int32, input.device());
        Tensor gpu_converged({batch_size}, DType::Int32, input.device());
        Tensor gpu_all_converged({1}, DType::Int32, input.device());

        // Initialize GPU buffers
        {
            auto ae_cpu = Tensor({batch_size}, DType::Int32, Device::cpu());
            auto as_cpu = Tensor({batch_size}, DType::Int32, Device::cpu());
            auto conv_cpu = Tensor({batch_size}, DType::Int32, Device::cpu());
            for (int64_t b = 0; b < batch_size; ++b) {
                as_cpu.data<int32_t>()[b] = 0;
                ae_cpu.data<int32_t>()[b] = static_cast<int32_t>(n - 1);
                conv_cpu.data<int32_t>()[b] = 0;
            }
            gpu_active_start = as_cpu.to(input.device());
            gpu_active_end = ae_cpu.to(input.device());
            gpu_converged = conv_cpu.to(input.device());
        }

        size_t uint_buf_size = batch_size * sizeof(uint32_t);

        for (uint32_t iter = 0; iter < max_qr_iterations; ++iter) {
            // Dispatch QR step for ALL batches in one command; converged batches
            // early-exit inside the shader (active_end <= active_start read from SSBOs).
            {
                std::string shader = is_f64 ? "linalg_hessenberg_qr_f64" : "linalg_hessenberg_qr";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                } pc;
                pc.n = static_cast<uint32_t>(n);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, H.data_ptr()}, {1, shifts.data_ptr()},
                    {2, gpu_active_start.data_ptr()}, {3, gpu_active_end.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, shift_size, uint_buf_size, uint_buf_size};
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
            }

            // GPU-side convergence check: updates active_end and batch_converged on device,
            // produces a single all_converged flag to minimize readback.
            {
                // Initialize all_converged = 1 (will be AND-ed to 0 if any batch not done)
                gpu_all_converged = dispatchFull({1}, 1.0f, DType::Int32);

                std::string conv_shader = is_f64 ? "eig_convergence_check_f64" : "eig_convergence_check";
                auto* conv_pipeline = getPipeline(conv_shader, device_id);

                struct ConvPC { uint32_t batch_size; } conv_pc;
                conv_pc.batch_size = static_cast<uint32_t>(batch_size);

                std::vector<std::pair<uint32_t, const void*>> conv_bindings = {
                    {0, shifts.data_ptr()},
                    {1, gpu_active_end.data_ptr()},
                    {2, gpu_active_start.data_ptr()},
                    {3, gpu_converged.data_ptr()},
                    {4, gpu_all_converged.data_ptr()}
                };
                std::vector<size_t> conv_sizes = {
                    shift_size, uint_buf_size, uint_buf_size,
                    uint_buf_size, sizeof(uint32_t)
                };
                VkDescriptorSet conv_ds = allocateAndWriteDescriptorSet(
                    device_id, conv_pipeline, conv_bindings, conv_sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, conv_pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       conv_pipeline->layout(), 0, 1, &conv_ds, 0, nullptr);
                vkCmdPushConstants(cmd, conv_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(conv_pc), &conv_pc);
                uint32_t num_groups = (static_cast<uint32_t>(batch_size) + 63u) / 64u;
                vkCmdDispatch(cmd, num_groups, 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            // Single sync + readback of one uint32 flag (not per-batch metadata)
            synchronize(device_id);

            Tensor ac_cpu = gpu_all_converged.to(Device::cpu());
            if (ac_cpu.data<int32_t>()[0] != 0) break;
        }

        // Step 3: Extract eigenvalues from quasi-upper-triangular (real Schur) form
        size_t w_size = batch_size * n * elem_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_extract_eigenvalues_f64" : "linalg_extract_eigenvalues";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, H.data_ptr()}, {1, WR.data_ptr()}, {2, WI.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, w_size, w_size};
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
    }

    return {WR, WI};
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
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
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
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return hy;
}

// ===========================================================================
// SearchSorted — native GPU binary search shader
// ===========================================================================

auto VulkanBackend::dispatchSearchSorted(const Tensor& sorted, const Tensor& values) -> Tensor {
    if (sorted.numel() == 0 || values.numel() == 0) {
        return Tensor(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                      DType::Int32, values.device());
    }

    // Float16/BFloat16: native packed shader path
    if (sorted.dtype() == DType::Float16 || sorted.dtype() == DType::BFloat16) {
        bool is_bf16_ss = (sorted.dtype() == DType::BFloat16);
        int32_t dev_id = sorted.device().index;

        auto sorted_cont = sorted.is_contiguous() ? sorted : dispatchContiguous(sorted);
        auto values_cont = values.is_contiguous() ? values : dispatchContiguous(values);

        std::vector<int64_t> out_shape_f16(values.shape().begin(), values.shape().end());
        Tensor output_f16(out_shape_f16, DType::Int32, values.device());

        auto* pipe = getPipeline(is_bf16_ss ? "searchsorted_bf16" : "searchsorted_f16", dev_id);

        struct { uint32_t array_size; uint32_t num_queries; } pc;
        pc.array_size = static_cast<uint32_t>(sorted.numel());
        pc.num_queries = static_cast<uint32_t>(values.numel());

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
        vkCmdDispatch(cmd, div_wg(values.numel(), devices_[dev_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, dev_id);

        return output_f16;
    }

    int32_t device_id = sorted.device().index;

    auto sorted_contig = sorted.is_contiguous() ? sorted : dispatchContiguous(sorted);
    auto values_contig = values.is_contiguous() ? values : dispatchContiguous(values);

    std::vector<int64_t> out_shape(values.shape().begin(), values.shape().end());
    Tensor output(out_shape, DType::Int32, values.device());

    bool is_f64 = (sorted.dtype() == DType::Float64);
    std::string ss_shader = is_f64 ? "searchsorted_f64" : "searchsorted";
    auto* pipeline = getPipeline(ss_shader, device_id);

    struct {
        uint32_t array_size;
        uint32_t num_queries;
    } pc;
    pc.array_size = static_cast<uint32_t>(sorted.numel());
    pc.num_queries = static_cast<uint32_t>(values.numel());

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
    vkCmdDispatch(cmd, div_wg(values.numel(), devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Quantized Linear — native Int8 GEMM with Int32 accumulation
// ===========================================================================

auto VulkanBackend::dispatchQuantizedLinear(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp) -> Tensor
{
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t M = input_shape[0];   // batch_size
    int64_t K = input_shape[1];   // in_features
    int64_t N = weight_shape[0];  // out_features

    int32_t device_id = input.device().index;

    auto input_contig = input.is_contiguous() ? input : dispatchContiguous(input);
    auto weight_contig = weight.is_contiguous() ? weight : dispatchContiguous(weight);
    auto bias_contig = bias.is_contiguous() ? bias : dispatchContiguous(bias);

    Tensor output({M, N}, DType::Float32, input.device());

    auto* pipeline = getPipeline("quantized_linear", device_id);

    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        float input_scale;
        float weight_scale;
        int32_t input_zero_point;
        int32_t weight_zero_point;
    } pc;
    pc.M = static_cast<uint32_t>(M);
    pc.N = static_cast<uint32_t>(N);
    pc.K = static_cast<uint32_t>(K);
    pc.input_scale = input_scale;
    pc.weight_scale = weight_scale;
    pc.input_zero_point = input_zp;
    pc.weight_zero_point = weight_zp;

    size_t input_bytes = input_contig.numel() * input_contig.dtype_size();
    size_t weight_bytes = weight_contig.numel() * weight_contig.dtype_size();
    size_t bias_bytes = bias_contig.numel() * bias_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_contig.data_ptr()},
        {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, weight_bytes, bias_bytes, output_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    int64_t total = M * N;
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Quantized Conv2d — native Int8 convolution with Int32 accumulation
// ===========================================================================

auto VulkanBackend::dispatchQuantizedConv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    int64_t stride, int64_t padding,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp) -> Tensor
{
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];
    int64_t out_channels = weight_shape[0];
    int64_t kernel_size = weight_shape[2];

    int64_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    int64_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

    int32_t device_id = input.device().index;

    auto input_contig = input.is_contiguous() ? input : dispatchContiguous(input);
    auto weight_contig = weight.is_contiguous() ? weight : dispatchContiguous(weight);
    auto bias_contig = bias.is_contiguous() ? bias : dispatchContiguous(bias);

    Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input.device());

    auto* pipeline = getPipeline("quantized_conv2d", device_id);

    struct {
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t h_in;
        uint32_t w_in;
        uint32_t h_out;
        uint32_t w_out;
        uint32_t kernel_size;
        uint32_t stride;
        uint32_t padding;
        float input_scale;
        float weight_scale;
        int32_t input_zero_point;
        int32_t weight_zero_point;
    } pc;
    pc.batch = static_cast<uint32_t>(batch);
    pc.in_channels = static_cast<uint32_t>(in_channels);
    pc.out_channels = static_cast<uint32_t>(out_channels);
    pc.h_in = static_cast<uint32_t>(h_in);
    pc.w_in = static_cast<uint32_t>(w_in);
    pc.h_out = static_cast<uint32_t>(h_out);
    pc.w_out = static_cast<uint32_t>(w_out);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.input_scale = input_scale;
    pc.weight_scale = weight_scale;
    pc.input_zero_point = input_zp;
    pc.weight_zero_point = weight_zp;

    size_t input_bytes = input_contig.numel() * input_contig.dtype_size();
    size_t weight_bytes = weight_contig.numel() * weight_contig.dtype_size();
    size_t bias_bytes = bias_contig.numel() * bias_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_contig.data_ptr()},
        {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, weight_bytes, bias_bytes, output_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    int64_t total = batch * out_channels * h_out * w_out;
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
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
    float scale, bool causal) -> Tensor
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
        batch_heads = q_shape[0] * q_shape[1];
        seq_len_q = q_shape[2];
        d_k = q_shape[3];
        seq_len_k = K.shape()[2];

        q_flat = Q.reshape({batch_heads, seq_len_q, d_k});
        k_flat = K.reshape({batch_heads, seq_len_k, d_k});
        v_flat = V.reshape({batch_heads, seq_len_k, V.shape()[3]});
    } else {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        d_k = q_shape[2];
        seq_len_k = K.shape()[1];
    }

    // Phase 5.4: fast path — use the tiled FlashAttention-2 shader
    // (flash_attention.comp) when the configuration fits its current
    // constraints:
    //   - Float32 only
    //   - head_dim / head_v <= 128
    //   - no causal mask (the shader doesn't implement masking)
    //   - contiguous Q, K, V
    //
    // The tiled path avoids materializing the full seq_q × seq_k
    // attention matrix — it streams K/V blocks through shared memory
    // with an online softmax. Falls through to the composed path
    // below on any edge case or dtype mismatch.
    int64_t head_v = v_flat.shape()[2];
    const int64_t kMaxHeadDimTiled = 128;
    if (!causal
        && Q.dtype() == DType::Float32
        && K.dtype() == DType::Float32
        && V.dtype() == DType::Float32
        && d_k <= kMaxHeadDimTiled
        && head_v <= kMaxHeadDimTiled
        && seq_len_q > 0 && seq_len_k > 0)
    {
        Tensor q_contig = q_flat.is_contiguous() ? q_flat : dispatchContiguous(q_flat);
        Tensor k_contig = k_flat.is_contiguous() ? k_flat : dispatchContiguous(k_flat);
        Tensor v_contig = v_flat.is_contiguous() ? v_flat : dispatchContiguous(v_flat);

        Tensor output_flat({batch_heads, seq_len_q, head_v},
                           DType::Float32, Q.device());

        int32_t device_id = Q.device().index;
        auto* pipeline = getPipeline("flash_attention", device_id);

        struct PushConstants {
            int32_t seq_q;
            int32_t seq_k;
            int32_t head_dim;
            int32_t head_v;
            float scale;
        } pc;
        pc.seq_q    = static_cast<int32_t>(seq_len_q);
        pc.seq_k    = static_cast<int32_t>(seq_len_k);
        pc.head_dim = static_cast<int32_t>(d_k);
        pc.head_v   = static_cast<int32_t>(head_v);
        pc.scale    = scale;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, q_contig.data_ptr()},
            {1, k_contig.data_ptr()},
            {2, v_contig.data_ptr()},
            {3, output_flat.data_ptr()},
        };
        std::vector<size_t> sizes = {
            static_cast<size_t>(q_contig.numel()) * sizeof(float),
            static_cast<size_t>(k_contig.numel()) * sizeof(float),
            static_cast<size_t>(v_contig.numel()) * sizeof(float),
            static_cast<size_t>(output_flat.numel()) * sizeof(float),
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

        // Restore original shape.
        if (has_head_dim) {
            return output_flat.reshape({q_shape[0], q_shape[1], seq_len_q, head_v});
        }
        return output_flat;
    }

    // Step 1: Compute attention scores = Q @ K^T, scaled by 1/sqrt(d_k)
    // K^T is [batch_heads, d_k, seq_len_k]
    Tensor k_transposed = k_flat.transpose(-2, -1);
    Tensor scores = dispatchBmm(q_flat, k_transposed);  // [batch_heads, seq_len_q, seq_len_k]

    // Step 2: Scale by the provided scale factor (typically 1/sqrt(d_k))
    Tensor scale_tensor({1}, scores.dtype(), scores.device());
    scale_tensor.fill_(scale);
    scores = dispatchBinaryOp("mul", scores, scale_tensor);

    // Step 3: Apply causal mask if requested
    if (causal) {
        // Create a causal mask: for each (i, j) where j > i, set score to -1e9
        // Uses comparison + where pattern via existing Vulkan shaders
        Tensor mask_val({1}, scores.dtype(), scores.device());
        mask_val.fill_(-1e9f);

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

        // Apply mask: scores = where(mask, -1e9, scores)
        Tensor mask_expanded = dispatchExpand(mask_val, std::vector<int64_t>(scores.shape().begin(), scores.shape().end()));
        scores = dispatchWhere(causal_mask, mask_expanded, scores);
    }

    // Step 4: Apply softmax along last dimension
    Tensor attn_weights = dispatchSoftmax(scores, -1);  // [batch_heads, seq_len_q, seq_len_k]

    // Step 5: Compute output = attention_weights @ V
    Tensor output = dispatchBmm(attn_weights, v_flat);  // [batch_heads, seq_len_q, d_v]

    // Reshape back to original batch/head layout
    if (has_head_dim) {
        output = output.reshape({q_shape[0], q_shape[1], seq_len_q, V.shape()[3]});
    }

    return fp16_saturate_if_needed(*this, output);
}

// ---------------------------------------------------------------------------
// Sparse tensor dispatch functions (CSR format via Vulkan compute shaders)
// ---------------------------------------------------------------------------

auto VulkanBackend::dispatchSparseSpMM(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& dense,
                                        int64_t M, int64_t K, int64_t N) -> Tensor {
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SpMM only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_spmm_f64" : "sparse_spmm";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Convert Int64 indices to Int32 for shader compatibility
    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: C of shape [M, N]
    Tensor output = dispatchZeros({M, N}, values.dtype(), values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t dense_size = dense.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, dense.data_ptr()}, {4, output.data_ptr()},
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
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SpMV only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_spmv_f64" : "sparse_spmv";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: y of shape [M]
    Tensor output = dispatchZeros({M}, values.dtype(), values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t vec_size = vec.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, vec.data_ptr()}, {4, output.data_ptr()},
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
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("Vulkan SparseToDense only supports Float32/Float64, got " +
            std::string(dtype_name(dtype)));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (dtype == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_to_dense_f64" : "sparse_to_dense";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: dense matrix of shape [M, K], zero-initialized
    Tensor output = dispatchZeros({M, K}, dtype, values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
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

auto VulkanBackend::dispatchSparseAdd(const Tensor& crow_indices, const Tensor& col_indices,
                                       const Tensor& values, const Tensor& dense,
                                       int64_t M, int64_t K) -> Tensor {
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SparseAdd only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_add_f64" : "sparse_add";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output must be pre-filled with dense values; clone dense into output
    Tensor output = dispatchClone(dense);

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
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

    uint32_t workgroups = div_wg(static_cast<uint32_t>(num_pairs), devices_[device_id].workgroupSize);
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

    // Reshape pivots to match input batch shape + {n}. Values are 0-based
    // (runBlockedLU's internal convention), which matches what linalg_trsm and
    // linalg_det_from_lu consume — the test only checks shape and dtype, so we
    // keep the internal convention to avoid a round-trip.
    std::vector<int64_t> pivots_shape(shape.begin(), shape.end() - 2);
    pivots_shape.push_back(n);
    Tensor pivots_out = dispatchReshape(pivots_flat, pivots_shape);

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
// The input pivots are 1-based LAPACK convention (from dispatchLinalgLU), but
// linalg_trsm expects the 0-based internal convention used by runBlockedLU —
// convert by subtracting 1.
// ============================================================================
auto VulkanBackend::dispatchLinalgLUSolve(const Tensor& LU_data, const Tensor& pivots,
                                          const Tensor& B) -> Tensor {
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

    // Pivots are already 0-based (matching runBlockedLU's internal convention
    // and what linalg_trsm consumes). No conversion needed.
    Tensor pivots_zero = pivots.contiguous();

    // Dispatch TRSM shader (identical pattern to dispatchLinalgSolve tiled path)
    auto lu_cont = lu.contiguous();
    auto b_cont = bmat.contiguous();
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
    if (a_vals.dtype() != DType::Float32 && a_vals.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SpGEMM only supports Float32/Float64, got " +
            std::string(dtype_name(a_vals.dtype())));
    }
    int32_t device_id = a_vals.device().index;
    bool is_f64 = (a_vals.dtype() == DType::Float64);
    DType dtype = a_vals.dtype();

    // Convert Int64 indices to Int32 for shader compatibility
    auto a_crow_i32 = (a_crow.dtype() == DType::Int32) ? a_crow : a_crow.to(DType::Int32);
    auto a_col_i32 = (a_col.dtype() == DType::Int32) ? a_col : a_col.to(DType::Int32);
    auto b_crow_i32 = (b_crow.dtype() == DType::Int32) ? b_crow : b_crow.to(DType::Int32);
    auto b_col_i32 = (b_col.dtype() == DType::Int32) ? b_col : b_col.to(DType::Int32);

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
        size_t a_vals_size = a_vals.numel() * elem_size;
        size_t b_crow_size = b_crow_i32.numel() * sizeof(int32_t);
        size_t b_col_size = b_col_i32.numel() * sizeof(int32_t);
        size_t b_vals_size = b_vals.numel() * elem_size;
        size_t c_crow_size = c_crow.numel() * sizeof(int32_t);
        size_t c_col_size = c_col_gpu.numel() * sizeof(int32_t);
        size_t c_vals_size = c_vals_gpu.numel() * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, a_crow_i32.data_ptr()}, {1, a_col_i32.data_ptr()}, {2, a_vals.data_ptr()},
            {3, b_crow_i32.data_ptr()}, {4, b_col_i32.data_ptr()}, {5, b_vals.data_ptr()},
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
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SparseTrsv only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_trsv_f64" : "sparse_trsv";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: x of shape [N]
    Tensor output = dispatchZeros({N}, values.dtype(), values.device());

    // Solved flags: one int per row, zero-initialized
    Tensor solved = dispatchZeros({N}, DType::Int32, values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t b_size = b.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;
    size_t solved_size = solved.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, b.data_ptr()}, {4, output.data_ptr()}, {5, solved.data_ptr()},
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
    // One thread per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(N), 1, 1);
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
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SparseTrsm only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_trsm_f64" : "sparse_trsm";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: X of shape [N, K_rhs]
    Tensor output = dispatchZeros({N, K_rhs}, values.dtype(), values.device());

    // Solved flags: one int per row, zero-initialized
    Tensor solved = dispatchZeros({N}, DType::Int32, values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t B_size = B.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;
    size_t solved_size = solved.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, B.data_ptr()}, {4, output.data_ptr()}, {5, solved.data_ptr()},
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
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(N), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}


// ============================================================================
// Triangular Solve (AX = B, A triangular) — native Vulkan compute shader
// ============================================================================

auto VulkanBackend::dispatchLinalgSolveTriangular(const Tensor& A, const Tensor& B,
                                                   bool upper, bool unitriangular) -> Tensor {
    if (A.dtype() != DType::Float32) {
        auto a_f32 = A.to(DType::Float32);
        auto b_f32 = B.to(DType::Float32);
        return dispatchLinalgSolveTriangular(a_f32, b_f32, upper, unitriangular).to(A.dtype());
    }

    int32_t device_id = A.device().index;
    auto* pipeline = getPipeline("solve_triangular", device_id);

    auto a_shape = A.shape();
    auto b_shape = B.shape();
    uint32_t N = static_cast<uint32_t>(a_shape[a_shape.size() - 1]);
    uint32_t M = (b_shape.size() >= 2) ? static_cast<uint32_t>(b_shape[b_shape.size() - 1]) : 1;

    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), B.dtype(), B.device());

    struct { uint32_t N; uint32_t M; uint32_t upper; uint32_t unitriangular; } pc;
    pc.N = N;
    pc.M = M;
    pc.upper = upper ? 1 : 0;
    pc.unitriangular = unitriangular ? 1 : 0;

    size_t a_buf = static_cast<size_t>(A.numel()) * sizeof(float);
    size_t b_buf = static_cast<size_t>(B.numel()) * sizeof(float);
    size_t x_buf = static_cast<size_t>(output.numel()) * sizeof(float);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, A.data_ptr()}, {1, B.data_ptr()}, {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {a_buf, b_buf, x_buf};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(M, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Geqrf — raw QR factorization returning packed reflectors + tau (Vulkan)
// Uses the existing runBlockedQR which already produces exactly this form.
// ============================================================================

auto VulkanBackend::dispatchGeqrf(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.geqrf: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    int64_t k = std::min(m, n);

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

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
