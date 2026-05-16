// test_cpu_sparse_dtype_native.cpp
//
// Wave G: CPU sparse SpMM/SpMV now natively supports F16/BF16/Int32/Int64
// without tensor-wide widen-narrow. The `sparse_acc_type<T>` traits in the
// kernel templates select F32 accumulator for half-types (in-register
// widen per element) while keeping the input/output buffers in the native
// dtype. Public `spmm`/`spmv` no longer cast inputs through F32/F64 for
// these dtypes.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/sparse/sparse_tensor.hpp>
#include <tenzor/sparse/sparse_ops.hpp>

using namespace tenzor;

namespace {

class CpuSparseDtypeNative : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// Build a 4x4 CSR matrix in the given dtype.  Pattern:
//   row 0: A[0,0] = 1, A[0,2] = 2
//   row 1: (empty)
//   row 2: A[2,1] = 3
//   row 3: A[3,3] = 4
static auto build_csr_4x4(DType dtype) -> SparseTensor {
    // CSR components.
    auto crow = zeros({5}, DType::Int64, Device::cpu());
    auto col  = zeros({4}, DType::Int64, Device::cpu());
    auto vals = zeros({4}, dtype, Device::cpu());

    int64_t* cp = crow.data<int64_t>();
    int64_t* lp = col.data<int64_t>();
    cp[0] = 0; cp[1] = 2; cp[2] = 2; cp[3] = 3; cp[4] = 4;
    lp[0] = 0; lp[1] = 2; lp[2] = 1; lp[3] = 3;

    // Fill values from a F32 template via dtype-appropriate cast.
    auto vals_f32 = zeros({4}, DType::Float32, Device::cpu());
    float* vf = vals_f32.data<float>();
    vf[0] = 1; vf[1] = 2; vf[2] = 3; vf[3] = 4;
    vals = vals_f32.to(dtype);

    return SparseTensor::sparse_csr(crow, col, vals, {4, 4});
}

}  // namespace

TEST_F(CpuSparseDtypeNative, SpMV_F16_NativeMatchesF32) {
    // F32 reference.
    auto sparse_f32 = build_csr_4x4(DType::Float32);
    auto vec_f32 = zeros({4}, DType::Float32, Device::cpu());
    auto* vf = vec_f32.data<float>();
    vf[0] = 1.0f; vf[1] = 0.5f; vf[2] = -0.25f; vf[3] = 2.0f;

    Tensor ref = sparse::spmv(sparse_f32, vec_f32);
    ASSERT_EQ(ref.dtype(), DType::Float32);

    // Native F16.
    auto sparse_f16 = build_csr_4x4(DType::Float16);
    auto vec_f16 = vec_f32.to(DType::Float16);
    Tensor out_f16 = sparse::spmv(sparse_f16, vec_f16);
    ASSERT_EQ(out_f16.dtype(), DType::Float16);  // no widen-narrow

    auto ref_to_f16 = ref.to(DType::Float16).to(DType::Float32);
    auto out_to_f32 = out_f16.to(DType::Float32);
    for (int64_t i = 0; i < ref.numel(); ++i) {
        EXPECT_NEAR(ref_to_f16.data<float>()[i], out_to_f32.data<float>()[i], 5e-3f);
    }
}

TEST_F(CpuSparseDtypeNative, SpMV_BF16_NativeMatchesF32) {
    auto sparse_f32 = build_csr_4x4(DType::Float32);
    auto vec_f32 = zeros({4}, DType::Float32, Device::cpu());
    auto* vf = vec_f32.data<float>();
    vf[0] = 1.0f; vf[1] = 0.5f; vf[2] = -0.25f; vf[3] = 2.0f;

    Tensor ref = sparse::spmv(sparse_f32, vec_f32);

    auto sparse_bf16 = build_csr_4x4(DType::BFloat16);
    auto vec_bf16 = vec_f32.to(DType::BFloat16);
    Tensor out_bf16 = sparse::spmv(sparse_bf16, vec_bf16);
    ASSERT_EQ(out_bf16.dtype(), DType::BFloat16);

    auto ref_to_bf16 = ref.to(DType::BFloat16).to(DType::Float32);
    auto out_to_f32 = out_bf16.to(DType::Float32);
    for (int64_t i = 0; i < ref.numel(); ++i) {
        EXPECT_NEAR(ref_to_bf16.data<float>()[i], out_to_f32.data<float>()[i], 1e-2f);
    }
}

TEST_F(CpuSparseDtypeNative, SpMV_Int32_Native) {
    auto sparse_i32 = build_csr_4x4(DType::Int32);
    auto vec_i32 = zeros({4}, DType::Int32, Device::cpu());
    auto* vp = vec_i32.data<int32_t>();
    vp[0] = 2; vp[1] = 3; vp[2] = -1; vp[3] = 5;

    Tensor out = sparse::spmv(sparse_i32, vec_i32);
    ASSERT_EQ(out.dtype(), DType::Int32);

    // Expected: row 0: 1*2 + 2*(-1) = 0; row 1: 0; row 2: 3*3 = 9; row 3: 4*5 = 20.
    auto* op = out.data<int32_t>();
    EXPECT_EQ(op[0], 0);
    EXPECT_EQ(op[1], 0);
    EXPECT_EQ(op[2], 9);
    EXPECT_EQ(op[3], 20);
}

// ----------------------------------------------------------------------------
// Wave G1 (deferred → landed): complex sparse SpMV.
// ----------------------------------------------------------------------------
TEST_F(CpuSparseDtypeNative, SpMV_Complex64_Native) {
    // Build a 2x2 CSR with complex values.
    //   row 0: A[0,0] = (1, 0), A[0,1] = (0, 1)
    //   row 1: A[1,0] = (1, 1), A[1,1] = (2, -1)
    auto crow = zeros({3}, DType::Int64, Device::cpu());
    auto col  = zeros({4}, DType::Int64, Device::cpu());
    auto vals = zeros({4}, DType::Complex64, Device::cpu());
    auto* cp = crow.data<int64_t>();
    auto* lp = col.data<int64_t>();
    auto* vp = vals.data<std::complex<float>>();
    cp[0] = 0; cp[1] = 2; cp[2] = 4;
    lp[0] = 0; lp[1] = 1; lp[2] = 0; lp[3] = 1;
    vp[0] = {1.0f,  0.0f};
    vp[1] = {0.0f,  1.0f};
    vp[2] = {1.0f,  1.0f};
    vp[3] = {2.0f, -1.0f};
    auto sp = SparseTensor::sparse_csr(crow, col, vals, {2, 2});

    auto x = zeros({2}, DType::Complex64, Device::cpu());
    auto* xp = x.data<std::complex<float>>();
    xp[0] = {1.0f, 0.0f};
    xp[1] = {0.0f, 1.0f};

    Tensor out = sparse::spmv(sp, x);
    ASSERT_EQ(out.dtype(), DType::Complex64);
    auto* op = out.data<std::complex<float>>();
    // Row 0: (1,0)*(1,0) + (0,1)*(0,1) = (1,0) + (-1,0) = (0, 0)
    // Row 1: (1,1)*(1,0) + (2,-1)*(0,1) = (1,1) + (1,2) = (2, 3)
    EXPECT_NEAR(op[0].real(), 0.0f, 1e-5f);
    EXPECT_NEAR(op[0].imag(), 0.0f, 1e-5f);
    EXPECT_NEAR(op[1].real(), 2.0f, 1e-5f);
    EXPECT_NEAR(op[1].imag(), 3.0f, 1e-5f);
}

TEST_F(CpuSparseDtypeNative, SpMV_Complex128_Native) {
    auto crow = zeros({3}, DType::Int64, Device::cpu());
    auto col  = zeros({2}, DType::Int64, Device::cpu());
    auto vals = zeros({2}, DType::Complex128, Device::cpu());
    crow.data<int64_t>()[0] = 0;
    crow.data<int64_t>()[1] = 1;
    crow.data<int64_t>()[2] = 2;
    col.data<int64_t>()[0] = 0;
    col.data<int64_t>()[1] = 1;
    vals.data<std::complex<double>>()[0] = {2.0, 0.0};
    vals.data<std::complex<double>>()[1] = {0.0, 3.0};
    auto sp = SparseTensor::sparse_csr(crow, col, vals, {2, 2});

    auto x = zeros({2}, DType::Complex128, Device::cpu());
    x.data<std::complex<double>>()[0] = {1.0, 1.0};
    x.data<std::complex<double>>()[1] = {1.0, 0.0};

    Tensor out = sparse::spmv(sp, x);
    ASSERT_EQ(out.dtype(), DType::Complex128);
    auto* op = out.data<std::complex<double>>();
    // Row 0: (2,0)*(1,1) = (2, 2)
    // Row 1: (0,3)*(1,0) = (0, 3)
    EXPECT_NEAR(op[0].real(), 2.0, 1e-10);
    EXPECT_NEAR(op[0].imag(), 2.0, 1e-10);
    EXPECT_NEAR(op[1].real(), 0.0, 1e-10);
    EXPECT_NEAR(op[1].imag(), 3.0, 1e-10);
}

TEST_F(CpuSparseDtypeNative, SpMM_F16_NativeMatchesF32) {
    auto sparse_f32 = build_csr_4x4(DType::Float32);
    auto dense_f32 = zeros({4, 3}, DType::Float32, Device::cpu());
    auto* dp = dense_f32.data<float>();
    for (int64_t i = 0; i < 12; ++i) dp[i] = static_cast<float>(i % 7) * 0.25f;

    Tensor ref = sparse::spmm(sparse_f32, dense_f32);
    ASSERT_EQ(ref.dtype(), DType::Float32);

    auto sparse_f16 = build_csr_4x4(DType::Float16);
    auto dense_f16 = dense_f32.to(DType::Float16);
    Tensor out_f16 = sparse::spmm(sparse_f16, dense_f16);
    ASSERT_EQ(out_f16.dtype(), DType::Float16);

    auto ref_to_f16 = ref.to(DType::Float16).to(DType::Float32);
    auto out_to_f32 = out_f16.to(DType::Float32);
    for (int64_t i = 0; i < ref.numel(); ++i) {
        EXPECT_NEAR(ref_to_f16.data<float>()[i], out_to_f32.data<float>()[i], 5e-3f);
    }
}
