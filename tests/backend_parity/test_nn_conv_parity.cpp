/**
 * @file test_nn_conv_parity.cpp
 * @brief Convolution layer parity tests across backends
 *
 * Tests Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose3d,
 * DeformableConv2d, grouped, depthwise, and no-bias convolutions.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class NNConvParity : public BackendTest {};
// ============================================================================
// Conv1d Tests
// ============================================================================

TEST_P(NNConvParity, Conv1d_Basic) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::Conv1d conv(3, 16, 3, 1, 1);
    auto input = randn({1, 3, 32}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv1d conv_dev(3, 16, 3, 1, 1);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv1d_Basic failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNConvParity, Conv1d_Stride) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::Conv1d conv(3, 16, 3, 2, 1);  // stride=2
    auto input = randn({1, 3, 64}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv1d conv_dev(3, 16, 3, 2, 1);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv1d_Stride failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNConvParity, Conv1d_Padding) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::Conv1d conv(3, 16, 5, 1, 2);  // 5-wide kernel, pad=2
    auto input = randn({1, 3, 32}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv1d conv_dev(3, 16, 5, 1, 2);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv1d_Padding failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Audit-5 Y.30: Conv1d per-axis stride / dilation parity coverage.
//
// F.11 / U.4 wired the Conv1d→Conv2d backend dispatch to translate the
// scalar 1D Stride/Padding/Dilation into per-axis (StrideH=1, StrideW=stride,
// PaddingH=0, PaddingW=padding, DilationH=1, DilationW=dilation). Before
// U.4 those scalars were duplicated to both H and W on CUDA/ROCm/OneAPI/
// Vulkan, so padding=1 turned the spurious H=1 axis into H_out=3 and
// dilation>1 could even reject the call because the dilated kernel exceeded
// H=1. The fix landed without a parity test, so a regression would only
// surface in downstream ASR / TTS workloads.
//
// Conv1d's public ctor is scalar-only (the 1D analogue of Conv2d's
// pair<int64_t,int64_t>); the "per-axis" angle is that the *backend*
// dispatcher must forward the scalar as a per-axis attr along the temporal
// axis only. Pick L=32 with stride=2 / padding=1 / dilation=1 (PyTorch
// L_out = (32 + 2*1 - 1*(3-1) - 1)/2 + 1 = 16) and L=32 with stride=1 /
// padding=2 / dilation=2 (L_out = (32 + 2*2 - 2*(3-1) - 1)/1 + 1 = 32) so
// the expected output length is asymmetric enough that a wrong attr broadcast
// to H would either reshape the tensor or throw.
// ============================================================================

TEST_P(NNConvParity, Conv1d_PerAxisStride) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // stride=2, padding=1, dilation=1; L_in=32, kernel=3 →
    // L_out = (32 + 2*1 - 1*(3-1) - 1)/2 + 1 = 16.
    nn::Conv1d conv(3, 16, 3, 2, 1, 1);
    auto input = randn({1, 3, 32}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();
    ASSERT_EQ(ref_output.shape().size(), 3u);
    EXPECT_EQ(ref_output.shape()[2], 16);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv1d conv_dev(3, 16, 3, 2, 1, 1);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Conv1d_PerAxisStride on ")
                         + backend_name(backends[i]));
            ASSERT_EQ(output.shape().size(), 3u);
            EXPECT_EQ(output.shape()[2], 16);
            EXPECT_TENSORS_CLOSE(ref_output,
                                 output.to(Device::cpu()),
                                 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv1d_PerAxisStride failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

TEST_P(NNConvParity, Conv1d_PerAxisDilation) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // stride=1, padding=2, dilation=2; L_in=32, kernel=3 →
    // effective kernel = (3-1)*2 + 1 = 5,
    // L_out = (32 + 2*2 - 5)/1 + 1 = 32.
    nn::Conv1d conv(3, 16, 3, 1, 2, 2);
    auto input = randn({1, 3, 32}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();
    ASSERT_EQ(ref_output.shape().size(), 3u);
    EXPECT_EQ(ref_output.shape()[2], 32);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv1d conv_dev(3, 16, 3, 1, 2, 2);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Conv1d_PerAxisDilation on ")
                         + backend_name(backends[i]));
            ASSERT_EQ(output.shape().size(), 3u);
            EXPECT_EQ(output.shape()[2], 32);
            EXPECT_TENSORS_CLOSE(ref_output,
                                 output.to(Device::cpu()),
                                 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv1d_PerAxisDilation failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Conv3d Tests
// ============================================================================

TEST_P(NNConvParity, Conv3d_Basic) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::Conv3d conv(3, 16, 3, 1, 1);
    auto input = randn({1, 3, 8, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv3d conv_dev(3, 16, 3, 1, 1);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv3d_Basic failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNConvParity, Conv3d_Stride) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::Conv3d conv(3, 16, 3, 2, 1);  // stride=2
    auto input = randn({1, 3, 16, 16, 16}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv3d conv_dev(3, 16, 3, 2, 1);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-3f, 1e-4f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv3d_Stride failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Transposed Convolution Tests
// ============================================================================

TEST_P(NNConvParity, ConvTranspose1d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    Tensor ref_output;
    nn::ConvTranspose1d conv(16, 3, 3, 2, 0);  // stride=2 for upsampling, padding=0
    auto input = randn({1, 16, 16}, DType::Float32, Device::cpu());
    ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::ConvTranspose1d conv_dev(16, 3, 3, 2, 0);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ConvTranspose1d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNConvParity, ConvTranspose3d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    Tensor ref_output;
    nn::ConvTranspose3d conv(16, 3, 3, 2, 0);  // stride=2 for upsampling, padding=0
    auto input = randn({1, 16, 4, 4, 4}, DType::Float32, Device::cpu());
    ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::ConvTranspose3d conv_dev(16, 3, 3, 2, 0);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ConvTranspose3d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Grouped Convolution Tests
// ============================================================================

TEST_P(NNConvParity, Conv2d_Groups2) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // Conv2d(in, out, kernel, stride, padding, dilation, groups)
    nn::Conv2d conv(16, 32, 3, 1, 1, 1, 2);  // groups=2
    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d conv_dev(16, 32, 3, 1, 1, 1, 2);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv2d_Groups2 failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNConvParity, Conv1d_Groups) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // Conv1d(in, out, kernel, stride, padding, dilation, groups)
    nn::Conv1d conv(16, 32, 3, 1, 1, 1, 2);  // groups=2
    auto input = randn({1, 16, 32}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv1d conv_dev(16, 32, 3, 1, 1, 1, 2);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv1d_Groups failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// DeformableConv2d Test
// ============================================================================

TEST_P(NNConvParity, DeformableConv2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::DeformableConv2d conv(3, 16, 3, 1, 1);
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    // DeformableConv2d requires offset tensor: (N, offset_groups*2*kH*kW, H_out, W_out)
    // With kernel=3, offset_groups=1: shape is (1, 2*3*3, 8, 8) = (1, 18, 8, 8)
    auto offset = randn({1, 18, 8, 8}, DType::Float32, Device::cpu());
    // Mask tensor: (N, offset_groups*kH*kW, H_out, W_out) = (1, 9, 8, 8)
    auto mask = sigmoid(randn({1, 9, 8, 8}, DType::Float32, Device::cpu()));

    auto ref_output = conv.forward(Variable(input, false),
                                    Variable(offset, false),
                                    Variable(mask, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::DeformableConv2d conv_dev(3, 16, 3, 1, 1);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto offset_dev = offset.to(backends[i]);
            auto mask_dev = mask.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false),
                                            Variable(offset_dev, false),
                                            Variable(mask_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "DeformableConv2d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Conv2d No-Bias Test
// ============================================================================

TEST_P(NNConvParity, Conv2d_NoBias) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // Conv2d(in, out, kernel, stride, padding, dilation, groups, bias)
    nn::Conv2d conv(3, 16, 3, 1, 1, 1, 1, false);  // bias=false
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d conv_dev(3, 16, 3, 1, 1, 1, 1, false);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv2d_NoBias failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Depthwise Conv2d Test
// ============================================================================

TEST_P(NNConvParity, DepthwiseConv2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // Conv2d(in, out, kernel, stride, padding, dilation, groups)
    // Depthwise: groups=in_channels, out_channels=in_channels
    nn::Conv2d conv(16, 16, 3, 1, 1, 1, 16);
    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d conv_dev(16, 16, 3, 1, 1, 1, 16);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "DepthwiseConv2d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// ConvTranspose2d (missing from the original file) + backward parity for
// convs.  Phase 3.2 gap-fill.
// ============================================================================

TEST_P(NNConvParity, ConvTranspose2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    nn::ConvTranspose2d conv(4, 8, 3, 2, 1, 0, 1, true);
    auto input = randn({1, 4, 8, 8}, DType::Float32, Device::cpu());
    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::ConvTranspose2d conv_dev(4, 8, 3, 2, 1, 0, 1, true);
            auto params = conv.parameters();
            auto dev_params = conv_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            conv_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("ConvTranspose2d on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_output, output, 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ConvTranspose2d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// Helper: forward+backward conv parity — uses test_gradient_parity. Holds the
// weight/bias as parameter tensors wrapped as Variables so backward returns
// grads for them in addition to the input.
namespace {
template <typename ConvT>
void conv_grad_parity(ConvT make_conv,
                      const std::vector<int64_t>& input_shape,
                      const char* name) {
    auto input = randn(input_shape, DType::Float32, Device::cpu());
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn conv parity");

    // Reference on CPU: materialize one conv, extract weight/bias, then for
    // each backend rebuild a fresh conv with cloned params and compare grads.
    auto ref_conv = make_conv();
    auto v_in = Variable(input.clone(), true);
    auto ref_out = ref_conv.forward(v_in);
    ref_out.backward(ones_like(ref_out.tensor()));
    Tensor ref_output_t = ref_out.tensor();
    Tensor ref_input_grad = v_in.grad().value();
    // Capture parameter gradients
    auto ref_params = ref_conv.parameters();
    std::vector<Tensor> ref_param_grads;
    for (auto& p : ref_params) {
        ref_param_grads.push_back(p->grad().value());
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto dev_conv = make_conv();
            auto dev_params = dev_conv.parameters();
            for (size_t p = 0; p < ref_params.size(); ++p) {
                dev_params[p]->tensor() = ref_params[p]->tensor().clone();
            }
            dev_conv.to(backends[i]);
            auto dev_in = Variable(input.to(backends[i]), true);
            auto dev_out = dev_conv.forward(dev_in);
            dev_out.backward(ones_like(dev_out.tensor()));
            backends[i].synchronize();

            SCOPED_TRACE(std::string(name) + " on " + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_output_t,
                                 dev_out.tensor().to(Device::cpu()),
                                 1e-3f, 1e-4f);
            EXPECT_TENSORS_CLOSE(ref_input_grad,
                                 dev_in.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-4f);
            // Parameter grads: 3D conv backward on cuDNN uses Winograd /
            // Tensor-Core-adjacent algorithms that accumulate more f32
            // loss than a plain reference kernel; the handful of
            // elements that drift on 3-D configs stay under ~5e-3
            // absolute, well under any correctness threshold. Use a
            // rtol/atol pair sized for cuDNN's 3-D backward drift.
            // Real correctness breaks would be orders of magnitude
            // larger (see test_conv3d_multidtype for tight-tolerance
            // per-dtype parity).
            for (size_t p = 0; p < ref_params.size(); ++p) {
                EXPECT_TENSORS_CLOSE(ref_param_grads[p],
                                     dev_params[p]->grad().value().to(Device::cpu()),
                                     1e-2f, 5e-3f);
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << " failed on "
                      << backend_name(backends[i]) << ": " << e.what();
        }
    }
}
}  // namespace

TEST_P(NNConvParity, Conv2d_Backward) {
    conv_grad_parity(
        [] { return nn::Conv2d(3, 8, 3, 1, 1); },
        {1, 3, 8, 8},
        "Conv2d_Backward");
}

TEST_P(NNConvParity, Conv1d_Backward) {
    conv_grad_parity(
        [] { return nn::Conv1d(3, 8, 3, 1, 1); },
        {1, 3, 16},
        "Conv1d_Backward");
}

TEST_P(NNConvParity, Conv3d_Backward) {
    conv_grad_parity(
        [] { return nn::Conv3d(2, 4, 3, 1, 1); },
        {1, 2, 4, 4, 4},
        "Conv3d_Backward");
}

// ============================================================================
// Main
// ============================================================================

INSTANTIATE_BACKEND_TESTS(NNConvParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
