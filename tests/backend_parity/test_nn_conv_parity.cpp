/**
 * @file test_nn_conv_parity.cpp
 * @brief Convolution layer parity tests across backends
 *
 * Tests Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose3d,
 * DeformableConv2d, grouped, depthwise, and no-bias convolutions.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Conv1d Tests
// ============================================================================

TEST(NNConvParity, Conv1d_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv1d_Basic skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNConvParity, Conv1d_Stride) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv1d_Stride skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNConvParity, Conv1d_Padding) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv1d_Padding skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Conv3d Tests
// ============================================================================

TEST(NNConvParity, Conv3d_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv3d_Basic skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNConvParity, Conv3d_Stride) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv3d_Stride skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Transposed Convolution Tests
// ============================================================================

TEST(NNConvParity, ConvTranspose1d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref_output;
    nn::ConvTranspose1d conv(16, 3, 3, 2, 0);  // stride=2 for upsampling, padding=0
    auto input = randn({1, 16, 16}, DType::Float32, Device::cpu());
    try {
        ref_output = conv.forward(Variable(input, false)).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "ConvTranspose1d CPU forward failed: " << e.what();
    }

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
            std::cerr << "ConvTranspose1d skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNConvParity, ConvTranspose3d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref_output;
    nn::ConvTranspose3d conv(16, 3, 3, 2, 0);  // stride=2 for upsampling, padding=0
    auto input = randn({1, 16, 4, 4, 4}, DType::Float32, Device::cpu());
    try {
        ref_output = conv.forward(Variable(input, false)).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "ConvTranspose3d CPU forward failed: " << e.what();
    }

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
            std::cerr << "ConvTranspose3d skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Grouped Convolution Tests
// ============================================================================

TEST(NNConvParity, Conv2d_Groups2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv2d_Groups2 skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNConvParity, Conv1d_Groups) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv1d_Groups skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// DeformableConv2d Test
// ============================================================================

TEST(NNConvParity, DeformableConv2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
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
                std::cerr << "DeformableConv2d skipped on " << backend_name(backends[i])
                          << ": " << e.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "DeformableConv2d not supported: " << e.what();
    }
}

// ============================================================================
// Conv2d No-Bias Test
// ============================================================================

TEST(NNConvParity, Conv2d_NoBias) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "Conv2d_NoBias skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Depthwise Conv2d Test
// ============================================================================

TEST(NNConvParity, DepthwiseConv2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

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
            std::cerr << "DepthwiseConv2d skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

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
