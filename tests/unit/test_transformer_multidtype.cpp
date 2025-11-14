#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// Helper function to convert DType to string
std::string dtype_to_string(DType dtype) {
    switch(dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        default: return "unknown";
    }
}

// ============================================================================
// Multi-DType Test Parameter Structure
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// ============================================================================
// Helper Functions for Multi-DType Testing
// ============================================================================

template<typename T>
T get_transformer_tolerance() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<T>(1e-4);  // Float32: moderate tolerance for transformers
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<T>(1e-8);  // Float64: high precision
    } else {  // Float16
        return static_cast<T>(1e-2);  // Float16: relaxed tolerance for complex ops
    }
}

template<typename T>
T get_consistency_tolerance() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<T>(1e-5);
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<T>(1e-10);
    } else {  // Float16
        return static_cast<T>(5e-3);  // Very relaxed for Float16
    }
}

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class BackendDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }
};

// ============================================================================
// PositionalEncoding Tests
// ============================================================================

class PositionalEncodingMultiDTypeTest : public BackendDTypeTest {};

TEST_P(PositionalEncodingMultiDTypeTest, Construction) {
    EXPECT_NO_THROW({
        PositionalEncoding pe(512);
        pe.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_NO_THROW({
        PositionalEncoding pe(768, 10000, 0.1);
        pe.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(PositionalEncodingMultiDTypeTest, ForwardShape) {
    PositionalEncoding pe(256, 1000, 0.0);
    pe.to(device);

    int64_t batch_size = 4;
    int64_t seq_len = 20;
    int64_t d_model = 256;

    Variable input(randn({batch_size, seq_len, d_model}, dtype, device), true);
    Variable output = pe.forward(input);

    // Shape should be preserved
    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(PositionalEncodingMultiDTypeTest, ExceedsMaxLen) {
    PositionalEncoding pe(128, 100, 0.0);
    pe.to(device);

    Variable input(randn({2, 150, 128}, dtype, device), true);

    // Should throw because seq_len (150) > max_len (100)
    EXPECT_THROW({
        pe.forward(input);
    }, std::runtime_error) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

TEST_P(PositionalEncodingMultiDTypeTest, WithDropout) {
    PositionalEncoding pe(256, 1000, 0.5);
    pe.to(device);

    Variable input(randn({2, 10, 256}, dtype, device), true);

    // Should not throw
    EXPECT_NO_THROW({
        Variable output = pe.forward(input);
        EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

// ============================================================================
// TransformerEncoderLayer Tests
// ============================================================================

class TransformerEncoderLayerMultiDTypeTest : public BackendDTypeTest {};

TEST_P(TransformerEncoderLayerMultiDTypeTest, Construction) {
    EXPECT_NO_THROW({
        TransformerEncoderLayer layer(512, 8);
        layer.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_NO_THROW({
        TransformerEncoderLayer layer(768, 12, 3072, 0.1, "gelu", true);
        layer.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerEncoderLayerMultiDTypeTest, InvalidActivation) {
    EXPECT_THROW({
        TransformerEncoderLayer layer(512, 8, 2048, 0.1, "invalid");
        layer.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerEncoderLayerMultiDTypeTest, ForwardShape) {
    TransformerEncoderLayer layer(256, 4, 1024, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;
    int64_t d_model = 256;

    Variable src(randn({batch_size, seq_len, d_model}, dtype, device), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    // Shape should be preserved
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerEncoderLayerMultiDTypeTest, BatchFirstFalse) {
    TransformerEncoderLayer layer(256, 4, 1024, 0.0, "relu", false);
    layer.to(device);

    int64_t seq_len = 10;
    int64_t batch_size = 2;
    int64_t d_model = 256;

    Variable src(randn({seq_len, batch_size, d_model}, dtype, device), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], seq_len) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], batch_size) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerEncoderLayerMultiDTypeTest, WithMask) {
    TransformerEncoderLayer layer(128, 4, 512, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Variable src(randn({batch_size, seq_len, 128}, dtype, device), true);
    Tensor mask = create_causal_mask(seq_len, device, dtype);

    EXPECT_NO_THROW({
        Variable output = layer.forward(src, mask);
        EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerEncoderLayerMultiDTypeTest, GeLUActivation) {
    TransformerEncoderLayer layer(256, 8, 1024, 0.0, "gelu", true);
    layer.to(device);

    Variable src(randn({2, 5, 256}, dtype, device), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

// ============================================================================
// TransformerEncoder Tests
// ============================================================================

class TransformerEncoderMultiDTypeTest : public BackendDTypeTest {};

TEST_P(TransformerEncoderMultiDTypeTest, Construction) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(512, 8);

    EXPECT_NO_THROW({
        TransformerEncoder encoder(encoder_layer, 6);
        encoder.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
    EXPECT_NO_THROW({
        TransformerEncoder encoder(encoder_layer, 6, norm);
        encoder.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerEncoderMultiDTypeTest, ForwardShape) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(256, 4, 1024, 0.0, "relu", true);
    TransformerEncoder encoder(encoder_layer, 3);
    encoder.to(device);

    Variable src(randn({2, 10, 256}, dtype, device), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerEncoderMultiDTypeTest, MultipleLayersProcessing) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(128, 4, 512, 0.0, "relu", true);
    TransformerEncoder encoder(encoder_layer, 6);  // 6 layers
    encoder.to(device);

    Variable src(randn({2, 5, 128}, dtype, device), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    // Output should still have same shape
    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerEncoderMultiDTypeTest, WithNormalization) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(256, 8, 1024, 0.0, "relu", true);
    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{256});
    TransformerEncoder encoder(encoder_layer, 6, norm);
    encoder.to(device);

    Variable src(randn({2, 10, 256}, dtype, device), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

// ============================================================================
// TransformerDecoderLayer Tests
// ============================================================================

class TransformerDecoderLayerMultiDTypeTest : public BackendDTypeTest {};

TEST_P(TransformerDecoderLayerMultiDTypeTest, Construction) {
    EXPECT_NO_THROW({
        TransformerDecoderLayer layer(512, 8);
        layer.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_NO_THROW({
        TransformerDecoderLayer layer(768, 12, 3072, 0.1, "gelu", true);
        layer.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerDecoderLayerMultiDTypeTest, ForwardShape) {
    TransformerDecoderLayer layer(256, 4, 1024, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t tgt_len = 8;
    int64_t src_len = 10;
    int64_t d_model = 256;

    Variable tgt(randn({batch_size, tgt_len, d_model}, dtype, device), true);
    Variable memory(randn({batch_size, src_len, d_model}, dtype, device), true);

    Variable output = layer.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], tgt_len) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerDecoderLayerMultiDTypeTest, WithCausalMask) {
    TransformerDecoderLayer layer(128, 4, 512, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t tgt_len = 5;
    int64_t src_len = 8;

    Variable tgt(randn({batch_size, tgt_len, 128}, dtype, device), true);
    Variable memory(randn({batch_size, src_len, 128}, dtype, device), true);

    Tensor tgt_mask = create_causal_mask(tgt_len, device, dtype);

    EXPECT_NO_THROW({
        Variable output = layer.forward(tgt, memory, tgt_mask);
        EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerDecoderLayerMultiDTypeTest, InvalidSingleInputForward) {
    TransformerDecoderLayer layer(256, 8);
    layer.to(device);

    Variable input(randn({2, 5, 256}, dtype, device), true);

    // Should throw because decoder needs both tgt and memory
    EXPECT_THROW({
        layer.forward(input);
    }, std::runtime_error) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

// ============================================================================
// TransformerDecoder Tests
// ============================================================================

class TransformerDecoderMultiDTypeTest : public BackendDTypeTest {};

TEST_P(TransformerDecoderMultiDTypeTest, Construction) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(512, 8);

    EXPECT_NO_THROW({
        TransformerDecoder decoder(decoder_layer, 6);
        decoder.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
    EXPECT_NO_THROW({
        TransformerDecoder decoder(decoder_layer, 6, norm);
        decoder.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerDecoderMultiDTypeTest, ForwardShape) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(256, 4, 1024, 0.0, "relu", true);
    TransformerDecoder decoder(decoder_layer, 3);
    decoder.to(device);

    Variable tgt(randn({2, 8, 256}, dtype, device), true);
    Variable memory(randn({2, 10, 256}, dtype, device), true);

    Variable output = decoder.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 8) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerDecoderMultiDTypeTest, MultipleLayersProcessing) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(128, 4, 512, 0.0, "relu", true);
    TransformerDecoder decoder(decoder_layer, 6);
    decoder.to(device);

    Variable tgt(randn({2, 5, 128}, dtype, device), true);
    Variable memory(randn({2, 7, 128}, dtype, device), true);

    Variable output = decoder.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

// ============================================================================
// Complete Transformer Tests
// ============================================================================

class TransformerMultiDTypeTest : public BackendDTypeTest {};

TEST_P(TransformerMultiDTypeTest, Construction) {
    EXPECT_NO_THROW({
        Transformer model(512, 8, 6, 6);
        model.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_NO_THROW({
        Transformer model(768, 12, 12, 12, 3072, 0.1, "gelu", true);
        model.to(device);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerMultiDTypeTest, ForwardShape) {
    Transformer model(256, 4, 3, 3, 1024, 0.0, "relu", true);
    model.to(device);

    int64_t batch_size = 2;
    int64_t src_len = 10;
    int64_t tgt_len = 8;
    int64_t d_model = 256;

    Variable src(randn({batch_size, src_len, d_model}, dtype, device), true);
    Variable tgt(randn({batch_size, tgt_len, d_model}, dtype, device), true);

    Variable output = model.forward(src, tgt);

    // Output should have target's sequence length
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], tgt_len) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerMultiDTypeTest, WithMasks) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);
    model.to(device);

    int64_t batch_size = 2;
    int64_t src_len = 10;
    int64_t tgt_len = 8;

    Variable src(randn({batch_size, src_len, 128}, dtype, device), true);
    Variable tgt(randn({batch_size, tgt_len, 128}, dtype, device), true);

    Tensor tgt_mask = create_causal_mask(tgt_len, device, dtype);

    EXPECT_NO_THROW({
        Variable output = model.forward(src, tgt, Tensor{}, tgt_mask);
        EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerMultiDTypeTest, InvalidSingleInputForward) {
    Transformer model(256, 8);
    model.to(device);

    Variable input(randn({2, 10, 256}, dtype, device), true);

    // Should throw because transformer needs both src and tgt
    EXPECT_THROW({
        model.forward(input);
    }, std::runtime_error) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerMultiDTypeTest, BERTConfig) {
    // Test BERT-like configuration: 768 dims, 12 heads, 12 layers
    // Use smaller config for Float16 to avoid precision issues
    int64_t d_model = (dtype == DType::Float16) ? 384 : 768;
    int64_t num_heads = (dtype == DType::Float16) ? 6 : 12;
    int64_t num_layers = (dtype == DType::Float16) ? 6 : 12;
    int64_t d_ff = (dtype == DType::Float16) ? 1536 : 3072;

    Transformer model(d_model, num_heads, num_layers, num_layers, d_ff, 0.1, "gelu", true);
    model.to(device);

    int64_t seq_len = (dtype == DType::Float16) ? 32 : 128;
    Variable src(randn({1, seq_len, d_model}, dtype, device), true);
    Variable tgt(randn({1, seq_len/2, d_model}, dtype, device), true);

    EXPECT_NO_THROW({
        Variable output = model.forward(src, tgt);
        EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string()
            << " with dtype " << dtype_to_string(dtype);
        EXPECT_EQ(output.shape()[1], seq_len/2) << "Failed on " << device.to_string()
            << " with dtype " << dtype_to_string(dtype);
        EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
            << " with dtype " << dtype_to_string(dtype);
        EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerMultiDTypeTest, GPTLikeConfig) {
    // Test GPT-like configuration with causal masking
    // Adjust model size based on dtype precision
    int64_t d_model = (dtype == DType::Float16) ? 256 : 512;
    int64_t num_heads = (dtype == DType::Float16) ? 4 : 8;
    int64_t num_layers = (dtype == DType::Float16) ? 3 : 6;
    int64_t d_ff = (dtype == DType::Float16) ? 1024 : 2048;

    Transformer model(d_model, num_heads, num_layers, num_layers, d_ff, 0.1, "gelu", true);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = (dtype == DType::Float16) ? 16 : 32;

    Variable src(randn({batch_size, seq_len, d_model}, dtype, device), true);
    Variable tgt(randn({batch_size, seq_len, d_model}, dtype, device), true);

    Tensor causal_mask = create_causal_mask(seq_len, device, dtype);

    Variable output = model.forward(src, tgt, Tensor{}, causal_mask);

    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.dtype(), dtype) << "Output dtype mismatch";
}

// ============================================================================
// Integration Tests
// ============================================================================

class TransformerIntegrationMultiDTypeTest : public BackendDTypeTest {};

TEST_P(TransformerIntegrationMultiDTypeTest, ForwardBackward) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);
    model.to(device);

    Variable src(randn({2, 5, 128}, dtype, device), true);
    Variable tgt(randn({2, 3, 128}, dtype, device), true);

    Variable output = model.forward(src, tgt);
    Variable loss = mean(output);

    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_TRUE(src.has_grad()) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_TRUE(tgt.has_grad()) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerIntegrationMultiDTypeTest, ParameterCount) {
    Transformer model(256, 4, 2, 2, 1024, 0.0, "relu", true);
    model.to(device);

    auto params = model.parameters();

    // Each encoder layer has: self_attn (8 params), linear1, linear2 (4 params), norm1, norm2 (4 params)
    // Each decoder layer has: self_attn (8), cross_attn (8), linear1, linear2 (4), norm1, norm2, norm3 (6)
    // Plus encoder norm and decoder norm
    // Total should be substantial
    EXPECT_GT(params.size(), 0) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerIntegrationMultiDTypeTest, TrainEvalSwitch) {
    Transformer model(128, 4, 2, 2);
    model.to(device);

    EXPECT_TRUE(model.is_training()) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);

    model.eval();
    EXPECT_FALSE(model.is_training()) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);

    model.train();
    EXPECT_TRUE(model.is_training()) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
}

TEST_P(TransformerIntegrationMultiDTypeTest, SmallModelConsistency) {
    // Test that a small model produces consistent outputs (sanity check)
    Transformer model(64, 2, 1, 1, 128, 0.0, "relu", true);
    model.to(device);
    model.eval();  // Use eval mode for deterministic forward passes

    Variable src(ones({1, 3, 64}, dtype, device), true);
    Variable tgt(ones({1, 2, 64}, dtype, device), true);

    Variable output1 = model.forward(src, tgt);

    // Run multiple forward passes - output should be consistent
    Variable output2 = model.forward(src, tgt);

    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());

    // Use dtype-specific comparison
    if (dtype == DType::Float32) {
        auto data1 = output1_cpu.data<float>();
        auto data2 = output2_cpu.data<float>();
        auto tol = get_consistency_tolerance<float>();
        for (int64_t i = 0; i < output1_cpu.numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol) << "Mismatch at index " << i
                << " on " << device.to_string() << " with dtype Float32";
        }
    } else if (dtype == DType::Float64) {
        auto data1 = output1_cpu.data<double>();
        auto data2 = output2_cpu.data<double>();
        auto tol = get_consistency_tolerance<double>();
        for (int64_t i = 0; i < output1_cpu.numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol) << "Mismatch at index " << i
                << " on " << device.to_string() << " with dtype Float64";
        }
    } else if (dtype == DType::Float16) {
        // Float16 requires conversion to Float32 for reading
        auto output1_f32 = output1_cpu.to(DType::Float32);
        auto output2_f32 = output2_cpu.to(DType::Float32);
        auto data1 = output1_f32.data<float>();
        auto data2 = output2_f32.data<float>();
        auto tol = get_consistency_tolerance<float>();
        for (int64_t i = 0; i < output1_f32.numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol) << "Mismatch at index " << i
                << " on " << device.to_string() << " with dtype Float16";
        }
    }
}

TEST_P(TransformerIntegrationMultiDTypeTest, DifferentSequenceLengths) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);
    model.to(device);

    // Test with different source and target lengths
    Variable src1(randn({2, 20, 128}, dtype, device), true);
    Variable tgt1(randn({2, 10, 128}, dtype, device), true);

    Variable output1 = model.forward(src1, tgt1);
    EXPECT_EQ(output1.shape()[1], 10) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output1.dtype(), dtype) << "Output dtype mismatch";

    Variable src2(randn({2, 5, 128}, dtype, device), true);
    Variable tgt2(randn({2, 15, 128}, dtype, device), true);

    Variable output2 = model.forward(src2, tgt2);
    EXPECT_EQ(output2.shape()[1], 15) << "Failed on " << device.to_string()
        << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output2.dtype(), dtype) << "Output dtype mismatch";
}

TEST_P(TransformerIntegrationMultiDTypeTest, DTypePrecisionComparison) {
    // This test validates that different dtypes produce reasonable outputs
    // but may have different precision characteristics
    Transformer model(64, 2, 1, 1, 128, 0.0, "relu", true);
    model.to(device);
    model.eval();

    Variable src(randn({1, 5, 64}, dtype, device), true);
    Variable tgt(randn({1, 3, 64}, dtype, device), true);

    Variable output = model.forward(src, tgt);

    // Verify output is finite and reasonable
    auto output_cpu = output.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        auto data = output_cpu.data<float>();
        for (int64_t i = 0; i < output_cpu.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(data[i])) << "Non-finite value at index " << i
                << " on " << device.to_string() << " with Float32";
            EXPECT_LE(std::abs(data[i]), 100.0f) << "Unexpectedly large value at index " << i;
        }
    } else if (dtype == DType::Float64) {
        auto data = output_cpu.data<double>();
        for (int64_t i = 0; i < output_cpu.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(data[i])) << "Non-finite value at index " << i
                << " on " << device.to_string() << " with Float64";
            EXPECT_LE(std::abs(data[i]), 100.0) << "Unexpectedly large value at index " << i;
        }
    } else if (dtype == DType::Float16) {
        // Float16 requires conversion to Float32 for reading
        auto output_f32 = output_cpu.to(DType::Float32);
        auto data = output_f32.data<float>();
        for (int64_t i = 0; i < output_f32.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(data[i])) << "Non-finite value at index " << i
                << " on " << device.to_string() << " with Float16";
            EXPECT_LE(std::abs(data[i]), 1000.0f) << "Unexpectedly large value at index " << i;
        }
    }
}

// ============================================================================
// Test Parameter Generation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    // Test with floating-point dtypes (transformer works with floating-point)
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Float16, "float16"},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

// Instantiate tests for all backend-dtype combinations
INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    PositionalEncodingMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    TransformerEncoderLayerMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    TransformerEncoderMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    TransformerDecoderLayerMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    TransformerDecoderMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    TransformerMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    TransformerIntegrationMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);
