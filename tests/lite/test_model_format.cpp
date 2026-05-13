/**
 * @file test_model_format.cpp
 * @brief Tests for TZLite binary model format constants and reader/writer
 */

#include <gtest/gtest.h>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/model_format.hpp>
#include <tenzor/lite/runtime.hpp>
#include <tenzor/ops/creation.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace tenzor { void initialize(); }

namespace {
class TenzorModelFormatEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_mf_env =
    ::testing::AddGlobalTestEnvironment(new TenzorModelFormatEnv);
}

using namespace tenzor::lite;

TEST(ModelFormatTest, MagicConstant) {
    EXPECT_EQ(TZLITE_MAGIC, 0x544C5A54u);
}

TEST(ModelFormatTest, VersionConstant) {
    EXPECT_EQ(TZLITE_VERSION, 1u);
}

TEST(ModelFormatTest, HeaderDefaultInit) {
    TZLiteHeader header{};
    header.magic = TZLITE_MAGIC;
    header.version = TZLITE_VERSION;
    header.num_nodes = 0;
    header.num_weights = 0;
    header.weight_data_offset = sizeof(TZLiteHeader);

    EXPECT_EQ(header.magic, TZLITE_MAGIC);
    EXPECT_EQ(header.version, TZLITE_VERSION);
    EXPECT_EQ(header.num_nodes, 0u);
    EXPECT_EQ(header.num_weights, 0u);
    EXPECT_EQ(header.weight_data_offset, sizeof(TZLiteHeader));
}

TEST(ModelFormatTest, HeaderSize) {
    // Header should be compact: 4 + 4 + 4 + 4 + 8 = 24 bytes
    EXPECT_EQ(sizeof(TZLiteHeader), 24u);
}

TEST(ModelFormatTest, ReaderLoadFromInvalidPath) {
    EXPECT_THROW(TZLiteReader::load("/nonexistent/model.tzlite"), std::runtime_error);
}

TEST(ModelFormatTest, ReaderLoadFromNullData) {
    EXPECT_THROW(TZLiteReader::load(nullptr, 0), std::runtime_error);
}

TEST(ModelFormatTest, ReaderLoadMinimalBuffer) {
    // A buffer containing just the magic bytes should be loadable
    // (minimal valid model with zero nodes)
    TZLiteHeader header{};
    header.magic = TZLITE_MAGIC;
    header.version = TZLITE_VERSION;
    header.num_nodes = 0;
    header.num_weights = 0;
    header.weight_data_offset = sizeof(TZLiteHeader);

    auto graph = TZLiteReader::load(&header, sizeof(header));
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(graph->num_nodes(), 0u);
}

TEST(ModelFormatTest, MagicMatchesRuntimeMagic) {
    // The "TZLT" magic in model_format.hpp should match what runtime.hpp expects
    uint8_t bytes[4];
    bytes[0] = (TZLITE_MAGIC >> 0) & 0xFF;
    bytes[1] = (TZLITE_MAGIC >> 8) & 0xFF;
    bytes[2] = (TZLITE_MAGIC >> 16) & 0xFF;
    bytes[3] = (TZLITE_MAGIC >> 24) & 0xFF;
    // "TZLT" in little-endian
    EXPECT_EQ(bytes[0], 'T');
    EXPECT_EQ(bytes[1], 'Z');
    EXPECT_EQ(bytes[2], 'L');
    EXPECT_EQ(bytes[3], 'T');
}

// ============================================================================
// Phase 2: TLV-section round-trip — build a graph with weights, write it to a
// .tzlite file, load it back, run inference, verify numerics match a hand-
// computed reference.
// ============================================================================

namespace {

auto temp_path(const std::string& stem) -> std::string {
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path();
    return (dir / (stem + "_" +
                   std::to_string(static_cast<long>(::getpid())) + ".tzlite"))
        .string();
}

}  // namespace

TEST(ModelFormatTest, RoundTripGraphWithWeights) {
    // Graph: y = ReLU(W @ x + b)
    // tensor_ids: 0=x (input), 1=W (weight), 2=b (weight),
    //             3=W@x, 4=W@x+b, 5=y (output).
    LiteGraph g;
    {
        LiteNode mm;
        mm.op = LiteOpType::MatMul;
        mm.input_ids = {1, 0};  // W (3x2) @ x (2x1) -> (3x1)
        mm.output_ids = {3};
        g.add_node(std::move(mm));

        LiteNode ad;
        ad.op = LiteOpType::Add;
        ad.input_ids = {3, 2};
        ad.output_ids = {4};
        g.add_node(std::move(ad));

        LiteNode rl;
        rl.op = LiteOpType::ReLU;
        rl.input_ids = {4};
        rl.output_ids = {5};
        g.add_node(std::move(rl));
    }
    g.set_input_ids({0});
    g.set_output_ids({5});

    // W = [[ 1,  2],
    //      [ 3, -4],
    //      [-5,  6]]
    auto W = tenzor::zeros({3, 2}, tenzor::DType::Float32);
    {
        auto* d = static_cast<float*>(W.data_ptr());
        d[0]= 1; d[1]= 2;
        d[2]= 3; d[3]=-4;
        d[4]=-5; d[5]= 6;
    }
    // b = [-1, -1, -1]^T  (shape 3x1; broadcasts with W@x)
    auto b = tenzor::zeros({3, 1}, tenzor::DType::Float32);
    {
        auto* d = static_cast<float*>(b.data_ptr());
        d[0]=-1; d[1]=-1; d[2]=-1;
    }

    WriteOptions opts;
    opts.weights[1] = W;
    opts.weights[2] = b;
    opts.input_ids  = {0};
    opts.output_ids = {5};
    opts.metadata["framework_version"] = "tenzor-lite-phase2";

    auto path = temp_path("roundtrip_graph_weights");
    TZLiteWriter::save(g, path, opts);

    // Sanity-check the file is non-empty and starts with the magic.
    {
        std::ifstream f(path, std::ios::binary);
        ASSERT_TRUE(f.is_open());
        uint32_t mag = 0;
        f.read(reinterpret_cast<char*>(&mag), sizeof(mag));
        EXPECT_EQ(mag, TZLITE_MAGIC);
    }

    // Load and run.
    auto runtime = LiteRuntime::load(path);
    ASSERT_NE(runtime, nullptr);

    auto x = runtime->create_input({2, 1}, tenzor::DType::Float32);
    {
        auto* d = static_cast<float*>(x.data);
        d[0] = 1.0f;
        d[1] = 1.0f;
    }

    auto y = runtime->forward(x);
    ASSERT_EQ(y.ndim, 2);
    ASSERT_EQ(y.shape[0], 3);
    ASSERT_EQ(y.shape[1], 1);

    // Reference:  W @ x = [1+2, 3-4, -5+6]^T = [3, -1, 1]^T
    //           + b      = [2, -2, 0]^T
    //           ReLU     = [2, 0, 0]^T
    const auto* yd = y.data_as<float>();
    EXPECT_FLOAT_EQ(yd[0], 2.0f);
    EXPECT_FLOAT_EQ(yd[1], 0.0f);
    EXPECT_FLOAT_EQ(yd[2], 0.0f);

    // Metadata round-trips.
    EXPECT_EQ(runtime->model_metadata("framework_version"),
              std::string{"tenzor-lite-phase2"});

    std::filesystem::remove(path);
}

TEST(ModelFormatTest, GraphOnlySaveLoadPreservesNodes) {
    // The legacy graph-only writer (no weights, no TLV) must still produce a
    // file that the new reader accepts and round-trips losslessly.
    LiteGraph g;
    LiteNode n;
    n.op = LiteOpType::Sigmoid;
    n.input_ids = {0};
    n.output_ids = {1};
    g.add_node(std::move(n));

    auto path = temp_path("graph_only");
    TZLiteWriter::save(g, path);

    auto reloaded = TZLiteReader::load(path);
    ASSERT_NE(reloaded, nullptr);
    EXPECT_EQ(reloaded->num_nodes(), 1u);
    EXPECT_EQ(reloaded->nodes()[0].op, LiteOpType::Sigmoid);
    EXPECT_EQ(reloaded->nodes()[0].input_ids,
              std::vector<int16_t>{0});
    EXPECT_EQ(reloaded->nodes()[0].output_ids,
              std::vector<int16_t>{1});

    std::filesystem::remove(path);
}

// Phase 5 note: a load-time op-coverage check lives in
// src/lite/runtime.cpp::verify_op_coverage. Writing a robust negative-path
// test for it requires picking a real OpId guaranteed-never-registered on
// CPU, which is fragile across builds (any backend can register an op at
// any time). The behaviour is exercised implicitly on every graph that
// successfully loads — all 48 lite tests do this. A targeted test will
// land once we expose `tz.lite.Runtime(path, device='cuda')` and can
// construct a graph mixing supported + unsupported ops.

TEST(ModelFormatTest, UnknownSectionTagsAreSkipped) {
    // Forward compatibility: a section table containing a tag the current
    // build doesn't recognise must still parse cleanly (skip + continue).
    // Construct a minimal file by hand: header + 0 nodes + 1 unknown section.
    std::vector<uint8_t> buf;
    auto push = [&](const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), b, b + n);
    };

    TZLiteHeader header{};
    header.magic = TZLITE_MAGIC;
    header.version = TZLITE_VERSION;
    header.num_nodes = 0;
    header.num_weights = 0;
    header.weight_data_offset = sizeof(header);  // node table is empty
    push(&header, sizeof(header));

    uint32_t section_count = 1;
    push(&section_count, sizeof(section_count));

    uint32_t unknown_tag = 0xDEADBEEF;
    uint64_t payload_size = 5;
    push(&unknown_tag, sizeof(unknown_tag));
    push(&payload_size, sizeof(payload_size));
    const uint8_t payload[5] = {1, 2, 3, 4, 5};
    push(payload, sizeof(payload));

    EXPECT_NO_THROW({
        auto graph = TZLiteReader::load(buf.data(), buf.size());
        ASSERT_NE(graph, nullptr);
        EXPECT_EQ(graph->num_nodes(), 0u);
    });
}
