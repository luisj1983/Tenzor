/**
 * @file test_model_format.cpp
 * @brief Tests for TZLite binary model format constants and reader/writer
 */

#include <gtest/gtest.h>
#include <tenzor/lite/model_format.hpp>

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
