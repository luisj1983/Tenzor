#include <gtest/gtest.h>
#include <tenzor/nn/pytorch_loader.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace tenzor::nn;

class PytorchLoaderTest : public ::testing::Test {
protected:
    std::string tmp_dir;

    void SetUp() override {
        tmp_dir = fs::temp_directory_path() / "tenzor_pytorch_loader_test";
        fs::create_directories(tmp_dir);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir);
    }

    std::string createFile(const std::string& name, const std::string& content) {
        auto path = (fs::path(tmp_dir) / name).string();
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(content.data(), content.size());
        return path;
    }
};

// ===========================================================================
// is_pytorch_file tests
// ===========================================================================

TEST_F(PytorchLoaderTest, IsPytorchFileReturnsFalseForNonexistent) {
    EXPECT_FALSE(is_pytorch_file("/nonexistent/path/model.pth"));
}

TEST_F(PytorchLoaderTest, IsPytorchFileReturnsFalseForEmptyFile) {
    auto path = createFile("empty.pth", "");
    EXPECT_FALSE(is_pytorch_file(path));
}

TEST_F(PytorchLoaderTest, IsPytorchFileReturnsFalseForTextFile) {
    auto path = createFile("text.pth", "this is not a pytorch file");
    EXPECT_FALSE(is_pytorch_file(path));
}

TEST_F(PytorchLoaderTest, IsPytorchFileReturnsFalseForShortFile) {
    auto path = createFile("short.pth", "PK");
    EXPECT_FALSE(is_pytorch_file(path));
}

TEST_F(PytorchLoaderTest, IsPytorchFileReturnsTrueForZipMagic) {
    // ZIP magic number: PK\x03\x04
    std::string zip_header = "PK\x03\x04" + std::string(26, '\0');
    auto path = createFile("valid.pth", zip_header);
    EXPECT_TRUE(is_pytorch_file(path));
}

// ===========================================================================
// load_pytorch_state_dict error path tests
// ===========================================================================

TEST_F(PytorchLoaderTest, LoadStateDictThrowsForNonexistent) {
    EXPECT_THROW(load_pytorch_state_dict("/nonexistent/path/model.pth"),
                 std::runtime_error);
}

TEST_F(PytorchLoaderTest, LoadStateDictThrowsForInvalidFile) {
    auto path = createFile("invalid.pth", "not a valid pytorch file at all");
    EXPECT_THROW(load_pytorch_state_dict(path), std::runtime_error);
}

TEST_F(PytorchLoaderTest, LoadStateDictThrowsForEmptyFile) {
    auto path = createFile("empty.pth", "");
    EXPECT_THROW(load_pytorch_state_dict(path), std::runtime_error);
}

TEST_F(PytorchLoaderTest, LoadStateDictThrowsForCorruptZip) {
    // Valid ZIP header but no actual content
    std::string corrupt = "PK\x03\x04" + std::string(26, '\0');
    auto path = createFile("corrupt.pth", corrupt);
    EXPECT_THROW(load_pytorch_state_dict(path), std::runtime_error);
}

// ===========================================================================
// list_pytorch_tensors error path tests
// ===========================================================================

TEST_F(PytorchLoaderTest, ListTensorsThrowsForNonexistent) {
    EXPECT_THROW(list_pytorch_tensors("/nonexistent/path/model.pth"),
                 std::runtime_error);
}

TEST_F(PytorchLoaderTest, ListTensorsThrowsForInvalidFile) {
    auto path = createFile("invalid.pth", "garbage content here");
    EXPECT_THROW(list_pytorch_tensors(path), std::runtime_error);
}

TEST_F(PytorchLoaderTest, ListTensorsThrowsForEmptyFile) {
    auto path = createFile("empty.pth", "");
    EXPECT_THROW(list_pytorch_tensors(path), std::runtime_error);
}
