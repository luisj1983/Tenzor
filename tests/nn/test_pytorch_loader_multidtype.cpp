/**
 * @file test_pytorch_loader_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for PyTorch loader
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/pytorch_loader.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class PytorchLoaderMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::string tmp_dir;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        tmp_dir = (fs::temp_directory_path() / "tenzor_pytorch_loader_multidtype").string();
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

TEST_P(PytorchLoaderMultiDTypeTest, IsPytorchFileReturnsFalseForNonexistent) {
    EXPECT_FALSE(is_pytorch_file("/nonexistent/path/model.pth"));
}

TEST_P(PytorchLoaderMultiDTypeTest, IsPytorchFileReturnsFalseForEmptyFile) {
    auto path = createFile("empty.pth", "");
    EXPECT_FALSE(is_pytorch_file(path));
}

TEST_P(PytorchLoaderMultiDTypeTest, IsPytorchFileReturnsTrueForZipMagic) {
    std::string zip_header = "PK\x03\x04" + std::string(26, '\0');
    auto path = createFile("valid.pth", zip_header);
    EXPECT_TRUE(is_pytorch_file(path));
}

TEST_P(PytorchLoaderMultiDTypeTest, LoadStateDictThrowsForNonexistent) {
    EXPECT_THROW(load_pytorch_state_dict("/nonexistent/path/model.pth"),
                 std::runtime_error);
}

TEST_P(PytorchLoaderMultiDTypeTest, LoadStateDictThrowsForInvalidFile) {
    auto path = createFile("invalid.pth", "not a valid pytorch file at all");
    EXPECT_THROW(load_pytorch_state_dict(path), std::runtime_error);
}

TEST_P(PytorchLoaderMultiDTypeTest, ListTensorsThrowsForNonexistent) {
    EXPECT_THROW(list_pytorch_tensors("/nonexistent/path/model.pth"),
                 std::runtime_error);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(PytorchLoaderMultiDTypeTest);
