/**
 * @file test_dataset_hardening.cpp
 * @brief Security/robustness regressions for the dataset parsers.
 *
 * These cover the untrusted-file hardening fixes:
 *   - MNIST IDX header: a truncated file must throw cleanly (the BE-u32 reader
 *     used to assemble a value from an uninitialized stack buffer on a short
 *     read instead of rejecting it).
 *   - ImageFolder: symlinks must be skipped so an attacker-controlled dataset
 *     tarball cannot redirect traversal/decoding outside the dataset root.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/data/datasets/mnist.hpp>
#include <tenzor/data/datasets/imagenet.hpp>
#include <tenzor/io/image.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace tenzor;

namespace {
std::string make_temp_dir(const std::string& tag) {
    auto dir = fs::temp_directory_path() /
               ("tenzor_ds_" + tag + "_" + std::to_string(::getpid()) + "_" +
                std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::create_directories(dir);
    return dir.string();
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& b) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
}
}  // namespace

// A truncated IDX file (fewer than the 4 header bytes) must throw, not read
// uninitialized stack memory.
TEST(DatasetHardening, MnistTruncatedHeaderThrows) {
    std::string dir = make_temp_dir("mnist_trunc");
    // train-images / train-labels: write only 2 bytes (header read needs 4).
    write_bytes(dir + "/train-images-idx3-ubyte", {0x00, 0x00});
    write_bytes(dir + "/train-labels-idx1-ubyte", {0x00, 0x00});

    EXPECT_THROW(data::datasets::MNIST(dir, /*train=*/true, /*normalize=*/false),
                 std::runtime_error)
        << "MNIST must reject a truncated IDX header instead of trusting a "
           "partially-read magic value";

    fs::remove_all(dir);
}

// ImageFolder must not follow a symlinked file (which resolves through to a
// regular file) out of the dataset tree.
TEST(DatasetHardening, ImageFolderSkipsSymlinkedFiles) {
    std::string root = make_temp_dir("imgfolder");
    // Create one real class dir with one real image and one symlink pointing
    // OUTSIDE the root to a regular (non-image-irrelevant) file.
    fs::create_directories(fs::path(root) / "class_a");

    // A genuine small PNG inside the class dir.
    auto img = zeros({3, 8, 8}, DType::UInt8, Device::cpu());
    auto png = io::encode_png(img);
    write_bytes((fs::path(root) / "class_a" / "real.png").string(),
                std::vector<uint8_t>(png.begin(), png.end()));

    // An out-of-tree target file, and a symlink to it with an image extension.
    std::string outside = make_temp_dir("imgfolder_outside");
    write_bytes(outside + "/secret.png", std::vector<uint8_t>(64, 0x00));
    std::error_code ec;
    fs::create_symlink(outside + "/secret.png",
                       fs::path(root) / "class_a" / "link.png", ec);

    if (ec) {
        GTEST_SKIP() << "symlink creation unsupported on this filesystem";
    }

    data::datasets::ImageFolder folder(root, /*image_size=*/8, {".png"});
    // Only the real image must be collected; the symlink must be skipped.
    EXPECT_EQ(folder.size(), 1u)
        << "ImageFolder followed a symlink out of the dataset root";

    fs::remove_all(root);
    fs::remove_all(outside);
}

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
    return RUN_ALL_TESTS();
}
