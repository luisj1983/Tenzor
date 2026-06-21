/**
 * @file test_tensorboard.cpp
 * @brief Unit tests for TensorBoard integration
 */

#include <gtest/gtest.h>
#include <tenzor/utils/tensorboard.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/tenzor.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>  // getpid — audit-5 Y.33

using namespace tenzor;

namespace {

// Locate the single events.out.tfevents file inside a log dir.
std::string find_event_file(const std::string& dir) {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().filename().string().find("events.out.tfevents") !=
            std::string::npos) {
            return entry.path().string();
        }
    }
    return {};
}

// Read an entire (binary) file into a byte buffer.
std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}

// Size of the (single) event file in a log dir, or 0 if none yet.
std::uintmax_t event_file_size(const std::string& dir) {
    const std::string p = find_event_file(dir);
    if (p.empty()) return 0;
    return std::filesystem::file_size(p);
}

// True if `needle` appears as a contiguous byte subsequence of `hay`.
bool contains_bytes(const std::vector<uint8_t>& hay,
                    const std::vector<uint8_t>& needle) {
    if (needle.empty() || needle.size() > hay.size()) return false;
    for (size_t i = 0; i + needle.size() <= hay.size(); ++i) {
        if (std::memcmp(hay.data() + i, needle.data(), needle.size()) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

// Global test environment for initialization
class TensorBoardTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tb_env =
    ::testing::AddGlobalTestEnvironment(new TensorBoardTestEnvironment);

class TensorBoardTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Audit-5 Y.33: per-test, per-process temp dir so parallel re-runs
        // and CI matrix shards don't race on a shared "/tmp/tenzor_tb_test"
        // directory. Mirrors tests/nn/test_safetensors.cpp.
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string test_name = info ? info->name() : "unknown";
        test_log_dir_ = (std::filesystem::temp_directory_path() /
                         ("tenzor_tensorboard_" + std::to_string(::getpid()) +
                          "_" + test_name)).string();

        // Clean up test log directory if it exists
        if (std::filesystem::exists(test_log_dir_)) {
            std::filesystem::remove_all(test_log_dir_);
        }
    }

    void TearDown() override {
        // Clean up test log directory
        if (std::filesystem::exists(test_log_dir_)) {
            std::filesystem::remove_all(test_log_dir_);
        }
    }

    std::string test_log_dir_;
};

// Test 1: Constructor creates directory
TEST_F(TensorBoardTest, ConstructorCreatesDirectory) {
    {
        SummaryWriter writer(test_log_dir_);
        EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
        EXPECT_TRUE(writer.is_open());
    }
}

// Test 2: Event file is created
TEST_F(TensorBoardTest, EventFileCreated) {
    {
        SummaryWriter writer(test_log_dir_);
    }  // Close writer

    // Check for event file
    bool found_event_file = false;
    for (const auto& entry : std::filesystem::directory_iterator(test_log_dir_)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("events.out.tfevents") != std::string::npos) {
            found_event_file = true;
            break;
        }
    }

    EXPECT_TRUE(found_event_file);
}

// Test 3: Add scalar writes the TAG and VALUE to the event file.
//
// The production encoder writes the tag as a length-delimited protobuf string
// (so the ASCII tag bytes appear verbatim) and the scalar as a `simple_value`
// field: the field tag byte 0x15 (field 2, wire-type 5 / fixed32) followed by
// the 4 little-endian IEEE-754 bytes of the float. We verify BOTH round-trip,
// so a writer that emitted the wrong tag, the wrong value, or unrelated bytes
// would fail — unlike the old size>0 check.
TEST_F(TensorBoardTest, AddScalar) {
    const std::string tag = "loss_unique_tag";
    const float v0 = 0.5f, v1 = 0.4f, v2 = 0.3f;
    {
        SummaryWriter writer(test_log_dir_);
        writer.add_scalar(tag, v0, 0);
        writer.add_scalar(tag, v1, 1);
        writer.add_scalar(tag, v2, 2);
        writer.flush();
    }

    const std::string event_path = find_event_file(test_log_dir_);
    ASSERT_FALSE(event_path.empty()) << "no event file was created";
    const auto bytes = read_all_bytes(event_path);
    ASSERT_GT(bytes.size(), 0u);

    // Tag must be serialized literally.
    const std::vector<uint8_t> tag_bytes(tag.begin(), tag.end());
    EXPECT_TRUE(contains_bytes(bytes, tag_bytes))
        << "scalar tag was not serialized into the event file";

    // Each scalar value must appear as simple_value: 0x15 + float LE bytes.
    auto simple_value_seq = [](float v) {
        std::vector<uint8_t> seq{0x15};
        uint8_t fb[4];
        std::memcpy(fb, &v, 4);  // host is little-endian on supported targets
        seq.insert(seq.end(), fb, fb + 4);
        return seq;
    };
    EXPECT_TRUE(contains_bytes(bytes, simple_value_seq(v0)))
        << "scalar value 0.5 not found as simple_value in event file";
    EXPECT_TRUE(contains_bytes(bytes, simple_value_seq(v1)))
        << "scalar value 0.4 not found as simple_value in event file";
    EXPECT_TRUE(contains_bytes(bytes, simple_value_seq(v2)))
        << "scalar value 0.3 not found as simple_value in event file";

    // A value never written must NOT be present (guards against a writer that
    // dumps arbitrary/constant bytes that happen to contain our values).
    EXPECT_FALSE(contains_bytes(bytes, simple_value_seq(123.456f)))
        << "a value that was never logged appeared in the event file";
}

// Test 4: Add histogram
TEST_F(TensorBoardTest, AddHistogram) {
    {
        SummaryWriter writer(test_log_dir_);

        // Create tensor with some data
        Tensor tensor({100}, DType::Float32, Device::cpu());
        float* data = tensor.data<float>();
        for (int i = 0; i < 100; ++i) {
            data[i] = static_cast<float>(i) / 100.0f;
        }

        // File already holds the version event from construction; capture its
        // size, then assert the histogram event strictly grows the file.
        writer.flush();
        const auto before = event_file_size(test_log_dir_);
        ASSERT_GT(before, 0u) << "version event missing before add_histogram";

        writer.add_histogram("weights", tensor, 0);
        writer.flush();
        const auto after = event_file_size(test_log_dir_);
        EXPECT_GT(after, before)
            << "add_histogram wrote nothing (no-op): size unchanged at " << before;
    }
}

// Test 5: Add image grayscale
TEST_F(TensorBoardTest, AddImageGrayscale) {
    {
        SummaryWriter writer(test_log_dir_);

        // Create grayscale image [1, 28, 28]
        Tensor img({1, 28, 28}, DType::Float32, Device::cpu());
        float* data = img.data<float>();
        for (int i = 0; i < 28*28; ++i) {
            data[i] = static_cast<float>(i % 256) / 255.0f;
        }

        // add_image must actually write bytes — the dir already exists from the
        // constructor, so the old exists() check passed even if add_image was a
        // complete no-op. Compare event-file size before/after instead.
        writer.flush();
        const auto before = event_file_size(test_log_dir_);
        ASSERT_GT(before, 0u);

        writer.add_image("mnist/sample", img, 0);
        writer.flush();
        const auto after = event_file_size(test_log_dir_);
        EXPECT_GT(after, before)
            << "add_image (grayscale) wrote nothing: size unchanged at " << before;
    }
}

// Test 6: Add image RGB
TEST_F(TensorBoardTest, AddImageRGB) {
    {
        SummaryWriter writer(test_log_dir_);

        // Create RGB image [3, 64, 64]
        Tensor img({3, 64, 64}, DType::Float32, Device::cpu());
        float* data = img.data<float>();
        for (int i = 0; i < 3*64*64; ++i) {
            data[i] = 0.5f;
        }

        writer.flush();
        const auto before = event_file_size(test_log_dir_);
        ASSERT_GT(before, 0u);

        writer.add_image("generated/sample", img, 0);
        writer.flush();
        const auto after = event_file_size(test_log_dir_);
        EXPECT_GT(after, before)
            << "add_image (RGB) wrote nothing: size unchanged at " << before;
    }
}

// Test 7: Add graph
TEST_F(TensorBoardTest, AddGraph) {
    {
        SummaryWriter writer(test_log_dir_);

        // Build a small autograd graph: y = x @ w + b
        Variable x(tenzor::randn({1, 3}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
        Variable w(tenzor::randn({3, 4}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
        Variable b(tenzor::randn({1, 4}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
        auto y = x.matmul(w) + b;

        writer.flush();
        const auto before = event_file_size(test_log_dir_);
        ASSERT_GT(before, 0u);

        writer.add_graph("ToyModel", y);
        writer.flush();
        const auto after = event_file_size(test_log_dir_);
        EXPECT_GT(after, before)
            << "add_graph wrote nothing (no-op): size unchanged at " << before;
    }
}

// Test 8: is_open status
TEST_F(TensorBoardTest, IsOpenStatus) {
    SummaryWriter writer(test_log_dir_);

    EXPECT_TRUE(writer.is_open());

    writer.close();

    EXPECT_FALSE(writer.is_open());
}

// Test 9: Close flushes data
TEST_F(TensorBoardTest, CloseFlushes) {
    {
        SummaryWriter writer(test_log_dir_);
        writer.add_scalar("test", 1.0f, 0);
        // Don't explicitly flush, close should do it
    }  // Destructor calls close

    // Verify file has content
    bool has_content = false;
    for (const auto& entry : std::filesystem::directory_iterator(test_log_dir_)) {
        if (entry.path().filename().string().find("events.out.tfevents") != std::string::npos) {
            has_content = std::filesystem::file_size(entry.path()) > 0;
        }
    }
    EXPECT_TRUE(has_content);
}

// Test 10: Multiple scalars same tag
TEST_F(TensorBoardTest, MultipleScalarsSameTag) {
    {
        SummaryWriter writer(test_log_dir_);

        for (int i = 0; i < 10; ++i) {
            writer.add_scalar("training/loss", static_cast<float>(10 - i) / 10.0f, i);
        }

        writer.flush();
    }

    EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
}

// Test 11: Multiple tags
TEST_F(TensorBoardTest, MultipleTags) {
    {
        SummaryWriter writer(test_log_dir_);

        writer.add_scalar("train/loss", 0.5f, 0);
        writer.add_scalar("train/accuracy", 0.85f, 0);
        writer.add_scalar("val/loss", 0.6f, 0);
        writer.add_scalar("val/accuracy", 0.80f, 0);

        writer.flush();
    }

    EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
}

// Test 12: Custom queue and flush settings
TEST_F(TensorBoardTest, CustomQueueFlush) {
    {
        SummaryWriter writer(test_log_dir_, 5, 60);  // max_queue=5, flush_secs=60

        // Add 10 scalars (should trigger 2 auto-flushes at 5 events each)
        for (int i = 0; i < 10; ++i) {
            writer.add_scalar("counter", static_cast<float>(i), i);
        }
    }

    EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
}

// Test 13: Exception on closed writer
TEST_F(TensorBoardTest, ExceptionOnClosed) {
    SummaryWriter writer(test_log_dir_);
    writer.close();

    EXPECT_THROW(
        writer.add_scalar("test", 1.0f, 0),
        TensorBoardException
    );
}

// Test 14: Invalid image shape throws
TEST_F(TensorBoardTest, InvalidImageShape) {
    SummaryWriter writer(test_log_dir_);

    // Wrong number of dimensions
    Tensor bad_img({64, 64}, DType::Float32, Device::cpu());

    EXPECT_THROW(
        writer.add_image("bad", bad_img, 0),
        TensorBoardException
    );
}

// Test 15: Invalid image channels throws
TEST_F(TensorBoardTest, InvalidImageChannels) {
    SummaryWriter writer(test_log_dir_);

    // Invalid number of channels (must be 1, 3, or 4)
    Tensor bad_img({5, 64, 64}, DType::Float32, Device::cpu());

    EXPECT_THROW(
        writer.add_image("bad", bad_img, 0),
        TensorBoardException
    );
}

// Test 16: Histogram with different bin sizes
TEST_F(TensorBoardTest, HistogramDifferentBins) {
    {
        SummaryWriter writer(test_log_dir_);

        Tensor tensor({100}, DType::Float32, Device::cpu());
        float* data = tensor.data<float>();
        for (int i = 0; i < 100; ++i) {
            data[i] = static_cast<float>(i);
        }

        writer.add_histogram("weights/bins10", tensor, 0, 10);
        writer.add_histogram("weights/bins30", tensor, 1, 30);
        writer.add_histogram("weights/bins50", tensor, 2, 50);

        writer.flush();
    }

    EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
}

// Test 17: Destructor closes writer
TEST_F(TensorBoardTest, DestructorCloses) {
    {
        SummaryWriter writer(test_log_dir_);
        EXPECT_TRUE(writer.is_open());
        // Writer goes out of scope, destructor should close it
    }

    // Verify file was created (destructor flushed)
    EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
}
