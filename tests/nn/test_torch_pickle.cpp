/**
 * @file test_torch_pickle.cpp
 * @brief Tests for the native PyTorch .pth checkpoint parser.
 *
 * Audit H2-followup verification. The tests build hand-crafted minimal
 * torch.save archives (ZIP + pickle bytestream + raw tensor storage) and
 * verify the parser reproduces the encoded state_dict end-to-end.
 *
 * We don't depend on a real PyTorch install at test time — the format is
 * stable enough to embed test vectors as byte literals.
 */

#include <gtest/gtest.h>

#include "tenzor/io/torch_pickle.hpp"
#include "tenzor/nn/pytorch_loader.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <unistd.h>  // HH.26: getpid() to disambiguate parallel ctest shards
#include <fstream>
#include <vector>

using namespace tenzor;

namespace {

// =========================================================================
// ZIP archive writer (minimal — STORED entries only, no compression).
// Mirrors what torch.save writes for tensor data; sufficient for tests.
// =========================================================================

struct ZipBuilder {
    struct Entry {
        std::string name;
        std::vector<uint8_t> data;
        uint32_t local_offset = 0;
    };
    std::vector<Entry> entries_;

    void add(const std::string& name, std::vector<uint8_t> data) {
        entries_.push_back({name, std::move(data), 0});
    }

    static void write_u16(std::vector<uint8_t>& out, uint16_t v) {
        out.push_back(v & 0xff);
        out.push_back((v >> 8) & 0xff);
    }
    static void write_u32(std::vector<uint8_t>& out, uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back((v >> (8 * i)) & 0xff);
    }

    auto build() -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        // Local file headers + data
        for (auto& e : entries_) {
            e.local_offset = static_cast<uint32_t>(out.size());
            write_u32(out, 0x04034b50);             // local file header sig
            write_u16(out, 20);                     // version needed
            write_u16(out, 0);                      // flags
            write_u16(out, 0);                      // method = STORED
            write_u16(out, 0);                      // last mod time
            write_u16(out, 0);                      // last mod date
            write_u32(out, 0);                      // crc-32 (not validated in our reader)
            write_u32(out, static_cast<uint32_t>(e.data.size()));  // comp size
            write_u32(out, static_cast<uint32_t>(e.data.size()));  // uncomp size
            write_u16(out, static_cast<uint16_t>(e.name.size()));  // name len
            write_u16(out, 0);                      // extra len
            out.insert(out.end(), e.name.begin(), e.name.end());
            out.insert(out.end(), e.data.begin(), e.data.end());
        }
        // Central directory
        uint32_t cd_start = static_cast<uint32_t>(out.size());
        for (auto& e : entries_) {
            write_u32(out, 0x02014b50);             // CD header sig
            write_u16(out, 20);                     // version made by
            write_u16(out, 20);                     // version needed
            write_u16(out, 0);                      // flags
            write_u16(out, 0);                      // method
            write_u16(out, 0);                      // mod time
            write_u16(out, 0);                      // mod date
            write_u32(out, 0);                      // crc
            write_u32(out, static_cast<uint32_t>(e.data.size()));
            write_u32(out, static_cast<uint32_t>(e.data.size()));
            write_u16(out, static_cast<uint16_t>(e.name.size()));
            write_u16(out, 0);                      // extra
            write_u16(out, 0);                      // comment
            write_u16(out, 0);                      // disk number start
            write_u16(out, 0);                      // internal attrs
            write_u32(out, 0);                      // external attrs
            write_u32(out, e.local_offset);
            out.insert(out.end(), e.name.begin(), e.name.end());
        }
        uint32_t cd_size = static_cast<uint32_t>(out.size()) - cd_start;
        // End of central directory
        write_u32(out, 0x06054b50);
        write_u16(out, 0);                          // disk #
        write_u16(out, 0);                          // disk CD start
        write_u16(out, static_cast<uint16_t>(entries_.size()));
        write_u16(out, static_cast<uint16_t>(entries_.size()));
        write_u32(out, cd_size);
        write_u32(out, cd_start);
        write_u16(out, 0);                          // comment len
        return out;
    }
};

// =========================================================================
// Pickle stream builder for `state_dict = {"weight": <tensor>}`-style
// dicts. We emit a single-tensor pickle that exercises every opcode the
// parser needs to support (PROTO, EMPTY_DICT, MARK, SHORT_BINUNICODE,
// BININT1, BININT, TUPLE2, TUPLE3, TUPLE, BINPUT, BINGET, EMPTY_TUPLE,
// STACK_GLOBAL, REDUCE, BINPERSID, MEMOIZE, SETITEMS, STOP).
// =========================================================================

class PickleBuilder {
public:
    void proto(uint8_t v) { put(0x80); put(v); }
    void frame_skip() {}  // we don't emit FRAME; reader handles either way

    void short_binunicode(const std::string& s) {
        put(0x8c); put(static_cast<uint8_t>(s.size()));
        for (char c : s) put(static_cast<uint8_t>(c));
    }
    void binint1(uint8_t v) { put('K'); put(v); }
    void binint(int32_t v) {
        put('J');
        for (int i = 0; i < 4; ++i) put((v >> (8 * i)) & 0xff);
    }
    void mark()         { put('('); }
    void empty_dict()   { put('}'); }
    void empty_tuple()  { put(')'); }
    void tuple()        { put('t'); }
    void tuple1()       { put(0x85); }
    void tuple2()       { put(0x86); }
    void tuple3()       { put(0x87); }
    void memoize()      { put(0x94); }
    void long_binput(uint32_t k) {
        put('r');
        for (int i = 0; i < 4; ++i) put((k >> (8 * i)) & 0xff);
    }
    void long_binget(uint32_t k) {
        put('j');
        for (int i = 0; i < 4; ++i) put((k >> (8 * i)) & 0xff);
    }
    void stack_global() { put(0x93); }
    void reduce()       { put('R'); }
    void binpersid()    { put('Q'); }
    void setitems()     { put('u'); }
    void stop()         { put('.'); }
    void newfalse()     { put(0x89); }
    void newtrue()      { put(0x88); }

    auto data() && -> std::vector<uint8_t> { return std::move(bytes_); }
    auto bytes() const -> const std::vector<uint8_t>& { return bytes_; }

private:
    void put(uint8_t b) { bytes_.push_back(b); }
    std::vector<uint8_t> bytes_;
};

// Helper: build a minimal torch.save-style archive containing a single
// Float32 tensor of shape [N] called "weight" whose raw bytes are
// `tensor_bytes`. Returns the path of the temporary archive on disk.
auto make_minimal_pth(const std::string& path,
                       const std::vector<float>& values) {
    // 1. Build the pickle stream that constructs
    //    { "weight": _rebuild_tensor_v2(<storage 0>, 0, (N,), (1,), False, OrderedDict()) }
    //    in the OrderedDict-wrapped state_dict style torch.save emits.
    //
    //    Stack trace (line by line):
    //      PROTO 2
    //      EMPTY_DICT          [ {} ]
    //      MEMOIZE             [ {} ]
    //      MARK
    //        SHORT_BINUNICODE "weight"
    //        MEMOIZE
    //        STACK_GLOBAL "torch._utils", "_rebuild_tensor_v2"
    //        MEMOIZE
    //        EMPTY_TUPLE                       <- args tuple start (we'll build manually)
    //        MARK
    //          (storage persistent id)
    //          BININT 0 (offset)
    //          TUPLE1 (size = (N,))    -- actually we use MARK+TUPLE
    //          TUPLE1 (stride = (1,))
    //          NEWFALSE
    //          ... etc
    //
    //    The simpler approach for this test: build the arg list with MARK+TUPLE.

    PickleBuilder p;
    p.proto(2);
    p.empty_dict();
    p.memoize();          // memo[0] = {}
    p.mark();

    // ---- Key "weight" ----
    p.short_binunicode("weight");
    p.memoize();

    // ---- Value: _rebuild_tensor_v2(storage_persid, 0, (N,), (1,), False, OrderedDict()) ----
    // GLOBAL torch._utils._rebuild_tensor_v2
    p.short_binunicode("torch._utils"); p.memoize();
    p.short_binunicode("_rebuild_tensor_v2"); p.memoize();
    p.stack_global();
    p.memoize();

    // Build args tuple: (storage_persid, 0, (N,), (1,), False, OrderedDict())
    p.mark();

    // -- storage persistent ID --
    // Persistent ID payload = ('storage', GLOBAL torch.FloatStorage, 'data/0', 'cpu', N)
    p.mark();
    p.short_binunicode("storage"); p.memoize();
    p.short_binunicode("torch"); p.memoize();
    p.short_binunicode("FloatStorage"); p.memoize();
    p.stack_global(); p.memoize();
    p.short_binunicode("data/0"); p.memoize();
    p.short_binunicode("cpu"); p.memoize();
    p.binint(static_cast<int32_t>(values.size()));
    p.tuple();
    p.binpersid();

    // -- storage_offset = 0 --
    p.binint(0);

    // -- size = (N,) --
    p.mark();
    p.binint(static_cast<int32_t>(values.size()));
    p.tuple();

    // -- stride = (1,) --
    p.mark();
    p.binint(1);
    p.tuple();

    // -- requires_grad = False --
    p.newfalse();

    // -- backward_hooks = OrderedDict() --
    // GLOBAL collections.OrderedDict + EMPTY_TUPLE + REDUCE
    p.short_binunicode("collections"); p.memoize();
    p.short_binunicode("OrderedDict"); p.memoize();
    p.stack_global(); p.memoize();
    p.empty_tuple();
    p.reduce();
    p.memoize();

    p.tuple();              // close the 6-arg tuple
    p.reduce();             // _rebuild_tensor_v2(*args) -> Tensor
    p.memoize();

    // SETITEMS to populate the dict, then STOP.
    p.setitems();
    p.stop();

    // 2. Build the data/0 entry: raw f32 bytes of `values`.
    std::vector<uint8_t> data0(values.size() * sizeof(float));
    std::memcpy(data0.data(), values.data(), data0.size());

    // 3. Assemble the ZIP: top-level `archive/` prefix simulates torch.save layout.
    ZipBuilder zip;
    zip.add("archive/data.pkl", std::move(p).data());
    zip.add("archive/data/0", std::move(data0));
    auto bytes = zip.build();

    // 4. Write to disk.
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

class TorchPickleEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const g_env = ::testing::AddGlobalTestEnvironment(new TorchPickleEnv);

class TorchPickleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // HH.26: include pid so parallel ctest shards (sharing the gtest
        // random seed) don't collide on the same temp file.
        test_path_ = (std::filesystem::temp_directory_path() /
                      ("tenzor_test_torch_pickle_" +
                       std::to_string(::getpid()) + "_" +
                       std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                       ".pth")).string();
    }
    void TearDown() override {
        std::filesystem::remove(test_path_);
    }
    std::string test_path_;
};

}  // namespace

TEST_F(TorchPickleTest, LoadsSingleFloat32Tensor) {
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    make_minimal_pth(test_path_, v);

    auto state = tenzor::io::load_torch_pickle(test_path_);
    ASSERT_EQ(state.size(), 1u);
    ASSERT_TRUE(state.count("weight"));

    const auto& t = state.at("weight");
    EXPECT_EQ(t.dtype(), DType::Float32);
    ASSERT_EQ(t.shape().size(), 1u);
    EXPECT_EQ(t.shape()[0], 5);

    const float* p = t.data<float>();
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_FLOAT_EQ(p[i], v[i]);
    }
}

TEST_F(TorchPickleTest, ErrorMessageForMissingFile) {
    EXPECT_THROW(tenzor::io::load_torch_pickle("/no/such/file.pth"),
                 std::runtime_error);
}

TEST_F(TorchPickleTest, ErrorForNonZipInput) {
    // Write a tiny text file that isn't a ZIP.
    std::ofstream f(test_path_, std::ios::binary);
    f << "not a zip";
    f.close();
    EXPECT_THROW(tenzor::io::load_torch_pickle(test_path_),
                 std::runtime_error);
}

TEST_F(TorchPickleTest, LoadsMultipleTensors) {
    // Verify the parser handles a state_dict with multiple keys + tensors
    // (the common case for real model checkpoints). We construct a minimal
    // dict with two Float32 tensors of different sizes referencing
    // different `data/N` storages.

    // Tensor 1: shape [4] called "fc.weight", values [10, 20, 30, 40].
    // Tensor 2: shape [2] called "fc.bias",   values [100, 200].

    PickleBuilder p;
    p.proto(2);
    p.empty_dict();
    p.memoize();
    p.mark();

    // ---- Entry 1: "fc.weight" → _rebuild_tensor_v2(...) ----
    p.short_binunicode("fc.weight"); p.memoize();
    p.short_binunicode("torch._utils"); p.memoize();
    p.short_binunicode("_rebuild_tensor_v2"); p.memoize();
    p.stack_global(); p.memoize();
    p.mark();
    // storage persistent ID
    p.mark();
    p.short_binunicode("storage"); p.memoize();
    p.short_binunicode("torch"); p.memoize();
    p.short_binunicode("FloatStorage"); p.memoize();
    p.stack_global(); p.memoize();
    p.short_binunicode("data/0"); p.memoize();
    p.short_binunicode("cpu"); p.memoize();
    p.binint(4);
    p.tuple();
    p.binpersid();
    p.binint(0);                    // offset
    p.mark(); p.binint(4); p.tuple();  // size = (4,)
    p.mark(); p.binint(1); p.tuple();  // stride = (1,)
    p.newfalse();
    p.short_binunicode("collections"); p.memoize();
    p.short_binunicode("OrderedDict"); p.memoize();
    p.stack_global(); p.memoize();
    p.empty_tuple();
    p.reduce();
    p.memoize();
    p.tuple();
    p.reduce();
    p.memoize();

    // ---- Entry 2: "fc.bias" → _rebuild_tensor_v2(...) ----
    p.short_binunicode("fc.bias"); p.memoize();
    p.short_binunicode("torch._utils"); p.memoize();
    p.short_binunicode("_rebuild_tensor_v2"); p.memoize();
    p.stack_global(); p.memoize();
    p.mark();
    p.mark();
    p.short_binunicode("storage"); p.memoize();
    p.short_binunicode("torch"); p.memoize();
    p.short_binunicode("FloatStorage"); p.memoize();
    p.stack_global(); p.memoize();
    p.short_binunicode("data/1"); p.memoize();
    p.short_binunicode("cpu"); p.memoize();
    p.binint(2);
    p.tuple();
    p.binpersid();
    p.binint(0);
    p.mark(); p.binint(2); p.tuple();
    p.mark(); p.binint(1); p.tuple();
    p.newfalse();
    p.short_binunicode("collections"); p.memoize();
    p.short_binunicode("OrderedDict"); p.memoize();
    p.stack_global(); p.memoize();
    p.empty_tuple();
    p.reduce();
    p.memoize();
    p.tuple();
    p.reduce();
    p.memoize();

    p.setitems();
    p.stop();

    std::vector<float> v0 = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> v1 = {100.0f, 200.0f};
    std::vector<uint8_t> d0(v0.size() * sizeof(float));
    std::vector<uint8_t> d1(v1.size() * sizeof(float));
    std::memcpy(d0.data(), v0.data(), d0.size());
    std::memcpy(d1.data(), v1.data(), d1.size());

    ZipBuilder zip;
    zip.add("archive/data.pkl", std::move(p).data());
    zip.add("archive/data/0", std::move(d0));
    zip.add("archive/data/1", std::move(d1));
    auto bytes = zip.build();
    std::ofstream f(test_path_, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    f.close();

    auto state = tenzor::io::load_torch_pickle(test_path_);
    ASSERT_EQ(state.size(), 2u);
    ASSERT_TRUE(state.count("fc.weight"));
    ASSERT_TRUE(state.count("fc.bias"));

    const auto& w = state.at("fc.weight");
    EXPECT_EQ(w.shape().size(), 1u);
    EXPECT_EQ(w.shape()[0], 4);
    for (size_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(w.data<float>()[i], v0[i]);

    const auto& b = state.at("fc.bias");
    EXPECT_EQ(b.shape().size(), 1u);
    EXPECT_EQ(b.shape()[0], 2);
    for (size_t i = 0; i < 2; ++i) EXPECT_FLOAT_EQ(b.data<float>()[i], v1[i]);
}

// Audit C.8: unsupported pickle opcodes must throw, not silently skip.
// Previously the default branch of the opcode switch did `break;`, so a
// pickle blob using an opcode the parser didn't recognise would parse to
// a garbage state dict.  The parser now surfaces unknown opcodes with the
// byte and offset.
TEST_F(TorchPickleTest, ErrorOnUnsupportedOpcode) {
    // Build a minimal pickle stream that starts with PROTO 2 / EMPTY_DICT
    // and then hits an undefined opcode byte (0xFE).  Wrap in the minimal
    // zip layout the loader requires.
    std::vector<uint8_t> pkl = {
        0x80, 0x02,          // PROTO 2
        '}',                 // EMPTY_DICT
        0xFE,                // unsupported opcode
        '.'                  // STOP (never reached)
    };

    ZipBuilder zip;
    zip.add("archive/data.pkl", pkl);
    auto data = zip.build();

    auto path = std::filesystem::temp_directory_path() /
                "tenzor_pickle_unsupported_opcode.pth";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }

    try {
        (void) tenzor::nn::load_pytorch_state_dict(path.string());
        FAIL() << "expected load_pytorch_state_dict to throw on unsupported opcode";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("unsupported opcode"), std::string::npos)
            << "actual error: " << msg;
        EXPECT_NE(msg.find("0xFE"), std::string::npos)
            << "actual error: " << msg;
    }

    std::filesystem::remove(path);
}

// SECURITY: LONG4 (0x8b) reads a 32-bit length and must validate it against the
// bytes remaining BEFORE allocating the buffer; otherwise a 5-byte fragment
// declaring a ~4 GB length forces a giant allocation (OOM/bad_alloc DoS). The
// parser must reject it as a clean "read past EOF", not crash or OOM.
TEST_F(TorchPickleTest, Long4LengthPastEofRejectedBeforeAllocating) {
    std::vector<uint8_t> pkl = {
        0x80, 0x02,                      // PROTO 2
        '}',                             // EMPTY_DICT
        0x8b, 0xFF, 0xFF, 0xFF, 0xFF,    // LONG4 with length 0xFFFFFFFF, no data
        '.'                              // STOP (never reached)
    };

    ZipBuilder zip;
    zip.add("archive/data.pkl", pkl);
    auto data = zip.build();

    auto path = std::filesystem::temp_directory_path() /
                ("tenzor_pickle_long4_oom_" + std::to_string(::getpid()) + ".pth");
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }

    EXPECT_THROW((void) tenzor::io::load_torch_pickle(path.string()),
                 std::runtime_error)
        << "LONG4 with an out-of-range length must be rejected, not allocated";

    std::filesystem::remove(path);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
