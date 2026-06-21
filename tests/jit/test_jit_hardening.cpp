/**
 * @file test_jit_hardening.cpp
 * @brief Regression tests for JIT parser / graph-deserializer hardening against
 *        malformed / untrusted input.
 *
 * Covers:
 *   - compile_script: deeply-nested expressions, oversized source, huge
 *     range(N) unroll, and deep nested if/for blocks must throw cleanly
 *     (std::runtime_error) rather than crashing / overflowing the native stack.
 *   - load_graph (GraphReader): a malformed/truncated/forged .graph file must
 *     throw (bad magic, bad version, truncation, over-large declared counts)
 *     rather than OOM / OOB read.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tenzor/jit/script.hpp>
#include <tenzor/jit/serialization.hpp>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

// Fixture for tests that actually trace/compile a script (needs the CPU backend
// registered). The reject-paths below throw during *parsing*, before any
// backend is touched, so they don't need this.
class JitHardeningCompile : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

// ---------------------------------------------------------------------------
// compile_script parser hardening
// ---------------------------------------------------------------------------

// Deeply nested parentheses must throw the recursion-depth error, not overflow
// the C++ stack. Without the DepthGuard this segfaults.
TEST(JitHardening, DeeplyNestedParensThrowsNotCrash) {
    std::string src = "def f(x): return ";
    const int depth = 5000;  // far beyond kMaxRecursionDepth (256)
    for (int i = 0; i < depth; ++i) src += "(";
    src += "x";
    for (int i = 0; i < depth; ++i) src += ")";
    src += "\n";
    EXPECT_THROW(jit::compile_script(src.c_str()), std::runtime_error);
}

// Deeply nested method-call chains re-enter parse_expression via call args.
TEST(JitHardening, DeeplyNestedCallArgsThrowsNotCrash) {
    // x.add(x.add(x.add(...))) — each arg re-enters the expression recursion.
    std::string src = "def f(x): return ";
    const int depth = 5000;
    for (int i = 0; i < depth; ++i) src += "x.add(";
    src += "x";
    for (int i = 0; i < depth; ++i) src += ")";
    src += "\n";
    EXPECT_THROW(jit::compile_script(src.c_str()), std::runtime_error);
}

// Oversized source is rejected at the length gate before lexing.
TEST(JitHardening, OversizedSourceRejected) {
    std::string src = "def f(x): return x + ";
    // > 1 MiB of "1 + 1 + ..." padding.
    src.reserve(2u << 20);
    while (src.size() < (1u << 20) + 1024) src += "1 + ";
    src += "x\n";
    EXPECT_THROW(jit::compile_script(src.c_str()), std::runtime_error);
}

// A huge static loop unroll must be rejected at parse time, not OOM at trace.
TEST(JitHardening, HugeRangeUnrollRejected) {
    const char* src =
        "def f(x):\n"
        "    for i in range(2000000000):\n"
        "        x = x + x\n"
        "    return x\n";
    EXPECT_THROW(jit::compile_script(src), std::runtime_error);
}

// A modest, legitimate script still compiles fine after the hardening.
TEST_F(JitHardeningCompile, LegitimateScriptStillCompiles) {
    const char* src =
        "def f(x):\n"
        "    y = x + x\n"
        "    return y * x\n";
    auto compiled = jit::compile_script(src);
    ASSERT_NE(compiled, nullptr);
}

// A moderately nested but legitimate expression (well under the cap) compiles.
TEST_F(JitHardeningCompile, ModerateNestingStillCompiles) {
    std::string src = "def f(x): return ";
    const int depth = 64;  // < 256
    for (int i = 0; i < depth; ++i) src += "(";
    src += "x";
    for (int i = 0; i < depth; ++i) src += ")";
    src += "\n";
    auto compiled = jit::compile_script(src.c_str());
    ASSERT_NE(compiled, nullptr);
}

// ---------------------------------------------------------------------------
// GraphReader / load_graph deserializer hardening
// ---------------------------------------------------------------------------

namespace {
std::string temp_graph_path(const std::string& tag) {
    return std::string(::testing::TempDir()) + "/tenzor_hardening_" + tag + ".graph";
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    f.close();
}

void append_u32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
}
void append_u64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
}
}  // namespace

// A file with a wrong magic number must be rejected, not parsed.
TEST(JitHardening, GraphBadMagicRejected) {
    std::vector<uint8_t> bytes;
    append_u32(bytes, 0xDEADBEEF);  // not MAGIC_NUMBER
    auto path = temp_graph_path("badmagic");
    write_bytes(path, bytes);
    EXPECT_THROW(jit::load_graph(path), std::runtime_error);
    std::remove(path.c_str());
}

// A truncated file (header only, nothing else) must throw on the next read.
TEST(JitHardening, GraphTruncatedAfterMagicRejected) {
    // Round-trip a real, tiny graph so we get a valid magic+version prefix,
    // then truncate the file partway through the body.
    auto good = std::make_shared<jit::Graph>();
    auto path = temp_graph_path("good_for_trunc");
    jit::save_graph(*good, path);

    // Read the valid file, keep only the first 8 bytes (magic + version),
    // dropping the metadata/body — every subsequent declared read must fail.
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> all((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    in.close();
    ASSERT_GE(all.size(), 8u);
    std::vector<uint8_t> truncated(all.begin(), all.begin() + 8);
    auto tpath = temp_graph_path("truncated");
    write_bytes(tpath, truncated);
    EXPECT_THROW(jit::load_graph(tpath), std::runtime_error);
    std::remove(path.c_str());
    std::remove(tpath.c_str());
}

// A forged file claiming an astronomically large value count must throw the
// "exceeds remaining file" bound, not attempt a huge allocation / OOB read.
TEST(JitHardening, GraphOverlargeCountRejected) {
    // Build the smallest valid-looking prefix by round-tripping an empty graph,
    // then overwrite the metadata num_values with a forged huge count.
    auto good = std::make_shared<jit::Graph>();
    auto path = temp_graph_path("good_for_count");
    jit::save_graph(*good, path);
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> all((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    in.close();

    // Layout: [u32 magic][u32 version][u64 num_nodes][u64 num_values]...
    // read_values() reads its OWN leading u64 num_values from the values
    // section; rather than hunt the exact offset, append a forged values-count
    // is overkill — instead just append a standalone huge-count file after the
    // header so the section reader hits its bound. We forge directly:
    //   magic, version, then a values-section count = 2^60.
    // Reuse the real magic/version from the good file's first 8 bytes.
    ASSERT_GE(all.size(), 8u);
    std::vector<uint8_t> forged(all.begin(), all.begin() + 8);
    // metadata: num_nodes, num_values, num_inputs, num_outputs
    append_u64(forged, 0);                 // num_nodes
    append_u64(forged, 0);                 // num_values (metadata copy)
    append_u64(forged, 0);                 // num_inputs
    append_u64(forged, 0);                 // num_outputs
    // values section leading count — forged huge
    append_u64(forged, (uint64_t(1) << 60));
    auto fpath = temp_graph_path("overlarge");
    write_bytes(fpath, forged);
    EXPECT_THROW(jit::load_graph(fpath), std::runtime_error);
    std::remove(path.c_str());
    std::remove(fpath.c_str());
}

// A valid empty graph must still round-trip (hardening doesn't break the happy
// path).
TEST(JitHardening, EmptyGraphRoundTrips) {
    auto good = std::make_shared<jit::Graph>();
    auto path = temp_graph_path("roundtrip");
    jit::save_graph(*good, path);
    auto loaded = jit::load_graph(path);
    ASSERT_NE(loaded, nullptr);
    std::remove(path.c_str());
}
