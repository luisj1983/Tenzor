/**
 * @file golden_util.hpp
 * @brief Golden-tensor record/replay for single-backend parity hosts.
 *
 * The parity tests compare each backend's output against CPU. On a CPU-only
 * host there is nothing to compare against, so tests historically became
 * no-ops. This header provides a pre-recorded "golden" fallback: on a
 * multi-backend host we record CPU outputs to disk once, commit them, and
 * single-backend CI jobs verify against the recorded tensor.
 *
 * Binary format (little-endian):
 *   offset 0  : 4 bytes magic 'TGLD'
 *   offset 4  : 4 bytes version (uint32_t, currently 1)
 *   offset 8  : 4 bytes dtype (DType enum as uint32_t)
 *   offset 12 : 4 bytes rank (uint32_t)
 *   offset 16 : rank * 8 bytes shape (int64_t each)
 *   offset 16 + 8*rank : raw data, row-major, contiguous, dtype-sized
 *
 * All integers are written in host byte order; we only target little-endian
 * platforms (x86_64, aarch64 LE) — if a big-endian port ever matters we'll
 * add swapping here.
 */

#pragma once

#include <tenzor/tenzor.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <system_error>
#include <string>
#include <vector>
#include <unistd.h>  // getpid (per-process coverage report filename)

namespace tenzor {
namespace testing {
namespace golden {

// -- Configuration -----------------------------------------------------------

inline bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && *v != '0';
}

inline bool recording_enabled() { return env_flag("TENZOR_RECORD_GOLDENS"); }
inline bool require_multi_backend() { return env_flag("TENZOR_REQUIRE_MULTI_BACKEND"); }

// -- Coverage accounting -----------------------------------------------------
//
// CPU-only CI shards (.github/workflows/ci.yml -> full-cpu-tests/backend_parity)
// run the parity suite against committed goldens. Without a counter, a host
// where every golden fingerprint mis-matches (e.g. goldens were never recorded
// or the input generator changed) silently degrades to zero real comparisons
// while still reporting "PASS" — exactly the rot this gate is meant to catch.
//
// `note_comparison()` is called once per *real* recorded-golden comparison
// (CPU output actually checked against a loaded golden). When
// TENZOR_GOLDEN_COVERAGE_REPORT names a file, the process-exit hook writes the
// running tally there so a CI post-step can assert a coverage floor. The
// counter is process-local and append-safe: each test binary writes its own
// count, and the CI step sums the files.
inline int& comparison_counter() {
    static int count = 0;
    return count;
}

// Counts golden LOAD ATTEMPTS (every maybe_load call), as opposed to
// comparison_counter() which counts successful loads that were actually
// compared. The gap between the two is the silent-disable failure mode the
// audit flagged: if the input generator / RNG-seed scheme changes, every
// fingerprint shifts, maybe_load returns nullopt for all of them, and the
// suite degrades to zero comparisons while still reporting PASS. By tracking
// attempts we can assert at process exit that a binary which TRIED to use
// goldens actually compared at least one — without false-failing a binary
// that legitimately has no goldens for its ops (attempts == 0).
inline int& load_attempt_counter() {
    static int count = 0;
    return count;
}

// Flush the comparison count to $TENZOR_GOLDEN_COVERAGE_REPORT (if set).
// Multiple parity test binaries run in one CI step and would clobber a shared
// path, so each process writes to "<report>.<pid>" — the CI step globs and
// SUMS them. Registered via atexit on first note_comparison() so it fires
// exactly once after RUN_ALL_TESTS().
inline void flush_coverage_report() {
    const char* path = std::getenv("TENZOR_GOLDEN_COVERAGE_REPORT");
    if (!path || !*path) return;
    std::string per_pid = std::string(path) + "." +
                          std::to_string(static_cast<long>(::getpid()));
    std::ofstream os(per_pid, std::ios::trunc);
    if (os) os << comparison_counter() << "\n";
}

inline void note_comparison() {
    static bool registered = [] {
        // Only arm the flush when a report path is requested; avoids touching
        // the filesystem for ordinary local runs.
        if (const char* p = std::getenv("TENZOR_GOLDEN_COVERAGE_REPORT"); p && *p) {
            std::atexit(&flush_coverage_report);
        }
        return true;
    }();
    (void)registered;
    ++comparison_counter();
}

// Process-exit gate: if this binary attempted to load goldens at all but never
// successfully compared even one, every golden it expected went missing — the
// fingerprints no longer match any committed .gold file (input generator or
// seed scheme changed, or the golden dir was lost). That silently turns the
// parity comparisons into no-ops while the suite still reports PASS, which is
// exactly the rot this check exists to surface. Force a non-zero process exit
// so ctest goes red instead of green-on-nothing. Opt out with
// TENZOR_GOLDEN_ALLOW_ZERO=1 (e.g. when intentionally re-recording goldens).
inline void enforce_golden_coverage_at_exit() {
    if (const char* a = std::getenv("TENZOR_GOLDEN_ALLOW_ZERO"); a && *a && *a != '0')
        return;
    // When recording, no comparisons happen by design — don't fail.
    if (recording_enabled()) return;
    // This gate is OPT-IN. Enforcing it unconditionally would turn every
    // legitimate "no committed golden for this fingerprint yet" case into a hard
    // process failure: a CPU-only dev host (or any binary whose ops simply have
    // no recorded goldens) attempts loads, matches none, and would be killed
    // despite all its gtest assertions passing/skipping. CI that wants the
    // coverage floor sets TENZOR_REQUIRE_GOLDEN_COVERAGE=1 (or names a
    // TENZOR_GOLDEN_COVERAGE_REPORT to collect the per-binary tallies); only
    // then do we fail a binary that attempted goldens but compared none.
    const char* require = std::getenv("TENZOR_REQUIRE_GOLDEN_COVERAGE");
    const char* report  = std::getenv("TENZOR_GOLDEN_COVERAGE_REPORT");
    const bool enforce =
        (require && *require && *require != '0') || (report && *report);
    if (!enforce) return;
    if (load_attempt_counter() > 0 && comparison_counter() == 0) {
        std::cerr << "[GOLDEN COVERAGE FAILURE] " << load_attempt_counter()
                  << " golden load(s) were attempted but NONE matched a committed "
                     ".gold file — every golden comparison silently became a no-op. "
                     "The input generator or seed scheme likely changed, abandoning "
                     "all goldens. Re-record with TENZOR_RECORD_GOLDENS=1 on a "
                     "multi-backend host (or set TENZOR_GOLDEN_ALLOW_ZERO=1 to bypass)."
                  << std::endl;
        std::_Exit(1);
    }
}

// Called on every maybe_load(); arms the exit gate on first attempt.
inline void note_load_attempt() {
    static bool registered = [] {
        std::atexit(&enforce_golden_coverage_at_exit);
        return true;
    }();
    (void)registered;
    ++load_attempt_counter();
}

/**
 * Directory containing recorded goldens.
 *
 * Resolution order:
 *   1. $TENZOR_GOLDEN_DIR env var (absolute or cwd-relative path)
 *   2. TENZOR_TEST_GOLDEN_DIR compile-time define (set by CMake)
 *   3. "./tests/backend_parity/golden" (fallback for ad-hoc runs from src tree)
 */
inline std::string golden_dir() {
    if (const char* env = std::getenv("TENZOR_GOLDEN_DIR"); env && *env) {
        return env;
    }
#ifdef TENZOR_TEST_GOLDEN_DIR
    return TENZOR_TEST_GOLDEN_DIR;
#else
    return "tests/backend_parity/golden";
#endif
}

// -- Key construction --------------------------------------------------------

/**
 * Sanitize a test name into a filename-safe token (keep alphanumerics and '_',
 * replace the rest with '_').
 */
inline std::string sanitize(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') out.push_back(c);
        else out.push_back('_');
    }
    return out;
}

/**
 * FNV-1a 64-bit hash over a byte span. Stable across builds/platforms.
 */
inline uint64_t fnv1a64(const void* data, std::size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

/**
 * Compute a stable fingerprint over (test_name, inputs). Inputs are fingerprinted
 * by shape + dtype + raw contiguous CPU bytes.
 */
inline uint64_t fingerprint_inputs(std::string_view test_name,
                                   const std::vector<Tensor>& inputs) {
    uint64_t h = fnv1a64(test_name.data(), test_name.size());
    for (const auto& t : inputs) {
        auto shape = t.shape();
        auto dtype = static_cast<uint32_t>(t.dtype());
        h ^= fnv1a64(shape.data(), shape.size() * sizeof(int64_t));
        h ^= fnv1a64(&dtype, sizeof(dtype));
        // Move to CPU for a stable byte image. Parity tests generate contiguous
        // inputs via generate_test_tensor, so we don't force a contiguous copy
        // here — if a non-contiguous tensor is ever passed, its strided byte
        // layout will still hash deterministically for a given test.
        Tensor cpu = t.device().type == Device::Type::CPU ? t : t.to(Device::cpu());
        const void* ptr = cpu.data_ptr();
        std::size_t bytes = static_cast<std::size_t>(cpu.numel()) * ::tenzor::dtype_size(cpu.dtype());
        if (ptr && bytes) h ^= fnv1a64(ptr, bytes);
    }
    return h;
}

inline std::string golden_path(std::string_view test_name, uint64_t fingerprint) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(fingerprint));
    return golden_dir() + "/" + sanitize(test_name) + "_" + buf + ".gold";
}

// -- Read / write ------------------------------------------------------------

inline constexpr uint32_t kMagic = 0x444c4754; // 'TGLD' little-endian
inline constexpr uint32_t kVersion = 1;

/**
 * Write `t` to `path`. Creates parent dirs. Overwrites existing.
 * Returns true on success.
 */
inline bool write_golden(const std::string& path, const Tensor& t) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    // Move to CPU, ensure contiguous, canonical dtype-sized bytes. The
    // contiguify is REQUIRED: view-producing ops (transpose/permute/chunk)
    // return non-contiguous tensors whose data_ptr() walks the underlying
    // storage in physical order, not logical order. Serializing those raw
    // bytes with the logical shape header would store transposed/permuted data
    // in the wrong order, so the golden would mismatch the (correctly
    // contiguified) live result on load.
    Tensor cpu = t.device().type == Device::Type::CPU ? t : t.to(Device::cpu());
    if (!cpu.is_contiguous()) cpu = cpu.contiguous();
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    uint32_t dtype = static_cast<uint32_t>(cpu.dtype());
    auto shape_vec = cpu.shape();
    uint32_t rank = static_cast<uint32_t>(shape_vec.size());
    os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
    os.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    os.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    for (int64_t d : shape_vec) {
        os.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }
    std::size_t bytes = static_cast<std::size_t>(cpu.numel()) * ::tenzor::dtype_size(cpu.dtype());
    if (bytes && cpu.data_ptr()) {
        os.write(reinterpret_cast<const char*>(cpu.data_ptr()),
                 static_cast<std::streamsize>(bytes));
    }
    return static_cast<bool>(os);
}

/**
 * Load tensor from `path`. Returns nullopt on any error.
 */
inline std::optional<Tensor> read_golden(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return std::nullopt;
    uint32_t magic = 0, version = 0, dtype_raw = 0, rank = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    is.read(reinterpret_cast<char*>(&dtype_raw), sizeof(dtype_raw));
    is.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    if (!is || magic != kMagic || version != kVersion) return std::nullopt;
    std::vector<int64_t> shape(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        is.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
    }
    if (!is) return std::nullopt;
    DType dtype = static_cast<DType>(dtype_raw);
    Tensor out = zeros(shape, dtype, Device::cpu());
    std::size_t bytes = static_cast<std::size_t>(out.numel()) * ::tenzor::dtype_size(dtype);
    if (bytes) {
        is.read(reinterpret_cast<char*>(out.data_ptr()),
                static_cast<std::streamsize>(bytes));
        if (!is) return std::nullopt;
    }
    return out;
}

// -- High-level API used by parity helpers -----------------------------------

/**
 * Try to record `result` under (test_name, inputs-fingerprint). No-op unless
 * $TENZOR_RECORD_GOLDENS is set. Returns the path written (or empty string).
 */
inline std::string maybe_record(std::string_view test_name,
                                const std::vector<Tensor>& inputs,
                                const Tensor& result) {
    if (!recording_enabled()) return "";
    // Do NOT record goldens for very large tensors. Goldens exist to give
    // CPU-only CI hosts a committed correctness reference for the small,
    // representative parity cases — not to store multi-MB/GB stress-test blobs
    // in git (GitHub warns >50MB and hard-rejects >100MB; a 2048x2048 f32
    // tensor is already 16MB, a 1GB stress tensor is 1GB). Large/stress tests
    // still compare LIVE across backends on a multi-backend host; they simply
    // skip the CPU-vs-golden fallback.
    constexpr std::size_t kMaxGoldenBytes = 8u * 1024u * 1024u;  // 8 MiB
    if (static_cast<std::size_t>(result.numel()) * ::tenzor::dtype_size(result.dtype())
            > kMaxGoldenBytes) {
        return "";
    }
    uint64_t fp = fingerprint_inputs(test_name, inputs);
    std::string path = golden_path(test_name, fp);
    if (!write_golden(path, result)) return "";
    return path;
}

/**
 * Try to load a previously recorded golden for (test_name, inputs-fingerprint).
 *
 * Phase 9 hardening: prints a loud warning to stderr each time a golden is
 * loaded so CI logs make stale-golden usage visible (the substitution that
 * the audit flagged as "silently masking divergence"). Also enforces an age
 * limit: if the golden file is older than the configurable threshold (default
 * 30 days, override via TENZOR_GOLDEN_MAX_AGE_DAYS), refuse to load and force
 * the test to skip — stale goldens are no better than no goldens.
 */
inline std::optional<Tensor> maybe_load(std::string_view test_name,
                                        const std::vector<Tensor>& inputs) {
    note_load_attempt();
    uint64_t fp = fingerprint_inputs(test_name, inputs);
    std::string path = golden_path(test_name, fp);

    // Staleness guard: refuse goldens older than max-age days.
    int max_age_days = 30;
    if (const char* v = std::getenv("TENZOR_GOLDEN_MAX_AGE_DAYS")) {
        try { max_age_days = std::max(1, std::stoi(v)); } catch (...) {}
    }
    std::error_code ec;
    auto ftime = std::filesystem::last_write_time(path, ec);
    if (!ec) {
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now()
                  + std::chrono::system_clock::now());
        auto age = std::chrono::system_clock::now() - sctp;
        auto age_days = std::chrono::duration_cast<std::chrono::hours>(age).count() / 24;
        if (age_days > max_age_days) {
            std::cerr << "[GOLDEN STALE] " << path << " is " << age_days
                      << " days old (max=" << max_age_days
                      << "). Refusing to use; record fresh goldens with "
                         "TENZOR_RECORD_GOLDENS=1.\n";
            return std::nullopt;
        }
    }

    auto loaded = read_golden(path);
    if (loaded) {
        std::cerr << "[GOLDEN FALLBACK] " << test_name
                  << " using recorded golden at " << path
                  << " — only one backend was available. Set "
                     "TENZOR_REQUIRE_MULTI_BACKEND=1 to fail fast in CI.\n";
    }
    return loaded;
}

} // namespace golden
} // namespace testing
} // namespace tenzor
