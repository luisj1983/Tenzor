// =============================================================================
// tests/backends/test_init_ocl_workaround.cpp
//
// S26: regression tests for the AMD-OCL ICD probe workaround in
// src/core/init.cpp.
//
// What the workaround does:
//   When OCL_ICD_VENDORS is unset, tenzor::initialize() discovers an Intel
//   OpenCL ICD (libintelocl.so) via env-var override or oneAPI install path,
//   writes a per-process vendor file under $TMPDIR/tenzor_ocl_vendors_<pid>/,
//   and sets OCL_ICD_VENDORS to point at that directory. If no ICD is
//   available, the workaround skips silently (warn-once).
//
// Constraints:
//   tenzor::initialize() is idempotent — it runs at most once per process.
//   Therefore each scenario must run in its own process. We use gtest's
//   "death-test style" subprocess invocation via fork() so each TEST owns a
//   fresh init() call. The parent process never calls tenzor::initialize().
//
// Scenarios covered:
//   1. TENZOR_OCL_ICD_PATH override is honoured: a fabricated path is written
//      verbatim into <tmpdir>/tenzor_ocl_vendors_<pid>/intel64.icd, and the
//      OCL_ICD_VENDORS env var points at that directory.
//   2. Non-existent TENZOR_OCL_ICD_PATH skips workaround silently (no /tmp
//      file created, OCL_ICD_VENDORS stays unset).
//   3. Missing oneAPI install (ONEAPI_ROOT=/nonexistent and ICD env vars
//      cleared) skips workaround silently when /opt/intel/oneapi is also
//      absent. This is best-effort: if the host has /opt/intel/oneapi we
//      GTEST_SKIP, since we can't prove the negative.
// =============================================================================

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "tenzor/tenzor.hpp"

namespace fs = std::filesystem;

namespace {

// Resolve TMPDIR the same way init.cpp does, so the test reads from the same
// directory the workaround writes to.
auto tmp_root() -> fs::path {
    const char* env = std::getenv("TMPDIR");
    return (env && *env) ? fs::path(env) : fs::path("/tmp");
}

// Run a child function in a forked subprocess and return its exit code.
// gtest assertions in the child still print, but only the exit code reaches
// the parent assertion. The child should exit(0) on success, non-zero on
// failure, ideally after printing what went wrong to stderr.
template <typename Fn>
auto run_in_subprocess(Fn&& fn) -> int {
    pid_t pid = fork();
    if (pid == 0) {
        // Child.
        int rc = 0;
        try {
            rc = fn();
        } catch (...) {
            rc = 99;
        }
        // _exit avoids running parent atexit handlers / gtest cleanup.
        _exit(rc);
    }
    // Parent.
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 128;  // abnormal termination
}

}  // namespace

// ---------------------------------------------------------------------------
// Scenario 1: TENZOR_OCL_ICD_PATH is honoured.
// ---------------------------------------------------------------------------
TEST(InitOclWorkaround, EnvVarOverrideIsHonoured) {
    // Fabricate a fake ICD payload file. The workaround only checks existence,
    // not that it's a real .so, so an empty regular file is enough.
    fs::path fake_icd = tmp_root() / "tenzor_test_fake_libintelocl.so";
    {
        std::ofstream f(fake_icd);
        ASSERT_TRUE(f.good()) << "could not create fake ICD at " << fake_icd;
        f << "fake-icd-marker\n";
    }

    int rc = run_in_subprocess([&]() -> int {
        // Wipe OCL_ICD_VENDORS so the workaround is actually invoked.
        ::unsetenv("OCL_ICD_VENDORS");
        ::unsetenv("INTEL_OPENCL_ICD_PATH");
        // Override picks up first; should win over ONEAPI_ROOT discovery.
        ::setenv("TENZOR_OCL_ICD_PATH", fake_icd.c_str(), 1);

        tenzor::initialize();

        // Verify per-process vendor dir exists with the expected ICD content.
        fs::path expected_dir = tmp_root() / ("tenzor_ocl_vendors_"
                                              + std::to_string(::getpid()));
        if (!fs::is_directory(expected_dir)) {
            std::fprintf(stderr, "vendor dir not created: %s\n",
                         expected_dir.c_str());
            return 1;
        }
        fs::path icd_file = expected_dir / "intel64.icd";
        if (!fs::is_regular_file(icd_file)) {
            std::fprintf(stderr, "intel64.icd not written: %s\n",
                         icd_file.c_str());
            return 2;
        }
        std::ifstream f(icd_file);
        std::string line;
        std::getline(f, line);
        if (line != fake_icd.string()) {
            std::fprintf(stderr,
                         "intel64.icd content mismatch: got '%s', expected '%s'\n",
                         line.c_str(), fake_icd.c_str());
            return 3;
        }
        // And OCL_ICD_VENDORS should point at the per-process dir.
        const char* vendors = std::getenv("OCL_ICD_VENDORS");
        if (!vendors || std::string(vendors) != expected_dir.string()) {
            std::fprintf(stderr,
                         "OCL_ICD_VENDORS mismatch: got '%s', expected '%s'\n",
                         vendors ? vendors : "(null)", expected_dir.c_str());
            return 4;
        }
        return 0;
    });

    // Best-effort cleanup of the fake ICD; the per-process vendor dir is
    // cleaned by tenzor::finalize() inside the child.
    std::error_code ec;
    fs::remove(fake_icd, ec);

    EXPECT_EQ(rc, 0) << "subprocess failed; see stderr above";
}

// ---------------------------------------------------------------------------
// Scenario 2: TENZOR_OCL_ICD_PATH points at a non-existent file -> silent skip.
// ---------------------------------------------------------------------------
TEST(InitOclWorkaround, NonExistentOverridePathSkipsSilently) {
    int rc = run_in_subprocess([]() -> int {
        ::unsetenv("OCL_ICD_VENDORS");
        ::unsetenv("INTEL_OPENCL_ICD_PATH");
        ::setenv("TENZOR_OCL_ICD_PATH",
                 "/this/path/does/not/exist/libintelocl.so", 1);

        tenzor::initialize();

        // Workaround must have skipped: no per-process vendor dir.
        fs::path expected_dir = tmp_root() / ("tenzor_ocl_vendors_"
                                              + std::to_string(::getpid()));
        if (fs::exists(expected_dir)) {
            std::fprintf(stderr,
                         "vendor dir unexpectedly created: %s\n",
                         expected_dir.c_str());
            return 1;
        }
        // And OCL_ICD_VENDORS must not have been set by the workaround.
        if (std::getenv("OCL_ICD_VENDORS") != nullptr) {
            std::fprintf(stderr,
                         "OCL_ICD_VENDORS unexpectedly set to '%s'\n",
                         std::getenv("OCL_ICD_VENDORS"));
            return 2;
        }
        return 0;
    });
    EXPECT_EQ(rc, 0) << "subprocess failed; see stderr above";
}

// ---------------------------------------------------------------------------
// Scenario 3: No oneAPI install at all -> silent skip.
// ---------------------------------------------------------------------------
TEST(InitOclWorkaround, NoOneApiInstallSkipsSilently) {
    // We can only prove the skip path if /opt/intel/oneapi is absent on this
    // host. Otherwise the workaround may legitimately succeed.
    if (fs::is_directory("/opt/intel/oneapi/compiler")) {
        GTEST_SKIP() << "/opt/intel/oneapi/compiler is present on host; "
                        "cannot test the no-oneAPI skip path here.";
    }

    int rc = run_in_subprocess([]() -> int {
        ::unsetenv("OCL_ICD_VENDORS");
        ::unsetenv("TENZOR_OCL_ICD_PATH");
        ::unsetenv("INTEL_OPENCL_ICD_PATH");
        ::setenv("ONEAPI_ROOT", "/definitely/not/a/real/path", 1);

        tenzor::initialize();

        fs::path expected_dir = tmp_root() / ("tenzor_ocl_vendors_"
                                              + std::to_string(::getpid()));
        if (fs::exists(expected_dir)) {
            std::fprintf(stderr,
                         "vendor dir unexpectedly created: %s\n",
                         expected_dir.c_str());
            return 1;
        }
        if (std::getenv("OCL_ICD_VENDORS") != nullptr) {
            std::fprintf(stderr,
                         "OCL_ICD_VENDORS unexpectedly set to '%s'\n",
                         std::getenv("OCL_ICD_VENDORS"));
            return 2;
        }
        return 0;
    });
    EXPECT_EQ(rc, 0) << "subprocess failed; see stderr above";
}
