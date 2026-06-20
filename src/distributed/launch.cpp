/**
 * @file launch.cpp
 * @brief Implementation of distributed training launch utility
 *
 * Implements process spawning via fork() for multi-process distributed
 * training on a single node. On POSIX systems, uses fork()/exec. On
 * Windows, uses CreateProcess() with custom environment blocks.
 * Sets environment variables for each worker process to enable
 * distributed communication setup.
 */

#include "tenzor/distributed/launch.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cctype>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "Ws2_32.lib")
#endif

namespace tenzor::distributed {

// ============================================================================
// spawn() Implementation
// ============================================================================

auto spawn(const LaunchConfig& config, WorkerFn worker_fn) -> std::vector<int> {
    if (config.nproc_per_node <= 0) {
        throw std::invalid_argument(
            "spawn: nproc_per_node must be positive, got " +
            std::to_string(config.nproc_per_node)
        );
    }

    if (config.nnodes <= 0) {
        throw std::invalid_argument(
            "spawn: nnodes must be positive, got " +
            std::to_string(config.nnodes)
        );
    }

    if (config.node_rank < 0 || config.node_rank >= config.nnodes) {
        throw std::invalid_argument(
            "spawn: node_rank " + std::to_string(config.node_rank) +
            " must be in range [0, " + std::to_string(config.nnodes) + ")"
        );
    }

    int world_size = config.world_size();
    int nproc = config.nproc_per_node;
    std::vector<int> exit_codes(nproc, -1);

#if defined(__linux__) || defined(__APPLE__)

    // Create log directory if needed
    if (config.log_to_file) {
        std::filesystem::create_directories(config.log_dir);
    }

    std::vector<pid_t> child_pids;
    child_pids.reserve(nproc);

    for (int local_rank = 0; local_rank < nproc; ++local_rank) {
        int global_rank = config.global_rank(local_rank);

        pid_t pid = fork();

        if (pid < 0) {
            // Fork failed - kill already spawned children
            for (auto cpid : child_pids) {
                kill(cpid, SIGTERM);
            }
            throw std::runtime_error(
                "spawn: fork() failed for local_rank " +
                std::to_string(local_rank)
            );
        }

        if (pid == 0) {
            // Child process

            // Set environment variables for distributed communication
            setenv("RANK", std::to_string(global_rank).c_str(), 1);
            setenv("LOCAL_RANK", std::to_string(local_rank).c_str(), 1);
            setenv("WORLD_SIZE", std::to_string(world_size).c_str(), 1);
            setenv("MASTER_ADDR", config.master_addr.c_str(), 1);
            setenv("MASTER_PORT", std::to_string(config.master_port).c_str(), 1);

            // Redirect output to log files if configured
            if (config.log_to_file) {
                std::string log_file = config.log_dir + "/worker_" +
                                       std::to_string(global_rank) + ".log";
                FILE* log_fp = fopen(log_file.c_str(), "w");
                if (log_fp) {
                    dup2(fileno(log_fp), STDOUT_FILENO);
                    dup2(fileno(log_fp), STDERR_FILENO);
                    fclose(log_fp);
                }
            }

            // Run the worker function
            try {
                worker_fn(global_rank, world_size);
                _exit(0);
            } catch (const std::exception& e) {
                std::cerr << "[Rank " << global_rank << "] Fatal error: "
                          << e.what() << std::endl;
                _exit(1);
            } catch (...) {
                std::cerr << "[Rank " << global_rank
                          << "] Unknown fatal error" << std::endl;
                _exit(1);
            }
        }

        // Parent process: record child PID
        child_pids.push_back(pid);
    }

    // Parent process: wait for all children to complete
    bool any_failed = false;

    for (int i = 0; i < nproc; ++i) {
        int status = 0;
        pid_t waited = waitpid(child_pids[i], &status, 0);

        if (waited < 0) {
            exit_codes[i] = -1;
            any_failed = true;
            continue;
        }

        if (WIFEXITED(status)) {
            exit_codes[i] = WEXITSTATUS(status);
            if (exit_codes[i] != 0) {
                any_failed = true;
            }
        } else if (WIFSIGNALED(status)) {
            int sig = WTERMSIG(status);
            exit_codes[i] = 128 + sig;  // Convention: signal N -> exit code 128+N
            any_failed = true;

            std::cerr << "[Launch] Worker " << i << " (rank "
                      << config.global_rank(i) << ") killed by signal "
                      << sig << std::endl;
        }
    }

    if (any_failed) {
        // If any worker failed, terminate remaining workers gracefully
        for (int i = 0; i < nproc; ++i) {
            if (exit_codes[i] < 0) {
                // Worker may still be running
                kill(child_pids[i], SIGTERM);
            }
        }
    }

#elif defined(_WIN32)

    // ---- Windows: CreateProcess()-based spawning ----

    // Create log directory if needed
    if (config.log_to_file) {
        std::filesystem::create_directories(config.log_dir);
    }

    // Maximum supported worker processes
    constexpr int max_workers = 256;
    if (nproc > max_workers) {
        throw std::invalid_argument(
            "spawn: nproc_per_node " + std::to_string(nproc) +
            " exceeds maximum supported workers (" +
            std::to_string(max_workers) + ")"
        );
    }

    // RAII wrapper for Win32 HANDLEs to ensure proper cleanup
    struct HandleGuard {
        HANDLE h{INVALID_HANDLE_VALUE};
        HandleGuard() = default;
        explicit HandleGuard(HANDLE handle) : h(handle) {}
        ~HandleGuard() {
            if (h != INVALID_HANDLE_VALUE && h != nullptr) {
                CloseHandle(h);
            }
        }
        HandleGuard(const HandleGuard&) = delete;
        HandleGuard& operator=(const HandleGuard&) = delete;
        HandleGuard(HandleGuard&& other) noexcept : h(other.h) {
            other.h = INVALID_HANDLE_VALUE;
        }
        HandleGuard& operator=(HandleGuard&& other) noexcept {
            if (this != &other) {
                if (h != INVALID_HANDLE_VALUE && h != nullptr) {
                    CloseHandle(h);
                }
                h = other.h;
                other.h = INVALID_HANDLE_VALUE;
            }
            return *this;
        }
    };

    std::vector<HandleGuard> process_handles(nproc);
    std::vector<HandleGuard> thread_handles(nproc);

    // Get path to current executable for re-launching workers
    wchar_t exe_path[MAX_PATH];
    DWORD exe_path_len = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (exe_path_len == 0 || exe_path_len >= MAX_PATH) {
        throw std::runtime_error(
            "spawn: GetModuleFileNameW failed (error " +
            std::to_string(GetLastError()) + ")"
        );
    }

    // Capture the parent's environment block so children inherit all
    // existing variables plus the distributed training overrides.
    // GetEnvironmentStringsW returns a double-null-terminated block of
    // "KEY=VALUE\0" pairs.
    wchar_t* parent_env_raw = GetEnvironmentStringsW();
    if (!parent_env_raw) {
        throw std::runtime_error(
            "spawn: GetEnvironmentStringsW failed (error " +
            std::to_string(GetLastError()) + ")"
        );
    }

    // Measure length of parent environment block (up to and including
    // the terminating double null).
    size_t parent_env_len = 0;
    {
        const wchar_t* p = parent_env_raw;
        while (*p) {
            size_t entry_len = wcslen(p) + 1;  // includes null terminator
            p += entry_len;
        }
        parent_env_len = static_cast<size_t>(p - parent_env_raw) + 1;
    }

    // Helper: convert a narrow (UTF-8) string to a wide string
    auto to_wide = [](const std::string& s) -> std::wstring {
        if (s.empty()) return {};
        int needed = MultiByteToWideChar(
            CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
        std::wstring result(static_cast<size_t>(needed), L'\0');
        MultiByteToWideChar(
            CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
            result.data(), needed);
        return result;
    };

    // Names of the environment variables we will override per worker
    static const std::wstring override_keys[] = {
        L"RANK", L"LOCAL_RANK", L"WORLD_SIZE", L"MASTER_ADDR", L"MASTER_PORT"
    };
    constexpr size_t num_override_keys = 5;

    for (int local_rank = 0; local_rank < nproc; ++local_rank) {
        int global_rank = config.global_rank(local_rank);

        // Build override entries for this worker
        std::wstring overrides[num_override_keys] = {
            L"RANK=" + to_wide(std::to_string(global_rank)),
            L"LOCAL_RANK=" + to_wide(std::to_string(local_rank)),
            L"WORLD_SIZE=" + to_wide(std::to_string(world_size)),
            L"MASTER_ADDR=" + to_wide(config.master_addr),
            L"MASTER_PORT=" + to_wide(std::to_string(config.master_port))
        };

        // Build child environment block: copy parent entries (skipping
        // any we are overriding), then append our overrides.
        std::wstring env_block;
        env_block.reserve(parent_env_len + 512);

        const wchar_t* p = parent_env_raw;
        while (*p) {
            std::wstring_view entry(p);
            // Check if this entry starts with one of our override keys
            bool skip = false;
            for (size_t k = 0; k < num_override_keys; ++k) {
                const auto& key = override_keys[k];
                if (entry.size() > key.size() &&
                    entry[key.size()] == L'=' &&
                    entry.substr(0, key.size()) == key) {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                env_block.append(entry.data(), entry.size());
                env_block.push_back(L'\0');
            }
            p += entry.size() + 1;
        }

        // Append our override variables
        for (size_t k = 0; k < num_override_keys; ++k) {
            env_block.append(overrides[k]);
            env_block.push_back(L'\0');
        }

        // Terminate the environment block with a final null
        env_block.push_back(L'\0');

        // Build command line: quote the executable path and pass worker args
        std::wstring cmd_line = L"\"" +
                                std::wstring(exe_path, exe_path_len) +
                                L"\" --tenzor-worker" +
                                L" --rank=" +
                                to_wide(std::to_string(global_rank)) +
                                L" --world-size=" +
                                to_wide(std::to_string(world_size));

        STARTUPINFOW si{};
        si.cb = sizeof(si);

        // Redirect output to log files if configured
        HandleGuard log_handle;
        if (config.log_to_file) {
            std::string log_file = config.log_dir + "/worker_" +
                                   std::to_string(global_rank) + ".log";
            SECURITY_ATTRIBUTES sa{};
            sa.nLength = sizeof(sa);
            sa.bInheritHandle = TRUE;

            HANDLE h = CreateFileW(
                to_wide(log_file).c_str(),
                GENERIC_WRITE,
                FILE_SHARE_READ,
                &sa,
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                nullptr
            );
            if (h != INVALID_HANDLE_VALUE) {
                log_handle = HandleGuard(h);
                si.dwFlags |= STARTF_USESTDHANDLES;
                si.hStdOutput = h;
                si.hStdError = h;
                si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            }
        }

        PROCESS_INFORMATION pi{};

        BOOL ok = CreateProcessW(
            nullptr,                                // lpApplicationName
            cmd_line.data(),                        // lpCommandLine (writable)
            nullptr,                                // lpProcessAttributes
            nullptr,                                // lpThreadAttributes
            config.log_to_file ? TRUE : FALSE,      // bInheritHandles
            CREATE_UNICODE_ENVIRONMENT,             // dwCreationFlags
            env_block.data(),                       // lpEnvironment
            nullptr,                                // lpCurrentDirectory
            &si,                                    // lpStartupInfo
            &pi                                     // lpProcessInformation
        );

        if (!ok) {
            DWORD err = GetLastError();
            // Terminate already-spawned workers before throwing
            for (int j = 0; j < local_rank; ++j) {
                TerminateProcess(process_handles[j].h, 1);
            }
            FreeEnvironmentStringsW(parent_env_raw);
            throw std::runtime_error(
                "spawn: CreateProcessW failed for local_rank " +
                std::to_string(local_rank) + " (error " +
                std::to_string(err) + ")"
            );
        }

        process_handles[local_rank] = HandleGuard(pi.hProcess);
        thread_handles[local_rank] = HandleGuard(pi.hThread);
    }

    FreeEnvironmentStringsW(parent_env_raw);

    // Wait for all workers to complete.
    // WaitForMultipleObjects supports at most MAXIMUM_WAIT_OBJECTS (64)
    // handles per call, so we wait in batches when nproc exceeds that.
    {
        int remaining = nproc;
        int offset = 0;

        while (remaining > 0) {
            int batch = std::min(remaining,
                                 static_cast<int>(MAXIMUM_WAIT_OBJECTS));

            // Build temporary array of raw handles for this batch
            std::vector<HANDLE> raw_handles(batch);
            for (int i = 0; i < batch; ++i) {
                raw_handles[i] = process_handles[offset + i].h;
            }

            DWORD wait_result = WaitForMultipleObjects(
                static_cast<DWORD>(batch),
                raw_handles.data(),
                TRUE,           // bWaitAll
                INFINITE
            );

            if (wait_result == WAIT_FAILED) {
                DWORD err = GetLastError();
                std::cerr << "[Launch] WaitForMultipleObjects failed (error "
                          << err << ")" << std::endl;
            }

            offset += batch;
            remaining -= batch;
        }
    }

    // Collect exit codes from all workers
    bool any_failed = false;
    for (int i = 0; i < nproc; ++i) {
        DWORD code = 0;
        if (GetExitCodeProcess(process_handles[i].h, &code)) {
            if (code == STILL_ACTIVE) {
                // Worker did not terminate; force-kill it
                TerminateProcess(process_handles[i].h, 1);
                exit_codes[i] = -1;
                any_failed = true;
                std::cerr << "[Launch] Worker " << i << " (rank "
                          << config.global_rank(i)
                          << ") still active, terminated" << std::endl;
            } else {
                exit_codes[i] = static_cast<int>(code);
                if (code != 0) {
                    any_failed = true;
                }
            }
        } else {
            exit_codes[i] = -1;
            any_failed = true;
        }
    }

    if (any_failed) {
        // Terminate any workers that may still be running
        for (int i = 0; i < nproc; ++i) {
            DWORD code = 0;
            if (GetExitCodeProcess(process_handles[i].h, &code) &&
                code == STILL_ACTIVE) {
                TerminateProcess(process_handles[i].h, 1);
            }
        }
    }

#else
    // Unsupported platform
    (void)worker_fn;
    throw std::runtime_error(
        "spawn: process spawning is not supported on this platform"
    );
#endif

    return exit_codes;
}

// ============================================================================
// init_from_env() Implementation
// ============================================================================

namespace {

/**
 * @brief Determine whether an address is obviously local (loopback).
 *
 * Recognizes the common loopback spellings without performing any DNS
 * resolution: "localhost", the IPv4 loopback block 127.0.0.0/8, and the
 * IPv6 loopback "::1" (with optional surrounding brackets). Used to skip
 * the blocking TCP reachability probe when the master is the local host,
 * since a loopback connect would either succeed trivially or block waiting
 * for a listener that has not started yet.
 *
 * @param addr Hostname or IP address (as provided in MASTER_ADDR)
 * @return true if the address is a recognized loopback address
 */
auto is_local_address(const std::string& addr) -> bool {
    // Strip optional IPv6 brackets, e.g. "[::1]".
    std::string host = addr;
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }

    // Case-insensitive hostname compare for "localhost".
    std::string lower = host;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "localhost") {
        return true;
    }

    // IPv6 loopback.
    if (host == "::1") {
        return true;
    }

    // IPv4 loopback block 127.0.0.0/8: any address of the form 127.x.y.z.
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    struct in_addr v4{};
    if (inet_pton(AF_INET, host.c_str(), &v4) == 1) {
        // ntohl gives host-order; high byte 127 means loopback block.
        return (ntohl(v4.s_addr) >> 24) == 127;
    }
#endif

    return false;
}

/**
 * @brief Check TCP reachability of master_addr:master_port with timeout.
 *
 * Attempts a non-blocking TCP connect to verify the address is resolvable
 * and the port is reachable. Times out after timeout_ms milliseconds.
 *
 * @param addr Hostname or IP address
 * @param port TCP port number
 * @param timeout_ms Connection timeout in milliseconds
 * @return Empty string on success, error description on failure
 */
auto check_master_reachability(const std::string& addr, int port,
                               int timeout_ms = 5000) -> std::string {
#if defined(__linux__) || defined(__APPLE__)
    // Resolve hostname
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;      // IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;  // TCP
    hints.ai_flags = AI_NUMERICSERV;

    struct addrinfo* result = nullptr;
    std::string port_str = std::to_string(port);
    int rv = getaddrinfo(addr.c_str(), port_str.c_str(), &hints, &result);
    if (rv != 0) {
        return "MASTER_ADDR '" + addr + "' cannot be resolved: " +
               std::string(gai_strerror(rv));
    }

    // Try each resolved address
    std::string last_error;
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        int sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sockfd < 0) {
            last_error = "socket() failed: " + std::string(strerror(errno));
            continue;
        }

        // Set non-blocking for timeout-controlled connect
        int flags = fcntl(sockfd, F_GETFL, 0);
        fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

        int connect_rv = connect(sockfd, rp->ai_addr, rp->ai_addrlen);
        if (connect_rv == 0) {
            // Connected immediately
            close(sockfd);
            freeaddrinfo(result);
            return "";  // Success
        }

        if (errno != EINPROGRESS) {
            last_error = "connect() failed: " + std::string(strerror(errno));
            close(sockfd);
            continue;
        }

        // Wait for connection with timeout
        struct pollfd pfd{};
        pfd.fd = sockfd;
        pfd.events = POLLOUT;

        int poll_rv = poll(&pfd, 1, timeout_ms);
        if (poll_rv > 0 && (pfd.revents & POLLOUT)) {
            // Check if connection actually succeeded
            int so_error = 0;
            socklen_t len = sizeof(so_error);
            getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &so_error, &len);
            close(sockfd);

            if (so_error == 0) {
                freeaddrinfo(result);
                return "";  // Success
            }
            last_error = "connect error: " + std::string(strerror(so_error));
        } else if (poll_rv == 0) {
            close(sockfd);
            last_error = "connection timed out after " +
                         std::to_string(timeout_ms) + "ms";
        } else {
            close(sockfd);
            last_error = "poll() failed: " + std::string(strerror(errno));
        }
    }

    freeaddrinfo(result);
    return "MASTER_ADDR '" + addr + ":" + std::to_string(port) +
           "' is not reachable: " + last_error;
#elif defined(_WIN32)
    // Windows: Winsock-based reachability check

    // Initialize Winsock (safe to call multiple times; refcounted)
    WSADATA wsa_data{};
    int wsa_rv = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (wsa_rv != 0) {
        return "WSAStartup failed (error " + std::to_string(wsa_rv) + ")";
    }

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICSERV;

    struct addrinfo* result = nullptr;
    std::string port_str = std::to_string(port);
    int rv = getaddrinfo(addr.c_str(), port_str.c_str(), &hints, &result);
    if (rv != 0) {
        WSACleanup();
        return "MASTER_ADDR '" + addr + "' cannot be resolved (error " +
               std::to_string(rv) + ")";
    }

    std::string last_error;
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        SOCKET sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sockfd == INVALID_SOCKET) {
            last_error = "socket() failed (error " +
                         std::to_string(WSAGetLastError()) + ")";
            continue;
        }

        // Set non-blocking mode
        u_long nonblocking = 1;
        ioctlsocket(sockfd, FIONBIO, &nonblocking);

        int connect_rv = connect(sockfd, rp->ai_addr,
                                 static_cast<int>(rp->ai_addrlen));
        if (connect_rv == 0) {
            closesocket(sockfd);
            freeaddrinfo(result);
            WSACleanup();
            return "";  // Success
        }

        if (WSAGetLastError() != WSAEWOULDBLOCK) {
            last_error = "connect() failed (error " +
                         std::to_string(WSAGetLastError()) + ")";
            closesocket(sockfd);
            continue;
        }

        // Wait for connection with timeout using select()
        fd_set write_fds;
        fd_set except_fds;
        FD_ZERO(&write_fds);
        FD_ZERO(&except_fds);
        FD_SET(sockfd, &write_fds);
        FD_SET(sockfd, &except_fds);

        struct timeval tv{};
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        int sel_rv = select(0, nullptr, &write_fds, &except_fds, &tv);
        if (sel_rv > 0 && FD_ISSET(sockfd, &write_fds)) {
            // Check if connection succeeded
            int so_error = 0;
            int len = sizeof(so_error);
            getsockopt(sockfd, SOL_SOCKET, SO_ERROR,
                       reinterpret_cast<char*>(&so_error), &len);
            closesocket(sockfd);

            if (so_error == 0) {
                freeaddrinfo(result);
                WSACleanup();
                return "";  // Success
            }
            last_error = "connect error (code " +
                         std::to_string(so_error) + ")";
        } else if (sel_rv == 0) {
            closesocket(sockfd);
            last_error = "connection timed out after " +
                         std::to_string(timeout_ms) + "ms";
        } else {
            closesocket(sockfd);
            last_error = "select() failed (error " +
                         std::to_string(WSAGetLastError()) + ")";
        }
    }

    freeaddrinfo(result);
    WSACleanup();
    return "MASTER_ADDR '" + addr + ":" + std::to_string(port) +
           "' is not reachable: " + last_error;
#else
    // On unsupported platforms, skip reachability check
    (void)addr;
    (void)port;
    (void)timeout_ms;
    return "";
#endif
}

} // anonymous namespace

auto init_from_env(const std::string& backend) -> void {
    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");
    const char* master_addr_env = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");

    // -- Validate required environment variables --
    if (!rank_env || !world_size_env) {
        throw std::runtime_error(
            "init_from_env: RANK and WORLD_SIZE environment variables must "
            "be set. Launch with tenzor::distributed::spawn() or an external "
            "launcher like torchrun."
        );
    }

    // -- Parse and validate WORLD_SIZE --
    int world_size = std::atoi(world_size_env);
    if (world_size <= 0) {
        throw std::invalid_argument(
            "init_from_env: WORLD_SIZE must be > 0, got '" +
            std::string(world_size_env) + "'"
        );
    }

    // -- Parse and validate RANK --
    int rank = std::atoi(rank_env);
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "init_from_env: RANK must be in range [0, WORLD_SIZE=" +
            std::to_string(world_size) + "), got '" +
            std::string(rank_env) + "'"
        );
    }

    // -- Parse and validate MASTER_PORT --
    std::string master_addr = master_addr_env ? master_addr_env : "localhost";
    int master_port = master_port_env ? std::atoi(master_port_env) : 29500;

    if (master_port < 1 || master_port > 65535) {
        throw std::invalid_argument(
            "init_from_env: MASTER_PORT must be in range [1, 65535], got '" +
            std::string(master_port_env ? master_port_env : "29500") + "'"
        );
    }

    // -- Validate MASTER_ADDR reachability with TCP socket timeout --
    // Only check reachability from non-rank-0 processes (rank 0 may be
    // the master itself and hasn't started listening yet) and when the
    // address is not obviously local (a loopback connect would either
    // succeed trivially or block on a listener that has not started yet,
    // so the probe carries no useful signal there).
    if (rank != 0 && !is_local_address(master_addr)) {
        std::string reach_error = check_master_reachability(
            master_addr, master_port, /*timeout_ms=*/5000);
        if (!reach_error.empty()) {
            throw std::runtime_error("init_from_env: " + reach_error);
        }
    }

    init_process_group(backend, rank, world_size, master_addr, master_port);
}

// ============================================================================
// get_local_rank() Implementation
// ============================================================================

auto get_local_rank() -> int {
    const char* local_rank_env = std::getenv("LOCAL_RANK");
    if (!local_rank_env) {
        return 0;
    }
    return std::atoi(local_rank_env);
}

} // namespace tenzor::distributed
