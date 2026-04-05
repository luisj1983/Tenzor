/**
 * @file tenzor_serve.cpp
 * @brief CLI entry point for the Tenzor inference server
 *
 * Usage:
 *   tenzor_serve [model_repository_path] [http_port]
 *
 * Defaults to port 8080 and the current directory as the model repository.
 */

#include <tenzor/serving/server.hpp>

#include <cstdlib>
#include <iostream>
#include <csignal>

namespace {

volatile std::sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int /*sig*/) {
    g_shutdown_requested = 1;
}

} // namespace

int main(int argc, char** argv) {
    tenzor::serving::ServerConfig config;
    config.http_port = 8080;

    if (argc > 1) {
        config.model_repository_path = argv[1];
    }
    if (argc > 2) {
        config.http_port = std::atoi(argv[2]);
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    tenzor::serving::InferenceServer server(config);
    server.start();

    std::cout << "Tenzor inference server listening on port " << config.http_port << "\n";
    std::cout << "Press Ctrl+C to stop..." << std::endl;

    server.wait();

    return 0;
}
