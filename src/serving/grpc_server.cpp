/**
 * @file grpc_server.cpp
 * @brief gRPC inference server implementation
 *
 * Implements the TenzorServing gRPC service defined in tenzor_serving.proto.
 * Compiled only when TENZOR_HAS_GRPC is defined (requires -DTENZOR_BUILD_SERVING=ON + system gRPC).
 */

#ifdef TENZOR_HAS_GRPC

#include <filesystem>
#include "tenzor/serving/server.hpp"
#include "tenzor/serving/auth.hpp"
#include "tenzor/serving/rate_limiter.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/dtype_from_string.hpp"

#include <grpcpp/grpcpp.h>
#include "tenzor_serving.grpc.pb.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>

namespace tenzor {
namespace serving {

namespace {

// Convert DType enum to its canonical string name (round-trips through
// tenzor::dtype_from_string on the input side). Use the central dtype_name()
// so every dtype — bfloat16/int8/uint8/int16/bool/complex/... — is labeled
// correctly instead of being silently mislabeled as "float32".
std::string dtype_to_string(DType dt) {
    return std::string(tenzor::dtype_name(dt));
}

} // anonymous namespace

class TenzorServingImpl final : public tenzor::serving::TenzorServing::Service {
public:
    TenzorServingImpl(ModelRepository& repo, AuthConfig auth, RateLimitConfig rl)
        : repository_(repo),
          auth_(std::move(auth)),
          rate_limiter_(std::make_unique<TokenBucketRateLimiter>(rl)) {}

    grpc::Status Predict(grpc::ServerContext* context,
                         const PredictRequest* request,
                         PredictResponse* response) override {
        if (auto st = enforce_access(context); !st.ok()) return st;

        auto start = std::chrono::high_resolution_clock::now();

        auto model = repository_.get_model(request->model_name());
        if (!model || model->state.load() != ModelState::READY) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "Model not ready");
        }

        try {
            // Deserialize input tensor
            const auto& tensor_data = request->input();
            std::vector<int64_t> shape(tensor_data.shape().begin(), tensor_data.shape().end());
            DType dtype = tenzor::dtype_from_string(tensor_data.dtype());

            // Validate dims and buffer length before copying out of the
            // protobuf-owned buffer. The old code reinterpret_cast the bytes as
            // float and memcpy'd numel*sizeof(float) with NO length check, so a
            // client sending shape=[1e9] with a few bytes triggered a massive
            // OOB read. Honor the declared dtype instead of always reading float.
            int64_t numel = 1;
            for (int64_t d : shape) {
                if (d < 0) {
                    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                        "negative tensor dimension");
                }
                if (__builtin_mul_overflow(numel, d, &numel)) {
                    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                        "tensor shape too large");
                }
            }
            // The int64 numel guard above does NOT bound `numel * dtype_size`:
            // a shape with numel ~2^61 passes the int64 check, then the size_t
            // byte-size multiply wraps small, the length check passes for a tiny
            // payload, and tenzor::empty() allocates from the full shape
            // (bad_alloc DoS). Use a checked multiply and reject on overflow.
            size_t required = 0;
            if (__builtin_mul_overflow(static_cast<size_t>(numel),
                                       tenzor::dtype_size(dtype), &required)) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                    "tensor byte size overflows size_t");
            }
            if (tensor_data.data().size() < required) {
                return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                    "input data length does not match shape * dtype size");
            }
            auto host_input = tenzor::empty(shape, dtype, tenzor::Device::cpu());
            if (required > 0) {
                std::memcpy(host_input.storage()->data(),
                            tensor_data.data().data(), required);
            }
            auto input = (model->device.type == tenzor::Device::Type::CPU)
                             ? host_input
                             : host_input.to(model->device);

            // Run inference via batcher
            auto future = model->batcher->submit(input);
            auto output = future.get();

            // Serialize output. Pull the result back to host first: a
            // GPU-resident model returns device-storage output, and
            // protobuf set_data() below does a host-side memcpy from
            // storage()->data(). Without the copy to CPU that reads a raw
            // device pointer as host memory (garbage/crash). Mirror the HTTP
            // path (server.cpp). contiguous() after to(cpu) guarantees the
            // bytes are tight-packed; it is a cheap no-op when already on CPU.
            auto output_cont = (output.device().type == tenzor::Device::Type::CPU)
                                   ? output.contiguous()
                                   : output.to(tenzor::Device::cpu()).contiguous();
            auto* out_tensor = response->mutable_output();
            for (auto dim : output_cont.shape()) {
                out_tensor->add_shape(dim);
            }
            out_tensor->set_dtype(dtype_to_string(output_cont.dtype()));
            size_t data_bytes = output_cont.numel() * dtype_size(output_cont.dtype());
            // Serialize raw bytes through the untyped storage pointer (matching
            // the input side above). data<float>() throws DTypeException for any
            // dtype != Float32, which would break inference for Float64/Float16/
            // Int32/Int64 outputs.
            out_tensor->set_data(output_cont.storage()->data(), data_bytes);

            auto end = std::chrono::high_resolution_clock::now();
            float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
            response->set_latency_ms(latency_ms);

            // Update metrics. Mirror the HTTP path (server.cpp): besides the
            // running totals, push the sample into the latency ring buffer so
            // Prometheus p50/p95/p99 gauges are populated for gRPC-only and
            // mixed HTTP+gRPC deployments (otherwise the percentiles stay 0).
            auto& metrics = MetricsRegistry::instance().get_metrics(request->model_name());
            auto latency_us = static_cast<uint64_t>(latency_ms * 1000.0f);
            metrics.total_requests.fetch_add(1);
            metrics.total_latency_us.fetch_add(latency_us);
            metrics.record_latency(latency_us);

            return grpc::Status::OK;
        } catch (const std::exception& e) {
            auto& metrics = MetricsRegistry::instance().get_metrics(request->model_name());
            metrics.error_count.fetch_add(1);
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    grpc::Status GetModelStatus(grpc::ServerContext* context,
                                const ModelStatusRequest* request,
                                ModelStatusResponse* response) override {
        if (auto st = enforce_access(context); !st.ok()) return st;

        auto model = repository_.get_model(request->model_name());
        if (!model) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "Model not found");
        }

        response->set_model_name(request->model_name());
        response->set_model_version(model->version);

        auto state = model->state.load();
        switch (state) {
            case ModelState::LOADING: response->set_status("LOADING"); break;
            case ModelState::READY: response->set_status("READY"); break;
            case ModelState::UNLOADING: response->set_status("UNLOADING"); break;
            case ModelState::FAILED: response->set_status("FAILED"); break;
        }

        return grpc::Status::OK;
    }

    grpc::Status LoadModel(grpc::ServerContext* context,
                           const LoadModelRequest* request,
                           LoadModelResponse* response) override {
        if (auto st = enforce_access(context); !st.ok()) return st;

        try {
            // Confine the client-supplied path to the repository root via the
            // SAME helper the HTTP handler uses (single source of truth), so a
            // client cannot make the server open/deserialize an arbitrary host
            // file through the untrusted-input graph/state loader. This catches
            // absolute paths, real ".." traversal components, and symlink
            // escapes — while still accepting filenames that merely contain the
            // substring ".." (e.g. "model..v2.tz"). The gRPC server has no
            // configured repository root, so sanitize_repository_path now
            // fails CLOSED (rejects) rather than silently resolving against the
            // current working directory — a network client must not be able to
            // load arbitrary files under the server's CWD. Configure a model
            // repository root to enable LoadModel over gRPC.
            std::string mp;
            try {
                mp = tenzor::serving::sanitize_repository_path(
                    request->model_path(), /*root_dir=*/"");
            } catch (const std::invalid_argument& e) {
                response->set_success(false);
                response->set_message(e.what());
                return grpc::Status::OK;
            }
            repository_.load_model(request->model_name(), mp, Device::cpu());
            response->set_success(true);
            response->set_message("Model loaded successfully");
            return grpc::Status::OK;
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_message(e.what());
            return grpc::Status::OK;
        }
    }

    grpc::Status UnloadModel(grpc::ServerContext* context,
                             const UnloadModelRequest* request,
                             UnloadModelResponse* response) override {
        if (auto st = enforce_access(context); !st.ok()) return st;

        try {
            repository_.unload_model(request->model_name());
            response->set_success(true);
            response->set_message("Model unloaded successfully");
            return grpc::Status::OK;
        } catch (const std::exception& e) {
            response->set_success(false);
            response->set_message(e.what());
            return grpc::Status::OK;
        }
    }

    grpc::Status HealthCheck(grpc::ServerContext* /*context*/,
                             const HealthCheckRequest* /*request*/,
                             HealthCheckResponse* response) override {
        response->set_healthy(true);
        response->set_version("0.1.0");
        return grpc::Status::OK;
    }

private:
    ModelRepository& repository_;
    AuthConfig auth_;
    std::unique_ptr<TokenBucketRateLimiter> rate_limiter_;

    /// Resolve the client identity used for per-client rate limiting. Prefer the
    /// peer URI (e.g. "ipv4:1.2.3.4:5678") which the gRPC core fills from the
    /// transport; strip the trailing ":port" so all connections from one host
    /// share a bucket, matching the HTTP path's per-IP keying.
    static std::string client_id(grpc::ServerContext* context) {
        std::string peer = context->peer(); // "ipv4:host:port" / "ipv6:[..]:port"
        auto last_colon = peer.rfind(':');
        if (last_colon != std::string::npos) {
            return peer.substr(0, last_colon);
        }
        return peer;
    }

    /// Shared gate applied to every authenticated RPC: API-key validation
    /// (reusing validate_token/ct_eq from serving/auth.hpp) followed by token
    /// bucket rate limiting (serving/rate_limiter.hpp). This brings the gRPC
    /// transport to parity with the HTTP path, which previously enforced both
    /// while gRPC enforced neither.
    grpc::Status enforce_access(grpc::ServerContext* context) {
        // ---- Authentication ----
        // Fail CLOSED when auth is enabled but misconfigured (no keys): deny
        // rather than waving every RPC through unauthenticated.
        if (auth_.enabled && auth_.api_keys.empty()) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "authentication is enabled but no API keys are configured");
        }
        if (auth_.enabled && !auth_.api_keys.empty()) {
            // gRPC metadata keys are lowercased by the core; look up the
            // configured header in lowercase. The token is sent verbatim
            // (optionally "Bearer "-prefixed), matching the HTTP Authorization
            // header so a single client can use either transport.
            std::string key = auth_.header_name;
            std::transform(key.begin(), key.end(), key.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            const auto& md = context->client_metadata();
            auto it = md.find(key);
            if (it == md.end()) {
                return grpc::Status(grpc::StatusCode::UNAUTHENTICATED,
                                    "missing authentication metadata");
            }
            std::string header_value(it->second.data(), it->second.size());
            if (!validate_token(auth_, header_value)) {
                auto& metrics = MetricsRegistry::instance().get_metrics("_auth");
                metrics.error_count.fetch_add(1, std::memory_order_relaxed);
                return grpc::Status(grpc::StatusCode::PERMISSION_DENIED,
                                    "invalid api key");
            }
        }

        // ---- Rate limiting ----
        if (rate_limiter_ && !rate_limiter_->allow(client_id(context))) {
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                "rate limit exceeded");
        }

        return grpc::Status::OK;
    }
};

namespace {

/// Read an entire file into a string (used for TLS cert/key material).
std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("gRPC TLS: cannot open credential file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// Build server credentials from the configuration. TLS is used whenever a
/// certificate/key pair is configured (the secure default once auth is on);
/// otherwise we fall back to insecure credentials. When auth is enabled but no
/// TLS material is supplied we WARN loudly, because API keys travel in cleartext
/// metadata over an insecure channel.
std::shared_ptr<grpc::ServerCredentials> make_credentials(const ServerConfig& config) {
    if (!config.tls_cert_path.empty() && !config.tls_key_path.empty()) {
        grpc::SslServerCredentialsOptions ssl_opts;
        grpc::SslServerCredentialsOptions::PemKeyCertPair pair;
        pair.private_key = read_file(config.tls_key_path);
        pair.cert_chain = read_file(config.tls_cert_path);
        ssl_opts.pem_key_cert_pairs.push_back(std::move(pair));
        if (!config.tls_client_ca_path.empty()) {
            ssl_opts.pem_root_certs = read_file(config.tls_client_ca_path);
            ssl_opts.client_certificate_request =
                GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
        }
        return grpc::SslServerCredentials(ssl_opts);
    }
    if (config.enable_auth && !config.api_keys.empty()) {
        std::cerr << "[TenzorServing] WARNING: gRPC auth is enabled but no TLS "
                     "cert/key configured; API keys will be sent in cleartext. "
                     "Set tls_cert_path/tls_key_path."
                  << std::endl;
    }
    return grpc::InsecureServerCredentials();
}

} // namespace

// Standalone function to start a gRPC server (called from InferenceServer if configured)
void start_grpc_server(ModelRepository& repository, const ServerConfig& config,
                       std::atomic<bool>& running) {
    std::string server_address = "0.0.0.0:" + std::to_string(config.grpc_port);

    // Mirror the HTTP path's security posture: validate the same API keys and
    // apply the same token-bucket rate limit inside every authenticated RPC.
    AuthConfig auth;
    auth.enabled = config.enable_auth;
    auth.api_keys = config.api_keys;
    auth.header_name = config.auth_header;

    RateLimitConfig rl;
    rl.enabled = config.enable_rate_limit;
    rl.requests_per_second = config.rate_limit_rps;
    rl.burst_size = config.rate_limit_burst;

    TenzorServingImpl service(repository, std::move(auth), rl);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, make_credentials(config));
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    if (!server) {
        // BuildAndStart returns null when the port can't be bound (in use, bad
        // address, bad TLS credentials). Fail loudly instead of printing a
        // false "listening" message and then dereferencing a null server.
        throw std::runtime_error(
            "[TenzorServing] failed to start gRPC server on " + server_address +
            " (port in use, invalid address, or bad TLS credentials)");
    }
    std::cout << "[TenzorServing] gRPC server listening on " << server_address << std::endl;

    // Wait until signalled to stop
    while (running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    server->Shutdown();
}

} // namespace serving
} // namespace tenzor

#endif // TENZOR_HAS_GRPC
