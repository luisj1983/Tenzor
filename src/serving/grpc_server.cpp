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
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/dtype_from_string.hpp"

#include <grpcpp/grpcpp.h>
#include "tenzor_serving.grpc.pb.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

namespace tenzor {
namespace serving {

namespace {

// Convert DType enum to string
std::string dtype_to_string(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        default: return "float32";
    }
}

} // anonymous namespace

class TenzorServingImpl final : public tenzor::serving::TenzorServing::Service {
public:
    explicit TenzorServingImpl(ModelRepository& repo) : repository_(repo) {}

    grpc::Status Predict(grpc::ServerContext* /*context*/,
                         const PredictRequest* request,
                         PredictResponse* response) override {
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

            // Serialize output
            auto output_cont = output.contiguous();
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

    grpc::Status GetModelStatus(grpc::ServerContext* /*context*/,
                                const ModelStatusRequest* request,
                                ModelStatusResponse* response) override {
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

    grpc::Status LoadModel(grpc::ServerContext* /*context*/,
                           const LoadModelRequest* request,
                           LoadModelResponse* response) override {
        try {
            // Confine the client-supplied path to the repository root via the
            // SAME helper the HTTP handler uses (single source of truth), so a
            // client cannot make the server open/deserialize an arbitrary host
            // file through the untrusted-input graph/state loader. This catches
            // absolute paths, real ".." traversal components, and symlink
            // escapes — while still accepting filenames that merely contain the
            // substring ".." (e.g. "model..v2.tz"). The gRPC server has no
            // configured repository root, so it resolves relative to the
            // current directory.
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

    grpc::Status UnloadModel(grpc::ServerContext* /*context*/,
                             const UnloadModelRequest* request,
                             UnloadModelResponse* response) override {
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
};

// Standalone function to start a gRPC server (called from InferenceServer if configured)
void start_grpc_server(ModelRepository& repository, int port, std::atomic<bool>& running) {
    std::string server_address = "0.0.0.0:" + std::to_string(port);
    TenzorServingImpl service(repository);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
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
