/**
 * @file server.cpp
 * @brief Implementation of inference serving infrastructure
 */

#include "tenzor/serving/server.hpp"
#include "tenzor/serving/auth.hpp"
#include "tenzor/jit/serialization.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/dtype.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <filesystem>
#include <stdexcept>
#include <vector>
#ifdef TENZOR_HAS_HTTPLIB
#include <httplib.h>
#endif

// G.12: vendored nlohmann/json (third_party/nlohmann/json.hpp). Used to
// replace the previous ad-hoc std::string::find('"') request parsing in
// the /v1/models/.../{load,predict} and /v1/experiments endpoints.
#include <nlohmann/json.hpp>

// G.12: explicit build-config gate. When TENZOR_BUILD_SERVING is ON,
// the serving subsystem requires at least one HTTP/gRPC transport. We
// don't fire when serving is OFF because this TU is unconditionally
// compiled into tenzor_core (the InferenceServer class needs to exist
// even in trimmed builds for header / ABI compatibility) — the runtime
// guard in serve_loop() handles the OFF case with a typed exception.
#if defined(TENZOR_BUILD_SERVING) && !defined(TENZOR_HAS_HTTPLIB) && !defined(TENZOR_HAS_GRPC)
static_assert(false,
    "tenzor::serving requires at least one HTTP/gRPC transport when "
    "TENZOR_BUILD_SERVING=ON. The httplib FetchContent should run "
    "automatically; if it didn't, check src/CMakeLists.txt.");
#endif

namespace tenzor {
namespace serving {

// ============================================================================
// Path sandboxing (shared by HTTP and gRPC LoadModel handlers)
// ============================================================================

auto sanitize_repository_path(const std::string& requested,
                              const std::string& root_dir) -> std::string {
    namespace fs = std::filesystem;
    fs::path req(requested);
    if (req.is_absolute()) {
        throw std::invalid_argument(
            "model_path must be a relative path within the model repository");
    }
    // Reject only REAL ".." path components — not filenames that merely contain
    // the substring "..", e.g. "model..v2.tz" (which has no traversal).
    for (const auto& part : req) {
        if (part == "..") {
            throw std::invalid_argument(
                "model_path must not contain a '..' path component");
        }
    }
    // Fail closed: refuse to resolve model paths when no repository root is
    // configured. Previously an empty root silently defaulted to "." (the
    // current working directory), letting a network client load and deserialize
    // ANY parseable file under the server's CWD subtree. A model repository must
    // be explicitly configured.
    if (root_dir.empty()) {
        throw std::invalid_argument(
            "model repository root is not configured; refusing to resolve model "
            "paths against the current working directory");
    }
    fs::path root = fs::path(root_dir);
    fs::path root_canon = fs::weakly_canonical(root);
    fs::path resolved = fs::weakly_canonical(root / req);
    // A raw string-prefix compare has no path-separator boundary, so a root of
    // "/srv/models" would accept "/srv/models-evil/x". Use fs::relative and
    // reject if it escapes (error, empty, or first component ".."), which also
    // accounts for symlinks resolved by weakly_canonical above.
    std::error_code ec;
    fs::path rel = fs::relative(resolved, root_canon, ec);
    bool escapes = ec || rel.empty() ||
                   (rel.begin() != rel.end() && *rel.begin() == "..");
    if (escapes) {
        throw std::invalid_argument("model_path escapes the model repository");
    }
    return resolved.string();
}

// ============================================================================
// DynamicBatcher
// ============================================================================

DynamicBatcher::DynamicBatcher(std::shared_ptr<jit::CompiledModule> model,
                               BatchConfig config,
                               std::string model_name)
    : model_(std::move(model)),
      config_(std::move(config)),
      model_name_(std::move(model_name)) {}

DynamicBatcher::~DynamicBatcher() {
    stop();
}

auto DynamicBatcher::submit(Tensor input) -> std::future<Tensor> {
    auto req = std::make_shared<InferRequest>(std::move(input));
    auto future = req->result.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!running_.load(std::memory_order_acquire)) {
            // Batcher already stopped (e.g. concurrent model unload): fail the
            // request now instead of enqueueing it where the exited batch loop
            // would never process it, leaving future.get() to block forever.
            req->result.set_exception(std::make_exception_ptr(std::runtime_error(
                "inference batcher is not running (model unloaded)")));
            return future;
        }
        queue_.push(std::move(req));
    }
    queue_cv_.notify_one();

    return future;
}

auto DynamicBatcher::start() -> void {
    running_.store(true, std::memory_order_release);
    batch_thread_ = std::thread([this] { batch_loop(); });
}

auto DynamicBatcher::stop() -> void {
    running_.store(false, std::memory_order_release);
    queue_cv_.notify_all();
    if (batch_thread_.joinable()) {
        batch_thread_.join();
    }
    // Defensively fail any request still queued: the batch loop has exited and
    // will never process it, so future.get() on it would block forever.
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!queue_.empty()) {
        auto req = std::move(queue_.front());
        queue_.pop();
        try {
            req->result.set_exception(std::make_exception_ptr(std::runtime_error(
                "inference batcher stopped before request was processed")));
        } catch (...) {
            // set_exception throws only if the promise was already satisfied.
        }
    }
}

auto DynamicBatcher::batch_loop() -> void {
    // Loop until stopped AND drained (see the inner break): a plain
    // `while (running_)` exited with requests still queued (> max_batch_size, or
    // submitted right before stop), permanently hanging their futures.
    while (true) {
        std::vector<std::shared_ptr<InferRequest>> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait for either batch full or timeout
            queue_cv_.wait_for(lock,
                std::chrono::microseconds(config_.max_latency_us),
                [this] {
                    return !queue_.empty() || !running_.load(std::memory_order_acquire);
                });

            if (!running_.load(std::memory_order_acquire) && queue_.empty()) {
                break;
            }

            // Drain up to max_batch_size requests
            while (!queue_.empty() &&
                   static_cast<int32_t>(batch.size()) < config_.max_batch_size) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop();
            }
        }

        if (!batch.empty()) {
            execute_batch(batch);
        }
    }
}

auto DynamicBatcher::execute_batch(
    std::vector<std::shared_ptr<InferRequest>>& batch) -> void {
    // Record dynamic-batching metrics for this group. batch_loop() only calls
    // execute_batch() with a non-empty batch, and each call corresponds to
    // exactly one batching decision, so total_batch_count is incremented once
    // here. total_batch_size accumulates the actual number of rows fed through
    // the model (each request may carry >1 leading-dim row, i.e. a client-side
    // batch), so format_prometheus()'s batch_sum/batch_count yields the true
    // mean rows-per-batch rather than mean requests-per-batch. A request with a
    // scalar (rank-0) input contributes 1 to the row count. Recorded for both
    // the batched and per-request fallback paths because the batching decision
    // (and therefore the metric) is independent of how the group ultimately
    // executes. Skipped when no model name was supplied (no metrics handle to
    // attribute the batch to).
    if (!model_name_.empty()) {
        uint64_t row_count = 0;
        for (const auto& req : batch) {
            const auto& in_shape = req->input.shape();
            row_count += in_shape.empty()
                             ? 1
                             : static_cast<uint64_t>(in_shape[0]);
        }
        auto& metrics = MetricsRegistry::instance().get_metrics(model_name_);
        metrics.total_batch_count.fetch_add(1, std::memory_order_relaxed);
        metrics.total_batch_size.fetch_add(row_count, std::memory_order_relaxed);
    }

    // Run every request in the batch on its own forward pass. Each request's
    // failure is isolated to that request so a single bad input never poisons
    // co-batched requests. Used both as the deliberate fallback path and as the
    // safety net whenever a batched (concatenated) execution cannot succeed for
    // the whole group (incompatible input shapes/dtypes/devices that make cat
    // throw, a model that collapses the batch dimension, etc.).
    auto run_per_request =
        [this](std::vector<std::shared_ptr<InferRequest>>& reqs) {
            for (auto& req : reqs) {
                try {
                    tenzor::Variable iv(req->input, false);
                    req->result.set_value(model_->forward(iv).tensor());
                } catch (...) {
                    try {
                        req->result.set_exception(std::current_exception());
                    } catch (...) {}
                }
            }
        };

    try {
        if (batch.size() == 1) {
            // Single request: no batching overhead
            tenzor::Variable input_var(batch[0]->input, false);
            auto output = model_->forward(input_var);
            batch[0]->result.set_value(output.tensor());
            return;
        }

        // Batch multiple requests: concatenate along dim 0, single forward,
        // split. Independent clients can submit requests whose non-batch dims,
        // dtype, or device differ; tenzor::cat then throws, and the model may
        // also fail on the combined batch or collapse the batch dimension. Any
        // of those is a property of the *group*, not of an individual request,
        // so the whole batched attempt is wrapped in its own try and degrades
        // to per-request execution rather than failing every co-batched
        // request. Per-request execution must produce identical results to
        // batched execution, so availability/correctness never depend on how
        // requests happen to be co-scheduled.
        std::vector<Tensor> inputs;
        inputs.reserve(batch.size());
        // Record each request's leading-dim row count BEFORE cat so the
        // batched output can be split back by per-request section sizes.
        // A request may legitimately carry >1 row (a client-side batch); a
        // fixed split_size=1 would mis-map rows to requests, dropping rows
        // or raising a spurious split mismatch.
        std::vector<int64_t> row_counts;
        row_counts.reserve(batch.size());
        bool inputs_ok = true;
        for (auto& req : batch) {
            const auto& in_shape = req->input.shape();
            if (in_shape.empty()) {
                // A scalar input cannot participate in dim-0 batching; fall
                // back so this request (and its co-batched peers) still run.
                inputs_ok = false;
                break;
            }
            row_counts.push_back(in_shape[0]);
            inputs.push_back(req->input);
        }

        if (inputs_ok) {
            try {
                // Concatenate all inputs along batch dimension. Throws if the
                // requests have incompatible non-batch shapes, dtypes, or
                // devices.
                auto batched_input = tenzor::cat(inputs, 0);

                // Single forward pass on the full batch
                tenzor::Variable batched_var(batched_input, false);
                auto batched_output = model_->forward(batched_var);

                // split_with_sizes assumes the model preserved the dim-0 row
                // count. Models that pool/reduce the batch dimension (or emit a
                // fixed-size output) break that assumption: the split would
                // throw. Validate the row count first and, on mismatch, fall
                // back to per-request execution so batching never changes
                // correctness/availability versus unbatched.
                int64_t total_rows = 0;
                for (int64_t rc : row_counts) total_rows += rc;
                const auto& out_shape = batched_output.tensor().shape();
                bool batch_dim_preserved =
                    !out_shape.empty() && out_shape[0] == total_rows;

                if (batch_dim_preserved) {
                    // Split the batched output by each request's actual row
                    // count so every request gets its full sub-tensor back.
                    auto split_outputs = tenzor::split_with_sizes(
                        batched_output.tensor(), row_counts, /*dim=*/0);
                    if (split_outputs.size() == batch.size()) {
                        for (size_t i = 0; i < batch.size(); ++i) {
                            batch[i]->result.set_value(split_outputs[i]);
                        }
                        return;
                    }
                    // Unexpected split arity — treat as a batched-path failure
                    // and re-run each request individually below.
                }
            } catch (...) {
                // Batched execution failed for the group as a whole (cat shape/
                // dtype/device mismatch, forward error on the combined batch,
                // split failure, ...). Do NOT poison the batch — fall through
                // to isolated per-request execution.
            }
        }

        // Fallback: execute each request independently.
        run_per_request(batch);
    } catch (...) {
        // Last-resort guard: anything outside the per-request loop (e.g. an
        // allocation failure) must still surface to callers rather than leaving
        // their futures unfulfilled.
        for (auto& req : batch) {
            try {
                req->result.set_exception(std::current_exception());
            } catch (...) {}
        }
    }
}

// ============================================================================
// ModelRepository
// ============================================================================

auto ModelRepository::load_model(const std::string& name, const std::string& path,
                                  Device device, BatchConfig batch_config) -> void {
    auto entry = std::make_shared<ModelEntry>();
    entry->name = name;
    entry->device = device;
    entry->state.store(ModelState::LOADING, std::memory_order_release);

    try {
        // Load the compiled module via CompiledModule::load() rather than the
        // raw jit::load_graph()+set_graph() pair (R1-03): set_graph() is a bare
        // `graph_ = std::move(g)` with no other bookkeeping, so a model loaded
        // that way never gets loaded_=true, never re-applies a dynamic-dims
        // configuration serialized via mark_dynamic_dims() (F031 — silently
        // reverting a dynamic-batch model to its trace-time concrete shape),
        // and — since throw_if_loaded_shape_mismatch short-circuits on
        // !loaded_ — never gets the ordinary shape/dtype/device compatibility
        // guard either (R1-02). CompiledModule::load() gives this serving path
        // the same safety net the documented save()/load() API already has.
        auto module = jit::CompiledModule::load(path);

        // Fail fast, at load time, instead of deferring to the first
        // inference request: a CompiledModule loaded from disk cannot
        // retrace (its constants/weights are baked at trace-time device --
        // see CompiledModule::throw_if_loaded_shape_mismatch), so serving
        // it on a DIFFERENT device than it was traced/saved on would
        // otherwise appear to load successfully and only fail confusingly
        // once the first real request arrived with an input placed on
        // `device` (the exact device tensor_from_json places every
        // inference request's input on -- server.cpp's predict handler).
        // A model with dynamic dims configured before it was saved can
        // legitimately retrace, so this check only applies to the common
        // (non-dynamic) case, matching throw_if_loaded_shape_mismatch's own
        // bypass condition.
        if (!module->has_dynamic_shapes()) {
            for (const auto& loaded_dev : module->loaded_devices()) {
                if (loaded_dev.type != device.type || loaded_dev.index != device.index) {
                    throw std::invalid_argument(
                        "ModelRepository::load_model: model '" + name + "' at '" +
                        path + "' was traced/saved for device " +
                        loaded_dev.to_string() + " but was requested on " +
                        device.to_string() + "; a loaded model cannot be "
                        "retraced to a different device. Re-trace/re-export "
                        "the model on the target device, or load it with "
                        "device=" + loaded_dev.to_string() + ".");
                }
            }
        }

        module->optimize_for_inference();

        entry->module = std::move(module);
        // Pass the serving name so the batcher can attribute its per-batch
        // metrics (total_batch_count / total_batch_size) to the same
        // ModelMetrics handle the request handlers update under this name.
        entry->batcher = std::make_unique<DynamicBatcher>(
            entry->module, batch_config, name);
        entry->batcher->start();
        entry->state.store(ModelState::READY, std::memory_order_release);
    } catch (...) {
        entry->state.store(ModelState::FAILED, std::memory_order_release);
        throw;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    models_[name] = std::move(entry);
}

auto ModelRepository::unload_model(const std::string& name) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(name);
    if (it != models_.end()) {
        it->second->state.store(ModelState::UNLOADING, std::memory_order_release);
        if (it->second->batcher) {
            it->second->batcher->stop();
        }
        models_.erase(it);
    }
}

auto ModelRepository::get_model(const std::string& name) -> std::shared_ptr<ModelEntry> {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(name);
    return it != models_.end() ? it->second : nullptr;
}

auto ModelRepository::list_models() const -> std::vector<std::string> {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (auto& [name, _] : models_) {
        names.push_back(name);
    }
    return names;
}

// ============================================================================
// MetricsRegistry
// ============================================================================

auto MetricsRegistry::instance() -> MetricsRegistry& {
    static MetricsRegistry registry;
    return registry;
}

auto MetricsRegistry::get_metrics(const std::string& model_name) -> ModelMetrics& {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metrics_.find(model_name);
    if (it == metrics_.end()) {
        metrics_[model_name] = std::make_unique<ModelMetrics>();
        return *metrics_[model_name];
    }
    return *it->second;
}

auto MetricsRegistry::format_prometheus() const -> std::string {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;

    // Escape Prometheus label values: model names come from the URL and may
    // contain quotes/backslashes/newlines that would corrupt the exposition
    // format or inject arbitrary metric lines.
    auto escape_label = [](const std::string& in) {
        std::string out;
        out.reserve(in.size());
        for (char c : in) {
            if (c == '\\' || c == '"') { out.push_back('\\'); out.push_back(c); }
            else if (c == '\n') { out += "\\n"; }
            else { out.push_back(c); }
        }
        return out;
    };
    for (auto& [name, m] : metrics_) {
        const std::string esc_name = escape_label(name);
        auto total = m->total_requests.load(std::memory_order_relaxed);
        auto latency = m->total_latency_us.load(std::memory_order_relaxed);
        auto errors = m->error_count.load(std::memory_order_relaxed);
        auto batches = m->total_batch_count.load(std::memory_order_relaxed);

        ss << "tenzor_requests_total{model=\"" << esc_name << "\"} " << total << "\n";
        ss << "tenzor_errors_total{model=\"" << esc_name << "\"} " << errors << "\n";
        if (total > 0) {
            ss << "tenzor_latency_avg_us{model=\"" << esc_name << "\"} "
               << (latency / total) << "\n";
        }
        if (batches > 0) {
            auto batch_sum = m->total_batch_size.load(std::memory_order_relaxed);
            ss << "tenzor_batch_size_avg{model=\"" << esc_name << "\"} "
               << (batch_sum / batches) << "\n";
        }

        // Compute latency percentiles from ring buffer
        if (total > 0) {
            auto count = std::min(total, static_cast<uint64_t>(ModelMetrics::kLatencyWindowSize));
            std::vector<uint64_t> latencies(count);
            for (uint64_t i = 0; i < count; ++i) {
                latencies[i] = m->latency_window[i].load(std::memory_order_relaxed);
            }
            std::sort(latencies.begin(), latencies.end());
            auto p50 = latencies[static_cast<size_t>(count * 0.50)];
            auto p95 = latencies[static_cast<size_t>(count * 0.95)];
            auto p99 = latencies[static_cast<size_t>(std::min(count - 1, static_cast<uint64_t>(count * 0.99)))];
            ss << "tenzor_latency_p50_us{model=\"" << esc_name << "\"} " << p50 << "\n";
            ss << "tenzor_latency_p95_us{model=\"" << esc_name << "\"} " << p95 << "\n";
            ss << "tenzor_latency_p99_us{model=\"" << esc_name << "\"} " << p99 << "\n";
        }
    }

    return ss.str();
}

// ============================================================================
// InferenceServer
// ============================================================================

InferenceServer::InferenceServer(ServerConfig config)
    : config_(std::move(config)) {}

InferenceServer::~InferenceServer() {
    stop();
}

auto InferenceServer::start() -> void {
    running_.store(true, std::memory_order_release);
    server_thread_ = std::thread([this] { serve_loop(); });
    std::cout << "[TenzorServing] Server started on port " << config_.http_port << std::endl;
}

auto InferenceServer::stop() -> void {
    running_.store(false, std::memory_order_release);
#ifdef TENZOR_HAS_HTTPLIB
    // Unblock svr.listen() in serve_loop(). Clearing running_ alone is not
    // enough: httplib::Server::listen() does not poll it, so the serve thread
    // would block forever and join() below would hang. We must call the
    // server's own stop().
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(server_mutex_);
            if (http_server_ == nullptr) {
                // serve_loop() either has not registered the handle yet (it
                // will observe running_==false under this same mutex and skip
                // listen()), or has already returned. Nothing to stop.
                break;
            }
            if (http_server_->is_running()) {
                // listen() has entered its accept loop, so stop() closes the
                // listening socket and unblocks it. Safe under the mutex: the
                // local svr in serve_loop() cannot be destroyed until listen()
                // returns and serve_loop() re-acquires this mutex to clear the
                // handle, which it cannot do while we hold it here.
                http_server_->stop();
                break;
            }
        }
        // Race: the handle is registered but listen() has not yet reached its
        // accept loop, so stop() would be a no-op and listen() would then block
        // forever. Release the mutex (so serve_loop() can make progress) and
        // retry. Bounded: listen() sets is_running() right after we registered
        // the handle, or returns (e.g. a bind failure) and clears the handle,
        // which the null check above then catches.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    std::cout << "[TenzorServing] Server stopped" << std::endl;
}

auto InferenceServer::wait() -> void {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

auto InferenceServer::wait_for(std::chrono::milliseconds timeout) -> bool {
    // Audit-11 QQ.19: bounded wait. We do not have a condition_variable on
    // running_ in this path, so poll on the atomic. Resolution is bounded by
    // a 10 ms tick — sufficient for KeyboardInterrupt latency, well below
    // any realistic server-shutdown deadline.
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + timeout;
    const auto tick = std::chrono::milliseconds(10);
    while (running_.load(std::memory_order_acquire)) {
        if (clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(tick);
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    return true;
}

auto InferenceServer::has_http_transport() -> bool {
#ifdef TENZOR_HAS_HTTPLIB
    return true;
#else
    return false;
#endif
}

auto InferenceServer::serve_loop() -> void {
#ifdef TENZOR_HAS_HTTPLIB
    using nlohmann::json;
    httplib::Server svr;

    // ---------- Helpers (JSON dtype <-> Tensor) ----------
    auto dtype_from_request = [](const std::string& s) -> DType {
        // Accept the canonical names emitted by dtype_name() so a client
        // round-trips an output_dtype string back into a predict request.
        if (s == "float32" || s == "f32") return DType::Float32;
        if (s == "float64" || s == "f64") return DType::Float64;
        if (s == "float16" || s == "f16" || s == "half") return DType::Float16;
        if (s == "bfloat16" || s == "bf16") return DType::BFloat16;
        if (s == "int32"   || s == "i32") return DType::Int32;
        if (s == "int64"   || s == "i64") return DType::Int64;
        if (s == "int8"    || s == "i8")  return DType::Int8;
        if (s == "uint8"   || s == "u8")  return DType::UInt8;
        if (s == "bool")                  return DType::Bool;
        throw std::runtime_error("unsupported dtype in request: " + s);
    };

    // Build a host-side Tensor of the given dtype/shape from a JSON array.
    // Numeric arrays are accepted as plain JSON numbers; the request always
    // travels in textual JSON (binary payloads use the gRPC path).
    auto tensor_from_json = [&](const json& shape_j, const json& data_j,
                                DType dtype, Device device) -> Tensor {
        if (!shape_j.is_array()) {
            throw std::runtime_error("\"shape\" must be a JSON array");
        }
        if (!data_j.is_array()) {
            throw std::runtime_error("\"data\" must be a JSON array");
        }
        std::vector<int64_t> shape;
        shape.reserve(shape_j.size());
        int64_t expected = 1;
        for (const auto& d : shape_j) {
            auto v = d.get<int64_t>();
            // Reject negative dimensions outright: the previous code mapped them
            // to 0, which let a crafted shape (e.g. [-1]) pass the length check
            // and then allocate a tensor with a negative dim.
            if (v < 0) {
                throw std::runtime_error("shape dimensions must be non-negative");
            }
            shape.push_back(v);
            // Guard the element-count product against int64 overflow; a wrapped
            // (small/negative) product would otherwise pass the length check and
            // under-allocate, leading to OOM/DoS or an out-of-bounds write.
            if (__builtin_mul_overflow(expected, v, &expected)) {
                throw std::runtime_error("shape product overflows int64");
            }
        }
        if (static_cast<int64_t>(data_j.size()) != expected) {
            throw std::runtime_error("data length does not match shape product");
        }

        // Allocate on host first, then move to target device via .to() if
        // needed. Avoids touching GPU memory from host JSON deserialization
        // code paths.
        auto host = tenzor::empty(shape, dtype, Device::cpu());

        auto write_typed = [&](auto* out_ptr, auto convert) {
            using T = std::remove_pointer_t<decltype(out_ptr)>;
            for (size_t i = 0; i < data_j.size(); ++i) {
                out_ptr[i] = static_cast<T>(convert(data_j[i]));
            }
        };

        switch (dtype) {
            case DType::Float32:
                write_typed(host.data<float>(),
                            [](const json& v){ return v.get<double>(); });
                break;
            case DType::Float64:
                write_typed(host.data<double>(),
                            [](const json& v){ return v.get<double>(); });
                break;
            case DType::Int32:
                write_typed(host.data<int32_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::Int64:
                write_typed(host.data<int64_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::Int8:
                write_typed(host.data<int8_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::UInt8:
                write_typed(host.data<uint8_t>(),
                            [](const json& v){ return v.get<int64_t>(); });
                break;
            case DType::Bool:
                write_typed(host.data<bool>(),
                            [](const json& v){ return v.get<bool>(); });
                break;
            case DType::Float16: {
                auto* out = host.data<Float16>();
                for (size_t i = 0; i < data_j.size(); ++i) {
                    out[i] = Float16(static_cast<float>(data_j[i].get<double>()));
                }
                break;
            }
            case DType::BFloat16: {
                auto* out = host.data<BFloat16>();
                for (size_t i = 0; i < data_j.size(); ++i) {
                    out[i] = BFloat16(static_cast<float>(data_j[i].get<double>()));
                }
                break;
            }
            default:
                throw std::runtime_error(
                    std::string("dtype not supported by JSON predict path: ") +
                    std::string(dtype_name(dtype)));
        }

        // Move to target device (no-op if already on CPU).
        if (device.type != Device::Type::CPU) {
            return host.to(device);
        }
        return host;
    };

    // Serialize a host Tensor (assumed contiguous) into the JSON output_data
    // array. Mirrors the dtype switch above.
    auto json_from_tensor = [](const Tensor& t) -> json {
        json arr = json::array();
        const auto n = static_cast<size_t>(t.numel());
        switch (t.dtype()) {
            case DType::Float32: {
                const auto* p = t.data<float>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Float64: {
                const auto* p = t.data<double>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Int32: {
                const auto* p = t.data<int32_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Int64: {
                const auto* p = t.data<int64_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Int8: {
                const auto* p = t.data<int8_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(static_cast<int>(p[i]));
                break;
            }
            case DType::UInt8: {
                const auto* p = t.data<uint8_t>();
                for (size_t i = 0; i < n; ++i) arr.push_back(static_cast<unsigned>(p[i]));
                break;
            }
            case DType::Bool: {
                const auto* p = t.data<bool>();
                for (size_t i = 0; i < n; ++i) arr.push_back(p[i]);
                break;
            }
            case DType::Float16: {
                const auto* p = t.data<Float16>();
                for (size_t i = 0; i < n; ++i) {
                    arr.push_back(static_cast<float>(p[i]));
                }
                break;
            }
            case DType::BFloat16: {
                const auto* p = t.data<BFloat16>();
                for (size_t i = 0; i < n; ++i) {
                    arr.push_back(static_cast<float>(p[i]));
                }
                break;
            }
            default:
                throw std::runtime_error(
                    std::string("dtype not supported by JSON predict response: ") +
                    std::string(dtype_name(t.dtype())));
        }
        return arr;
    };

    auto json_error = [](httplib::Response& res, int status,
                         const std::string& msg) {
        json j;
        j["error"] = msg;
        res.status = status;
        res.set_content(j.dump(), "application/json");
    };

    // ---------- Authentication middleware ----------
    // Gate on enable_auth alone and fail CLOSED: if auth is enabled but no API
    // keys are configured, deny every request rather than serving the whole
    // server unauthenticated (the previous `&& !api_keys.empty()` condition
    // silently disabled auth on misconfiguration).
    if (config_.enable_auth) {
        svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
            // Skip auth for health and metrics endpoints
            if (req.path == "/health" || req.path == "/metrics") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            if (config_.api_keys.empty()) {
                res.status = 503;
                res.set_content(
                    json{{"error", "authentication is enabled but no API keys are configured"}}.dump(),
                    "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            auto it = req.headers.find(config_.auth_header);
            if (it == req.headers.end()) {
                res.status = 401;
                res.set_content(json{{"error","missing authentication header"}}.dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            // Extract Bearer token and validate via the SHARED helpers in
            // serving/auth.hpp (single source of truth) so the live path and
            // the reusable header cannot drift on the Bearer-strip boundary or
            // the constant-time comparison.
            std::string token = strip_bearer(it->second);
            // Constant-time comparison against every configured key (no early
            // break) so response time does not leak how many leading bytes of a
            // guessed key are correct.
            bool valid = false;
            for (const auto& key : config_.api_keys) {
                valid |= ct_eq(token, key);
            }
            if (!valid) {
                res.status = 403;
                res.set_content(json{{"error","invalid api key"}}.dump(), "application/json");
                auto& metrics = MetricsRegistry::instance().get_metrics("_auth");
                metrics.error_count.fetch_add(1, std::memory_order_relaxed);
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    // ---------- Rate limiting (token bucket per client IP) ----------
    struct TokenBucket {
        double tokens;
        std::chrono::steady_clock::time_point last_refill;
    };
    std::mutex rate_limit_mutex;
    std::unordered_map<std::string, TokenBucket> rate_limit_buckets;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status","ok"},{"version","0.1.0"}}.dump(),
                        "application/json");
    });

    // Prometheus metrics
    svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(MetricsRegistry::instance().format_prometheus(), "text/plain");
    });

    // List models
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = repository_.list_models();
        json j;
        j["models"] = models;
        res.set_content(j.dump(), "application/json");
    });

    // Model status
    svr.Get(R"(/v1/models/([^/]+)/status)", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        auto model = repository_.get_model(name);
        if (!model) {
            res.status = 404;
            res.set_content(json{{"error","model not found"}}.dump(), "application/json");
            return;
        }
        auto state = model->state.load();
        const char* state_str = "UNKNOWN";
        switch (state) {
            case ModelState::LOADING: state_str = "LOADING"; break;
            case ModelState::READY: state_str = "READY"; break;
            case ModelState::UNLOADING: state_str = "UNLOADING"; break;
            case ModelState::FAILED: state_str = "FAILED"; break;
        }
        json j;
        j["model_name"] = name;
        j["version"]    = model->version;
        j["status"]     = state_str;
        res.set_content(j.dump(), "application/json");
    });

    // Load model
    svr.Post(R"(/v1/models/([^/]+)/load)", [this, &json_error](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        // Expect JSON body: {"model_path": "/path/to/model", "device": "cpu"}
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error& e) {
            json_error(res, 400, std::string("malformed JSON: ") + e.what());
            return;
        }
        if (!body.contains("model_path") || !body["model_path"].is_string()) {
            json_error(res, 400, "missing or non-string model_path");
            return;
        }
        std::string model_path = body["model_path"].get<std::string>();

        // Sandbox the path: a client must not be able to make the server open and
        // deserialize an ARBITRARY host file (the graph/state loader is itself an
        // untrusted-input parser). Confine the resolved path inside the
        // configured repository root via the shared helper (same policy as the
        // gRPC LoadModel handler — single source of truth).
        try {
            model_path = sanitize_repository_path(model_path,
                                                  config_.model_repository_path);
        } catch (const std::invalid_argument& e) {
            json_error(res, 400, e.what());
            return;
        }

        Device target_device = Device::cpu();
        if (body.contains("device") && body["device"].is_string()) {
            target_device = Device::from_string(body["device"].get<std::string>());
        }

        try {
            repository_.load_model(name, model_path, target_device);
            res.set_content(json{{"success",true}}.dump(), "application/json");
        } catch (const std::exception& e) {
            json j;
            j["success"] = false;
            j["message"] = e.what();
            res.status = 500;
            res.set_content(j.dump(), "application/json");
        }
    });

    // Unload model
    svr.Delete(R"(/v1/models/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        try {
            repository_.unload_model(name);
            res.set_content(json{{"success",true}}.dump(), "application/json");
        } catch (const std::exception& e) {
            json j;
            j["success"] = false;
            j["message"] = e.what();
            res.status = 500;
            res.set_content(j.dump(), "application/json");
        }
    });

    // Predict — JSON in, JSON out.
    // Request:  {"shape":[...], "dtype":"float32", "data":[...]}
    // Response: {"output_shape":[...], "output_dtype":"...", "output_data":[...]}
    svr.Post(R"(/v1/models/([^/]+)/predict)",
             [this, &rate_limit_mutex, &rate_limit_buckets,
              &dtype_from_request, &tensor_from_json, &json_from_tensor,
              &json_error]
             (const httplib::Request& req, httplib::Response& res) {
        // Rate limiting check
        if (config_.enable_rate_limit) {
            auto client_ip = req.remote_addr;
            std::lock_guard<std::mutex> lock(rate_limit_mutex);
            auto now = std::chrono::steady_clock::now();
            // Bound memory: an attacker spoofing many source IPs would otherwise
            // grow this map without limit. When it gets large, evict buckets that
            // are idle past a TTL (they will have fully refilled anyway).
            if (rate_limit_buckets.size() > 10000) {
                const auto ttl = std::chrono::seconds(300);
                for (auto it = rate_limit_buckets.begin(); it != rate_limit_buckets.end();) {
                    if (now - it->second.last_refill > ttl) {
                        it = rate_limit_buckets.erase(it);
                    } else {
                        ++it;
                    }
                }
                // The TTL sweep only reclaims *idle* buckets. An attacker
                // spoofing one source IP per request every <TTL keeps every
                // bucket "recently refilled", so the sweep frees nothing and the
                // map grows without bound — an active-id memory-exhaustion DoS.
                // Enforce a hard absolute cap by evicting the least-recently-
                // refilled entries down to kHardMaxBuckets. Evicting a stale
                // bucket only resets its tokens to full burst, so it can never
                // let a throttled client exceed its rate. Matches the library
                // limiter (include/tenzor/serving/rate_limiter.hpp).
                constexpr std::size_t kHardMaxBuckets = 100000;
                if (rate_limit_buckets.size() > kHardMaxBuckets) {
                    std::vector<std::chrono::steady_clock::time_point> refills;
                    refills.reserve(rate_limit_buckets.size());
                    for (const auto& kv : rate_limit_buckets) {
                        refills.push_back(kv.second.last_refill);
                    }
                    const std::size_t to_evict =
                        rate_limit_buckets.size() - kHardMaxBuckets;
                    std::nth_element(refills.begin(), refills.begin() + to_evict,
                                     refills.end());
                    const auto cutoff = refills[to_evict];
                    std::size_t evicted = 0;
                    for (auto it = rate_limit_buckets.begin();
                         it != rate_limit_buckets.end() && evicted < to_evict;) {
                        if (it->second.last_refill < cutoff) {
                            it = rate_limit_buckets.erase(it);
                            ++evicted;
                        } else {
                            ++it;
                        }
                    }
                    // Mop up any remaining over-cap entries that tie the cutoff
                    // timestamp, but ONLY those at-or-below the cutoff — never a
                    // newer (possibly recently-throttled) bucket, whose recreation
                    // with a full burst would let it momentarily exceed its rate.
                    for (auto it = rate_limit_buckets.begin();
                         it != rate_limit_buckets.end() &&
                         rate_limit_buckets.size() > kHardMaxBuckets;) {
                        if (it->second.last_refill <= cutoff) {
                            it = rate_limit_buckets.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
            auto& bucket = rate_limit_buckets[client_ip];
            if (bucket.tokens == 0 && bucket.last_refill == std::chrono::steady_clock::time_point{}) {
                // Initialize new bucket
                bucket.tokens = static_cast<double>(config_.rate_limit_burst);
                bucket.last_refill = now;
            }
            // Refill tokens
            auto elapsed = std::chrono::duration<double>(now - bucket.last_refill).count();
            bucket.tokens = std::min(
                static_cast<double>(config_.rate_limit_burst),
                bucket.tokens + elapsed * config_.rate_limit_rps);
            bucket.last_refill = now;
            if (bucket.tokens < 1.0) {
                json_error(res, 429, "rate limit exceeded");
                return;
            }
            bucket.tokens -= 1.0;
        }

        auto name = req.matches[1].str();
        // A/B routing: if `name` names a configured experiment, route this
        // request to the selected variant (model_a / model_b) per the
        // experiment's traffic split, and count it against the experiment.
        // select_model() returns "" when no experiment matches, so direct model
        // requests are unaffected. (Previously experiments were configurable via
        // /v1/experiments but never consulted on the predict path.)
        if (std::string routed = traffic_router_.select_model(name); !routed.empty()) {
            name = std::move(routed);
        }
        auto model = repository_.get_model(name);
        if (!model || model->state.load() != ModelState::READY) {
            json_error(res, 404, "model not ready");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        try {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const json::parse_error& e) {
                json_error(res, 400, std::string("malformed JSON: ") + e.what());
                return;
            }

            if (!body.contains("shape") || !body.contains("data")) {
                json_error(res, 400, "request must contain \"shape\" and \"data\"");
                return;
            }
            // dtype defaults to float32 to preserve the previous ad-hoc
            // float-only behavior when clients omit the field.
            DType dtype = DType::Float32;
            if (body.contains("dtype") && body["dtype"].is_string()) {
                try {
                    dtype = dtype_from_request(body["dtype"].get<std::string>());
                } catch (const std::exception& e) {
                    json_error(res, 400, e.what());
                    return;
                }
            }

            Tensor input;
            try {
                input = tensor_from_json(body["shape"], body["data"], dtype, model->device);
            } catch (const std::exception& e) {
                json_error(res, 400, e.what());
                return;
            }

            // Submit to batcher and wait for result
            auto future = model->batcher->submit(input);
            auto output = future.get();

            auto end = std::chrono::high_resolution_clock::now();
            auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // Update metrics
            auto& metrics = MetricsRegistry::instance().get_metrics(name);
            metrics.total_requests.fetch_add(1);
            metrics.total_latency_us.fetch_add(latency_us);
            metrics.record_latency(latency_us);

            // Pull output back to host (cheap no-op when already on CPU) and
            // serialize to JSON. contiguous() guarantees the data<T>() reads
            // are tight-packed.
            Tensor output_host = (output.device().type == Device::Type::CPU)
                ? output.contiguous()
                : output.to(Device::cpu()).contiguous();

            json resp;
            resp["output_shape"] = output_host.shape();
            resp["output_dtype"] = std::string(dtype_name(output_host.dtype()));
            resp["output_data"]  = json_from_tensor(output_host);

            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            auto& metrics = MetricsRegistry::instance().get_metrics(name);
            metrics.error_count.fetch_add(1);
            json_error(res, 500, e.what());
        }
    });

    // A/B experiment management
    svr.Post("/v1/experiments", [this, &json_error](const httplib::Request& req, httplib::Response& res) {
        // {"name":"exp1","model_a":"v1","model_b":"v2","fraction_b":0.1}
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error& e) {
            json_error(res, 400, std::string("malformed JSON: ") + e.what());
            return;
        }
        auto str_or_empty = [&](const char* k) -> std::string {
            return (body.contains(k) && body[k].is_string())
                ? body[k].get<std::string>()
                : std::string{};
        };
        auto name    = str_or_empty("name");
        auto model_a = str_or_empty("model_a");
        auto model_b = str_or_empty("model_b");
        if (name.empty() || model_a.empty() || model_b.empty()) {
            json_error(res, 400, "missing name, model_a, or model_b");
            return;
        }
        double fraction_b = 0.1;
        if (body.contains("fraction_b") && body["fraction_b"].is_number()) {
            fraction_b = body["fraction_b"].get<double>();
        }
        // fraction_b is compared against a uniform [0,1) draw in select_model,
        // so it must lie in [0,1]; a value outside that range silently routes
        // all traffic to one model. Reject it instead of accepting garbage.
        if (!(fraction_b >= 0.0 && fraction_b <= 1.0)) {
            json_error(res, 400, "fraction_b must be in the range [0, 1]");
            return;
        }
        traffic_router_.set_experiment(name, {model_a, model_b, fraction_b});
        res.set_content(json{{"success",true}}.dump(), "application/json");
    });

    svr.Get("/v1/experiments", [this](const httplib::Request&, httplib::Response& res) {
        auto experiments = traffic_router_.list_experiments();
        json arr = json::array();
        for (const auto& e : experiments) {
            auto [a_count, b_count] = traffic_router_.get_metrics(e);
            arr.push_back({
                {"name",       e},
                {"requests_a", a_count},
                {"requests_b", b_count},
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    svr.Delete(R"(/v1/experiments/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto name = req.matches[1].str();
        traffic_router_.remove_experiment(name);
        res.set_content(json{{"success",true}}.dump(), "application/json");
    });

    std::cout << "[TenzorServing] HTTP server listening on port " << config_.http_port << std::endl;
    // Publish the server handle so stop() can unblock listen(). Re-check
    // running_ under the same mutex stop() uses: if stop() ran before we got
    // here, running_ is already false and we must NOT enter listen() (nothing
    // would ever unblock it). Once the handle is registered, any concurrent
    // stop() observes it and calls svr.stop().
    {
        std::lock_guard<std::mutex> lk(server_mutex_);
        if (!running_.load(std::memory_order_acquire)) {
            return;
        }
        http_server_ = &svr;
    }
    svr.listen("0.0.0.0", config_.http_port);
    // listen() returned (stopped or bind failure). Clear the handle before svr
    // goes out of scope so stop() never dereferences a dangling pointer.
    {
        std::lock_guard<std::mutex> lk(server_mutex_);
        http_server_ = nullptr;
    }
#else
    // No HTTP transport compiled in (TENZOR_BUILD_SERVING=OFF). serve_loop runs
    // on the background server_thread_, so throwing here is fatal: an uncaught
    // exception on a non-main thread calls std::terminate and aborts the whole
    // process (this broke the start/stop lifecycle — start() returns, then the
    // serve thread throws asynchronously and kills the process). start() also
    // cannot throw synchronously because callers legitimately expect a server
    // built without a transport to support the lifecycle API (construct, start,
    // query repository/router, stop) — only actual request serving is
    // unavailable.
    //
    // So address the G.12 audit's real concern — "don't *silently* pretend to
    // serve" — by emitting one loud warning, then idling on running_ until
    // stop() clears it, allowing a clean join(). This is observable (not
    // silent) and lifecycle-safe (not a process abort).
    std::cerr << "[TenzorServing] WARNING: this build has no HTTP transport "
                 "(TENZOR_BUILD_SERVING=OFF). start()/stop() are supported but "
                 "no HTTP requests will be served. Configure with "
                 "TENZOR_BUILD_SERVING=ON for httplib, or use the gRPC server "
                 "entry point."
              << std::endl;
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#endif
}

} // namespace serving
} // namespace tenzor
