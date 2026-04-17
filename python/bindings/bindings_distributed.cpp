// tenzor.distributed Python bindings. Extracted from python/bindings.cpp
// as part of P3.4 (incremental split of the ~10k-line monolith).
//
// Covers: process group management, DDP, FSDP, gradient compression, RPC.

#include "register.hpp"

#include <pybind11/stl.h>

#include <tenzor/distributed/distributed.hpp>
#include <tenzor/distributed/ddp.hpp>
#include <tenzor/distributed/fsdp.hpp>
#include <tenzor/distributed/gradient_compression.hpp>
#include <tenzor/distributed/rpc/rpc.hpp>
#include <tenzor/distributed/rpc/function_registry.hpp>
#include <tenzor/nn/module.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_distributed(py::module_& m) {
    auto distributed = m.def_submodule("distributed", "Distributed training");

    py::enum_<tenzor::distributed::ReduceOp>(distributed, "ReduceOp")
        .value("SUM", tenzor::distributed::ReduceOp::SUM)
        .value("PRODUCT", tenzor::distributed::ReduceOp::PRODUCT)
        .value("MIN", tenzor::distributed::ReduceOp::MIN)
        .value("MAX", tenzor::distributed::ReduceOp::MAX)
        .value("AVG", tenzor::distributed::ReduceOp::AVG)
        .export_values();

    distributed.def("init_process_group", &tenzor::distributed::init_process_group,
        "Initialize distributed process group",
        py::arg("backend") = "nccl",
        py::arg("rank") = -1,
        py::arg("world_size") = -1,
        py::arg("master_addr") = "localhost",
        py::arg("master_port") = 29500);

    distributed.def("destroy_process_group", &tenzor::distributed::destroy_process_group,
        "Destroy process group and cleanup resources");

    distributed.def("get_rank", &tenzor::distributed::get_rank,
        "Get current process rank");

    distributed.def("get_world_size", &tenzor::distributed::get_world_size,
        "Get total number of processes");

    distributed.def("is_initialized", &tenzor::distributed::is_initialized,
        "Check if distributed training is initialized");

    distributed.def("barrier", &tenzor::distributed::barrier,
        "Barrier synchronization across all processes");

    distributed.def("get_process_group", []() {
        return tenzor::distributed::DistributedContext::get_process_group();
    }, "Get the default process group initialized by init_process_group()");

    distributed.def("all_reduce", &tenzor::distributed::all_reduce,
        "All-reduce operation on tensor",
        py::arg("tensor"), py::arg("op") = tenzor::distributed::ReduceOp::SUM);

    distributed.def("broadcast", &tenzor::distributed::broadcast,
        "Broadcast tensor from source rank",
        py::arg("tensor"), py::arg("src_rank") = 0);

    py::class_<tenzor::distributed::ProcessGroup, std::shared_ptr<tenzor::distributed::ProcessGroup>>(
        distributed, "ProcessGroup")
        .def_property_readonly("rank", &tenzor::distributed::ProcessGroup::rank,
            "Get process rank")
        .def_property_readonly("world_size", &tenzor::distributed::ProcessGroup::world_size,
            "Get world size")
        .def("broadcast", &tenzor::distributed::ProcessGroup::broadcast,
            "Broadcast tensor from source rank",
            py::arg("tensor"), py::arg("src_rank") = 0)
        .def("all_reduce", &tenzor::distributed::ProcessGroup::all_reduce,
            "All-reduce operation",
            py::arg("tensor"), py::arg("op") = tenzor::distributed::ReduceOp::SUM)
        .def("barrier", &tenzor::distributed::ProcessGroup::barrier,
            "Barrier synchronization");

    py::class_<tenzor::distributed::DistributedDataParallel>(distributed, "DistributedDataParallel")
        .def(py::init<tenzor::nn::Module&, tenzor::distributed::ProcessGroup&, size_t>(),
            "Construct DDP wrapper",
            py::arg("module"), py::arg("process_group"),
            py::arg("bucket_size_bytes") = tenzor::distributed::DistributedDataParallel::DEFAULT_BUCKET_SIZE)
        .def("forward", &tenzor::distributed::DistributedDataParallel::forward,
            "Forward pass through wrapped module",
            py::arg("input"),
            py::call_guard<py::gil_scoped_release>())
        .def("synchronize_gradients", &tenzor::distributed::DistributedDataParallel::synchronize_gradients,
            "Synchronize gradients across all processes")
        .def("sync_comm", &tenzor::distributed::DistributedDataParallel::sync_comm,
            "Wait for pending async all-reduce operations")
        .def("auto_sync_gradients", &tenzor::distributed::DistributedDataParallel::auto_sync_gradients,
            "Enable or disable automatic gradient synchronization",
            py::arg("enabled"))
        .def("reset_buckets", &tenzor::distributed::DistributedDataParallel::reset_buckets,
            "Reset bucket ready states for next iteration");

    // FSDP (Fully Sharded Data Parallel)
    py::enum_<tenzor::distributed::ShardingStrategy>(distributed, "ShardingStrategy")
        .value("FULL_SHARD", tenzor::distributed::ShardingStrategy::FULL_SHARD)
        .value("SHARD_GRAD_OP", tenzor::distributed::ShardingStrategy::SHARD_GRAD_OP)
        .value("NO_SHARD", tenzor::distributed::ShardingStrategy::NO_SHARD);

    py::class_<tenzor::distributed::FSDPConfig>(distributed, "FSDPConfig")
        .def(py::init<>())
        .def_readwrite("strategy", &tenzor::distributed::FSDPConfig::strategy)
        .def_readwrite("cpu_offload", &tenzor::distributed::FSDPConfig::cpu_offload)
        .def_readwrite("auto_wrap_min_params", &tenzor::distributed::FSDPConfig::auto_wrap_min_params)
        .def_readwrite("mixed_precision", &tenzor::distributed::FSDPConfig::mixed_precision)
        .def_readwrite("forward_prefetch", &tenzor::distributed::FSDPConfig::forward_prefetch)
        .def_readwrite("backward_prefetch", &tenzor::distributed::FSDPConfig::backward_prefetch);

    py::class_<tenzor::distributed::FullyShardedDataParallel>(distributed, "FullyShardedDataParallel")
        .def(py::init<tenzor::nn::Module&, tenzor::distributed::ProcessGroup&,
                       const tenzor::distributed::FSDPConfig&>(),
             py::arg("module"), py::arg("process_group"),
             py::arg("config") = tenzor::distributed::FSDPConfig{})
        .def("forward", &tenzor::distributed::FullyShardedDataParallel::forward,
             py::arg("input"))
        .def("finalize_backward", &tenzor::distributed::FullyShardedDataParallel::finalize_backward)
        .def("summon_full_params", &tenzor::distributed::FullyShardedDataParallel::summon_full_params)
        .def("release_full_params", &tenzor::distributed::FullyShardedDataParallel::release_full_params)
        .def("total_params", &tenzor::distributed::FullyShardedDataParallel::total_params)
        .def("sharded_param_bytes", &tenzor::distributed::FullyShardedDataParallel::sharded_param_bytes);

    // Gradient Compression
    py::class_<tenzor::distributed::CompressedGradient>(distributed, "CompressedGradient")
        .def_readonly("data", &tenzor::distributed::CompressedGradient::data)
        .def_readonly("original_shape", &tenzor::distributed::CompressedGradient::original_shape)
        .def_readonly("compression_ratio", &tenzor::distributed::CompressedGradient::compression_ratio);

    py::class_<tenzor::distributed::FP16Compressor>(distributed, "FP16Compressor")
        .def(py::init<>())
        .def("compress", &tenzor::distributed::FP16Compressor::compress, py::arg("gradient"))
        .def("decompress", &tenzor::distributed::FP16Compressor::decompress, py::arg("compressed"))
        .def("name", &tenzor::distributed::FP16Compressor::name)
        .def("reset", &tenzor::distributed::FP16Compressor::reset);

    py::class_<tenzor::distributed::TopKCompressor>(distributed, "TopKCompressor")
        .def(py::init<double>(), py::arg("ratio") = 0.01)
        .def("compress", &tenzor::distributed::TopKCompressor::compress, py::arg("gradient"))
        .def("decompress", &tenzor::distributed::TopKCompressor::decompress, py::arg("compressed"))
        .def("name", &tenzor::distributed::TopKCompressor::name)
        .def("reset", &tenzor::distributed::TopKCompressor::reset);

    // RPC submodule
    auto rpc = distributed.def_submodule("rpc", "Remote Procedure Call framework");

    py::class_<tenzor::distributed::rpc::RpcAgentConfig>(rpc, "RpcAgentConfig",
        "Configuration for the RPC agent")
        .def(py::init<>())
        .def_readwrite("num_io_threads",
            &tenzor::distributed::rpc::RpcAgentConfig::num_io_threads,
            "I/O threads for socket operations (default: 2)")
        .def_readwrite("num_worker_threads",
            &tenzor::distributed::rpc::RpcAgentConfig::num_worker_threads,
            "Worker threads for RPC execution (default: 4)")
        .def_readwrite("timeout_ms",
            &tenzor::distributed::rpc::RpcAgentConfig::timeout_ms,
            "RPC timeout in milliseconds (default: 60000)")
        .def_readwrite("heartbeat_interval_ms",
            &tenzor::distributed::rpc::RpcAgentConfig::heartbeat_interval_ms,
            "Heartbeat interval in milliseconds (default: 5000)")
        .def_readwrite("enable_heartbeat",
            &tenzor::distributed::rpc::RpcAgentConfig::enable_heartbeat,
            "Enable health monitoring (default: true)");

    rpc.def("init_rpc", &tenzor::distributed::rpc::init_rpc,
        py::arg("name"), py::arg("rank"), py::arg("world_size"),
        py::arg("config") = tenzor::distributed::rpc::RpcAgentConfig{},
        "Initialize the RPC framework");

    rpc.def("shutdown_rpc", &tenzor::distributed::rpc::shutdown_rpc,
        "Shut down the RPC framework");

    rpc.def("rpc_sync", &tenzor::distributed::rpc::rpc_sync,
        py::arg("dst"), py::arg("func_name"), py::arg("args"),
        "Synchronous RPC call to a remote worker");

    rpc.def("rpc_async", [](int32_t dst, const std::string& func_name,
                             const std::vector<tenzor::Tensor>& args) {
        auto future = tenzor::distributed::rpc::rpc_async(dst, func_name, args);
        return future.get();  // Block in Python for simplicity
    },
    py::arg("dst"), py::arg("func_name"), py::arg("args"),
    "Asynchronous RPC call (blocks until result available in Python)");

    rpc.def("register_function",
        [](const std::string& name, py::function fn) {
            // Wrap the Python callable so it can be invoked from the C++
            // RPC handler thread. We acquire the GIL before calling in.
            auto py_fn = fn.cast<py::object>();
            tenzor::distributed::rpc::FunctionRegistry::instance()
                .register_function(name,
                    [py_fn](const std::vector<tenzor::Tensor>& args)
                        -> std::vector<tenzor::Tensor> {
                        py::gil_scoped_acquire gil;
                        py::object result = py_fn(args);
                        return result.cast<std::vector<tenzor::Tensor>>();
                    });
        },
        py::arg("name"), py::arg("fn"),
        "Register a Python callable under `name` for remote invocation. "
        "The callable receives and returns a list of Tensors.");

    rpc.def("has_function",
        [](const std::string& name) -> bool {
            return tenzor::distributed::rpc::FunctionRegistry::instance()
                .has_function(name);
        },
        py::arg("name"),
        "Check whether a function is registered.");
}

} // namespace tenzor::python
