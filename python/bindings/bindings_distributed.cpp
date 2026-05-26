// tenzor.distributed Python bindings. Extracted from python/bindings.cpp
// as part of P3.4 (incremental split of the ~10k-line monolith).
//
// Covers: process group management, DDP, FSDP, gradient compression, RPC.

#include "register.hpp"
#include "future_binding.hpp"  // TensorListFuture (audit C.6)

#include <pybind11/stl.h>

#include <tenzor/distributed/distributed.hpp>
#include <tenzor/distributed/ddp.hpp>
#include <tenzor/distributed/fsdp.hpp>
#include <tenzor/distributed/gradient_compression.hpp>
#include <tenzor/distributed/pipeline_parallel.hpp>
#include <tenzor/distributed/rpc/rpc.hpp>
#include <tenzor/distributed/rpc/function_registry.hpp>
#include <tenzor/distributed/sequence_parallel.hpp>
#include <tenzor/distributed/tensor_parallel.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/autograd/variable.hpp>

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

    // HH.17: release the GIL while bootstrapping the process group — the
    // call blocks on a TCP rendezvous and other Python threads (e.g. dataset
    // workers) need to keep running. Same rationale for init_rpc /
    // shutdown_rpc below.
    distributed.def("init_process_group", &tenzor::distributed::init_process_group,
        "Initialize distributed process group",
        py::arg("backend") = "nccl",
        py::arg("rank") = -1,
        py::arg("world_size") = -1,
        py::arg("master_addr") = "localhost",
        py::arg("master_port") = 29500,
        py::call_guard<py::gil_scoped_release>());

    distributed.def("destroy_process_group", &tenzor::distributed::destroy_process_group,
        "Destroy process group and cleanup resources");

    distributed.def("get_rank", &tenzor::distributed::get_rank,
        "Get current process rank");

    distributed.def("get_world_size", &tenzor::distributed::get_world_size,
        "Get total number of processes");

    distributed.def("is_initialized", &tenzor::distributed::is_initialized,
        "Check if distributed training is initialized");

    distributed.def("barrier", &tenzor::distributed::barrier,
        "Barrier synchronization across all processes",
        py::call_guard<py::gil_scoped_release>());

    distributed.def("get_process_group", []() {
        return tenzor::distributed::DistributedContext::get_process_group();
    }, "Get the default process group initialized by init_process_group()");

    distributed.def("all_reduce", &tenzor::distributed::all_reduce,
        "All-reduce operation on tensor",
        py::arg("tensor"), py::arg("op") = tenzor::distributed::ReduceOp::SUM,
        py::call_guard<py::gil_scoped_release>());

    distributed.def("broadcast", &tenzor::distributed::broadcast,
        "Broadcast tensor from source rank",
        py::arg("tensor"), py::arg("src_rank") = 0,
        py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::distributed::ProcessGroup, std::shared_ptr<tenzor::distributed::ProcessGroup>>(
        distributed, "ProcessGroup")
        .def_property_readonly("rank", &tenzor::distributed::ProcessGroup::rank,
            "Get process rank")
        .def_property_readonly("world_size", &tenzor::distributed::ProcessGroup::world_size,
            "Get world size")
        .def("broadcast", &tenzor::distributed::ProcessGroup::broadcast,
            "Broadcast tensor from source rank",
            py::arg("tensor"), py::arg("src_rank") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def("all_reduce", &tenzor::distributed::ProcessGroup::all_reduce,
            "All-reduce operation",
            py::arg("tensor"), py::arg("op") = tenzor::distributed::ReduceOp::SUM,
            py::call_guard<py::gil_scoped_release>())
        .def("reduce", &tenzor::distributed::ProcessGroup::reduce,
            "Reduce tensor to a single destination rank",
            py::arg("tensor"), py::arg("dst_rank"),
            py::arg("op") = tenzor::distributed::ReduceOp::SUM,
            py::call_guard<py::gil_scoped_release>())
        // gather / all_gather / reduce_scatter: the underlying C++ signatures
        // take `std::vector<Tensor>&` for the output. pybind11's default STL
        // caster copies that list at entry and does NOT propagate C++-side
        // reassignments back to Python. Gloo's implementations reassign output
        // slots (e.g. `output[src] = zeros_like(tensor); recv into it`), so
        // the Python list never observed the received data. Wrap these to
        // take the output by value and return the filled vector — Python
        // callers then assign the return value back.
        .def("gather", [](tenzor::distributed::ProcessGroup& self,
                           const tenzor::Tensor& tensor,
                           py::list output_list,
                           int dst_rank) -> py::object {
            // Audit item H.5: match torch.distributed.gather convention.
            // The destination list is mutated in place on dst_rank; the
            // call returns None on every rank.  Previous binding took
            // output by value, modified the local copy, and returned it
            // — forcing callers to re-bind their list and producing the
            // documented "other ranks return the unmodified placeholder"
            // oddity.
            std::vector<tenzor::Tensor> output;
            output.reserve(output_list.size());
            for (auto item : output_list) {
                output.push_back(item.cast<tenzor::Tensor>());
            }
            {
                // Q.15: release the GIL across the actual collective so
                // other Python threads (DataLoader workers, NCCL streams)
                // make progress while we're blocked in Gloo.
                py::gil_scoped_release release;
                self.gather(tensor, output, dst_rank);
            }
            if (self.rank() == dst_rank) {
                for (size_t i = 0; i < output.size(); ++i) {
                    output_list[i] = py::cast(output[i]);
                }
            }
            return py::none();
        }, py::arg("tensor"), py::arg("output"), py::arg("dst_rank"),
           "Gather per-rank tensors onto dst_rank.  The `output` list is "
           "mutated in place on dst_rank (other ranks leave it unchanged). "
           "Returns None — matches torch.distributed.gather semantics.")
        .def("scatter", &tenzor::distributed::ProcessGroup::scatter,
            "Scatter per-rank tensors from a source rank to each rank",
            py::arg("tensors"), py::arg("output"), py::arg("src_rank"),
            py::call_guard<py::gil_scoped_release>())
        .def("all_gather", [](tenzor::distributed::ProcessGroup& self,
                               const tenzor::Tensor& tensor,
                               py::list output_list) -> py::object {
            // S.19: mirror gather's "mutate the supplied output list in
            // place, return None" semantics for PyTorch parity with
            // torch.distributed.all_gather. Previously this took a
            // std::vector by value, mutated the copy, and returned the
            // filled vector — Python callers had to re-bind the return
            // value or silently see the original list untouched.
            std::vector<tenzor::Tensor> output;
            output.reserve(output_list.size());
            for (auto item : output_list) {
                output.push_back(item.cast<tenzor::Tensor>());
            }
            {
                // Q.15: release the GIL across the actual collective.
                py::gil_scoped_release release;
                self.all_gather(tensor, output);
            }
            for (size_t i = 0; i < output.size(); ++i) {
                output_list[i] = py::cast(output[i]);
            }
            return py::none();
        }, py::arg("tensor"), py::arg("output"),
           "All-gather per-rank tensors onto every rank. The `output` list "
           "is mutated in place on all ranks. Returns None — matches "
           "torch.distributed.all_gather semantics.")
        .def("reduce_scatter", &tenzor::distributed::ProcessGroup::reduce_scatter,
            "Reduce-scatter: each rank ends up with one slice of the reduction",
            py::arg("tensors"), py::arg("output"),
            py::arg("op") = tenzor::distributed::ReduceOp::SUM,
            py::call_guard<py::gil_scoped_release>())
        .def("barrier", &tenzor::distributed::ProcessGroup::barrier,
            "Barrier synchronization",
            py::call_guard<py::gil_scoped_release>());

    // --- Tensor-parallel linear layers ---
    // Split Linear by column (output features) or row (input features)
    // across a ProcessGroup. Use ColumnParallel → RowParallel for MLPs;
    // the ParallelAttention wrapper composes both for attention blocks.
    py::class_<tenzor::distributed::ColumnParallelLinear, tenzor::nn::Module,
               std::shared_ptr<tenzor::distributed::ColumnParallelLinear>>(
        distributed, "ColumnParallelLinear",
        "Column-parallel Linear: weight split across output-feature dim")
        .def(py::init<int64_t, int64_t, tenzor::distributed::ProcessGroup&,
                      bool, bool>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("process_group"),
             py::arg("bias") = true, py::arg("gather_output") = true)
        .def_property_readonly("in_features",
            &tenzor::distributed::ColumnParallelLinear::in_features)
        .def_property_readonly("out_features",
            &tenzor::distributed::ColumnParallelLinear::out_features)
        .def_property_readonly("local_out_features",
            &tenzor::distributed::ColumnParallelLinear::local_out_features)
        .def("forward",
             &tenzor::distributed::ColumnParallelLinear::forward_impl,
             py::arg("input"),
             "Forward pass: column-parallel matmul + optional all-gather",
             // KK.21: release GIL — forward issues blocking collective comms
             // and large matmuls that would otherwise stall other Python
             // threads.  Mirrors DDP's L257 binding.
             py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::distributed::RowParallelLinear, tenzor::nn::Module,
               std::shared_ptr<tenzor::distributed::RowParallelLinear>>(
        distributed, "RowParallelLinear",
        "Row-parallel Linear: weight split across input-feature dim")
        .def(py::init<int64_t, int64_t, tenzor::distributed::ProcessGroup&,
                      bool, bool>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("process_group"),
             py::arg("bias") = true, py::arg("input_is_parallel") = true)
        .def("forward",
             &tenzor::distributed::RowParallelLinear::forward_impl,
             py::arg("input"),
             "Forward pass: row-parallel matmul + all-reduce",
             // KK.21: see ColumnParallelLinear above.
             py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::distributed::ParallelAttention, tenzor::nn::Module,
               std::shared_ptr<tenzor::distributed::ParallelAttention>>(
        distributed, "ParallelAttention",
        "Multi-head attention with heads sharded across a ProcessGroup")
        .def(py::init<int64_t, int64_t, tenzor::distributed::ProcessGroup&>(),
             py::arg("embed_dim"), py::arg("num_heads"), py::arg("process_group"))
        .def("forward",
             &tenzor::distributed::ParallelAttention::forward_impl,
             py::arg("input"),
             "Forward pass: multi-head attention with sharded heads",
             // KK.21: see ColumnParallelLinear above.
             py::call_guard<py::gil_scoped_release>());

    // --- Pipeline / Sequence parallel ---
    // Pipeline stages are typically built from user code so we expose the
    // base PipelineStage class; full scheduler bindings require task-graph
    // wiring that is beyond the Python surface today.
    py::class_<tenzor::distributed::PipelineStage,
               std::shared_ptr<tenzor::distributed::PipelineStage>>(
        distributed, "PipelineStage",
        "One stage of a pipeline-parallel execution graph")
        .def(py::init<std::shared_ptr<tenzor::nn::Module>, int, int>(),
             py::arg("module"), py::arg("stage_id"), py::arg("num_stages"),
             "Wrap a module as a pipeline stage. stage_id is this rank's "
             "position (0..num_stages-1); num_stages is the total pipeline depth.")
        .def("forward", &tenzor::distributed::PipelineStage::forward,
             py::arg("input"),
             "Run the local sub-module's forward pass. Send/recv across "
             "stages is handled by a scheduler, not this method.",
             // KK.21: see ColumnParallelLinear above.
             py::call_guard<py::gil_scoped_release>());

    py::class_<tenzor::distributed::SequenceParallel,
               std::shared_ptr<tenzor::distributed::SequenceParallel>>(
        distributed, "SequenceParallel",
        "Sequence-parallel scatter/gather helpers for activations");

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
            "Synchronize gradients across all processes",
            py::call_guard<py::gil_scoped_release>())
        // V.36: release the GIL across DDP comm/state ops.  sync_comm waits on
        // outstanding async all-reduce work; auto_sync_gradients and
        // reset_buckets touch per-parameter bucket state which under large
        // models is several thousand entries — all blocking, all heavy enough
        // to starve other Python threads.
        .def("sync_comm", &tenzor::distributed::DistributedDataParallel::sync_comm,
            "Wait for pending async all-reduce operations",
            py::call_guard<py::gil_scoped_release>())
        .def("auto_sync_gradients", &tenzor::distributed::DistributedDataParallel::auto_sync_gradients,
            "Enable or disable automatic gradient synchronization",
            py::arg("enabled"),
            py::call_guard<py::gil_scoped_release>())
        .def("reset_buckets", &tenzor::distributed::DistributedDataParallel::reset_buckets,
            "Reset bucket ready states for next iteration",
            py::call_guard<py::gil_scoped_release>());

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
             py::arg("input"),
             py::call_guard<py::gil_scoped_release>())
        // V.36: FSDP backward/unshard/reshard each perform full-shard
        // all-gather or reduce-scatter under the hood — blocking collective
        // work that must drop the GIL.
        .def("finalize_backward", &tenzor::distributed::FullyShardedDataParallel::finalize_backward,
             py::call_guard<py::gil_scoped_release>())
        .def("summon_full_params", &tenzor::distributed::FullyShardedDataParallel::summon_full_params,
             py::call_guard<py::gil_scoped_release>())
        .def("release_full_params", &tenzor::distributed::FullyShardedDataParallel::release_full_params,
             py::call_guard<py::gil_scoped_release>())
        .def("total_params", &tenzor::distributed::FullyShardedDataParallel::total_params)
        .def("sharded_param_bytes", &tenzor::distributed::FullyShardedDataParallel::sharded_param_bytes)
        // audit-9 JJ.4: per-rank sharded checkpoint round-trip.  Without these
        // bindings FSDP training literally could not be checkpointed from
        // Python.  Both methods iterate every shard — release the GIL.
        .def("state_dict", &tenzor::distributed::FullyShardedDataParallel::state_dict,
             py::call_guard<py::gil_scoped_release>(),
             "Per-rank sharded checkpoint dictionary.  Each rank returns its "
             "own shard plus metadata; caller orchestrates cross-rank "
             "checkpoint coordination.")
        .def("load_state_dict", &tenzor::distributed::FullyShardedDataParallel::load_state_dict,
             py::arg("state"),
             py::call_guard<py::gil_scoped_release>(),
             "Restore from a per-rank sharded checkpoint produced by "
             "state_dict().  Validates world_size/rank/numel/shape "
             "consistency; throws on mismatch.");

    // Gradient Compression
    py::class_<tenzor::distributed::CompressedGradient>(distributed, "CompressedGradient")
        .def_readonly("data", &tenzor::distributed::CompressedGradient::data)
        .def_readonly("original_shape", &tenzor::distributed::CompressedGradient::original_shape)
        .def_readonly("compression_ratio", &tenzor::distributed::CompressedGradient::compression_ratio);

    py::class_<tenzor::distributed::FP16Compressor>(distributed, "FP16Compressor")
        .def(py::init<>())
        .def("compress", &tenzor::distributed::FP16Compressor::compress, py::arg("gradient"),
             py::call_guard<py::gil_scoped_release>())
        .def("decompress", &tenzor::distributed::FP16Compressor::decompress, py::arg("compressed"),
             py::call_guard<py::gil_scoped_release>())
        .def("name", &tenzor::distributed::FP16Compressor::name)
        .def("reset", &tenzor::distributed::FP16Compressor::reset);

    py::class_<tenzor::distributed::TopKCompressor>(distributed, "TopKCompressor")
        .def(py::init<double>(), py::arg("ratio") = 0.01)
        .def("compress", &tenzor::distributed::TopKCompressor::compress, py::arg("gradient"),
             py::call_guard<py::gil_scoped_release>())
        .def("decompress", &tenzor::distributed::TopKCompressor::decompress, py::arg("compressed"),
             py::call_guard<py::gil_scoped_release>())
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

    // HH.17: GIL release — both blocking on a network setup / teardown.
    rpc.def("init_rpc", &tenzor::distributed::rpc::init_rpc,
        py::arg("name"), py::arg("rank"), py::arg("world_size"),
        py::arg("config") = tenzor::distributed::rpc::RpcAgentConfig{},
        "Initialize the RPC framework",
        py::call_guard<py::gil_scoped_release>());

    rpc.def("shutdown_rpc", &tenzor::distributed::rpc::shutdown_rpc,
        "Shut down the RPC framework",
        py::call_guard<py::gil_scoped_release>());

    // V.36: rpc_sync blocks the calling Python thread on a network round-trip
    // (potentially seconds).  The C++ implementation in `rpc.cpp` returns
    // plain Tensors and does not call back into Python before returning, so
    // GIL release is safe — no need for the request handler to reacquire.
    rpc.def("rpc_sync", &tenzor::distributed::rpc::rpc_sync,
        py::arg("dst"), py::arg("func_name"), py::arg("args"),
        "Synchronous RPC call to a remote worker",
        py::call_guard<py::gil_scoped_release>());

    rpc.def("rpc_async", [](int32_t dst, const std::string& func_name,
                             const std::vector<tenzor::Tensor>& args) {
        // Audit C.6: previously this blocked on `.get()`. Now we return a
        // TensorListFuture so callers can overlap compute; they must call
        // `.result()` (or `.wait()`) explicitly.
        return TensorListFuture(
            tenzor::distributed::rpc::rpc_async(dst, func_name, args));
    },
    py::arg("dst"), py::arg("func_name"), py::arg("args"),
    "Asynchronous RPC call; returns a TensorListFuture (call .result()).");

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
                        // HH.18: registered callables that return
                        // List[Variable] previously triggered a py::cast_error
                        // here because the cast<List[Tensor]> path doesn't
                        // know how to walk Variable wrappers. Try the
                        // Variable path first and extract .tensor() so the
                        // C++ RPC wire (Tensor-only) is happy; fall back to
                        // the plain-Tensor cast for the common case.
                        // Note: Variable wire serialisation (preserving the
                        // grad_fn chain across the network) is out of scope
                        // here — this only routes the forward result.
                        try {
                            auto vars = result.cast<std::vector<
                                tenzor::Variable>>();
                            std::vector<tenzor::Tensor> tensors;
                            tensors.reserve(vars.size());
                            for (auto& v : vars) {
                                tensors.push_back(v.tensor());
                            }
                            return tensors;
                        } catch (const py::cast_error&) {
                            return result.cast<std::vector<tenzor::Tensor>>();
                        }
                    });
        },
        py::arg("name"), py::arg("fn"),
        "Register a Python callable under `name` for remote invocation. "
        "The callable receives a list of Tensors and returns a list of "
        "Tensors (or Variables; their .tensor() is extracted for the wire).");

    rpc.def("has_function",
        [](const std::string& name) -> bool {
            return tenzor::distributed::rpc::FunctionRegistry::instance()
                .has_function(name);
        },
        py::arg("name"),
        "Check whether a function is registered.");
}

} // namespace tenzor::python
