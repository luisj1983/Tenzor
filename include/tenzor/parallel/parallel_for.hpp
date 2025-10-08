#pragma once

#include <functional>
#include <cstdint>

namespace tenzor {

// Parallel for execution
template<typename F>
auto parallel_for(int64_t begin, int64_t end, F&& func) -> void;

// Parallel for with grain size
template<typename F>
auto parallel_for(int64_t begin, int64_t end, int64_t grain_size, F&& func) -> void;

// Parallel reduce
template<typename T, typename F, typename R>
auto parallel_reduce(int64_t begin, int64_t end,
                    T init,
                    F&& map_func,
                    R&& reduce_func) -> T;

} // namespace tenzor
