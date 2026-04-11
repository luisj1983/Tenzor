// Deprecation shim — see include/tenzor/parallel/threadpool.hpp for the
// full Phase 4.1 rename rationale.
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#  warning "tenzor/parallel/atomic.hpp is deprecated. Use tenzor/utils/threading/atomic.hpp instead."
#elif defined(_MSC_VER)
#  pragma message("warning: tenzor/parallel/atomic.hpp is deprecated. Use tenzor/utils/threading/atomic.hpp instead.")
#endif

#include "tenzor/utils/threading/atomic.hpp"
