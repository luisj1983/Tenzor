// Deprecation shim — the ThreadPool header moved to
// <tenzor/utils/threading/threadpool.hpp> in Phase 4.1 of the plan.
//
// `src/parallel/` was renamed to `src/utils/threading/` because the
// previous directory name implied data/pipeline/tensor parallelism,
// which actually lives under `src/distributed/`. The content was just
// a plain thread pool.
//
// Third-party code that includes the old path still compiles, but
// receives a deprecation warning so the migration is visible. This
// shim will be removed in a future Tenzor release — update your
// `#include` paths before then.
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#  warning "tenzor/parallel/threadpool.hpp is deprecated. Use tenzor/utils/threading/threadpool.hpp instead."
#elif defined(_MSC_VER)
#  pragma message("warning: tenzor/parallel/threadpool.hpp is deprecated. Use tenzor/utils/threading/threadpool.hpp instead.")
#endif

#include "tenzor/utils/threading/threadpool.hpp"
