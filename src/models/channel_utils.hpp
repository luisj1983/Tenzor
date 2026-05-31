#pragma once

#include <algorithm>
#include <cstdint>

namespace tenzor {
namespace models {

// Round `v` to the nearest multiple of `divisor`.
//
// When `round_down_guard` is true (MobileNet convention) the result is floored
// at `divisor` and bumped up one step if rounding reduced the value by more than
// 10%. When false (YOLO / CSPDarknet convention) it is a plain nearest-multiple
// round with no floor or guard. The two conventions are intentionally distinct.
inline int64_t make_divisible(int64_t v, int64_t divisor, bool round_down_guard) {
    int64_t new_v = (v + divisor / 2) / divisor * divisor;
    if (round_down_guard) {
        new_v = std::max(divisor, new_v);
        if (new_v < 0.9 * static_cast<double>(v)) {
            new_v += divisor;
        }
    }
    return new_v;
}

}  // namespace models
}  // namespace tenzor
