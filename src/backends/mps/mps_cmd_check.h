// mps_cmd_check.h
// Helper for surfacing Metal command-buffer errors. Without this every
// `[cmd commit]` / `[cmd waitUntilCompleted]` pair silently swallows GPU
// errors — shader compile failures, buffer misalignments, OOM — and returns
// undefined results to the caller (audit C5 / MPS A8).
//
// Usage (Objective-C++ only): after a wait, call mps_cmd_check(cmd, "kernel_name").
// It throws std::runtime_error with the Metal error description when
// cmd.error != nil.

#pragma once

#import <Metal/Metal.h>
#include <stdexcept>
#include <string>

namespace tenzor::mps {

inline void mps_cmd_check(id<MTLCommandBuffer> cmd, const char* tag) {
    if (cmd != nil && cmd.error != nil) {
        NSString* desc = [cmd.error localizedDescription];
        const char* utf8 = desc ? [desc UTF8String] : "(no description)";
        throw std::runtime_error(
            std::string("MPS command buffer error in ") + (tag ? tag : "?") +
            ": " + (utf8 ? utf8 : "(no description)"));
    }
}

}  // namespace tenzor::mps
