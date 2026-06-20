// mps_buffer_util.h
// Shared host-side MTLBuffer resolution for tensor operands.
//
// Tensors backed by the MPS allocator expose their MTLBuffer through the
// allocator's pooled map (keyed by allocation-base pointer). Contiguous tensors
// own their allocation, so their data_ptr() IS the base and they hit the pool
// at offset 0. Views/slices point partway into an allocation: their data_ptr()
// is generally not page-aligned, and newBufferWithBytesNoCopy requires a
// page-aligned pointer/length (returning nil otherwise, which crashes on
// encode). For those, we materialize a contiguous copy (a fresh allocation
// base) and tie the copy's storage lifetime to the returned MTLBuffer via an
// associated keep-alive box, so it survives the synchronous dispatch and is
// freed when ARC releases the buffer.
//
// Objective-C++ only.

#pragma once

#import <Metal/Metal.h>
#import <objc/runtime.h>

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cstdint>
#include <stdexcept>

// ObjC box owning a heap Tensor, freed on dealloc. Declared at file scope —
// ObjC classes cannot live in a C++ namespace.
@interface TZMpsTensorBox : NSObject {
@public
    tenzor::Tensor* _tensor;
}
@end

namespace tenzor::mps {

// The allocator's buffer lookup lives in mps_backend.mm.
id<MTLBuffer> pooled_buffer_for(void* ptr);

namespace detail {
inline const void* mps_keep_alive_key() {
    static const char key = 0;
    return &key;
}
}  // namespace detail

// Resolve an MTLBuffer for `tensor` on `device`, bound at offset 0:
//  - allocation-base tensor: return the pooled buffer directly (no copy);
//  - offset view: materialize .contiguous() and return its (kept-alive) buffer.
inline id<MTLBuffer> mps_buffer_for(id<MTLDevice> device, const Tensor& tensor) {
    void* ptr = const_cast<void*>(tensor.data_ptr());
    if (id<MTLBuffer> pooled = pooled_buffer_for(ptr)) {
        return pooled;
    }

    auto attach_keepalive = [](id<MTLBuffer> buf, const Tensor& c) {
        if (!buf) return;
        TZMpsTensorBox* box = [[TZMpsTensorBox alloc] init];
        box->_tensor = new Tensor(c);
        objc_setAssociatedObject(buf, detail::mps_keep_alive_key(), box,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    };

    // Offset view (or any pointer that is not the allocation base): copy to a
    // contiguous allocation-base tensor and use that.
    if (tensor.offset() != 0 || (tensor.storage() && tensor.storage()->data() != ptr)) {
        Tensor c = tensor.contiguous();
        void* cptr = const_cast<void*>(c.data_ptr());
        if (id<MTLBuffer> pooled = pooled_buffer_for(cptr)) {
            attach_keepalive(pooled, c);
            return pooled;
        }
        size_t cbytes = c.numel() * dtype_size(c.dtype());
        id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:cptr
                                                      length:cbytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
        attach_keepalive(buf, c);
        return buf;
    }

    // Allocation-base pointer not in the pooled map (no active allocator): wrap
    // no-copy. The base is page-aligned by the allocator; fail loudly rather
    // than silently returning nil if that invariant is somehow violated.
    if ((reinterpret_cast<uintptr_t>(ptr) % 16384) != 0) {
        throw std::runtime_error(
            "MPS mps_buffer_for: non-page-aligned base pointer for no-copy wrap");
    }
    size_t bytes = tensor.numel() * dtype_size(tensor.dtype());
    return [device newBufferWithBytesNoCopy:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

}  // namespace tenzor::mps
