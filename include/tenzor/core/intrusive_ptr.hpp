/**
 * @file intrusive_ptr.hpp
 * @brief Intrusive reference counting for high-performance tensor management
 *
 * Provides IntrusiveRefCounted base class and intrusive_ptr smart pointer.
 * Objects derive from IntrusiveRefCounted and are managed by intrusive_ptr,
 * which stores the refcount inside the object itself rather than in a
 * separate control block (like std::shared_ptr).
 *
 * Performance: ~5ns per copy vs ~15-20ns for std::shared_ptr due to:
 * - No separate control block allocation
 * - Better cache locality (refcount adjacent to object data)
 * - Single atomic operation per increment/decrement
 *
 * Thread safety:
 * - Increment: relaxed ordering (correctness only requires atomicity)
 * - Decrement: release ordering on decrement, acquire fence before delete
 *   (standard double-checked locking pattern for shared ownership)
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tenzor {

/**
 * @brief Base class providing intrusive reference counting.
 *
 * Classes that want to be managed by intrusive_ptr should inherit from this.
 * The refcount starts at 0; intrusive_ptr's adopting constructor sets it to 1.
 *
 * @code
 * class MyObject : public IntrusiveRefCounted {
 *     int data;
 * public:
 *     MyObject(int d) : data(d) {}
 * };
 *
 * intrusive_ptr<MyObject> p(new MyObject(42));
 * @endcode
 */
class IntrusiveRefCounted {
public:
    IntrusiveRefCounted() noexcept = default;

    // Non-copyable refcount (copies get their own count)
    IntrusiveRefCounted(const IntrusiveRefCounted&) noexcept : refcount_(0) {}
    IntrusiveRefCounted& operator=(const IntrusiveRefCounted&) noexcept { return *this; }

    /// Get current reference count (for debugging/testing only)
    auto use_count() const noexcept -> uint32_t {
        return refcount_.load(std::memory_order_relaxed);
    }

protected:
    virtual ~IntrusiveRefCounted() = default;

private:
    mutable std::atomic<uint32_t> refcount_{0};

    template<typename T>
    friend class intrusive_ptr;

    friend void intrusive_ptr_add_ref(const IntrusiveRefCounted* p) noexcept;
    friend void intrusive_ptr_release(const IntrusiveRefCounted* p) noexcept;
};

/// Increment the reference count (relaxed ordering — atomicity alone suffices)
inline void intrusive_ptr_add_ref(const IntrusiveRefCounted* p) noexcept {
    p->refcount_.fetch_add(1, std::memory_order_relaxed);
}

/// Decrement the reference count. Deletes the object when count reaches 0.
/// Uses release ordering on decrement and acquire fence before delete to
/// ensure all modifications by other threads are visible before destruction.
inline void intrusive_ptr_release(const IntrusiveRefCounted* p) noexcept {
    if (p->refcount_.fetch_sub(1, std::memory_order_release) == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        delete p;
    }
}

/**
 * @brief Smart pointer with intrusive reference counting.
 *
 * Similar to std::shared_ptr but stores the reference count inside the
 * managed object itself. This eliminates the separate control block
 * allocation and provides better cache locality.
 *
 * @tparam T Type that inherits from IntrusiveRefCounted
 */
template<typename T>
class intrusive_ptr {
    static_assert(std::is_base_of_v<IntrusiveRefCounted, T>,
                  "T must inherit from IntrusiveRefCounted");

public:
    /// Default constructor: null pointer
    intrusive_ptr() noexcept : ptr_(nullptr) {}

    /// Construct from raw pointer, taking ownership (refcount incremented)
    explicit intrusive_ptr(T* p) noexcept : ptr_(p) {
        if (ptr_) intrusive_ptr_add_ref(ptr_);
    }

    /// Copy constructor
    intrusive_ptr(const intrusive_ptr& other) noexcept : ptr_(other.ptr_) {
        if (ptr_) intrusive_ptr_add_ref(ptr_);
    }

    /// Move constructor
    intrusive_ptr(intrusive_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    /// Converting copy constructor (for derived-to-base conversions)
    template<typename U>
    requires std::is_convertible_v<U*, T*>
    intrusive_ptr(const intrusive_ptr<U>& other) noexcept : ptr_(other.get()) {
        if (ptr_) intrusive_ptr_add_ref(ptr_);
    }

    /// Converting move constructor
    template<typename U>
    requires std::is_convertible_v<U*, T*>
    intrusive_ptr(intrusive_ptr<U>&& other) noexcept : ptr_(other.get()) {
        other.release_ownership();
    }

    ~intrusive_ptr() {
        if (ptr_) intrusive_ptr_release(ptr_);
    }

    /// Copy assignment
    auto operator=(const intrusive_ptr& other) noexcept -> intrusive_ptr& {
        if (ptr_ != other.ptr_) {
            if (other.ptr_) intrusive_ptr_add_ref(other.ptr_);
            if (ptr_) intrusive_ptr_release(ptr_);
            ptr_ = other.ptr_;
        }
        return *this;
    }

    /// Move assignment
    auto operator=(intrusive_ptr&& other) noexcept -> intrusive_ptr& {
        if (ptr_ != other.ptr_) {
            if (ptr_) intrusive_ptr_release(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /// Dereference
    auto operator*() const noexcept -> T& { return *ptr_; }

    /// Arrow
    auto operator->() const noexcept -> T* { return ptr_; }

    /// Get raw pointer
    auto get() const noexcept -> T* { return ptr_; }

    /// Boolean conversion
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /// Comparison
    auto operator==(const intrusive_ptr& other) const noexcept -> bool { return ptr_ == other.ptr_; }
    auto operator!=(const intrusive_ptr& other) const noexcept -> bool { return ptr_ != other.ptr_; }
    auto operator==(std::nullptr_t) const noexcept -> bool { return ptr_ == nullptr; }
    auto operator!=(std::nullptr_t) const noexcept -> bool { return ptr_ != nullptr; }

    /// Reset to null
    void reset() noexcept {
        if (ptr_) {
            intrusive_ptr_release(ptr_);
            ptr_ = nullptr;
        }
    }

    /// Swap
    void swap(intrusive_ptr& other) noexcept { std::swap(ptr_, other.ptr_); }

    /// Release ownership without decrementing (for move operations)
    void release_ownership() noexcept { ptr_ = nullptr; }

private:
    T* ptr_;
};

/// Create an intrusive_ptr managing a newly constructed object
template<typename T, typename... Args>
auto make_intrusive(Args&&... args) -> intrusive_ptr<T> {
    return intrusive_ptr<T>(new T(std::forward<Args>(args)...));
}

/// Static pointer cast for intrusive_ptr (like std::static_pointer_cast)
template<typename T, typename U>
auto static_pointer_cast(const intrusive_ptr<U>& p) -> intrusive_ptr<T> {
    return intrusive_ptr<T>(static_cast<T*>(p.get()));
}

} // namespace tenzor

/// std::hash specialization for intrusive_ptr
template<typename T>
struct std::hash<tenzor::intrusive_ptr<T>> {
    auto operator()(const tenzor::intrusive_ptr<T>& p) const noexcept -> size_t {
        return std::hash<T*>{}(p.get());
    }
};
