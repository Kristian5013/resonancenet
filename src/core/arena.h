#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace rnet::core {

/// Simple memory arena allocator.
/// Allocates from a pre-allocated block. Fast bump allocation.
/// Call reset() to reuse without freeing individual allocations.
class Arena {
public:
    explicit Arena(size_t capacity);
    ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) noexcept;
    Arena& operator=(Arena&&) noexcept;

    /// Allocate bytes with specified alignment.
    /// Returns nullptr if not enough space.
    void* alloc(size_t bytes, size_t alignment = alignof(std::max_align_t));

    /// Typed allocation
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = alloc(sizeof(T), alignof(T));
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }

    /// Allocate an array of T
    template<typename T>
    T* alloc_array(size_t count) {
        void* mem = alloc(sizeof(T) * count, alignof(T));
        if (!mem) return nullptr;
        // Default-construct each element
        T* arr = static_cast<T*>(mem);
        for (size_t i = 0; i < count; ++i) {
            new (&arr[i]) T();
        }
        return arr;
    }

    /// Reset the arena — all allocations become invalid.
    /// Does NOT call destructors.
    void reset();

    /// Get total capacity
    size_t capacity() const { return capacity_; }

    /// Get bytes currently allocated
    size_t used() const { return offset_; }

    /// Get bytes remaining
    size_t remaining() const { return capacity_ - offset_; }

    /// Check if arena contains a pointer
    bool contains(const void* ptr) const;

private:
    uint8_t* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
};

/// GrowableArena — automatically allocates new blocks when full.
class GrowableArena {
public:
    explicit GrowableArena(size_t block_size = 4096);
    ~GrowableArena() = default;

    GrowableArena(const GrowableArena&) = delete;
    GrowableArena& operator=(const GrowableArena&) = delete;
    GrowableArena(GrowableArena&&) = default;
    GrowableArena& operator=(GrowableArena&&) = default;

    /// Allocate bytes. Always succeeds (allocates new block if needed).
    void* alloc(size_t bytes,
                size_t alignment = alignof(std::max_align_t));

    /// Typed allocation
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        void* mem = alloc(sizeof(T), alignof(T));
        return new (mem) T(std::forward<Args>(args)...);
    }

    /// Reset all blocks (keeps allocated memory for reuse)
    void reset();

    /// Total bytes allocated across all blocks
    size_t total_capacity() const;

    /// Total bytes used across all blocks
    size_t total_used() const;

    /// Number of blocks allocated
    size_t num_blocks() const { return blocks_.size(); }

private:
    size_t block_size_;
    std::vector<Arena> blocks_;
    size_t current_block_ = 0;

    void add_block(size_t min_size);
};

/// PoolAllocator — fixed-size object pool.
/// Extremely fast allocation/deallocation for same-size objects.
template<typename T>
class PoolAllocator {
public:
    explicit PoolAllocator(size_t pool_size = 256)
        : pool_size_(pool_size) {
        add_pool();
    }

    ~PoolAllocator() {
        for (auto* pool : pools_) {
            ::operator delete(pool);
        }
    }

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    /// Allocate one T (uninitialized memory)
    T* allocate() {
        if (!free_list_) {
            add_pool();
        }
        Node* node = free_list_;
        free_list_ = node->next;
        ++allocated_count_;
        return reinterpret_cast<T*>(node);
    }

    /// Deallocate one T (does NOT call destructor)
    void deallocate(T* ptr) {
        Node* node = reinterpret_cast<Node*>(ptr);
        node->next = free_list_;
        free_list_ = node;
        --allocated_count_;
    }

    /// Create a T with constructor args
    template<typename... Args>
    T* create(Args&&... args) {
        T* ptr = allocate();
        return new (ptr) T(std::forward<Args>(args)...);
    }

    /// Destroy and deallocate
    void destroy(T* ptr) {
        ptr->~T();
        deallocate(ptr);
    }

    size_t allocated_count() const { return allocated_count_; }
    size_t pool_count() const { return pools_.size(); }

private:
    union Node {
        Node* next;
        alignas(T) uint8_t storage[sizeof(T)];
    };

    void add_pool() {
        auto* pool = static_cast<Node*>(
            ::operator new(sizeof(Node) * pool_size_));
        pools_.push_back(pool);

        // Build free list
        for (size_t i = 0; i < pool_size_ - 1; ++i) {
            pool[i].next = &pool[i + 1];
        }
        pool[pool_size_ - 1].next = free_list_;
        free_list_ = pool;
    }

    size_t pool_size_;
    Node* free_list_ = nullptr;
    std::vector<Node*> pools_;
    size_t allocated_count_ = 0;
};

/// StringArena: arena-allocate immutable string copies.
/// Useful for interning strings without individual heap allocations.
class StringArena {
public:
    explicit StringArena(size_t block_size = 4096)
        : arena_(block_size) {}

    /// Copy a string into the arena. Returns a view into arena memory.
    std::string_view store(std::string_view str) {
        if (str.empty()) return {};
        char* mem = static_cast<char*>(
            arena_.alloc(str.size(), 1));
        if (!mem) return {};
        std::memcpy(mem, str.data(), str.size());
        return std::string_view(mem, str.size());
    }

    /// Store a null-terminated C string
    const char* store_cstr(std::string_view str) {
        char* mem = static_cast<char*>(
            arena_.alloc(str.size() + 1, 1));
        if (!mem) return nullptr;
        std::memcpy(mem, str.data(), str.size());
        mem[str.size()] = '\0';
        return mem;
    }

    void reset() { arena_.reset(); }
    size_t total_used() const { return arena_.total_used(); }
    size_t total_capacity() const { return arena_.total_capacity(); }

private:
    GrowableArena arena_;
};

/// ScopedArena: RAII wrapper that resets arena on destruction.
/// Useful for temporary allocations within a scope.
class ScopedArena {
public:
    explicit ScopedArena(Arena& arena)
        : arena_(arena), saved_offset_(arena.used()) {}

    ~ScopedArena() {
        // Reset to saved position (only valid if no allocations
        // were made from the arena by other users in between)
        arena_.reset();
    }

    ScopedArena(const ScopedArena&) = delete;
    ScopedArena& operator=(const ScopedArena&) = delete;

    void* alloc(size_t bytes, size_t alignment =
                    alignof(std::max_align_t)) {
        return arena_.alloc(bytes, alignment);
    }

    template<typename T, typename... Args>
    T* create(Args&&... args) {
        return arena_.create<T>(std::forward<Args>(args)...);
    }

private:
    Arena& arena_;
    size_t saved_offset_;
};

}  // namespace rnet::core
