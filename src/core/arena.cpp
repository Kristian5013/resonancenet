#include "core/arena.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>

namespace rnet::core {

// ─── Arena ───────────────────────────────────────────────────────────

Arena::Arena(size_t capacity)
    : capacity_(capacity), offset_(0) {
    if (capacity > 0) {
        buffer_ = static_cast<uint8_t*>(
            ::operator new(capacity, std::align_val_t{
                alignof(std::max_align_t)}));
    }
}

Arena::~Arena() {
    if (buffer_) {
        ::operator delete(buffer_, std::align_val_t{
            alignof(std::max_align_t)});
    }
}

Arena::Arena(Arena&& other) noexcept
    : buffer_(other.buffer_),
      capacity_(other.capacity_),
      offset_(other.offset_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
            ::operator delete(buffer_, std::align_val_t{
                alignof(std::max_align_t)});
        }
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

void* Arena::alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;

    // Align the current offset
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

    if (aligned_offset + bytes > capacity_) {
        return nullptr;
    }

    void* ptr = buffer_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void Arena::reset() {
    offset_ = 0;
}

bool Arena::contains(const void* ptr) const {
    auto* p = static_cast<const uint8_t*>(ptr);
    return p >= buffer_ && p < buffer_ + capacity_;
}

// ─── GrowableArena ──────────────────────────────────────────────────

GrowableArena::GrowableArena(size_t block_size)
    : block_size_(block_size) {}

void* GrowableArena::alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;

    // Try current block first
    if (current_block_ < blocks_.size()) {
        void* ptr = blocks_[current_block_].alloc(bytes, alignment);
        if (ptr) return ptr;
    }

    // Try subsequent existing blocks (after reset)
    for (size_t i = current_block_ + 1; i < blocks_.size(); ++i) {
        void* ptr = blocks_[i].alloc(bytes, alignment);
        if (ptr) {
            current_block_ = i;
            return ptr;
        }
    }

    // Allocate a new block
    size_t min_size = std::max(bytes + alignment, block_size_);
    add_block(min_size);
    current_block_ = blocks_.size() - 1;

    void* ptr = blocks_.back().alloc(bytes, alignment);
    return ptr;
}

void GrowableArena::reset() {
    for (auto& block : blocks_) {
        block.reset();
    }
    current_block_ = 0;
}

size_t GrowableArena::total_capacity() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.capacity();
    }
    return total;
}

size_t GrowableArena::total_used() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.used();
    }
    return total;
}

void GrowableArena::add_block(size_t min_size) {
    size_t size = std::max(min_size, block_size_);
    blocks_.emplace_back(size);
}

}  // namespace rnet::core
