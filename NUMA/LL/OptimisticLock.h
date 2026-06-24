// File: OptimisticLock.h
#pragma once

#include <atomic>
#include <thread> // For std::this_thread::yield

class OptimisticLock {
public:
    OptimisticLock() : version(0) {}

    uint64_t read_begin() const {
        return version.load(std::memory_order_acquire);
    }

    bool read_validate(uint64_t old_version) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return (old_version % 2 == 0) && (version.load(std::memory_order_relaxed) == old_version);
    }

    void begin_write() {
        version.fetch_add(1, std::memory_order_release); 
        std::atomic_thread_fence(std::memory_order_release);
    }

    void end_write() {
        std::atomic_thread_fence(std::memory_order_release); 
        version.fetch_add(1, std::memory_order_release); 
    }
    
    uint64_t get_current_version_for_write_validation() const {
        return version.load(std::memory_order_acquire);
    }

private:
    std::atomic<uint64_t> version;
};
