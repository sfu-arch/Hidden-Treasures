// File: NumaAllocator.h
#pragma once

#include <numa.h>
#include <stdexcept>
#include <iostream> // For perror
#include <cstdlib>  // For posix_memalign, free
#include <new>      // For std::bad_alloc

// Define a cache line size, common values are 64 or 128 bytes.
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

template <typename T>
class NumaAllocator {
public:
    // Allocate memory on a specific NUMA node, aligned to cache line size
    static T* allocate(size_t n_elements, int node_id) {
        if (numa_available() < 0) {
            // Fallback or error if NUMA is not available but code expects it
            // Forcing an error is safer for a NUMA-specific benchmark
            throw std::runtime_error("NUMA API not available, but NumaAllocator was used.");
        }

        void* ptr = nullptr;
        ptr = numa_alloc_onnode(n_elements * sizeof(T), node_id);

        if (!ptr) {
            perror("NumaAllocator::allocate - numa_alloc_onnode");
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    static void deallocate(T* ptr, size_t n_elements, int node_id) {
        if (ptr) {
            numa_free(ptr, n_elements * sizeof(T));
        }
    }
};
