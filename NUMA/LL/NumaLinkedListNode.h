// File: NumaLinkedListNode.h
#pragma once

#include "NumaAllocator.h"
#include "OptimisticLock.h"
#include <mutex>

using KeyType = int64_t;
using ValueType = int64_t;

struct NumaLinkedListNode {
    KeyType key;
    ValueType value;
    NumaLinkedListNode* next;
    int allocation_node_id;

    mutable OptimisticLock opt_lock; // For optimistic reads of value AND versioning changes to 'next'
    mutable std::mutex node_mutex;   // For exclusive access to value updates AND 'next' pointer updates

    NumaLinkedListNode(KeyType k, ValueType v, int alloc_node, NumaLinkedListNode* nxt = nullptr)
        : key(k), value(v), next(nxt), allocation_node_id(alloc_node) {}
};
