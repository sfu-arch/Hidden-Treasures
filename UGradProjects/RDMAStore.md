# DEX–Outback: A Split-Index for One-RTT Point Lookups and High-Throughput Range Scans on Disaggregated Memory

## 1. Motivation

Modern disaggregated memory systems, accessed via **RDMA** or **CXL**, force a tradeoff:

- **Hash-based stores** (e.g., Outback) excel at *point lookups* by precomputing offsets client-side, enabling **one-round-trip RDMA**.  
- **Tree-based stores** (e.g., DEX) excel at *range scans* by laying out sorted leaves for sequential RDMA streaming, but point lookups incur extra metadata hops.

Applications in **transactional databases, analytics engines, and key–value stores** often demand *both* access modes. For example:
- **Hybrid OLTP/OLAP workloads**: transactions issue many small point reads, but analytical queries scan ranges.  
- **Time-series stores**: hot inserts and point lookups, plus periodic range aggregations.  
- **Graph stores**: point fetches for neighbors, plus scans across adjacency lists.  

We propose **DEX–Outback**, a split-index design combining **Dynamic Minimal Perfect Hashing (DMPH)** from Outback with **partitioned range-leaf layouts** from DEX.

---

## 2. Core Idea

**Split index architecture:**
- **Compute-side (client-resident):**  
  - Bucketized **DMPH directory** maps key → `(leaf_id, slot|miss)` without remote I/O.  
  - Local seed tables (few bits per key) are disseminated after bucket rebuilds.  

- **Memory-side (disaggregated memory pool):**  
  - **DEX-style sorted leaf segments**, laid out contiguously for RDMA streaming.  
  - **Delta logs** per leaf absorb writes (append-only).  
  - **Compactor** periodically merges deltas into new leaves and rebuilds optional in-leaf MPHFs.  

**Result:**  
- **One RTT point GET** (client computes address, issues RDMA read).  
- **Streaming range SCANs** (start at fence, RDMA-read contiguous leaves).  
- **Cheap updates** (append-only; background compaction).  

---

## 3. System Design

### 3.1 Data Layout (Memory Pool)
- **Leaf segments:**  
  - Sorted key-value pairs, sized to align with 2–8 RDMA MTUs.  
  - Each has a header (epoch, length, fence key).  
- **Delta logs:**  
  - Append-only, per-leaf. RDMA write appends.  
  - Background compactor merges logs into new leaf segment.  
- **Optional in-leaf MPHFs:**  
  - Provide O(1) slot offset lookup inside each leaf for exact positioning.

### 3.2 Client-Side DMPH
- **Bucketization:** `h1(k) → bucket`.  
- **Bucket-local MPHF:** deterministic, compact (few bits/key).  
- **Insertions/deletions:** localized to one bucket; rebuild by disseminating new seeds.  
- **Fallback:** if key absent, client probes delta head via a small RDMA read.

### 3.3 Operations

**Point GET(k):**
```text
1. Client computes (leaf_id, pos) via DMPH.
2. RDMA read directly from leaf[pos].
3. Optionally check delta head if key may be in log.
````

**Range SCAN(\[l, r]):**

```text
1. Client maps l → leaf_id (via DMPH or fence table).
2. RDMA-stream leaf segments sequentially until r.
```

**PUT/DELETE(k, v):**

```text
1. Append entry to leaf’s delta log (RDMA write).
2. Background compactor merges logs into sorted leaf, rebuilds MPHF if enabled.
3. Publish new leaf via epoch CAS; retire old leaf after grace period.
```

---

## 4. Concurrency and Synchronization

* **Leaf epochs:** Each leaf has a version word (coherent line or atomic CAS).
* **Copy-on-write compaction:** Writer builds new leaf offline, installs via CAS.
* **Bucket-local rebuilds:** Only the affected bucket’s MPHF seeds need refresh.
* **Split/merge handling:** If a leaf grows/shrinks, adjust fences and rebuild affected buckets.

---

## 5. Evaluation Plan

### Workloads

* **YCSB** mixes A/B/C/F with scan fractions {0, 5, 20, 50%}.
* **Zipf skew** θ ∈ \[0.0, 1.0].
* **Burst tests:** compaction storms, hot-leaf churn.
* **Hybrid workloads:** OLTP (point-heavy) + OLAP (range-heavy).

### Baselines

* **DEX** (pure range-optimized).
* **Outback** (pure DMPH point-optimized).
* **HiStore** (hash+ordered hybrid for RDMA).

### Metrics

* Median/99p GET latency.
* SCAN throughput (MB/s).
* Ops/sec under mixed workloads.
* Network bytes/op.
* Rebuild & compaction overhead.
* Tail latency under churn/skew.

### Platforms

* **RDMA**: 100–200 Gb RoCE/IB cluster.
* **CXL** (if available): load/store access, measure coherence constraints and epoch CAS costs.

---

## 6. Anticipated Results

* **Point-heavy mixes:** Expect ≥1.5–5× improvement vs. DEX, guided by Outback’s reported gains.
* **Scan-heavy mixes:** Comparable throughput to DEX.
* **Mixed workloads:** Outperforms either pure approach, achieving balanced point + range performance.
* **Space efficiency:** Metadata cost \~2–4 bits/key for DMPH + small fence arrays.

---

## 7. Risks and Mitigations

* **Bucket hot-spotting:** Use extendible hashing + stash.
* **Compaction interference:** Use epochs and copy-on-write to isolate readers.
* **Churn under skew:** Combine with lease-based software coherence or coherence credits when running atop CXL pods.

---

## 8. Contributions

1. A **split-index design** combining Outback-style DMPH with DEX leaf partitioning.
2. Protocols for **one-RTT point lookups and high-throughput range scans** in the same system.
3. Mechanisms for **localized updates**: delta logs, bucket-local rehash, epoch-based compaction.
4. **Empirical study** showing balanced performance across point-heavy, range-heavy, and mixed workloads.

---

## 9. Next Steps

* Implement client DMPH directory and seed distribution.
* Build leaf store with deltas + compactor.
* Prototype in RDMA cluster; then extend to CXL pod simulation.
* Evaluate under controlled workloads, then scale to TPC-C/YCSB mixes.

            +--------------------------------------+
            |         Client (Compute Node)        |
            |--------------------------------------|
            |  Dynamic Minimal Perfect Hash (DMPH) |
            |  - Bucketized directory              |
            |  - Seeds (few bits per key)          |
            |  - Maps key -> (leaf_id, slot|miss)  |
            +-------------------+------------------+
                                |
                                | one RDMA read/write
                                v
            +--------------------------------------+
            |  Disaggregated Memory (CXL/RDMA)     |
            |--------------------------------------|
            |   Leaf Segments (sorted runs)        |
            |   +------------------------------+   |
            |   | Fence Key | Epoch | Payload |   |
            |   +------------------------------+   |
            |                                      |
            |   Delta Logs (append-only)           |
            |   +------------------------------+   |
            |   | Append(k,v) → log entry      |   |
            |   +------------------------------+   |
            |                                      |
            |   Background Compactor               |
            |   - Merges delta + leaf              |
            |   - Rebuilds optional in-leaf MPHF   |
            |   - Publishes new epoch via CAS      |
            +--------------------------------------+
