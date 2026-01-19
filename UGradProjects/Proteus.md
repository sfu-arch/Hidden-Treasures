# Software-Defined Sparsity on a Software-Defined Vector Fabric: A Paradigm Shift for Irregular Tensor Computation


## Abstract

The relentless scaling of deep neural networks and scientific datasets has precipitated a fundamental architectural crisis: the divergence between the irregularity of sparse data and the rigidity of high-performance hardware. While sparsity offers a theoretical reduction in arithmetic complexity, realizing this gain in silicon has historically required "Hardware-Defined Sparsity"—specialized accelerators with fixed intersection logic, rigid dataflow constraints, and format-specific memory controllers. This report proposes a radical departure from this orthodoxy: **Software-Defined Sparsity on a Software-Defined Vector Fabric (SDS-SDSF)**. By leveraging a malleable grid of processing elements that can dynamically aggregate into logical vector engines, and coupling this with a compiler-driven sparsity management layer utilizing Compressed Frames and Decoupled Runahead Scheduling, the proposed architecture, **Proton-SDS**, achieves the efficiency of vector processing with the flexibility of general-purpose scalar execution. This document provides an exhaustive survey of the related work—ranging from the foundational Rockcress architecture to state-of-the-art sparse accelerators like ExTensor and Gamma—and articulates the novelty of the SDS-SDSF paradigm in addressing the challenges of variable sparsity length and irregular memory latency.


---

## 1. The Sparsity-Efficiency Paradox


### 1.1 The End of Regularity

Deep learning models, particularly in the domains of Graph Neural Networks (GNNs), Recommendation Systems, and Mixture-of-Experts (MoE) Large Language Models (LLMs), are inherently sparse. They contain vast tensors where the majority of elements are zero. Processing these zeros wastes energy and cycles, yet skipping them introduces irregularity. The indices of non-zero elements are not known until runtime, requiring indirect memory accesses (A[i]]) and data-dependent control flow that shatters the efficiency of wide SIMD units.


### 1.2 The Hardware Lottery and the Fragmentation of Accelerators

In response to this efficiency gap, the industry has turned to Domain-Specific Architectures (DSAs). A plethora of sparse accelerators—such as ExTensor <sup>1</sup>, SpArch <sup>2</sup>, and Gamma <sup>3</sup>—have been proposed. Each of these designs effectively wins a specific "hardware lottery," optimizing for a particular combination of:



1. **Sparsity Format:** Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), Coordinate List (COO), or hierarchical bitmaps.
2. **Sparsity Density:** High sparsity (hypersparse graphs) vs. moderate sparsity (pruned neural networks).
3. **Dataflow:** Inner product, outer product, or row-wise merging.

The limitation of this approach is evident: a hardware accelerator optimized for the row-wise merging of hypersparse matrices (like Gamma) performs sub-optimally on the inner-product-heavy workloads of dense-sparse matrix multiplication. The logic required to intersect compressed fiber trees (as in ExTensor) is "baked" into the silicon. If a new compression format emerges—driven perhaps by a new algorithmic insight in machine learning—the hardware cannot adapt.


### 1.3 The Thesis: Software-Defined Adaptation

This report argues for a return to flexibility without sacrificing performance. The convergence of **Software-Defined Vector Architectures** (like Rockcress <sup>4</sup>) and **Tensor Compilers** (like TACO <sup>5</sup> and COMET <sup>6</sup>) creates a unique opportunity. Instead of building the sparsity logic into the hardware, we can define the sparsity traversal in software and execute it on a hardware fabric that reconfigures itself to match the "shape" of the computation. This is the essence of **Software-Defined Sparsity on a Software-Defined Vector Fabric**.


---

## 2. Theoretical Foundations and Related Work

To contextualize the proposal, one must rigorously examine the evolution of vector processing, the landscape of current sparse accelerators, and the software compilers that drive them. This section surveys the literature, highlighting the specific mechanisms that Proton-SDS builds upon or diverges from.


### 2.1 Software-Defined Vector Architectures

The concept of decoupling the logical vector instruction from the physical execution resources is the bedrock of the SDSF.


#### 2.1.1 Rockcress: The Architectural Substrate

The primary antecedent for the proposed vector fabric is **Rockcress** <sup>4</sup>, developed by researchers at Cornell. Rockcress addresses the limitations of standard manycore processors, which excel at Thread-Level Parallelism (TLP) but struggle with Data-Level Parallelism (DLP) due to the overhead of managing thousands of independent threads.



* **Mechanism:** Rockcress introduces the concept of "composable cores." It utilizes a standard mesh of RISC-V scalar cores but augments them with a dedicated **Instruction Forwarding Network (inet)**. At runtime, software can group a cluster of adjacent cores into a "Vector Group." One core acts as the "Commander" (or instruction fetch unit), while the neighbors act as "Workers" (or vector lanes).<sup>4</sup>
* **Instruction Forwarding:** The Commander fetches a single instruction stream and broadcasts it via the inet to the Workers. This amortizes the fetch and decode energy across the group, effectively synthesizing a SIMD engine from MIMD building blocks.<sup>4</sup>
* **Significance for Sparsity:** Rockcress demonstrates that vector width need not be a design-time constant. It can be a runtime variable. However, Rockcress focuses primarily on *dense* vectors. Its synchronization mechanism, based on barrier flags and naive Decoupled Access/Execute (DAE), is designed for regular memory access patterns.<sup>9</sup> It lacks specific support for the irregular, data-dependent divergence inherent in sparse tensor algebra.
* **Snippet Analysis:** As noted in snippet <sup>4</sup>, Rockcress tracks readiness at a "frame" granularity. A frame is a block of data in the scratchpad. This coarse-grained synchronization is key to hiding latency, a concept Proton-SDS expands upon.


#### 2.1.2 Vector Lane Threading (VLT)

The problem of "Variable Sparsity Length"—where different rows in a sparse matrix have varying numbers of non-zeros—was presaged by work on **Vector Lane Threading (VLT)**.<sup>10</sup>



* **Concept:** Standard vector processors mask off lanes when the vector length is shorter than the hardware width, wasting resources. VLT proposes allowing idle vector lanes to execute independent scalar threads or short vectors from other contexts.<sup>10</sup>
* **Relevance:** This is a crucial precedent for "Elastic Lanes." In sparse computing, load imbalance is endemic. VLT proves that breaking the lock-step rigidness of SIMD lanes to handle diverse workloads is architecturally viable. Proton-SDS adapts this by letting the *software scheduler* (the compiler) explicitly assign different "short rows" to different lanes within a vector group.


#### 2.1.3 Libra and Flexible SIMD

**Libra** <sup>12</sup> explores the configuration of SIMD execution for mobile workloads. It uses a flexible network to augment SIMD, allowing for "tailoring" the execution to the loop characteristics (ILP vs. DLP).



* **Insight:** Libra highlights the energy benefits of right-sizing the compute resources. If a sparse kernel has low arithmetic intensity, powering up a massive vector width is wasteful. Libra’s approach of dynamic configurability aligns with the "Software-Defined" ethos, allowing the runtime to match the vector group size to the sparsity density of the current tensor tile.<sup>13</sup>


#### 2.1.4 The Cray X1 Legacy

The **Cray X1** <sup>15</sup> implemented Multi-Streaming Processors (MSPs) by ganging four Single-Streaming Processors (SSPs). This hardware-heavy approach to aggregation provided immense bandwidth but lacked the fine-grained malleability of Rockcress.



* **Lesson:** The X1 showed that high bandwidth and decoupled memory (vector caches) are non-negotiable for feeding aggregated vector units. The Proton-SDS proposal modernizes this by replacing the crossbar-based ganging with a mesh-based inet, reducing the hardware overhead while maintaining the bandwidth scaling.<sup>17</sup>


### 2.2 The Landscape of Hardware-Defined Sparsity

To articulate novelty, one must contrast the proposal with the rigid accelerators that currently dominate research.


#### 2.2.1 ExTensor: Hierarchical Intersection

**ExTensor** <sup>1</sup> is designed around the principle of "intersection." It views tensors as hierarchical trees (based on the Coordinate format) and intersects them to find non-zero collisions.



* **Mechanism:** It decouples metadata scanning from compute. Specialized "Scanner" units traverse the directory structure of the tensor, feeding coordinate streams to an "Intersection" unit.
* **Rigidity:** ExTensor is bound to the hierarchical format. If the software wants to use a flat bitmap format or a blocked format not supported by the scanners, the hardware is useless. The "Scanner" logic is hardwired state machines. In Proton-SDS, scanning is performed by the Scalar Core running compiled code, making it infinitely programmable.<sup>1</sup>


#### 2.2.2 SpArch: The Merger Bottleneck

**SpArch** <sup>2</sup> tackles the matrix multiplication output problem. When multiplying sparse matrices, partial results are generated unpredictably. SpArch uses a high-radix hardware merger to condense these partials.



* **Rigidity:** The merger tree has a fixed width. If the number of partial products exceeds this width (a common occurrence in "power-law" graphs), the system must stall and spill to memory. SpArch essentially hardcodes the "Outer Product" dataflow logic.<sup>22</sup> Proton-SDS handles merging via software-scheduled vector reductions, which can spill to scratchpad memory flexibly without pipeline stalls.


#### 2.2.3 Gamma and the Gustavson Algorithm

**Gamma** <sup>3</sup> is optimized for Gustavson’s algorithm, which accumulates full rows of the output matrix to avoid the partial sum explosion of outer products.



* **Mechanism:** It uses a "FiberCache" and Processing Elements (PEs) that walk along fibers (rows) of the input matrices.
* **Rigidity:** Gamma is locked into row-major traversal. It struggles with "Inner Product" reuse patterns that might be more efficient for certain matrix shapes (e.g., tall and skinny matrices). Snippet <sup>24</sup> (Flexagon) explicitly notes this limitation, stating that Gamma degrades when the dataflow preference shifts.


#### 2.2.4 SIGMA: Flexible Distribution

**SIGMA** <sup>25</sup> uses a "Forwarding Adder Network" (FAN) to route non-zeros to PEs. It represents a move toward flexibility compared to SpArch.



* **Constraint:** SIGMA relies on **Bitmap** compression for metadata distribution. It requires a global controller to decode bitmaps and map them to the FAN. This centralized control becomes a bottleneck and limits the architecture to bitmap-friendly sparsity (structured sparsity). Proton-SDS’s distributed control (via Scalar Commanders) avoids this bottleneck.<sup>25</sup>


#### 2.2.5 SparTen: Handling Load Imbalance

**SparTen** <sup>27</sup> focuses on CNNs and introduces "Greedy Balancing" to handle the variable number of non-zeros in different filters.



* **Mechanism:** It uses heavy hardware logic to perform online load balancing, swapping work between units.
* **Relevance:** SparTen highlights the "Variable Sparsity Length" challenge. However, it solves it with reactive hardware. Proton-SDS solves it with *proactive software scheduling* (Decoupled Runahead), where the scalar core sees the imbalance ahead of time and packs the frames accordingly.<sup>29</sup>


### 2.3 Compilers: The Enablers of Software Definition

The shift to software-defined sparsity is powered by advances in tensor compilers.


#### 2.3.1 TACO: The Tensor Algebra Compiler

**TACO** <sup>5</sup> provides the theoretical framework for generating code that iterates over sparse formats.



* **Mechanism:** TACO defines "Format Descriptors" (dense, compressed, singleton) and generates the nested loops to traverse them.
* **Integration:** Proton-SDS is designed to be the *target* for TACO. Instead of generating scalar C code (which runs slowly on a CPU), a modified TACO backend generates "Frame Descriptors" and "Vector Micro-ops" for the SDSF. The "Split" and "Reorder" transformations in TACO <sup>32</sup> map directly to the dimensioning of the vector groups.


#### 2.3.2 MLIR: Multi-Level IR

**MLIR** <sup>33</sup> enables progressive lowering. The sparse_tensor dialect allows the compiler to reason about sparsity at a high level before lowering to hardware specifics.



* **Relevance:** This allows Proton-SDS to support a wide variety of frontend languages (Python, PyTorch) by hooking into the MLIR pipeline. The sparse_tensor encoding attributes can be mapped to the Frame Configuration registers of the Scalar Core.<sup>35</sup>


#### 2.3.3 COMET: Domain-Specific Optimization

**COMET** <sup>6</sup> performs high-level optimizations for tensor contractions, often finding "Block Sparsity" in scientific codes.



* **Synergy:** When COMET identifies a block structure (e.g., $4 \times 4$ blocks), it can instruct the SDSF to configure vector groups of size 4 or 16. This perfect matching of hardware granularity to data granularity is the "Holy Grail" of efficiency that fixed-width SIMD (AVX-512) cannot achieve.<sup>36</sup>


### 2.4 Decoupled Execution and Runahead

Addressing the memory latency of indirect access requires advanced microarchitectural techniques.


#### 2.4.1 Decoupled Access/Execute (DAE)

DAE architectures <sup>37</sup> split execution into an Access stream and an Execute stream.



* **In Rockcress:** DAE is used for dense frames.<sup>9</sup> The Access core loads data; the Execute core computes.
* **Evolution:** Proton-SDS extends this to *Dependent* Access. The Access stream involves pointer chasing.


#### 2.4.2 Vector Runahead

**Vector Runahead** <sup>40</sup> allows an out-of-order processor to speculatively execute dependent chains and reorder them into vector loads.



* **Mechanism:** It creates a "Runhead" thread that runs ahead of the commit point, warming up the cache.
* **Novelty in Proton-SDS:** We employ a **Constructive Runahead**. The scalar core (Runahead engine) does not just warm the cache; it writes the valid data into the Compressed Frame. The work is saved, not discarded. This efficiency is crucial for energy-constrained manycore environments.<sup>40</sup>


---

## 3. Proposal: The Proton-SDS Architecture


### 3.1 Design Philosophy: The "Proton" Concept

Named after the shape-shifting Greek sea god, the **Proton-SDS** (Software-Defined Sparsity) architecture is built on the premise that hardware should provide a **malleable fabric** of raw compute and memory bandwidth, while software should define the **topology** of the data flow. This stands in direct contrast to the rigid "pipelines" of ExTensor or SpArch.

The architecture addresses three critical requirements for sparse computing:



1. **Flexibility:** Support for any sparsity format (current or future).
2. **Efficiency:** High utilization of floating-point units despite data irregularity.
3. **Latency Hiding:** Mitigating the cost of indirect memory accesses (col_ind).


### 3.2 System Architecture Overview

Proton-SDS is a tiled manycore architecture comprising a 2D mesh of **Proton Tiles**.


#### 3.2.1 The Proton Tile

Each tile contains:



* **Scalar Core (SC):** A lightweight RISC-V core (RV64GC) modified with the "Runahead Extension."
* **Vector Unit (VU):** A dense FPU array capable of executing vector micro-ops.
* **Scratchpad Memory (SPM):** A banked SRAM, partitioned into "Frame Buffers."
* **Instruction Forwarding Unit (IFU):** The interface to the inet.
* **Router:** Connecting the tile to the mesh network.


#### 3.2.3 Proton ISA Extensions

The Proton-SDS architecture extends the base RISC-V (RV64GC) ISA with custom instructions for vector group management, frame operations, and decoupled runahead scheduling. The table below summarizes the new instructions:

| **Category** | **Instruction** | **Format** | **Description** |
|--------------|-----------------|------------|-----------------|
| **Group Configuration** | `CFG_GROUP` | `CFG_GROUP size, topology` | Configures a Virtual Vector Engine by aggregating `size` adjacent tiles. The Commander core broadcasts instructions; Workers become vector lanes. |
| | `SPLIT_GROUP` | `SPLIT_GROUP factor` | Splits the current group into `factor` smaller sub-groups (for hypersparse regions). |
| | `MERGE_GROUP` | `MERGE_GROUP target_size` | Merges adjacent groups into a larger vector engine (for dense phases). |
| | `RELEASE_GROUP` | `RELEASE_GROUP` | Dissolves the current vector group, returning Workers to independent scalar mode. |
| **Frame Management** | `FRAME_ALLOC` | `FRAME_ALLOC frame_id, size` | Allocates a Compressed Frame buffer in the scratchpad with the specified ID and size. |
| | `FRAME_FREE` | `FRAME_FREE frame_id` | Releases a previously allocated frame buffer. |
| | `FRAME_SET_HDR` | `FRAME_SET_HDR frame_id, vlen, mask_fmt, opcode` | Sets the frame header: vector length, mask format, and associated micro-op. |
| | `FRAME_CHAIN` | `FRAME_CHAIN frame_id, next_frame_id` | Links frames for rows exceeding vector width; sets CONTINUATION bit. |
| **Runahead & Gather** | `GATHER` | `GATHER base, index, frame_id, lane_id` | Non-binding prefetch-and-place: fetches `base[index]` and writes result to specified frame/lane slot. |
| | `GATHER_INDIRECT` | `GATHER_INDIRECT base, index_stream, frame_id, mask` | Batch gather for sparse access: fetches `base[index_stream[i]]` for lanes specified by mask. |
| | `GATHER_STRIDE` | `GATHER_STRIDE base, stride, count, frame_id` | Strided gather for semi-regular patterns; places `count` elements into frame. |
| **Scheduling & Sync** | `QUEUE_VEC_OP` | `QUEUE_VEC_OP opcode, frame_id, mask` | Enqueues a vector operation to execute on the specified frame when data is ready. |
| | `FRAME_WAIT` | `FRAME_WAIT frame_id` | Blocks Commander until the specified frame is marked Ready by the scoreboard. |
| | `FRAME_POLL` | `FRAME_POLL frame_id, rd` | Non-blocking check; writes 1 to `rd` if frame is ready, 0 otherwise. |
| | `SYNC_BARRIER` | `SYNC_BARRIER group_id` | Synchronization barrier across all tiles in the specified vector group. |
| **Vector Micro-ops** | `V_LOAD` | `V_LOAD vd, frame_id, buffer_type` | Loads vector register `vd` from frame's Value_Buf or Gathered buffer. |
| | `V_STORE` | `V_STORE vs, frame_id, buffer_type` | Stores vector register `vs` to frame buffer. |
| | `V_FMA` | `V_FMA vd, vs1, vs2` | Fused multiply-add on vector registers (masked by frame's Valid_Mask). |
| | `V_REDUCE` | `V_REDUCE vd, vs, op` | Horizontal reduction (sum, max, etc.) within active lanes; result to `vd[0]`. |
| | `V_COMPRESS` | `V_COMPRESS vd, vs, mask` | Compresses active elements of `vs` (per mask) into contiguous positions in `vd`. |
| | `V_EXPAND` | `V_EXPAND vd, vs, mask` | Expands contiguous elements of `vs` into positions specified by mask in `vd`. |
| **Metadata Primitives** | `V_AND_POPC` | `V_AND_POPC rd, vs1, vs2` | Bitwise AND of two mask vectors; returns popcount to scalar `rd`. |
| | `V_PREFIX_SUM` | `V_PREFIX_SUM vd, vs` | Computes prefix sum on mask/index vector (for lane packing calculations). |
| | `SCHEDULE_LANES` | `SCHEDULE_LANES rd, length, policy` | Software-assisted lane scheduling: returns optimal mask for packing `length` elements. |

**Notes:**
- All frame-related instructions operate on the local Scratchpad Memory (SPM).
- The Frame Scoreboard hardware tracks Ready/Valid status and updates masks atomically as GATHER operations complete.
- Vector micro-ops inherit masking from the frame's Valid_Mask unless explicitly overridden.
- The metadata primitives (`V_AND_POPC`, `V_PREFIX_SUM`) are optional accelerants that allow the Commander to perform traversal/packing at near-hardware speed while remaining format-agnostic.


#### 3.2.2 The Software-Defined Vector Fabric (SDSF)

Unlike a static GPU with fixed "Warps," Proton-SDS creates **Virtual Vector Engines** at runtime.



* **Group Formation:** The compiler analyzes the workload. If the kernel operates on dense $16 \times 16$ blocks, it emits a configuration instruction: CFG_GROUP(size=16).
* **The Commander:** One Scalar Core is designated the "Commander." It fetches the instruction stream.
* **The Workers:** 15 adjacent tiles are designated "Workers." Their Scalar Cores enter a sleep state (clock-gated), and their Vector Units become slaves to the Commander's IFU.
* **Elasticity:** If the sparsity pattern changes (e.g., entering a hypersparse region), the compiler can issue a reconfiguration to split the 16-core group into 4 groups of 4 cores. This **Dynamic Dimensioning** ensures that hardware parallelism always matches the available data parallelism, a feature missing in architectures like SpArch or NVIDIA GPUs.<sup>4</sup>


### 3.3 The "Compressed Frame" Abstraction

The central innovation for handling **Variable Sparsity Length** is the **Compressed Frame**. In standard DAE (like Rockcress <sup>9</sup>), a frame is a dense block of data. In Proton-SDS, a Compressed Frame is a structured container that decouples the *topology* of the data from the *values*.

**Frame Structure:**

| **Field** | **Description** |
|-----------|----------------|
| **Header** | Contains metadata: Vector_Length, Mask_Format, Opcode. |
| **Value_Buf** | Contiguous array of floating-point values (the Non-Zeros). |
| **Coord_Buf** | (Optional) Array of indices (column indices for SpGEMM). |
| **Valid_Mask** | A flexible bitmask indicating which vector lanes are active. |
| **Next_Ptr** | Pointer to the next frame (for chaining long rows). |

**Mechanism:**

The Compressed Frame acts as the interface between the Software-Defined Sparsity (running on the Commander) and the Hardware Execution (running on the SDSF). The hardware vector units do not need to understand CSR or Blocked-ELL. They simply execute based on the Valid_Mask and Value_Buf populated by the Commander.


### 3.4 Decoupled Runahead Scheduling (DRS)

To handle the irregular latency of sparse accesses, Proton-SDS introduces a novel scheduling mechanism.


#### 3.4.1 The Challenge of Indirection

In a sparse matrix-vector multiplication (SpMV): y[row] += val[k] * x[col[k]].

The access to x depends on loading col[k]. This dependency chain stalls standard pipelines.


#### 3.4.2 The DRS Solution



1. Runahead Phase (Commander Core): \
The Commander runs a "Skeleton Program." This program contains only the control flow and index calculations of the sparse loop. It strips out the floating-point math.
    * It loads col[k].
    * It computes the address &x[col[k]].
    * It issues a **non-binding prefetch-and-place** instruction: GATHER(addr, frame_id, lane_id).
2. The Frame Scoreboard: \
The hardware memory controller receives the GATHER request. It fetches the data from the L2 cache or DRAM. Crucially, it places the result directly into the specified frame_id and lane_id slot in the Worker's Scratchpad.
    * The Valid_Mask for that frame is updated atomically as data arrives.
3. Execute Phase (Vector Fabric): \
The Vector Units poll the Frame Scoreboard. When a frame is marked "Ready" (meaning all gathered data has arrived), the Vector Unit executes the compute micro-ops (e.g., FMA) on the buffer.

**Novelty:** This decouples the *latency of the irregularity* (handled by the Commander's runahead) from the *throughput of the math* (handled by the Vector Fabric). The Commander can run thousands of cycles ahead, filling frames for future execution, effectively hiding the memory wall.<sup>40</sup>


### 3.5 Handling Variable Sparsity Length

A major inefficiency in SIMD is "ragged rows"—one row has 2 elements, another has 20.



* **Elastic Lane Mapping:** The Commander's software scheduler (compiler-generated) detects these imbalances. It can pack multiple short rows into a single frame (e.g., Lane 0-3 do Row A, Lane 4-7 do Row B).
* **Frame Chaining:** For a row that exceeds the vector width, the scheduler spans it across multiple frames. The CONTINUATION bit in the Frame Header tells the Vector Unit to preserve the accumulator state between frames.
* **Result:** The Vector Fabric sees a continuous stream of fully packed frames, oblivious to the underlying jaggedness of the sparse matrix. This software-driven packing achieves load balancing without the complex, area-hungry "work stealer" hardware found in SparTen.<sup>29</sup>


---

## 4. Detailed Novelty Analysis

The claim of novelty rests on the specific comparison with the state-of-the-art architectures surveyed in Section 2.


### 4.1 Comparison Table

| **Feature** | **ExTensor / SpArch** | **NVIDIA GPU (SIMT)** | **Rockcress** | **Proton-SDS** |
|-------------|----------------------|----------------------|---------------|-----------------|
| **Sparsity Logic** | Fixed Hardware Units (Intersectors/Mergers) | Software (CUDA Threads) | N/A (Dense only) | **Software-Defined (Runahead Commander)** |
| **Format Support** | Rigid (Specific Format required) | Flexible (Software) | N/A | **Flexible (Any Format via Compiler)** |
| **Vector Length** | Fixed (Architecture constant) | Fixed (Warp Size = 32) | Variable (Dense only) | **Variable (Dynamic Grouping)** |
| **Irregularity Handling** | Hardware Stalls / Padding | Divergence (SIMT Stack serialization) | N/A | **Decoupled Runahead (Latency Hiding)** |
| **Memory Access** | Coupled (Pipeline Stalls) | Coupled (Memory Coalescing needed) | Decoupled (Dense Frames) | **Decoupled (Gather-to-Frame)** |



### 4.2 Novelty Statement 1: The Separation of Topology and Value

Existing accelerators (ExTensor) couple the topology (metadata) and value processing in monolithic hardware pipelines. Proton-SDS introduces the **Compressed Frame** as an architectural cut. By processing topology on a scalar runahead core and values on a vector fabric, we achieve the flexibility of a CPU with the throughput of a TPU. This architecture allows "Late Binding" of the sparsity format—the hardware doesn't know if it's processing CSR or DCSR until the software loads the frame.


### 4.3 Novelty Statement 2: Elastic Vector Dimensioning

While Rockcress introduced the idea of composable cores, it applied it to dense loops. Proton-SDS applies it to **Sparsity Density**.



* In **Hypersparse** workloads (GNNs), Proton-SDS configures itself as a massive array of tiny vector engines (e.g., 256 groups of 4 cores).
* In Dense phases (MLP layers), it reconfigures as a small array of massive vector engines (e.g., 16 groups of 64 cores). \
This dynamic adaptation is impossible in fixed-array architectures like SIGMA or rigid SIMD architectures like GPUs.


### 4.4 Novelty Statement 3: Constructive Runahead for Sparsity

Standard Runahead (as discussed in <sup>40</sup>) is speculative and often discards work. Proton-SDS's **Decoupled Runahead** is constructive. The address calculation performed by the Commander is effectively "saved" by issuing the GATHER to the frame. The Vector Unit never re-calculates the address; it simply consumes the data. This significantly reduces total energy per operation compared to standard OoO Runahead.


---

## 5. Software Stack and Compilation Flow

The hardware is only as good as the compiler that drives it. The Proton-SDS stack leverages the modern ecosystem of MLIR and TACO.


### 5.1 The Compiler Pipeline



1. **High-Level IR (MLIR/PyTorch):** The user defines the computation in a high-level framework. The tensor formats are annotated (e.g., #CSR, #DCSR).
2. **Sparsity Analysis (TACO):** The TACO engine <sup>5</sup> analyzes the iteration graph. It generates the "Sparsity Lattice" that defines how to iterate over the indices.
3. **The SDSF Backend (Custom MLIR Dialect):**
    * **Phase Analysis:** The compiler identifies phases of computation. It inserts CFG_GROUP instructions to dimension the vector fabric.
    * **Runahead Slicing:** The compiler slices the loop into two:
        * The **Commander Slice** (Metadata & Control): Contains row_ptr traversal and GATHER emission.
        * The **Worker Slice** (Compute): Contains the math operations acting on abstract Frame Buffers.
4. **Optimization - Lane Packing:** The compiler uses profiling data or heuristics to estimate the non-zero distribution. It generates the scheduling logic for the Commander to pack multiple short rows into frames efficiently.


### 5.2 Example: Compiling SpMV (CSR Format)

**Original Kernel:**

```c
for (i = 0; i < N; i++) {
    for (j = row_ptr[i]; j < row_ptr[i+1]; j++) {
        y[i] += val[j] * x[col[j]];
    }
}
```


**Commander Slice (Scalar Core):**

```c
Frame_Alloc(ID=0);
for (i = 0; i < N; i++) {
    start = row_ptr[i]; end = row_ptr[i+1];
    len = end - start;
    // Decision: Pack into current frame or start new?
    mask = Schedule_Lanes(len);
    // Issue Async Gather
    GATHER_INDIRECT(base=x, index_stream=&col[start],
                    frame=0, mask=mask);
    // Queue compute instruction
    QUEUE_VEC_OP(OP_FMA, frame=0, mask=mask);
}
```


**Worker Slice (Vector Fabric):**

```asm
; (Hardware FSM) Wait for Frame 0 Ready
V_LOAD   V0, Frame.Values    ; Data from val
V_LOAD   V1, Frame.Gathered  ; Data from x
V_FMA    V2, V0, V1
V_REDUCE V2                  ; if end of row
```


---

## 6. Hypothetical Evaluation and Methodology

To validate the Proton-SDS architecture, we propose a rigorous evaluation methodology extending the infrastructure used in the Rockcress and ExTensor studies.


### 6.1 Simulation Environment

We utilize **gem5-mesh** <sup>9</sup>, the fork used for Rockcress, as the baseline.



* **Modifications:** We implement the inet modifications to support variable-mask broadcasting. We model the Scratchpad Memory with "Frame Logic" extensions (Scoreboarding).
* **Energy Model:** We integrate **McPAT** for area and energy estimation, calibrating the "Scalar Core" energy against a standard RISC-V in-order core and the "Vector Unit" against a standard FPU.


### 6.2 Benchmarks

We select a diverse suite to stress the "Variable Sparsity" aspect:



1. **SuiteSparse Matrix Collection:** Real-world matrices from engineering and physics (unstructured).
2. **Graph500:** Power-law graphs (extreme irregularity).
3. **DeepBench (Baidu):** Sparse LSTM and CNN kernels (structured sparsity).
4. **GNN-Benchmark:** Graph Neural Network aggregation phases.


### 6.3 Metrics



* **Speedup:** vs. Single Thread CPU, AVX-512 CPU, and NVIDIA A100 GPU (normalized to peak BW).
* **Effective Utilization:** Percentage of FPU cycles doing useful work (not multiplying by zero, not idle).
* **Energy Efficiency:** J/Op. We expect Proton-SDS to beat GPUs on sparse workloads due to the elimination of divergence energy, and to beat CPUs due to vector amortization.


### 6.4 Anticipated Results (Based on Analytical Modeling)



* **Vs. ExTensor:** Proton-SDS may have slightly lower *peak* throughput on perfectly hierarchical data (CSF) due to the overhead of software scheduling. However, on "wild" data (arbitrary sparsity), Proton-SDS should maintain high performance while ExTensor would fall back to inefficient modes or require format conversion.
* **Vs. Rockcress:** We anticipate a 2-5x speedup on sparse kernels due to the **Decoupled Runahead** hiding the memory latency, which Rockcress's simple barrier synchronization exposes.<sup>4</sup>
* **Vs. VLT:** Compared to simple VLT, Proton-SDS's "Compressed Frame" approach allows for much larger "lookahead," enabling better memory system pipelining.


---

## 7. Challenges and Future Directions


### 7.1 The Compiler Complexity Barrier

The complexity of the Proton-SDS approach is shifted from hardware to software. Writing the backend to perform "Lane Packing" and "Runahead Slicing" is non-trivial. The reliance on **Bernstein's Exocompilation** principles <sup>41</sup> is critical—we must expose the frame abstraction to the library writer, not just the compiler writer, to allow for hand-tuned kernels where the auto-scheduler fails.


### 7.2 Scalability of the inet

As vector groups grow larger (e.g., 64 cores), the latency of the instruction distribution network increases.

* **Mitigation:** We propose **Hierarchical Groups**. A "Lieutenant" core could buffer instructions for a sub-group, reducing the load on the Commander. This is a topic for future architectural exploration.



---



---

## Summary: Comparison of Key Architectures

| **Architecture** | **Sparsity Handling** | **Vector Granularity** | **Flexibility** | **Latency Hiding** |
|------------------|----------------------|----------------------|-----------------|-------------------|
| **ExTensor** [1] | Hardware Intersection | Fixed | Low (Format Bound) | Scanner Lookahead |
| **SpArch** [2] | Hardware Merger | Fixed | Low (Outer Prod Only) | Prefetcher |
| **Gamma** [3] | Hardware Fiber Walk | Fixed | Low (Row-Wise Only) | Fiber Cache |
| **Rockcress** [4] | Software (Dense) | Variable (Dense) | High (Dense) | Simple DAE |
| **Proton-SDS** | **Software Runahead** | **Variable (Sparse)** | **Maximum** | **Decoupled Frame** |

*Note: Citations correspond to the references listed below.*


#### Works cited



1. ExTensor: An Accelerator for Sparse Tensor Algebra - Neal Crago, PhD, accessed January 17, 2026, [https://www.nealcrago.com/wp-content/uploads/ExTensor_MICRO2019.pdf](https://www.nealcrago.com/wp-content/uploads/ExTensor_MICRO2019.pdf)
2. Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling - arXiv, accessed January 17, 2026, [https://arxiv.org/pdf/2205.05826](https://arxiv.org/pdf/2205.05826)
3. Gamma: Leveraging Gustavson's Algorithm to Accelerate Sparse Matrix Multiplication - People, accessed January 17, 2026, [https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf](https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf)
4. Software-Defined Vector Processing on Manycore Fabrics - Cornell: Computer Science, accessed January 17, 2026, [https://www.cs.cornell.edu/~asampson/media/papers/SDS-micro2021-preprint.pdf](https://www.cs.cornell.edu/~asampson/media/papers/SDS-micro2021-preprint.pdf)
5. Portable Accelerated Learning - Stanford Portal Center, accessed January 17, 2026, [https://portal.stanford.edu/portal-whitepaper-short.pdf](https://portal.stanford.edu/portal-whitepaper-short.pdf)
6. pnnl/COMET - GitHub, accessed January 17, 2026, [https://github.com/pnnl/COMET](https://github.com/pnnl/COMET)
7. LOGICAL ACCELERATORS ON MANYCORE PROCESSORS - Cornell eCommons, accessed January 17, 2026, [https://ecommons.cornell.edu/bitstreams/70397a74-eb3f-4a03-b025-38f26de89631/download](https://ecommons.cornell.edu/bitstreams/70397a74-eb3f-4a03-b025-38f26de89631/download)
8. big.VLITTLE: On-Demand Data-Parallel Acceleration for Mobile Systems on Chip, accessed January 17, 2026, [https://www.csl.cornell.edu/~cbatten/pdfs/ta-big-vlittle-micro2022.pdf](https://www.csl.cornell.edu/~cbatten/pdfs/ta-big-vlittle-micro2022.pdf)
9. cucapra/gem5-mesh: Fork of gem5 with support for manycore architectures. Includes models and scripts to evaluate a software-defined-vector architecture. - GitHub, accessed January 17, 2026, [https://github.com/cucapra/gem5-mesh](https://github.com/cucapra/gem5-mesh)
10. Vector Lane Threading - CECS, accessed January 17, 2026, [https://www.cecs.uci.edu/~papers/icpp06/ICPP/papers/06_srivoire-vector.pdf](https://www.cecs.uci.edu/~papers/icpp06/ICPP/papers/06_srivoire-vector.pdf)
11. Vector Lane Threading, accessed January 17, 2026, [https://rivoire.cs.sonoma.edu/pubs/rivoire.2006.vlt.icpp.slides.pdf](https://rivoire.cs.sonoma.edu/pubs/rivoire.2006.vlt.icpp.slides.pdf)
12. Breaking SIMD Shackles with an Exposed Flexible Microarchitecture and the Access Execute PDG - Computer Sciences Dept., accessed January 17, 2026, [https://research.cs.wisc.edu/vertical/papers/2013/pact13-dyser.pdf](https://research.cs.wisc.edu/vertical/papers/2013/pact13-dyser.pdf)
13. Libra: Tailoring SIMD Execution using Heterogeneous Hardware and Dynamic Configurability PowerPoint Presentation - ID:2069126 - SlideServe, accessed January 17, 2026, [https://www.slideserve.com/shiri/libra-tailoring-simd-execution-using-heterogeneous-hardware-and-dynamic-configurability](https://www.slideserve.com/shiri/libra-tailoring-simd-execution-using-heterogeneous-hardware-and-dynamic-configurability)
14. R-Blocks: an Energy-Efficient, Flexible, and Programmable CGRA - TUE Research portal, accessed January 17, 2026, [https://research.tue.nl/files/328730999/3656642.pdf](https://research.tue.nl/files/328730999/3656642.pdf)
15. Performance evaluation of the Cray X1 distributed shared memory architecture, accessed January 17, 2026, [https://ieeexplore.ieee.org/document/1375194](https://ieeexplore.ieee.org/document/1375194)
16. A Performance Evaluation of the Cray X1 for Scientific Applications, accessed January 17, 2026, [https://ntrs.nasa.gov/api/citations/20040010771/downloads/20040010771.pdf](https://ntrs.nasa.gov/api/citations/20040010771/downloads/20040010771.pdf)
17. Optimizing Performance of Superscalar Codes For a Single Cray X1 MSP processor - OSTI, accessed January 17, 2026, [https://www.osti.gov/servlets/purl/860889](https://www.osti.gov/servlets/purl/860889)
18. TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators - People, accessed January 17, 2026, [https://people.csail.mit.edu/emer/media/papers/2023.10.micro.teaal.pdf](https://people.csail.mit.edu/emer/media/papers/2023.10.micro.teaal.pdf)
19. ExTensor: An Accelerator for Sparse Tensor Algebra | Request PDF - ResearchGate, accessed January 17, 2026, [https://www.researchgate.net/publication/336450044_ExTensor_An_Accelerator_for_Sparse_Tensor_Algebra](https://www.researchgate.net/publication/336450044_ExTensor_An_Accelerator_for_Sparse_Tensor_Algebra)
20. ExTensor: An accelerator for sparse tensor algebra – Related Work - Alastair Reid, accessed January 17, 2026, [https://alastairreid.github.io/RelatedWork/papers/hedge:micro:2019/](https://alastairreid.github.io/RelatedWork/papers/hedge:micro:2019/)
21. Stellar: An Automated Design Framework for Dense and Sparse Spatial Accelerators - People @EECS, accessed January 17, 2026, [https://people.eecs.berkeley.edu/~ysshao/assets/papers/stellar-micro2024.pdf](https://people.eecs.berkeley.edu/~ysshao/assets/papers/stellar-micro2024.pdf)
22. Spatula: A Hardware Accelerator for Sparse Matrix Factorization - DSpace@MIT, accessed January 17, 2026, [https://dspace.mit.edu/bitstream/handle/1721.1/153276/3613424.3623783.pdf?sequence=1&isAllowed=y](https://dspace.mit.edu/bitstream/handle/1721.1/153276/3613424.3623783.pdf?sequence=1&isAllowed=y)
23. Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow, accessed January 17, 2026, [https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/spada.asplos23.pdf](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/spada.asplos23.pdf)
24. Flexagon: A Multi-Dataflow Sparse-Sparse Matrix Multiplication Accelerator for Efficient DNN Processing - Universidad de Murcia, accessed January 17, 2026, [https://digitum.um.es/server/api/core/bitstreams/80c3e79a-8cfe-4204-abab-ed261af81906/content](https://digitum.um.es/server/api/core/bitstreams/80c3e79a-8cfe-4204-abab-ed261af81906/content)
25. SIGMA: A Sparse and Irregular GEMM Accelerator with Flexible Interconnects for DNN Training, accessed January 17, 2026, [https://www.cse.wustl.edu/~roger/566S.s21/09065523.pdf](https://www.cse.wustl.edu/~roger/566S.s21/09065523.pdf)
26. Sparkle: A High Efficient Sparse Matrix Multiplication Accelerator for Deep Learning, accessed January 17, 2026, [https://ieeexplore.ieee.org/document/9978530/](https://ieeexplore.ieee.org/document/9978530/)
27. Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling - People, accessed January 17, 2026, [https://people.csail.mit.edu/emer/media/papers/2022.10.micro.sparseloop.pdf](https://people.csail.mit.edu/emer/media/papers/2022.10.micro.sparseloop.pdf)
28. Sparse-TPU: Adapting Systolic Arrays for Sparse Matrices - Trevor Mudge, accessed January 17, 2026, [http://tnm.engin.umich.edu/wp-content/uploads/sites/353/2020/08/2020.6.sparse-tpu_ics2020.pdf](http://tnm.engin.umich.edu/wp-content/uploads/sites/353/2020/08/2020.6.sparse-tpu_ics2020.pdf)
29. SparTen: A Sparse Tensor Accelerator for Convolutional Neural Networks - ResearchGate, accessed January 17, 2026, [https://www.researchgate.net/publication/336450382_SparTen_A_Sparse_Tensor_Accelerator_for_Convolutional_Neural_Networks](https://www.researchgate.net/publication/336450382_SparTen_A_Sparse_Tensor_Accelerator_for_Convolutional_Neural_Networks)
30. A Tensor Algebra Compiler Library Interface and Runtime Patricio Noyola, accessed January 17, 2026, [http://tensor-compiler.org/files/noyola-meng-thesis-taco-interface.pdf](http://tensor-compiler.org/files/noyola-meng-thesis-taco-interface.pdf)
31. Sparse Tensor Algebra Compilation, accessed January 17, 2026, [http://tensor-compiler.org/files/kjolstad-phd-thesis-taco-compiler.pdf](http://tensor-compiler.org/files/kjolstad-phd-thesis-taco-compiler.pdf)
32. A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra - Fredrik Kjolstad, accessed January 17, 2026, [http://fredrikbk.com/publications/taco-scheduling.pdf](http://fredrikbk.com/publications/taco-scheduling.pdf)
33. MLIR Sparsifier - MPACT Research Group | Google for Developers, accessed January 17, 2026, [https://developers.google.com/mlir-sparsifier](https://developers.google.com/mlir-sparsifier)
34. Compiler Support for Sparse Tensor Computations in MLIR - arXiv, accessed January 17, 2026, [https://arxiv.org/pdf/2202.04305](https://arxiv.org/pdf/2202.04305)
35. [RFC] Sparse tensor support in torch-mlir - LLVM Discussion Forums, accessed January 17, 2026, [https://discourse.llvm.org/t/rfc-sparse-tensor-support-in-torch-mlir/63627](https://discourse.llvm.org/t/rfc-sparse-tensor-support-in-torch-mlir/63627)
36. COMET: A Domain-Specific Compilation of High-Performance Computational Chemistry - William & Mary, accessed January 17, 2026, [https://www.cs.wm.edu/~bren/Scholarship/4.Peer-Reviewed-Workshop-and-Posters/3.LCPC20.pdf](https://www.cs.wm.edu/~bren/Scholarship/4.Peer-Reviewed-Workshop-and-Posters/3.LCPC20.pdf)
37. A Tensor Processing Framework for CPU-Manycore Heterogeneous Systems - IEEE Xplore, accessed January 17, 2026, [https://ieeexplore.ieee.org/ielaam/43/9778249/9509755-aam.pdf](https://ieeexplore.ieee.org/ielaam/43/9778249/9509755-aam.pdf)
38. PROGRAMMING FRAMEWORKS FOR IMPROVING THE PRODUCTIVITY AND PERFORMANCE OF MANYCORE ARCHITECTURES - Computer Systems Laboratory, accessed January 17, 2026, [https://www.csl.cornell.edu/~cbatten/pdfs/chen-manycore-prog-cuthesis2022.pdf](https://www.csl.cornell.edu/~cbatten/pdfs/chen-manycore-prog-cuthesis2022.pdf)
39. (PDF) SWOOP: software-hardware co-design for non-speculative, execute-ahead, in-order cores - ResearchGate, accessed January 17, 2026, [https://www.researchgate.net/publication/329406297_SWOOP_software-hardware_co-design_for_non-speculative_execute-ahead_in-order_cores](https://www.researchgate.net/publication/329406297_SWOOP_software-hardware_co-design_for_non-speculative_execute-ahead_in-order_cores)
40. Vector Runahead, accessed January 17, 2026, [https://www.repository.cam.ac.uk/bitstreams/441f133e-613c-433a-8168-92069dd8ab9e/download](https://www.repository.cam.ac.uk/bitstreams/441f133e-613c-433a-8168-92069dd8ab9e/download)
41. Exocompilation for Productive Programming of Hardware Accelerators - DSpace@MIT, accessed January 17, 2026, [https://dspace.mit.edu/bitstream/handle/1721.1/146372/3519939.3523446.pdf?sequence=1&isAllowed=y](https://dspace.mit.edu/bitstream/handle/1721.1/146372/3519939.3523446.pdf?sequence=1&isAllowed=y)
https://www.csl.cornell.edu/~cbatten/pdfs/srinath-xloops-slides-micro2014.pdf
https://www.csl.cornell.edu/~cbatten/pdfs/srinath-xloops-slides-micro2014.pdf
---

## Appendix A: Comparison with Trapezoid (ISCA'24)

Trapezoid and Proton-SDS address the same fundamental problem—"sparsity is a continuum; don't build a one-trick accelerator"—but arrive at very different architectural solutions regarding where "flexibility" should reside.

### A.1 Trapezoid Overview

**Trapezoid** is a matrix-multiplication accelerator that maintains efficiency across the full density range by switching among specialized dataflows backed by reusable hardware blocks.

**Key characteristics:**

- **Fixed compute substrate with flexible dataflow selection:** Trapezoid uses a 2D spatial array (128×128 PEs) with sparse-handling hardware per PE row.
- **Four operational modes:**
  - Standard inner-product (IP) for Dense × Dense
  - TrIP for Mid-Sparse × Dense and Mid-Sparse × Mid-Sparse
  - TrGT (Gustavson-style) for High-Sparse × High-Sparse
  - TrGS (Gustavson-style) for High-Sparse × Mid-Sparse and High-Sparse × Dense

**Hardware mechanisms:**

- **Multi-Fiber Intersection Unit (MFIU):** Performs pairwise bitmask intersections (AND), prefix sums, and index shifting to generate routing metadata.
- **Distribution networks:** Route operands to multipliers.
- **Merge-reduction tree:** Accumulates partial results.
- **Multi-level memory hierarchy:** Row-local buffers + global cache clusters for high gather bandwidth.

### A.2 Architectural Comparison

| Dimension | Trapezoid (ISCA'24) | Proton-SDS |
|-----------|---------------------|-------------|
| **Primary target** | Dense→HS matrix multiplication | General irregular tensor kernels (SpMV/SpMM/SpGEMM/…) |
| **Flexibility mechanism** | Switch among 4 dataflows (IP/TrIP/TrGT/TrGS) | Software-defined traversal + dynamic vector grouping |
| **Mild sparsity approach** | MFIU (AND+prefix+shift) + distribution nets + reduction tree | Commander builds Compressed Frames; VUs run masked micro-ops |
| **High sparsity approach** | Gustavson dataflows + multi-level memory hierarchy for gather BW | Decoupled runahead + gather-to-frame + scoreboard readiness |
| **Output accumulation** | Merge-reduction tree + banked local buffer | Software-scheduled reductions / scratchpad accumulators |
| **Format assumptions** | Bitmasks/tiling machinery for intersection/packing | Format-agnostic via software-defined metadata traversal |
| **Adaptation granularity** | Mode switch + limited dynamic packing | Mode switch + variable vector width/topology + software packing policies |

### A.3 Key Differentiators

**Scope of generality:**
- Trapezoid is "unified across sparsity *levels* for GEMM"
- Proton is "unified across sparsity *formats + kernels*"

**Packing mechanism:**
- Trapezoid packs in hardware (MFIU → routing → multipliers)
- Proton packs in software (Commander constructs Compressed Frames)

**Latency hiding:**
- Trapezoid: bandwidth/traffic-optimal for HS matmul via cache + schedule
- Proton: latency-tolerant for irregular indirection via runahead + frame materialization

### A.4 Trade-offs

**When Trapezoid excels:**
- Structured sparse matmul-like kernels where traversal/packing can be performed efficiently by dedicated hardware
- Mid-sparsity tiled/bitmask formats where lane validity can be decided quickly in hardware
- Workloads with predictable merge/reduce patterns

**When Proton-SDS excels:**
- Diverse or evolving sparsity formats
- Kernels beyond matrix multiplication (sparse attention, GNN aggregation)
- Workloads with irregular pointer chasing / indirect gathers
- Mixed dense-sparse phases requiring dynamic vector width adaptation

### A.5 Lessons for Proton-SDS

To address potential performance gaps while maintaining the software-defined philosophy, Proton-SDS incorporates:

1. **Bitmask/compaction primitives:** The `V_AND_POPC` and `V_PREFIX_SUM` instructions allow the Commander to generate packed lane maps at near-hardware speed without hardwiring a specific format.

2. **Hierarchical gather bandwidth:** Group-local buffer levels can sustain gather-to-frame operations without creating scratchpad port contention.


It supports four main modes (their words):

* standard inner-product (IP) for **D×D**
* **TrIP** for **MS×D** and **MS×MS**
* **TrGT** (Gustavson-style) for **HS×HS**
* **TrGS** (Gustavson-style) for **HS×MS** and **HS×D** 

So Trapezoid’s flexibility is primarily: **pick the right schedule/dataflow for the sparsity regime**.

### 2) Mild sparsity: hardware “packs” effectual work into dense MACs (TrIP)

Key mechanism: **Multi-Fiber Intersection Unit (MFIU)** + **two distribution networks** + **merge-reduction tree** + **banked local buffer** per PE row. 

The MFIU explicitly does (in hardware):

* pairwise bitmask intersections (AND),
* prefix sums to count effectual positions,
* shifting indices to produce routing metadata for the distribution networks. 

They also **dynamically choose how many B columns to stream** so that “effectual computations” don’t exceed available multipliers (e.g., 128 per PE row). 

### 3) High sparsity: Gustavson + **multi-level memory hierarchy** for gather bandwidth

For HS, Trapezoid’s key claim is that Gustavson-like schedules need **high gather bandwidth**, and they provide it via a **multi-level memory hierarchy** (row-local buffers + global cache clusters). 

They also *resource-slice* a PE row into “subrows” based on nnz, allocating registers/multipliers/buffer bank/network/tree slices proportionally. 

---

## What “flexible sparsity” means in Proton-SDS (your proposal)

**Proton-SDS = a *general* sparse/irregular execution substrate where flexibility comes from (1) software-defined traversal + packing, and (2) software-defined vector width/topology via dynamic grouping.**

Your key levers (as written) are:

* **Software-defined vector fabric (SDSF):** dynamically form vector engines out of a manycore mesh (variable group size/width).
* **Software-defined sparsity:** the “Commander” runs metadata/control (and runahead), constructs **Compressed Frames**, and the VUs consume those frames.
* **Decoupled Runahead Scheduling:** hide irregular latency by issuing async gather-to-frame and letting compute run when frames are ready.

So Proton’s flexibility is primarily: **make the *hardware* a malleable vector substrate, and let software define the sparsity logic and packing strategy**.

---

## Compare & contrast: where each puts “adaptation”

### A) Scope of generality

* **Trapezoid:** *Matrix multiplication* across D/MS/HS by switching among a few matmul-specific schedules (IP + two Gustavson variants + a novel sparse IP variant). 
* **Proton-SDS:** intended as a **general irregular tensor substrate** (SpMV/SpMM/SpGEMM, reductions, sparse attention-style gathers, etc.), because traversal is software-defined.

**Positioning angle:** Trapezoid is “unified across sparsity *levels* for GEMM,” while Proton is “unified across sparsity *formats + kernels*.”

### B) What is hardwired vs programmable

* **Trapezoid hardwires** critical sparse mechanisms:

  * bitmask-based multi-fiber intersection (MFIU) + routing metadata generation 
  * Benes-style distribution networks (for A and B) and a merge-reduction tree 
  * explicit memory hierarchy to satisfy Gustavson gather bandwidth 
* **Proton pushes these into software**:

  * intersection/merging/packing become compiler+runtime strategies that fill frames (optionally aided by small primitives, but not required).

**Novelty hook for you:** “Trapezoid is versatile *within* a fixed matmul template; Proton is versatile by moving sparsity topology decisions out of fixed-function hardware.”

### C) How “packing” happens (key difference)

* **Trapezoid packs in hardware**: it turns multiple fiber intersections into a dense vector-like stream of effectual multiplies (AND + prefix sum + shift → route values to multipliers). 
* **Proton packs in software**: the Commander constructs **Compressed Frames** that already present “dense-ish” work to the VUs; variable-length rows are handled by lane packing + frame chaining (in your design).

**Practical implication:**

* Trapezoid’s packing is extremely fast/area-efficient for *its assumed representations* (bitmasks/tiles), but is less open-ended.
* Proton’s packing can support “whatever the compiler can describe,” but you must prove the software overhead doesn’t erase gains.

### D) Handling high sparsity: cache-based gather vs runahead gather-to-frame

* **Trapezoid (HS):** Gustavson-based schedules + **multi-level cache/buffers** to reduce B traffic and raise gather bandwidth. 
* **Proton (HS):** **constructive runahead** and **async gather-to-frame**, hiding latency by running ahead and materializing gathered operands directly into scratchpad frames.

**Positioning angle:** Trapezoid is “bandwidth/traffic-optimal for HS matmul via cache + schedule.” Proton is “latency-tolerant for irregular indirection via runahead + frame materialization,” which can apply beyond matmul.

### E) Adaptation granularity

* **Trapezoid:** adapts mainly by **choosing among a few modes** (IP/TrIP/TrGT/TrGS) and by **some dynamic packing decisions** (e.g., B columns streamed to match multiplier capacity). 
* **Proton:** adapts by **changing the hardware’s logical vector topology/width** (dynamic grouping) *and* by changing software packing policy.

This is one of your strongest “paradigm shift” claims: **the vector width becomes a function of sparsity structure, not a constant.**

---

## Side-by-side table

| Dimension               | Trapezoid (ISCA’24)                                                   | Proton-SDS (your proposal)                                                  |
| ----------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Primary target          | Dense→HS **matrix multiplication**                                    | **General irregular tensor** kernels (SpMV/SpMM/SpGEMM/…)                    |
| “Flexibility” mechanism | Switch among **4 dataflows** (IP/TrIP/TrGT/TrGS)                      | Software-defined traversal + **dynamic vector grouping**                     |
| Mild sparsity core      | **MFIU** (AND+prefix+shift) + distribution nets + reduction tree      | Commander builds **Compressed Frames**; VUs run masked micro-ops             |
| High sparsity core      | Gustavson dataflows + **multi-level memory hierarchy** for gather BW  | **Decoupled runahead** + gather-to-frame + scoreboard readiness              |
| Output accumulation     | Merge-reduction tree + banked local buffer                            | Software-scheduled reductions / scratchpad accumulators (as written)         |
| Format assumptions      | Leans on **bitmasks/tiling** machinery for intersection/packing       | Explicit goal: **format-agnostic** via software-defined metadata traversal   |
| Adaptation granularity  | Mode switch + limited dynamic packing                                 | Mode switch + **variable vector width/topology** + software packing policies |

---

## How to position your novelty *against Trapezoid* (recommended wording)

If you want the cleanest “novelty delta,” I’d frame it like this:

1. **Trapezoid generalizes across sparsity *levels* for one kernel (GEMM)** using multiple specialized dataflows and dedicated sparse hardware blocks. 
2. **Proton generalizes across sparsity *formats and kernels*** by separating:

   * topology/metadata traversal (**software, Commander + compiler**), from
   * numeric throughput (**hardware vector fabric**),
     and by making vector width/topology itself software-defined.

Then add a grounded concession + advantage:

* Concession: Trapezoid’s hardwired MFIU/distribution/tree will likely beat you on “its sweet spot” (bitmask-friendly MS×MS matmul) because it packs intersections at very high throughput. 
* Advantage: your approach should win when *formats change*, *kernels diversify*, or *sparsity is coupled with pointer chasing / irregular gathers* that don’t map neatly onto a fixed set of matmul dataflows.

---

## What you should steal from Trapezoid (to strengthen Proton without losing the thesis)

Even if you keep Proton “software-defined,” Trapezoid suggests two optional accelerants that fit your story:

1. **Bitmask-intersection as a vector primitive**
   Instead of a full MFIU block, consider adding ISA support for:

* vector AND + popcount/prefix-sum over small tiles,
* fast “compress/expand” (like a shift/permute network).
  This lets the Commander build frames faster while keeping formats programmable (software chooses what the masks *mean*).

2. **Local gather bandwidth hierarchy**
   Trapezoid is explicit that Gustavson needs high gather BW and they solve it with a multi-level hierarchy. 
   For Proton, the analog is: **don’t rely on one scratchpad + global cache**; add a small “row-local”/“group-local” buffer level to sustain gather-to-frame.

---

## 1) Traversal/packing throughput can become the bottleneck (the classic “front-end can’t feed the MACs” problem)

**Trapezoid** puts a lot of the “figure out which multiplies are effectual and where to send them” work into hardware—e.g., its Multi-Fiber Intersection Unit does bitmask AND + prefix sums + shifting to generate routing/compaction metadata.
That hardware pipeline is designed to run at array speed so the multipliers stay busy.

**Proton**, as written, has the Commander scalar core:

* traverse metadata (CSR pointers, bitmasks, coordinate lists, etc.)
* decide lane packing / segmentation
* emit gathers and fill Compressed Frames

If that Commander can’t produce “ready frames” fast enough, **VUs idle** even if memory is fine. This is the most direct way Proton loses on structured sparse matmul-like kernels where traversal is cheap to hardwire.

**When it hurts most:** mid-sparsity tiled/bitmask forms where “which lanes are valid” can be decided extremely quickly in hardware (Trapezoid), but takes a nontrivial number of scalar instructions in software.

---

## 2) Trapezoid’s flexibility is *within* a matmul template; Proton’s flexibility is broader (and that costs overhead)

Trapezoid flexes by switching among a small set of dataflows (IP / TrIP / TrGT / TrGS) chosen for density regimes.
It also dynamically controls how much work is streamed to match multiplier capacity.

Proton tries to be format/kernal agnostic, so its “flexibility” comes from **general mechanisms** (frame packing, dynamic grouping, runahead). The overhead of those general mechanisms is paid even when the problem could be handled by a highly tuned matmul-specific pipeline.

---

## 3) Routing + reduction can be cheaper in Trapezoid’s fixed networks than in frame-based “software-defined” flow

Trapezoid explicitly includes:

* distribution networks to steer operands
* a merge-reduction tree for accumulation

Proton, as described, tends to:

* write packed operands into SPM frames
* then execute masked vector micro-ops
* then reduce/commit through software-scheduled patterns (or generic reductions)

For workloads where **the merge/reduce pattern is frequent and predictable** (common in matmul-style contractions), Trapezoid’s fixed networks can simply do less work per effectual multiply than a general “pack → store → load → reduce” frame pipeline.

---

## 4) Scratchpad “gather-to-frame” placement can create port/bank pressure that Trapezoid avoids

Proton’s DRS is powerful, but operationally it means:

* lots of fine-grain writes arriving into frame slots,
* plus mask/scoreboard updates,
* plus subsequent vector loads from those frame buffers.

That can stress **SPM write ports/banks** (especially with many outstanding gathers), becoming the bottleneck even if DRAM/L2 bandwidth is okay.

Trapezoid, by contrast, is architected around feeding a PE array with local buffering and structured data motion for its schedules.
So in some regimes, **data motion is more streaming/regular**, and less “random lane placement into a scratchpad.”

---

## 5) Commander + inet overhead can dominate when the kernel is already “nicely structured”

If the sparsity is structured enough that a dedicated sparse matmul engine can keep its pipeline full, then Proton’s:

* commander instruction overhead,
* inet broadcast overhead,
* frame bookkeeping,
* possible group reconfiguration costs,

can be pure tax relative to Trapezoid’s “always-on” spatial schedule.

---

# The short answer (focused on traversal)

Proton does worse when **the act of discovering/packing effectual work is itself the critical path** and can be performed extremely efficiently by specialized hardware (e.g., Trapezoid’s bitmask intersection + compaction/routing pipeline).
In those cases, moving traversal/packing to a scalar Commander risks turning Proton into a **metadata front-end bottleneck** that starves the math units.