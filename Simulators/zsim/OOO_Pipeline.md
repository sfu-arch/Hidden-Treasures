# OOO Core Pipeline Modeling Documentation

## Overview

The `OOOCore` simulates a detailed out-of-order processor pipeline based on Intel Nehalem microarchitecture. The simulation is **issue-centric**, meaning `curCycle` tracks the current issue cycle, and all pipeline activities are modeled relative to instruction issue.

## Invocation Flow: From Pin to Pipeline Simulation

### 1. Pin Instrumentation Callbacks

After **functional emulation** by Pin, static callback functions are invoked to drive timing simulation:

```cpp
// Pin calls these after emulating instructions
LoadFunc(tid, addr)         → records load address
StoreFunc(tid, addr)        → records store address  
BranchFunc(tid, pc, ...)    → records branch info
BblFunc(tid, bblAddr, ...)  → simulates entire BBL pipeline
```

These are **static trampoline functions** that dispatch to the appropriate core instance:

```cpp
void OOOCore::BblFunc(THREADID tid, ADDRINT bblAddr, BblInfo* bblInfo) {
    OOOCore* core = static_cast<OOOCore*>(cores[tid]);
    core->bbl(bblAddr, bblInfo);  // Call instance method
    
    // Phase boundary check
    while (core->curCycle > core->phaseEndCycle) {
        core->phaseEndCycle += zinfo->phaseLength;
        uint32_t newCid = TakeBarrier(tid, cid);
        if (newCid != cid) break;  // context-switch
    }
}
```

### 2. Address Recording Phase

Memory operations are **recorded** but not immediately simulated:

```cpp
inline void OOOCore::load(Address addr) {
    loadAddrs[loads++] = addr;  // Buffer for later
}

inline void OOOCore::store(Address addr) {
    storeAddrs[stores++] = addr;  // Buffer for later
}
```

Branches are also recorded:

```cpp
void OOOCore::branch(Address pc, bool taken, Address takenNpc, Address notTakenNpc) {
    branchPc = pc;
    branchTaken = taken;
    branchTakenNpc = takenNpc;
    branchNotTakenNpc = notTakenNpc;
}
```

### 3. BBL Simulation (The Core Engine)

**`bbl(bblAddr, bblInfo)`** simulates the **previous BBL** once the next BBL starts. This delay allows all memory addresses to be collected.

```cpp
inline void OOOCore::bbl(Address bblAddr, BblInfo* bblInfo) {
    if (!prevBbl) {
        prevBbl = bblInfo;  // First BBL, nothing to simulate yet
        loads = stores = 0;
        return;
    }
    
    /* Simulate execution of previous BBL */
    DynBbl* bbl = &(prevBbl->oooBbl[0]);
    prevBbl = bblInfo;  // Update for next iteration
    
    // ... full pipeline simulation happens here ...
}
```

## Pipeline Stages

The OOO core models these pipeline stages (loosely based on Nehalem):

```
Stage 1:  FETCH       → I-cache access, branch prediction
Stage 4:  DECODE      → Instruction decode to micro-ops
Stage 7:  ISSUE       → Issue from instruction window
Stage 13: DISPATCH    → RAT + ROB + RS allocation
          EXECUTE     → Execution units
          MEMORY      → L1D access, LSU
          COMMIT      → Retire from ROB
```

## Backend: Dispatch/Issue/Execute Simulation

### Per-Uop Processing Loop

```cpp
for (uint32_t i = 0; i < bbl->uops; i++) {
    DynUop* uop = &(bbl->uop[i]);
    
    // 1. DECODE stage modeling
    // 2. Issue width limiting (4 uops/cycle)
    // 3. Register scoreboard dependency tracking
    // 4. RF read port limiting
    // 5. Instruction window scheduling
    // 6. Type-specific execution (LOAD/STORE/GENERAL/FENCE)
    // 7. ROB commit tracking
}
```

### 1. Decode Stage Stalls

Models decode throughput and predecode delays:

```cpp
uint32_t decDiff = uop->decCycle - prevDecCycle;
decodeCycle = MAX(decodeCycle + decDiff, uopQueue.minAllocCycle());

if (decodeCycle > curCycle) {
    uint32_t cdDiff = decodeCycle - curCycle;
    curCycleIssuedUops = 0;
    curCycleRFReads = 0;
    for (uint32_t i = 0; i < cdDiff; i++) 
        insWindow.advancePos(curCycle);  // curCycle increments
}

uopQueue.markLeave(curCycle);
```

**Effect**: Advances `curCycle` when decode can't keep up with issue.

### 2. Issue Width Limiting

Enforces 4 uops/cycle issue limit:

```cpp
if (curCycleIssuedUops >= ISSUES_PER_CYCLE) {
    curCycleIssuedUops = 0;
    curCycleRFReads = 0;
    insWindow.advancePos(curCycle);  // curCycle++
}
curCycleIssuedUops++;
```

### 3. Register Dependency Tracking

Uses a **register scoreboard** to track when registers become available:

```cpp
regScoreboard[0] = curCycle;  // R0 always available (invalid reg)

uint64_t c0 = regScoreboard[uop->rs[0]];  // Source 1 ready cycle
uint64_t c1 = regScoreboard[uop->rs[1]];  // Source 2 ready cycle
uint64_t cOps = MAX(c0, c1);              // Operands ready
```

### 4. RF Read Port Limiting

Models register file read bandwidth (3 reads/cycle):

```cpp
curCycleRFReads += ((c0 < curCycle)? 1 : 0) + ((c1 < curCycle)? 1 : 0);

if (curCycleRFReads > RF_READS_PER_CYCLE) {
    curCycleRFReads -= RF_READS_PER_CYCLE;
    curCycleIssuedUops = 0;
    insWindow.advancePos(curCycle);  // curCycle++
}
```

**Logic**: If operands aren't ready (future cycles), they're already forwarded via bypass network. Otherwise, must read RF.

### 5. Dispatch Cycle Calculation

Combines dependencies with structural hazards:

```cpp
uint64_t c2 = rob.minAllocCycle();        // ROB availability
uint64_t c3 = curCycle;                   // Current cycle

uint64_t dispatchCycle = MAX(cOps, MAX(c2, c3) + (DISPATCH_STAGE - ISSUE_STAGE));
```

**Dispatch delay**: 6 cycles (RAT + ROB + RS)

### 6. Instruction Window Scheduling

The core scheduling mechanism - models execution ports:

```cpp
insWindow.schedule(curCycle, dispatchCycle, uop->portMask, uop->extraSlots);
```

**What this does:**
- Searches for free execution port matching `portMask`
- May advance `curCycle` if window is full
- May increase `dispatchCycle` if ports are busy
- Models 36-entry instruction window

**Port masks** (Nehalem execution ports):
- Port 0: Integer ALU, FP multiply, etc.
- Port 1: Integer ALU, FP add, etc.
- Port 2: Load operations
- Port 3: Store address
- Port 4: Store data
- Port 5: Integer ALU, branch, etc.

### 7. Type-Specific Execution

#### UOP_GENERAL (ALU/FPU operations)

```cpp
case UOP_GENERAL:
    commitCycle = dispatchCycle + uop->lat;
    break;
```

Simple: dispatch + latency.

#### UOP_LOAD (Memory loads)

```cpp
case UOP_LOAD:
    // Load queue allocation
    uint64_t lqCycle = loadQueue.minAllocCycle();
    if (lqCycle > dispatchCycle) {
        dispatchCycle = lqCycle;  // Backpressure
    }
    
    // Wait for previous store addresses
    dispatchCycle = MAX(lastStoreAddrCommitCycle+1, dispatchCycle);
    
    // L1D access
    Address addr = loadAddrs[loadIdx++];
    uint64_t reqSatisfiedCycle = l1d->load(addr, dispatchCycle) + L1D_LAT;
    cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);
    
    // Store-to-load forwarding
    uint32_t fwdIdx = (addr>>2) & (FWD_ENTRIES-1);
    if (fwdArray[fwdIdx].addr == addr) {
        reqSatisfiedCycle = MAX(reqSatisfiedCycle, fwdArray[fwdIdx].storeCycle);
    }
    
    commitCycle = reqSatisfiedCycle;
    loadQueue.markRetire(commitCycle);
    break;
```

**Key features:**
- Load queue modeling (32 entries, 4-wide retire)
- Memory ordering: waits for prior store addresses
- L1D timing simulation via FilterCache
- Store-to-load forwarding (direct-mapped, 32 entries)
- Contention recording via `cRec.record()`

#### UOP_STORE (Memory stores)

```cpp
case UOP_STORE:
    // Store queue allocation
    uint64_t sqCycle = storeQueue.minAllocCycle();
    if (sqCycle > dispatchCycle) {
        dispatchCycle = sqCycle;  // Backpressure
    }
    
    // Memory ordering
    dispatchCycle = MAX(lastStoreAddrCommitCycle+1, dispatchCycle);
    
    // L1D access
    Address addr = storeAddrs[storeIdx++];
    uint64_t reqSatisfiedCycle = l1d->store(addr, dispatchCycle) + L1D_LAT;
    cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);
    
    // Update forwarding table
    fwdArray[(addr>>2) & (FWD_ENTRIES-1)].set(addr, reqSatisfiedCycle);
    
    commitCycle = reqSatisfiedCycle;
    lastStoreCommitCycle = MAX(lastStoreCommitCycle, reqSatisfiedCycle);
    storeQueue.markRetire(commitCycle);
    break;
```

#### UOP_STORE_ADDR (Store address calculation)

```cpp
case UOP_STORE_ADDR:
    commitCycle = dispatchCycle + uop->lat;
    lastStoreAddrCommitCycle = MAX(lastStoreAddrCommitCycle, commitCycle);
    break;
```

Tracks when store addresses are known (for load ordering).

#### UOP_FENCE (Memory fences)

```cpp
case UOP_FENCE:
    commitCycle = dispatchCycle + uop->lat;
    // Force future load serialization
    lastStoreAddrCommitCycle = MAX(commitCycle, 
        MAX(lastStoreAddrCommitCycle, lastStoreCommitCycle + uop->lat));
    break;
```

Enforces memory ordering across the fence.

### 8. ROB Commit

```cpp
rob.markRetire(commitCycle);
```

Models 128-entry ROB with 4-wide retire bandwidth.

### 9. Register Dependency Update

```cpp
regScoreboard[uop->rd[0]] = commitCycle;
regScoreboard[uop->rd[1]] = commitCycle;
```

Updates when destination registers become available for dependent ops.

## Frontend: Fetch & Branch Prediction

Simulated **after** backend (working backwards from decode):

### 1. Compute Fetch Cycle

```cpp
uint64_t fetchCycle = decodeCycle - (DECODE_STAGE - FETCH_STAGE);
```

### 2. Branch Prediction

```cpp
if (branchPc && !branchPred.predict(branchPc, branchTaken)) {
    mispredBranches++;
    
    // Wrong-path fetches (up to mispred penalty)
    Address wrongPathAddr = branchTaken? branchNotTakenNpc : branchTakenNpc;
    uint64_t reqCycle = fetchCycle;
    
    for (uint32_t i = 0; i < 5*64/lineSize; i++) {
        uint64_t fetchLat = l1i->load(wrongPathAddr + lineSize*i, curCycle) - curCycle;
        cRec.record(curCycle, curCycle, curCycle + fetchLat);
        uint64_t respCycle = reqCycle + fetchLat;
        
        if (respCycle > lastCommitCycle) break;  // Mispred resolved
        
        reqCycle = respCycle + lineSize/FETCH_BYTES_PER_CYCLE;
    }
    
    fetchCycle = lastCommitCycle;  // Pipeline flush penalty
}
```

**Branch predictor**: 2-level PAg (11-bit index, 18-bit history, 14-bit PHT)

**Wrong-path modeling**: Fetches wrong path instructions (up to ~5 cache lines) until misprediction resolves.

#### Wrong-Path Execution Modeling Details

When a branch mispredicts, the simulator models speculative wrong-path instruction fetches:

**A. Determine Wrong-Path Address**
```cpp
Address wrongPathAddr = branchTaken? branchNotTakenNpc : branchTakenNpc;
```
- If predicted **not-taken** but actually **taken**: fetches fall-through path
- If predicted **taken** but actually **not-taken**: fetches target path

**B. Fetch Wrong-Path Instructions**
```cpp
uint64_t reqCycle = fetchCycle;
for (uint32_t i = 0; i < 5*64/lineSize; i++) {
    uint64_t fetchLat = l1i->load(wrongPathAddr + lineSize*i, curCycle) - curCycle;
    cRec.record(curCycle, curCycle, curCycle + fetchLat);
    uint64_t respCycle = reqCycle + fetchLat;
    
    if (respCycle > lastCommitCycle) {
        break;  // Misprediction detected, stop
    }
    
    reqCycle = respCycle + lineSize/FETCH_BYTES_PER_CYCLE;
}
```

**Key Characteristics:**

1. **Limited Duration**: Fetches until branch resolves (`lastCommitCycle`)
   - Branch resolution depends on when the branch instruction commits
   - Typical misprediction penalty: ~17 cycles on Nehalem
   - Can be longer if branch depends on slow loads
   
   **Understanding `lastCommitCycle`:**
   ```cpp
   uint64_t lastCommitCycle = 0;  // used to find misprediction penalty
   
   // Updated after each uop in the BBL
   for (uint32_t i = 0; i < bbl->uops; i++) {
       // ... uop simulation ...
       lastCommitCycle = commitCycle;  // Track last commit
   }
   ```
   
   `lastCommitCycle` represents **when the final uop in the BBL commits from the ROB**:
   
   - **Updated continuously**: After each uop simulation, holds the commit time of that uop
   - **Final value**: By loop end, holds the commit time of the last uop (often the branch)
   - **Branch resolution time**: Even if branch isn't the last uop, it can't resolve until all prior instructions commit (in-order retirement)
   - **Misprediction detection**: When branch commits, its result is known and misprediction detected
   
   **Why use commit (not execution) time?**
   - Branch **executes** (result known) before it **commits**
   - Pipeline can't recover until commit because:
     - Must wait for in-order commit for correctness
     - Branch must reach ROB head before pipeline flush
     - Conservative model: uses commit time for worst-case recovery
   
   **Variability factors:**
   - Dependencies: Branch depending on slow load → late commit
   - ROB fullness: Full ROB → delayed commit
   - Execution latency: Memory ops slower than ALU ops
   - Prior instructions: Must wait for all preceding uops
   
   **Example timing:**
   ```
   Cycle 100: Branch uop issues
   Cycle 110: Branch uop dispatches  
   Cycle 115: Branch executes (result known in backend)
   Cycle 125: Branch commits from ROB  ← lastCommitCycle = 125
   
   Wrong-path fetches:
     Cycle 120: Fetch line 1 (completes cycle 124) ✓ (124 < 125)
     Cycle 124: Fetch line 2 (completes cycle 128) ✗ STOP (128 > 125)
   
   Pipeline flush:
     fetchCycle = 125 (wait for branch commit)
     Restart from correct path at cycle 125
   ```
   
   **Usage in wrong-path loop:**
   ```cpp
   if (respCycle > lastCommitCycle) {
       break;  // Branch resolved, stop wrong-path fetches
   }
   ```
   
   Wrong-path fetches continue until `respCycle` (when a fetch would complete) exceeds `lastCommitCycle` (when branch resolves).

2. **Upper Bound**: Maximum 5 cache lines (320 bytes with 64B lines)
   - Represents finite pipeline depth:
     - IQ: 18 instructions
     - Uop queue: 28 uops
     - Instruction window: 36 uops
     - Predecode buffer: 16 bytes
   - At ~3.5 bytes/instr, 1.2 uops/instr ≈ pipeline capacity

3. **Fetch Bandwidth Modeling**:
   - 16 bytes/cycle fetch rate: `lineSize/FETCH_BYTES_PER_CYCLE`
   - 64-byte line takes 4 cycles minimum (throughput limit)
   - Plus I-cache access latency

4. **Cache Pollution** (Critical Detail):
   ```cpp
   l1i->load(wrongPathAddr + lineSize*i, curCycle)
   ```
   - Wrong-path fetches **actually access the I-cache**
   - Can cause **cache misses** on wrong-path code
   - **Pollutes I-cache** by potentially evicting useful instructions
   - **Recorded for contention simulation** (can delay other cores)
   - Models realistic microarchitectural side effects

5. **Pipeline Flush**:
   ```cpp
   fetchCycle = lastCommitCycle;
   ```
   - All speculatively-fetched instructions squashed
   - `fetchCycle` jumps to branch commit time
   - Represents pipeline flush penalty

**What's Modeled:**
- ✅ I-cache pollution from speculative fetches
- ✅ Fetch bandwidth consumption
- ✅ Pipeline flush penalty
- ✅ Variable wrong-path duration (until branch resolves)
- ✅ Contention effects on other cores
- ✅ Multiple cache line fetches

**What's NOT Modeled:**

1. **No Wrong-Path Execution**: Only fetches are modeled, not backend execution
   - Wrong-path uops don't dispatch/issue/execute
   - No resource consumption in instruction window, ROB, etc.
   - Assumes squash happens before execution resources used
   - Reasonable: branch resolution typically fast, wrong-path rarely executes

2. **Perfect BTB** (Branch Target Buffer):
   ```cpp
   // NOTE: Resteering due to BTB misses is done at the BAC unit, is
   // relatively rare, and carries an 8-cycle penalty, which should be
   // partially hidden if the branch is predicted correctly
   ```
   - Assumes BTB always has correct target address
   - Real BTB misses cause 8-cycle resteering penalty (not modeled)

3. **No Wrong-Path Memory Operations**: Only instruction fetches
   - Wrong-path loads/stores not executed
   - No speculative L1D accesses

4. **Next Branch Assumption**:
   - Assumes wrong-path contains non-taken branches
   - Simplifies nested misprediction scenarios

**Design Rationale:**

This is a **pragmatic model** that:
- Captures primary performance impact (I-cache effects + flush penalty)
- Avoids complexity of full speculative execution state
- Accurately models cross-core interference from wrong-path pollution
- Validated against real hardware behavior (see ZSim ISCA 2013 paper)

### 3. Current BBL Instruction Fetch

```cpp
Address endAddr = bblAddr + bblInfo->bytes;
for (Address fetchAddr = bblAddr; fetchAddr < endAddr; fetchAddr += lineSize) {
    uint64_t fetchLat = l1i->load(fetchAddr, curCycle) - curCycle;
    cRec.record(curCycle, curCycle, curCycle + fetchLat);
    fetchCycle += fetchLat;
}
```

Models 16 bytes/cycle fetch bandwidth.

### 4. Decode Stage Update

```cpp
decodeCycle++;  // BBL boundary bubble
uint64_t minFetchDecCycle = fetchCycle + (DECODE_STAGE - FETCH_STAGE);
if (minFetchDecCycle > decodeCycle) {
    decodeCycle = minFetchDecCycle;
}
```

## Contention Recording

Throughout execution, cache accesses are recorded for later contention simulation:

```cpp
cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);
```

This feeds into the `OOOCoreRecorder` which builds the event DAG.

### What Events Are Recorded?

Only **memory hierarchy accesses** that need contention simulation are recorded. The core records timing for **cross-core shared resources** (caches, interconnect, memory), while microarchitectural resources (ALU ops, register dependencies, port conflicts) are simulated directly without events.

#### 1. Data Memory Accesses (L1D)

**Load Operations:**
```cpp
case UOP_LOAD:
    uint64_t reqSatisfiedCycle = l1d->load(addr, dispatchCycle) + L1D_LAT;
    cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);
```

**Store Operations:**
```cpp
case UOP_STORE:
    uint64_t reqSatisfiedCycle = l1d->store(addr, dispatchCycle) + L1D_LAT;
    cRec.record(curCycle, dispatchCycle, reqSatisfiedCycle);
```

**Parameters:**
- `curCycle`: When the uop issues from instruction window
- `dispatchCycle`: When load/store dispatches to LSU
- `reqSatisfiedCycle`: When L1D responds (includes cache hierarchy delays)

#### 2. Instruction Fetches (L1I)

**Correct-Path Fetches:**
```cpp
for (Address fetchAddr = bblAddr; fetchAddr < endAddr; fetchAddr += lineSize) {
    uint64_t fetchLat = l1i->load(fetchAddr, curCycle) - curCycle;
    cRec.record(curCycle, curCycle, curCycle + fetchLat);
    fetchCycle += fetchLat;
}
```

**Wrong-Path Fetches (Branch Misprediction):**
```cpp
for (uint32_t i = 0; i < 5*64/lineSize; i++) {
    uint64_t fetchLat = l1i->load(wrongPathAddr + lineSize*i, curCycle) - curCycle;
    cRec.record(curCycle, curCycle, curCycle + fetchLat);
    // Fetch until misprediction resolves
}
```

**Parameters:**
- All three parameters are `curCycle` or `curCycle + latency`
- Models each cache line fetch from I-cache

#### 3. The Recording Process

When `cRec.record()` is called, it triggers `recordAccess()` in the recorder:

```cpp
void OOOCoreRecorder::recordAccess(uint64_t curCycle, uint64_t dispatchCycle, uint64_t respCycle) {
    assert(eventRecorder.hasRecord());
    TimingRecord tr = eventRecorder.popRecord();  // From FilterCache
    
    if (IsGet(tr.type)) {
        // Build event chain for loads/fetches
    } else {
        // Build event chain for stores
    }
}
```

**TimingRecord (from cache hierarchy) contains:**
- **type**: GET (read) or PUT (write)
- **reqCycle**: When request enters cache hierarchy
- **respCycle**: When cache hierarchy responds
- **startEvent**: Event from cache that started the request
- **endEvent**: Event from cache that completes the request

#### 4. Event DAG Construction

**For GET (Loads/Instruction Fetches):**

Creates complete event chain:
- **OOOIssueEvent**: Marks issue cycle
- **OOODispatchEvent**: Marks dispatch to LSU (loads only)
- **DelayEvents**: Model pipeline delays between stages
- **OOORespEvent**: Marks when response returns to core
- Links all events into dependency graph with responses in `futureResponses` heap

Event flow:
```
Issue → Dispatch → L1D Request (cache hierarchy) → L1D Response → Core Response
```

**For PUT (Stores):**

Simpler fire-and-forget:
- **OOOIssueEvent**: Marks issue cycle
- **DelayEvent**: Pipeline delay to cache
- Links to cache hierarchy start event
- No response tracking (stores don't block execution)

Event flow:
```
Issue → L1D Request (cache hierarchy)
```

#### 5. What's NOT Recorded

These are simulated **directly** without creating events:

- **ALU/FPU operations** (UOP_GENERAL): `commitCycle = dispatchCycle + uop->lat`
- **Register dependencies**: Tracked via `regScoreboard[]`
- **Execution port conflicts**: Handled by `insWindow.schedule()`
- **Decode stalls**: Modeled by advancing `decodeCycle`
- **Issue width limits**: Enforced by `curCycleIssuedUops` counter
- **RF read port limits**: Enforced by `curCycleRFReads` counter
- **ROB/queue fullness**: Tracked by `rob.minAllocCycle()`, etc.

#### 6. Summary Table

| Access Type | Source | Timing Parameters | Purpose |
|-------------|--------|-------------------|---------|
| **Load** | L1D | (issue, dispatch, response) | Model load latency + contention |
| **Store** | L1D | (issue, dispatch, response) | Model store acknowledgment |
| **I-Fetch** | L1I | (curCycle, curCycle, response) | Model fetch latency + contention |
| **Wrong-path Fetch** | L1I | (curCycle, curCycle, response) | Model misprediction pollution |

**Key insight:** The recorder focuses exclusively on **memory hierarchy contention** across cores. All single-core microarchitectural effects (dependencies, structural hazards) are deterministically simulated without events, enabling fast simulation while accurately capturing multi-core interference.

## Cycle Advancement Mechanisms

### 1. Incremental (During BBL Simulation)

```cpp
insWindow.advancePos(curCycle)  → curCycle++
```

Called when:
- Decode stalls
- Issue width exceeded
- RF read ports saturated
- Instruction window full

### 2. Bulk (After Contention Simulation)

```cpp
void OOOCore::advance(uint64_t targetCycle) {
    decodeCycle += targetCycle - curCycle;
    insWindow.longAdvance(curCycle, targetCycle);  // Updates curCycle
    curCycleRFReads = 0;
    curCycleIssuedUops = 0;
}
```

Called from:
- `join()` - when thread rejoins after being descheduled
- `cSimStart()` - before contention simulation
- `cSimEnd()` - after contention simulation adds skew

### 3. Recorder-Driven (Contention Integration)

```cpp
void OOOCore::cSimEnd() {
    uint64_t targetCycle = cRec.cSimEnd(curCycle);
    if (targetCycle > curCycle) advance(targetCycle);
}
```

Applies contention delays calculated by bound-weave simulation.

## Key Data Structures

### WindowStructure<1024, 36>
- **Size**: 36 uops
- **Horizon**: 1024 cycles (2 windows + unbounded overflow)
- **Function**: Schedules uops to execution ports
- **Operations**: `schedule()`, `advancePos()`, `longAdvance()`

### ReorderBuffer<128, 4>
- **Size**: 128 entries
- **Width**: 4 retires/cycle
- **Function**: Models in-order retirement
- **Operation**: `markRetire(commitCycle)`

### Load/Store Queues
- **Type**: `ReorderBuffer<32, 4>`
- **Function**: Track in-flight memory ops
- **Ordering**: Enforces memory consistency model

### Register Scoreboard
- **Type**: `uint64_t regScoreboard[MAX_REGISTERS]`
- **Function**: Tracks when each register becomes available
- **Usage**: Data dependency enforcement

### Forwarding Array
- **Type**: `FwdEntry fwdArray[32]`
- **Structure**: Direct-mapped by `(addr >> 2) & 31`
- **Function**: Store-to-load forwarding
- **Entry**: `{Address addr, uint64_t storeCycle}`

### Branch Predictor
- **Type**: `BranchPredictorPAg<11, 18, 14>`
- **Structure**: 2-level adaptive (BHSR → PHT)
- **BHSR**: 2^11 = 2048 entries, 18 bits history each
- **PHT**: 2^14 = 16384 entries, 2-bit counters

## Statistics Tracked

```cpp
instrs              // Total instructions
uops                // Total micro-ops retired
bbls                // Basic blocks executed
approxInstrs        // Instructions with approximate decode
mispredBranches     // Branch mispredictions
cycles              // Unhalted cycles (via cRec)
cCycles             // Contention cycles (via cRec)

// Optional (OOO_STALL_STATS)
profFetchStalls     // Cycles stalled on fetch
profDecodeStalls    // Cycles stalled on decode
profIssueStalls     // Cycles stalled on issue width
```

## Example Execution Flow

```
Pin Functional Emulation:
  MOV rax, [rbx]
  ADD rax, rcx
  MOV [rdx], rax

↓ Pin callbacks collect addresses

LoadFunc(tid, [rbx])
StoreFunc(tid, [rdx])
BblFunc(tid, bblAddr, bblInfo)

↓ OOOCore::bbl() simulates PREVIOUS BBL

For each uop:
  1. Check decode stall      → advance curCycle if needed
  2. Check issue width       → advance curCycle if needed  
  3. Read register scoreboard → check dependencies
  4. Check RF ports          → advance curCycle if needed
  5. Calculate dispatch time → MAX(deps, ROB, curCycle+delay)
  6. Schedule to port        → may advance curCycle, dispatchCycle
  7. Execute:
     - LOAD: L1D access, forwarding check, record contention
     - STORE: L1D access, update forwarding, record contention
     - GENERAL: dispatch + latency
  8. Commit to ROB           → track retire
  9. Update scoreboard       → deps for future uops

After all uops:
  - Branch prediction
  - Wrong-path fetches (if mispred)
  - Current BBL I-cache fetches
  - Update decode cycle

↓ Phase boundary check

if (curCycle > phaseEndCycle):
  TakeBarrier()  → synchronize with other cores

↓ Contention Simulation (periodic)

cSimStart()  → taper phase
  [Bound-weave simulator runs event DAG]
cSimEnd()    → apply skew, advance(targetCycle)
```

## Summary

The OOO core models a complex superscalar pipeline through:

1. **Delayed simulation**: Collect addresses first, simulate previous BBL
2. **Issue-centric**: All timing relative to issue stage
3. **Structural hazards**: Decode, issue width, RF ports, ROB, LSU queues
4. **Data dependencies**: Register scoreboard tracking
5. **Execution resources**: Instruction window + port scheduling
6. **Memory system**: L1I/L1D via FilterCache, forwarding, ordering
7. **Branch prediction**: 2-level predictor with wrong-path fetches
8. **Contention integration**: Record events, apply delays from bound-weave

The result is cycle-accurate modeling of out-of-order execution that captures both microarchitectural effects (port conflicts, decode stalls) and memory hierarchy effects (cache misses, contention) through the separation of functional and timing simulation.
