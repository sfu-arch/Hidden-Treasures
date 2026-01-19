# zsim Design Documentation

This document provides a detailed technical overview of zsim's architecture, focusing on its clock and timing systems, phase transitions, and DRAM modeling.

## Table of Contents

1. [Overview](#overview)
2. [Clock and Timing System](#clock-and-timing-system)
3. [Phase Transitions](#phase-transitions)
4. [DRAM Tick Modeling and Minimum Latency](#dram-tick-modeling-and-minimum-latency)
5. [Synchronization and Barriers](#synchronization-and-barriers)
6. [Performance and Accuracy Trade-offs](#performance-and-accuracy-trade-offs)

## Overview

zsim is a fast and accurate microarchitectural simulator that uses a **two-phase simulation approach** to balance simulation speed with timing accuracy. The key innovation is separating fast approximate execution (bound phase) from detailed timing simulation (weave phase), with periodic synchronization between cores.

### Key Design Principles

- **Phase-based execution**: Alternates between fast bound phase and detailed weave phase
- **Independent core clocks**: Each core advances independently during bound phase
- **Periodic synchronization**: Cores sync at configurable phase boundaries
- **Multiple clock domains**: Different components (cores, memory, interconnect) in separate timing domains
- **Contention modeling**: Detailed resource conflicts modeled in weave phase

## Clock and Timing System

zsim employs a sophisticated multi-clock system with several interrelated timing concepts that enable both performance and accuracy.

### Core Clock Types

#### 1. Per-Core Clock (`curCycle`)
```cpp
class TimingCore {
    uint64_t curCycle;           // Core's local execution cycle
    uint64_t phaseEndCycle;      // When this core must sync
};
```

**Purpose**: Tracks each core's independent execution progress
- **Updates**: During instruction execution, memory accesses, pipeline events
- **Independence**: Can diverge between cores during bound phase
- **Synchronization**: Forced to sync at phase boundaries

#### 2. Global Phase Clock (`globPhaseCycles`)
```cpp
// In zsim.h and updated in scheduler.h
zinfo->globPhaseCycles += zinfo->phaseLength;  // Updated at phase boundaries
```

**Purpose**: Defines global coordination points for all cores
- **Updates**: Every `phaseLength` cycles (default 10,000) when all cores sync
- **Scope**: Global reference for phase transitions and cross-core coordination
- **Calculation**: `globPhaseCycles = numPhases * phaseLength`

#### 3. ZLL Clock (Zero-Latency-Load Clock)
```cpp
// The ZLL clock is defined as: curCycle - gapCycles
uint64_t zllClock = curCycle - gapCycles;
```

**Purpose**: Provides a consistent reference frame for timing calculations
- **Definition**: Represents what the core's cycle would be without contention delays
- **Invariant**: Remains constant when applying skew corrections from contention simulation
- **Usage**: Enables correct timing translation across multiple phases

#### 4. Gap Cycles and Contention Tracking
```cpp
// In CoreRecorder::cSimEnd()
uint64_t skew = lastEvCycle2 - lastEvCycle1;
curCycle += skew;      // Advance core clock by contention delay
gapCycles += skew;     // Track cumulative contention
prevRespCycle += skew; // Adjust response timing
```

**Purpose**: Tracks cumulative delays due to resource contention
- **gapCycles**: Total cycles added due to contention across all phases
- **skew**: Additional delay from current phase's detailed simulation
- **Maintenance**: ZLL clock invariant preserved: `ZLL = curCycle - gapCycles`

### Clock Update Mechanisms

#### Bound Phase Updates
```cpp
// Core execution advances local clock
void TimingCore::bblAndRecord(Address bblAddr, BblInfo* bblInfo) {
    instrs += bblInfo->instrs;
    curCycle += bblInfo->instrs;  // Advance by instruction count
    
    // Memory accesses update clock with minimum latencies
    uint64_t startCycle = curCycle;
    curCycle = l1cache->load(addr, curCycle);  // Cache returns respCycle
}
```

#### Memory System Updates
```cpp
// Memory controllers return optimistic response cycles
uint64_t DDRMemory::access(MemReq& req) {
    uint64_t respCycle = req.cycle + (isWrite? minWrLatency : minRdLatency);
    // Record detailed event for weave phase
    return respCycle;  // Core updates curCycle to this value
}
```

#### Weave Phase Corrections
```cpp
// Apply contention simulation results
uint64_t lastEvCycle1 = lastEventSimulatedOrigStartCycle + gapCycles; // ZLL time
uint64_t lastEvCycle2 = lastEventSimulatedStartCycle;                 // Actual time
uint64_t skew = lastEvCycle2 - lastEvCycle1;  // Additional contention delay

// Update all clocks to maintain consistency
curCycle += skew;
gapCycles += skew;
prevRespCycle += skew;
```

### Clock Synchronization Properties

#### Core Independence
- **Bound Phase**: Cores execute independently with their own `curCycle`
- **No Lockstep**: Cores do not advance in synchronized cycles
- **Variable Progress**: Fast cores can get ahead of slow cores

#### Periodic Synchronization
- **Trigger**: When any core's `curCycle > phaseEndCycle`
- **Barrier**: All cores must reach sync point before any can proceed
- **Global Update**: `globPhaseCycles` advances by `phaseLength`

#### Timing Consistency
- **ZLL Reference**: Provides consistent timing baseline across phases
- **Contention Correction**: Detailed simulation adjusts optimistic bound phase timing
- **Causality**: Events maintain proper ordering despite multi-phase execution

## Phase Transitions

zsim's two-phase execution model is the core of its performance-accuracy balance. Understanding when and how phase transitions occur is crucial for understanding the simulator's behavior.

### Phase Types

#### Bound Phase (Fast Execution)
**Characteristics**:
- Fast approximate execution
- Minimum latency assumptions
- Independent core progress
- Memory accesses use optimistic timing
- No detailed contention modeling

**Code Path**:
```cpp
// Cores execute instructions and advance curCycle
void TimingCore::BblAndRecordFunc(THREADID tid, ADDRINT bblAddr, BblInfo* bblInfo) {
    TimingCore* core = static_cast<TimingCore*>(cores[tid]);
    core->bblAndRecord(bblAddr, bblInfo);  // Fast execution
    
    // Check for phase boundary
    while (core->curCycle > core->phaseEndCycle) {
        core->phaseEndCycle += zinfo->phaseLength;
        uint32_t newCid = TakeBarrier(tid, cid);  // TRIGGERS TRANSITION
    }
}
```

#### Weave Phase (Detailed Simulation)
**Characteristics**:
- Cycle-accurate contention simulation
- Resource conflict modeling
- Cross-domain event coordination
- Actual completion timing determination
- Clock correction and synchronization

**Code Path**:
```cpp
// In EndOfPhaseActions() - triggered by barrier
VOID EndOfPhaseActions() {
    zinfo->profSimTime->transition(PROF_WEAVE);
    
    // Execute detailed contention simulation
    zinfo->contentionSim->simulatePhase(zinfo->globPhaseCycles + zinfo->phaseLength);
    
    zinfo->eventQueue->tick();  // Process global events
    zinfo->profSimTime->transition(PROF_BOUND);  // Return to bound phase
}
```

### Transition Triggers

#### 1. Phase Boundary Detection
```cpp
// Any core exceeding phase boundary triggers transition
phaseEndCycle = zinfo->globPhaseCycles + zinfo->phaseLength;
if (curCycle > phaseEndCycle) {
    TakeBarrier(tid, cid);  // Enter synchronization
}
```

#### 2. Barrier Synchronization
```cpp
// All cores must reach barrier before proceeding
uint32_t TakeBarrier(uint32_t tid, uint32_t cid) {
    uint32_t newCid = zinfo->sched->sync(procIdx, tid, cid);  // BARRIER POINT
    // Context switch handling...
    return newCid;
}
```

#### 3. Phase End Actions
```cpp
// Last core to reach barrier triggers phase transition
void Scheduler::callback() {
    if (atSyncFunc) atSyncFunc(); // Calls EndOfPhaseActions()
    
    // Update global phase state
    zinfo->numPhases++;
    zinfo->globPhaseCycles += zinfo->phaseLength;
}
```

### Transition Sequence

1. **Bound Phase Execution**: Cores execute independently with minimum latencies
2. **Phase Boundary Hit**: First core exceeds `phaseEndCycle`
3. **Barrier Entry**: Core enters barrier, waits for others
4. **All Cores Synchronized**: Last core triggers phase end actions
5. **Weave Phase Start**: Detailed contention simulation begins
6. **Event Processing**: All recorded events simulated with full detail
7. **Clock Correction**: Core clocks adjusted based on actual timing
8. **Global State Update**: Phase counters and boundaries updated
9. **Bound Phase Resume**: Cores continue with corrected timing

### Timing Corrections During Transitions

#### Pre-Weave State
```cpp
// Before detailed simulation - optimistic timing
uint64_t boundPhaseCompletion = req.cycle + minLatency;
```

#### Weave Phase Simulation
```cpp
// Detailed simulation with actual resource conflicts
void ContentionSim::simulatePhase(uint64_t limit) {
    // Process all recorded events with full timing detail
    // Account for bank conflicts, queue delays, refresh cycles
    uint64_t actualCompletion = detailedSimulation(events);
}
```

#### Post-Weave Correction
```cpp
// Apply timing corrections to core clocks
uint64_t skew = actualCompletion - boundPhaseCompletion;
curCycle += skew;        // Adjust core clock
gapCycles += skew;       // Track total contention
```

## DRAM Tick Modeling and Minimum Latency

zsim's DRAM modeling demonstrates the bound/weave phase approach, using minimum latencies during bound phase and detailed timing during weave phase.

### Bound Phase DRAM Interface

#### Minimum Latency Calculation
```cpp
// DDR Memory initialization - ddr_mem.cpp
DDRMemory::DDRMemory(...) {
    // Calculate minimum latencies based on DRAM timing parameters
    minRdLatency = controllerSysLatency + memToSysCycle(tCL + tBL - 1);
    minWrLatency = controllerSysLatency;
    
    // Where:
    // controllerSysLatency: Controller overhead in system cycles
    // tCL: CAS Latency in memory cycles
    // tBL: Burst Length in memory cycles
    // memToSysCycle(): Converts memory cycles to system cycles
}
```

#### Bound Phase Access
```cpp
// Fast path - returns minimum latency
uint64_t DDRMemory::access(MemReq& req) {
    if (req.type == PUTS) {
        return req.cycle; // 0 latency for writebacks
    } else {
        bool isWrite = (req.type == PUTX);
        uint64_t respCycle = req.cycle + (isWrite? minWrLatency : minRdLatency);
        
        // Record detailed event for weave phase
        if (zinfo->eventRecorders[req.srcId]) {
            DDRMemoryAccEvent* memEv = new DDRMemoryAccEvent(this, ...);
            TimingRecord tr = {addr, req.cycle, respCycle, req.type, memEv, memEv};
            zinfo->eventRecorders[req.srcId]->pushRecord(tr);
        }
        
        return respCycle;  // Optimistic completion time
    }
}
```

### DRAM Tick Mechanism

#### Periodic Tick Events
```cpp
// DRAMSim integration with cycle-level ticking
template <class T>
class TickEvent : public TimingEvent {
    void simulate(uint64_t startCycle) {
        uint32_t delay = obj->tick(startCycle);  // Call DRAM's tick method
        if (delay) {
            requeue(startCycle + delay);         // Schedule next tick
        }
    }
};
```

#### DRAMSim Integration
```cpp
// DRAMSim memory controller tick - dramsim_mem_ctrl.cpp
uint32_t DRAMSimMemory::tick(uint64_t cycle) {
    dramCore->update();    // Advance DRAMSim's internal simulation
    curCycle++;           // Track local cycle count
    return 1;             // Return delay until next tick (1 cycle)
}
```

#### Tick Event Lifecycle
1. **Initialization**: TickEvent created and queued at cycle 0
2. **Periodic Execution**: Every cycle, `tick()` method called
3. **DRAM Update**: `dramCore->update()` advances internal DRAM state
4. **Automatic Rescheduling**: TickEvent requeues itself for next cycle
5. **Continuous Operation**: Process repeats throughout simulation

### Weave Phase DRAM Simulation

#### Detailed Event Processing
```cpp
// Weave phase - detailed DRAM simulation
void DDRMemory::enqueue(DDRMemoryAccEvent* ev, uint64_t sysCycle) {
    uint64_t memCycle = sysToMemCycle(sysCycle);
    
    // Create detailed DRAM request with full timing
    DDRRequest* req = createDetailedRequest(ev);
    
    // Model bank conflicts, row buffer hits/misses
    uint64_t actualCycle = detailedDRAMSimulation(req);
    
    // Complete event with actual timing
    ev->done(actualCycle);
}
```

#### Resource Conflict Modeling
```cpp
// Bank and timing conflict resolution
AddrLoc loc = mapLineAddr(ev->getAddr());
Bank& bank = banks[loc.rank][loc.bank];

// Check for bank conflicts
if (bank.state == BANK_ACTIVE && bank.openRow != loc.row) {
    // Row buffer miss - precharge + activate required
    uint64_t prechargeDelay = tRP;
    uint64_t activateDelay = tRCD;
    actualCycle += prechargeDelay + activateDelay;
}

// Check timing constraints (tRAS, tRC, etc.)
actualCycle = MAX(actualCycle, bank.earliestCmdCycle + tConstraint);
```

### Minimum Latency Design Rationale

#### Why Minimum Latencies?
1. **Performance**: Bound phase executes quickly without detailed resource modeling
2. **Accuracy Foundation**: Provides realistic baseline timing for instruction scheduling
3. **Consistency**: All memory controllers provide minimum latency guarantees
4. **Optimization**: Enables aggressive instruction scheduling during bound phase

#### Latency Components
```cpp
// Minimum latency breakdown
minLatency = controllerLatency +     // Controller processing overhead
             networkLatency +        // Interconnect traversal  
             deviceLatency;          // DRAM device minimum access time
```

#### Different Controller Types
```cpp
// Simple Memory
uint64_t respCycle = req.cycle + latency;  // Fixed latency

// DDR Memory  
uint64_t respCycle = req.cycle + (isWrite? minWrLatency : minRdLatency);

// DRAMSim Memory
uint64_t respCycle = req.cycle + minLatency;

// MD1 Memory (load-dependent)
uint64_t respCycle = req.cycle + curLatency;  // Based on current load
```

### DRAM Timing Accuracy

#### Bound Phase Assumptions
- **Best-case timing**: No bank conflicts, row buffer hits
- **Minimum delays**: Controller + device minimum latencies only
- **No queuing**: Requests processed immediately

#### Weave Phase Reality
- **Resource conflicts**: Bank busy, row buffer misses
- **Queuing delays**: Request buffers full, refresh cycles
- **Timing constraints**: tRAS, tRC, tRRD, etc. modeled accurately
- **Cross-request dependencies**: Proper ordering and timing

#### Correction Mechanism
```cpp
// Compare bound vs. weave phase timing
uint64_t boundTime = req.cycle + minLatency;      // Optimistic
uint64_t weaveTime = detailedSimulation(req);     // Realistic
uint64_t additionalDelay = weaveTime - boundTime; // Contention penalty

// Apply correction to core clock
core->curCycle += additionalDelay;
core->gapCycles += additionalDelay;
```

## Synchronization and Barriers

### Barrier Implementation

#### Barrier State Machine
```cpp
enum State {OFFLINE, WAITING, RUNNING, LEFT};

struct ThreadSyncInfo {
    volatile State state;
    volatile uint32_t futexWord;
    uint32_t lastIdx;
};
```

#### Synchronization Protocol
```cpp
void Barrier::sync(uint32_t tid, lock_t* schedLock) {
    threadList[tid].state = WAITING;
    runningThreads--;
    tryWakeNext(tid);  // May trigger phase end
    futex_unlock(schedLock);
    
    // Wait for wakeup from phase end
    while (threadList[tid].state == WAITING) {
        futex_wait(&threadList[tid].futexWord, 1);
    }
}
```

### Phase End Detection
```cpp
void Barrier::checkEndPhase(uint32_t tid) {
    if (curThreadIdx == runListSize && runningThreads == 0) {
        // All cores have reached barrier
        sched->callback();  // Trigger EndOfPhaseActions()
        curThreadIdx = 0;   // Reset for next phase
        wakeAllWaitingThreads();
    }
}
```

## Performance and Accuracy Trade-offs

### Simulation Speed
- **Bound Phase**: ~10-100x faster than detailed simulation
- **Phase Length**: Configurable trade-off (default 10,000 cycles)
  - Longer phases: Better bound phase speedup, less frequent sync overhead
  - Shorter phases: More accurate timing, higher sync overhead

### Timing Accuracy
- **Overall Accuracy**: Within 1-2% of detailed cycle-by-cycle simulation
- **Bound Phase Error**: Optimistic timing accumulates during phase
- **Weave Phase Correction**: Detailed simulation corrects accumulated error
- **Error Bounds**: Limited by phase length and memory system complexity

### Scalability
- **Core Count**: Barrier synchronization overhead grows with core count
- **Memory Complexity**: Weave phase cost scales with memory system detail
- **Phase Length Tuning**: Critical for performance vs. accuracy balance

This design enables zsim to simulate 1000+ core systems with cycle-level accuracy while maintaining reasonable simulation performance, making it suitable for large-scale microarchitectural studies.
