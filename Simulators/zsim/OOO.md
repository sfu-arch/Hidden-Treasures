# OOOCoreRecorder Documentation

## Overview

`OOOCoreRecorder` is a timing event recorder for out-of-order (OOO) CPU cores in the zsim microarchitectural simulator. It builds a dependency graph of timing events to model how instructions and memory operations execute with contention delays.

## Core Concept

The recorder doesn't directly simulate execution - instead, it **records events as a directed acyclic graph (DAG)** that will be simulated later by the contention simulation infrastructure. This is part of zsim's "bound-weave" model that separates functional simulation from timing simulation.

## State Machine

```
HALTED ──notifyJoin──> RUNNING ──notifyLeave──> DRAINING ──cSimEnd──> HALTED
   ^                                                                      |
   └──────────────────────────────────────────────────────────────────────┘
```

### States

- **HALTED**: Core is inactive (thread blocked/sleeping). No events are being produced.
- **RUNNING**: Core is actively executing, producing issue events for operations.
- **DRAINING**: Thread has left (e.g., blocking syscall), waiting for in-flight events to complete before transitioning to HALTED.

## Two-Clock System

The recorder maintains two parallel clock references:

### curCycle (Physical Clock)
- Actual simulation time including all contention delays
- Advances with `gapCycles` adjustments
- Used for setting `minStartCycle` on events

### zllCycle (Logical/Zero-Latency-Less Clock)
- Computed as: `zllCycle = curCycle - gapCycles`
- **Stable across `gapCycles` adjustments**
- Used for event ordering and delay calculations
- Stored in event objects as `zllStartCycle`

### Why Two Clocks?

When contention simulation completes, both `curCycle` and `gapCycles` increase by the same `skew` amount:

```cpp
curCycle += skew;
gapCycles += skew;
```

Since both increase equally, `zllCycle = curCycle - gapCycles` remains constant. This stability is crucial for:
1. Correctly ordering events created in different phases
2. Computing delays between events with different `gapCycles` values
3. Maintaining consistent event timestamps across clock adjustments

### Why Not Just Use curCycle?

**If we used `curCycle` directly to timestamp events, contention simulation would corrupt the timing.**

Consider this scenario across multiple phases:

```
Phase 1:
  - Create issue event A at curCycle=1000, gapCycles=0
  - Store A.timestamp = 1000

Phase 2 (after contention sim):
  - Skew detected = 400
  - curCycle += 400 → now 1400
  - gapCycles += 400 → now 400
  
Phase 3:
  - Create issue event B at curCycle=2000, gapCycles=400
  - Need to compute delay from A to B
  
  Using curCycle (WRONG):
    delay = B.curCycle - A.curCycle
    delay = 2000 - 1000 = 1000 ❌
    Problem: This includes the 400 cycles of contention!
    
  Using zllCycle (CORRECT):
    A.zllCycle = 1000 - 0 = 1000
    B.zllCycle = 2000 - 400 = 1600
    delay = 1600 - 1000 = 600 ✓
    Correct: Only the logical execution time
```

#### The Problem with curCycle

When computing `delay = curCycle_B - curCycle_A`:
- **Double-counts contention**: The 400 cycles of skew are already tracked in `gapCycles`
- **Wrong delays**: Events appear to take longer than they actually do in logical time
- **Broken ordering**: Events from different phases can't be correctly compared

#### Why zllCycle Solves This

The `zllCycle` "factors out" the contention accumulation:

```
Event created at curCycle=X with gapCycles=G
  → Event.zllStartCycle = X - G

Later, after skew S is added:
  → curCycle becomes X + S
  → gapCycles becomes G + S
  → Event.zllStartCycle still equals (X+S) - (G+S) = X - G ✓
```

**The zllCycle coordinate system doesn't shift when contention is added!**

This means:
- Events created before contention adjustments maintain correct relative timing
- Delays computed using zllCycle represent **logical execution time only**
- Contention is tracked separately in `gapCycles` and applied globally
- The timeline is **stable** across phase boundaries

## Event Types

### OOOIssueEvent
- Marks when instructions/operations are issued from the core
- Forms the "spine" of the event chain
- Each new issue event links to the previous one via delay edges
- Tracked via `lastEvProduced` pointer
- When simulated, calls `reportIssueEventSimulated()` to update tracking

### OOODispatchEvent
- Created when memory operations dispatch to the memory hierarchy
- Generated in `recordAccess()` for GET operations
- Links with preceding response events to model response-to-dispatch dependencies

### OOORespEvent
- Represents when memory responses return to the core
- Stored in `futureResponses` priority queue (min-heap ordered by `zllStartCycle`)
- Links to subsequent issue events to model response-to-issue dependencies
- Sets `cRec = nullptr` when simulated to mark completion

## Event Graph Construction

### addIssueEvent(uint64_t evCycle)

This is the core linking function that creates two types of dependencies:

#### 1. Link with Pending Responses (Vertical Dependencies)
```cpp
while (!futureResponses.empty()) {
    FutureResponse firstResp = futureResponses.top();
    if (firstResp.zllStartCycle >= zllCycle) break;
    if (firstResp.ev) {
        firstResp.ev->addChild(ev, eventRecorder);
        maxCycle = firstResp.zllStartCycle;
    }
    futureResponses.pop();
}
```
- Pops all response events with `zllStartCycle < zllCycle`
- These earlier responses become parent events
- Models: "memory responses affect subsequent execution"
- Computes `preDelay` based on the latest response

#### 2. Link with Previous Issue (Horizontal Dependencies)
```cpp
uint32_t issueDelay = zllCycle - lastEvProduced->zllStartCycle - preDelay;
DelayEvent* dIssue = new (eventRecorder) DelayEvent(issueDelay);
lastEvProduced->addChild(dIssue, eventRecorder)->addChild(ev, eventRecorder);
```
- Creates delay edge from previous issue event
- Models: "instruction stream ordering"
- Ensures monotonic progression of issue events

**Why Compute This Delay?**

The delay between consecutive issue events encodes the **logical execution time** that elapsed in the core's instruction stream. This is essential for building the event DAG that contention simulation will process.

**The Delay Formula:**
```cpp
issueDelay = zllCycle - lastEvProduced->zllStartCycle - preDelay
```

Breaking this down:
- `zllCycle`: Current event's logical time
- `lastEvProduced->zllStartCycle`: Previous event's logical time
- `preDelay`: Already accounted for by response dependencies
- Result: Net delay for the horizontal edge

**Why Subtract preDelay?**

If responses have already created dependencies, we don't want to double-count that time. The `preDelay` represents the gap between the latest response and the current issue, which is already captured by the vertical edges.

### Concrete Example: Building the Event Chain

**Scenario**: Core executes 3 operations across multiple phases

```
Phase 1 (gapCycles=0):
  Operation A issues at curCycle=1000
    → zllCycle = 1000 - 0 = 1000
    → Create IssueEvent_A with zllStartCycle=1000
    → lastEvProduced = IssueEvent_A

Phase 2: Contention simulation runs
  - Memory delays detected: skew=400
  - curCycle += 400 → now 1400
  - gapCycles += 400 → now 400
  - zllCycle = 1400 - 400 = 1000 (unchanged!)

Phase 3 (gapCycles=400):
  Operation B issues at curCycle=2000
    → zllCycle = 2000 - 400 = 1600
    → Need to link B to A with correct delay
```

**Computing the Delay (No Prior Responses):**

```cpp
// In addIssueEvent for Operation B
zllCycle = 2000 - 400 = 1600
lastEvProduced->zllStartCycle = 1000
preDelay = 0  // No pending responses

issueDelay = 1600 - 1000 - 0 = 600 cycles
```

**Result**: The delay edge between A and B is **600 cycles**

This correctly represents:
- Core did 600 cycles of logical work between A and B
- The 400 cycles of contention are NOT included (already in gapCycles)
- When contention simulation runs, it applies gapCycles globally

**Event Graph Created:**
```
IssueEvent_A (zll=1000) → [DelayEvent(600)] → IssueEvent_B (zll=1600)
```

**With Responses: More Complex Example**

```
Phase 3 continued:
  Operation B issues at curCycle=2000
    - zllCycle = 1600
    - futureResponses contains: {(zll=1200, RespEvent_X), (zll=1400, RespEvent_Y)}
```

**Step 1: Link with Earlier Responses**
```cpp
// Pop responses with zllStartCycle < 1600
while (!futureResponses.empty()) {
    firstResp = futureResponses.top()
    
    // Response X: zll=1200 < 1600 ✓
    if (1200 < 1600) {
        RespEvent_X->addChild(IssueEvent_B)
        maxCycle = 1200
        futureResponses.pop()
    }
    
    // Response Y: zll=1400 < 1600 ✓
    if (1400 < 1600) {
        RespEvent_Y->addChild(IssueEvent_B)
        maxCycle = 1400
        futureResponses.pop()
    }
    
    // No more responses with zll < 1600
}
```

**Step 2: Calculate PreDelay**
```cpp
preDelay = (maxCycle < zllCycle) ? (zllCycle - maxCycle) : 0
preDelay = (1400 < 1600) ? (1600 - 1400) : 0
preDelay = 200 cycles
```

This represents the gap between the latest response (zll=1400) and current issue (zll=1600).

**Step 3: Calculate Issue Delay**
```cpp
issueDelay = zllCycle - lastEvProduced->zllStartCycle - preDelay
issueDelay = 1600 - 1000 - 200
issueDelay = 400 cycles
```

**Why subtract preDelay?**
- Total logical time: 1600 - 1000 = 600 cycles
- Time already captured by response edge: 200 cycles (response Y at 1400 → issue B at 1600)
- Remaining time for horizontal edge: 400 cycles

**Event Graph Created:**
```
IssueEvent_A (zll=1000)
    ↓ [DelayEvent(400)]
IssueEvent_B (zll=1600)
    ↑ (from RespEvent_X at zll=1200)
    ↑ (from RespEvent_Y at zll=1400, with implicit 200-cycle gap)
```

**Why This Matters:**

1. **Correct Timing**: Total path from A to B:
   - Horizontal: 400 cycles
   - Vertical (via latest response): 1400 - 1000 = 400 cycles (from A) + 200 cycles (gap) = 600 total
   - Either path gives 600 cycles ✓

2. **No Double-Counting**: The 200-cycle gap isn't counted in both edges

3. **Contention Separation**: The 400 cycles of skew (gapCycles) isn't in any delay
   - When simulated, physical times will be:
   - IssueEvent_A: 1000 + 400 = 1400
   - IssueEvent_B: 1600 + 400 = 2000
   - Delay preserved: 2000 - 1400 = 600 ✓

**What If We Used curCycle?**

```cpp
// WRONG: Using curCycle directly
issueDelay = curCycle_B - curCycle_A - preDelay
issueDelay = 2000 - 1000 - 200 = 800 ❌
```

This would:
- Double-count the 400 cycles of contention
- Create an 800-cycle delay instead of 400
- Result in total path of 1000 cycles (wrong!)
- Break the event graph timing

This creates a DAG where:
- **Vertical edges**: Response → Issue (memory delays affect execution)
- **Horizontal edges**: Issue → Issue (instruction ordering with no double-counting)

### Comparison: Why OOO Needs preDelay (In-Order Cores Don't)

**preDelay only exists because OOO cores allow execution to overlap with memory operations.**

#### CoreRecorder (In-Order Cores) - No preDelay

In `core_recorder.cpp` for TimingCore:

```cpp
void CoreRecorder::recordAccess(uint64_t startCycle) {
    if (IsGet(tr.type)) {
        uint64_t delay = tr.reqCycle - prevRespCycle;  // ← Simple subtraction!
        TimingEvent* ev = new (eventRecorder) TimingCoreEvent(delay, prevRespCycle - gapCycles, this);
        prevRespEvent->addChild(ev, eventRecorder)->addChild(tr.startEvent, eventRecorder);
        prevRespEvent = tr.endEvent;      // ← Wait for THIS response
        prevRespCycle = tr.respCycle;      // ← Update to THIS response time
    }
}
```

**Key differences:**
- **Single response tracking**: Only `prevRespEvent` pointer (not a queue)
- **Core stalls**: Each load blocks until response returns
- **No overlap**: Next operation waits until current one completes
- **Simple delay**: `delay = reqCycle - prevRespCycle` (no preDelay subtraction)
- **Linear chain**: Events form a simple sequence, not a DAG

**Event model for in-order:**
```
IssueA → Load → [STALL] → Response → IssueB → Load → [STALL] → Response → IssueC
```

Everything is serialized. No need to track multiple pending responses or compute overlaps.

#### OOOCoreRecorder - Requires preDelay

**Event model for out-of-order:**
```
IssueA → LoadA dispatched → IssueB → IssueC → LoadB dispatched → ResponseA arrives → IssueD
```

Core continues executing while waiting for responses. This creates:
- **futureResponses queue**: Track multiple pending responses
- **preDelay subtraction**: Avoid double-counting overlapped execution
- **Complex DAG**: Both horizontal (issue→issue) and vertical (response→issue) edges

| Aspect | CoreRecorder (In-Order) | OOOCoreRecorder (OOO) |
|--------|------------------------|------------------------|
| Response tracking | `prevRespEvent` (single pointer) | `futureResponses` (min-heap) |
| Pipeline behavior | Stalls on each load | Continues executing |
| Delay formula | `reqCycle - prevRespCycle` | `zllCycle - lastEvProduced->zllStartCycle - preDelay` |
| preDelay needed? | **No** - no overlap | **Yes** - captures overlap |
| Event structure | Simple chain | Complex DAG |
| Execution overlap | None (serialized) | Full (memory + core parallel) |

**The fundamental insight:** In-order cores have no overlap between memory system work and core execution, so they use simple serialized event chains. OOO cores exploit memory-level parallelism by continuing execution during memory operations, requiring sophisticated dependency tracking via `preDelay` to avoid double-counting the overlapped time.

## Memory Access Recording

### recordAccess(curCycle, dispatchCycle, respCycle)

Handles memory operations by popping a `TimingRecord` and constructing appropriate event chains.

#### GET Operations (Loads)
```
lastEvProduced → delay → dispatch → delay → tr.startEvent
                    ↓
             (preceding responses also link to dispatch)
                    
tr.endEvent → respEvent → (added to futureResponses)
```

Key steps:
1. Create new issue event at `curCycle`
2. Create dispatch event with links from:
   - Previous issue (via delay)
   - All earlier pending responses (models dispatch stalling on prior loads)
3. Link dispatch → request to memory hierarchy
4. Create response event, add to `futureResponses` heap

#### PUT Operations (Stores)
```
lastEvProduced → delay → tr.startEvent
```

Simpler:
- Fire-and-forget: issue → request
- No response tracking (writebacks don't block execution)
- `tr.endEvent` not linked (it's a writeback in cache hierarchy)

## Contention Simulation Integration

### cSimStart(uint64_t curCycle)

Called before contention simulation begins:

**RUNNING state:**
- Adds "taper" issue event at `nextPhaseCycle` if not already tapered
- Ensures event chain reaches the phase boundary

**DRAINING state:**
- Flushes `futureResponses` (long leave, won't complete this phase)
- Advances `curCycle` to phase boundary if needed

### cSimEnd(uint64_t curCycle)

Called after contention simulation completes:

1. **Calculate skew:**
```cpp
uint64_t lastEvCycle1 = lastEvSimulatedZllStartCycle + gapCycles;  // ideal time
uint64_t lastEvCycle2 = lastEvSimulatedStartCycle;                 // actual time
uint64_t skew = lastEvCycle2 - lastEvCycle1;
```

2. **Adjust clocks:**
```cpp
curCycle += skew;
gapCycles += skew;
```
This keeps `zllCycle` constant while accounting for contention delays.

3. **Clean up simulated responses:**
- Responses with `cRec != this` have completed → set `ev = nullptr`
- Prevents linking with stale events

4. **Detect DRAINING → HALTED transition:**
- If `lastEvProduced == nullptr`, all events simulated
- Transition to HALTED, record `lastUnhaltedCycle`

## Key Invariants

1. **Monotonic zllCycle**: `lastEvProduced->zllStartCycle` never decreases
2. **Ordered responses**: `futureResponses` heap maintains `zllStartCycle` ordering
3. **Null lastEvProduced**: Indicates all produced events have simulated (enables HALTED transition)
4. **Stable zllCycle**: Remains constant across `gapCycles` adjustments
5. **Non-negative skew**: Contention cannot make execution faster than ideal (asserted)

## Call Sites for addIssueEvent()

| Location | Called With | Purpose |
|----------|-------------|---------|
| `notifyJoin()` | `curCycle` | Resume execution from HALTED or DRAINING |
| `notifyLeave()` | `curCycle` | Taper execution before DRAINING |
| `recordAccess()` | `curCycle` | Issue event for memory operation |
| `cSimStart()` | `nextPhaseCycle` | Taper phase boundary when RUNNING |

Note: Only `cSimStart()` passes a value other than `curCycle` (specifically `nextPhaseCycle` for phase tapering).

## Statistics

### getUnhaltedCycles()
- Total cycles the core was active (excluding HALTED periods)
- Accounts for current HALTED state if applicable
```cpp
uint64_t haltedCycles = totalHaltedCycles + 
    ((state == HALTED) ? (cycle - lastUnhaltedCycle) : 0);
return cycle - haltedCycles;
```

### getContentionCycles()
- Total accumulated contention delays
```cpp
return totalGapCycles + gapCycles;
```

## Multi-Domain Support

When `zinfo->numDomains > 1`, issue events produce "crossing" events:
```cpp
lastEvProduced->produceCrossings(&eventRecorder);
eventRecorder.getCrossingStack().clear();
```

These handle synchronization between different clock domains in multi-domain simulations.

## Memory Management

Events are allocated using placement new with `eventRecorder`:
```cpp
OOOIssueEvent* ev = new (eventRecorder) OOOIssueEvent(...);
```

The `eventRecorder` manages a pool that's recycled after events complete, avoiding heap fragmentation.

## Important Implementation Notes

### Use-After-Free in cSimEnd()
The code contains a known use-after-free when checking `fr.ev->cRec`:
```cpp
for (FutureResponse& fr : GetPrioQueueContainer(futureResponses)) {
    if (fr.ev && fr.ev->cRec != this) {
        fr.ev = nullptr;  // Mark as simulated
    }
}
```

This is safe because:
- Called at end of weave phase
- Event space hasn't been recycled yet
- Check happens every phase

A proper fix would have events clear their own pointers.

### GetPrioQueueContainer Hack
Uses inheritance trick to access private `.c` member of `priority_queue`:
```cpp
template <class T, class S, class C>
S& GetPrioQueueContainer(std::priority_queue<T, S, C>& q) {
    struct PQE : private std::priority_queue<T, S, C> {
        static S& Container(std::priority_queue<T, S, C>& q) {
            return q.*&PQE::c;
        }
    };
    return PQE::Container(q);
}
```

This allows iterating over the heap without destructive pops.

## Summary

`OOOCoreRecorder` orchestrates timing event creation for OOO cores by:
1. Managing state transitions (HALTED ↔ RUNNING ↔ DRAINING)
2. Building event DAGs with proper dependencies
3. Using two-clock system for stability across contention adjustments
4. Tracking responses to model memory-to-execution dependencies
5. Integrating with zsim's bound-weave contention simulation

The result is accurate timing simulation that captures:
- Out-of-order execution effects
- Memory hierarchy contention
- Multi-cycle latencies
- Cross-domain synchronization

All while maintaining fast functional simulation through event-graph recording.
