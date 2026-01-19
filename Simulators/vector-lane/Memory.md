# BRG_SP Cache Coherence Protocol Documentation
## SLICC-Based Memory System for Software-Defined Vector Processing

This document describes the BRG_SP (Berkeley Research Group Scratchpad) cache coherence protocol implemented in SLICC (Specification Language for Implementing Cache Coherence) for the gem5-mesh simulator.

---

## Table of Contents
1. [Overview](#overview)
2. [SLICC Language Basics](#slicc-language-basics)
3. [Protocol Architecture](#protocol-architecture)
4. [L1 Instruction Cache (L1Icache)](#l1-instruction-cache-l1icache)
5. [L2 Cache / LLC](#l2-cache--llc)
6. [Vector Load Support](#vector-load-support)
7. [State Transitions](#state-transitions)
8. [Message Types](#message-types)
9. [Integration with Mesh Network](#integration-with-mesh-network)

---

## Overview

The BRG_SP protocol is a custom cache coherence protocol designed for the Software-Defined Vector Processing architecture. It implements:

- **Two-level cache hierarchy**: L1 Instruction Cache + Shared L2/LLC
- **Vector load acceleration**: Special handling for vector group memory operations
- **Scratchpad integration**: Direct stores to scratchpad memories for vector data distribution
- **LL/SC support**: Load-Linked/Store-Conditional for synchronization

### Protocol Files

| File | Description |
|------|-------------|
| `BRG_SP.slicc` | Top-level protocol definition |
| `BRG_SP-L1Icache.sm` | L1 Instruction Cache state machine |
| `BRG_SP-L2cache.sm` | L2/LLC state machine with vector support |

### Protocol Declaration (BRG_SP.slicc)

```slicc
protocol "BRG_SP";
include "RubySlicc_interfaces.slicc";
include "BRG_SP-L2cache.sm";
include "BRG_SP-L1Icache.sm";
```

---

## SLICC Language Basics

SLICC (Specification Language for Implementing Cache Coherence) is gem5's domain-specific language for defining cache coherence protocols.

### Key SLICC Constructs

| Construct | Description |
|-----------|-------------|
| `machine()` | Defines a cache controller state machine |
| `state_declaration()` | Defines the set of states |
| `enumeration()` | Defines events that trigger transitions |
| `structure()` | Defines data structures (cache entries, TBEs) |
| `in_port()` | Defines input message ports |
| `out_port()` | Defines output message ports |
| `action()` | Defines atomic actions |
| `transition()` | Defines state transitions |

### Example Action Definition

```slicc
action(sdc_sendDataFromCache, "sdc", desc = "Send response data from cache") {
    peek(requestNetwork_in, LLCRequestMsg) {
        enqueue(responseNetwork_out, LLCResponseMsg, cache_resp_latency) {
            out_msg.Type        := LLCResponseType:DATA;
            out_msg.LineAddress := in_msg.LineAddress;
            out_msg.Destination.add(in_msg.Requestor);
            out_msg.DataBlk     := cache_entry.DataBlk;
        }
    }
}
```

### Example Transition

```slicc
transition(I, READ, IV) {
    atb_allocateTBE;           // Allocate transaction buffer
    art_addRequestorToTBE;     // Store requestor info
    acb_allocateCacheBlock;    // Allocate cache line
    imr_issueMemReadRequest;   // Send memory request
    prq_popRequestQueue;       // Pop input message
    pms_profileMissAccess;     // Update statistics
}
```

---

## Protocol Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU Cores (Mesh)                         │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│  Core 0 │  Core 1 │  Core 2 │  Core 3 │   ...   │ Core N  │     │
│ ┌─────┐ │ ┌─────┐ │ ┌─────┐ │ ┌─────┐ │         │ ┌─────┐ │     │
│ │L1 I$│ │ │L1 I$│ │ │L1 I$│ │ │L1 I$│ │         │ │L1 I$│ │     │
│ └──┬──┘ │ └──┬──┘ │ └──┬──┘ │ └──┬──┘ │         │ └──┬──┘ │     │
│    │    │    │    │    │    │    │    │         │    │    │     │
│ ┌──┴──┐ │ ┌──┴──┐ │ ┌──┴──┐ │ ┌──┴──┐ │         │ ┌──┴──┐ │     │
│ │SPAD │ │ │SPAD │ │ │SPAD │ │ │SPAD │ │         │ │SPAD │ │     │
│ └─────┘ │ └─────┘ │ └─────┘ │ └─────┘ │         │ └─────┘ │     │
└────┬────┴────┬────┴────┬────┴────┬────┴─────────┴────┬────┴─────┘
     │         │         │         │                   │
     └─────────┴─────────┴────┬────┴───────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Ruby Network    │
                    │    (Garnet)       │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  L2 Cache / LLC   │
                    │   (BRG_SP-L2)     │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Memory Controller │
                    │      (DRAM)        │
                    └───────────────────┘
```

---

## L1 Instruction Cache (L1Icache)

**File:** `BRG_SP-L1Icache.sm`

### Machine Declaration

```slicc
machine(MachineType:L1Cache, "SP_LLC (L1I Cache)")
    : Sequencer* sequencer;
      CacheMemory * L1cache;
      Cycles issue_latency := 80;   // Time to send to L2
      Cycles l2_hit_latency := 18;  // L2 hit latency
```

### States

| State | Permission | Description |
|-------|------------|-------------|
| `I` | Invalid | Block not present in cache |
| `V` | Read_Only | Block is valid and clean |

### Events

| Event | Description |
|-------|-------------|
| `Fetch` | Instruction fetch request from core |
| `Repl` | Cache line replacement needed |
| `Data` | Data response received from L2 |

### Cache Entry Structure

```slicc
structure(Entry, desc="...", interface="AbstractCacheEntry") {
    State CacheState,  desc="cache state";
    DataBlock DataBlk, desc="data for the block";
}
```

### Transaction Buffer Entry (TBE)

```slicc
structure(TBE, desc="...") {
    State TBEState,    desc="Transient state";
    DataBlock DataBlk, desc="data for the block";
}
```

### Message Ports

| Port | Direction | Virtual Network | Type |
|------|-----------|-----------------|------|
| `requestToNetwork` | Out | 1 | Request |
| `responseFromNetwork` | In | 0 | Response |
| `mandatoryQueue` | In | - | CPU requests |

### Key Transitions

```
State: I (Invalid)
  │
  ├── Fetch ──────────> Issue READ to L2, remain in I (wait for data)
  │
  └── Data ───────────> V (allocate, write data, signal done)

State: V (Valid)
  │
  ├── Fetch ──────────> Hit! Return data, stay in V
  │
  └── Repl ───────────> I (invalidate cache entry)
```

### Actions

| Action | Code | Description |
|--------|------|-------------|
| `pht_profileHitAccess` | `pht` | Increment hit counter |
| `pms_profileMissAccess` | `pms` | Increment miss counter |
| `ic_invCache` | `ic` | Invalidate cache entry |
| `nS_issueRdBlkS` | `nS` | Issue read request to L2 |
| `a_allocate` | `a` | Allocate cache block |
| `w_writeCache` | `w` | Write data to cache |
| `l_loadDone` | `l` | Signal load complete to sequencer |

---

## L2 Cache / LLC

**File:** `BRG_SP-L2cache.sm`

### Machine Declaration

```slicc
machine(MachineType:L2Cache, "SP LLC cache")
    : CacheMemory* cacheMemory;
      Cycles cache_resp_latency := 1;
      Cycles to_memory_controller_latency := 1;
      Cycles mem_to_cpu_latency := 1;
      int meshDimX := 2;
      int meshDimY := 2;
      int netWidth := 1;  // Network width in words
```

### States

| State | Permission | Description |
|-------|------------|-------------|
| `I` | Invalid | Not present/invalid |
| `IV` | Busy | Waiting for data (read miss) |
| `IM` | Busy | Waiting for data (write miss) |
| `V` | Read_Write | Valid & clean |
| `M` | Read_Write | Valid & dirty (modified) |
| `VV` | Read_Write | Valid & clean, vector read in progress |
| `VM` | Read_Write | Valid & dirty, vector read in progress |
| `MI` | Busy | Waiting for writeback ACK |
| `IV_LL` | Busy | Waiting for data (LL request) |
| `IA` | Busy | Waiting for data (ATOMIC request) |

### Events

| Event | Description |
|-------|-------------|
| `READ` | Scalar read request |
| `VREAD0` | Initial vector read |
| `VREAD1` | Middle vector read(s) |
| `VREAD2` | Final vector read |
| `SPLOAD` | Store to scratchpad |
| `WRITE` | Write request |
| `LL` | Load-Linked request |
| `SC` | Store-Conditional request |
| `ATOMIC` | Atomic operation request |
| `Repl` | Replacement trigger |
| `STALL` | Stall for cycle |
| `VEC_STALL_WAIT` | Stall due to vector port busy |
| `VEC_STALL_WAIT_MEM` | Stall due to vector mem port busy |
| `Memory_Data` | Data from memory |
| `VData0/1/2` | Vector data from memory |
| `SP_Mem_Data` | Scratchpad data from memory |
| `Memory_Ack` | Writeback acknowledgment |

### Cache Entry Structure

```slicc
structure(Entry, desc = "...", interface = "AbstractCacheEntry") {
    State CacheState,     desc = "Cache state";
    DataBlock DataBlk,    desc = "Data in the block";
    WriteMask writeMask,  desc = "Dirty byte mask";
    MachineID LLSC_owner, desc = "Owner of LLSC lock";
}
```

### TBE Structure (Extended for Vector)

```slicc
structure(TBE, desc = "...") {
    State     TBEState,        desc = "Transient state";
    DataBlock DataBlk,         desc = "Data for the block";
    WriteMask writeMask,       desc = "Dirty byte mask";
    MachineID Requestor,       desc = "Requestor's ID";
    int       SeqNum,          desc = "Sequence number";
    int       XDim,            desc = "X dimension for vector";
    int       YDim,            desc = "Y dimension for vector";
    Addr      WordAddress,     desc = "Word address of element";
    Addr      PrefetchAddress, desc = "Prefetch address";
    int       CoreOffset,      desc = "Offset from core origin";
    int       SubCoreOffset,   desc = "Offset within core's data";
    int       CountPerCore,    desc = "Responses per core";
    int       RespCnt,         desc = "Total responses needed";
    int       PrefetchConfig,  desc = "Wide load config";
    LLCRequestType InType,     desc = "Original request type";
}
```

### Global Registers (Vector Support)

```slicc
// Counters for tracking vector operations
int _vecCntCpuSide, default=0;
int _vecCntMemSide, default=0;

Tick _lastVecRespTimeCpuSide, default=0;
Tick _lastVecRespTimeMemSide, default=0;

Addr _lastServicedVecAddrCpuSide, default=0;
Addr _lastServicedVecAddrMemSide, default=0;
```

---

## Vector Load Support

The L2 cache includes specialized support for vector memory operations that distribute data to multiple cores in the mesh.

### Vector Load Flow

```
1. Master core issues SPLOAD/VREAD request with:
   - XDim, YDim: Vector group dimensions
   - RespCnt: Total words to distribute
   - CountPerCore: Words per destination core
   - CoreOffset: Starting core in group

2. L2 Cache processes request:
   - VREAD0: First response, transition to VV/VM state
   - VREAD1: Middle responses (may span multiple cycles)
   - VREAD2: Final response, return to V/M state

3. For each word, L2 calculates:
   - Which core should receive it (getCoreIdx)
   - Scratchpad address for that core (getSpadIdx)
   - Word offset in cache line (getWordIdx)

4. Data sent as REVDATA/REDATA messages to scratchpads
```

### Core Index Calculation

```slicc
int getCoreIdx(int coreIdx, int coreOffset, int subCoreOffset, 
               int vecDimX, int vecDimY, int countPerCore, bool isCpuSide) {
    int vecCnt := getPortVecCnt(isCpuSide);
    
    // Vector core in group
    int vecFlat := coreOffset + (subCoreOffset + vecCnt) / countPerCore;
    
    // Get X,Y in vector group
    int vecX := mod(vecFlat, vecDimX);
    int vecY := vecFlat / vecDimX;
    
    // Linearize on mesh
    int meshVecFlat := vecX + vecY * meshDimX;
    
    return coreIdx + meshVecFlat;
}
```

### Vector Response Throttling

Only one vector response can be issued per cycle per port:

```slicc
bool canIssueVecLoad(bool isCpuSide) {
    if (getPortLastVecRespTime(isCpuSide) == clockEdge()) {
        return false;  // Already issued this cycle
    }
    return true;
}
```

---

## State Transitions

### Scalar Read Path

```
┌───┐                      ┌────┐                      ┌───┐
│ I │ ──── READ ────────> │ IV │ ── Memory_Data ───> │ V │
└───┘   (allocate TBE,    └────┘   (write cache,     └───┘
         issue mem req)             send response)
         
┌───┐
│ V │ ──── READ ────────> (hit, send data, stay in V)
└───┘

┌───┐
│ M │ ──── READ ────────> (hit, send data, stay in M)
└───┘
```

### Write Path

```
┌───┐                      ┌────┐                      ┌───┐
│ I │ ──── WRITE ───────> │ IM │ ── Memory_Data ───> │ M │
└───┘   (allocate,        └────┘   (write data,      └───┘
         save data in TBE)          send ACK)

┌───┐
│ V │ ──── WRITE ───────> │ M │  (write, send ACK)
└───┘                      └───┘

┌───┐
│ M │ ──── WRITE ───────> (write, send ACK, stay in M)
└───┘
```

### Vector Read Path (Hit)

```
┌───┐                      ┌────┐                      ┌───┐
│ V │ ──── VREAD0 ──────> │ VV │ ── VREAD1..N ─────> │ VV │
└───┘   (first vec resp)   └────┘   (middle resps)    └────┘
                                          │
                                          │ VREAD2 (last)
                                          ▼
                                       ┌───┐
                                       │ V │
                                       └───┘
```

### Vector Read Path (Miss)

```
┌───┐                      ┌────┐
│ I │ ──── VREAD0 ──────> │ IV │ ── VData0 ──> (first vec resp from mem)
└───┘                      └────┘       │
                                        │ VData1..N
                                        ▼
                                   (middle resps)
                                        │
                                        │ VData2
                                        ▼
                                     ┌───┐
                                     │ V │
                                     └───┘
```

### Replacement

```
┌───┐
│ V │ ──── Repl ────────> │ I │  (deallocate)
└───┘                      └───┘

┌───┐                      ┌────┐                      ┌───┐
│ M │ ──── Repl ────────> │ MI │ ── Memory_Ack ───> │ I │
└───┘   (issue writeback) └────┘   (deallocate)     └───┘
```

### Load-Linked / Store-Conditional

```
┌───┐                      
│ V │ ──── LL ──────────> (set LLSC_owner, return data)
└───┘                      

┌───┐                      
│ V │ ──── SC ──────────> (check owner, conditional write)
└───┘                      │
                           ├── owner matches: write, send success ACK
                           └── owner mismatch: send fail ACK
```

---

## Message Types

### Request Messages (LLCRequestMsg)

| Field | Type | Description |
|-------|------|-------------|
| `LineAddress` | `Addr` | Cache line address |
| `WordAddress` | `Addr` | Word address (for vector) |
| `Type` | `LLCRequestType` | READ, WRITE, SPLOAD, LL, SC, ATOMIC |
| `Requestor` | `MachineID` | Source of request |
| `Destination` | `NetDest` | Destination(s) |
| `DataBlk` | `DataBlock` | Data for writes |
| `writeMask` | `WriteMask` | Bytes being written |
| `XDim`, `YDim` | `int` | Vector dimensions |
| `RespCnt` | `int` | Response count |
| `CoreOffset` | `int` | Core offset in group |
| `CountPerCore` | `int` | Responses per core |
| `PrefetchAddress` | `Addr` | Base scratchpad address |
| `PrefetchConfig` | `int` | Wide load configuration |

### Response Messages (LLCResponseMsg)

| Field | Type | Description |
|-------|------|-------------|
| `LineAddress` | `Addr` | Address (or SPAD addr for vector) |
| `Type` | `LLCResponseType` | DATA, ACK, REDATA, REVDATA |
| `Destination` | `NetDest` | Destination(s) |
| `DataBlk` | `DataBlock` | Data payload |
| `BlkIdx` | `int` | Word index in block |
| `Len` | `int` | Number of words |
| `SC_Success` | `bool` | SC success flag |
| `SeqNum` | `int` | Sequence number |

### Response Types

| Type | Description |
|------|-------------|
| `DATA` | Normal data response |
| `ACK` | Write/SC acknowledgment |
| `REDATA` | Remote store (single) to scratchpad |
| `REVDATA` | Remote store (vector) to scratchpad |

---

## Integration with Mesh Network

### Virtual Networks

| VNet | Direction | Type |
|------|-----------|------|
| 0 | Response | Data/ACK from L2 to cores |
| 1 | Request | Requests from cores to L2 |

### Message Size Types

```slicc
MessageSizeType getFlitMessageSize(MachineID requestor) {
    if (machineIDToMachineType(requestor) == MachineType:L1Cache) {
        return MessageSizeType:Response_Data;  // Full cache line
    } else if (machineIDToMachineType(requestor) == MachineType:Scratchpad) {
        return MessageSizeType:SingleWordData; // Single word
    }
}
```

### Scratchpad Integration

The protocol directly stores vector load results to scratchpad memories:

```slicc
action(rsc_remoteStoreFromCache, "rsc", desc = "Remote store to spad") {
    peek(requestNetwork_in, LLCRequestMsg) {
        // Calculate destination scratchpad
        int spadIdx := getCoreIdx(...);
        MachineID spad := createMachineID(MachineType:Scratchpad, intToID(spadIdx));
        
        // Calculate scratchpad address
        Addr spadAddr := getSpadIdx(baseCntAddr, coreOffset, ...);
        
        enqueue(responseNetwork_out, LLCResponseMsg, cache_resp_latency) {
            out_msg.Type := LLCResponseType:REVDATA;
            out_msg.LineAddress := spadAddr;
            out_msg.Destination.add(spad);
            out_msg.DataBlk := cache_entry.DataBlk;
            out_msg.BlkIdx := wordOffset;
        }
    }
}
```

---

## Statistics

### L1 Cache Statistics
- `demand_hits` - Number of cache hits
- `demand_misses` - Number of cache misses

### L2 Cache Statistics
- `demand_hits` - Scalar hit count
- `demand_misses` - Scalar miss count
- Vector hits/misses counted per-word

### Profiling Actions

```slicc
action(pvh_profileVectorAccessHit, "pvh", desc = "Profile vector hit") {
    peek(requestNetwork_in, LLCRequestMsg) {
        cacheMemory.incDemandHits(in_msg.RespCnt);
    }
}

action(pvm_profileVectorAccessMiss, "pvm", desc = "Profile vector miss") {
    peek(memQueue_in, MemoryMsg) {
        cacheMemory.incDemandHits(tbe.RespCnt-1);  // All but first are "hits"
    }
}
```

---

## Key Design Decisions

1. **Shared L2/LLC**: Single shared last-level cache for all cores simplifies coherence
2. **No L1 Data Cache**: Data goes directly through scratchpads for vector operations
3. **Vector State Lock**: VV/VM states prevent replacement during multi-cycle vector reads
4. **Two Port Counters**: Separate CPU-side and memory-side vector counters prevent conflicts
5. **Per-Cycle Throttling**: Only one vector response per port per cycle for timing accuracy
6. **Direct Scratchpad Writes**: Bypass core for vector data distribution
