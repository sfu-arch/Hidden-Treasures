# Software-Defined Vector Processing on Manycore Fabrics
## gem5-mesh Custom Module Documentation

This documentation covers the custom gem5 extensions implementing Software-Defined Vector Processing on Manycore Fabrics. The system enables a mesh of simple cores to operate as a configurable vector processor through CSR-based configuration.

---

# C++ Class Documentation

## Table of Contents
1. [Harness Class](#harness-class)
2. [Vector Class](#vector-class)
3. [VecInstSel Class](#vecinstsel-class)
4. [MeshHelper Class](#meshhelper-class)
5. [MeshPacketData Class](#meshpacketdata-class)
6. [ToMeshPort Class](#tomeshport-class)
7. [FromMeshPort Class](#frommeshport-class)
8. [CPIStack Class](#cpistack-class)
9. [Bind Specification (bind_spec.hh)](#bind-specification)

---

## Harness Class

**File:** `harness.hh`, `harness.cc`

### Overview
The `Harness` class is a test harness for debugging and testing the mesh network ports. It inherits from `ClockedObject` and provides a simple pass-through interface between CPU-side ports.

### Class Hierarchy
```
ClockedObject
    └── Harness
```

### Nested Classes

#### CPUSideSlavePort
A slave port that receives requests from the CPU side.

| Member | Type | Description |
|--------|------|-------------|
| `id` | `int` | Port identifier for vector ports |
| `owner` | `Harness*` | Pointer to owning Harness object |
| `needRetry` | `bool` | Flag indicating retry is needed |
| `blockedPacket` | `PacketPtr` | Stored packet when blocked |

**Key Methods:**
- `sendPacket(PacketPtr pkt)` - Send a packet with flow control
- `getAddrRanges()` - Return address ranges this port handles
- `trySendRetry()` - Send retry if port is free
- `recvTimingReq(PacketPtr pkt)` - Receive timing requests from master
- `recvFunctional(PacketPtr pkt)` - Handle functional (debug) accesses

#### CPUSideMasterPort
A master port that sends requests to the CPU side.

| Member | Type | Description |
|--------|------|-------------|
| `id` | `int` | Port identifier |
| `owner` | `Harness*` | Pointer to owning Harness |
| `blockedPacket` | `PacketPtr` | Stored packet when blocked |

**Key Methods:**
- `sendPacket(PacketPtr pkt)` - Send packet using `sendTimingReq()`
- `recvTimingResp(PacketPtr pkt)` - Handle timing responses
- `recvReqRetry()` - Handle retry requests
- `recvRangeChange()` - Handle address range changes

### Main Class Members

| Member | Type | Description |
|--------|------|-------------|
| `slavePorts` | `std::vector<CPUSideSlavePort>` | Vector of slave ports |
| `masterPorts` | `std::vector<CPUSideMasterPort>` | Vector of master ports |
| `blocked` | `bool` | Blocking state flag |
| `originalPacket` | `PacketPtr` | Original packet for upgrades |
| `waitingPortId` | `int` | Port ID awaiting response |

### Key Methods
- `handleRequest(PacketPtr pkt, int port_id)` - Process incoming requests
- `handleResponse(PacketPtr pkt)` - Process responses
- `sendResponse(PacketPtr pkt)` - Send response to CPU
- `handleFunctional(PacketPtr pkt)` - Handle functional accesses
- `getPort(const std::string& if_name, PortID idx)` - Get port by name

---

## Vector Class

**File:** `vector.hh`, `vector.cc`

### Overview
The `Vector` class is the core component handling instruction communication over the mesh network. It manages sending, receiving, and synchronizing instructions with neighboring cores. This class implements the software-defined vector processing paradigm.

### Class Hierarchy
```
Stage
    └── Vector
```

### Enumerations

#### InstSource
```cpp
typedef enum InstSource {
    None = 0,      // No source
    Pipeline,      // From local instruction pipeline
    Mesh           // From mesh network
} InstSource;
```

### Nested Structures

#### SenderState
Used to attach metadata to mesh packets.
```cpp
struct SenderState : public Packet::SenderState {
    std::shared_ptr<VecInstSel::MasterData> master_data;
};
```

### Key Members

| Member | Type | Description |
|--------|------|-------------|
| `_numInPortsActive` | `int` | Number of active input ports |
| `_numOutPortsActive` | `int` | Number of active output ports |
| `_stage` | `SensitiveStage` | Pipeline stage (FETCH/EXECUTE) |
| `_curCsrVal` | `RegVal` | Current CSR configuration value |
| `_stolenCredits` | `int` | Credits stolen for stall control |
| `_canRootSend` | `bool` | Whether core can be root master |
| `_canRecv` | `bool` | Whether stage receives from mesh |
| `_vecPassThrough` | `bool` | Pass-through mode flag |
| `_meshRevecId` | `int` | Mesh REVEC instruction ID |
| `_pipeRevecId` | `int` | Pipeline REVEC instruction ID |
| `_vecUops` | `VecInstSel` | Micro-op instruction selector |

### Core Methods

#### Configuration
- `setupConfig(int csrId, RegVal csrVal)` - Configure mesh ports based on CSR value
- `getConfigured()` - Check if non-default configuration is active
- `resetActive()` - Reset all port activity states

#### Tick and Pipeline
- `tick()` - Main execution tick, handles instruction routing
- `passInstructions()` - Pass instructions through without vector processing
- `doSquash(SquashComm::BaseSquash&, StageIdx)` - Handle pipeline squashes

#### Instruction Movement
- `pullPipeInstruction(IODynInstPtr&)` - Get instruction from local pipeline
- `pullMeshInstruction(IODynInstPtr&)` - Get instruction from mesh network
- `pushPipeInstToNextStage(const IODynInstPtr&)` - Send to local pipeline
- `pushMeshInstToNextStage(const IODynInstPtr&)` - Send to next mesh stage
- `forwardInstruction(const IODynInstPtr&)` - Broadcast to mesh neighbors

#### Stall Detection
- `canPullMesh()` - Check if mesh input can be pulled
- `canPullPipe()` - Check if pipeline input can be pulled
- `canPushMesh()` - Check if mesh output is ready
- `canPushPipe()` - Check if pipeline output is ready
- `isOutMeshStalled()` - Check mesh output stall status
- `isInMeshStalled()` - Check mesh input stall status

#### Role Detection
- `isRootMaster()` - Is this the vector group leader?
- `isMaster()` - Does this core send to mesh?
- `isSlave()` - Does this core receive from mesh?
- `canWriteMesh()` - Can write to mesh ports?
- `canReadMesh()` - Can read from mesh ports?

#### Packet Creation
- `createMeshPacket(RegVal payload)` - Create packet with data payload
- `createMeshPacket(const VecInstSel::MasterData&)` - Create packet with instruction

#### Vector Group Info
- `getXLen()`, `getYLen()` - Get vector dimensions
- `getXOrigin()`, `getYOrigin()` - Get vector group origin

### Statistics
| Stat | Description |
|------|-------------|
| `m_revec_stalls` | Stalls due to REVEC synchronization |
| `m_backpressure_stalls` | Stalls due to mesh backpressure |
| `m_no_mesh_stalls` | Stalls waiting for mesh input |
| `m_no_pipe_stalls` | Stalls waiting for fetch input |
| `m_cycles_in_vec` | Cycles in vector mode |

---

## VecInstSel Class

**File:** `vec_inst_sel.hh`, `vec_inst_sel.cc`

### Overview
`VecInstSel` handles storage and selection of received instructions/PCs from the mesh network. It manages the micro-op decomposition when receiving PC-based vector commands.

### Class Hierarchy
```
Named
    └── VecInstSel
```

### Nested Structures

#### MasterData
Information to create local `IODynInst` from mesh communication.

| Member | Type | Description |
|--------|------|-------------|
| `inst` | `IODynInstPtr` | Full instruction (if sent) |
| `isInst` | `bool` | True if instruction, false if PC |
| `pc` | `TheISA::PCState` | Program counter (if PC mode) |
| `recvCnt` | `int` | Receive count for ordering |

### Key Members

| Member | Type | Description |
|--------|------|-------------|
| `m_cpu_p` | `IOCPU*` | CPU pointer |
| `_uopPC` | `TheISA::PCState` | Current micro-op PC |
| `_uopIssueLen` | `int` | Number of micro-ops to fetch |
| `_uopCnt` | `int` | Current micro-op count |
| `_vecCmds` | `std::queue<std::shared_ptr<MasterData>>` | Pending commands |
| `_maxVecCmds` | `int` | Maximum queue size |
| `_lastICacheResp` | `IODynInstPtr` | Last fetched instruction |
| `_pendingICacheReq` | `bool` | Pending I-cache request flag |
| `_stallUntilJumpPC` | `bool` | Stall for control flow resolution |
| `_waitingForTerminator` | `bool` | Waiting for block terminator |

### Core Methods

#### Enqueue/Dequeue
- `enqueueTiming(PacketPtr pkt)` - Enqueue instruction/PC from mesh packet
- `dequeueInst()` - Dequeue next instruction for pipeline
- `getRdy()` - Check if can receive new packet
- `getVal()` - Check if instruction is available

#### PC Generation (Micro-op Decomposition)
- `setPCGen(TheISA::PCState, int cnt)` - Setup PC generator for uop fetch
- `isPCGenActive()` - Check if fetching micro-ops
- `tryReqNextUop()` - Try to fetch next micro-op
- `sendICacheReq(int tid, Addr)` - Send I-cache request for micro-op

#### Response Handling
- `recvIcacheResp(PacketPtr pkt)` - Handle I-cache response
- `doSquash(SquashComm::BaseSquash&, StageIdx)` - Handle branch squash

#### Internal Management
- `processHead()` - Process command at queue head
- `cleanCurIssueBlock()` - Clear current issue block state
- `reset()` - Full reset of state

### Statistics
| Stat | Description |
|------|-------------|
| `MeshQueueSize` | Queue occupancy histogram |
| `NoFetchedInst` | Cycles with block but no instruction |
| `TryFetchAgain` | Failed fetch attempts |
| `StallsOnControl` | Stalls for control resolution |
| `AlreadyInstruction` | Blocked due to pending instruction |
| `DequeueReqs` | Dequeue requests from Vector stage |

---

## MeshHelper Class

**File:** `mesh_helper.hh`, `mesh_helper.cc`

### Overview
Static utility class for parsing CSR values and extracting mesh configuration information. All methods are static.

### Enumerations

#### SensitiveStage
```cpp
typedef enum SensitiveStage {
    EXECUTE = 0,  // Execute stage sensitive
    FETCH,        // Fetch stage sensitive
    NUM_STAGES,
    NONE          // Not sensitive
} SensitiveStage;
```

### Key Static Methods

#### Execute Stage CSR Parsing
- `exeCsrToOutSrcs(uint64_t csrVal, std::vector<Mesh_DS_t>&)` - Get output sources
- `exeCsrToOp(uint64_t csrVal, int opIdx, Mesh_Dir&)` - Get operand direction
- `exeCsrToInSrc(uint64_t csrVal, std::vector<Mesh_Dir>&)` - Get input sources
- `exeCsrToOutDests(uint64_t csrVal, std::vector<Mesh_Dir>&)` - Get output destinations

#### Fetch Stage CSR Parsing
- `fetCsrToInSrc(uint64_t csrVal, Mesh_Dir&)` - Get fetch input source
- `fetCsrToOutDests(uint64_t csrVal, std::vector<Mesh_Dir>&)` - Get fetch outputs

#### Generic CSR Methods
- `csrToInSrcs(uint64_t csr, uint64_t csrVal, std::vector<Mesh_Dir>&)` - Generic input parser
- `csrToOutDests(uint64_t csr, uint64_t csrVal, std::vector<Mesh_Dir>&)` - Generic output parser
- `csrToOutSrcs(uint64_t csr, uint64_t csrVal, std::vector<Mesh_DS_t>&)` - Generic source parser

#### Configuration Queries
- `isBindCSR(int csrIdx)` - Check if CSR is a binding CSR
- `getCSRCodes()` - Get all binding CSR codes
- `isCSRDefault(uint64_t csrVal)` - Check if default (disabled)
- `isVectorMaster(uint64_t csrVal)` - Check if configured as master
- `isVectorSlave(uint64_t csrVal)` - Check if configured as slave
- `isDecoupledAccess(RegVal csrVal)` - Check decoupled access mode
- `hasForwardingPath(RegVal csrVal)` - Check for forwarding path

#### Vector Group Information
- `getXOrigin(RegVal)`, `getYOrigin(RegVal)` - Get group origin
- `getXLen(uint64_t, uint64_t)`, `getYLen(uint64_t, uint64_t)` - Get dimensions
- `doVecLoad(uint64_t csrVal)` - Should perform vector load?

#### Stage Conversion
- `csrToStage(uint64_t csr)` - CSR index to stage enum
- `stageToCsr(SensitiveStage stage)` - Stage enum to CSR index

#### Prefetch
- `numPrefetchRegions(RegVal)` - Get prefetch region count
- `prefetchRegionSize(RegVal)` - Get prefetch region size

---

## MeshPacketData Class

**File:** `mesh_ports.hh`

### Overview
Data container for packets in mesh network queues. Implements bubble and report interfaces for Minor-style hardware queues.

### Members

| Member | Type | Description |
|--------|------|-------------|
| `pkt` | `PacketPtr` | The mesh packet |

### Methods
- `bubbleFill()` - Mark as bubble (empty)
- `isBubble()` - Check if bubble
- `reportData(std::ostream&)` - Report interface (empty)
- `getPacket()` - Get stored packet

---

## ToMeshPort Class

**File:** `mesh_ports.hh`, `mesh_ports.cc`

### Overview
Master port for sending data/instructions to neighboring cores over the mesh network.

### Class Hierarchy
```
MasterPort
    └── ToMeshPort
```

### Members

| Member | Type | Description |
|--------|------|-------------|
| `cpu` | `IOCPU*` | CPU reference |
| `idx` | `int` | Port index |
| `active` | `SensitiveStage` | Active pipeline stage |
| `vec` | `Vector*` | Driving Vector stage |

### Key Methods
- `setDriver(Vector*)` - Set the driving vector stage
- `getPairRdy()` - Get paired slave port ready status
- `setActive(SensitiveStage)` - Set active stage
- `getActive()` - Get active stage
- `isActive()` - Check if active
- `tryUnblockNeighbor()` - Trigger neighbor unblock
- `recvTimingResp(PacketPtr)` - Handle responses (always returns true)
- `recvReqRetry()` - Handle retry (no-op)

---

## FromMeshPort Class

**File:** `mesh_ports.hh`, `mesh_ports.cc`

### Overview
Slave port for receiving data/instructions from neighboring cores over the mesh network.

### Class Hierarchy
```
SlavePort
    └── FromMeshPort
```

### Members

| Member | Type | Description |
|--------|------|-------------|
| `cpu` | `IOCPU*` | CPU reference |
| `idx` | `int` | Port index |
| `recvPkt_d` | `PacketPtr` | Received packet (delayed) |
| `recvEvent` | `EventFunctionWrapper` | Receive processing event |
| `wakeupCPUEvent` | `EventFunctionWrapper` | CPU wakeup event |
| `active` | `SensitiveStage` | Active pipeline stage |
| `_meshQueue` | `Minor::Queue<MeshPacketData>` | 2-slot receive queue |
| `vec` | `Vector*` | Driving Vector stage |

### Key Methods
- `getPacket()` - Dequeue and return packet
- `getPacketData(PacketPtr)` - Extract uint64 data from packet
- `getRdy()` - Check if ready to receive
- `getPairVal()` - Check if data available
- `pktExists()` - Check queue non-empty
- `tryUnblockCPU()` - Trigger CPU wakeup
- `setupEvents()` - Initialize event wrappers
- `recvTimingReq(PacketPtr)` - Handle incoming requests
- `getAddrRanges()` - Return address ranges (empty list)

---

## CPIStack Class

**File:** `cpi_stack.hh`, `cpi_stack.cc`

### Overview
CPI (Cycles Per Instruction) stack profiler that tracks execution events to understand performance bottlenecks.

### Class Hierarchy
```
Named, Clocked
    └── CPIStack
```

### CPIStackInterface Events
```cpp
enum CPIEvents {
    Issued,             // Instruction issued
    Stallon_Load,       // Stall on load
    Stallon_Frame,      // Stall on frame
    Stallon_Fetch,      // Stall on fetch
    Stallon_INET_Pull,  // Stall on interconnect pull
    Stallon_ROB_Store,  // Stall on ROB store
    Stallon_ROB_Other,  // Stall on ROB other
    Stallon_ExeUnit,    // Stall on execution unit
    Stallon_DepOther,   // Stall on other dependency
    Stallon_DepNone,    // No dependency stall
    Stallon_MemBarrier, // Stall on memory barrier
    Stallon_Remem,      // Stall on remote memory
    Num_Events
};
```

### Members
| Member | Type | Description |
|--------|------|-------------|
| `_stack` | `Stats::Vector2d` | Event counters |
| `_lastCycleRecorded` | `Cycles` | Last recorded cycle |

### Methods
- `setEventThisCycle(CPIEvents event)` - Record event for current cycle
- `regStats()` - Register statistics

---

## Bind Specification

**File:** `bind_spec.hh`

### Overview
Defines the CSR encoding format for mesh port bindings. This header contains all the bit-field definitions for the EXECUTE and FETCH stage CSRs.

### Mesh Direction Enum
```cpp
typedef enum Mesh_Dir {
    RIGHT = 0,
    DOWN = 1,
    LEFT = 2,
    UP = 3,
    NUM_DIR
} Mesh_Dir;
```

### Mesh Output Source Enum
```cpp
typedef enum Mesh_Out_Src {
    RD = 0,   // Destination register
    RS1 = 1,  // Source register 1
    RS2 = 2,  // Source register 2
    INST = 3  // Instruction bits
} Mesh_Out_Src;
```

### Execute Stage Encoding (20 bits)

| Field | Bits | Description |
|-------|------|-------------|
| UP output | 0-1 | Output source for UP direction |
| RIGHT output | 2-3 | Output source for RIGHT direction |
| DOWN output | 4-5 | Output source for DOWN direction |
| LEFT output | 6-7 | Output source for LEFT direction |
| RS1 input | 8-10 | Input direction for RS1 |
| RS2 input | 11-13 | Input direction for RS2 |

### Fetch Stage Encoding (42 bits)

| Field | Bits | Description |
|-------|------|-------------|
| Input source | 0-2 | Direction for instruction input |
| RIGHT output | 3 | Send to RIGHT neighbor |
| DOWN output | 4 | Send to DOWN neighbor |
| LEFT output | 5 | Send to LEFT neighbor |
| UP output | 6 | Send to UP neighbor |
| X length | 7-11 | Vector X dimension |
| Y length | 12-16 | Vector Y dimension |
| DAE mode | 17 | Decoupled access enable |
| X origin | 18-24 | Vector group X origin |
| Y origin | 25-31 | Vector group Y origin |

### Prefetch Encoding

| Field | Bits | Description |
|-------|------|-------------|
| Num regions | 0-9 | Number of prefetch regions |
| Region size | 10-19 | Size of each region |
