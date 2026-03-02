# gem5 Interface and Components Documentation
## Port, Packet, and sendTimingReq

This document focuses on how the custom mesh networking code interfaces with gem5's memory system, specifically covering Ports, Packets, and the timing request protocol.

---

## Table of Contents
1. [gem5 Memory System Overview](#gem5-memory-system-overview)
2. [Port Architecture](#port-architecture)
3. [Packet Structure](#packet-structure)
4. [sendTimingReq Protocol](#sendtimingreq-protocol)
5. [Mesh Network Port Implementation](#mesh-network-port-implementation)
6. [Data Flow Examples](#data-flow-examples)
7. [Credit-Based Flow Control](#credit-based-flow-control)

---

## gem5 Memory System Overview

The gem5 simulator uses a port-based communication model for all memory transactions. In this mesh implementation, we repurpose the memory port infrastructure to handle instruction and data communication between cores in a 2D mesh network.

### Key Concepts
- **Ports**: Connection points between objects
- **Packets**: Container for requests/responses with data
- **Timing Mode**: Cycle-accurate simulation with proper stall handling
- **Master/Slave**: Initiator/responder relationship

---

## Port Architecture

### Port Hierarchy in gem5

```
Port (base class)
├── BaseMasterPort
│   └── MasterPort
│       └── ToMeshPort (custom)
└── BaseSlavePort
    └── SlavePort
        └── FromMeshPort (custom)
```

### MasterPort Key Methods

| Method | Description |
|--------|-------------|
| `sendTimingReq(PacketPtr)` | Send a timing request to slave |
| `sendFunctional(PacketPtr)` | Send a functional (debug) request |
| `recvTimingResp(PacketPtr)` | **Virtual** - Handle timing response |
| `recvReqRetry()` | **Virtual** - Handle retry notification |

### SlavePort Key Methods

| Method | Description |
|--------|-------------|
| `sendTimingResp(PacketPtr)` | Send a timing response to master |
| `sendRetryReq()` | Send retry notification to master |
| `recvTimingReq(PacketPtr)` | **Virtual** - Handle timing request |
| `recvRespRetry()` | **Virtual** - Handle response retry |
| `getAddrRanges()` | **Virtual** - Return handled address ranges |

### Port Binding

Ports must be bound during simulation initialization:

```python
# From Harness.py
from_cpu = VectorSlavePort("From CPU, receives requests")
to_cpu = VectorMasterPort("To CPU, sends requests")
```

Port binding in C++:
```cpp
Port&
Harness::getPort(const std::string &if_name, PortID idx) {
    if (if_name == "to_cpu" && idx < masterPorts.size()) {
        return masterPorts[idx];
    } else if (if_name == "from_cpu" && idx < slavePorts.size()) {
        return slavePorts[idx];
    } else {
        return ClockedObject::getPort(if_name, idx);
    }
}
```

---

## Packet Structure

### PacketPtr

`PacketPtr` is a `Packet*` pointer. Packets are the fundamental unit of communication in gem5's memory system.

### Key Packet Members

| Member | Type | Description |
|--------|------|-------------|
| `cmd` | `MemCmd` | Command type (Read, Write, etc.) |
| `req` | `RequestPtr` | Associated request object |
| `data` | `uint8_t*` | Data payload |
| `senderState` | `SenderState*` | Custom state attached to packet |

### Creating Packets in the Mesh

#### Data Payload Packet
```cpp
PacketPtr Vector::createMeshPacket(RegVal payload) {
    int size = sizeof(payload);
    uint8_t *data = new uint8_t[size];
    
    // Convert RegVal to bytes (little endian)
    for (int i = 0; i < size; i++) {
        data[i] = (uint8_t)(payload >> (i * 8));
    }
    
    Addr addr = 0;
    RequestPtr req = std::make_shared<Request>(addr, size, 0, 0);
    PacketPtr new_pkt = new Packet(req, MemCmd::WritebackDirty, size);
    new_pkt->dataDynamic(data);  // Packet owns the data now
    
    return new_pkt;
}
```

#### Instruction Metadata Packet
```cpp
PacketPtr Vector::createMeshPacket(const VecInstSel::MasterData& data) {
    auto copy = std::make_shared<VecInstSel::MasterData>(data);
    
    Addr addr = 0;
    int size = 0;  // No actual data payload
    RequestPtr req = std::make_shared<Request>(addr, size, 0, 0);
    PacketPtr new_pkt = new Packet(req, MemCmd::WritebackDirty, size);
    
    // Attach instruction info as SenderState
    new_pkt->pushSenderState(new Vector::SenderState(copy));
    
    return new_pkt;
}
```

### Extracting Data from Packets

```cpp
uint64_t FromMeshPort::getPacketData(PacketPtr pkt) {
    return pkt->getUintX(LittleEndianByteOrder);
}
```

### SenderState Pattern

The `SenderState` pattern allows attaching custom data to packets:

```cpp
// Define custom state
struct SenderState : public Packet::SenderState {
    std::shared_ptr<VecInstSel::MasterData> master_data;
    SenderState(std::shared_ptr<VecInstSel::MasterData> _master_data)
        : master_data(_master_data) { }
};

// Attach to packet
pkt->pushSenderState(new Vector::SenderState(data));

// Retrieve from packet
Vector::SenderState* ss = safe_cast<Vector::SenderState*>(pkt->popSenderState());
auto msg = ss->master_data;
```

---

## sendTimingReq Protocol

### Overview

`sendTimingReq()` is the primary method for sending cycle-accurate requests in gem5. It returns a boolean indicating success.

### Return Values

| Return | Meaning |
|--------|---------|
| `true` | Request accepted, will receive response later |
| `false` | Request rejected, must retry later |

### Protocol Flow

```
Master                                  Slave
  │                                       │
  │──── sendTimingReq(pkt) ──────────────>│
  │                                       │
  │<─── true/false ───────────────────────│
  │                                       │
  │     (if false, wait for retry)        │
  │                                       │
  │<─── recvReqRetry() ───────────────────│
  │                                       │
  │──── sendTimingReq(pkt) ──────────────>│
  │                                       │
  │     (response comes later)            │
  │                                       │
  │<─── recvTimingResp(pkt) ──────────────│
  │                                       │
```

### Implementation in ToMeshPort

```cpp
void ToMeshPort::sendPacket(PacketPtr pkt) {
    panic_if(blockedPacket != nullptr, 
             "Should never try to send if blocked!");
    
    // sendTimingReq is inherited from MasterPort
    if (!sendTimingReq(pkt)) {
        // Failed - store for retry
        blockedPacket = pkt;
    }
}
```

### Implementation in FromMeshPort (Receiving Side)

```cpp
bool FromMeshPort::recvTimingReq(PacketPtr pkt) {
    // Forward to Vector stage for processing
    if (vec->enqueueMeshPkt(pkt)) {
        return true;   // Accepted
    }
    return false;      // Rejected - sender must retry
}
```

### Retry Mechanism

When `sendTimingReq()` returns false, the master must wait for `recvReqRetry()`:

```cpp
void Harness::CPUSideMasterPort::recvReqRetry() {
    assert(blockedPacket != nullptr);
    
    PacketPtr pkt = blockedPacket;
    blockedPacket = nullptr;
    
    // Try to resend
    sendPacket(pkt);
}
```

---

## Mesh Network Port Implementation

### ToMeshPort (Sending to Neighbors)

**Purpose**: Send instructions/data to neighboring cores in the mesh.

```cpp
ToMeshPort::ToMeshPort(IOCPU *_cpu, int idx)
    : MasterPort(_cpu->name() + ".mesh_out_port" + csprintf("[%d]", idx), _cpu),
      cpu(_cpu), idx(idx), active(NONE), vec(nullptr)
{ }
```

**Key Features**:
- One port per mesh direction (UP, DOWN, LEFT, RIGHT)
- `active` field indicates which pipeline stage drives this port
- `getPairRdy()` queries the connected slave's readiness

```cpp
bool ToMeshPort::getPairRdy() {
    BaseSlavePort *slavePort = &(getSlavePort());
    if (FromMeshPort *slaveMeshPort = dynamic_cast<FromMeshPort*>(slavePort)) {
        return slaveMeshPort->getRdy();
    }
    // Edge harness always ready
    return true;
}
```

### FromMeshPort (Receiving from Neighbors)

**Purpose**: Receive instructions/data from neighboring cores.

```cpp
FromMeshPort::FromMeshPort(IOCPU *_cpu, int idx)
    : SlavePort(_cpu->name() + ".mesh_in_port" + csprintf("[%d]", idx), _cpu),
      cpu(_cpu), idx(idx), recvPkt_d(nullptr),
      recvEvent([this] { process(); }, name()),
      wakeupCPUEvent([this] { tryUnblockCPU(); }, name()),
      active(NONE),
      _meshQueue(name(), "pkt", MESH_QUEUE_SLOTS),  // 2-slot queue
      vec(nullptr)
{ }
```

**Key Features**:
- 2-slot queue to handle pipeline stalls
- Event-based processing for proper timing
- `getRdy()` checks if Vector stage can accept packets

```cpp
bool FromMeshPort::getRdy() {
    if (!vec) return false;
    return vec->canRecvMeshPkt();
}
```

### I-Cache Port Usage

The Vector stage can also send requests to the instruction cache when operating in PC-forwarding mode:

```cpp
void VecInstSel::sendICacheReq(int tid, Addr instAddr) {
    int fetchSize = sizeof(RiscvISA::MachInst);
    
    RequestPtr req = std::make_shared<Request>(
        tid, instAddr, fetchSize,
        Request::INST_FETCH,
        m_cpu_p->instMasterId(),
        instAddr,
        m_cpu_p->tcBase(tid)->contextId()
    );
    
    // Translate address
    Fault fault = m_cpu_p->itb->translateAtomic(req, 
        m_cpu_p->tcBase(tid), BaseTLB::Execute);
    
    // Create and send packet
    PacketPtr inst_pkt = new Packet(req, MemCmd::ReadReq);
    inst_pkt->dataDynamic(new uint8_t[fetchSize]);
    
    if (!m_cpu_p->getInstPort().sendTimingReq(inst_pkt)) {
        // Handle failure - save for retry
        _pendingFailedReq = true;
        _failedReqVirtAddr = instAddr;
        delete inst_pkt;
    } else {
        _pendingICacheReq = true;
        _pendingICacheReqAddr = inst_pkt->getAddr();
    }
}
```

---

## Data Flow Examples

### Example 1: Master Sending Instruction to Slave

```
1. Vector::forwardInstruction() is called
   
2. Create packet with instruction metadata:
   PacketPtr new_pkt = createMeshPacket(forwardInst);
   
3. Send via appropriate mesh port:
   getMeshMasterPorts()[dir].sendTimingReq(new_pkt);
   
4. Slave's FromMeshPort::recvTimingReq() called:
   - Checks if vec->canRecvMeshPkt()
   - If true, enqueues packet
   - Returns true/false

5. If accepted, packet enters _meshQueue in VecInstSel

6. Vector::pullMeshInstruction() dequeues when ready
```

### Example 2: Stall and Retry Flow

```
Cycle N:
  Master: sendTimingReq(pkt) → false (slave buffer full)
  Master: Store pkt in blockedPacket
  
Cycle N+1:
  Slave: Processes queued packet, frees space
  Slave: Calls master's recvReqRetry()
  
Cycle N+1 (later in cycle):
  Master: recvReqRetry() called
  Master: Retrieves blockedPacket
  Master: sendTimingReq(pkt) → true
```

### Example 3: Vector Group Communication

```
           ┌─────────────────────────────────────┐
           │          Vector Group               │
           │                                     │
    ┌──────┴──────┐    ┌──────────┐    ┌────────┴────┐
    │   Master    │───>│  Slave   │───>│   Slave     │
    │ (Root Send) │    │ (Recv)   │    │   (Recv)    │
    │             │    │          │    │             │
    │ fetchInst() │    │ pullMesh │    │  pullMesh   │
    │ forward()   │    │ pushPipe │    │  pushPipe   │
    └─────────────┘    └──────────┘    └─────────────┘
         │                  │                │
         ▼                  ▼                ▼
    [Local Decode]    [Local Decode]   [Local Decode]
```

---

## Credit-Based Flow Control

The mesh network uses credits to prevent buffer overflow and manage backpressure.

### Credit Management in Vector Stage

```cpp
void Vector::stealCredits() {
    if (m_is_sequential) {
        int remainingCred = m_input_queue_size - outputCredit();
        outputCredit() = -1 * remainingCred;
        _stolenCredits = m_input_queue_size;
    } else {
        _stolenCredits = m_max_num_credits;
        auto& pipeline = m_cpu_p->getPipeline();
        pipeline.setPrevStageUnemployed(m_stage_idx, true);
    }
}

void Vector::restoreCredits() {
    if (m_is_sequential) {
        outputCredit() += _stolenCredits;
    } else {
        if (_stolenCredits > 0) {
            m_outgoing_credit_wire->to_prev_stage(m_next_stage_credit_idx) 
                += m_num_credits;
            m_num_credits = 0;
            
            auto& pipeline = m_cpu_p->getPipeline();
            pipeline.setPrevStageUnemployed(m_stage_idx, false);
        }
    }
    _stolenCredits = 0;
}
```

### Stall Detection

```cpp
bool Vector::canPushMesh() {
    auto pipeSrc = getOutPipeSource();
    auto meshSrc = getOutMeshSource();
    
    bool coupledStalls = (pipeSrc == meshSrc);
    
    bool outMeshStall = (meshSrc != None) && 
        (isOutMeshStalled() ||
         (meshSrc == Pipeline && isInPipeStalled()) ||
         (meshSrc == Mesh     && isInMeshStalled()) ||
         (coupledStalls       && isOutPipeStalled()));
    
    return !outMeshStall;
}
```

---

## Configuration via CSR

The mesh network is configured through Control and Status Registers (CSRs).

### CSR Values

| CSR | Stage | Purpose |
|-----|-------|---------|
| `MISCREG_EXE` | Execute | Configure data operand routing |
| `MISCREG_FETCH` | Fetch | Configure instruction routing |

### Configuration Flow

```cpp
void Vector::setupConfig(int csrId, RegVal csrVal) {
    if (csrId != RiscvISA::MISCREG_FETCH) return;
    
    _curCsrVal = csrVal;
    resetActive();
    
    // Configure output ports
    std::vector<Mesh_Dir> outDirs;
    MeshHelper::csrToOutDests(csrId, csrVal, outDirs);
    for (auto dir : outDirs) {
        getMeshMasterPorts()[dir].setActive(_stage);
        _numOutPortsActive++;
    }
    
    // Configure input ports
    std::vector<Mesh_Dir> inDirs;
    MeshHelper::csrToInSrcs(csrId, csrVal, inDirs);
    for (auto dir : inDirs) {
        getMeshSlavePorts()[dir].setActive(_stage);
        _numInPortsActive++;
    }
    
    // Handle credit management
    if (!getConfigured()) {
        restoreCredits();
    }
}
```

---

## Build System Integration

### SConscript

```python
Import('*')

# Python config files
SimObject('Harness.py')

# C++ source files
Source('harness.cc')
Source('mesh_helper.cc')
Source('mesh_ports.cc')
Source('vec_inst_sel.cc')
Source('vector.cc')
Source('cpi_stack.cc')

# Debug flag
DebugFlag('Harness', "Custom test harness")
```

### Python SimObject (Harness.py)

```python
from m5.params import *
from m5.proxy import *
from m5.objects.ClockedObject import ClockedObject

class Harness(ClockedObject):
    type = 'Harness'
    cxx_header = "custom/harness.hh"
    
    from_cpu = VectorSlavePort("From CPU, receives requests")
    to_cpu = VectorMasterPort("To CPU, sends requests")
    system = Param.System(Parent.any, "The system this object is part of")
```

---

## Debugging

### Debug Flags

| Flag | Description |
|------|-------------|
| `Harness` | Test harness operations |
| `Mesh` | Mesh network operations (in mesh_ports.cc, vector.cc) |

### Example Debug Usage

```cpp
DPRINTF(Mesh, "Forward to mesh net %s\n", inst->toString(true));
DPRINTF(Mesh, "Got request for addr %#x\n", pkt->getAddr());
```

### Enable Debug Output

```bash
./gem5.opt --debug-flags=Mesh config.py
```
