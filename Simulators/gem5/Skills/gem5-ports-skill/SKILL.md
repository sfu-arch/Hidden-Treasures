---
name: gem5-ports-tutorial
description: >
  Expert guide for teaching and explaining gem5 Ports (RequestPort, ResponsePort),
  Packets, timing/atomic/functional access modes, and the InspectorGadget SimObject tutorial.
  Use this skill whenever a student or developer asks about gem5 ports, memory interfaces,
  packet communication in gem5, how SimObjects connect to memory, RecvTimingReq/RecvTimingResp,
  retry protocols, ClockedObject, or the InspectorGadget tutorial steps. Also use when
  explaining port-to-port connection, blocked packets, sendRetry patterns, or any of the
  4-step InspectorGadget implementation (buffering, inspection latency, throughput, pipelining).
  Trigger this skill even for partial or conversational questions like "why does my port
  return false?", "what is sendRetryReq?", or "how do I add a port to my SimObject?"
---

# gem5 Ports Tutorial Skill

This skill covers **Module 04 – Modeling Memory Objects in gem5: Ports**, a bootcamp tutorial
building on SimObjects (01), Debugging (02), and Event-Driven Simulation (03).

---

## Core Concepts

### What Are Ports?

Ports are gem5's **main interface to the memory system**. A `SimObject` uses ports to send/request data.

| Port Type       | Role                                      |
|-----------------|-------------------------------------------|
| `RequestPort`   | Makes requests, awaits responses          |
| `ResponsePort`  | Awaits requests, sends responses          |

> Key distinction: Both requests **and** responses can carry data. Do not conflate "request/response" with "data direction".

A `RequestPort` can only connect to a `ResponsePort` (and vice versa). Each port connects to exactly one peer.

---

### Packets

`PacketPtr` (= `Packet*`) facilitates communication through ports.

| Field         | Description                                          |
|---------------|------------------------------------------------------|
| `Addr`        | Address of the memory location being accessed        |
| `Data`        | Data payload                                         |
| `MemCmd`      | Kind of packet: `readReq`, `readResp`, `writeReq`, `writeResp`, etc. |
| `RequestorID` | ID of the SimObject that created the request         |

> A `Packet` can **change** from request to response when it arrives at a SimObject that handles it.

Defined in: `gem5/src/mem/packet.hh`

---

### Access Modes

| Mode         | Time advances? | Interleaved? | Use case                                       |
|--------------|----------------|--------------|------------------------------------------------|
| `timing`     | Yes            | Yes          | Realistic simulation (the only realistic mode) |
| `atomic`     | No (requestor moves time) | No | Fast-forwarding                           |
| `functional` | No             | No           | Initialization from files (host → simulator)   |

---

### Port API in C++

```cpp
class RequestPort {
  public:
    bool sendTimingReq(PacketPtr pkt);
    virtual bool recvTimingResp(PacketPtr pkt) = 0;  // pure virtual
    virtual void sendRetryResp();
};

class ResponsePort {
  public:
    bool sendTimingResp(PacketPtr pkt);
    virtual bool recvTimingReq(PacketPtr pkt) = 0;   // pure virtual
    virtual void sendRetryReq();
};
```

`sendTimingReq` calls `peer::recvTimingReq` internally (and similarly for responses).

---

## Timing Protocol Scenarios

### Scenario 1: Everything Goes Smoothly (no retry)

Diagram: `assets/port_ladder_no_retry_drawio.svg`

1. **Requestor** calls `sendTimingReq` → triggers `Responder::recvTimingReq`
2. **Responder** returns **true** (not busy, accepted)
3. Time advances; Responder prepares response
4. **Responder** calls `sendTimingResp` → triggers `Requestor::recvTimingResp`
5. **Requestor** returns **true** → transaction complete

### Scenario 2: Responder Is Busy (request retry)

Diagram: `assets/port_ladder_req_retry_drawio.svg`

1. **Requestor** sends request
2. **Responder** returns **false** (busy) — requestor holds the blocked packet
3. When Responder becomes free, calls `sendReqRetry` → `Requestor::recvReqRetry`
4. **Requestor** resends the blocked packet
5. **Responder** now accepts (returns true)
6. Transaction proceeds to completion

### Scenario 3: Requestor Is Busy (response retry)

Diagram: `assets/port_ladder_resp_retry_drawio.svg`

Similar flow but the Requestor returns **false** for `recvTimingResp`, and Responder later calls `sendRespRetry`.

> ⚠️ **Design Caution**: A Requestor returning false on `recvTimingResp` is unusual and indicates a potential design problem. The Requestor should generally ensure it can accept a response before sending a request.

---

## InspectorGadget: Overview

`InspectorGadget` is a `ClockedObject` placed between the CPU and memory to monitor/inspect all memory traffic.

Diagram: `assets/inspector-gadget_drawio.svg`

It has:
- `cpu_side_port` → `ResponsePort` (receives requests from CPU side)
- `mem_side_port` → `RequestPort` (forwards requests to memory side)

### 4-Step Implementation Plan

| Step | Feature Added                                    |
|------|--------------------------------------------------|
| 1    | Forward traffic with buffering (queuing latency) |
| 2    | Add 1-cycle inspection latency                   |
| 3    | Multiple inspection units + `inspection_latency` param |
| 4    | Pipelining of inspections                        |

---

## Step 1: Buffering Traffic

### ClockedObject Basics

```cpp
class ClockedObject : public SimObject, public Clocked {
  public:
    ClockedObject(const ClockedObjectParams &p);
};
```

Key helper functions:
- `clockEdge(Cycles n)` → time of nth clock edge into the future
- `nextCycle()` → equivalent to `clockEdge(Cycles(1))`

### Python SimObject Declaration (`InspectorGadget.py`)

```python
from m5.objects.ClockedObject import ClockedObject
from m5.params import *

class InspectorGadget(ClockedObject):
    type = "InspectorGadget"
    cxx_header = "bootcamp/inspector-gadget/inspector_gadget.hh"
    cxx_class = "gem5::InspectorGadget"

    cpu_side_port = ResponsePort("ResponsePort to receive requests from CPU side.")
    mem_side_port = RequestPort("RequestPort to send received requests to memory side.")

    inspection_buffer_entries = Param.Int("Number of entries in the inspection buffer.")
    response_buffer_entries = Param.Int("Number of entries in the response buffer.")
```

### SConscript

```python
Import("*")
SimObject("InspectorGadget.py", sim_objects=["InspectorGadget"])
Source("inspector_gadget.cc")
DebugFlag("InspectorGadget")
```

### Extending Ports in C++

Since `recvTimingReq` and `recvTimingResp` are **pure virtual**, you must subclass the ports.

**CPUSidePort** (extends `ResponsePort`):
```cpp
class CPUSidePort: public ResponsePort {
  private:
    InspectorGadget* owner;
    bool needToSendRetry;
    PacketPtr blockedPacket;
  public:
    CPUSidePort(InspectorGadget* owner, const std::string& name):
        ResponsePort(name), owner(owner), needToSendRetry(false), blockedPacket(nullptr) {}
    bool needRetry() const { return needToSendRetry; }
    bool blocked() const { return blockedPacket != nullptr; }
    void sendPacket(PacketPtr pkt);
    virtual AddrRangeList getAddrRanges() const override;
    virtual bool recvTimingReq(PacketPtr pkt) override;
    virtual Tick recvAtomic(PacketPtr pkt) override;
    virtual void recvFunctional(PacketPtr pkt) override;
    virtual void recvRespRetry() override;
};
```

**MemSidePort** (extends `RequestPort`):
```cpp
class MemSidePort: public RequestPort {
  private:
    InspectorGadget* owner;
    PacketPtr blockedPacket;
  public:
    MemSidePort(InspectorGadget* owner, const std::string& name):
        RequestPort(name), owner(owner), blockedPacket(nullptr) {}
    bool blocked() const { return blockedPacket != nullptr; }
    void sendPacket(PacketPtr pkt);
    virtual bool recvTimingResp(PacketPtr pkt) override;
    virtual void recvReqRetry() override;
};
```

### Key Design Patterns

**`getPort` override** — gem5 requires implementing `getPort` to return the correct port by name:
```cpp
Port& InspectorGadget::getPort(const std::string& if_name, PortID idx) {
    if (if_name == "cpu_side_port") return cpuSidePort;
    if (if_name == "mem_side_port") return memSidePort;
    return ClockedObject::getPort(if_name, idx);
}
```

**Blocked packet pattern** — when a port can't forward a packet, it stores it and waits for retry:
```cpp
void CPUSidePort::sendPacket(PacketPtr pkt) {
    panic_if(blocked(), "Should never try to send if blocked!");
    if (!owner->sendResponse(pkt)) {
        blockedPacket = pkt;
    }
}
void CPUSidePort::recvRespRetry() {
    assert(blocked());
    PacketPtr pkt = blockedPacket;
    blockedPacket = nullptr;
    sendPacket(pkt);
}
```

**TimedQueue** — used for `inspectionBuffer` and `outputBuffer` to respect timing:
```cpp
// Push with a ready time; front() only returns packets ready at curTick()
inspectionBuffer.push(pkt, curTick());
outputBuffer.push(pkt, clockEdge(Cycles(1)));
```

---

## Step 2: Inspection Latency

Add `inspection_latency` parameter and schedule `nextInspectionEvent` to fire 1 cycle after a packet arrives in `inspectionBuffer`.

Key pattern: `processNextInspectionEvent` pops from `inspectionBuffer`, calls `inspectRequest(pkt)`, and pushes to `outputBuffer` with timestamp `clockEdge(Cycles(1))`.

---

## Step 3: Multiple Inspection Units + Configurable Latency

### New Parameters

```python
insp_window = Param.Int("Number of entries in front of inspectionBuffer to try to inspect every cycle.")
num_insp_units = Param.Int("Number of inspection units.")
insp_tot_latency = Param.Cycles("Latency to complete one inspection.")
```

### C++ Additions

```cpp
int inspectionWindow;
int numInspectionUnits;
Cycles totalInspectionLatency;
std::vector<Tick> inspectionUnitAvailableTimes;
```

### scheduleNextInspectionEvent with Unit Availability

```cpp
void InspectorGadget::scheduleNextInspectionEvent(Tick when) {
    bool have_packet = !inspectionBuffer.empty();
    bool have_entry = outputBuffer.size() < outputBufferEntries;
    if (have_packet && have_entry && !nextInspectionEvent.scheduled()) {
        Tick first_avail = *std::min_element(
            inspectionUnitAvailableTimes.begin(),
            inspectionUnitAvailableTimes.end());
        Tick schedule_time = align(std::max({when,
            inspectionBuffer.firstReadyTime(), first_avail}));
        schedule(nextInspectionEvent, schedule_time);
    }
}
```

### processNextInspectionEvent Loop

```cpp
void InspectorGadget::processNextInspectionEvent() {
    int insp_window_left = inspectionWindow;
    for (int i = 0; i < numInspectionUnits; i++) {
        if (inspectionUnitAvailableTimes[i] > curTick()) continue; // unit busy
        if (inspectionBuffer.empty()) break;
        if (outputBuffer.size() >= outputBufferEntries) break;
        PacketPtr pkt = inspectionBuffer.front();
        inspectRequest(pkt);
        outputBuffer.push(pkt, clockEdge(totalInspectionLatency));
        inspectionBuffer.pop();
        inspectionUnitAvailableTimes[i] = clockEdge(totalInspectionLatency);
        if (--insp_window_left == 0) break;
    }
    // Ensure all unit times are >= nextCycle (even idle units)
    for (int i = 0; i < numInspectionUnits; i++)
        inspectionUnitAvailableTimes[i] = std::max(inspectionUnitAvailableTimes[i], nextCycle());
    scheduleNextReqSendEvent(nextCycle());
    scheduleNextReqRetryEvent(nextCycle());
    scheduleNextInspectionEvent(nextCycle());
}
```

> **Why update idle unit times?** To prevent scheduling inspection events at the current tick when rescheduling immediately — idle units would otherwise allow scheduling at `curTick()`.

---

## Step 4: Pipelining

Extend the design to pipeline inspections so multiple requests can be in-flight through inspection units simultaneously, tracked via `inspectionUnitAvailableTimes` already in place from Step 3.

---

## Common Issues & Debugging Tips

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Segfault on port send | `blockedPacket` not null-checked | Always check `blocked()` before sending |
| Simulation hangs | Retry never sent | Ensure `sendReqRetry`/`sendRespRetry` is called when unblocked |
| Wrong schedule time | Not using `align()` | Always wrap `Tick` computations with `align()` |
| Port not found | `getPort()` not overridden | Implement `getPort` in your SimObject |
| Build error: cannot instantiate Port | Forgot to implement pure virtual | Extend Port class and override `recvTimingReq`/`recvTimingResp` |

---

## File Locations Reference

| Purpose | Path |
|---------|------|
| Port declarations | `gem5/src/mem/port.hh` |
| Packet class | `gem5/src/mem/packet.hh` |
| ClockedObject | `gem5/src/sim/clocked_object.hh` |
| Python params | `gem5/src/python/m5/params.py` |
| Timing protocol | `gem5/src/mem/protocol/timing.hh` |
| Tutorial code (step-by-step) | `04-ports/step-{1,2,3,4}/` |

---

## Teaching Tips

When explaining port communication to students:
1. Start with the **happy path** (Scenario 1) using the ladder diagram
2. Introduce the **retry protocol** only after the happy path is clear
3. Emphasize: `sendTimingReq` returns immediately; the **response arrives in a future event**
4. The `blocked packet` pattern is a key gem5 idiom — make sure students understand why it's needed
5. Use the `InspectorGadget` progression (Steps 1–4) as a concrete running example of increasing complexity
