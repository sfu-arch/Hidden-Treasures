# gem5 v20+ Migration Guide
## Migrating gem5-mesh from Pre-v20 to Modern gem5

This document provides a comprehensive guide for migrating the gem5-mesh codebase from pre-v20 gem5 (which uses master/slave terminology) to gem5 v20 and later versions (which use requestor/responder terminology).

---

## Table of Contents
1. [Overview of Changes](#overview-of-changes)
2. [Port API Changes](#port-api-changes)
3. [File-by-File Migration Guide](#file-by-file-migration-guide)
4. [Python SimObject Changes](#python-simobject-changes)
5. [Request/RequestPtr Changes](#requestrequestptr-changes)
6. [MemObject Removal](#memobject-removal)
7. [Statistics API Changes](#statistics-api-changes)
8. [SLICC Protocol Changes](#slicc-protocol-changes)
9. [Migration Scripts](#migration-scripts)
10. [Testing and Validation](#testing-and-validation)

---

## Overview of Changes

gem5 v20 (released 2020) introduced significant terminology changes to remove master/slave naming. gem5 v21+ continued refactoring the port infrastructure.

### Terminology Mapping

| Old Term (Pre-v20) | New Term (v20+) |
|--------------------|-----------------|
| `MasterPort` | `RequestorPort` |
| `SlavePort` | `ResponsePort` |
| `BaseMasterPort` | `Port` (unified base) |
| `BaseSlavePort` | `Port` (unified base) |
| `MasterID` | `RequestorID` |
| `masterId()` | `requestorId()` |
| `isMaster()` | `isRequestor()` |
| `isSlave()` | `isResponder()` |

### Key Header Changes

| Old Header | New Header |
|------------|------------|
| `mem/port.hh` | `mem/port.hh` (same, but classes renamed) |
| `mem/mem_object.hh` | Removed (use `sim/sim_object.hh` or `sim/clocked_object.hh`) |

---

## Port API Changes

### Class Inheritance Changes

**Old (Pre-v20):**
```cpp
#include "mem/port.hh"

class ToMeshPort : public MasterPort {
    // ...
};

class FromMeshPort : public SlavePort {
    // ...
};
```

**New (v20+):**
```cpp
#include "mem/port.hh"

class ToMeshPort : public RequestorPort {
    // ...
};

class FromMeshPort : public ResponsePort {
    // ...
};
```

### Method Renames

| Old Method | New Method |
|------------|------------|
| `getSlavePort()` | `getPeer()` (returns `Port&`) |
| `getMasterPort()` | `getPeer()` (returns `Port&`) |
| `sendTimingReq()` | `sendTimingReq()` (unchanged) |
| `sendTimingResp()` | `sendTimingResp()` (unchanged) |
| `recvTimingReq()` | `recvTimingReq()` (unchanged) |
| `recvTimingResp()` | `recvTimingResp()` (unchanged) |

### Peer Access Pattern Change

**Old (Pre-v20):**
```cpp
bool ToMeshPort::getPairRdy() {
    BaseSlavePort *slavePort = &(getSlavePort());
    if (FromMeshPort *meshPort = dynamic_cast<FromMeshPort*>(slavePort)) {
        return meshPort->getRdy();
    }
    return true;
}
```

**New (v20+):**
```cpp
bool ToMeshPort::getPairRdy() {
    Port &peer = getPeer();
    if (FromMeshPort *meshPort = dynamic_cast<FromMeshPort*>(&peer)) {
        return meshPort->getRdy();
    }
    return true;
}
```

---

## File-by-File Migration Guide

### custom/mesh_ports.hh

```diff
 #ifndef __CUSTOM_MESH_PORTS_HH__
 #define __CUSTOM_MESH_PORTS_HH__
 
 #include "mem/port.hh"
 #include "mem/packet.hh"
 #include "sim/clocked_object.hh"
 #include "custom/mesh_helper.hh"
 #include "cpu/minor/buffers.hh"
 
 class IOCPU;
 class Vector;
 
 // ... MeshPacketData unchanged ...
 
-class ToMeshPort : public MasterPort {
+class ToMeshPort : public RequestorPort {
   public:
     ToMeshPort(IOCPU *_cpu, int idx);
     
     // ... rest unchanged ...
     
   protected:
     virtual bool recvTimingResp(PacketPtr pkt);
     virtual void recvReqRetry();
     
     IOCPU *cpu;
     int idx;
     SensitiveStage active;
     Vector *vec;
 };
 
-class FromMeshPort : public SlavePort {
+class FromMeshPort : public ResponsePort {
   public:
     FromMeshPort(IOCPU *_cpu, int idx);
     
     virtual AddrRangeList getAddrRanges() const;
     
     // ... rest unchanged ...
     
   protected:
     virtual bool recvTimingReq(PacketPtr pkt);
     virtual void recvRespRetry();
     virtual Tick recvAtomic(PacketPtr pkt) { panic("recvAtomic unimpl"); };
     virtual void recvFunctional(PacketPtr pkt);
     
     IOCPU* cpu;
     int idx;
     // ... rest unchanged ...
 };
 
 #endif
```

### custom/mesh_ports.cc

```diff
 #include "custom/mesh_ports.hh"
 #include "cpu/io/cpu.hh"
 #include "custom/vector.hh"
 #include "debug/Mesh.hh"
 
 ToMeshPort::ToMeshPort(IOCPU *_cpu, int idx)
-        : MasterPort(
+        : RequestorPort(
           _cpu->name() + ".mesh_out_port" + csprintf("[%d]", idx), _cpu), 
           cpu(_cpu), idx(idx), active(NONE), vec(nullptr)
     { }
 
 bool
 ToMeshPort::getPairRdy() {
-  BaseSlavePort *slavePort = &(getSlavePort());
-  if (FromMeshPort *slaveMeshPort = dynamic_cast<FromMeshPort*>(slavePort)) {
+  Port &peer = getPeer();
+  if (FromMeshPort *slaveMeshPort = dynamic_cast<FromMeshPort*>(&peer)) {
     return slaveMeshPort->getRdy();
   }
   return true;
 }
 
 void
 ToMeshPort::tryUnblockNeighbor() {
-  BaseSlavePort *slavePort = &(getSlavePort());
-  if (FromMeshPort *slaveMeshPort = dynamic_cast<FromMeshPort*>(slavePort)) {
+  Port &peer = getPeer();
+  if (FromMeshPort *slaveMeshPort = dynamic_cast<FromMeshPort*>(&peer)) {
     slaveMeshPort->tryUnblockCPU();
   }
 }
 
 FromMeshPort::FromMeshPort(IOCPU *_cpu, int idx)
-        : SlavePort(
+        : ResponsePort(
           _cpu->name() + ".mesh_in_port" + csprintf("[%d]", idx), _cpu), 
           cpu(_cpu), idx(idx), recvPkt_d(nullptr),
           recvEvent([this] { process(); }, name()), 
           wakeupCPUEvent([this] { tryUnblockCPU(); }, name()), 
           active(NONE), _meshQueue(name(), "pkt", MESH_QUEUE_SLOTS), 
           vec(nullptr)
     {}
```

### custom/harness.hh

```diff
 #ifndef __CUSTOM_HARNESS_HH__
 #define __CUSTOM_HARNESS_HH__
 
 #include <vector>
 
 #include "sim/clocked_object.hh"
-#include "mem/mem_object.hh"
+#include "mem/port.hh"
 #include "params/Harness.hh"
 
 class Harness : public ClockedObject
 {
   private:
-    class CPUSideSlavePort : public SlavePort
+    class CPUSideSlavePort : public ResponsePort
     {
       private:
         int id;
         Harness *owner;
         bool needRetry;
         PacketPtr blockedPacket;
 
       public:
         CPUSideSlavePort(const std::string& name, int id, Harness *owner) :
-            SlavePort(name, owner), id(id), owner(owner), needRetry(false),
+            ResponsePort(name, owner), id(id), owner(owner), needRetry(false),
             blockedPacket(nullptr)
         { }
         
         // ... methods unchanged ...
     };
 
-    class CPUSideMasterPort : public MasterPort
+    class CPUSideMasterPort : public RequestorPort
     {
       private:
         int id;
         Harness *owner;
         PacketPtr blockedPacket;
 
       public:
         CPUSideMasterPort(const std::string& name, int id, Harness *owner) :
-            MasterPort(name, owner), id(id), owner(owner), blockedPacket(nullptr)
+            RequestorPort(name, owner), id(id), owner(owner), blockedPacket(nullptr)
         { }
         
         // ... methods unchanged ...
     };
     
     // ... rest unchanged ...
 };
 
 #endif
```

### custom/harness.cc

```diff
 #include "custom/harness.hh"
 
 #include "debug/Harness.hh"
 #include "sim/system.hh"
 
 // No significant changes needed in harness.cc
 // The port class changes are in the header
```

---

## Python SimObject Changes

### custom/Harness.py

```diff
 from m5.params import *
 from m5.proxy import *
 from m5.objects.ClockedObject import ClockedObject
 
 class Harness(ClockedObject):
     type = 'Harness'
     cxx_header = "custom/harness.hh"
 
-    from_cpu = VectorSlavePort("From CPU, receives requests")
-    to_cpu = VectorMasterPort("To CPU, sends requests")
+    from_cpu = VectorResponsePort("From CPU, receives requests")
+    to_cpu = VectorRequestorPort("To CPU, sends requests")
 
     system = Param.System(Parent.any, "The system this object is part of")
```

### Mesh Configuration Files

Any Python configuration scripts using port names need updating:

```diff
 # Old
-cpu.icache_port = cache.slave
-cache.master = bus.slave
+# New
+cpu.icache_port = cache.cpu_side
+cache.mem_side = bus.cpu_side_ports
```

---

## Request/RequestPtr Changes

### MasterID → RequestorID

```diff
-MasterID masterId = pkt->req->masterId();
+RequestorID requestorId = pkt->req->requestorId();
```

### Request Constructor Changes

The Request constructor signature may have changed:

**Old:**
```cpp
RequestPtr req = std::make_shared<Request>(
    tid, instAddr, fetchSize,
    Request::INST_FETCH,
    cpu->instMasterId(),    // MasterID
    instAddr,
    cpu->tcBase(tid)->contextId()
);
```

**New (v21+):**
```cpp
RequestPtr req = std::make_shared<Request>(
    instAddr, fetchSize,
    Request::INST_FETCH,
    cpu->instRequestorId(),  // RequestorID
    instAddr,
    cpu->tcBase(tid)->contextId()
);
```

### Files Requiring Request Changes

- `custom/vec_inst_sel.cc` - `sendICacheReq()` method
- `custom/vector.cc` - `createMeshPacket()` method

---

## MemObject Removal

`MemObject` was removed in gem5 v20. Objects should inherit from:
- `SimObject` - Basic simulation object
- `ClockedObject` - Object with clock domain

### Changes Required

```diff
-#include "mem/mem_object.hh"
+#include "sim/clocked_object.hh"
+// or
+#include "sim/sim_object.hh"
```

The `Harness` class already inherits from `ClockedObject`, so no class hierarchy changes are needed there.

---

## Statistics API Changes

gem5 v21+ changed the statistics registration API:

### Old Style (Pre-v21)

```cpp
void regStats() override {
    m_revec_stalls
        .name(name() + ".revec_stalls")
        .desc("number of stalls due to revec")
    ;
}
```

### New Style (v21+)

```cpp
void regStats() override {
    ClockedObject::regStats();  // or parent class
    
    m_revec_stalls
        .name(name() + ".revec_stalls")
        .desc("number of stalls due to revec")
    ;
}

// Stats may need to be declared differently
struct VectorStats : public statistics::Group {
    VectorStats(Vector *parent);
    
    statistics::Scalar revec_stalls;
    statistics::Scalar backpressure_stalls;
    // ...
} stats;
```

### Files Requiring Stats Changes

- `custom/vector.cc` - `regStats()` method
- `custom/vec_inst_sel.cc` - `regStats()` method
- `custom/cpi_stack.cc` - `regStats()` method

---

## SLICC Protocol Changes

### Message Types

SLICC protocols may need updates for message type names:

```diff
-out_msg.Requestor := machineID;
+out_msg.Requestor := machineID;  // Usually unchanged
```

### MachineType Changes

Some machine types may have been renamed in newer Ruby versions.

---

## Migration Scripts

### Automated Sed Script

Create a file `migrate_ports.sh`:

```bash
#!/bin/bash

# Port class renames
find src/custom -name "*.hh" -o -name "*.cc" | xargs sed -i '' \
    -e 's/MasterPort/RequestorPort/g' \
    -e 's/SlavePort/ResponsePort/g' \
    -e 's/BaseMasterPort/Port/g' \
    -e 's/BaseSlavePort/Port/g'

# Method renames
find src/custom -name "*.hh" -o -name "*.cc" | xargs sed -i '' \
    -e 's/getSlavePort()/getPeer()/g' \
    -e 's/getMasterPort()/getPeer()/g' \
    -e 's/masterId()/requestorId()/g' \
    -e 's/instMasterId()/instRequestorId()/g'

# Python port types
find src/custom -name "*.py" | xargs sed -i '' \
    -e 's/VectorSlavePort/VectorResponsePort/g' \
    -e 's/VectorMasterPort/VectorRequestorPort/g' \
    -e 's/SlavePort/ResponsePort/g' \
    -e 's/MasterPort/RequestorPort/g'

# Header includes
find src/custom -name "*.hh" -o -name "*.cc" | xargs sed -i '' \
    -e 's/#include "mem\/mem_object.hh"/#include "sim\/clocked_object.hh"/g'

echo "Migration complete. Manual review required."
```

### Verification Script

Create `verify_migration.sh`:

```bash
#!/bin/bash

echo "Checking for remaining old terminology..."

# Check for old port classes
grep -rn "class.*: public MasterPort" src/custom/
grep -rn "class.*: public SlavePort" src/custom/

# Check for old method calls
grep -rn "getSlavePort" src/custom/
grep -rn "getMasterPort" src/custom/

# Check for old Python port types
grep -rn "VectorSlavePort\|VectorMasterPort" src/custom/

# Check for MemObject
grep -rn "mem_object.hh" src/custom/
grep -rn ": public MemObject" src/custom/

echo "Verification complete."
```

---

## Testing and Validation

### Build Verification

```bash
# Clean build
scons --clean build/RISCV/gem5.opt

# Rebuild
scons build/RISCV/gem5.opt -j$(nproc) 2>&1 | tee build.log

# Check for errors
grep -i "error:" build.log
```

### Runtime Verification

```bash
# Run with debug flags
./build/RISCV/gem5.opt \
    --debug-flags=Mesh,Harness \
    configs/example/your_config.py

# Check for port binding issues
./build/RISCV/gem5.opt \
    --debug-flags=Port \
    configs/example/your_config.py
```

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `MasterPort not found` | Class renamed | Change to `RequestorPort` |
| `getSlavePort undefined` | Method removed | Use `getPeer()` |
| `mem_object.hh not found` | Header removed | Use `clocked_object.hh` |
| `VectorSlavePort unknown` | Python type renamed | Use `VectorResponsePort` |

---

## Compatibility Wrapper (Optional)

For gradual migration, you can create a compatibility header:

**compat/port_compat.hh:**
```cpp
#ifndef __COMPAT_PORT_COMPAT_HH__
#define __COMPAT_PORT_COMPAT_HH__

#include "base/compiler.hh"

// Check gem5 version
#if GEM5_VERSION >= 0x140000  // v20+

#include "mem/port.hh"

// Aliases for backward compatibility
using MasterPort = RequestorPort;
using SlavePort = ResponsePort;

// Helper macro for peer access
#define GET_SLAVE_PORT() getPeer()
#define GET_MASTER_PORT() getPeer()

#else  // Pre-v20

#include "mem/port.hh"

#define GET_SLAVE_PORT() getSlavePort()
#define GET_MASTER_PORT() getMasterPort()

#endif

#endif // __COMPAT_PORT_COMPAT_HH__
```

---

## Migration Checklist

### Phase 1: Core Port Changes
- [ ] `mesh_ports.hh` - Port class inheritance
- [ ] `mesh_ports.cc` - Constructor calls, peer access
- [ ] `harness.hh` - Nested port classes
- [ ] `harness.cc` - Any port-related code

### Phase 2: Python Changes
- [ ] `Harness.py` - Port type declarations
- [ ] Configuration scripts - Port bindings

### Phase 3: Request/ID Changes
- [ ] `vec_inst_sel.cc` - Request creation
- [ ] `vector.cc` - Request creation
- [ ] Any files using `MasterID`

### Phase 4: Header Cleanup
- [ ] Remove `mem_object.hh` includes
- [ ] Update to `clocked_object.hh` where needed

### Phase 5: Statistics (v21+)
- [ ] `vector.cc` - Stats registration
- [ ] `vec_inst_sel.cc` - Stats registration
- [ ] `cpi_stack.cc` - Stats registration

### Phase 6: IOCPU Integration
- [ ] Update `getMeshMasterPorts()` naming (optional)
- [ ] Update `getMeshSlavePorts()` naming (optional)
- [ ] Port binding code in CPU

### Phase 7: Testing
- [ ] Clean build successful
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Benchmark comparison with pre-migration

---

## Version-Specific Notes

### gem5 v20.0
- Initial master/slave removal
- `MemObject` removed

### gem5 v20.1
- Port infrastructure stabilized
- Some method signatures finalized

### gem5 v21.0
- Statistics API overhaul
- Further port cleanup

### gem5 v21.1+
- Additional deprecations
- New stats group pattern

---

## References

- [gem5 v20 Release Notes](https://www.gem5.org/project/2020/05/21/gem5-20.html)
- [gem5 v21 Release Notes](https://www.gem5.org/project/2021/03/19/gem5-21-0.html)
- [gem5 Terminology Change Discussion](https://gem5-review.googlesource.com/c/public/gem5/+/24919)
- [gem5 Port Documentation](https://www.gem5.org/documentation/general_docs/memory_system/port/)
