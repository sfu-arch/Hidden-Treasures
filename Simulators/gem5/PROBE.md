# gem5 Probe System Notes

This document explains how gem5 probes connect event sources (probe points) to callbacks (probe listeners), with focus on:

- `ProbeListenerArg::notify(...)` in `src/sim/probe/probe.hh`
- `connectListener<...>(..., "RetiredInsts", ...)` in `src/cpu/probes/inst_tracker.cc`

## Key Line

From `src/sim/probe/probe.hh`:

```cpp
void notify(const Arg &val) override { (object->*function)(val); }
```

What it means:

- `object` is a pointer to the listener owner object (`T*`).
- `function` is a pointer to one member function on `T` with signature:
  - `void (T::*)(const Arg&)`
- `(object->*function)(val)` invokes that member function on that object.

This is the final dispatch step from probe event to your handler.

## End-to-End Flow (CPU -> Handler)

### 1) Listener registration

In `src/cpu/probes/inst_tracker.cc`, `LocalInstTracker::regProbeListeners()` does:

```cpp
connectListener<LocalInstTrackerListener>(
    this, "RetiredInsts", &LocalInstTracker::retiredInstsHandler);
```

`LocalInstTrackerListener` is:

- `ProbeListenerArg<LocalInstTracker, uint64_t>`
- Defined in `src/cpu/probes/inst_tracker.hh`

### 2) Listener creation + attach by name

`connectListener(...)` in `src/sim/probe/probe_listener_object.hh` calls:

- `ProbeManager::connect<T>(...)` (`src/sim/probe/probe.hh`)

That:

- constructs the listener (`new T(...)`)
- registers it with `addListener(listener_name, listener)`

`ProbeManager::addListener(...)` (`src/sim/probe/probe.cc`) finds probe points with matching name (here `"RetiredInsts"`) and attaches the listener.

### 3) Probe point definition in CPU

`BaseCPU::regProbePoints()` (`src/cpu/base.cc`) creates:

- `ppRetiredInsts = pmuProbePoint("RetiredInsts");`

So the probe point exists before listeners are connected.

### 4) Runtime event emission

On commit, CPU models call `probeInstCommit(...)`:

- `src/cpu/simple/base.cc`
- `src/cpu/o3/cpu.cc`
- `src/cpu/minor/execute.cc`

Then `BaseCPU::probeInstCommit(...)` in `src/cpu/base.cc` emits:

```cpp
ppRetiredInsts->notify(1);
```

### 5) Dispatch chain

- `ProbePointArg<uint64_t>::notify(const Arg&)` iterates attached listeners.
- For each listener, `ProbeListenerArg::notify(...)` runs:
  - `(object->*function)(val)`
- This invokes `LocalInstTracker::retiredInstsHandler(const uint64_t&)`.

## Why `Args&&...` Is Variadic (and What It Is Not)

In `ProbeManager::connect`:

```cpp
template <typename Listener, typename... Args>
ProbeListenerPtr<Listener> connect(Args &&...args)
{
    ProbeListenerPtr<Listener> result(
        new Listener(std::forward<Args>(args)...),
        ProbeListenerCleanup(this));
    addListener(result->getName(), *result);
    return result;
}
```

`Args&&...` is for constructor-time flexibility.

- It allows `connect` to construct any listener type with whatever constructor that listener needs.
- It does not make probe events multi-argument.

Runtime event delivery is still single-payload:

- `ProbePointArg<Arg>::notify(const Arg&)`
- `ProbeListenerArg<T, Arg>::notify(const Arg&)`
- handler signature: `void (T::*)(const Arg&)`

### Example A: Variadic constructor arguments (works today)

```cpp
// Listener type:
using L = ProbeListenerArg<LocalInstTracker, uint64_t>;

// connect() forwards three constructor args to L(...)
connect<L>(this, "RetiredInsts", &LocalInstTracker::retiredInstsHandler);
```

### Example B: Why payload still stays one argument

This is valid:

```cpp
ppRetiredInsts->notify(1); // Arg = uint64_t
```

This is not valid with current API:

```cpp
ppRetiredInsts->notify(1, pc, tid); // no notify overload with 3 payload args
```

To carry multiple values at runtime, wrap them in one payload type (struct/tuple/pair).

### Example C: Multi-field payload via struct (recommended)

```cpp
struct RetiredInstEvent
{
    uint64_t count;
    Addr pc;
    ThreadID tid;
};

using RetiredInstListener = ProbeListenerArg<MyObj, RetiredInstEvent>;

// emission
ppRetiredInstsEx->notify({.count = 1, .pc = pc, .tid = tid});

// callback
void MyObj::onRetiredInst(const RetiredInstEvent &e)
{
    // e.count, e.pc, e.tid
}
```

## Initialization Ordering

`src/sim/cxx_manager.cc` instantiation sequence is:

1. `regProbePoints()`
2. `regProbeListeners()`

This ordering ensures probe points exist before listeners try to connect.

## How To Add a Probe Point

Use this when you are instrumenting a SimObject and want others to subscribe.

### Step 1: Choose payload type and probe name

- Payload type is the single runtime argument (`Arg`) delivered by `notify(const Arg&)`.
- Probe name is the string listeners will subscribe to (for example `"RetiredInsts"`).

If you need multiple runtime values, define a struct payload:

```cpp
struct MyEvent
{
    uint64_t count;
    Addr pc;
};
```

### Step 2: Add probe point member in the producer class

In your class header:

```cpp
#include "sim/probe/probe.hh"

class MySimObject : public SimObject
{
  private:
    ProbePointArg<MyEvent> *ppMyEvent = nullptr;

  public:
    void regProbePoints() override;
};
```

### Step 3: Instantiate probe point in `regProbePoints()`

In your `.cc` file:

```cpp
void
MySimObject::regProbePoints()
{
    SimObject::regProbePoints();
    ppMyEvent = new ProbePointArg<MyEvent>(getProbeManager(), "MyEvent");
}
```

Important:

- Pass `getProbeManager()` from the producing SimObject.
- Use the exact name listeners will use in `connect(..., "MyEvent", ...)`.

### Step 4: Emit events at runtime

At the event site:

```cpp
ppMyEvent->notify({.count = 1, .pc = pc});
```

Optionally skip expensive payload assembly if nobody listens:

```cpp
if (ppMyEvent->hasListeners()) {
    ppMyEvent->notify({.count = 1, .pc = pc});
}
```

### Step 5: Attach a listener

Consumer side (a `ProbeListenerObject` subclass):

```cpp
using MyEventListener = ProbeListenerArg<MyListenerObj, MyEvent>;

connectListener<MyEventListener>(this, "MyEvent", &MyListenerObj::onMyEvent);
```

Handler:

```cpp
void
MyListenerObj::onMyEvent(const MyEvent &e)
{
    // consume e.count, e.pc
}
```

### BaseCPU-specific note

For PMU-style `uint64_t` probes in `BaseCPU`, there is a helper:

- `pmuProbePoint("Name")` in `src/cpu/base.cc`

Example from current code:

- registration: `ppRetiredInsts = pmuProbePoint("RetiredInsts");`
- emission: `ppRetiredInsts->notify(1);`

## Attaching Multiple Trackers to One Core (Python config)

In gem5 stdlib, core wrappers expose the underlying CPU SimObject as
`core.core` (for example in `BaseCPUCore`).

- `core.core` is already defined by the wrapper.
- Attributes like `core.core.probeListener` are usually dynamic child
  assignments of SimObjects.

### Key rule

Do not reuse the same attribute name if you want more than one tracker.

This overwrites:

```python
core.core.probeListener = tracker_a
core.core.probeListener = tracker_b  # tracker_a replaced
```

Use unique child names instead:

```python
for c in processor.get_cores():
    inst_tracker = LocalInstTracker(
        manager=c.core,
        global_inst_tracker=global_inst_tracker,
        start_listening=True,
    )

    pc_tracker = PcCountTracker(
        manager=c.core,
        core=c.core,
        ptmanager=pc_manager,
        targets=pc_targets,
    )

    c.core.instTracker = inst_tracker
    c.core.pcTracker = pc_tracker
```

Notes:

- Setting `manager=c.core` makes the target probe manager explicit.
- Each tracker can subscribe to different probe names, or even the same probe
  name; probe points support multiple listeners.
- One tracker object should be attached once (a SimObject cannot have multiple
  parents).

## Ownership and Cleanup

- `ProbeListenerObject` stores listeners in:
  - `std::vector<ProbeListenerPtr<>> listeners;`
- `ProbeListenerPtr` has a cleanup deleter that first disconnects from the manager, then deletes the listener.
- Clearing `listeners` disconnects callbacks safely.

Relevant files:

- `src/sim/probe/probe_listener_object.hh`
- `src/sim/probe/probe_listener_object.cc`
- `src/sim/probe/probe.hh`

## If You Need More Than One Logical Argument

The probe API is single-payload (`Arg`) by design. Use a struct as payload.

### Recommended pattern

```cpp
struct RetiredInstEvent
{
    uint64_t count;
    Addr pc;
    ThreadID tid;
};
```

Then:

- Probe point type: `ProbePointArg<RetiredInstEvent>`
- Listener type: `ProbeListenerArg<MyListener, RetiredInstEvent>`
- Handler signature: `void handler(const RetiredInstEvent &e)`

This keeps type safety and clarity, and avoids changing the core probe template API.

## Common Pitfalls

- Name mismatch: listener name must match probe point name exactly.
- Type mismatch: listener payload type must match probe point payload type.
- Silent non-attach risk: if no matching probe name exists at attach time, registration fails for that name.
- Semantics mismatch: if emitter sends `count > 1` in future, handlers that always increment by 1 will undercount.
