# SE Mode Fixed File Mapping (Approach 3 Only)

This document describes one implementation path in SE mode:

- preload multiple host files into fixed physical addresses (PA)
- map each file at fixed virtual addresses (VA)
- hard-reserve those PA ranges in `MemPools` so normal SE allocation cannot collide

## Goal

Support config entries like:

```python
process.fixed_file_maps = [
    "weights0.bin,0x40000000,0x10000000,0x200000,0",
    "weights1.bin,0x42000000,0x10200000,0x100000,0",
]
```

Fields are:

- `host_path`
- `va` (target VA base)
- `pa` (target PA base)
- `size` (bytes to copy from file)
- `offset` (byte offset in host file)

## Required Invariants

1. `va` and `pa` must be page-aligned.
2. Mapping length is `roundUp(size, page_size)`.
3. File-copy length is `size`; remaining bytes to mapping length are zero-filled.
4. VMA metadata is registered (`MemState::mapRegion`) before/with page table mapping.
5. PA ranges used for fixed mappings are reserved in `MemPools` before runtime allocation.

## SEWorkload vs Process (Important)

In SE-mode configs you usually set both:

1. `system.workload = SEWorkload.init_compatible(binary)`
2. `cpu.workload = process`

These are different objects with different roles.

### What `SEWorkload.init_compatible(...)` does

`init_compatible(path)` inspects the target binary and picks the matching
`SEWorkload` subclass for that ISA/format. It is not the application state.

### Purpose of `system.workload` (`SEWorkload`)

- System-level SE shim (syscall emulation entry point).
- Syscall path is `system->workload->syscall(tc)` and then forwarded to the
  current process.
- Owns global SE physical page allocator (`MemPools`).

### Purpose of `cpu.workload` / `core.workload` (`Process`)

- Per-program state: executable/cmd/env/pid, page tables, VMAs.
- Owns per-process mappings (`Process::map`) and process memory initialization.
- This is where `fixed_file_maps` must be configured.

### Configuration rule for fixed file mappings

Set file mapping directives on `Process`, not on `SEWorkload`:

```python
system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary]
process.fixed_file_maps = [
    "weights0.bin,0x40000000,0x10000000,0x200000,0",
    "weights1.bin,0x42000000,0x10200000,0x100000,0",
]
system.cpu.workload = process
```

## Files To Modify And Exact Changes

### 1) `sim/Process.py`

Add a new process parameter:

```python
fixed_file_maps = VectorParam.String(
    [],
    "List of host_path,va,pa,size,offset entries for fixed SE preload mapping",
)
```

Why: exposes mapping directives from Python configs to C++ `ProcessParams`.

### 2) `sim/process.hh`

Add parsed representation and helpers in `Process`:

- New struct:
  - `std::string hostPath;`
  - `Addr va;`
  - `Addr pa;`
  - `Addr size;`
  - `Addr offset;`
- Member: `std::vector<FixedFileMap> fixedFileMaps;`
- Private helper declarations:
  - parser for one CSV entry
  - `applyFixedFileMaps()` invoked from `initState()`
  - loader for host bytes into buffer (with offset/size checks)

Why: keep parsing and mapping logic explicit and testable.

### 3) `sim/process.cc`

#### 3.1 Parse params in `Process::Process(...)`

- Read `params.fixed_file_maps`.
- Parse each string into `FixedFileMap`.
- Validate syntax and numeric conversion with `fatal_if(...)`.

#### 3.2 Apply mappings in `Process::initState()`

After current image/interpreter writes:

- for each `FixedFileMap`:
  - validate page alignment for `va`/`pa`
  - compute `map_len = roundUp(size, page_size)`
  - reserve PA range via new API on `SEWorkload` (see below)
  - register VMA:
    - `memState->mapRegion(va, map_len, "fixed_file:" + hostPath, -1, 0);`
  - map VA->PA:
    - `pTable->map(va, pa, map_len, EmulationPageTable::Clobber);`
  - load `size` bytes from `hostPath` at `offset`
  - write to PA:
    - `system->physProxy.writeBlob(pa, buf.data(), size);`
  - zero-fill:
    - `system->physProxy.memsetBlob(pa + size, 0, map_len - size);`

#### 3.3 Fault/overlap checks

- `fatal_if(!memState->isUnmapped(va, map_len), ...)`
- `fatal_if(pTable->lookup(va) != nullptr, ...)` if needed for clearer diagnostics
- `fatal_if(size == 0, ...)` unless empty mappings are intentionally supported

Why: enforce deterministic behavior and avoid silent overlap.

### 4) `sim/se_workload.hh`

Add wrapper API:

- `void reservePhysRange(Addr paddr, int npages, int pool_id=0);`

Why: keep Process code decoupled from `MemPools` internals.

### 5) `sim/se_workload.cc`

Implement wrapper:

- `memPools.reservePhysPages(paddr, npages, pool_id);`

### 6) `sim/mem_pool.hh`

Add reserve APIs:

- in `MemPool`:
  - `void reserve(Addr start, Addr npages);`
- in `MemPools`:
  - `void reservePhysPages(Addr page_addr, int npages, int pool_id=0);`

Optional but useful:

- `bool isFree(Addr start, Addr npages) const;` for pre-check and diagnostics.

### 7) `sim/mem_pool.cc`

Implement hard reservation by removing range from free list:

- `MemPool::reserve(start, npages)`:
  - convert to page numbers
  - verify requested range lies in pool bounds
  - remove exact range from `freePhysPages`
  - `fatal_if` any page in range was already allocated/reserved
- `MemPools::reservePhysPages(...)` delegates to selected pool

Implementation detail:

- Because `FreeList` exposes ranges, implement reserve by rebuilding range list:
  - iterate free ranges
  - subtract reserved interval
  - rebuild new `FreeList`
- keep behavior strict (`fatal_if`) instead of silently clobbering allocator state

### 8) `src/sim/SConscript` (only if needed)

If new source helpers are split into a new `.cc`, add build entry.
If all changes stay inside existing files, no SConscript change is required.

## Execution Order In Code

Recommended order during `Process::initState()`:

1. Existing ELF/interpreter load (`image.write`, `interpImage.write`).
2. Parse/validate fixed mappings.
3. Reserve all PA ranges first (fail fast on overlap).
4. For each entry: register VMA, install VA->PA mapping, preload bytes, zero-fill tail.

Rationale: reserving all PAs upfront avoids partial state if one later entry collides.

## What Must Not Be Left In Old Pattern

Remove/avoid these anti-patterns:

1. Calling `pTable->map()` for fixed VA/PA without `memState->mapRegion()`.
2. Loading file bytes into PA without reserving same PA in `MemPools`.
3. Using ad-hoc hardcoded VA/PA in random callsites instead of centralized `fixed_file_maps`.
4. Relying on later mmap/brk/fault allocation to "avoid" hardcoded PA by luck.

## Minimal Validation Plan

1. Add two non-overlapping mappings and verify:
   - VMA list contains both ranges (`printVmaList()` output)
   - expected bytes visible at mapped VAs
2. Force overlap in PA ranges and expect early fatal from reserve path.
3. Force overlap in VA ranges and expect `mapRegion`/pre-check failure.
4. Trigger regular `malloc`/`mmap` activity and confirm no allocation lands in reserved PA ranges.

## Notes

- This approach is intentionally strict and deterministic.
- It is compatible with multiple files and explicit VA/PA placement.
- The key correctness point is memory-pool reservation; without that, fixed PA mappings are unsafe in long SE runs.
