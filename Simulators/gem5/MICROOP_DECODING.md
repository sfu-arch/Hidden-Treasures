# Micro-op Decomposition in `decode.cc`

This note explains how MinorCPU Decode turns macro-ops into micro-ops in:
- `decode.cc`
- `decode.hh`

## Overview

Decode receives instructions from Fetch2 and emits instructions to Execute.
If an instruction is a macro-op (`static_inst->isMacroop()`), Decode emits one
micro-op per output slot/cycle until the macro-op is fully decomposed.

Core loop: `Decode::evaluate()` in `decode.cc`.

## Per-thread Decode State

Decode keeps per-thread progress in `DecodeThreadInfo` (`decode.hh`):
- `inputIndex`: current position in the head input packet.
- `inMacroop`: true while a macro-op is mid-decomposition.
- `microopPC`: saved micro-op PC state for resuming decomposition.
- `execSeqNum`: sequence number source for emitted instructions.

This state allows Decode to pause decomposition when output bandwidth is full
and continue in a later cycle.

## Decomposition Flow

For each selected thread, Decode:
1. Reads the current instruction at `inputIndex`.
2. If bubble: skip.
3. If fault: pass fault onward (no decomposition).
4. If non-macro instruction: pass through unchanged.
5. If macro-op:
   - On first micro-op, initialize `microopPC` from macro-op `inst->pc` and set
     `inMacroop = true`.
   - Fetch the static micro-op using:
     - `fetchRomMicroop(microPC, static_inst)` for ROM microPCs, or
     - `static_inst->fetchMicroop(microPC)` otherwise.
   - Create a new dynamic instruction for the micro-op.
   - Set output instruction PC to `microopPC`.
   - Copy prediction fields only if this is the last micro-op.
   - Advance `microopPC` via `static_micro_inst->advancePC(...)`.
   - If last micro-op, consume the input macro-op (`inputIndex++`) and clear
     `inMacroop`.

Every emitted output instruction (micro-op or pass-through instruction) receives
its own `execSeqNum`, then `execSeqNum` increments.

## What `microopPC` Is

`microopPC` is Decode’s saved program-counter state while expanding a macro-op.
Type: `std::unique_ptr<PCStateBase>`.

It tracks both:
- instruction address (`PCStateBase::_pc`)
- micro-op index/state (`PCStateBase::_upc`, exposed as `microPC()`)

Decode uses `microopPC->microPC()` to fetch the next micro-op and updates it
after each emitted micro-op. This is what makes multi-cycle macro-op expansion
possible.

## What `PCStateBase` Is

`PCStateBase` (in `arch/generic/pcstate.hh`) is gem5’s ISA-independent base
class for program counter state. It provides:
- `instAddr()` for instruction address
- `microPC()` for micro-op PC
- virtual `clone()`, `update(...)`, `advance()`, and formatting hooks

Using `PCStateBase` lets MinorCPU decode logic remain generic across ISAs while
still supporting ISA-specific concrete PC state implementations.

## Important Behavioral Details

- Decode only advances `inputIndex` for a macro-op when the last micro-op is
  emitted.
- Therefore, one macro-op may occupy multiple output slots/cycles.
- An assertion ensures input packets are only popped when not mid-macro-op.
- Branch prediction metadata (`predictedTaken`, `predictedTarget`) is attached
  only to the final micro-op.

## Minor vs O3: Common Micro-op Handling Patterns

Both cores use the same underlying micro-op concepts (`microPC`, `isLastMicroop`,
macro-op parent context), but place decomposition in different stages.

### 1) Where decomposition happens

- MinorCPU: decomposition happens in Decode (`src/cpu/minor/decode.cc`).
  Decode explicitly checks `isMacroop()`, fetches micro-ops, and emits them.
- O3CPU: decomposition happens in Fetch (`src/cpu/o3/fetch.cc`), not Decode.
  O3 Decode (`src/cpu/o3/decode.cc`) primarily forwards already-built dyn insts
  to Rename with flow-control/squash handling.

### 2) Persistent per-thread macro-op state

- Minor: `DecodeThreadInfo` keeps `inMacroop` + `microopPC`.
- O3: Fetch keeps `macroop[tid]` (`src/cpu/o3/fetch.hh`) plus per-thread PC
  state (`pc[tid]`) and writes back `macroop[tid] = curMacroop` each cycle.

Pattern: decomposition state is persisted per thread so expansion can span
multiple cycles without losing position.

### 3) Micro-op selection method

Both Minor and O3 select next micro-op from either:
- ROM microcode via `fetchRomMicroop(microPC, macroop)`, or
- macro-op-local stream via `fetchMicroop(microPC)`.

Pattern: micro-op source is abstracted behind decoder/static-inst APIs; pipeline
logic stays ISA-agnostic.

### 4) End-of-macro-op boundary

Both use `isLastMicroop()` as the boundary signal for finishing current macro-op
and moving to next macro-op/instruction stream.

Pattern: explicit last-micro-op markers simplify control flow, prediction
handoff, and commit-level architectural boundaries.

### 5) Decode-stage role differences

- Minor Decode does real transformation (macro-op -> micro-op dyn inst objects)
  and assigns per-emitted-inst sequence numbers.
- O3 Decode (`decodeInsts`) mostly:
  - drains per-thread input/skid queues,
  - filters squashed instructions,
  - forwards valid dyn insts to Rename up to `decodeWidth`.

Pattern: front-end decomposition can be done either in Fetch or Decode; later
stages work best when they only see normalized micro-op granularity.
