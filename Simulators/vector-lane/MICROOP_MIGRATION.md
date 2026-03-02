# Custom Vector Micro-Op Migration Guide

## Goal
Align `custom/` vector instruction flow with gem5 Minor/O3 micro-op handling:

1. Select the correct `i`th micro-op from ISA macro-ops.
2. Create a new dynamic context per receiving core.
3. Preserve micro-op identity and flags across mesh forwarding.

## Current Mismatch Summary

1. `custom/vec_inst_sel.cc` decodes raw instruction bits per fetch response and manually does `PC += 4`.
2. `custom/vector.cc` rebuilds mesh-received instructions by re-decoding `machInst`.
3. `_uopCnt` tracks fetched words, not ISA instruction boundaries.

These patterns break macro-op decomposition and can lose `isMicroop`, `isFirstMicroop`, `isLastMicroop`, and delayed-commit behavior.

## Required Changes In `custom/`

## 1) Make `VecInstSel` Macro-Op Aware
Files: `custom/vec_inst_sel.hh`, `custom/vec_inst_sel.cc`

1. Add macro-op tracking state:
- `StaticInstPtr _activeMacroInst;`
- Optional: `bool hasActiveMacro()` helper.

2. When a fetched instruction decodes to a macro-op:
- Save macro-op in `_activeMacroInst`.
- Emit `fetchMicroop(_uopPC->microPC())` from that macro-op.
- Keep emitting micro-ops until `isLastMicroop()` is true.
- Clear `_activeMacroInst` at last micro-op.

3. Advance PC using static instruction semantics:
- Replace manual `pc += 4` logic with `static_inst->advancePC(*_uopPC)`.
- This preserves `microPC` behavior and matches Minor/O3 decomposition style.

4. Update `_uopCnt` semantics:
- Increment only at instruction boundary:
`!static_inst->isMicroop() || static_inst->isLastMicroop()`.
- Do not increment on intermediate micro-ops.

5. Keep block active while macro decomposition is still in progress:
- `isPCGenActive()` must consider `_activeMacroInst != nullptr`.

## 2) Preserve Forwarded Micro-Ops When Rebuilding Dynamic Context
File: `custom/vector.cc`

1. In `Vector::createInstruction`, do not re-decode from `machInst` for mesh-forwarded instructions.
2. Build a new `IODynInst` using the incoming `static_inst_p` directly.
3. Keep generating a new sequence number and local dynamic context.

This keeps micro-op identity correct while still avoiding shared mutable dynamic state across cores.

## 3) Keep Sender/Receiver Contract Explicit
Files: `custom/vector.cc`, `custom/vec_inst_sel.hh`, `custom/vec_inst_sel.cc`

1. `MasterData.isInst == true`: receiver should treat payload as already-selected instruction/micro-op.
2. `MasterData.isInst == false`: receiver should start local PC-driven fetch/decompose path.

Do not mix these paths by re-decoding `machInst` in the `isInst == true` path.

## Old Patterns To Remove/Clean Up

## A) Remove Manual Per-Word Decode Path As The Source Of Uop Identity
File: `custom/vec_inst_sel.cc`

Remove/replace:
- Manual decode as final instruction selection:
`decoder.decode(mach_inst, 0x0)` as the only emitted op source.
- Manual PC increment logic:
`riscv_pc.pc(riscv_pc.instAddr() + sizeof(RiscvISA::MachInst));`
`riscv_pc.npc(...)`.

Keep decode only as macro-op entry decode, then use `fetchMicroop`.

## B) Remove Re-Decode Of Mesh-Forwarded Dynamic Instructions
File: `custom/vector.cc`

Remove/replace:
- Rebuilding `static_inst` via:
`uint32_t machInst = (uint32_t)inst->static_inst_p->machInst;`
`static_inst = extractInstruction(machInst, cur_pc);`

This is the key behavior that drops micro-op identity.

## C) Remove Ambiguous `_uopCnt` Meaning
Files: `custom/vec_inst_sel.hh`, `custom/vec_inst_sel.cc`

Clean up:
- `_uopCnt` should mean "completed ISA instructions in current vissue block."
- Remove comments/usages implying "_uopCnt == number of emitted micro-ops."

## D) Remove Unused Decode Helpers After Migration
File: `custom/vector.cc`, `custom/vector.hh`

If no remaining callsites:
- Remove `Vector::extractInstruction(...)`.

## Optional But Recommended Cleanup

1. Pick one block-termination scheme:
- Count-based (`imm5`) or terminator-based.
- Avoid maintaining both modes unless required by ISA contract.

2. Rename fields for clarity:
- `_uopIssueLen` -> `_issueInstLen`
- `_uopCnt` -> `_issuedInstBoundaryCnt`

3. Add local helper methods in `VecInstSel`:
- `StaticInstPtr nextStaticInstFromPC();`
- `bool onInstructionBoundary(const StaticInstPtr &si) const;`

## Behavior Checks After Migration

1. For RVV macro-op instructions, all cores emit matching micro-op sequences and terminate on `isLastMicroop`.
2. No macro-op reaches execute as a direct executable instruction.
3. Forwarded uops retain micro-op flags after `Vector::createInstruction`.
4. `_uopCnt` reaches block length at ISA instruction boundaries only.

## Minimal Verification Plan

1. Enable mesh/vector debug flags and log:
- static instruction name
- `isMicroop`
- `isFirstMicroop`
- `isLastMicroop`
- PC and microPC
- `_uopCnt`

2. Compare one vector block on:
- root master core
- one slave core

3. Confirm same micro-op order and same boundary count evolution.

