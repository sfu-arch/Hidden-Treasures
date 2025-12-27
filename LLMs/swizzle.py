import math
import numpy as np
import matplotlib.pyplot as plt

def bank8_id(addr: int, s: int) -> int:
    """
    Map address to **8‑bank** shared memory using Hopper‑style XOR swizzle.

    Parameters
    ----------
    addr : int
        Byte address (must be a 16‑byte core‑matrix start).
    s : int
        Swizzle selector (0‑3).

    Returns
    -------
    int
        Bank index 0‑7.
    """
    word = addr >> 2 # Convert byte address to word address
    # print binary
    print(f"address: {addr:#08b}")
    print(f"word: {word:#08b}")
    folded_bits = (word >> 3) & ((1 << s) - 1) # Fold bits according to swizzle selector
    print(f"folded_bits: {folded_bits:#08b}")
    print(f"word ^ folded_bits: {(word ^ folded_bits):#08b}")
    return (word ^ folded_bits) & 0x07   # 3‑bit mask → 8 banks


def group_by_bank8(limit: int, s: int):
    """Return dict{bank: [addr, ...]} for addresses 0..limit (step 16 B)."""
    d = {b: [] for b in range(8)}
    for addr in range(0, limit + 1, 16):
        d[bank8_id(addr, s)].append(addr)
    return d


def plot_group_by_bank8(limit: int, s: int):
    banks = group_by_bank8(limit, s)
    width = max(len(lst) for lst in banks.values())
    mat   = np.full((8, width), -1, dtype=int)
    text  = np.full((8, width), "", dtype=object)

#     for b, lst in banks.items():
#         for c, addr in enumerate(lst):
#             mat[b, c]  = b
#             text[b, c] = f"{addr:#04x}"

#     fig, ax = plt.subplots(figsize=(width * 0.45, 3))
#     im = ax.imshow(mat, vmin=0, vmax=7, aspect="auto")

#     # annotate addresses
#     for r in range(8):
#         for c in range(width):
#             if text[r, c]:
#                 ax.text(c, r, text[r, c], ha="center", va="center",
#                         fontsize=7, color="white")

#     ax.set_yticks(range(8))
#     ax.set_ylabel("Bank ID (0‑7)")
#     ax.set_xlabel("Address slots per bank (sorted ↑)")
#     ax.set_title(f"Addresses grouped by 8‑bank mapping — swizzle s={s}  (limit 0x{limit:X})")
#     plt.tight_layout()
#     return fig

# # === DEMO PARAMETERS ===
LIMIT    = 0x100
# SWIZZLES = [0, 1, 2, 3]

# figs = []
# for s in SWIZZLES:
#     figs.append(plot_group_by_bank8(LIMIT, s))

#plt.show()

for addr in range(0, LIMIT + 1, 4):
    print(f"Address: {addr:#04x}, Bank: {bank8_id(addr, 0)}")