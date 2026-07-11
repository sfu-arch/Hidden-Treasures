#!/usr/bin/env bash
# Profile each masked-SpGEMM execution mode with Nsight Compute (one op per
# invocation so kernel->op attribution is unambiguous), then render the roofline.
#
#   ./run_profile.sh
#
# Requires: ncu (Nsight Compute) on PATH, and CUDA permission for counters
# (run as root, or set: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
#  and load nvidia driver with NVreg_RestrictProfilingToAdminUsers=0).
set -euo pipefail

M=${M:-8192}; N=${N:-8192}; K=${K:-128}
DENSITY=${DENSITY:-0.05}
MASK=${MASK:-block_global}
OUTDIR=${OUTDIR:-prof}
PY=${PY:-python3}
SCRIPT=${SCRIPT:-masked_spgemm_roofline.py}

METRICS="gpu__time_duration.sum,dram__bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"

mkdir -p "$OUTDIR"
#    --nvtx --nvtx-include "measure/" \\
for OP in gemm  spmm sddmm_sampled sddmm_densemask; do
  echo "=== profiling $OP ==="
  sudo bash << EOF || echo "WARNING: ncu exited non-zero for $OP, continuing"
. /etc/profile.d/envmod-rcg.sh
module load LIB/CUDA/13.0
source /localhome/ashriram/envs/llmdev/bin/activate
/usr/local/cuda-13.0/bin/ncu --csv --log-file "$OUTDIR/$OP.csv" \\
    --metrics "$METRICS" \\
    --target-processes all \\
    "$PY" "$SCRIPT" profile --op "$OP" \\
      --M "$M" --N "$N" --K "$K" --density "$DENSITY" --mask "$MASK" \\
      --meta-out "$OUTDIR/$OP.meta.json"
EOF
done

echo "=== rendering roofline ==="
"$PY" "$SCRIPT" plot --prof "$OUTDIR" --out roofline.svg
echo "done -> roofline.svg"
