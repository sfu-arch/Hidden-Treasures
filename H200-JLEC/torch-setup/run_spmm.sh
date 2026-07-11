#!/usr/bin/env bash
# Profile A@A^T SpGEMM (fp32, cuSPARSE) per dataset with Nsight Compute, one op
# per invocation for clean kernel->op attribution, then render the roofline.
#   ./run_profile_spgemm.sh
# Requires ncu on PATH and counter permission (see run_profile.sh header).
set -euo pipefail

INDIR=${INDIR:-data}
OUTDIR=${OUTDIR:-prof_spgemm}
DATASETS=${DATASETS:-"OGBN RDT AMZ"}
PY=${PY:-python3}
SCRIPT=${SCRIPT:-spgemm_roofline.py}

METRICS="gpu__time_duration.sum,dram__bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"

mkdir -p "$OUTDIR"

for OP in gemm copy $DATASETS; do
  echo "=== profiling $OP ==="
  sudo bash << EOF || echo "WARNING: ncu exited non-zero for $OP, continuing"
. /etc/profile.d/envmod-rcg.sh
module load LIB/CUDA/13.0
source /localhome/ashriram/envs/llmdev/bin/activate
/usr/local/cuda-13.0/bin/ncu --csv --log-file "$OUTDIR/$OP.csv" \
      --nvtx --nvtx-include "measure/" \
      --metrics "$METRICS" --target-processes all \
      "$PY" "$SCRIPT" profile --op "$OP" --indir "$INDIR" \
        --meta-out "$OUTDIR/$OP.meta.json"
EOF
done

echo "=== rendering roofline ==="
"$PY" "$SCRIPT" plot --prof "$OUTDIR" --out spgemm_roofline.svg
echo "done -> spgemm_roofline.svg"