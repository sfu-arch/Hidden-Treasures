#!/usr/bin/env bash
#
# Toggle all documented Intel HW prefetchers
# and run membench on CPU 0 for both states.
#
# Needs: msr-tools, membench, taskset (util-linux)

set -euo pipefail

MASK=$((0x0F))          # bits 0-3, 9, 19 → disable every prefetcher
MSR=0x1a4                  # IA32_MISC_ENABLE
BYTES=$((512*1024*1024))   # 512 MiB test size; change if you like

toggle_prefetch() {        # $1 = 0 → enable, 1 → disable
    for cpu in /dev/cpu/[0]*; do
	cur_hex=$(rdmsr -p "${cpu##*/}" $MSR) # e.g. 8f020f
	cur=$(( 0x$cur_hex ))
	echo $cur
	if (( $1 )); then
            new=$(( cur |  MASK ))   # set bits → disable
        else
            new=$(( cur & ~MASK ))   # clear bits → enable
        fi
        wrmsr -p "${cpu##*/}" $MSR "$new"
    done
}

run_on_cpu0() {
    echo "+ taskset -c 0 ./membench $BYTES"
    taskset -c 0 ./membench "$BYTES"
}

echo "=== Prefetch ENABLED ==="
toggle_prefetch 0
run_on_cpu0

echo
echo "=== Prefetch DISABLED ==="
toggle_prefetch 1
run_on_cpu0

toggle_prefetch 0
