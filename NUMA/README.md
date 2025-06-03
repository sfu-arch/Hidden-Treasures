- [Dependencies](#dependencies)
- [Good benchmarking practices](#good-benchmarking-practices)
- [Memory Latency Checker Usage.](#memory-latency-checker-usage)
  - [Measure memory latencies](#measure-memory-latencies)
  - [numactl core and memory Bindings](#numactl-core-and-memory-bindings)
- [Memory benchmarking](#memory-benchmarking)
- [Example output](#example-output)

# Dependencies
```bash
sudo apt-get install msr-tools
sudo apt-get install linux-tools-common
sudo apt-get install linux-tools-generic
sudo apt-get install numactl
# Older systems may require:
sudo apt-get install numactl-devel
sudo apt-get install libnuma-devel
```


# Good benchmarking practices

1. **Isolate the Benchmark**: Run benchmarks on a dedicated system or ensure minimal background processes to avoid interference.

```bash
# Assigns the threads created on benchmark to specific CPU cores.
# Note that typically linux assigns odd cores to one socket and even cores to another.
# In this example thus is an inefficient allocation with 2 threads per socket and threads spread out.
taskset -c 4,5,6,7 ./benchmark
# Better allocation would be:
taskset -c 0,2,4,6 ./benchmark
```



2. **Enable highest performance mode**: Set the CPU governor to "performance" mode to ensure maximum CPU speed.
   ```bash
   sudo cpufreq-set -r -g performance
   # OR 
   sudo cpupower frequency-set -g performance
   ```
3. Disable Enable Prefetchers
```bash
cpu-info | # Note down processor model and number
# Model specific registers (MSRs) are used to control CPU features.
# For prefetchers on Intel CPUs it is typically MSR_IA32_MISC_ENABLE 0x1a4
sudo rdmsr --all 0x1a4
sudo rdmsr -p 0 0x1a4
# IA32 bit0 = L2 hardware prefetcher, bit1 = L2 adjacent-line,
# bit2 = DCU streamer, bit3 = DCU IP, L2 prefetcher has option to prefetch into L2 or L3. Disabling it will disable both options
# Note that setting the bit disable the prefetcher, clearing the bit enables it.
sudo wrmsr -p 0 0x1a4 0xF # Disable all prefetchers
sudo wrmsr -p 0 0x1a4 0x0 # Enable all prefetchers
```


# Memory Latency Checker Usage.
Intel Memory Latency Checker (MLC) is a tool to measure memory latency on Intel processors. 
```bash
# Check numa settings
$ numactl -H                                    ∞
available: 2 nodes (0-1)
node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94
node 0 size: 257556 MB
node 0 free: 209967 MB
node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95
node 1 size: 258031 MB
node 1 free: 219370 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
# on cs-arch04
```

## Measure memory latencies
```bash
sudo ./mlc ∞
Intel(R) Memory Latency Checker - v3.11b
Measuring idle latencies for sequential access (in ns)...
		Numa node
Numa node	     0	     1
       0	  82.1	 141.0
       1	 141.5	  81.5

Measuring Peak Injection Memory Bandwidths for the system
Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
Using all the threads from each core if Hyper-threading is enabled
Using traffic with the following read-write ratios
ALL Reads        :	158844.5
3:1 Reads-Writes :	150658.6
2:1 Reads-Writes :	149341.2
1:1 Reads-Writes :	143355.4
Stream-triad like:	128158.7

Measuring Memory Bandwidths between nodes within system
Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
Using all the threads from each core if Hyper-threading is enabled
Using Read-only traffic type
		Numa node
Numa node	     0	     1
       0	79670.3	34393.4
       1	34385.7	79678.3

Measuring Loaded Latencies for the system
Using all the threads from each core if Hyper-threading is enabled
Using Read-only traffic type
Inject	Latency	Bandwidth
Delay	(ns)	MB/sec
==========================
 00000	329.26	 159154.0
 00002	330.68	 159225.6
 00008	330.86	 159201.7
 00015	329.26	 159040.5
 00050	326.68	 159171.9
 00100	325.98	 158992.6
 00200	293.81	 158637.6
 00300	127.88	 132998.9
 00400	110.64	 102244.7
 00500	101.15	  83217.0
 00700	 96.70	  60426.1
 01000	 91.30	  42927.2
 01300	 89.38	  33361.5
 01700	 88.10	  25747.4
 02500	 86.39	  17829.5
 03500	 84.41	  13006.8
 05000	 83.80	   9351.6
 09000	 82.63	   5555.4
 20000	 82.30	   2932.1

Measuring cache-to-cache transfer latency (in ns)...
Local Socket L2->L2 HIT  latency	49.5
Local Socket L2->L2 HITM latency	49.5
Remote Socket L2->L2 HITM latency (data address homed in writer socket)
			Reader Numa Node
Writer Numa Node     0	     1
            0	     -	 115.2
            1	 115.5	     -
Remote Socket L2->L2 HITM latency (data address homed in reader socket)
			Reader Numa Node
Writer Numa Node     0	     1
            0	     -	 184.8
            1	 185.9	     -
```

## numactl core and memory Bindings

On multi-socket systems, ensure your benchmark process and the memory it allocates are on the same NUMA node if you want to measure local NUMA latency. You can use tools like numactl for this:

```bash
numactl --cpunodebind=0 --membind=0 ./latency_benchmark
```

# Memory benchmarking
```bash
gcc -O3 -march=native -o membench membench.c
chmod +x run.sh
sudo ./run.sh
```
# Example output

```
$ sudo ./run.sh
=== Prefetch ENABLED ===
0
+ taskset -c 0 ./membench 536870912
seq 11.3 GB/s  (checksum=0x1ffffff8000000)

=== Prefetch DISABLED ===
0
+ taskset -c 0 ./membench 536870912
seq 6.9 GB/s  (checksum=0x1ffffff8000000)
chase: 104.908 ns/hop  (final idx 40447003)
```

```bash
# NUMA bind
# Disable prefetchers on cpu 0
sudo wrmsr -p 0 0x1a4 0xf
# Run the benchmark with prefetchers disabled
# Memory allocation is bound to NUMA node 0
numactl --physcpubind=0 --membind=0 ./membench 536870912
# Memory allocation is bound to NUMA node 1
numactl --physcpubind=0 --membind=1 ./membench 536870912

$ sudo wrmsr -p 0 0x1a4 0xf # Disable prefetchers on cpu 0
$ numactl --physcpubind=0 --membind=0 ./membench 536870912
seq 7.1 GB/s  (checksum=0x1ffffff8000000)
chase: 103.760 ns/hop  (final idx 40447003)
$ numactl --physcpubind=0 --membind=1 ./membench 536870912

```


| Prefetch state | Memory binding (`--membind`) | Sequential bandwidth (GiB/s) | Pointer-chase latency (ns / hop) |
| -------------- | ---------------------------- | ---------------------------- | -------------------------------- |
| **Disabled**   | 0                            | **7.1**                      | **103.760**                      |
| **Disabled**   | 1 (remote)                   | 4.9                          | 152.016                          |
| **Enabled**    | 0                            | 11.4                         | 103.715                          |
| **Enabled**    | 1 (remote)                   | 8.4                          | 151.913                          |

**Notes**

* `sudo wrmsr -p 0 0x1a4 0x0` cleared the disable bits in MSR 0x1A4, re-enabling all hardware prefetchers before the last two runs.
* As expected, sequential throughput rises markedly when prefetchers are on and/or data is local to the socket, while the pointer-chase workload remains dominated by NUMA distance rather than prefetch.

