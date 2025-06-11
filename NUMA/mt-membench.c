#define _GNU_SOURCE // Must be the first line to enable CPU affinity macros
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>

/* -------- timing helpers -------- */
static inline uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Struct to hold data for each thread
typedef struct {
    int thread_id;
    int cpu_id;
    int memory_node;    // Node where memory is allocated
    uint64_t *buf;      // Pointer to shared buffer
    size_t start_idx;   // Starting index for this thread
    size_t n_per_thread;
    pthread_barrier_t *barrier;
    double *bandwidth_gb_s;
    double *latency_ns_hop;
} thread_data_t;


void *worker_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;

    // Pin thread to its assigned CPU core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->cpu_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
        return NULL;
    }

    // Allocate large local array to flush buf from cache
    const size_t CACHE_FLUSH_SIZE = 64 * 1024 * 1024; // 64MB
    volatile char *flush_array = (volatile char*)malloc(CACHE_FLUSH_SIZE);
    if (flush_array) {
        // Write to every cache line to ensure flush
        for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) {
            flush_array[i] = (char)(i & 0xFF);
        }
        // Read back to ensure memory access
        volatile char dummy = 0;
        for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) {
            dummy += flush_array[i];
        }
        free((void*)flush_array);
    }

    // Get pointer to this thread's section of the buffer
    uint64_t *buf = data->buf + data->start_idx;
    const size_t n = data->n_per_thread;

    // Initialize buffer section for sequential scan
    for (size_t i = 0; i < n; ++i) {
        buf[i] = data->start_idx + i;
    }
    
    // Wait for all threads to finish initialization
    pthread_barrier_wait(data->barrier);

    /* -------- 1. Sequential Scan (Bandwidth) -------- */
    uint64_t t0_bw = ns_now();
    volatile uint64_t sum = 0;
    const int repeats = 4;
    for (int r = 0; r < repeats; r++) {
        for (size_t i = 0; i < n; i++) {
            sum += buf[i];
        }
    }
    uint64_t t1_bw = ns_now();
    *(data->bandwidth_gb_s) = (double)(repeats * n * sizeof(uint64_t)) / (double)(t1_bw - t0_bw);
    
    // Wait for all threads to finish bandwidth test
    pthread_barrier_wait(data->barrier);

    /* -------- 2. Pointer-Chase (Latency) -------- */
    // Build the permutation in this thread's section of the buffer
    srand(data->thread_id + 1);
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = (size_t)rand() % (i + 1);
        uint64_t tmp = buf[i];
        buf[i] = buf[j];
        buf[j] = tmp;
    }

    // Wait for all threads to finish setting up pointer chase
    pthread_barrier_wait(data->barrier);
    
    volatile size_t idx = 0;
    uint64_t t0_lat = ns_now();
    const size_t chase_length = n;
    for (size_t i = 0; i < chase_length; ++i) {
        idx = buf[idx % n]; // Ensure we stay within this thread's section
    }
    uint64_t t1_lat = ns_now();
    *(data->latency_ns_hop) = (double)(t1_lat - t0_lat) / (double)chase_length;

    return NULL;
}

void run_benchmark(int num_threads, size_t total_bytes, const char *test_name, int execution_node, int memory_node) {
    printf("\n--- Running Test: %s ---\n", test_name);
    printf("Config: %d threads on Node %d, Memory on Node %d\n", num_threads, execution_node, memory_node);
    printf("Total memory size: %.0f MiB.\n", (double)total_bytes / (1024*1024));

    struct bitmask *cpumask = numa_allocate_cpumask();
    if(numa_node_to_cpus(execution_node, cpumask) != 0) {
        perror("numa_node_to_cpus");
        numa_free_cpumask(cpumask);
        exit(1);
    }
    
    int cpus_on_node[numa_num_possible_cpus()];
    int num_cpus_on_node = 0;
    for (int i = 0; i < numa_num_possible_cpus(); i++) {
        if (numa_bitmask_isbitset(cpumask, i)) {
            cpus_on_node[num_cpus_on_node++] = i;
        }
    }
    numa_free_cpumask(cpumask);

    if (num_cpus_on_node == 0) {
        fprintf(stderr, "Error: No CPUs found on execution node %d\n", execution_node);
        exit(1);
    }
    if (num_threads > num_cpus_on_node) {
        printf("Warning: %d threads requested, but only %d CPUs on node %d. CPUs will be reused.\n",
               num_threads, num_cpus_on_node, execution_node);
    }

    // Allocate shared memory buffer on the specified NUMA node
    const size_t n_total = total_bytes / sizeof(uint64_t);
    uint64_t *shared_buf = numa_alloc_onnode(total_bytes, memory_node);
    if (!shared_buf) {
        perror("numa_alloc_onnode");
        exit(1);
    }

    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];
    double bandwidths[num_threads];
    double latencies[num_threads];
    pthread_barrier_t barrier;

    // Initialize barrier for 3 synchronization points: init, bandwidth done, latency setup
    pthread_barrier_init(&barrier, NULL, num_threads);

    const size_t n_per_thread = n_total / num_threads;
    const size_t remainder = n_total % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].buf = shared_buf;
        // Fix start index calculation for remainder distribution
        thread_data[i].start_idx = i * n_per_thread + (i < remainder ? i : remainder);
        thread_data[i].n_per_thread = n_per_thread + (i < remainder ? 1 : 0);
        thread_data[i].barrier = &barrier;
        thread_data[i].bandwidth_gb_s = &bandwidths[i];
        thread_data[i].latency_ns_hop = &latencies[i];
        thread_data[i].cpu_id = cpus_on_node[i % num_cpus_on_node];
        thread_data[i].memory_node = memory_node;
        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }
    
    // Main thread just waits for all worker threads to complete
    double total_bandwidth = 0;
    double avg_latency = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_bandwidth += bandwidths[i];
        avg_latency += latencies[i];
    }
    avg_latency /= num_threads;

    printf("\n## Results for %s ##\n", test_name);
    printf("Total Sequential Bandwidth:    %.2f GB/s\n", total_bandwidth);
    printf("Average Pointer-Chase Latency: %.3f ns/hop\n\n", avg_latency);

    // Cleanup shared buffer
    numa_free(shared_buf, total_bytes);
    pthread_barrier_destroy(&barrier);
}

int main(int argc, char **argv) {
    if (numa_available() == -1) {
        fprintf(stderr, "NUMA not available on this system.\n");
        return 1;
    }

    const int num_threads = (argc > 1) ? atoi(argv[1]) : 4;
    const size_t bytes = (argc > 2)
                         ? strtoull(argv[2], NULL, 0) * (1UL << 20)
                         : (1UL << 30);

    if (num_threads <= 0) {
        fprintf(stderr, "Usage: %s <num_threads> [total_size_MiB]\n", argv[0]);
        return 1;
    }
    
    const int num_nodes = numa_num_configured_nodes();
    printf("System has %d NUMA nodes and %d CPUs.\n", num_nodes, sysconf(_SC_NPROCESSORS_ONLN));

    const int EXECUTION_NODE = 0;

    run_benchmark(num_threads, bytes, "Local Access (Node 0 -> Node 0)", EXECUTION_NODE, EXECUTION_NODE);

    if (num_nodes > 1) {
        const int REMOTE_MEMORY_NODE = 1;
        run_benchmark(num_threads, bytes, "Remote Access (Node 0 -> Node 1)", EXECUTION_NODE, REMOTE_MEMORY_NODE);
    } else {
        printf("\n--- Skipping Remote Access Test: Single NUMA node detected ---\n");
    }

    return 0;
}