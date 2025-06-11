#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <getopt.h>
#include <string.h>

/* -------- timing helpers -------- */
static inline uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* MiB → bytes; GiB factor for later */
#define GiB (1024.0 * 1024.0 * 1024.0)

void print_usage(const char *program_name)
{
    printf("Usage: %s [OPTIONS] [BYTES]\n", program_name);
    printf("NUMA memory access pattern benchmarks\n\n");
    printf("Arguments:\n");
    printf("  BYTES               Memory size to test (default: 512 MiB)\n\n");
    printf("Options:\n");
    printf("  -s, --sequential    Run sequential memory access pattern only\n");
    printf("  -p, --pointer       Run pointer chasing access pattern only\n");
    printf("  -a, --all           Run both access patterns (default)\n");
    printf("  -h, --help          Show this help message\n");
}

int main(int argc, char **argv)
{
    int run_sequential = 0;
    int run_pointer = 0;
    int run_all = 0;
    int c;
    
    static struct option long_options[] = {
        {"sequential", no_argument, 0, 's'},
        {"pointer", no_argument, 0, 'p'},
        {"all", no_argument, 0, 'a'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    while ((c = getopt_long(argc, argv, "spah", long_options, NULL)) != -1) {
        switch (c) {
            case 's':
                run_sequential = 1;
                break;
            case 'p':
                run_pointer = 1;
                break;
            case 'a':
                run_all = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // If no specific scan is requested, default to all
    if (!run_sequential && !run_pointer && !run_all) {
        run_all = 1;
    }
    
    const size_t bytes = (optind < argc)
        ? strtoull(argv[optind], NULL, 0)
        : (1UL << 29);               /* default 512 MiB */

    const size_t n = bytes / sizeof(uint64_t);
    uint64_t  *buf = aligned_alloc(64, n * sizeof(uint64_t));
    if (!buf) { perror("alloc"); return 1; }

    /* Commit pages and initialise */
    for (size_t i = 0; i < n; ++i) buf[i] = i;

    srand(1);                        /* deterministic run-to-run */

    if (run_all || run_sequential) {
        // -------- sequential scan --------
        uint64_t t0 = ns_now(), sum = 0;
        for (size_t r=0;r<4;r++)          // repeat to amplify effect
            for (size_t i=0;i<n;i=i+8)
                sum += buf[i];
        uint64_t t1 = ns_now();
        double gb_s = (double)(4*bytes) / (t1-t0);   // bytes/ns  -> GB/s
        printf("seq %.1f GB/s  (checksum=%#llx)\n", gb_s, (unsigned long long)sum);
    }

    if (run_all || run_pointer) {
        /* -------- pointer-chase build -------- */
        /* Create a random permutation so every element points to the next */
        for (size_t i = n - 1; i > 0; --i) {
            size_t j = (size_t)rand() % (i + 1);
            uint64_t tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp;
        }
        for (size_t i = 0; i < n; ++i) buf[i] %= n;

        /* -------- pointer-chase timing -------- */
        size_t idx = 0;
        uint64_t t0 = ns_now();
        for (size_t i = 0; i < n; ++i)
            idx = buf[idx];
        uint64_t t1 = ns_now();

        double hop_ns = (double)(t1 - t0) / (double)n;   /* ← fixed! */
        printf("chase: %.3f ns/hop  (final idx %zu)\n", hop_ns, idx);
    }

    free(buf);
    return 0;
}
