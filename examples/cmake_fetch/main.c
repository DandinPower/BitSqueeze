#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// include the bitsqueeze API
#include "bitsqueeze.h"

void generate_data(float *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float r = (float)rand() / (float)RAND_MAX; // 0.0 to 1.0
        data[i] = (r * 20.0f) - 10.0f;             // -10.0 to 10.0
    }
}

int main(void) {
    const size_t N = 1024; // Small size for demonstration
    srand((unsigned int)time(NULL));

    // 1. Prepare Data
    float *original = (float *)malloc(N * sizeof(float));
    float *recovered = (float *)malloc(N * sizeof(float));
    if (!original || !recovered) return 1;

    generate_data(original, N);

    printf("Original Data (First 5): %f %f %f %f %f\n", 
           original[0], original[1], original[2], original[3], original[4]);

    // 2. Compress using BitSqueeze
    bitsqueeze_buffer_t *buf = NULL;
    int c_res = bsq_compress_1d(original, N, MXFP4, &buf, NULL);

    if (c_res != 0 || !buf) {
        fprintf(stderr, "Compression failed!\n");
        free(original); free(recovered);
        return 1;
    }

    // 3. Check Compression Ratio
    int64_t packed_size = bsq_get_packed_size(buf);
    int64_t raw_size = N * sizeof(float);
    printf("\n--- Stats ---\n");
    printf("Raw Size:    %lld bytes\n", (long long)raw_size);
    printf("Packed Size: %lld bytes\n", (long long)packed_size);
    printf("Ratio:       %.2fx\n", (float)raw_size / (float)packed_size);

    // 4. Decompress
    int d_res = bsq_decompress(buf, recovered, N);
    if (d_res != 0) {
        fprintf(stderr, "Decompression failed!\n");
        bsq_free(buf); free(original); free(recovered);
        return 1;
    }

    printf("\nRecovered Data (First 5): %f %f %f %f %f\n", 
           recovered[0], recovered[1], recovered[2], recovered[3], recovered[4]);

    // 5. Cleanup
    bsq_free(buf);
    free(original);
    free(recovered);

    return 0;
}