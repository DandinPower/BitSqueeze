#include "utils/random.h"

float **gen_random_float_arrays(uint64_t count,
                                uint64_t N,
                                float minv,
                                float maxv,
                                unsigned int seed) {
    if (!count || !N || !isfinite(minv) || !isfinite(maxv) || maxv < minv)
        return NULL;

    if (!seed) seed = (unsigned int)time(NULL);
    srand(seed);

    float **arrs = calloc(count, sizeof(*arrs));
    if (!arrs) return NULL;

    for (uint64_t i = 0; i < count; ++i) {
        arrs[i] = malloc(N * sizeof(float));
        if (!arrs[i]) {
            for (uint64_t j = 0; j < i; ++j) free(arrs[j]);
            free(arrs);
            return NULL;
        }
        for (uint64_t j = 0; j < N; ++j) {
            float u = (float)rand() / (float)RAND_MAX;
            arrs[i][j] = minv + u * (maxv - minv);
        }
    }
    return arrs;
}

void free_random_float_arrays(float **arrs, uint64_t count) {
    if (!arrs) return;
    for (uint64_t i = 0; i < count; ++i) free(arrs[i]);
    free(arrs);
}
