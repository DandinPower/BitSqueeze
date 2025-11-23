#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "bitsqueeze.h"
#include "utils/random.h"

static void measure_metrics(const float *orig, const float *decomp, uint64_t N,
                            double *mae, double *mse, double *max_abs) {
    double m = 0.0, s = 0.0, mx = 0.0;
    for (uint64_t i = 0; i < N; ++i) {
        double e = (double)decomp[i] - (double)orig[i];
        double ae = fabs(e);
        m   += ae;
        s   += e * e;
        if (ae > mx) mx = ae;
    }
    *mae     = m / (double)N;
    *mse     = s / (double)N;
    *max_abs = mx;
}

int main(void) {
    const uint64_t X              = 5;              /* number of random arrays            */
    const uint16_t NUM_TOKENS     = 512;            /* rows in 2D shape                   */
    const uint16_t NUM_FEATURES   = 8192;           /* columns in 2D shape                */
    const uint64_t N              = (uint64_t)NUM_TOKENS * NUM_FEATURES;
    const float  MINV             = -10.0f;
    const float  MAXV             =  10.0f;
    const unsigned int SEED       = 12345;          /* deterministic seed                 */
    const float  SPARSE_RATIOS[]  = {0.2f, 0.1f};   /* sparsity levels to evaluate        */
    const size_t NUM_RATIOS       = sizeof(SPARSE_RATIOS) / sizeof(SPARSE_RATIOS[0]);

    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    for (uint64_t k = 0; k < X; ++k) {
        printf("[array %lu] N=%lu (tokens=%u, features=%u), original_size=%.3f KB\n",
               k, N, NUM_TOKENS, NUM_FEATURES, N * sizeof(float) / 1024.0);

        for (size_t r = 0; r < NUM_RATIOS; ++r) {
            const float sparse_ratio = SPARSE_RATIOS[r];

            bitsqueeze_buffer_t *buf = NULL;
            if (bsq_compress_2d(inputs[k], NUM_TOKENS, NUM_FEATURES, sparse_ratio, TOPK, &buf) || !buf) {
                fprintf(stderr, "TOPK compress failed for array %lu, ratio %.2f\n", k, sparse_ratio);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            float *decomp = malloc(N * sizeof(float));
            if (!decomp) {
                fprintf(stderr, "malloc failed for decomp buffer (array %lu, ratio %.2f)\n", k, sparse_ratio);
                bsq_free(buf);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            if (bsq_decompress(buf, decomp, N)) {
                fprintf(stderr, "TOPK decompress failed for array %lu, ratio %.2f\n", k, sparse_ratio);
                free(decomp);
                bsq_free(buf);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            double mae, mse, maxabs;
            measure_metrics(inputs[k], decomp, N, &mae, &mse, &maxabs);

            double size_kb = bsq_get_packed_size(buf) / 1024.0;
            double bw = 8.0 * size_kb * 1024.0 / (double)N;
            const sparse_array_t *arr = (const sparse_array_t *)buf->payload;
            double sparsity_ratio_actual = (double)arr->num_sparse_features / (double)arr->num_features;

            printf("   TOPK%.2f: sparsity=%.3f, size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
                   sparse_ratio, sparsity_ratio_actual, size_kb, bw, mae, mse, maxabs);

            free(decomp);
            bsq_free(buf);
        }
        printf("\n");
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
