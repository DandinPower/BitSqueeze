#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "bitsqueeze.h"
#include "utils/random.h"
#include "utils/evaluation.h"
#include <inttypes.h>

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
        printf("[array %" PRIu64 "] N=%" PRIu64 " (tokens=%u, features=%u), original_size=%.3f KB\n",
               k, N, NUM_TOKENS, NUM_FEATURES, N * sizeof(float) / 1024.0);

        for (size_t r = 0; r < NUM_RATIOS; ++r) {
            const float sparse_ratio = SPARSE_RATIOS[r];

            bitsqueeze_buffer_t *buf = NULL;
            double t0 = get_time_ms();
            int c_res = bsq_compress_2d(inputs[k], NUM_TOKENS, NUM_FEATURES, sparse_ratio, TOPK, &buf);
            double t1 = get_time_ms();
            double comp_time = t1 - t0;
            if (c_res || !buf) {
                fprintf(stderr, "TOPK compress failed for array %" PRIu64 ", ratio %.2f\n", k, sparse_ratio);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            float *decomp = (float*)malloc(N * sizeof(float));
            if (!decomp) {
                fprintf(stderr, "malloc failed for decomp buffer (array %" PRIu64 ", ratio %.2f)\n", k, sparse_ratio);
                bsq_free(buf);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            double t2 = get_time_ms();
            int d_res = bsq_decompress(buf, decomp, N);
            double t3 = get_time_ms();
            double decomp_time = t3 - t2;
            if (d_res) {
                fprintf(stderr, "TOPK decompress failed for array %" PRIu64 ", ratio %.2f\n", k, sparse_ratio);
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
            printf("            CompTime=%.3f ms, DecompTime=%.3f ms\n", comp_time, decomp_time);

            free(decomp);
            bsq_free(buf);
        }
        printf("\n");
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
