#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "sparsity.h"
#include "random.h"

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
    /* ---- configuration --------------------------------------------------- */
    const uint64_t X              = 10;            /* number of random arrays            */
    const uint16_t NUM_TOKENS     = 512;            /* rows in 2D shape                    */
    const uint16_t NUM_FEATURES   = 8192;           /* columns in 2D shape                 */
    const uint64_t N              = (uint64_t)NUM_TOKENS * NUM_FEATURES;  /* total elements (flattened) */
    const float  MINV             = -10.0f;
    const float  MAXV             =  10.0f;
    const unsigned int SEED       = 12345;         /* deterministic seed                  */
    const float  SPARSE_RATIOS[]  = {0.2f, 0.1f};/* two sparsity levels to evaluate     */
    const size_t NUM_RATIOS       = sizeof(SPARSE_RATIOS) / sizeof(SPARSE_RATIOS[0]);

    /* ---- generate random inputs ----------------------------------------- */
    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    /* ---- loop over each array ------------------------------------------- */
    for (uint64_t k = 0; k < X; ++k) {
        printf("[array %lu] N=%lu (tokens=%u, features=%u), original_size=%.3f KB\n",
               k, N, NUM_TOKENS, NUM_FEATURES, N * sizeof(float) / 1024.0);

        /* ---- loop over sparsity ratios ---------------------------------- */
        for (size_t r = 0; r < NUM_RATIOS; ++r) {
            const float sparse_ratio = SPARSE_RATIOS[r];

            sparse_array_t *sparse_array = NULL;
            /* ---- compress ------------------------------------------------- */
            if (compress(inputs[k], NUM_TOKENS, NUM_FEATURES, sparse_ratio, &sparse_array)) {
                fprintf(stderr, "compress failed for array %lu, ratio %.2f\n", k, sparse_ratio);
                free_sparse_array(sparse_array);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            /* ---- decompress ----------------------------------------------- */
            float *decomp = malloc(N * sizeof(float));
            if (!decomp) {
                fprintf(stderr, "malloc failed for decomp buffer (array %lu, ratio %.2f)\n", k, sparse_ratio);
                free_sparse_array(sparse_array);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            if (decompress(sparse_array, decomp)) {
                fprintf(stderr, "decompress failed for array %lu, ratio %.2f\n", k, sparse_ratio);
                free(decomp);
                free_sparse_array(sparse_array);
                free_random_float_arrays(inputs, X);
                return EXIT_FAILURE;
            }

            /* ---- metrics --------------------------------------------------- */
            double mae, mse, maxabs;
            measure_metrics(inputs[k], decomp, N, &mae, &mse, &maxabs);

            /* ---- report ---------------------------------------------------- */
            double sparsity_ratio_actual = (double)sparse_array->num_sparse_features / (double)sparse_array->num_features;
            double size_sparse_kb = get_sparse_array_size(sparse_array) / 1024.0;
            double bw = 8.0 * size_sparse_kb * 1024.0 / (double)N;  /* bits per original element */

            printf("   Sparse%.2f: sparsity=%.3f, size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
                   sparse_ratio, sparsity_ratio_actual, size_sparse_kb, bw, mae, mse, maxabs);

            /* ---- clean ----------------------------------------------------- */
            free(decomp);
            free_sparse_array(sparse_array);
        }
        printf("\n");  // Spacer between arrays
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
