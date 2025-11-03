#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "k_quantization.h"
#include "random.h"

static void measure_metrics(const float *orig, const float *deq, uint64_t N,
                            double *mae, double *mse, double *max_abs) {
    double m = 0.0, s = 0.0, mx = 0.0;
    for (uint64_t i = 0; i < N; ++i) {
        double e = (double)deq[i] - (double)orig[i];
        double ae = fabs(e);
        m    += ae;
        s    += e * e;
        if (ae > mx) mx = ae;
    }
    *mae     = m / (double)N;
    *mse     = s / (double)N;
    *max_abs = mx;
}

int main(void)
{
    /* ---- configuration --------------------------------------------------- */
    const uint64_t X    = 10;           /* number of random arrays            */
    // N must be a multiple of WEIGHT_PER_SUPER_BLOCK (256)
    const uint64_t N    = 4194304;      /* length of each array (2^22, mult of 256) */
    const float  MINV = -10.0f;
    const float  MAXV =  10.0f;
    const unsigned int SEED = 12345;    /* deterministic seed                 */

    if (N % WEIGHT_PER_SUPER_BLOCK != 0) {
        fprintf(stderr, "Test error: N (%lu) must be a multiple of %d\n",
                (unsigned long)N, WEIGHT_PER_SUPER_BLOCK);
        return EXIT_FAILURE;
    }

    /* ---- generate random inputs ----------------------------------------- */
    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    printf("Testing Q2_K Quantization\n");
    printf("Super-block size: %d weights\n", WEIGHT_PER_SUPER_BLOCK);
    printf("Block size: %d weights\n", Q2_K_BLOCK_SIZE);
    printf("Blocks per super-block: %d\n\n", Q2_K_SUPER_BLOCK_SIZE);


    /* ---- loop over each array ------------------------------------------- */
    for (uint64_t k = 0; k < X; ++k) {
        /* ---- q2_k ------------------------------------------------------- */
        quantized_array_q2_k_t *qk = NULL;
        if (k_quantize(inputs[k], N, &qk) || !qk) {
            fprintf(stderr, "q2_k quantisation failed on array %lu\n", k);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *y_k = (float *)malloc(N * sizeof(float));
        if (!y_k) {
            fprintf(stderr, "malloc failed for q2_k dequant buffer\n");
            free_quantized_q2_k_array(qk);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        if (k_dequantize(qk, y_k)) {
            fprintf(stderr, "q2_k dequantisation failed on array %lu\n", k);
            free(y_k);
            free_quantized_q2_k_array(qk);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double mae_k, mse_k, maxabs_k;
        measure_metrics(inputs[k], y_k, N, &mae_k, &mse_k, &maxabs_k);

        /* ---- report ------------------------------------------------------ */
        double size_k_kb = get_quantized_q2_k_array_size(qk) / 1024.0;
        double bw_k = 8.0 * size_k_kb * 1024.0 / (double)N;

        printf("[array %lu] N=%lu, super_blocks=%u, original_size=%.3f KB\n",
               k, N, qk->num_super_blocks, N * sizeof(float) / 1024.0);
        printf("    Q2_K:  size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               size_k_kb, bw_k, mae_k, mse_k, maxabs_k);

        /* ---- clean ------------------------------------------------------- */
        free(y_k);
        free_quantized_q2_k_array(qk);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
