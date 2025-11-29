#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "bitsqueeze.h"
#include "utils/random.h"
#include "utils/evaluation.h"
#include <inttypes.h>

int main(void) {
    const uint64_t X   = 5;            /* number of random arrays            */
    const uint64_t N   = 4194304;      /* length of each array               */
    const float  MINV  = -10.0f;
    const float  MAXV  =  10.0f;
    const unsigned int SEED = 12345;

    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    for (uint64_t k = 0; k < X; ++k) {
        bitsqueeze_buffer_t *buf = NULL;

        double t0 = get_time_ms();
        int c_res = bsq_compress_1d(inputs[k], N, BF16, &buf);
        double t1 = get_time_ms();
        double comp_time = t1 - t0;

        if (c_res || !buf) {
            fprintf(stderr, "bf16 compress failed on array %" PRIu64 "\n", k);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *deq = (float *)malloc(N * sizeof(float));
        if (!deq) {
            fprintf(stderr, "malloc failed for bf16 dequant buffer\n");
            bsq_free(buf);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        // PROFILE DECOMPRESS
        double t2 = get_time_ms();
        int d_res = bsq_decompress(buf, deq, N);
        double t3 = get_time_ms();
        double decomp_time = t3 - t2;

        if (d_res) {
            fprintf(stderr, "bf16 decompress failed on array %" PRIu64 "\n", k);
            free(deq);
            bsq_free(buf);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double mae, mse, maxabs;
        measure_metrics(inputs[k], deq, N, &mae, &mse, &maxabs);

        double size_kb = bsq_get_packed_size(buf) / 1024.0;
        double bw = 8.0 * size_kb * 1024.0 / (double)N;

        printf("[array %" PRIu64 "] N=%" PRIu64 ", original_size=%.3f KB\n",
               k, N, N * sizeof(float) / 1024.0);
        printf("   BF16: size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               size_kb, bw, mae, mse, maxabs);
        printf("         CompTime=%.3f ms, DecompTime=%.3f ms\n", comp_time, decomp_time);

        free(deq);
        bsq_free(buf);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
