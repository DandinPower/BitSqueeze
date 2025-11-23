#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "bitsqueeze.h"
#include "utils/random.h"

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
        if (bsq_compress_1d(inputs[k], N, FP16, &buf) || !buf) {
            fprintf(stderr, "fp16 compress failed on array %lu\n", k);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *deq = (float *)malloc(N * sizeof(float));
        if (!deq) {
            fprintf(stderr, "malloc failed for fp16 dequant buffer\n");
            bsq_free(buf);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        if (bsq_decompress(buf, deq, N)) {
            fprintf(stderr, "fp16 decompress failed on array %lu\n", k);
            free(deq);
            bsq_free(buf);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double mae, mse, maxabs;
        measure_metrics(inputs[k], deq, N, &mae, &mse, &maxabs);

        double size_kb = bsq_get_packed_size(buf) / 1024.0;
        double bw = 8.0 * size_kb * 1024.0 / (double)N;

        printf("[array %lu] N=%lu, original_size=%.3f KB\n",
               k, N, N * sizeof(float) / 1024.0);
        printf("   FP16: size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               size_kb, bw, mae, mse, maxabs);

        free(deq);
        bsq_free(buf);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
