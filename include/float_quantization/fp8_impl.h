#ifndef FP8_IMPL_H
#define FP8_IMPL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FP8_MAX_NORM_VALUE 448.0f

typedef struct {
    uint64_t num_elements;
    float    scale;  /* FP32 tensor scale */
    uint8_t *data;   /* FP8 E4M3 payload, length = num_elements */
} fp8_array_t;

fp8_array_t *allocate_fp8_array(uint64_t num_elements);

void free_fp8_array(fp8_array_t *fp8_array);

int64_t get_fp8_array_size(const fp8_array_t *fp8_array);

fp8_array_t *load_fp8_array_from_buffer(const void *buffer, int64_t buffer_size);

int fp8_compress(const float *float_array,
                 uint64_t num_elements,
                 fp8_array_t **fp8_array);

int fp8_decompress(const fp8_array_t *fp8_array,
                   float *float_array);

#ifdef __cplusplus
}
#endif

#endif
