#ifndef FP4_IMPL_H
#define FP4_IMPL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FP4_MAX_NORM_VALUE 6.0f

typedef struct {
    uint64_t num_elements;
    float    scale;  /* FP32 tensor scale */
    uint8_t *data;   /* packed FP4 E2M1 payload, length = ceil(num_elements / 2) bytes */
} fp4_array_t;

fp4_array_t *allocate_fp4_array(uint64_t num_elements);

void free_fp4_array(fp4_array_t *fp4_array);

int64_t get_fp4_array_size(const fp4_array_t *fp4_array);

fp4_array_t *load_fp4_array_from_buffer(const void *buffer, int64_t buffer_size);

int fp4_compress(const float *float_array,
                 uint64_t num_elements,
                 fp4_array_t **fp4_array);

int fp4_decompress(const fp4_array_t *fp4_array,
                   float *float_array);

#ifdef __cplusplus
}
#endif

#endif
