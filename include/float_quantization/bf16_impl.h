#ifndef BF16_IMPL_H
#define BF16_IMPL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "datatype/bf16.h"

typedef struct {
    uint64_t  num_elements;
    uint16_t *data;         /* BF16 payload, length = num_elements */
} bf16_array_t;

bf16_array_t *allocate_bf16_array(uint64_t num_elements);

void free_bf16_array(bf16_array_t *bf16_array);

int64_t get_bf16_array_size(const bf16_array_t *bf16_array);

bf16_array_t *load_bf16_array_from_buffer(const void *buffer, int64_t buffer_size);

int bf16_compress(const float *float_array,
                  uint64_t num_elements,
                  bf16_array_t **bf16_array);

int bf16_decompress(const bf16_array_t *bf16_array,
                    float *float_array);

#endif
