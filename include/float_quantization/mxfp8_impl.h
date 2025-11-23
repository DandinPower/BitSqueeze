#ifndef MXFP8_IMPL_H
#define MXFP8_IMPL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MXFP8_BLOCK_SIZE 32
#define MXFP8_MAX_NORM_VALUE     448.0f

typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of 32-value blocks */
    uint64_t block_size;     /* elements per block (default = 32) */
    int8_t  *scales;         /* per-block scale exponent (power-of-two), length = num_blocks */
    uint8_t *data;           /* FP8 E4M3 payload, length = num_elements */
} mxfp8_array_t;

mxfp8_array_t *allocate_mxfp8_array(uint64_t num_elements,
                                    uint64_t block_size);

void free_mxfp8_array(mxfp8_array_t *mxfp8_array);

int64_t get_mxfp8_array_size(const mxfp8_array_t *mxfp8_array);

mxfp8_array_t *load_mxfp8_array_from_buffer(const void *buffer, int64_t buffer_size);

int mxfp8_compress(const float *float_array,
                   uint64_t num_elements,
                   mxfp8_array_t **mxfp8_array);

int mxfp8_decompress(const mxfp8_array_t *mxfp8_array,
                     float *float_array);

#endif
