#ifndef MXFP4_IMPL_H
#define MXFP4_IMPL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MXFP4_BLOCK_SIZE 32
#define MXFP4_MAX_NORM_VALUE     6.0f

typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of 32-value blocks */
    uint64_t block_size;     /* elements per block (default = 32) */
    int8_t  *scales;         /* per-block scale exponent (power-of-two), length = num_blocks */
    uint8_t *data;           /* packed FP4 E2M1 payload, length = ceil(num_elements / 2) bytes */
} mxfp4_array_t;

mxfp4_array_t *allocate_mxfp4_array(uint64_t num_elements,
                                    uint64_t block_size);

void free_mxfp4_array(mxfp4_array_t *mxfp4_array);

int64_t get_mxfp4_array_size(const mxfp4_array_t *mxfp4_array);

mxfp4_array_t *load_mxfp4_array_from_buffer(const void *buffer, int64_t buffer_size);

int mxfp4_compress(const float *float_array,
                   uint64_t num_elements,
                   mxfp4_array_t **mxfp4_array);

int mxfp4_decompress(const mxfp4_array_t *mxfp4_array,
                     float *float_array);

#endif
