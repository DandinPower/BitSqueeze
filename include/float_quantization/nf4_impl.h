#ifndef NF4_IMPL_H
#define NF4_IMPL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_NF4_BLOCK_SIZE 64

/*
 * NF4 (NormalFloat4) without double quantization of block scales:
 *  - data: 4-bit NF4 codes packed 2 per byte
 *  - per-block FP32 scale stored directly in block_scales
 */
typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of block_size-sized blocks */
    uint64_t block_size;     /* elements per block (default = 64) */
    float   *block_scales;   /* FP32 per-block scales, length = num_blocks */
    uint8_t *data;           /* packed NF4 codes, length = ceil(num_elements / 2) bytes */
} nf4_array_t;

nf4_array_t *allocate_nf4_array(uint64_t num_elements,
                                uint64_t block_size);

void free_nf4_array(nf4_array_t *nf4_array);

int64_t get_nf4_array_size(const nf4_array_t *nf4_array);

nf4_array_t *load_nf4_array_from_buffer(const void *buffer, int64_t buffer_size);

int nf4_compress(const float *float_array,
                 uint64_t num_elements,
                 nf4_array_t **nf4_array);

int nf4_decompress(const nf4_array_t *nf4_array,
                   float *float_array);

#endif
