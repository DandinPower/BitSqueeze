#ifndef NVFP4_IMPL_H
#define NVFP4_IMPL_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_NVFP4_BLOCK_SIZE 16
#define NVFP4_MAX_NORM_VALUE     6.0f
#define NVFP4_FP8_MAX_NORM       448.0f

typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of 16-value blocks */
    uint64_t block_size;     /* elements per block (default = 16) */
    float    tensor_scale;   /* FP32 per-tensor scale */
    uint8_t *block_scales;   /* FP8 E4M3 per-block scale codes, length = num_blocks */
    uint8_t *data;           /* packed FP4 E2M1 payload, length = ceil(num_elements / 2) bytes */
} nvfp4_array_t;

nvfp4_array_t *allocate_nvfp4_array(uint64_t num_elements,
                                    uint64_t block_size);

void free_nvfp4_array(nvfp4_array_t *nvfp4_array);

int64_t get_nvfp4_array_size(const nvfp4_array_t *nvfp4_array);

nvfp4_array_t *load_nvfp4_array_from_buffer(const void *buffer, int64_t buffer_size);

int nvfp4_compress(const float *float_array,
                   uint64_t num_elements,
                   nvfp4_array_t **nvfp4_array);

int nvfp4_decompress(const nvfp4_array_t *nvfp4_array,
                     float *float_array);

#endif
