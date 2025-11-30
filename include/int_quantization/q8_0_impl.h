#ifndef Q8_0_IMPL_H
#define Q8_0_IMPL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The setting is refer to https://huggingface.co/docs/hub/en/gguf */
#define DEFAULT_Q8_0_BLOCK_SIZE 32

typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of blocks (for block‑wised formats) */
    uint64_t block_size;     /* elements per block */
    float  *scales;          /* length = num_blocks (or num_superblocks for kquant formats) */
    int8_t *data;            /* store quantized value*/
} q8_0_array_t;

q8_0_array_t *allocate_q8_0_array(uint64_t num_elements,
                                       uint64_t block_size);
                                    
void free_q8_0_array(q8_0_array_t *q8_0_array);

int64_t get_q8_0_array(const q8_0_array_t *q8_0_array);

q8_0_array_t *load_quantized_array_from_buffer(const void *buffer, int64_t buffer_size);

int q8_0_compress(const float *float_array,
             uint64_t num_elements,
             q8_0_array_t **q8_0_array);

int q8_0_decompress(const q8_0_array_t *q8_0_array,
               float *float_array);

#ifdef __cplusplus
}
#endif

#endif
