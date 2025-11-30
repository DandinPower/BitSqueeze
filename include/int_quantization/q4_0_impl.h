#ifndef Q4_0_IMPL_H
#define Q4_0_IMPL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The setting is refer to https://huggingface.co/docs/hub/en/gguf */
#define DEFAULT_Q4_0_BLOCK_SIZE 32
#define DEFAULT_Q4_K_SUPER_BLOCK_SIZE 8

typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of blocks (for block‑wised formats) */
    uint64_t block_size;     /* elements per block */
    float  *scales;          /* length = num_blocks (or num_superblocks for kquant formats) */
    int8_t *data;            /* for kquant, here need to contain quantized scale value + quantized value, otherwise it only need to store quantized value*/
} q4_0_array_t;

q4_0_array_t *allocate_q4_0_array(uint64_t num_elements,
                                       uint64_t block_size);                                       

void free_q4_0_array(q4_0_array_t *q4_0_array);

int64_t get_q4_0_array_size(const q4_0_array_t *q4_0_array);

q4_0_array_t *load_q4_0_array_from_buffer(const void *buffer, int64_t buffer_size);

int q4_0_compress(const float *float_array,
             uint64_t num_elements,
             uint8_t quantized_type,
             q4_0_array_t **q4_0_array);

int q4_0_decompress(const q4_0_array_t *q4_0_array,
               float *float_array);

#ifdef __cplusplus
}
#endif

#endif
