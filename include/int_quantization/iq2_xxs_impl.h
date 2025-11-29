#ifndef IQ2_XXS_IMPL_H
#define IQ2_XXS_IMPL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "datatype/fp16/fp16.h"

#define IQ2_XXS_SUPER_BLOCK_SIZE 256

/**
 * @brief IQ2_XXS quantized array structure
 * 
 * Each super block of 256 values is encoded as:
 * - 2 bytes: fp16 block scale
 * - 64 bytes: quantized data (8 groups × 8 bytes)
 * 
 * Per-group (32 values) encoding in 8 bytes:
 * - bytes 0-3: 4 grid indices (8 bits each, indexing 256-entry grid)
 * - bytes 4-7: 4×7-bit sign patterns (bits 0-27) + 4-bit group scale (bits 28-31)
 * 
 * Effective: 2.0625 bits per weight
 */
typedef struct {
    uint64_t num_elements;
    uint64_t num_super_blocks;
    uint16_t *scales;       /* fp16 block scales, length = num_super_blocks */
    uint8_t  *qs;           /* quantized data, 64 bytes per super block */
} iq2_xxs_array_t;

/**
 * @brief Initialize IQ2 quantization tables (must be called before quantization)
 */
void iq2_xxs_init(void);

/**
 * @brief Free IQ2 quantization tables
 */
void iq2_xxs_free_tables(void);

iq2_xxs_array_t *allocate_iq2_xxs_array(uint64_t num_elements);

void free_iq2_xxs_array(iq2_xxs_array_t *arr);

int64_t get_iq2_xxs_array_size(const iq2_xxs_array_t *arr);

iq2_xxs_array_t *load_iq2_xxs_array_from_buffer(const void *buffer, int64_t buffer_size);

int iq2_xxs_compress(const float *float_array,
                     uint64_t num_elements,
                     iq2_xxs_array_t **out);

int iq2_xxs_decompress(const iq2_xxs_array_t *arr,
                       float *float_array);

#endif
