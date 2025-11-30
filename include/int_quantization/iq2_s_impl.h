#ifndef IQ2_S_IMPL_H
#define IQ2_S_IMPL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IQ2_S_SUPER_BLOCK_SIZE 256

/**
 * @brief IQ2_S quantized array structure (2.5625 bpw)
 * 
 * Each super block of 256 values is encoded as:
 * - 2 bytes: fp16 block scale
 * - 64 bytes: qs (32 bytes grid low bits + 32 bytes sign patterns)
 * - 8 bytes: qh (high 2 bits of grid indices, packed)
 * - 8 bytes: group scales
 * 
 * Total: 82 bytes per 256 values = 2.5625 bpw
 */
typedef struct {
    uint64_t num_elements;
    uint64_t num_super_blocks;
    uint16_t *d;            /* fp16 block scales, length = num_super_blocks */
    uint8_t  *qs;           /* 64 bytes per super block (grid low + signs) */
    uint8_t  *qh;           /* 8 bytes per super block (grid high bits) */
    uint8_t  *scales;       /* 8 bytes per super block (group scales) */
} iq2_s_array_t;

/**
 * @brief Initialize IQ2_S quantization tables (must be called before quantization)
 */
void iq2_s_init(void);

/**
 * @brief Free IQ2_S quantization tables
 */
void iq2_s_free_tables(void);

iq2_s_array_t *allocate_iq2_s_array(uint64_t num_elements);

void free_iq2_s_array(iq2_s_array_t *arr);

int64_t get_iq2_s_array_size(const iq2_s_array_t *arr);

iq2_s_array_t *load_iq2_s_array_from_buffer(const void *buffer, int64_t buffer_size);

int iq2_s_compress(const float *float_array,
                   uint64_t num_elements,
                   iq2_s_array_t **out);

int iq2_s_decompress(const iq2_s_array_t *arr,
                     float *float_array);

#ifdef __cplusplus
}
#endif

#endif
