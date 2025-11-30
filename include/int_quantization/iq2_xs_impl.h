#ifndef IQ2_XS_IMPL_H
#define IQ2_XS_IMPL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IQ2_XS_SUPER_BLOCK_SIZE 256

/**
 * @brief IQ2_XS quantized array structure (2.3125 bpw)
 * 
 * Each super block of 256 values is encoded as:
 * - 2 bytes: fp16 block scale
 * - 64 bytes: quantized data (32 × uint16_t, each with 9-bit grid index + 7-bit signs)
 * - 8 bytes: group scales (8 groups × 2 nibbles for 16 sub-scales)
 * 
 * Total: 74 bytes per 256 values = 2.3125 bpw
 */
typedef struct {
    uint64_t num_elements;
    uint64_t num_super_blocks;
    uint16_t *d;            /* fp16 block scales, length = num_super_blocks */
    uint16_t *qs;           /* 32 uint16_t per super block (grid index + signs) */
    uint8_t  *scales;       /* 8 bytes per super block (group scales) */
} iq2_xs_array_t;

/**
 * @brief Initialize IQ2_XS quantization tables (must be called before quantization)
 */
void iq2_xs_init(void);

/**
 * @brief Free IQ2_XS quantization tables
 */
void iq2_xs_free_tables(void);

iq2_xs_array_t *allocate_iq2_xs_array(uint64_t num_elements);

void free_iq2_xs_array(iq2_xs_array_t *arr);

int64_t get_iq2_xs_array_size(const iq2_xs_array_t *arr);

iq2_xs_array_t *load_iq2_xs_array_from_buffer(const void *buffer, int64_t buffer_size);

int iq2_xs_compress(const float *float_array,
                    uint64_t num_elements,
                    iq2_xs_array_t **out);

int iq2_xs_decompress(const iq2_xs_array_t *arr,
                      float *float_array);

#ifdef __cplusplus
}
#endif

#endif
