#ifndef BITSQUEEZE_H
#define BITSQUEEZE_H

#include <stdint.h>
#include "float_quantization/bf16_impl.h"
#include "float_quantization/fp16_impl.h"
#include "float_quantization/fp8_impl.h"
#include "float_quantization/fp4_impl.h"
#include "float_quantization/mxfp8_impl.h"
#include "float_quantization/mxfp4_impl.h"
#include "float_quantization/nvfp4_impl.h"
#include "float_quantization/nf4_impl.h"
#include "float_quantization/nf4_dq_impl.h"
#include "int_quantization/q8_0_impl.h"
#include "int_quantization/q4_0_impl.h"
#include "int_quantization/q2_k_impl.h"
#include "int_quantization/iq2_xxs_impl.h"
#include "int_quantization/iq2_xs_impl.h"
#include "int_quantization/iq2_s_impl.h"
#include "sparsity/topk_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Q8_0 = 0,
    Q4_0 = 1,
    Q2_K = 2,
    TOPK = 3,
    BF16 = 4,
    FP16 = 5,
    FP8  = 6,
    FP4  = 7,
    MXFP8 = 8,
    MXFP4 = 9,
    NVFP4 = 10,
    NF4_DQ = 11,
    NF4 = 12,
    IQ2_XXS = 13,
    IQ2_XS = 14,
    IQ2_S = 15,
} bsq_method_t;

typedef struct {
    uint64_t num_elements;    /* for 1D formats */
    uint16_t num_tokens;      /* for 2D sparsity */
    uint16_t num_features;    /* for 2D sparsity */
    float    sparse_ratio;    /* only meaningful for TOPK */
} bsq_shape_t;

typedef struct bitsqueeze_buffer {
    bsq_method_t method;
    bsq_shape_t  shape;
    void        *payload;
} bitsqueeze_buffer_t;

int bsq_compress_1d(const float *src,
                    uint64_t num_elements,
                    bsq_method_t method,
                    bitsqueeze_buffer_t **out);

int bsq_compress_2d(const float *src,
                    uint16_t num_tokens,
                    uint16_t num_features,
                    float sparse_ratio,
                    bsq_method_t method,
                    bitsqueeze_buffer_t **out);

int bsq_decompress(const bitsqueeze_buffer_t *buf,
                   float *dst,
                   uint64_t dst_num_elements);

int64_t bsq_get_packed_size(const bitsqueeze_buffer_t *buf);

bitsqueeze_buffer_t *load_bsq_from_buffer(const void *buffer, int64_t buffer_size);

void bsq_free(bitsqueeze_buffer_t *buf);

#ifdef __cplusplus
}
#endif

#endif
