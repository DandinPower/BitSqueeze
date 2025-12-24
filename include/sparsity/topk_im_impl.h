#ifndef TOPK_IM_IMPL_H
#define TOPK_IM_IMPL_H

#include "sparsity/topk_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Given a 2D float array of size num_tokens by num_features, and a 2D importance array of size num_tokens by num_features that holds the importance score for the corresponding indexed values in the float array, use this information to find the top k values, where k is determined by spase_ratio multiplied by num_features, since the top k is selected per token, and then wrap everything inside sparse_array. */
int topk_im_compress(const float *float_array, const float *importance_array, uint16_t num_tokens, uint16_t num_features,  float sparse_ratio, sparse_array_t **sparse_array);

/* Given a sparse_array, recover the original 2D float array by filling the zero values with sparse values, this should be identical to topk_decompress. */
int topk_im_decompress(const sparse_array_t *sparse_array, float *float_array);

/* Given a sparse_array, apply the changes to the given float_array by recording sparse values, this use case appears when the sparse_array contains sparse importance values with higher precision and needs to apply them to a low precision quantized recovered float_array. */
int topk_im_apply(const sparse_array_t *sparse_array, float *float_array);

#ifdef __cplusplus
}
#endif

#endif
