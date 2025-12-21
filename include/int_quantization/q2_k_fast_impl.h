#ifndef Q2_K_FAST_IMPL_H
#define Q2_K_FAST_IMPL_H

#include "int_quantization/q2_k_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fast/min-max Q2_K variant. */
int q2_k_fast_compress(const float *float_array,
                       uint64_t num_elements,
                       q2_k_array_t **q2_k_array);

int q2_k_fast_decompress(const q2_k_array_t *q2_k_array,
                         float *float_array);

#ifdef __cplusplus
}
#endif

#endif
