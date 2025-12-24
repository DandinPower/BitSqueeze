#include "sparsity/topk_impl.h"

sparse_array_t *allocate_sparse_array(uint16_t num_tokens, uint16_t num_features, float sparse_ratio) {
    if (!num_tokens || !num_features) return NULL;
    if (sparse_ratio < 0.0f || sparse_ratio > 1.0f) return NULL;
    
    float raw_sparse = (float)num_features * sparse_ratio;
    uint16_t num_sparse_features = (uint16_t)roundf(raw_sparse);
    
    // clamp to valid range
    if (num_sparse_features > num_features) {
        num_sparse_features = num_features;
    } else if (num_sparse_features == 0 && sparse_ratio > 0.0f) {
        num_sparse_features = 1;  // Avoid total sparsity if ratio positive;
    }

    uint32_t sparse_elements = (uint32_t)num_tokens * num_sparse_features;
    uint64_t total = sizeof(sparse_array_t) + sparse_elements * (sizeof(float) + sizeof(uint16_t));
    sparse_array_t *sparse_array = (sparse_array_t*)calloc(1, total);
    if (!sparse_array) return NULL;

    /* initialise the header fields */
    sparse_array->num_tokens = num_tokens;
    sparse_array->num_features = num_features;
    sparse_array->num_sparse_features = num_sparse_features;
    sparse_array->sparse_indices = (uint16_t*)(sparse_array + 1);    /* just after the header */
    sparse_array->values = (float*)(sparse_array->sparse_indices + sparse_elements);     /* after the sparse_indices */

    return sparse_array;
}                          

void free_sparse_array(sparse_array_t *sparse_array) {
    if (!sparse_array) return;
    free(sparse_array);
}

uint64_t get_sparse_array_size(const sparse_array_t *sparse_array) {
    if (!sparse_array) return 0;

    uint32_t sparse_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_sparse_features;
    
    return sizeof(sparse_array_t) + sparse_elements * (sizeof(float) + sizeof(uint16_t));
}

sparse_array_t *load_sparse_array_from_buffer(const void *buffer, uint64_t buffer_size) {
    sparse_array_t *sparse_array = (sparse_array_t*)calloc(1, buffer_size);
    if (!sparse_array) return NULL;
    
    memcpy(sparse_array, buffer, buffer_size);

    uint32_t sparse_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_sparse_features;

    sparse_array->sparse_indices   = (uint16_t*)(sparse_array + 1);
    sparse_array->values = (float*)(sparse_array->sparse_indices + sparse_elements);

    return sparse_array;
}

typedef struct {
    float abs_val;    // importance key = fabsf(val) with NaN policy
    float val;        // original value
    uint16_t idx;     // original feature index
} heap_entry_t;

static inline float importance_abs(float v) {
    float a = fabsf(v);
    return (a == a) ? a : -INFINITY;   // treat NaN as least important
}

static inline int is_worse(heap_entry_t a, heap_entry_t b) {
    return a.abs_val < b.abs_val;
}

static inline void sift_down_min(heap_entry_t *h, uint16_t K, uint16_t p) {
    while (1) {
        uint32_t left = (uint32_t)2 * p + 1;
        if (left >= K) break;

        uint32_t right = left + 1;
        uint16_t c = (uint16_t)left;

        if (right < K && is_worse(h[(uint16_t)right], h[c])) {
            c = (uint16_t)right;
        }

        if (is_worse(h[c], h[p])) {
            heap_entry_t tmp = h[p];
            h[p] = h[c];
            h[c] = tmp;
            p = c;
        } else {
            break;
        }
    }
}

static inline void heapify_min(heap_entry_t *h, uint16_t K) {
    if (K <= 1) return;
    for (int32_t p = (int32_t)(K / 2) - 1; p >= 0; --p) {
        sift_down_min(h, K, (uint16_t)p);
    }
}

int topk_compress(const float *float_array,
                  uint16_t num_tokens,
                  uint16_t num_features,
                  float sparse_ratio,
                  sparse_array_t **sparse_array) {
    if (!float_array || !sparse_array) return 1;
    if (num_tokens == 0 || num_features == 0) return 1;
    if (*sparse_array) return 1;

    *sparse_array = allocate_sparse_array(num_tokens, num_features, sparse_ratio);
    if (!*sparse_array) return 1;

    sparse_array_t *sa = *sparse_array;
    const uint16_t K = sa->num_sparse_features;
    const uint16_t F = num_features;
    if (K == 0) return 0;

    int alloc_error = 0;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel
    {
#endif
        heap_entry_t *heap = (heap_entry_t *)malloc((size_t)K * sizeof(heap_entry_t));
        if (!heap) {
#if defined(__linux__) && defined(_OPENMP)
#pragma omp critical
#endif
            { alloc_error = 1; }
        }
#if defined(__linux__) && defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int t = 0; t < (int)num_tokens; ++t) {
            if (!heap) continue; // this thread cannot do work

            const uint32_t dense_base  = (uint32_t)t * (uint32_t)F;
            const uint32_t sparse_base = (uint32_t)t * (uint32_t)K;
            const float *x = float_array + dense_base;

            for (uint16_t i = 0; i < K; ++i) {
                float v = x[i];
                heap[i].idx = i;
                heap[i].val = v;
                heap[i].abs_val = importance_abs(v);
            }

            heapify_min(heap, K);

            for (uint16_t i = K; i < F; ++i) {
                float v = x[i];
                float a = importance_abs(v);
                if (a > heap[0].abs_val) {
                    heap[0].idx = i;
                    heap[0].val = v;
                    heap[0].abs_val = a;
                    sift_down_min(heap, K, 0);
                }
            }

            for (uint16_t j = 0; j < K; ++j) {
                sa->sparse_indices[sparse_base + j] = heap[j].idx;
                sa->values[sparse_base + j] = heap[j].val;
            }
        }

        free(heap);
#if defined(__linux__) && defined(_OPENMP)
    }
#endif

    if (alloc_error) {
        free_sparse_array(*sparse_array);
        *sparse_array = NULL;
        return 1;
    }

    return 0;
}

int topk_decompress(const sparse_array_t *sparse_array, float *float_array) {
    if (!float_array || !sparse_array) return 1;

    uint32_t num_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_features;
    memset(float_array, 0, num_elements * sizeof(float));

    for (uint16_t cur_token_index = 0; cur_token_index < sparse_array->num_tokens; cur_token_index++) {
        uint32_t dense_base = (uint32_t)cur_token_index * sparse_array->num_features;
        uint32_t sparse_base = (uint32_t)cur_token_index * sparse_array->num_sparse_features;

        for (uint16_t keep_feature_index = 0; keep_feature_index < sparse_array->num_sparse_features; keep_feature_index++) {
            uint16_t original_feature_index = sparse_array->sparse_indices[sparse_base + keep_feature_index];
            float_array[dense_base + original_feature_index] = sparse_array->values[sparse_base + keep_feature_index];
        }
    }

    return 0;
}
