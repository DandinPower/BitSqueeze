#include "sparsity/topk_im_impl.h"

typedef struct {
    float im_val;    // importance key
    float val;        // original value
    uint16_t idx;     // original feature index
} heap_entry_t;

static inline float importance_key(float v) {
    return (v == v) ? v : -INFINITY;   // treat NaN as least important
}

static inline int is_worse(heap_entry_t a, heap_entry_t b) {
    return a.im_val < b.im_val;
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

int topk_im_compress(const float *float_array, const float *importance_array, uint16_t num_tokens, uint16_t num_features,  float sparse_ratio, sparse_array_t **sparse_array) {
    if (!float_array || !sparse_array || !importance_array) return 1;
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
            const float *im = importance_array + dense_base;

            for (uint16_t i = 0; i < K; ++i) {
                heap[i].idx = i;
                heap[i].val = x[i];
                heap[i].im_val = importance_key(im[i]);
            }

            heapify_min(heap, K);

            for (uint16_t i = K; i < F; ++i) {
                float v = x[i];
                float im_v = importance_key(im[i]);
                if (im_v > heap[0].im_val) {
                    heap[0].idx = i;
                    heap[0].val = v;
                    heap[0].im_val = im_v;
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

int topk_im_decompress(const sparse_array_t *sparse_array, float *float_array) {
    if (!float_array || !sparse_array) return 1;

    uint32_t num_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_features;
    memset(float_array, 0, num_elements * sizeof(float));

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
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

int topk_im_apply(const sparse_array_t *sparse_array, float *float_array) {
    if (!float_array || !sparse_array) return 1;

#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
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
