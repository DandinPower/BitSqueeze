#include "bitsqueeze.h"
#include <string.h>

static bitsqueeze_buffer_t *_allocate_bsq_buffer(size_t payload_size) {
    size_t total = sizeof(bitsqueeze_buffer_t) + payload_size;
    bitsqueeze_buffer_t *buf = (bitsqueeze_buffer_t *)calloc(1, total);
    if (!buf) return NULL;
    buf->payload = ((uint8_t *)buf) + sizeof(bitsqueeze_buffer_t);
    return buf;
}

static void _fixup_payload_pointers(bitsqueeze_buffer_t *buf) {
    if (!buf || !buf->payload) return;

    switch (buf->method) {
        case Q8_0: {
            q8_0_array_t *arr = (q8_0_array_t *)buf->payload;
            arr->scales = (float *)(arr + 1);
            arr->data = (int8_t *)(arr->scales + arr->num_blocks);
            break;
        }
        case Q4_0: {
            q4_0_array_t *arr = (q4_0_array_t *)buf->payload;
            const uint64_t packed_elems = (arr->num_elements + 1) / 2;
            arr->scales = (float *)(arr + 1);
            arr->data = (int8_t *)(arr->scales + arr->num_blocks);
            (void)packed_elems; /* silence unused warning in case of static analysis */
            break;
        }
        case Q2_K: {
            q2_k_array_t *arr = (q2_k_array_t *)buf->payload;
            arr->super_blocks = (super_block_q2_k *)(arr + 1);
            break;
        }
        case TOPK: {
            sparse_array_t *arr = (sparse_array_t *)buf->payload;
            uint32_t sparse_elements = (uint32_t)arr->num_tokens * arr->num_sparse_features;
            arr->sparse_indices = (uint16_t *)(arr + 1);
            arr->values = (float *)(arr->sparse_indices + sparse_elements);
            break;
        }
        case BF16: {
            bf16_array_t *arr = (bf16_array_t *)buf->payload;
            arr->data = (uint16_t *)(arr + 1);
            break;
        }
        case FP16: {
            fp16_array_t *arr = (fp16_array_t *)buf->payload;
            arr->data = (uint16_t *)(arr + 1);
            break;
        }
        default:
            break;
    }
}

static int64_t _get_payload_size(const bitsqueeze_buffer_t *buf) {
    if (!buf) return 0;
    switch (buf->method) {
        case Q8_0:
            return get_q8_0_array((const q8_0_array_t *)buf->payload);
        case Q4_0:
            return get_q4_0_array_size((const q4_0_array_t *)buf->payload);
        case Q2_K:
            return get_q2_k_array_size((const q2_k_array_t *)buf->payload);
        case TOPK:
            return (int64_t)get_sparse_array_size((const sparse_array_t *)buf->payload);
        case BF16:
            return get_bf16_array_size((const bf16_array_t *)buf->payload);
        case FP16:
            return get_fp16_array_size((const fp16_array_t *)buf->payload);
        default:
            return 0;
    }
}

int bsq_compress_1d(const float *src,
                    uint64_t num_elements,
                    bsq_method_t method,
                    bitsqueeze_buffer_t **out) {
    if (!src || num_elements == 0 || !out || *out) return 1;

    switch (method) {
        case Q8_0: {
            q8_0_array_t *arr = NULL;
            if (q8_0_compress(src, num_elements, &arr) || !arr) return 1;
            const size_t payload_size = (size_t)get_q8_0_array(arr);

            bitsqueeze_buffer_t *buf = _allocate_bsq_buffer(payload_size);
            if (!buf) {
                free_q8_0_array(arr);
                return 1;
            }

            buf->method = Q8_0;
            buf->shape.num_elements = num_elements;
            memcpy(buf->payload, arr, payload_size);
            free_q8_0_array(arr);
            _fixup_payload_pointers(buf);
            *out = buf;
            return 0;
        }
        case Q4_0: {
            q4_0_array_t *arr = NULL;
            if (q4_0_compress(src, num_elements, 0, &arr) || !arr) return 1;
            const size_t payload_size = (size_t)get_q4_0_array_size(arr);

            bitsqueeze_buffer_t *buf = _allocate_bsq_buffer(payload_size);
            if (!buf) {
                free_q4_0_array(arr);
                return 1;
            }

            buf->method = Q4_0;
            buf->shape.num_elements = num_elements;
            memcpy(buf->payload, arr, payload_size);
            free_q4_0_array(arr);
            _fixup_payload_pointers(buf);
            *out = buf;
            return 0;
        }
        case Q2_K: {
            q2_k_array_t *arr = NULL;
            if (q2_k_compress(src, num_elements, &arr) || !arr) return 1;
            const size_t payload_size = (size_t)get_q2_k_array_size(arr);

            bitsqueeze_buffer_t *buf = _allocate_bsq_buffer(payload_size);
            if (!buf) {
                free_q2_k_array(arr);
                return 1;
            }

            buf->method = Q2_K;
            buf->shape.num_elements = num_elements;
            memcpy(buf->payload, arr, payload_size);
            free_q2_k_array(arr);
            _fixup_payload_pointers(buf);
            *out = buf;
            return 0;
        }
        case BF16: {
            bf16_array_t *arr = NULL;
            if (bf16_compress(src, num_elements, &arr) || !arr) return 1;
            const size_t payload_size = (size_t)get_bf16_array_size(arr);

            bitsqueeze_buffer_t *buf = _allocate_bsq_buffer(payload_size);
            if (!buf) {
                free_bf16_array(arr);
                return 1;
            }

            buf->method = BF16;
            buf->shape.num_elements = num_elements;
            memcpy(buf->payload, arr, payload_size);
            free_bf16_array(arr);
            _fixup_payload_pointers(buf);
            *out = buf;
            return 0;
        }
        case FP16: {
            fp16_array_t *arr = NULL;
            if (fp16_compress(src, num_elements, &arr) || !arr) return 1;
            const size_t payload_size = (size_t)get_fp16_array_size(arr);

            bitsqueeze_buffer_t *buf = _allocate_bsq_buffer(payload_size);
            if (!buf) {
                free_fp16_array(arr);
                return 1;
            }

            buf->method = FP16;
            buf->shape.num_elements = num_elements;
            memcpy(buf->payload, arr, payload_size);
            free_fp16_array(arr);
            _fixup_payload_pointers(buf);
            *out = buf;
            return 0;
        }
        case TOPK:
        default:
            return 1; /* invalid method for 1D compression */
    }
}

int bsq_compress_2d(const float *src,
                    uint16_t num_tokens,
                    uint16_t num_features,
                    float sparse_ratio,
                    bsq_method_t method,
                    bitsqueeze_buffer_t **out) {
    if (!src || !out || *out || num_tokens == 0 || num_features == 0) return 1;
    if (method != TOPK) return 1;

    sparse_array_t *arr = NULL;
    if (topk_compress(src, num_tokens, num_features, sparse_ratio, &arr) || !arr) return 1;

    const size_t payload_size = (size_t)get_sparse_array_size(arr);
    bitsqueeze_buffer_t *buf = _allocate_bsq_buffer(payload_size);
    if (!buf) {
        free_sparse_array(arr);
        return 1;
    }

    buf->method = TOPK;
    buf->shape.num_tokens = num_tokens;
    buf->shape.num_features = num_features;
    buf->shape.sparse_ratio = sparse_ratio;
    memcpy(buf->payload, arr, payload_size);
    free_sparse_array(arr);
    _fixup_payload_pointers(buf);
    *out = buf;
    return 0;
}

int bsq_decompress(const bitsqueeze_buffer_t *buf,
                   float *dst,
                   uint64_t dst_num_elements) {
    if (!buf || !dst || !buf->payload) return 1;

    switch (buf->method) {
        case Q8_0: {
            const q8_0_array_t *arr = (const q8_0_array_t *)buf->payload;
            if (dst_num_elements < arr->num_elements) return 1;
            return q8_0_decompress(arr, dst);
        }
        case Q4_0: {
            const q4_0_array_t *arr = (const q4_0_array_t *)buf->payload;
            if (dst_num_elements < arr->num_elements) return 1;
            return q4_0_decompress(arr, dst);
        }
        case Q2_K: {
            const q2_k_array_t *arr = (const q2_k_array_t *)buf->payload;
            if (dst_num_elements < arr->num_elements) return 1;
            return q2_k_decompress(arr, dst);
        }
        case BF16: {
            const bf16_array_t *arr = (const bf16_array_t *)buf->payload;
            if (dst_num_elements < arr->num_elements) return 1;
            return bf16_decompress(arr, dst);
        }
        case FP16: {
            const fp16_array_t *arr = (const fp16_array_t *)buf->payload;
            if (dst_num_elements < arr->num_elements) return 1;
            return fp16_decompress(arr, dst);
        }
        case TOPK: {
            const sparse_array_t *arr = (const sparse_array_t *)buf->payload;
            uint64_t expected = (uint64_t)arr->num_tokens * arr->num_features;
            if (dst_num_elements < expected) return 1;
            return topk_decompress(arr, dst);
        }
        default:
            return 1;
    }
}

int64_t bsq_get_packed_size(const bitsqueeze_buffer_t *buf) {
    if (!buf) return 0;
    int64_t payload = _get_payload_size(buf);
    if (payload <= 0) return 0;
    return (int64_t)sizeof(bitsqueeze_buffer_t) + payload;
}

bitsqueeze_buffer_t *load_bsq_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(bitsqueeze_buffer_t)) return NULL;

    bitsqueeze_buffer_t *buf = (bitsqueeze_buffer_t *)calloc(1, buffer_size);
    if (!buf) return NULL;

    memcpy(buf, buffer, buffer_size);
    buf->payload = ((uint8_t *)buf) + sizeof(bitsqueeze_buffer_t);

    const int64_t expected_size = bsq_get_packed_size(buf);
    if (expected_size == 0 || buffer_size < expected_size) {
        free(buf);
        return NULL;
    }

    _fixup_payload_pointers(buf);
    return buf;
}

void bsq_free(bitsqueeze_buffer_t *buf) {
    if (!buf) return;
    free(buf);
}
