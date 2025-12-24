#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sparsity/topk_im_impl.h"

#define NUM_TOKENS 2
#define NUM_FEATURES 5
#define SPARSE_RATIO 0.4f

static int index_in_expected(uint16_t idx, const uint16_t *expected, uint16_t expected_len) {
    for (uint16_t i = 0; i < expected_len; ++i) {
        if (expected[i] == idx) return 1;
    }
    return 0;
}

static int verify_token(const sparse_array_t *sa,
                        const float *src,
                        uint16_t token,
                        const uint16_t *expected,
                        uint16_t expected_len) {
    const uint16_t K = sa->num_sparse_features;
    if (K != expected_len) {
        fprintf(stderr, "token %u: expected K=%u, got %u\n", token, expected_len, K);
        return 0;
    }

    uint8_t seen[NUM_FEATURES] = {0};
    const uint32_t sparse_base = (uint32_t)token * K;
    const uint32_t dense_base = (uint32_t)token * NUM_FEATURES;

    for (uint16_t i = 0; i < K; ++i) {
        uint16_t idx = sa->sparse_indices[sparse_base + i];
        if (idx >= NUM_FEATURES) {
            fprintf(stderr, "token %u: index %u out of range\n", token, idx);
            return 0;
        }
        if (!index_in_expected(idx, expected, expected_len)) {
            fprintf(stderr, "token %u: unexpected index %u\n", token, idx);
            return 0;
        }
        if (seen[idx]) {
            fprintf(stderr, "token %u: duplicate index %u\n", token, idx);
            return 0;
        }
        seen[idx] = 1;

        float got = sa->values[sparse_base + i];
        float want = src[dense_base + idx];
        if (got != want) {
            fprintf(stderr, "token %u: value mismatch at index %u (got %.6f, want %.6f)\n",
                    token, idx, got, want);
            return 0;
        }
    }

    for (uint16_t i = 0; i < expected_len; ++i) {
        if (!seen[expected[i]]) {
            fprintf(stderr, "token %u: missing expected index %u\n", token, expected[i]);
            return 0;
        }
    }

    return 1;
}

int main(void) {
    const float src[NUM_TOKENS * NUM_FEATURES] = {
        10.0f, -9.0f, 8.0f, -7.0f, 6.0f,
         1.0f,  2.0f, 3.0f,  4.0f, 5.0f
    };
    const float importance[NUM_TOKENS * NUM_FEATURES] = {
        0.1f, 0.9f, 0.2f, 0.8f, 0.3f,
        5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };

    sparse_array_t *sa = NULL;
    int rc = topk_im_compress(src, importance, NUM_TOKENS, NUM_FEATURES, SPARSE_RATIO, &sa);
    if (rc || !sa) {
        fprintf(stderr, "topk_im_compress failed (rc=%d)\n", rc);
        return EXIT_FAILURE;
    }

    if (sa->num_sparse_features != 2) {
        fprintf(stderr, "unexpected num_sparse_features=%u\n", sa->num_sparse_features);
        free_sparse_array(sa);
        return EXIT_FAILURE;
    }

    const uint16_t expected_token0[] = {1, 3};
    const uint16_t expected_token1[] = {0, 1};

    if (!verify_token(sa, src, 0, expected_token0, 2) ||
        !verify_token(sa, src, 1, expected_token1, 2)) {
        free_sparse_array(sa);
        return EXIT_FAILURE;
    }

    float decomp[NUM_TOKENS * NUM_FEATURES];
    if (topk_im_decompress(sa, decomp) != 0) {
        fprintf(stderr, "topk_im_decompress failed\n");
        free_sparse_array(sa);
        return EXIT_FAILURE;
    }

    float expected_decomp[NUM_TOKENS * NUM_FEATURES] = {0};
    expected_decomp[0 * NUM_FEATURES + 1] = src[0 * NUM_FEATURES + 1];
    expected_decomp[0 * NUM_FEATURES + 3] = src[0 * NUM_FEATURES + 3];
    expected_decomp[1 * NUM_FEATURES + 0] = src[1 * NUM_FEATURES + 0];
    expected_decomp[1 * NUM_FEATURES + 1] = src[1 * NUM_FEATURES + 1];

    for (uint16_t i = 0; i < NUM_TOKENS * NUM_FEATURES; ++i) {
        if (decomp[i] != expected_decomp[i]) {
            fprintf(stderr, "decompress mismatch at %u (got %.6f, want %.6f)\n",
                    i, decomp[i], expected_decomp[i]);
            free_sparse_array(sa);
            return EXIT_FAILURE;
        }
    }

    float applied[NUM_TOKENS * NUM_FEATURES];
    for (uint16_t i = 0; i < NUM_TOKENS * NUM_FEATURES; ++i) {
        applied[i] = -1.0f;
    }

    if (topk_im_apply(sa, applied) != 0) {
        fprintf(stderr, "topk_im_apply failed\n");
        free_sparse_array(sa);
        return EXIT_FAILURE;
    }

    for (uint16_t i = 0; i < NUM_TOKENS * NUM_FEATURES; ++i) {
        float want = expected_decomp[i] != 0.0f ? expected_decomp[i] : -1.0f;
        if (applied[i] != want) {
            fprintf(stderr, "apply mismatch at %u (got %.6f, want %.6f)\n",
                    i, applied[i], want);
            free_sparse_array(sa);
            return EXIT_FAILURE;
        }
    }

    free_sparse_array(sa);
    return EXIT_SUCCESS;
}
