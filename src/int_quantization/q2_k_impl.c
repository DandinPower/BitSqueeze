#include "int_quantization/q2_k_impl.h"

#define MAX_VAL(a, b) ((a) > (b) ? (a) : (b))
#define MIN_VAL(a, b) ((a) < (b) ? (a) : (b))
/* The implementation is refer to https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L622 */

q2_k_array_t *allocate_q2_k_array(uint64_t num_elements) {
    if (!num_elements) return NULL;

    uint64_t num_elements_aligned = (num_elements % WEIGHT_PER_SUPER_BLOCK == 0)
                                        ? num_elements
                                        : num_elements + (WEIGHT_PER_SUPER_BLOCK - (num_elements % WEIGHT_PER_SUPER_BLOCK));

    uint64_t num_super_blocks = num_elements_aligned / WEIGHT_PER_SUPER_BLOCK;
    
    size_t total = sizeof(q2_k_array_t) + num_super_blocks * sizeof(super_block_q2_k);
    q2_k_array_t *qa = (q2_k_array_t *)calloc(1, total);
    if (!qa) return NULL;
    
    qa->num_elements = num_elements;
    qa->num_elements_aligned = num_elements_aligned;
    qa->num_super_blocks = num_super_blocks;
    qa->super_blocks = (super_block_q2_k *)(qa + 1);

    return qa;
}

void free_q2_k_array(q2_k_array_t *q2_k_array) {
    if (!q2_k_array) return;
    free(q2_k_array);
}

int64_t get_q2_k_array_size(const q2_k_array_t *q2_k_array) {
    if (!q2_k_array) return 0;
    return sizeof(q2_k_array_t) + q2_k_array->num_super_blocks * sizeof(super_block_q2_k);
}

q2_k_array_t *load_q2_k_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(q2_k_array_t)) return NULL;

    q2_k_array_t *q2_k_array = (q2_k_array_t *)calloc(1, buffer_size);
    if (!q2_k_array) return NULL;
    
    memcpy(q2_k_array, buffer, buffer_size);
    const int64_t expected = get_q2_k_array_size(q2_k_array);
    if (buffer_size < expected) {
        free(q2_k_array);
        return NULL;
    }

    q2_k_array->super_blocks = (super_block_q2_k *)(q2_k_array + 1);
    return q2_k_array;
}

static void find_optimal_scale_and_min(const float *weights, float *scale, float *min_val) {
    const float q2_scale = 3.f;
    float local_min = INFINITY;
    float local_max = -INFINITY;
    
    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        if (weights[l] < local_min) local_min = weights[l];
    }

    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        float shifted = weights[l] - local_min;
        if (shifted > local_max) local_max = shifted;
    }

    *scale = local_max / q2_scale;
    *min_val = local_min;
}

int q2_k_compress(const float *float_array, uint64_t num_elements, q2_k_array_t **q2_k_array) {
    const float q4_scale = 15.f;
    
    uint8_t L[WEIGHT_PER_SUPER_BLOCK];
    float weights[Q2_K_BLOCK_SIZE];
    float mins[Q2_K_SUPER_BLOCK_SIZE];
    float scales[Q2_K_SUPER_BLOCK_SIZE];
    
    if (!float_array || num_elements == 0 || !q2_k_array || *q2_k_array) {
        return 1;
    }
    
    *q2_k_array = allocate_q2_k_array(num_elements);
    if (!*q2_k_array) {
        return 1;
    }
    q2_k_array_t *qa = *q2_k_array;

    float *float_array_aligned = (float *)calloc(qa->num_elements_aligned, sizeof(float));
    if (!float_array_aligned) {
        free_q2_k_array(qa);
        *q2_k_array = NULL;
        return 1;
    }
    memcpy(float_array_aligned, float_array, qa->num_elements * sizeof(float));

    for (uint32_t curr_super_block_index = 0; curr_super_block_index < qa->num_super_blocks; curr_super_block_index++) {
        super_block_q2_k *curr_super_block = &qa->super_blocks[curr_super_block_index];
        const float *sb_base = float_array_aligned + (uint64_t)curr_super_block_index * WEIGHT_PER_SUPER_BLOCK;
        
        float max_scale = -INFINITY;
        float max_abs_min = 0.f;

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            memcpy(weights, sb_base + j * Q2_K_BLOCK_SIZE, Q2_K_BLOCK_SIZE * sizeof(float));
            find_optimal_scale_and_min(weights, &scales[j], &mins[j]);
            if (scales[j] > max_scale) {
                max_scale = scales[j];
            }
            if (fabsf(mins[j]) > max_abs_min) {
                max_abs_min = fabsf(mins[j]);
            }
        }

        if (max_scale > 0) {
            float iscale = q4_scale / max_scale;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * scales[j]);
                curr_super_block->scales[j] = l;
            }
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(max_scale / q4_scale);
        } else {
            memset(curr_super_block->scales, 0, sizeof(curr_super_block->scales));
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(0.f);
        }

        if (max_abs_min > 0) {
            const float iscale = 7.f / max_abs_min;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * mins[j]);
                l = MAX_VAL(-8, MIN_VAL(7, l));
                curr_super_block->scales[j] |= ((l & 0xF) << 4);
            }
            curr_super_block->super_min = fp16_ieee_from_fp32_value(max_abs_min / 7.f);
        } else {
            curr_super_block->super_min = fp16_ieee_from_fp32_value(0.f);
        }

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            const float temp_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale) * (curr_super_block->scales[j] & 0xF);
            const float m = fp16_ieee_to_fp32_value(curr_super_block->super_min);
            const int8_t min_q = (curr_super_block->scales[j] >> 4);
            const float temp_min = m * ((int8_t)(min_q << 4) >> 4);
        
            for (int ii = 0; ii < Q2_K_BLOCK_SIZE; ii++) {
                float val = (temp_scale > 0.f) ? (sb_base[j * Q2_K_BLOCK_SIZE + ii] - temp_min) / temp_scale : 0.f;
                int l = (int)lrintf(val);
                l = MAX_VAL(0, MIN_VAL(3, l));
                L[j * Q2_K_BLOCK_SIZE + ii] = (uint8_t)l;
            }
        }

        uint32_t packed_run = WEIGHT_PER_SUPER_BLOCK / 2; // 128
        for (int j = 0; j < WEIGHT_PER_SUPER_BLOCK; j += packed_run) {
            for (int l = 0; l < Q2_K_BLOCK_SIZE * 2; l++) { // l = 0..31
                uint8_t b0 = L[j + l + 0];
                uint8_t b1 = L[j + l + 32];
                uint8_t b2 = L[j + l + 64];
                uint8_t b3 = L[j + l + 96];
                curr_super_block->data[j / 4 + l] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
            }
        }
    }

    free(float_array_aligned);
    return 0;
}

int q2_k_decompress(const q2_k_array_t *q2_k_array, float *float_array) {
    if (!q2_k_array || !float_array || q2_k_array->num_super_blocks == 0) {
        return 1;
    }

    const uint64_t total_elements = q2_k_array->num_elements;

    for (uint32_t s = 0; s < q2_k_array->num_super_blocks; ++s) {
        const super_block_q2_k *curr_super_block = &q2_k_array->super_blocks[s];
        const float super_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale);
        const float super_min   = fp16_ieee_to_fp32_value(curr_super_block->super_min);

        float scales[Q2_K_SUPER_BLOCK_SIZE];
        float mins[Q2_K_SUPER_BLOCK_SIZE];

        for (int i = 0; i < Q2_K_SUPER_BLOCK_SIZE; ++i) {
            uint8_t packed_val = curr_super_block->scales[i];
            scales[i] = super_scale * (packed_val & 0x0F);
            
            int8_t min_q = (packed_val >> 4);
            mins[i] = super_min * ((int8_t)(min_q << 4) >> 4);
        }

        const uint8_t *q = curr_super_block->data;
        const uint64_t base_idx = (uint64_t)s * WEIGHT_PER_SUPER_BLOCK;

        for (int l = 0; l < 32; ++l) {
            uint8_t packed_byte = q[l];
            
            uint64_t idx0 = base_idx + (uint64_t)l;
            uint64_t idx1 = base_idx + (uint64_t)l + 32;
            uint64_t idx2 = base_idx + (uint64_t)l + 64;
            uint64_t idx3 = base_idx + (uint64_t)l + 96;

            const int local0 = l;
            const int local1 = l + 32;
            const int local2 = l + 64;
            const int local3 = l + 96;

            if (idx0 < total_elements) float_array[idx0] = mins[local0/16] + scales[local0/16] * ((packed_byte >> 0) & 3);
            if (idx1 < total_elements) float_array[idx1] = mins[local1/16] + scales[local1/16] * ((packed_byte >> 2) & 3);
            if (idx2 < total_elements) float_array[idx2] = mins[local2/16] + scales[local2/16] * ((packed_byte >> 4) & 3);
            if (idx3 < total_elements) float_array[idx3] = mins[local3/16] + scales[local3/16] * ((packed_byte >> 6) & 3);
        }

        for (int l = 0; l < 32; ++l) {
            uint8_t packed_byte = q[32 + l];
            
            uint64_t idx0 = base_idx + 128 + (uint64_t)l;
            uint64_t idx1 = base_idx + 160 + (uint64_t)l;
            uint64_t idx2 = base_idx + 192 + (uint64_t)l;
            uint64_t idx3 = base_idx + 224 + (uint64_t)l;

            if (idx0 < total_elements) float_array[idx0] = mins[(idx0 - base_idx)/16] + scales[(idx0 - base_idx)/16] * ((packed_byte >> 0) & 3);
            if (idx1 < total_elements) float_array[idx1] = mins[(idx1 - base_idx)/16] + scales[(idx1 - base_idx)/16] * ((packed_byte >> 2) & 3);
            if (idx2 < total_elements) float_array[idx2] = mins[(idx2 - base_idx)/16] + scales[(idx2 - base_idx)/16] * ((packed_byte >> 4) & 3);
            if (idx3 < total_elements) float_array[idx3] = mins[(idx3 - base_idx)/16] + scales[(idx3 - base_idx)/16] * ((packed_byte >> 6) & 3);
        }
    }
    return 0;
}
