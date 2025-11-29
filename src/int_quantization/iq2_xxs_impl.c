#include "int_quantization/iq2_xxs_impl.h"

/* ============================================================================
 * Lookup Tables
 * ============================================================================ */

/* Sign mask for each of 8 positions */
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* 7-bit sign index -> 8-bit sign pattern (with even parity) */
static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

/* 256-entry grid: each uint64_t contains 8 dequantized values (odd: 1,3,5,7)
 * PLACEHOLDER - copy from ggml-common.h iq2xxs_grid table */
static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/* ============================================================================
 * Quantization helper tables (built at runtime)
 * ============================================================================ */

static uint64_t *kgrid_q2xs = NULL;      /* Decoded grid values */
static int      *kmap_q2xs = NULL;       /* Maps 16-bit pattern -> grid index (or negative for neighbor lookup) */
static uint16_t *kneighbors_q2xs = NULL; /* Neighbor lists for off-grid points */
static int       iq2_xxs_initialized = 0;

#define KMAP_SIZE 43692  /* 4^8 rounded up for lookup + some slack */

static int iq2_compare_func(const void *a, const void *b) {
    const int *l = (const int *)a;
    const int *r = (const int *)b;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq2_xxs_init(void) {
    if (iq2_xxs_initialized) return;
    
    const int grid_size = 256;
    const int nwant = 2;  /* Number of neighbor distance levels to keep */
    
    /* Build decoded grid */
    kgrid_q2xs = (uint64_t *)malloc(grid_size * sizeof(uint64_t));
    if (!kgrid_q2xs) return;
    
    // for (int k = 0; k < grid_size; ++k) {
    //     kgrid_q2xs[k] = iq2xxs_grid[k];
    // }
    for (int k = 0; k < grid_size; ++k) {
        uint64_t packed = iq2xxs_grid[k];
        uint64_t normalized = 0;
        
        for (int i = 0; i < 8; ++i) {
            // Extract the byte (e.g., 0x08, 0x19, 0x2b...)
            uint8_t val = (packed >> (i * 8)) & 0xFF;
            uint8_t q_val;

            // Map dequant values back to normalized grid coordinates (1, 3, 5, 7)
            // Values in table are approx: 8, 25, 43, 60
            if      (val < 15) q_val = 1;
            else if (val < 35) q_val = 3;
            else if (val < 55) q_val = 5;
            else               q_val = 7;

            normalized |= ((uint64_t)q_val << (i * 8));
        }
        kgrid_q2xs[k] = normalized;
    }
    
    /* Build map from 16-bit patterns to grid indices */
    kmap_q2xs = (int *)malloc(KMAP_SIZE * sizeof(int));
    if (!kmap_q2xs) {
        free(kgrid_q2xs);
        kgrid_q2xs = NULL;
        return;
    }
    
    for (int i = 0; i < KMAP_SIZE; ++i) kmap_q2xs[i] = -1;
    
    /* Populate direct mappings for grid points */
    for (int i = 0; i < grid_size; ++i) {
        const uint8_t *aux8 = (const uint8_t *)&kgrid_q2xs[i];
        uint16_t index = 0;
        for (int k = 0; k < 8; ++k) {
            uint16_t q = (aux8[k] - 1) / 2;  /* Map {1,3,5,7} back to {0,1,2,3} */
            index |= (q << (2 * k));
        }
        kmap_q2xs[index] = i;
    }
    
    /* Build neighbor lists for off-grid points */
    int8_t pos[8];
    int *dist2 = (int *)malloc(2 * grid_size * sizeof(int));
    if (!dist2) {
        free(kgrid_q2xs); kgrid_q2xs = NULL;
        free(kmap_q2xs);  kmap_q2xs = NULL;
        return;
    }
    
    int num_neighbors = 0, num_not_in_map = 0;
    
    /* First pass: count neighbors needed */
    for (int i = 0; i < KMAP_SIZE; ++i) {
        if (kmap_q2xs[i] >= 0) continue;
        ++num_not_in_map;
        
        for (int k = 0; k < 8; ++k) {
            int l = (i >> (2 * k)) & 0x3;
            pos[k] = 2 * l + 1;
        }
        
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) {
                d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        
        int n = 0, d2_prev = dist2[0], nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2_prev) {
                if (nhave == nwant) break;
                d2_prev = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    
    /* Allocate neighbor storage */
    kneighbors_q2xs = (uint16_t *)malloc((num_neighbors + num_not_in_map) * sizeof(uint16_t));
    if (!kneighbors_q2xs) {
        free(kgrid_q2xs); kgrid_q2xs = NULL;
        free(kmap_q2xs);  kmap_q2xs = NULL;
        free(dist2);
        return;
    }
    
    /* Second pass: populate neighbor lists */
    int counter = 0;
    for (int i = 0; i < KMAP_SIZE; ++i) {
        if (kmap_q2xs[i] >= 0) continue;
        
        for (int k = 0; k < 8; ++k) {
            int l = (i >> (2 * k)) & 0x3;
            pos[k] = 2 * l + 1;
        }
        
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) {
                d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        
        kmap_q2xs[i] = -(counter + 1);  /* Negative = offset into neighbor list */
        
        int d2_prev = dist2[0];
        uint16_t *start = &kneighbors_q2xs[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2_prev) {
                if (nhave == nwant) break;
                d2_prev = dist2[2*j];
                ++nhave;
            }
            kneighbors_q2xs[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;  /* Store count at beginning of neighbor list */
    }
    
    free(dist2);
    iq2_xxs_initialized = 1;
}

void iq2_xxs_free_tables(void) {
    if (kgrid_q2xs) { free(kgrid_q2xs); kgrid_q2xs = NULL; }
    if (kmap_q2xs)  { free(kmap_q2xs);  kmap_q2xs = NULL; }
    if (kneighbors_q2xs) { free(kneighbors_q2xs); kneighbors_q2xs = NULL; }
    iq2_xxs_initialized = 0;
}

/* ============================================================================
 * Array allocation and management
 * ============================================================================ */

static int64_t _get_iq2_xxs_array_size(const iq2_xxs_array_t *arr) {
    if (!arr) return 0;
    /* Header + scales (2 bytes each) + qs (64 bytes per super block) */
    return (int64_t)(sizeof(iq2_xxs_array_t) 
                   + arr->num_super_blocks * sizeof(uint16_t)
                   + arr->num_super_blocks * 64);
}

iq2_xxs_array_t *allocate_iq2_xxs_array(uint64_t num_elements) {
    if (!num_elements) return NULL;
    
    uint64_t num_super_blocks = (num_elements + IQ2_XXS_SUPER_BLOCK_SIZE - 1) / IQ2_XXS_SUPER_BLOCK_SIZE;
    
    size_t total = sizeof(iq2_xxs_array_t)
                 + num_super_blocks * sizeof(uint16_t)   /* scales */
                 + num_super_blocks * 64;                /* qs data */
    
    iq2_xxs_array_t *arr = (iq2_xxs_array_t *)calloc(1, total);
    if (!arr) return NULL;
    
    arr->num_elements = num_elements;
    arr->num_super_blocks = num_super_blocks;
    arr->scales = (uint16_t *)(arr + 1);
    arr->qs = (uint8_t *)(arr->scales + num_super_blocks);
    
    return arr;
}

void free_iq2_xxs_array(iq2_xxs_array_t *arr) {
    if (arr) free(arr);
}

int64_t get_iq2_xxs_array_size(const iq2_xxs_array_t *arr) {
    return _get_iq2_xxs_array_size(arr);
}

iq2_xxs_array_t *load_iq2_xxs_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(iq2_xxs_array_t)) return NULL;
    
    iq2_xxs_array_t *arr = (iq2_xxs_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;
    
    memcpy(arr, buffer, buffer_size);
    
    const int64_t expected = _get_iq2_xxs_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }
    
    arr->scales = (uint16_t *)(arr + 1);
    arr->qs = (uint8_t *)(arr->scales + arr->num_super_blocks);
    
    return arr;
}

/* ============================================================================
 * Dequantization (the simpler direction)
 * ============================================================================ */

int iq2_xxs_decompress(const iq2_xxs_array_t *arr, float *float_array) {
    if (!arr || !float_array) return 1;
    
    const uint64_t num_super_blocks = arr->num_super_blocks;
    const uint64_t num_elements = arr->num_elements;
    
    uint64_t out_idx = 0;
    
    for (uint64_t sb = 0; sb < num_super_blocks; ++sb) {
        const float d = fp16_ieee_to_fp32_value(arr->scales[sb]);
        const uint8_t *qs_block = arr->qs + sb * 64;
        
        /* Process 8 groups of 32 values each */
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32[2];
            memcpy(aux32, qs_block + ib32 * 8, 8);
            
            const uint8_t *aux8 = (const uint8_t *)aux32;
            
            /* Group scale: upper 4 bits of aux32[1] give value 0-15 */
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            
            /* Process 4 sub-groups of 8 values each */
            for (int l = 0; l < 4; ++l) {
                const uint8_t grid_idx = aux8[l];
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + grid_idx);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                
                for (int j = 0; j < 8; ++j) {
                    if (out_idx < num_elements) {
                        float val = db * (float)grid[j];
                        float_array[out_idx++] = (signs & kmask_iq2xs[j]) ? -val : val;
                    }
                }
            }
        }
    }
    
    return 0;
}

/* ============================================================================
 * Quantization helpers
 * ============================================================================ */

static inline int nearest_int(float f) {
    return (int)(f + 0.5f - (f < 0));
}

/* Find best neighbor from neighbor list */
static int iq2_find_best_neighbour(const uint16_t *neighbours, const uint64_t *grid,
                                   const float *xval, const float *weight, 
                                   float scale, int8_t *L) {
    int num_neighbors = neighbours[0];
    if (num_neighbors <= 0) return -1;
    
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = (float)pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    
    if (grid_index >= 0) {
        const int8_t *pg = (const int8_t *)(grid + grid_index);
        for (int i = 0; i < 8; ++i) {
            L[i] = (pg[i] - 1) / 2;
        }
    }
    
    return grid_index;
}

/* ============================================================================
 * Quantization (the complex direction)
 * ============================================================================ */

int iq2_xxs_compress(const float *float_array, uint64_t num_elements, iq2_xxs_array_t **out) {
    if (!float_array || num_elements == 0 || !out || *out) return 1;
    
    /* Ensure tables are initialized */
    if (!iq2_xxs_initialized) {
        iq2_xxs_init();
        if (!iq2_xxs_initialized) return 1;
    }
    
    iq2_xxs_array_t *arr = allocate_iq2_xxs_array(num_elements);
    if (!arr) return 1;
    
    const int kMaxQ = 3;  /* Max quantization level (0-3 maps to 1,3,5,7) */
    const float GROUP_MAX_EPS = 1e-8f;
    
    float weight[32];
    float xval[32];
    float waux[32];
    int8_t L[32];
    int8_t Laux[32];
    uint8_t block_signs[4];
    uint32_t q2[16];  /* 2 uint32 per group × 8 groups = 16 */
    float scales[8];
    
    // uint64_t in_idx = 0;
    
    for (uint64_t sb = 0; sb < arr->num_super_blocks; ++sb) {
        memset(q2, 0, sizeof(q2));
        
        /* Compute variance for importance weighting */
        float sumx2 = 0;
        uint64_t block_start = sb * IQ2_XXS_SUPER_BLOCK_SIZE;
        uint64_t block_end = block_start + IQ2_XXS_SUPER_BLOCK_SIZE;
        if (block_end > num_elements) block_end = num_elements;
        // uint64_t block_len = block_end - block_start;
        
        for (uint64_t i = block_start; i < block_end; ++i) {
            sumx2 += float_array[i] * float_array[i];
        }
        float sigma2 = sumx2 / (float)IQ2_XXS_SUPER_BLOCK_SIZE;
        
        float max_scale = 0;
        
        /* Process 8 groups of 32 values */
        for (int ib = 0; ib < 8; ++ib) {
            uint64_t group_start = block_start + ib * 32;
            uint64_t group_end = group_start + 32;
            if (group_end > num_elements) group_end = num_elements;
            
            /* Build weight and absolute values with sign handling */
            for (int i = 0; i < 32; ++i) {
                uint64_t idx = group_start + i;
                float v = (idx < num_elements) ? float_array[idx] : 0.0f;
                weight[i] = sqrtf(sigma2 + v * v);
                waux[i] = sqrtf(weight[i]);
            }
            
            /* Handle signs with parity constraint */
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                
                for (int i = 0; i < 8; ++i) {
                    uint64_t idx = group_start + 8 * k + i;
                    float v = (idx < num_elements) ? float_array[idx] : 0.0f;
                    
                    if (v >= 0) {
                        xval[8*k + i] = v;
                    } else {
                        xval[8*k + i] = -v;
                        ++nflip;
                        s |= (1 << i);
                    }
                }
                
                /* Enforce even parity by flipping least important sign */
                if (nflip % 2) {
                    int imin = 0;
                    float min = weight[8*k] * xval[8*k] * xval[8*k];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i] * xval[8*k+i] * xval[8*k+i];
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8*k + imin] = -xval[8*k + imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            
            /* Find max for initial scale estimate */
            float max = xval[0];
            for (int i = 1; i < 32; ++i) {
                if (xval[i] > max) max = xval[i];
            }
            
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
            } else {
                /* Search for optimal scale */
                float best = 0;
                float scale = max / (2 * kMaxQ - 1);
                
                for (int is = -6; is <= 6; ++is) {
                    float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                    float this_scale = 1.0f / id;
                    
                    /* Quantize each sub-group */
                    for (int k = 0; k < 4; ++k) {
                        for (int i = 0; i < 8; ++i) {
                            int l = nearest_int(0.5f * (id * xval[8*k+i] - 1));
                            if (l < 0) l = 0;
                            if (l > kMaxQ - 1) l = kMaxQ - 1;
                            Laux[8*k + i] = (int8_t)l;
                        }
                        
                        /* Check if on grid, find neighbors if not */
                        uint16_t u = 0;
                        for (int i = 0; i < 8; ++i) {
                            u |= (Laux[8*k+i] << (2*i));
                        }
                        
                        int grid_index = kmap_q2xs[u];
                        if (grid_index < 0) {
                            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                            iq2_find_best_neighbour(neighbours, kgrid_q2xs, 
                                                   xval + 8*k, waux + 8*k, 
                                                   this_scale, Laux + 8*k);
                        }
                    }
                    
                    /* Compute weighted error and optimal scale */
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 32; ++i) {
                        float w = weight[i];
                        float q = 2 * Laux[i] + 1;
                        sumqx += w * xval[i] * q;
                        sumq2 += w * q * q;
                    }
                    
                    if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                        scale = sumqx / sumq2;
                        best = scale * sumqx;
                        memcpy(L, Laux, 32);
                    }
                }
                
                /* Final refinement */
                if (scale > 0) {
                    float id = 1.0f / scale;
                    for (int k = 0; k < 4; ++k) {
                        uint16_t u = 0;
                        for (int i = 0; i < 8; ++i) {
                            int l = nearest_int(0.5f * (id * xval[8*k+i] - 1));
                            if (l < 0) l = 0;
                            if (l > kMaxQ - 1) l = kMaxQ - 1;
                            u |= (l << (2*i));
                        }
                        
                        int grid_index = kmap_q2xs[u];
                        if (grid_index < 0) {
                            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                            iq2_find_best_neighbour(neighbours, kgrid_q2xs, 
                                                   xval + 8*k, waux + 8*k, 
                                                   scale, L + 8*k);
                        } else {
                            const int8_t *pg = (const int8_t *)(kgrid_q2xs + grid_index);
                            for (int i = 0; i < 8; ++i) {
                                L[8*k+i] = (pg[i] - 1) / 2;
                            }
                        }
                    }
                    
                    /* Recompute optimal scale */
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 32; ++i) {
                        float w = weight[i];
                        float q = 2 * L[i] + 1;
                        sumqx += w * xval[i] * q;
                        sumq2 += w * q * q;
                    }
                    if (sumq2 > 0) scale = sumqx / sumq2;
                }
                
                /* Handle negative scale (shouldn't happen but just in case) */
                if (scale < 0) {
                    scale = -scale;
                    for (int k = 0; k < 4; ++k) {
                        block_signs[k] = (~block_signs[k]) & 127;
                    }
                }
                
                scales[ib] = scale;
                if (scale > max_scale) max_scale = scale;
            }
            
            /* Pack grid indices and signs into q2 */
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    u |= (L[8*k+i] << (2*i));
                }
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    /* This shouldn't happen after optimization, but handle gracefully */
                    grid_index = 0;
                }
                q2[2*ib + 0] |= ((uint32_t)grid_index << (8*k));
                q2[2*ib + 1] |= ((uint32_t)block_signs[k] << (7*k));
            }
        }
        
        /* Encode block scale */
        if (max_scale == 0) {
            arr->scales[sb] = 0;
            memset(arr->qs + sb * 64, 0, 64);
        } else {
            float d = max_scale / 31.0f;
            arr->scales[sb] = fp16_ieee_from_fp32_value(d);
            float id = 1.0f / d;
            
            /* Encode group scales into upper 4 bits */
            for (int ib = 0; ib < 8; ++ib) {
                int l = nearest_int(0.5f * (id * scales[ib] - 1));
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                q2[2*ib + 1] |= ((uint32_t)l << 28);
            }
            
            memcpy(arr->qs + sb * 64, q2, 64);
        }
    }
    
    *out = arr;
    return 0;
}
