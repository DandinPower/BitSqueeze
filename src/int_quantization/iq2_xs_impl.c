#include "int_quantization/iq2_xs_impl.h"
#include "datatype/fp16/fp16.h"

/* ============================================================================
 * Lookup Tables
 * ============================================================================ */

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

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

/* 512-entry grid for IQ2_XS
 * PLACEHOLDER - copy from ggml-common.h iq2xs_grid table */
static const uint64_t iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

/* ============================================================================
 * Quantization helper tables (built at runtime)
 * ============================================================================ */

static uint64_t *kgrid_q2xs = NULL;
static int      *kmap_q2xs = NULL;
static uint16_t *kneighbors_q2xs = NULL;
static int       iq2_xs_initialized = 0;

#define KMAP_SIZE 43692

static int iq2_compare_func(const void *a, const void *b) {
    const int *l = (const int *)a;
    const int *r = (const int *)b;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq2_xs_init(void) {
    if (iq2_xs_initialized) return;
    
    const int grid_size = 512;
    const int nwant = 2;
    
    kgrid_q2xs = (uint64_t *)malloc(grid_size * sizeof(uint64_t));
    if (!kgrid_q2xs) return;
    
    // for (int k = 0; k < grid_size; ++k) {
    //     kgrid_q2xs[k] = iq2xs_grid[k];
    // }
    for (int k = 0; k < grid_size; ++k) {
        uint64_t packed = iq2xs_grid[k];
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
    
    kmap_q2xs = (int *)malloc(KMAP_SIZE * sizeof(int));
    if (!kmap_q2xs) {
        free(kgrid_q2xs); kgrid_q2xs = NULL;
        return;
    }
    
    for (int i = 0; i < KMAP_SIZE; ++i) kmap_q2xs[i] = -1;
    
    for (int i = 0; i < grid_size; ++i) {
        const uint8_t *aux8 = (const uint8_t *)&kgrid_q2xs[i];
        uint16_t index = 0;
        for (int k = 0; k < 8; ++k) {
            uint16_t q = (aux8[k] - 1) / 2;
            index |= (q << (2 * k));
        }
        kmap_q2xs[index] = i;
    }
    
    int8_t pos[8];
    int *dist2 = (int *)malloc(2 * grid_size * sizeof(int));
    if (!dist2) {
        free(kgrid_q2xs); kgrid_q2xs = NULL;
        free(kmap_q2xs);  kmap_q2xs = NULL;
        return;
    }
    
    int num_neighbors = 0, num_not_in_map = 0;
    
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
    
    kneighbors_q2xs = (uint16_t *)malloc((num_neighbors + num_not_in_map) * sizeof(uint16_t));
    if (!kneighbors_q2xs) {
        free(kgrid_q2xs); kgrid_q2xs = NULL;
        free(kmap_q2xs);  kmap_q2xs = NULL;
        free(dist2);
        return;
    }
    
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
        
        kmap_q2xs[i] = -(counter + 1);
        
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
        *start = n;
    }
    
    free(dist2);
    iq2_xs_initialized = 1;
}

void iq2_xs_free_tables(void) {
    if (kgrid_q2xs) { free(kgrid_q2xs); kgrid_q2xs = NULL; }
    if (kmap_q2xs)  { free(kmap_q2xs);  kmap_q2xs = NULL; }
    if (kneighbors_q2xs) { free(kneighbors_q2xs); kneighbors_q2xs = NULL; }
    iq2_xs_initialized = 0;
}

/* ============================================================================
 * Array allocation and management
 * ============================================================================ */

static int64_t _get_iq2_xs_array_size(const iq2_xs_array_t *arr) {
    if (!arr) return 0;
    /* Header + d (2 bytes) + qs (64 bytes) + scales (8 bytes) per super block */
    return (int64_t)(sizeof(iq2_xs_array_t) 
                   + arr->num_super_blocks * sizeof(uint16_t)      /* d */
                   + arr->num_super_blocks * 32 * sizeof(uint16_t) /* qs */
                   + arr->num_super_blocks * 8);                   /* scales */
}

iq2_xs_array_t *allocate_iq2_xs_array(uint64_t num_elements) {
    if (!num_elements) return NULL;
    
    uint64_t num_super_blocks = (num_elements + IQ2_XS_SUPER_BLOCK_SIZE - 1) / IQ2_XS_SUPER_BLOCK_SIZE;
    
    size_t total = sizeof(iq2_xs_array_t)
                 + num_super_blocks * sizeof(uint16_t)      /* d */
                 + num_super_blocks * 32 * sizeof(uint16_t) /* qs */
                 + num_super_blocks * 8;                    /* scales */
    
    iq2_xs_array_t *arr = (iq2_xs_array_t *)calloc(1, total);
    if (!arr) return NULL;
    
    arr->num_elements = num_elements;
    arr->num_super_blocks = num_super_blocks;
    arr->d = (uint16_t *)(arr + 1);
    arr->qs = (uint16_t *)(arr->d + num_super_blocks);
    arr->scales = (uint8_t *)(arr->qs + num_super_blocks * 32);
    
    return arr;
}

void free_iq2_xs_array(iq2_xs_array_t *arr) {
    if (arr) free(arr);
}

int64_t get_iq2_xs_array_size(const iq2_xs_array_t *arr) {
    return _get_iq2_xs_array_size(arr);
}

iq2_xs_array_t *load_iq2_xs_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(iq2_xs_array_t)) return NULL;
    
    iq2_xs_array_t *arr = (iq2_xs_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;
    
    memcpy(arr, buffer, buffer_size);
    
    const int64_t expected = _get_iq2_xs_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }
    
    arr->d = (uint16_t *)(arr + 1);
    arr->qs = (uint16_t *)(arr->d + arr->num_super_blocks);
    arr->scales = (uint8_t *)(arr->qs + arr->num_super_blocks * 32);
    
    return arr;
}

/* ============================================================================
 * Dequantization
 * ============================================================================ */

int iq2_xs_decompress(const iq2_xs_array_t *arr, float *float_array) {
    if (!arr || !float_array) return 1;
    
    const uint64_t num_super_blocks = arr->num_super_blocks;
    const uint64_t num_elements = arr->num_elements;
    
    uint64_t out_idx = 0;
    
    for (uint64_t sb = 0; sb < num_super_blocks; ++sb) {
        const float d = fp16_ieee_to_fp32_value(arr->d[sb]);
        const uint16_t *qs_block = arr->qs + sb * 32;
        const uint8_t *scales_block = arr->scales + sb * 8;
        
        /* Process 8 groups of 32 values */
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            /* Two 4-bit scales per group (for 16 values each) */
            float db[2];
            db[0] = d * (0.5f + (float)(scales_block[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (float)(scales_block[ib32] >> 4)) * 0.25f;
            
            /* Process 4 sub-groups of 8 values */
            for (int l = 0; l < 4; ++l) {
                uint16_t qs_val = qs_block[4 * ib32 + l];
                uint16_t grid_idx = qs_val & 511;  /* 9 bits */
                uint8_t sign_idx = qs_val >> 9;    /* 7 bits */
                
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + grid_idx);
                const uint8_t signs = ksigns_iq2xs[sign_idx];
                
                const float dl = db[l / 2];
                
                for (int j = 0; j < 8; ++j) {
                    if (out_idx < num_elements) {
                        float val = dl * (float)grid[j];
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
 * Quantization
 * ============================================================================ */

int iq2_xs_compress(const float *float_array, uint64_t num_elements, iq2_xs_array_t **out) {
    if (!float_array || num_elements == 0 || !out || *out) return 1;
    
    if (!iq2_xs_initialized) {
        iq2_xs_init();
        if (!iq2_xs_initialized) return 1;
    }
    
    iq2_xs_array_t *arr = allocate_iq2_xs_array(num_elements);
    if (!arr) return 1;
    
    const int kMaxQ = 3;
    const float GROUP_MAX_EPS = 1e-8f;
    
    float weight[16];
    float xval[16];
    float waux[16];
    int8_t L[16];
    int8_t Laux[16];
    uint8_t block_signs[2];
    uint16_t q2[32];
    float scales[16];  /* 16 sub-group scales per super block */
    
    for (uint64_t sb = 0; sb < arr->num_super_blocks; ++sb) {
        memset(q2, 0, sizeof(q2));
        
        uint64_t block_start = sb * IQ2_XS_SUPER_BLOCK_SIZE;
        uint64_t block_end = block_start + IQ2_XS_SUPER_BLOCK_SIZE;
        if (block_end > num_elements) block_end = num_elements;
        
        float sumx2 = 0;
        for (uint64_t i = block_start; i < block_end; ++i) {
            sumx2 += float_array[i] * float_array[i];
        }
        float sigma2 = sumx2 / (float)IQ2_XS_SUPER_BLOCK_SIZE;
        
        float max_scale = 0;
        
        /* Process 16 sub-groups of 16 values (8 groups × 2 halves) */
        for (int ib = 0; ib < 16; ++ib) {
            uint64_t group_start = block_start + ib * 16;
            
            for (int i = 0; i < 16; ++i) {
                uint64_t idx = group_start + i;
                float v = (idx < num_elements) ? float_array[idx] : 0.0f;
                weight[i] = sqrtf(sigma2 + v * v);
                waux[i] = sqrtf(weight[i]);
            }
            
            /* Handle signs with parity constraint for 2 sub-groups of 8 */
            for (int k = 0; k < 2; ++k) {
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
                
                if (nflip % 2) {
                    int imin = 0;
                    float min = weight[8*k] * xval[8*k] * xval[8*k];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i] * xval[8*k+i] * xval[8*k+i];
                        if (ax < min) { min = ax; imin = i; }
                    }
                    xval[8*k + imin] = -xval[8*k + imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            
            float max = xval[0];
            for (int i = 1; i < 16; ++i) {
                if (xval[i] > max) max = xval[i];
            }
            
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 16);
            } else {
                float best = 0;
                float scale = max / (2 * kMaxQ - 1);
                int is_on_grid[2] = {1, 1};
                
                for (int is = -9; is <= 9; ++is) {
                    float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                    float this_scale = 1.0f / id;
                    int is_on_grid_aux[2] = {1, 1};
                    
                    for (int k = 0; k < 2; ++k) {
                        for (int i = 0; i < 8; ++i) {
                            int l = nearest_int(0.5f * (id * xval[8*k+i] - 1));
                            if (l < 0) l = 0;
                            if (l > kMaxQ - 1) l = kMaxQ - 1;
                            Laux[8*k + i] = (int8_t)l;
                        }
                        
                        uint16_t u = 0;
                        for (int i = 0; i < 8; ++i) {
                            u |= (Laux[8*k+i] << (2*i));
                        }
                        
                        int grid_index = kmap_q2xs[u];
                        if (grid_index < 0) {
                            is_on_grid_aux[k] = 0;
                            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                            iq2_find_best_neighbour(neighbours, kgrid_q2xs, 
                                                   xval + 8*k, waux + 8*k, 
                                                   this_scale, Laux + 8*k);
                        }
                    }
                    
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 16; ++i) {
                        float w = weight[i];
                        float q = 2 * Laux[i] + 1;
                        sumqx += w * xval[i] * q;
                        sumq2 += w * q * q;
                    }
                    
                    if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                        scale = sumqx / sumq2;
                        best = scale * sumqx;
                        memcpy(L, Laux, 16);
                        is_on_grid[0] = is_on_grid_aux[0];
                        is_on_grid[1] = is_on_grid_aux[1];
                    }
                }
                
                /* Refinement for off-grid points */
                int n_not_ongrid = (is_on_grid[0] ? 0 : 1) + (is_on_grid[1] ? 0 : 1);
                if (n_not_ongrid > 0 && scale > 0) {
                    float id = 1.0f / scale;
                    for (int k = 0; k < 2; ++k) {
                        if (is_on_grid[k]) continue;
                        uint16_t u = 0;
                        for (int i = 0; i < 8; ++i) {
                            int l = nearest_int(0.5f * (id * xval[8*k+i] - 1));
                            if (l < 0) l = 0;
                            if (l > kMaxQ - 1) l = kMaxQ - 1;
                            u |= (l << (2*i));
                            L[8*k + i] = l;
                        }
                        int grid_index = kmap_q2xs[u];
                        if (grid_index < 0) {
                            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                            iq2_find_best_neighbour(neighbours, kgrid_q2xs, 
                                                   xval + 8*k, waux + 8*k, 
                                                   scale, L + 8*k);
                        }
                    }
                    
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 16; ++i) {
                        float w = weight[i];
                        float q = 2 * L[i] + 1;
                        sumqx += w * xval[i] * q;
                        sumq2 += w * q * q;
                    }
                    if (sumq2 > 0) scale = sumqx / sumq2;
                }
                
                if (scale < 0) {
                    scale = -scale;
                    for (int k = 0; k < 2; ++k) {
                        block_signs[k] = (~block_signs[k]) & 127;
                    }
                }
                
                scales[ib] = scale;
                if (scale > max_scale) max_scale = scale;
            }
            
            /* Pack into q2: grid index (9 bits) | sign pattern (7 bits) */
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    u |= (L[8*k+i] << (2*i));
                }
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) grid_index = 0;
                
                q2[2 * ib + k] = (uint16_t)grid_index | ((uint16_t)block_signs[k] << 9);
            }
        }
        
        /* Encode block scale */
        if (max_scale == 0) {
            arr->d[sb] = 0;
            memset(arr->qs + sb * 32, 0, 64);
            memset(arr->scales + sb * 8, 0, 8);
        } else {
            float d = max_scale / 31.0f;
            arr->d[sb] = fp16_ieee_from_fp32_value(d);
            float id = 1.0f / d;
            
            /* Encode group scales: 2 × 4-bit scales per byte */
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                int l0 = nearest_int(0.5f * (id * scales[2*ib32 + 0] - 1));
                int l1 = nearest_int(0.5f * (id * scales[2*ib32 + 1] - 1));
                if (l0 < 0) l0 = 0; if (l0 > 15) l0 = 15;
                if (l1 < 0) l1 = 0; if (l1 > 15) l1 = 15;
                arr->scales[sb * 8 + ib32] = (uint8_t)(l0 | (l1 << 4));
            }
            
            memcpy(arr->qs + sb * 32, q2, 64);
        }
    }
    
    *out = arr;
    return 0;
}
