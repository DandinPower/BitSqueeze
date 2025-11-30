# A Deep Dive into GGML IQ2 Quantization

This tutorial explains how GGML's IQ2 (Importance-matrix Quantization, 2-bit) family works, based entirely on the source code provided. IQ2 is a sophisticated approach to achieving near-2-bit-per-weight compression while maintaining quality through clever encoding tricks.

## High-Level Concept

### The Core Idea: Constrained Codebook Quantization

Traditional 2-bit quantization would allow each weight to independently take one of 4 values (00, 01, 10, 11). IQ2 takes a radically different approach: instead of quantizing individual values, it quantizes **groups of 8 values together** by constraining them to lie on a precomputed **grid** (codebook) of valid combinations.

Think of it this way: if you have 8 values each taking 4 possible states independently, you'd have 4^8 = 65,536 possible combinations. But IQ2 restricts this to only 256, 512, or 1024 carefully chosen combinations (depending on the variant). This restriction actually *improves* quality because the valid grid points are selected to match typical neural network weight distributions.

### The Three IQ2 Variants

The code defines three variants with increasing bits-per-weight (bpw):

| Variant | bpw | Grid Size | Key Tradeoff |
|---------|-----|-----------|--------------|
| IQ2_XXS | 2.0625 | 256 | Smallest, most compressed |
| IQ2_XS | 2.3125 | 512 | Middle ground |
| IQ2_S | 2.5625 | 1024 | Highest quality |

The "extra" bits beyond 2.0 come from storing per-group scales and using larger grids.

---

## Data Structures

Let me walk through the block structures, since understanding them is essential:

```c
#define QK_K 256  // Super-block size: 256 values per block

// IQ2_XXS: 2.0625 bpw
typedef struct {
    ggml_half d;              // 16-bit block scale
    uint16_t qs[QK_K/8];      // 32 uint16's = 64 bytes of quantized data
} block_iq2_xxs;              // Total: 2 + 64 = 66 bytes for 256 values
```

For 256 values in 66 bytes: 66 * 8 / 256 = **2.0625 bpw**. The math checks out.

```c
// IQ2_XS: 2.3125 bpw
typedef struct {
    ggml_half d;              // 16-bit block scale
    uint16_t qs[QK_K/8];      // 32 uint16's
    uint8_t scales[QK_K/32];  // 8 additional scale bytes
} block_iq2_xs;               // Total: 2 + 64 + 8 = 74 bytes
```

74 * 8 / 256 = **2.3125 bpw**.

---

## The Grid Tables: The Heart of IQ2

The magic lies in these lookup tables. Let's examine `iq2xxs_grid`:

```c
GGML_TABLE_BEGIN(uint64_t, iq2xxs_grid, 256)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, ...
GGML_TABLE_END()
```

Each 64-bit entry encodes 8 values (one per byte). The bytes represent the **dequantized values directly**. Looking at the initialization code reveals the encoding:

```c
for (int k = 0; k < grid_size; ++k) {
    int8_t * pos = (int8_t *)(the_grid + k);
    for (int i = 0; i < 8; ++i) {
        int l = (kgrid[k] >> 2*i) & 0x3;  // Extract 2-bit index (0-3)
        pos[i] = 2*l + 1;                  // Map to {1, 3, 5, 7}
    }
}
```

So the 2-bit indices {0, 1, 2, 3} map to odd values {1, 3, 5, 7}. The actual grid entries store these decoded values directly for fast dequantization.

**Example**: The hex value `0x08` = 8 in decimal, but looking at the raw grid source:
- `kgrid_2bit_256[0] = 0` means all 2-bit indices are 0, giving eight 1's → stored as `0x0101010101010101`

Wait, let me re-examine. The grid stores the *decoded* odd values {1, 3, 5, 7}... actually looking at `0x08`, that's just 8. Looking more carefully at the bytes:

- `0x08` = 8, `0x19` = 25, `0x2b` = 43

These don't immediately correspond to {1,3,5,7}. Let me trace through the dequantization to understand the actual encoding.

---

## Dequantization Flow (The Easier Direction)

Let's trace through `dequantize_row_iq2_xxs` to understand how data is unpacked:

```c
void dequantize_row_iq2_xxs(const block_iq2_xxs * x, float * y, int64_t k) {
    const int64_t nb = k / QK_K;  // Number of blocks
    
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);  // Block scale
        
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {  // 8 groups of 32 values
            // Load 8 bytes of quantized data
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            
            // Extract group scale from upper 4 bits of aux32[1]
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            
            for (int l = 0; l < 4; ++l) {  // 4 sub-groups of 8 values
                // aux8[l] is the grid index (0-255)
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                
                // Extract 7-bit sign pattern
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}
```

### Breaking Down the Dequantization Step-by-Step

**Step 1: Block Structure**
- Each block contains 256 values
- Processed in 8 groups of 32 values each
- Each 32-value group uses 8 bytes (64 bits) of `qs` data

**Step 2: Data Packing in aux32**
The 8 bytes for each 32-value group are interpreted as two 32-bit words:
- `aux32[0]`: Contains 4 grid indices (bytes 0-3), each indexing into `iq2xxs_grid`
- `aux32[1]`: Contains sign patterns and the group scale

**Step 3: Scale Computation**
```c
const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
```
- `aux32[1] >> 28` extracts a 4-bit scale factor (0-15)
- This gives a multiplier range of 0.5 to 15.5, then divided by 4
- Combined with block scale `d`, this gives fine-grained control

**Step 4: Grid Lookup**
```c
const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
```
Each byte `aux8[l]` (values 0-255) indexes directly into the 256-entry grid to get 8 dequantized values.

**Step 5: Sign Application**
```c
const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
```

This is clever! The sign pattern for each 8-value sub-group is stored in 7 bits (not 8). How? Through a **parity constraint**: the product of all 8 signs is forced to be +1 (even number of negatives). Given 7 bits, the 8th sign is determined.

The `ksigns_iq2xs` table maps 7-bit patterns to 8-bit sign masks:
```c
GGML_TABLE_BEGIN(uint8_t, ksigns_iq2xs, 128)
    0, 129, 130, 3, 132, 5, 6, 135, ...
GGML_TABLE_END()
```

Looking at a few entries:
- Index 0 → 0 (00000000): no negatives
- Index 1 → 129 (10000001): bits 0 and 7 set
- Index 3 → 3 (00000011): bits 0 and 1 set

Each output has an even number of 1-bits, confirming the parity constraint.

---

## Quantization Flow (The Complex Direction)

Now let's trace `quantize_row_iq2_xxs_impl`, which is more involved:

### Phase 1: Setup and Importance Weighting

```c
const float * xbl = x + QK_K*ibl;
float sumx2 = 0;
for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
float sigma2 = sumx2/QK_K;  // Variance estimate

// Later, for each value:
weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
```

The quantization is **importance-weighted**: values with larger magnitude get more weight in the optimization. This is the "importance matrix" in IQ - it's not a literal matrix, but rather per-value weights that prioritize accurate representation of larger weights.

### Phase 2: Sign Handling with Parity Constraint

```c
for (int k = 0; k < 4; ++k) {
    int nflip = 0;
    uint8_t s = 0;
    for (int i = 0; i < 8; ++i) {
        if (xb[8*k + i] >= 0) 
            xval[8*k + i] = xb[8*k + i];
        else {
            xval[8*k + i] = -xb[8*k + i];  // Make positive
            ++nflip; 
            s |= (1 << i);  // Record sign
        }
    }
    // Enforce even parity
    if (nflip % 2) {
        // Find the value with smallest weighted magnitude
        int imin = 0; 
        float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
        for (int i = 1; i < 8; ++i) {
            float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
            if (ax < min) { min = ax; imin = i; }
        }
        // Flip the least important sign
        xval[8*k+imin] = -xval[8*k+imin];
        s ^= (1 << imin);
    }
    block_signs[k] = s & 127;  // Store only 7 bits
}
```

This is beautiful! When there's an odd number of negatives (violating parity), the algorithm finds the **least important** value and flips its sign. The 8th sign bit is implicit from parity.

### Phase 3: Scale Search and Grid Matching

```c
float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
float best = 0;

for (int is = -6; is <= 6; ++is) {  // Search around initial scale
    float id = (2*kMaxQ-1+is*0.1f)/eff_max;
    float this_scale = 1/id;
    
    for (int k = 0; k < 4; ++k) {
        // Quantize to 2-bit indices
        for (int i = 0; i < 8; ++i) {
            int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
            Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
        }
        
        // Check if this combination is on the grid
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
        int grid_index = kmap_q2xs[u];
        
        if (grid_index < 0) {
            // Not on grid! Find nearest neighbor
            const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, 
                                                  xval + 8*k, waux + 8*k, 
                                                  this_scale, Laux + 8*k);
        }
    }
    
    // Compute weighted reconstruction error
    float sumqx = 0, sumq2 = 0;
    for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2*Laux[i] + 1;  // Dequantized value
        sumqx += w*xval[i]*q;
        sumq2 += w*q*q;
    }
    
    // Keep best scale (least weighted MSE)
    if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
        scale = sumqx/sumq2; 
        best = scale*sumqx;
        memcpy(L, Laux, 32);
    }
}
```

### Phase 4: The Neighbor Search System

The initialization code (`iq2xs_init_impl`) builds a neighbor lookup for points not on the grid:

```c
// For each possible 16-bit combination (4^8 = 65536 total)
for (int i = 0; i < kmap_size; ++i) {
    if (kmap_q2xs[i] >= 0) continue;  // Already on grid
    
    // Compute distance to all grid points
    for (int j = 0; j < grid_size; ++j) {
        const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
        int d2 = 0;
        for (int k = 0; k < 8; ++k) 
            d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
        dist2[2*j+0] = d2;
        dist2[2*j+1] = j;
    }
    qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
    
    // Store nearest neighbors (typically 2 closest distinct distances)
    kmap_q2xs[i] = -(counter + 1);  // Negative = index into neighbors
    // ... store neighbor list
}
```

The `kmap_q2xs` array serves dual purpose:
- **Positive value**: direct grid index (point is on grid)
- **Negative value**: encoded pointer to neighbor list (point is off grid)

---

## Implementation Details You Should Care About

### 1. The Parity Trick Saves 12.5% on Signs

By constraining each 8-value group to have even parity, you encode 8 signs in 7 bits. Over 256 values, that's 256/8 * 7 = 224 sign bits instead of 256. This is a 12.5% savings on sign storage.

### 2. Hierarchical Scaling

IQ2 uses two levels of scales:
- **Block scale** (`d`): 16-bit float for the entire 256-value block
- **Group scales**: 4 bits per 32-value group (16 groups = 64 bits = 8 bytes for IQ2_XS)

The dequantization formula reveals the scale combination:
```c
db = d * (0.5f + group_scale) * 0.25f
```

The `0.5 + group_scale` with `group_scale` in [0,15] gives range [0.5, 15.5], then scaled by 0.25 to give effective multiplier range [0.125, 3.875]. This provides good dynamic range within each block.

### 3. Grid Selection is Critical

The grids (`kgrid_2bit_256`, etc.) aren't arbitrary - they're specifically chosen to minimize expected quantization error for neural network weights. The raw grid values like:

```c
static const uint16_t kgrid_2bit_256[256] = {
    0, 2, 5, 8, 10, 17, 20, 32, ...
};
```

Each 16-bit value encodes 8 × 2-bit indices. These were likely found through optimization on representative weight distributions.

### 4. Importance Weighting During Quantization

```c
weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
```

This weighting scheme:
- Prioritizes values with larger magnitude (they contribute more to output)
- Uses global variance (`sigma2`) as a baseline
- Accepts external weights (`qw`) for additional control (e.g., from activation statistics)

### 5. The IQ2_XS vs IQ2_XXS Difference

IQ2_XS uses a 512-entry grid (vs 256) and stores explicit per-group scales. Looking at `dequantize_row_iq2_xs`:

```c
for (int l = 0; l < 4; ++l) {
    const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
    const uint8_t signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
```

Here, each 16-bit `qs` entry contains:
- Bits 0-8: grid index (512 possibilities)
- Bits 9-15: 7-bit sign pattern

The separate `scales` array stores the group scales explicitly rather than packing them into the data stream.

### 6. IQ2_S Uses Even Larger Grid + High Bits

```c
const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
```

IQ2_S uses a 1024-entry grid, requiring 10 bits per grid index. It stores:
- 8 low bits in `qs`
- 2 high bits in `qh` (packed, 4 groups share one byte)
- Signs separately from the grid index

---

## Summary: The Complete Picture

**What IQ2 achieves**: Near-2-bit quantization by constraining 8 weights to lie on a carefully chosen 256/512/1024-point codebook, with smart sign encoding that exploits parity.

**The quantization pipeline**:
1. Compute importance weights based on magnitude and variance
2. Make all values positive, record signs, enforce even parity (flip least important if needed)
3. Search for optimal scale by trying candidates around initial estimate
4. For each scale candidate, snap to nearest grid point (direct or via neighbor search)
5. Pick scale that minimizes weighted reconstruction error
6. Pack grid indices, signs (7 bits), and scales into the block structure

**The dequantization pipeline**:
1. Load block scale
2. For each group: compute local scale, look up grid values, apply signs, multiply by scale
3. Output float values

**Key insights**:
- Grid-based quantization beats independent quantization by exploiting weight structure
- Parity constraint on signs saves 12.5% storage
- Hierarchical scales (block + group) give fine-grained dynamic range
- Importance weighting ensures large weights are accurately represented
- Precomputed neighbor lists make non-grid-point handling efficient

## References:
1. datastructure: [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
2. quantization: [ggml-quant.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
3. visual explanation: [Reverse-engineering GGUF | Post-Training Quantization](https://www.youtube.com/watch?v=vW30o4U9BFE)