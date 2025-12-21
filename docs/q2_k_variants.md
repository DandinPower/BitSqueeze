# Q2_K Variants Report

## Scope

This report documents the Q2_K and Q2_K_FAST integer quantization variants in BitSqueeze. It focuses on the shared packed format, the correct Q2_K scale/min search, and how Q2_K_FAST trades accuracy for speed. Sources: `include/int_quantization/q2_k_impl.h`, `src/int_quantization/q2_k_impl.c`, and `src/int_quantization/q2_k_fast_impl.c`.

## Shared format and parameters

Both variants use the same data layout and decode path.

- Block shape:
  - `Q2_K_BLOCK_SIZE = 16` values per block
  - `Q2_K_SUPER_BLOCK_SIZE = 16` blocks per super-block
  - `WEIGHT_PER_SUPER_BLOCK = 256` values
- Storage layout (`super_block_q2_k`):
  - `super_scale` (fp16): scale for per-block scales
  - `super_min` (fp16): scale for per-block mins
  - `scales[16]` (1 byte each):
    - low nibble (bits 0..3): scale in [0, 15]
    - high nibble (bits 4..7): signed min in [-8, 7]
  - `data[64]` (2 bits per weight for 256 values)
- Bits per weight: (256 * 2 + 16 * 8 + 2 * 16) / 256 = 2.625 bpw

Reconstruction per block `j` (used by both variants):

```text
scale_j = super_scale * (scales[j] & 0x0F)
min_j   = super_min * sign_extend(scales[j] >> 4)
x_hat   = min_j + scale_j * q   where q in {0,1,2,3}
```

`sign_extend` is implemented as `(int8_t)(min_q << 4) >> 4` to preserve the signed 4-bit range.

## Correct the implementation of Q2_K

The Q2_K variant uses a per-block optimization to choose scale and min that reduce weighted error, matching the ggml Q2_K algorithm referenced in the code.

### Per-block optimal scale/min (`find_optimal_scale_and_min`)

For each 16-value block:

1. Compute `min`, `max`, and absolute weights `abs(x[i])`.
2. Clamp `min` to zero if all values are positive (the format expects `min <= 0`).
3. If `max == min`, set scale to 0 and all quantized values to 0.
4. Initialize a candidate scale from the `[min, max]` range.
5. Perform a small search over 16 candidate `iscale` values (range is based on the ggml recipe).
6. For each candidate:
   - Quantize into 2-bit indices `l` in [0, 3].
   - Solve for `scale` and `min` via weighted least squares.
   - Clamp `min` again to keep it <= 0.
   - Compute weighted L1 error and keep the best parameters.

The result is a block-specific `(scale, min)` pair that favors larger-magnitude weights (since errors are weighted by `abs(x[i])`).

### Super-block packing (correct Q2_K flow)

For each super-block of 256 values:

1. Compute `scale[j]` and `min[j]` for each block using the optimal search.
2. Find:
   - `max_scale = max(scale[j])`
   - `max_abs_min = max(abs(min[j]))`
3. Quantize per-block scales into the low nibble:
   - `super_scale = fp16(max_scale / 15)`
   - `scales[j].low = round(15 * scale[j] / max_scale)`
4. Quantize per-block mins into the high nibble:
   - `super_min = fp16(max_abs_min / 7)`
   - `scales[j].high = clamp(round(7 * min[j] / max_abs_min), -8, 7)`
5. Quantize each value to 2-bit `q`:
   - `q = round((x - min_j) / scale_j)`, clamp to [0, 3]
6. Pack 4x 2-bit values into each byte (see layout below).

## Q2_K_FAST (fast min-max variant)

Q2_K_FAST preserves the exact same packed format and decode logic, but uses a fast, non-iterative scale/min for each block:

```text
min_j   = min(x)
scale_j = (max(x) - min(x)) / 3
```

This maps the 2-bit range [0, 3] directly to the block's min/max interval. The rest of the encoding (super-block quantization, nibble packing, and data packing) is identical to Q2_K.

Tradeoff summary:

- Q2_K: higher compute cost, better accuracy (optimal search).
- Q2_K_FAST: much faster compression, lower accuracy (min-max only).

## 2-bit packing layout

The 256 2-bit values are packed into 64 bytes. Each byte stores four 2-bit values from a strided pattern:

```text
byte = L[i + 0] | (L[i + 32] << 2) | (L[i + 64] << 4) | (L[i + 96] << 6)
```

This is performed twice: once for indices [0..127] and once for [128..255]. Decompression mirrors this exact mapping.

## API integration and tests

- API:
  - `bsq_compress_1d(..., Q2_K, ...)` calls `q2_k_compress`
  - `bsq_compress_1d(..., Q2_K_FAST, ...)` calls `q2_k_fast_compress`
  - `bsq_decompress` dispatches to `q2_k_decompress` or `q2_k_fast_decompress` (same decode path)
- Tests:
  - `test/test_q2_k_impl.c`
  - `test/test_q2_k_fast_impl.c`

These tests generate random inputs, compress/decompress, and report size, bits-per-weight, and error metrics (MAE/MSE/MaxAbs).