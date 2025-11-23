# BitSqueeze

## Introduction
BitSqueeze is a tiny C library for compressing float32 tensors with GGML-style integer quantization (Q8_0, Q4_0, Q2_K, NF4, NVFP4), compact floating formats (FP4, MXFP4, NF4_DQ, FP8, MXFP8, FP16, BF16), and Top-K sparsity. Implementations live in src/, headers in include/, and ready-to-run tests in test/. The focus is small, dependency-free C11 code that can be dropped into inference pipelines to trade accuracy for bandwidth.

## Quick start (pre-written tests)
Prereqs: C toolchain + make. All binaries land in `build/`.

```bash
# Build everything
make

# 1D quantizers and float downcasts
./build/test_q2_k_impl
./build/test_q4_0_impl
./build/test_q8_0_impl
./build/test_nf4_impl
./build/test_nf4_dq_impl
./build/test_nvfp4_impl
./build/test_fp4_impl
./build/test_fp8_impl
./build/test_fp16_impl
./build/test_bf16_impl
./build/test_mxfp4_impl
./build/test_mxfp8_impl

# 2D Top-K sparsity
./build/test_topk_impl
```

Notes:
- Each test emits per-array size and error metrics (MAE, MSE, MaxAbs).
- `test/legacy` is currently out-of-date; ignore it for now.

## bitsqueeze API essentials

### Core types
- `bsq_method_t` methods: `Q8_0`, `Q4_0`, `Q2_K`, `TOPK`, `BF16`, `FP16`.
- `bsq_shape_t`: captures 1D length or 2D token/feature counts (plus requested `sparse_ratio` for TOPK).
- `bitsqueeze_buffer_t`: opaque holder for compressed payloads. Always free with `bsq_free`.

### Entry points
- `bsq_compress_1d(const float *src, uint64_t num_elements, bsq_method_t method, bitsqueeze_buffer_t **out);`
- `bsq_compress_2d(const float *src, uint16_t num_tokens, uint16_t num_features, float sparse_ratio, bsq_method_t method, bitsqueeze_buffer_t **out);` (use with `TOPK`)
- `bsq_decompress(const bitsqueeze_buffer_t *buf, float *dst, uint64_t dst_num_elements);`
- `bsq_get_packed_size(const bitsqueeze_buffer_t *buf);` returns packed byte count.
- `load_bsq_from_buffer(const void *buffer, int64_t buffer_size);` to rehydrate from serialized bytes.
- `bsq_free(bitsqueeze_buffer_t *buf);`

### Minimal 1D usage
```c
#include "bitsqueeze.h"

const uint64_t N = 1048576;
float *src = ...;  /* your float32 data */

bitsqueeze_buffer_t *buf = NULL;
if (bsq_compress_1d(src, N, Q8_0, &buf) == 0) {
    float *dst = malloc(N * sizeof(float));
    bsq_decompress(buf, dst, N);

    int64_t packed_bytes = bsq_get_packed_size(buf);
    /* ... use dst ... */

    free(dst);
    bsq_free(buf);
}
```

### Minimal 2D TOPK usage
```c
#include "bitsqueeze.h"

const uint16_t TOKENS = 512, FEATURES = 8192;
const float SPARSE_RATIO = 0.1f; /* keep top 10% features per token */
const uint64_t N = (uint64_t)TOKENS * FEATURES;
float *src = ...;  /* flattened row-major [TOKENS, FEATURES] */

bitsqueeze_buffer_t *buf = NULL;
if (bsq_compress_2d(src, TOKENS, FEATURES, SPARSE_RATIO, TOPK, &buf) == 0) {
    float *dst = malloc(N * sizeof(float));
    bsq_decompress(buf, dst, N);
    bsq_free(buf);
    free(dst);
}
```

## Current method comparison (synthetic random data)
The following results come from the bundled tests using 5 arrays of length 4,194,304 drawn uniformly from [-10, 10] (Top-K uses 512x8,192). `B/W` is bits-per-value; lower means smaller storage. Originals are 32 b/value.

| Method        | Shape                         | Packed size (KB) | B/W      | MAE       | MSE        | MaxAbs   | Notes                     |
|---------------|-------------------------------|------------------|----------|-----------|------------|----------|---------------------------|
| BF16          | N=4,194,304                   | 8,192.047        | 16.00009 | 0.007295  | 0.000102   | 0.031250 | BF16 mantissa drop        |
| FP16          | N=4,194,304                   | 8,192.047        | 16.00009 | 0.000912  | 0.000002   | 0.003906 | 2-byte IEEE half          |
| TOPK (20%)    | 512x8,192 tokens/features     | 4,914.055        | 9.59776  | 3.198804  | 17.057529  | 8.116994 | Keeps 20% largest values  |
| Q8_0          | N=4,194,304                   | 4,608.070        | 9.00014  | 0.018508  | 0.000472   | 0.039366 | 8-bit per 32-value block  |
| MXFP8         | N=4,194,304                   | 4,224.070        | 8.25014  | 0.116640  | 0.026185   | 0.499999 | Mixed-precision 8-bit     |
| FP8           | N=4,194,304                   | 4,096.055        | 8.00011  | 0.110497  | 0.021687   | 0.357143 | 8-bit float               |
| Q4_0          | N=4,194,304                   | 2,560.070        | 5.00014  | 0.335504  | 0.155091   | 0.714271 | 4-bit per 32-value block  |
| TOPK (10%)    | 512x8,192 tokens/features     | 2,457.055        | 4.79893  | 4.048771  | 24.292203  | 9.102604 | Keeps 10% largest values  |
| NVFP4         | N=4,194,304                   | 2,304.078        | 4.50015  | 0.440604  | 0.342659   | 1.666665 | NVIDIA FP4                |
| NF4           | N=4,194,304                   | 2,304.070        | 4.50014  | 0.404979  | 0.277983   | 1.518787 | Normal-fused 4-bit        |
| MXFP4         | N=4,194,304                   | 2,176.070        | 4.25014  | 0.500061  | 0.433629   | 1.999998 | Mixed-precision 4-bit     |
| NF4_DQ        | N=4,194,304                   | 2,112.078        | 4.12515  | 0.413354  | 0.285722   | 1.519034 | Normal-fused 4-bit, decompressed |
| FP4           | N=4,194,304                   | 2,048.055        | 4.00011  | 0.485940  | 0.404944   | 1.666666 | Tiny float, 1 exponent bit |
| Q2_K          | N=4,194,304 (sb=256, b=16)    | 1,344.062        | 2.62512  | 1.335691  | 2.579342   | 3.330325 | K-quants, 2 bits + scales |

## License and contribution
- License: MIT (see `LICENSE`).
- Contributions: Issues and PRs welcome. Please keep changes focused, add/refresh tests under `test/`, and follow the existing C11 style (`-Wall -Wextra -Wpedantic`).
