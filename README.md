# BitSqueeze

## Introduction
BitSqueeze is a tiny C library for compressing float32 tensors with GGML-style integer quantization (Q8_0, Q4_0, Q2_K), floating-point downcasts (FP16, BF16), and Top-K sparsity. Implementations live in src/, headers in include/, and ready-to-run tests in test/. The focus is small, dependency-free C11 code that can be dropped into inference pipelines to trade accuracy for bandwidth.

## Quick start (pre-written tests)
Prereqs: C toolchain + make. All binaries land in `build/`.

```bash
# Build everything
make

# 1D quantizers and float downcasts
./build/test_q8_0_impl
./build/test_q4_0_impl
./build/test_q2_k_impl
./build/test_fp16_impl
./build/test_bf16_impl

# 2D Top-K sparsity
./build/test_topk_impl
```

Notes:
- Each test emits per-array size and error metrics (MAE, MSE, MaxAbs).
- `test_real_example` and `test_datatype` are currently out-of-date; ignore them for now.

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
The following results come from the bundled tests (uniform random floats in [-10, 10]). `B/W` is bits-per-value; lower means smaller storage. Originals are 32 b/value.

| Method        | Shape                         | Packed size (KB) | B/W     | MAE      | MSE        | MaxAbs   | Notes                     |
|---------------|-------------------------------|------------------|---------|----------|------------|----------|---------------------------|
| FP16          | N=1,048,576                   | 2,048.047        | 16.00037 | 0.000912 | 0.000002   | 0.003906 | 2-byte IEEE half          |
| BF16          | N=1,048,576                   | 2,048.047        | 16.00037 | 0.007297 | 0.000103   | 0.031250 | BF16 mantissa drop        |
| Q8_0          | N=4,194,304                   | 4,608.070        | 9.00014  | 0.018492 | 0.000471   | 0.039367 | 8-bit per 32-value block  |
| Q4_0          | N=4,194,304                   | 2,560.070        | 5.00014  | 0.335426 | 0.155028   | 0.714271 | 4-bit per 32-value block  |
| Q2_K          | N=4,194,304 (sb=256, b=16)    | 1,344.062        | 2.62512  | 1.335775 | 2.579      | 3.330325 | K-quants, 2 bits + scales |
| TOPK (20%)    | 512x8,192 tokens/features     | 4,914.055        | 9.59776  | 3.200    | 17.07      | 8.117    | Keeps 20% largest values  |
| TOPK (10%)    | 512x8,192 tokens/features     | 2,457.055        | 4.79893  | 4.050    | 24.30      | 9.121    | Keeps 10% largest values  |

## License and contribution
- License: MIT (see `LICENSE`).
- Contributions: Issues and PRs welcome. Please keep changes focused, add/refresh tests under `test/`, and follow the existing C11 style (`-Wall -Wextra -Wpedantic`).
