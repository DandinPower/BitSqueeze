# BitSqueeze

## Introduction

BitSqueeze is a tiny C library for compressing float32 tensors with GGML-style integer quantization (Q8\_0, Q4\_0, Q2\_K, IQ2\_XXS, IQ2\_XS, IQ2\_S, NF4, NVFP4), compact floating formats (FP4, MXFP4, NF4\_DQ, FP8, MXFP8, FP16, BF16), and Top-K sparsity. Implementations live in `src/`, headers in `include/`, and ready-to-run tests in `test/`. The focus is small, dependency-free C11 code that can be dropped into inference pipelines to trade accuracy for bandwidth.

## Quick start

Prereqs: C toolchain (gcc/clang) + make + bash.

You can build the library, compile all tests and run the benchmark by following commands:

```bash
make
bash run_all_tests.sh
```

This script will compile the project into `build/`, execute every `test_*` binary, and generate a performance summary table at the end.

## bitsqueeze API essentials

### Core types

  - `bsq_method_t` methods:
      - Integer: `Q8_0`, `Q4_0`, `Q2_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`
      - Float: `BF16`, `FP16`, `FP8`, `MXFP8`, `FP4`, `MXFP4`, `NVFP4`, `NF4`, `NF4_DQ`
      - Sparse: `TOPK`
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
if (bsq_compress_1d(src, N, IQ2_XS, &buf) == 0) {
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

The following results were generated using `run_all_tests.sh` on **5 arrays of length 4,194,304** (Top-K uses 512x8,192).

**Test Environment:** Macbook Pro 2023, 16-inch, M2 Max with 32GB RAM.

| Method | B/W | Comp(ms) | Decomp(ms) | MAE | MSE | MaxAbs | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IQ2\_XXS** | **2.06262** | 954.178 | 2.156 | 1.541585 | 3.647318 | 10.177097 | 256-entry grid, 2-bit quantization |
| **IQ2\_XS** | **2.31264** | 1786.841 | 2.319 | 1.309299 | 2.731655 | 9.922921 | 512-entry grid, 2.31 bpw |
| **IQ2\_S** | **2.56265** | 571.447 | 2.309 | 1.101375 | 1.844577 | 6.680949 | 1024-entry grid, 2.56 bpw |
| **Q2\_K** | **2.62512** | 7.278 | 2.272 | 1.335575 | 2.578867 | 3.329085 | K-quants, 2 bits + scales |
| **FP4** | **4.00011** | 26.612 | 6.665 | 0.486186 | 0.405222 | 1.666666 | Tiny float, 1 exponent bit |
| **NF4\_DQ** | **4.12515** | 34.503 | 2.521 | 0.413350 | 0.285706 | 1.519034 | NF4 with double-quantized scales |
| **MXFP4** | **4.25014** | 21.262 | 7.612 | 0.499998 | 0.433414 | 1.999998 | Mixed-precision 4-bit |
| **NF4** | **4.50014** | 33.871 | 2.551 | 0.405029 | 0.278039 | 1.518812 | Normal-fused 4-bit |
| **NVFP4** | **4.50015** | 41.014 | 7.409 | 0.440844 | 0.342865 | 1.666663 | NVIDIA FP4 (Block + Tensor scale) |
| **TOPK0.10** | **4.79893** | 247.814 | 0.620 | 4.050239 | 24.303226 | 9.098263 | Keeps 10% largest values |
| **Q4\_0** | **5.00014** | 8.862 | 2.400 | 0.335426 | 0.155008 | 0.714212 | 4-bit per 32-value block |
| **FP8** | **8.00011** | 25.218 | 5.876 | 0.110532 | 0.021690 | 0.357143 | 8-bit float |
| **MXFP8** | **8.25014** | 23.295 | 7.393 | 0.116669 | 0.026197 | 0.499999 | Mixed-precision 8-bit |
| **Q8\_0** | **9.00014** | 8.801 | 0.634 | 0.018493 | 0.000471 | 0.039366 | 8-bit per 32-value block |
| **TOPK0.20** | **9.59776** | 250.321 | 0.918 | 3.200389 | 17.070546 | 8.125494 | Keeps 20% largest values |
| **BF16** | **16.00009** | 2.113 | 0.668 | 0.007294 | 0.000102 | 0.031250 | BF16 mantissa drop |
| **FP16** | **16.00009** | 1.353 | 0.636 | 0.000912 | 0.000002 | 0.003906 | 2-byte IEEE half |

*Originals are 32 bits per value. B/W = Bits per Weight (lower is smaller storage).*

## License and contribution

  - License: MIT (see `LICENSE`).
  - Contributions: Issues and PRs welcome. Please keep changes focused, add/refresh tests under `test/`, and follow the existing C11 style (`-Wall -Wextra -Wpedantic`).