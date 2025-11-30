# BitSqueeze

## Introduction

BitSqueeze is a tiny C library for compressing float32 tensors with GGML-style integer quantization (Q8\_0, Q4\_0, Q2\_K, IQ2\_XXS, IQ2\_XS, IQ2\_S, NF4, NVFP4), compact floating formats (FP4, MXFP4, NF4\_DQ, FP8, MXFP8, FP16, BF16), and Top-K sparsity. Implementations live in `src/`, headers in `include/`, and ready-to-run tests in `test/`. The focus is small, dependency-free C/C++ code that can be dropped into inference pipelines to trade accuracy for bandwidth.

## Quick start

Prereqs: C toolchain (gcc/clang) + make + bash. Currently well tested on macOS and Linux with both C and C++ builds (gcc/g++); set `CC` in the `Makefile` if you want to compile with C++ instead. On Linux, OpenMP is enabled to parallelize super-block compression where safe.

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

## OMP comparison (i9-13900K, Linux)

OpenMP is used on Linux to parallelize compression across super blocks. The table below shows results on an i9-13900K with OMP support enabled:

| Method | B/W | Comp(ms) | Decomp(ms) | w/o omp Comp(ms) | w/o omp Decomp(ms) | MAE | MSE | MaxAbs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IQ2\_XXS** | **2.06262** | 61.984 | 0.980 | 964.637 | 10.593 | 1.541779 | 3.647489 | 10.516134 |
| **IQ2\_XS** | **2.31264** | 113.537 | 0.850 | 1820.878 | 10.894 | 1.309543 | 2.732195 | 10.169125 |
| **IQ2\_S** | **2.56265** | 34.700 | 0.764 | 654.450 | 10.710 | 1.101184 | 1.843772 | 6.522045 |
| **Q2\_K** | **2.62512** | 2.814 | 0.314 | 9.956 | 2.422 | 1.335282 | 2.578281 | 3.328746 |
| **FP4** | **4.00011** | 5.058 | 1.670 | 31.585 | 25.351 | 0.485991 | 0.404891 | 1.666666 |
| **NF4\_DQ** | **4.12515** | 2.986 | 0.302 | 44.733 | 3.571 | 0.413326 | 0.285641 | 1.519035 |
| **MXFP4** | **4.25014** | 2.724 | 2.252 | 37.601 | 27.340 | 0.500091 | 0.433483 | 1.999999 |
| **NF4** | **4.50014** | 3.358 | 0.438 | 44.226 | 3.111 | 0.404978 | 0.277936 | 1.518667 |
| **NVFP4** | **4.50015** | 6.023 | 2.202 | 41.141 | 26.779 | 0.440718 | 0.342614 | 1.666662 |
| **TOPK0.10** | **4.79893** | 218.125 | 1.498 | 217.062 | 1.803 | 4.049933 | 24.299142 | 9.096730 |
| **Q4\_0** | **5.00014** | 1.045 | 0.370 | 9.626 | 3.876 | 0.335517 | 0.155077 | 0.714198 |
| **FP8** | **8.00011** | 5.132 | 2.689 | 15.404 | 22.261 | 0.110551 | 0.021696 | 0.357142 |
| **MXFP8** | **8.25014** | 2.015 | 1.279 | 19.663 | 23.614 | 0.116666 | 0.026187 | 0.500000 |
| **Q8\_0** | **9.00014** | 1.304 | 0.247 | 8.055 | 1.536 | 0.018494 | 0.000471 | 0.039367 |
| **TOPK0.20** | **9.59776** | 218.329 | 1.921 | 219.847 | 2.082 | 3.200113 | 17.066934 | 8.120811 |
| **BF16** | **16.00009** | 2.410 | 0.460 | 5.181 | 2.920 | 0.007291 | 0.000102 | 0.031250 |
| **FP16** | **16.00009** | 2.551 | 0.465 | 7.378 | 5.124 | 0.000912 | 0.000002 | 0.003906 |

## License and contribution

  - License: MIT (see `LICENSE`).
  - Contributions: Issues and PRs welcome. Please keep changes focused, add/refresh tests under `test/`, and follow the existing C11 style (`-Wall -Wextra -Wpedantic`).
