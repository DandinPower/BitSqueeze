# BitSqueeze

## Introduction

BitSqueeze is a tiny C library for compressing float32 tensors with GGML-style integer quantization (Q8\_0, Q4\_0, Q2\_K, Q2\_K\_FAST, IQ2\_XXS, IQ2\_XS, IQ2\_S, NF4, NVFP4), compact floating formats (FP4, MXFP4, NF4\_DQ, FP8, MXFP8, FP16, BF16), and Top-K sparsity (either absolute-value TOPK or user-supplied importance via TOPK_IM). Implementations live in `src/`, headers in `include/`, and ready-to-run tests in `test/`. The focus is small, dependency-free C/C++ code that can be dropped into inference pipelines to trade accuracy for bandwidth.

## Important Notes

**Cross-Platform Compatibility**: Current serialization implementations are not portable across different architectures. You must ensure that the machine loading a BitSqueeze buffer shares the same endianness and bit-width (32-bit vs. 64-bit) as the machine that created it.

**Risk**: Loading a buffer on a mismatched architecture will cause a segmentation fault. A fix for endian-swapping and architecture-agnostic headers is planned for a future update.

## Quick start

**Prerequisites:** C toolchain (gcc/clang), CMake (3.10+), and Make/Ninja.

You can build the library, compile all tests, and run the benchmark using the standard CMake workflow:

### Configure
`cmake -B build -DCMAKE_BUILD_TYPE=Release`

### Build
`cmake --build build --config Release`

### Run Tests (Two Options)

#### Option A: Standard CMake testing (Checks pass/fail only)
`cd build && ctest --output-on-failure`

#### Option B: Run custom benchmark script (Generates performance table)

Must be run from project root: `bash run_all_tests.sh build`

## bitsqueeze API essentials

### Core types

  - `bsq_method_t` methods:
      - Integer: `Q8_0`, `Q4_0`, `Q2_K`, `Q2_K_FAST`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`
      - Float: `BF16`, `FP16`, `FP8`, `MXFP8`, `FP4`, `MXFP4`, `NVFP4`, `NF4`, `NF4_DQ`
      - Sparse: `TOPK`, `TOPK_IM`
  - `bsq_shape_t`: captures 1D length or 2D token/feature counts (plus requested `sparse_ratio` for TOPK/TOPK_IM).
  - `bitsqueeze_buffer_t`: opaque holder for compressed payloads. Always free with `bsq_free`.

### Entry points

  - `bsq_compress_1d(const float *src, uint64_t num_elements, bsq_method_t method, bitsqueeze_buffer_t **out, const float *im);` (im currently only support Q2_K)
  - `bsq_compress_2d(const float *src, uint16_t num_tokens, uint16_t num_features, float sparse_ratio, bsq_method_t method, bitsqueeze_buffer_t **out, const float *im);` (use with `TOPK` or `TOPK_IM`; pass `NULL` for `TOPK`)
  - `bsq_decompress(const bitsqueeze_buffer_t *buf, float *dst, uint64_t dst_num_elements);`
  - `bsq_apply(const bitsqueeze_buffer_t *buf, float *dst, uint64_t dst_num_elements);` (applies sparse values, used with `TOPK_IM`)
  - `bsq_get_packed_size(const bitsqueeze_buffer_t *buf);` returns packed byte count.
  - `load_bsq_from_buffer(const void *buffer, int64_t buffer_size);` to rehydrate from serialized bytes.
  - `bsq_free(bitsqueeze_buffer_t *buf);`

### Minimal 1D usage

```c
#include "bitsqueeze.h"

const uint64_t N = 1048576;
float *src = ...;  /* your float32 data */

bitsqueeze_buffer_t *buf = NULL;
if (bsq_compress_1d(src, N, IQ2_XS, &buf, NULL) == 0) {
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
if (bsq_compress_2d(src, TOKENS, FEATURES, SPARSE_RATIO, TOPK, &buf, NULL) == 0) {
    float *dst = malloc(N * sizeof(float));
    bsq_decompress(buf, dst, N);
    bsq_free(buf);
    free(dst);
}
```

### Minimal 2D TOPK_IM usage
```c
#include "bitsqueeze.h"

const uint16_t TOKENS = 512, FEATURES = 8192;
const float SPARSE_RATIO = 0.1f; /* keep top 10% features per token */
const uint64_t N = (uint64_t)TOKENS * FEATURES;
float *src = ...;       /* flattened row-major [TOKENS, FEATURES] */
float *importance = ...;/* same shape as src; values used directly (no abs) */

bitsqueeze_buffer_t *buf = NULL;
if (bsq_compress_2d(src, TOKENS, FEATURES, SPARSE_RATIO, TOPK_IM, &buf, importance) == 0) {
    float *dst = malloc(N * sizeof(float));
    bsq_decompress(buf, dst, N); /* or use bsq_apply to overwrite existing values */
    bsq_free(buf);
    free(dst);
}
```

## Current method comparison (synthetic random data)

The following results were generated using `run_all_tests.sh` on **5 arrays of length 4,194,304** (Top-K uses 512x8,192).

**Test Environment:** Macbook Pro 2023, 16-inch, M2 Max with 32GB RAM. (Without OpenMP)

| Method | B/W | Comp(ms) | Decomp(ms) | MAE | MSE | MaxAbs | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TOPK0.01** | **0.48058** | 9.635 | 0.263 | 4.900416 | 32.343510 | 9.929792 | Keeps 1% largest values |
| **IQ2\_XXS** | **2.06262** | 903.515 | 2.167 | 1.541585 | 3.647318 | 10.177097 | 256-entry grid, 2-bit quantization |
| **IQ2\_XS** | **2.31264** | 1749.120 | 2.168 | 1.309299 | 2.731655 | 9.922921 | 512-entry grid, 2.31 bpw |
| **TOPK0.05** | **2.40245** | 30.680 | 0.457 | 4.512059 | 28.576108 | 9.566334 | Keeps 5% largest values |
| **IQ2\_S** | **2.56265** | 570.729 | 2.384 | 1.101375 | 1.844577 | 6.680949 | 1024-entry grid, 2.56 bpw |
| **Q2\_K** | **2.62512** | 164.133 | 2.273 | 1.127911 | 1.867995 | 4.753401 | K-quants with optimal scale/min search (more compute, better accuracy) |
| **Q2\_K\_FAST** | **2.62512** | 6.956 | 2.259 | 1.335575 | 2.578867 | 3.329085 | K-quants without scale/min search (faster, lower accuracy) |
| **FP4** | **4.00011** | 27.737 | 6.882 | 0.486186 | 0.405222 | 1.666666 | Tiny float, 1 exponent bit |
| **NF4\_DQ** | **4.12515** | 33.438 | 2.593 | 0.413350 | 0.285706 | 1.519034 | NF4 with double-quantized scales |
| **MXFP4** | **4.25014** | 21.915 | 7.516 | 0.499998 | 0.433414 | 1.999998 | Mixed-precision 4-bit |
| **NF4** | **4.50014** | 33.344 | 2.476 | 0.405029 | 0.278039 | 1.518812 | Normal-fused 4-bit |
| **NVFP4** | **4.50015** | 39.699 | 7.272 | 0.440844 | 0.342865 | 1.666663 | NVIDIA FP4 (Block + Tensor scale) |
| **TOPK0.10** | **4.79893** | 50.754 | 0.587 | 4.050239 | 24.303226 | 9.098263 | Keeps 10% largest values |
| **Q4\_0** | **5.00014** | 8.866 | 2.419 | 0.335426 | 0.155008 | 0.714212 | 4-bit per 32-value block |
| **FP8** | **8.00011** | 26.708 | 6.368 | 0.110532 | 0.021690 | 0.357143 | 8-bit float |
| **MXFP8** | **8.25014** | 22.171 | 6.847 | 0.116669 | 0.026197 | 0.499999 | Mixed-precision 8-bit |
| **Q8\_0** | **9.00014** | 8.723 | 0.610 | 0.018493 | 0.000471 | 0.039366 | 8-bit per 32-value block |
| **TOPK0.20** | **9.59776** | 78.810 | 0.719 | 3.200389 | 17.070546 | 8.125494 | Keeps 20% largest values |
| **TOPK0.30** | **14.40245** | 100.999 | 0.872 | 2.449609 | 11.430751 | 7.149782 | Keeps 30% largest values |
| **BF16** | **16.00009** | 2.144 | 0.626 | 0.007294 | 0.000102 | 0.031250 | BF16 mantissa drop |
| **FP16** | **16.00009** | 1.412 | 0.622 | 0.000912 | 0.000002 | 0.003906 | 2-byte IEEE half |
| **TOPK0.40** | **19.20128** | 97.600 | 0.954 | 1.799844 | 7.199153 | 6.163503 | Keeps 40% largest values |
| **TOPK0.50** | **24.00011** | 90.549 | 1.052 | 1.250046 | 4.167010 | 5.165112 | Keeps 50% largest values |
| **TOPK0.60** | **28.79893** | 89.426 | 1.377 | 0.800220 | 2.134463 | 4.154192 | Keeps 60% largest values |

*Originals are 32 bits per value. B/W = Bits per Weight (lower is smaller storage).*

## OMP-enabled results (i9-13900K, Linux)

OpenMP is used on Linux to parallelize compression across super blocks. The table below shows results on an i9-13900K with OMP support enabled:

| Method | B/W | Comp(ms) | Decomp(ms) | MAE | MSE | MaxAbs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TOPK0.01** | **0.48058** | 0.423 | 0.433 | 4.900092 | 32.339085 | 9.931082 |
| **IQ2\_XXS** | **2.06262** | 60.418 | 0.763 | 1.541779 | 3.647489 | 10.516134 |
| **IQ2\_XS** | **2.31264** | 111.861 | 1.050 | 1.309543 | 2.732195 | 10.169125 |
| **TOPK0.05** | **2.40245** | 1.313 | 0.518 | 4.511744 | 28.571871 | 9.572968 |
| **IQ2\_S** | **2.56265** | 36.413 | 0.769 | 1.101184 | 1.843772 | 6.522045 |
| **Q2\_K** | **2.62512** | 15.540 | 0.366 | 1.128283 | 1.868942 | 4.596894 |
| **Q2\_K\_FAST** | **2.62512** | 2.857 | 0.320 | 1.335282 | 2.578281 | 3.328746 |
| **FP4** | **4.00011** | 5.513 | 2.285 | 0.485991 | 0.404891 | 1.666666 |
| **NF4\_DQ** | **4.12515** | 3.667 | 0.394 | 0.413326 | 0.285641 | 1.519035 |
| **MXFP4** | **4.25014** | 2.866 | 1.951 | 0.500091 | 0.433483 | 1.999999 |
| **NF4** | **4.50014** | 3.642 | 0.452 | 0.404978 | 0.277936 | 1.518667 |
| **NVFP4** | **4.50015** | 6.095 | 2.412 | 0.440718 | 0.342614 | 1.666662 |
| **TOPK0.10** | **4.79893** | 2.217 | 0.685 | 4.049933 | 24.299142 | 9.096730 |
| **Q4\_0** | **5.00014** | 1.302 | 0.356 | 0.335517 | 0.155077 | 0.714198 |
| **FP8** | **8.00011** | 4.983 | 1.625 | 0.110551 | 0.021696 | 0.357142 |
| **MXFP8** | **8.25014** | 2.011 | 1.308 | 0.116666 | 0.026187 | 0.500000 |
| **Q8\_0** | **9.00014** | 1.383 | 0.256 | 0.018494 | 0.000471 | 0.039367 |
| **TOPK0.20** | **9.59776** | 3.791 | 0.765 | 3.200113 | 17.066934 | 8.120811 |
| **TOPK0.30** | **14.40245** | 4.927 | 3.626 | 2.449429 | 11.428593 | 7.158865 |
| **BF16** | **16.00009** | 2.504 | 0.455 | 0.007291 | 0.000102 | 0.031250 |
| **FP16** | **16.00009** | 2.492 | 0.461 | 0.000912 | 0.000002 | 0.003906 |
| **TOPK0.40** | **19.20128** | 6.051 | 3.699 | 1.799659 | 7.196919 | 6.160163 |
| **TOPK0.50** | **24.00011** | 9.429 | 3.952 | 1.249968 | 4.165911 | 5.170439 |
| **TOPK0.60** | **28.79893** | 6.036 | 3.674 | 0.800245 | 2.134279 | 4.185278 |

## Integration via CMake FetchContent

The recommended way to integrate BitSqueeze into your project is using CMake's `FetchContent`. This module automatically downloads and builds the library as a dependency.
Add the following configuration to your project's `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  bitsqueeze
  GIT_REPOSITORY https://github.com/DandinPower/BitSqueeze.git
  GIT_TAG        v0.1.3
)

# Disable BitSqueeze tests to speed up your build
set(BITSQUEEZE_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(bitsqueeze)

# Link your executable against the library alias
add_executable(your_app main.c)
target_link_libraries(your_app PRIVATE BitSqueeze::bitsqueeze)
```

For a complete, working implementation of this integration method, refer to the project in `examples/cmake_fetch/`.

## Installation as a Shared Library

If you prefer to install BitSqueeze system-wide or need to link against it using a non-CMake build system (such as raw Makefiles), you can build it as a shared library.

### 1. Build and Install

Use the `BUILD_SHARED_LIBS` option to generate a `.so` (Linux) or `.dylib` (macOS) file, and then install it to your system paths.

```bash
# 1. Configure with shared libraries enabled
cmake -B build_shared -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

# 2. Build the library
cmake --build build_shared --config Release

# 3. Install (requires sudo for system directories like /usr/local)
sudo cmake --install build_shared

sudo ldconfig
```

### 2. Linking

The installed library is named `libbitsqz`. When compiling your own projects, link against it using the `-lbitsqz` flag. (remember to copy the bitqueeze.h into your include folder)

```bash
gcc main.c -I include -o my_app -lbitsqz
```

### 3. Example Project

A complete, standalone example demonstrating how to link against the installed shared library using a standard `Makefile` can be found in:

`examples/shared_library/`

This example assumes you have already run the installation steps above. To run it, simply navigate to that directory and type `make`.

## License and contribution

  - License: MIT (see `LICENSE`).
  - Contributions: Issues and PRs welcome. Please keep changes focused, add/refresh tests under `test/`, and follow the existing C11 style (`-Wall -Wextra -Wpedantic`).
