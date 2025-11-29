# Floating-point Quantization Formats: FP16/BF16 → FP8 → FP4/NF4

## 16-bit formats: FP16 and BF16

These share the same width (16 bits) but trade exponent vs mantissa differently.

![image.png](Floating-point%20Quantization%20Formats%20FP16%20BF16%20%E2%86%92%20FP/image.png)

### Format and properties

- **FP16 (half precision)**
    - Bit layout: 1 sign, 5 exponent, 10 mantissa bits.
    - Higher precision than BF16 for values within its range, but significantly narrower dynamic range than FP32.
- **BF16 (bfloat16)**
    - Bit layout: 1 sign, 8 exponent, 7 mantissa bits.
    - Same exponent range as FP32, so almost the same *dynamic range* but with much less precision.
    - Widely supported on TPUs and modern GPUs for large-model training.

### Quantization / dequantization

From an FP32 tensor:

- **Quantization**
    - Convert each FP32 value to FP16 or BF16 via IEEE rounding to nearest representable value.
    - No explicit scaling factor; the exponent/mantissa structure is fixed.
- **Dequantization**
    - Cast back to FP32. The cast is exact in the sense that it perfectly recovers the FP16/BF16 value encoded; the error is solely from the earlier rounding.

### Bits per weight (8,192-element tensor for 72b activation)

For both FP16 and BF16:

- Data: 16 bits per weight.
- Total: 8,192 × 16 = 131,072 bits.
- Effective bits/weight: **16.00**.

---

## 8-bit formats: FP8 E4M3 and E5M2 (general FP8)

Proposed by NVIDIA, INTEL, ARM: https://arxiv.org/abs/2209.05433

![Screenshot 2025-11-17 at 1.46.28 PM.png](Floating-point%20Quantization%20Formats%20FP16%20BF16%20%E2%86%92%20FP/Screenshot_2025-11-17_at_1.46.28_PM.png)

### Format and properties

- **E4M3**
    - 1 sign, 4 exponent, 3 mantissa bits.
    - Better *precision* inside a narrower dynamic range.
    - Typically used for *weights and activations* in FP8 training.
- **E5M2**
    - 1 sign, 5 exponent, 2 mantissa bits.
    - Larger dynamic range but less precision.
    - Typically used for *gradients*, which need range more than fine mantissa.

### Hypothesis and motivation

- Hypothesis: we can push precision down to 8 bits if we:
    - Carefully choose which tensors use which FP8 variant.
    - Introduce scaling factors per tensor (or block) instead of a single global loss scale.
- Motivation:
    - 2× memory reduction versus FP16.
    - Higher Tensor Core throughput, especially on Hopper/Blackwell.
    - With proper scaling “recipes” (*one* FP32 scale per tensor.), training convergence can match BF16 for many LLMs.

### Quantization / dequantization

From FP32 to FP8:

1. Choose a scaling factor `s` (FP32), e.g. via amax history or current tensor stats.
2. For each FP32 element `x`:
    - Compute `x_scaled = x / s`.
    - Round `x_scaled` to the nearest representable FP8 value in E4M3 or E5M2.
    - Store `x_fp8`.

Dequantization back to FP32:

- Read `x_fp8`, cast to FP32: `x_fp8_fp32`.
- Reconstruct `x_hat = s * x_fp8_fp32`.

The error comes from:

- Rounding into FP8.
- Any mismatch between the tensor’s actual distribution and the chosen `s`.

### Bits per weight (8,192-element tensor)

Assuming 1 FP32 scale per whole tensor:

- Data: 8 bits × 8,192 = 65,536 bits.
- Scale: 1 FP32 = 32 bits.
- Total bits: 65,536 + 32 = 65,568 bits.
- Effective bits/weight: 65,568 / 8,192 ≈ **8.00** bits (≈ 8.004).

---

## MXFP8 (micro-scaling FP8)

MXFP8 is a *block-scaled* FP8 scheme introduced by Open Compute Project: [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

![image.png](Floating-point%20Quantization%20Formats%20FP16%20BF16%20%E2%86%92%20FP/image%201.png)

### Format and properties

- Data: FP8 values in **E4M3** for all tensors.
- Scaling:
    - Tensor is partitioned into contiguous blocks of **32 values**.
    - Each 32-value block has its own scale, encoded as an **E8M0** 8-bit exponent-only format (power-of-two scaling).

### Hypothesis and motivation

- Standard FP8 with a *single* FP32 scale per tensor can struggle with tensors that have wide dynamic range; some values saturate or collapse to zero.
- Hypothesis: if we assign a scale per small block, each block can use the high-precision E4M3 format without needing E5M2.
- Motivation:
    - Improve FP8 accuracy by better matching local magnitude structure.
    - Still keep 8-bit data and compact 8-bit scales managed in hardware.

### Quantization / dequantization

Given FP32 tensor `X`:

1. Partition `X` into blocks of 32 consecutive values.
2. For each block `b`:
    - Select scale `s_b` as an E8M0 value (power-of-two) that best fits that block (e.g., via amax or MSE minimization).
    - Compute `x_scaled = X_b / s_b`.
    - Quantize `x_scaled` elementwise to FP8 E4M3, storing 8-bit codes.
3. Dequantization:
    - For each block, convert `s_b` (E8M0) to FP32.
    - Cast FP8 codes back to FP32 values, multiply by `s_b` to reconstruct.

### Bits per weight (8,192-element tensor)

- Data: 8 bits × 8,192 = 65,536 bits.
- Scales:
    - Blocks: 8,192 / 32 = 256 blocks.
    - One 8-bit E8M0 scale per block → 256 × 8 = 2,048 bits.
- Total: 65,536 + 2,048 = 67,584 bits.
- Effective bits/weight: 67,584 / 8,192 = **8.25** bits.

---

## FP4 family: FP4 (E2M1), MXFP4, NVFP4

Mainstream float4 datatypes include three main 4-bit floating point schemes based on an E2M1 core: FP4, MXFP4, and NVFP4.

All of them use a 4-bit float:

- 1 sign, 2 exponent, 1 mantissa bit.
- Values roughly in range about −6 to 6 (examples given: 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6, and negatives).

### Plain FP4 (E2M1 + scale)

**Format and properties**

- Data: 4-bit E2M1 FP plus a “FP32 Scale Factor”

**Hypothesis and motivation**

- Hypothesis: For some inference workloads, 4 bits are enough if we allow a good float scale.
- Motivation:
    - Up to 4× less memory than FP16.
    - But quantization error can be large; accuracy often noticeably worse than FP8.

**Quantization / dequantization**

From FP32 tensor `X`:

1. Choose a scale `s` (FP32) for the entire tensor or for a coarse block.
2. Quantization:
    - `x_scaled = X / s`.
    - Round `x_scaled` into the nearest representable E2M1 value, store as 4-bit FP4.
3. Dequantization:
    - Cast FP4 to FP32, then multiply by `s`.

**Bits per weight (8,192-element tensor)**

Assume one FP32 scale for entire tensor:

- Data: 4 bits × 8,192 = 32,768 bits.
- Scale: 32 bits.
- Total: 32,768 + 32 = 32,800 bits.
- Effective bits/weight: 32,800 / 8,192 ≈ **4.00** bits (≈ 4.004).

---

### MXFP4 (E2M1 + per-block E8M0 scale)

also introduced by Open Compute Project: [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

**Format and properties**

- Data: 4-bit E2M1.
- Scaling: One shared **power-of-two scale** (E8M0) per block of **32 values**.

**Hypothesis and motivation**

- FP4 with only a per-tensor scale can drop accuracy aggressively.
- Hypothesis: finer-grained per-block scaling cuts error significantly while keeping formats simple.
- Motivation:
    - Still up to 4× less memory than FP16.
    - Simpler scales (power-of-two) are cheap, but may be coarse so better than pure FP4

**Quantization / dequantization**

Given FP32 tensor:

1. Partition into 32-value blocks.
2. For each block:
    - Choose E8M0 scale `s_b` (power-of-two) that minimizes quantization error for that block.
    - Compute `x_scaled = X_b / s_b`.
    - Quantize `x_scaled` into 4-bit FP4 E2M1.
3. Dequantization:
    - `x_hat = s_b * cast_fp4_to_fp32(x_q)` for each value in the block.

**Bits per weight (8,192-element tensor)**

- Data: 4 bits × 8,192 = 32,768 bits.
- Scales:
    - Blocks: 8,192 / 32 = 256 blocks.
    - Each block has one 8-bit E8M0 scale → 256 × 8 = 2,048 bits.
- Total: 32,768 + 2,048 = 34,816 bits.
- Effective bits/weight: 34,816 / 8,192 = **4.25** bits.

---

### NVFP4 (E2M1 + FP8 E4M3 micro-block scale + FP32 tensor scale)

![Screenshot 2025-11-17 at 1.58.09 PM.png](Floating-point%20Quantization%20Formats%20FP16%20BF16%20%E2%86%92%20FP/Screenshot_2025-11-17_at_1.58.09_PM.png)

NVFP4 is a new 4-bit format introduced for Blackwell, designed to achieve near-FP8 accuracy at FP4 cost.

**Format and properties**

- Core data: 4-bit E2M1 (weights in −6 to 6-ish).
- Scaling:
    - Per 16-value micro-block: a shared FP8 **E4M3** scale factor.
    - Per tensor: a second FP32 scale factor.
- So reconstructed values look like:
    - `x ≈ S_tensor * s_block * x_q`, where:
        - `x_q` is 4-bit E2M1,
        - `s_block` is FP8 E4M3,
        - `S_tensor` is FP32.

**Hypothesis and motivation**

![image.png](Floating-point%20Quantization%20Formats%20FP16%20BF16%20%E2%86%92%20FP/image%202.png)

- Hypothesis: 4-bit E2M1 plus *two-level scaling* can preserve “model intelligence” close to FP8.
- Motivation:
    - Smaller block size (16 vs 32 in MXFP4) improves how well scaling matches local dynamic range.
    - E4M3 scales allow fractional (non-power-of-two) scaling, lowering MSE compared to E8M0.
    - NVIDIA reports roughly 3.5× memory reduction relative to FP16 and 1.8× relative to FP8 with small accuracy drop on LLM benchmarks.

**Quantization / dequantization**

For FP32 tensor `X`:

1. Compute a per-tensor scale `S_tensor` (FP32) to normalize the overall range so that per-block E4M3 scales stay in a good range.
2. Partition `X` (already normalized by `S_tensor`) into blocks of **16 values**.
3. For each 16-value block:
    - Choose FP8 E4M3 scale `s_block` that minimizes block error.
    - Compute `x_norm = X_block / (S_tensor * s_block)`.
    - Quantize `x_norm` to 4-bit E2M1 (FP4) values `x_q`.
4. Dequantization:
    - For each block: `x_hat = S_tensor * s_block * cast_fp4_to_fp32(x_q)`.

**Bits per weight (8,192-element tensor)**

- Data: 4 bits × 8,192 = 32,768 bits.
- FP8 E4M3 per-block scales:
    - Blocks: 8,192 / 16 = 512.
    - Each scale: 8 bits → 512 × 8 = 4,096 bits.
- FP32 per-tensor scale: 32 bits.
- Total: 32,768 + 4,096 + 32 = 36,896 bits.
- Effective bits/weight: 36,896 / 8,192 ≈ **4.50** bits (≈ 4.504).

---

## NF4 (NormalFloat 4 from QLoRA)

NF4 is *not* a sign/exponent/mantissa float. It is a 4-bit nonuniform quantizer designed specifically for zero-mean normal distributions, used in QLoRA for 4-bit LLM finetuning: https://arxiv.org/abs/2305.14314

### Format and properties

- NF4 defines 16 real-valued levels `q_i` in the range [−1, 1], chosen as quantiles of a standard normal distribution N(0, 1) and then normalized.
- Its key property: *each quantization bin has equal expected probability* under N(0, 1), making it information-theoretically optimal for normally distributed weights.
- Zero is explicitly included as one of the levels so padding or exact zeros can be represented losslessly.

In practice QLoRA uses NF4 for weights and BF16 for compute.

### Hypothesis and motivation

- Empirical observation: pretrained transformer weights are close to zero-mean, Gaussian-distributed with varying standard deviations.
- Hypothesis: if we design a quantizer matched to N(0, 1), then after a simple blockwise normalization, we can:
    - Use only 4 bits per weight.
    - Achieve much lower quantization error than uniform/linear or log quantization.
- Motivation:
    - Enable PEFT finetuning of large LLMs with 4-bit base weights while retaining near 16-bit quality.

### Quantization / dequantization in QLoRA

QLoRA uses *block-wise* NF4 with **block size B = 64** and **Double Quantization (DQ)** for the scales.

For each block of 64 FP32 weights:

1. **Compute first-level quantization constant `c2` (FP32)**
    - Typically via absmax or similar normalization:
        - Normalize weights: `w_norm = w / c2`, bringing them into [−1, 1].
2. **Quantize to NF4**
    - For each normalized value `w_norm`:
        - Find nearest NF4 level `q_i`.
            
            ```c
            [-1.0, -0.6961928009986877, -0.5250730514526367,
            -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
            -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
            0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
            0.5626170039176941, 0.7229568362236023, 1.0]
            ```
            
        - Store index `i` as a 4-bit code.
3. **Double Quantization of scales**
    - All first-level FP32 scales `c2` are themselves grouped (block size 256) and quantized using an FP8 format with its own FP32 scale `c1`.
    - This reduces per-weight scale overhead from 32/64 = 0.5 bits to:
        - `8/64 + 32/(64 · 256) ≈ 0.127` bits per parameter.

Dequantization:

- Recover `c2` via FP8 dequantization using `c1`, then recover each block:
    - `w_hat ≈ c2 * q_i`, then cast to BF16 for compute.

### Bits per weight (8,192-element tensor)

- NF4 codes:
    - 4 bits × 8,192 = 32,768 bits.
- Scale overhead:
    - We have 8192 / 64 = 128 blocks
    - FP8 Block scaling overhead bits = 128 x 8 = 1024 bits
    - Fp32 Scaling factor for FP8 Block Scaling factors = 32 bits
- Total bits: 32,768 + 1,024 + 32 =  = 33,824 bits.
- Effective bits/weight: 33,824 / 8,192 ≈ **4.13** bits.

For comparison, without Double Quantization, NF4 with FP32 scales per 64 weights would use 4.50 bits/weight (4 bits + 32/64).

---

## Summary Comparison Table (8192 FP32 weights)

| Format / Scheme | Width & core structure | Scaling strategy | Bits per weight |
| --- | --- | --- | --- |
| FP32 | 32-bit, 1-8-23 | None | **32.00** |
| FP16 | 16-bit, 1-5-10 | None | **16.00** |
| BF16 | 16-bit, 1-8-7 | None | **16.00** |
| MXFP8 (E4M3 + E8M0 per 32 blk) | 8-bit E4M3 | E8M0 scale per 32-value block | **8.25** |
| FP8 (E4M3/E5M2 + FP32 tensor) | 8-bit E4M3 or E5M2 | 1 FP32 scale per tensor | **8.00** |
| NVFP4 (E2M1 + E4M3 + FP32) | 4-bit E2M1 | FP8 E4M3 per 16-value block + FP32 per tensor | **4.50** |
| NF4 (w/o DQ) | 4-bit NF4 (16 quantile-based levels) | FP32 scale per 64-value block | 4.50 |
| MXFP4 (E2M1 + E8M0 per 32 blk) | 4-bit E2M1 | E8M0 scale per 32-value block | **4.25** |
| NF4 (w DQ) | 4-bit NF4 (16 quantile-based levels) | FP32+FP8 Double Quantization of block scales | **4.13** |
| FP4 (E2M1 + FP32 tensor) | 4-bit E2M1 | 1 FP32 scale per tensor | **4.00** |

---

## References:

1. MXFP8, MXFP4: [https://arxiv.org/pdf/2310.10537](https://arxiv.org/pdf/2310.10537)
2. MXFP8, MXFP4: [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
3. FP8: [https://arxiv.org/pdf/2209.05433](https://arxiv.org/pdf/2209.05433)
4. NVFP4: [https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
5. NVFP4: [https://arxiv.org/pdf/2509.25149v1](https://arxiv.org/pdf/2509.25149v1)
6. NF4: [https://arxiv.org/pdf/2305.14314](https://arxiv.org/pdf/2305.14314)