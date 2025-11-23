#ifndef BF16_H_
#define BF16_H_

#include <stdint.h> // For uint16_t, uint32_t
#include <string.h> // For memcpy
#include <math.h>   // For isnan

// C++ compatibility
#ifdef __cplusplus
#include <cstdint>
#include <cstring>
#include <cmath>
#endif

// C++20 and later provides a dedicated, safe intrinsic
#if defined(__cplusplus) && __cplusplus >= 202002L
#include <bit>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts a 16-bit BF16 value to a 32-bit float.
 *
 * This function performs the bit-level conversion from BFloat16 to FP32.
 * It takes the 16-bit BF16, casts it to a 32-bit integer, and shifts it
 * left by 16 bits. This places the sign, exponent, and 7-bit mantissa
 * in the correct positions for an FP32, padding the new (lower) 16 bits
 * of the mantissa with zeros.
 *
 * @param b The 16-bit BFloat16 value.
 * @return The corresponding 32-bit float value.
 */
static inline float fp32_from_bf16_value(uint16_t b) {
    uint32_t u_b = (uint32_t)b << 16;
    float f;

#if defined(__cplusplus) && __cplusplus >= 202002L
    // C++20: Use std::bit_cast for safe, zero-cost type punning
    f = std::bit_cast<float>(u_b);
#else
    // C / C++11: Use memcpy, which compilers recognize and optimize
    // to a single instruction, while respecting aliasing rules.
    memcpy(&f, &u_b, sizeof(f));
#endif
    
    return f;
}

/**
 * @brief Converts a 32-bit float to a 16-bit BF16 value using
 * **Round-to-Nearest-Even (RNE)**.
 *
 * This is the numerically superior and standard method for conversion.
 * It rounds the float to the *closest* representable BF16 value.
 * In cases of a tie (exactly halfway), it rounds to the "even"
 * value (where the least significant bit is 0).
 * This method avoids the statistical bias of truncation.
 *
 * @param f The 32-bit float value.
 * @return The 16-bit BFloat16 value (rounded).
 */
static inline uint16_t bf16_from_fp32_value(float f) {
    uint32_t u_f;

#if defined(__cplusplus) && __cplusplus >= 202002L
    u_f = std::bit_cast<uint32_t>(f);
#else
    memcpy(&u_f, &f, sizeof(u_f));
#endif

    // Check for NaN (all exponent bits 1, mantissa non-zero)
    // (u_f & 0x7FFFFFFF) > 0x7F800000 is a portable way to check for NaN
    if ((u_f & 0x7F800000) == 0x7F800000 && (u_f & 0x007FFFFF) != 0) {
        // Preserve NaN payload. Set top bit of mantissa to make it a qNaN.
        return (uint16_t)(u_f >> 16) | 0x0040;
    }

    // Get the 16 bits that will be truncated
    uint32_t remainder = u_f & 0xFFFF;
    // Get the LSB of the 16 bits that will be *kept*
    uint32_t lsb = (u_f >> 16) & 1; 

    // Round up if:
    // 1. The remainder is greater than halfway (0x8000)
    // OR
    // 2. The remainder is *exactly* halfway (0x8000) AND the LSB is 1 (odd),
    //    so we round up to the nearest *even* number.
    if (remainder > 0x8000 || (remainder == 0x8000 && lsb == 1)) {
        u_f += 0x10000;
    }

    return (uint16_t)(u_f >> 16);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF16_H_
