#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "datatype/bf16.h"

// --- Globals for test summary ---
static int global_pass_count = 0;
static int global_fail_count = 0;

// --- Helper functions for float/u32 conversion ---

/**
 * @brief Re-interprets a 32-bit unsigned integer as a 32-bit float.
 */
static float float_from_u32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/**
 * @brief Re-interprets a 32-bit float as a 32-bit unsigned integer.
 */
static uint32_t u32_from_float(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

// --- Test functions ---

/**
 * @brief Tests the FP32 -> BF16 (rounding) conversion.
 */
void test_fp32_to_bf16(float f_in, uint16_t u_expected, const char* test_name) {
    uint16_t u_actual = bf16_from_fp32_value(f_in);
    
    if (u_actual == u_expected) {
        printf("[PASS] %s\n", test_name);
        global_pass_count++;
    } else {
        printf("[FAIL] %s: Input=%.10f (0x%08X), Expected=0x%04X, Got=0x%04X\n",
               test_name, f_in, u32_from_float(f_in), u_expected, u_actual);
        global_fail_count++;
    }
}

/**
 * @brief Tests the BF16 -> FP32 (expansion) conversion.
 */
void test_bf16_to_fp32(uint16_t u_in, float f_expected, const char* test_name) {
    float f_actual = fp32_from_bf16_value(u_in);
    
    // This conversion should be exact, so a direct bit-level check is fine.
    if (u32_from_float(f_actual) == u32_from_float(f_expected)) {
        printf("[PASS] %s\n", test_name);
        global_pass_count++;
    } else {
        printf("[FAIL] %s: Input=0x%04X, Expected=%.10f (0x%08X), Got=%.10f (0x%08X)\n",
               test_name, u_in, f_expected, u32_from_float(f_expected), 
               f_actual, u32_from_float(f_actual));
        global_fail_count++;
    }
}


int main(void) {
    printf("--- Starting BF16 Conversion Tests ---\n\n");

    // === Test 1: BF16 -> FP32 (Expansion) ===
    // This function is simpler and just adds 16 zero bits.
    printf("--- Testing bf16_from_fp32_value (BF16 -> FP32) ---\n");
    test_bf16_to_fp32(0x0000, 0.0f,          "BF16->FP32: Positive Zero");
    test_bf16_to_fp32(0x8000, -0.0f,         "BF16->FP32: Negative Zero");
    test_bf16_to_fp32(0x3F80, 1.0f,          "BF16->FP32: 1.0");
    test_bf16_to_fp32(0x4000, 2.0f,          "BF16->FP32: 2.0");
    test_bf16_to_fp32(0xC000, -2.0f,         "BF16->FP32: -2.0");
    test_bf16_to_fp32(0x7F80, INFINITY,      "BF16->FP32: Positive Infinity");
    test_bf16_to_fp32(0xFF80, -INFINITY,     "BF16->FP32: Negative Infinity");
    test_bf16_to_fp32(0x7FC0, NAN,           "BF16->FP32: Quiet NaN (qNaN)");
    test_bf16_to_fp32(0x420C, 35.0f,         "BF16->FP32: 35.0"); // 0x420C0000
    test_bf16_to_fp32(0x3DCD,                // 0.10009765625
                      float_from_u32(0x3DCD0000), 
                                             "BF16->FP32: 0.1 (approx)");

    // === Test 2: FP32 -> BF16 (Rounding) ===
    // This tests the complex RNE logic.
    printf("\n--- Testing bf16_from_fp32_value (FP32 -> BF16) ---\n");
    test_fp32_to_bf16(0.0f, 0x0000,          "FP32->BF16: Positive Zero");
    test_fp32_to_bf16(-0.0f, 0x8000,         "FP32->BF16: Negative Zero");
    test_fp32_to_bf16(1.0f, 0x3F80,          "FP32->BF16: 1.0 (no rounding)");
    test_fp32_to_bf16(INFINITY, 0x7F80,      "FP32->BF16: Positive Infinity");
    test_fp32_to_bf16(-INFINITY, 0xFF80,     "FP32->BF16: Negative Infinity");
    test_fp32_to_bf16(NAN, 0x7FC0,           "FP32->BF16: Quiet NaN (qNaN)");

    // --- RNE: Rounding Down ---
    test_fp32_to_bf16(float_from_u32(0x3F800001), 0x3F80, "RNE: Round Down (small fraction)");
    test_fp32_to_bf16(float_from_u32(0x3F807FFF), 0x3F80, "RNE: Round Down (just below half)");
    
    // --- RNE: Rounding Up ---
    test_fp32_to_bf16(float_from_u32(0x3F808001), 0x3F81, "RNE: Round Up (just above half)");
    test_fp32_to_bf16(float_from_u32(0x3F80FFFF), 0x3F81, "RNE: Round Up (large fraction)");

    // --- RNE: Tie-Breaking (These will likely FAIL with your code) ---
    printf("\n--- RNE Tie-Breaking Tests (EXPECTED TO FAIL with original code) ---\n");
    // Input 0x3F808000: Exactly halfway. LSB of 0x3F80 is 0 (even).
    // Should round DOWN to even.
    test_fp32_to_bf16(float_from_u32(0x3F808000), 0x3F80, 
        "[!!] RNE: Tie-to-Even (0x3F808000 -> 0x3F80)");

    // Input 0x3F818000: Exactly halfway. LSB of 0x3F81 is 1 (odd).
    // Should round UP to even (0x3F82).
    test_fp32_to_bf16(float_from_u32(0x3F818000), 0x3F82, 
        "[!!] RNE: Tie-to-Even (0x3F818000 -> 0x3F82)");

    // Input 0x3F828000: Exactly halfway. LSB of 0x3F82 is 0 (even).
    // Should round DOWN to even.
    test_fp32_to_bf16(float_from_u32(0x3F828000), 0x3F82, 
        "[!!] RNE: Tie-to-Even (0x3F828000 -> 0x3F82)");

    // Input 0x3F838000: Exactly halfway. LSB of 0x3F83 is 1 (odd).
    // Should round UP to even (0x3F84).
    test_fp32_to_bf16(float_from_u32(0x3F838000), 0x3F84, 
        "[!!] RNE: Tie-to-Even (0x3F838000 -> 0x3F84)");


    // --- Summary ---
    printf("\n--- Test Summary ---\n");
    printf("Total Tests: %d\n", global_pass_count + global_fail_count);
    printf("Passed:      %d\n", global_pass_count);
    printf("Failed:      %d\n", global_fail_count);
    printf("\n");

    if (global_fail_count > 0) {
        printf("NOTE: Failures in RNE Tie-Breaking tests are expected due to incorrect rounding logic in bf16.h.\n");
        return 1; // Return error code
    }


    return 0; // All tests passed
}
