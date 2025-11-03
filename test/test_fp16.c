#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "fp16/fp16.h"

static uint32_t f32_bits(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof(u));
  return u;
}

int main(void) {
  float vals[] = {
    0.0f, -0.0f, 1.0f, -1.0f,
    3.1415926f, 1.0e-8f, 1.0e-4f,
    65504.0f,         /* max finite fp16 */
    INFINITY, -INFINITY, NAN
  };
  const size_t n = sizeof(vals)/sizeof(vals[0]);

  printf("C test: fp32 -> fp16 -> fp32\n");
  for (size_t i = 0; i < n; i++) {
    float f = vals[i];
    uint16_t h = fp16_ieee_from_fp32_value(f);
    float fr = fp16_ieee_to_fp32_value(h);

    double abs_err = fabs((double)f - (double)fr);
    int f_nan = isnan(f);
    int fr_nan = isnan(fr);

    printf("[%02zu] in=%-12a bits32=0x%08X  h=0x%04X  out=%-12a bits32=0x%08X",
           i, f, f32_bits(f), h, fr, f32_bits(fr));

    if (f_nan || fr_nan) {
      printf("  note=NaN case\n");
    } else {
      printf("  abs_err=%g\n", abs_err);
    }
  }
  return 0;
}
