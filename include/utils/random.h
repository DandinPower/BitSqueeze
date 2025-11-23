#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float **gen_random_float_arrays(uint64_t count,
                                uint64_t N,
                                float minv,
                                float maxv,
                                unsigned int seed);

void free_random_float_arrays(float **arrs, uint64_t count);

#endif
