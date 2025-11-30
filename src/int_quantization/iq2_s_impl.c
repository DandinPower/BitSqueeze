#include "int_quantization/iq2_s_impl.h"
#include "datatype/fp16/fp16.h"

/* ============================================================================
 * Lookup Tables
 * ============================================================================ */

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* 1024-entry grid for IQ2_S
 * PLACEHOLDER - copy from ggml-common.h iq2s_grid table */
static const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

/* ============================================================================
 * Quantization helper tables (built at runtime)
 * ============================================================================ */

static uint64_t *kgrid_q2s = NULL;
static int      *kmap_q2s = NULL;
static uint16_t *kneighbors_q2s = NULL;
static int       iq2_s_initialized = 0;

#define KMAP_SIZE 43692

static int iq2_compare_func(const void *a, const void *b) {
    const int *l = (const int *)a;
    const int *r = (const int *)b;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq2_s_init(void) {
    if (iq2_s_initialized) return;
    
    const int grid_size = 1024;
    const int nwant = 1;  /* IQ2_S uses only 1 neighbor level */
    
    kgrid_q2s = (uint64_t *)malloc(grid_size * sizeof(uint64_t));
    if (!kgrid_q2s) return;
    
    // for (int k = 0; k < grid_size; ++k) {
    //     kgrid_q2s[k] = iq2s_grid[k];
    // }
    // Replace this loop in iq2_s_init:
    // for (int k = 0; k < grid_size; ++k) {
    //     kgrid_q2s[k] = iq2s_grid[k];
    // }

    // With this corrected loop:
    for (int k = 0; k < grid_size; ++k) {
        uint64_t packed = iq2s_grid[k];
        uint64_t normalized = 0;
        
        for (int i = 0; i < 8; ++i) {
            // Extract the byte (e.g., 0x08, 0x19, 0x2b...)
            uint8_t val = (packed >> (i * 8)) & 0xFF;
            uint8_t q_val;

            // Map dequant values back to normalized grid coordinates (1, 3, 5, 7)
            // Values in table are approx: 8, 25, 43, 60
            if      (val < 15) q_val = 1;
            else if (val < 35) q_val = 3;
            else if (val < 55) q_val = 5;
            else               q_val = 7;

            normalized |= ((uint64_t)q_val << (i * 8));
        }
        kgrid_q2s[k] = normalized;
    }
    
    kmap_q2s = (int *)malloc(KMAP_SIZE * sizeof(int));
    if (!kmap_q2s) {
        free(kgrid_q2s); kgrid_q2s = NULL;
        return;
    }
    
    for (int i = 0; i < KMAP_SIZE; ++i) kmap_q2s[i] = -1;
    
    for (int i = 0; i < grid_size; ++i) {
        const uint8_t *aux8 = (const uint8_t *)&kgrid_q2s[i];
        uint16_t index = 0;
        for (int k = 0; k < 8; ++k) {
            uint16_t q = (aux8[k] - 1) / 2;
            index |= (q << (2 * k));
        }
        kmap_q2s[index] = i;
    }
    
    int8_t pos[8];
    int *dist2 = (int *)malloc(2 * grid_size * sizeof(int));
    if (!dist2) {
        free(kgrid_q2s); kgrid_q2s = NULL;
        free(kmap_q2s);  kmap_q2s = NULL;
        return;
    }
    
    int num_neighbors = 0, num_not_in_map = 0;
    
    for (int i = 0; i < KMAP_SIZE; ++i) {
        if (kmap_q2s[i] >= 0) continue;
        ++num_not_in_map;
        
        for (int k = 0; k < 8; ++k) {
            int l = (i >> (2 * k)) & 0x3;
            pos[k] = 2 * l + 1;
        }
        
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(kgrid_q2s + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) {
                d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        
        int n = 0, d2_prev = dist2[0], nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2_prev) {
                if (nhave == nwant) break;
                d2_prev = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    
    kneighbors_q2s = (uint16_t *)malloc((num_neighbors + num_not_in_map) * sizeof(uint16_t));
    if (!kneighbors_q2s) {
        free(kgrid_q2s); kgrid_q2s = NULL;
        free(kmap_q2s);  kmap_q2s = NULL;
        free(dist2);
        return;
    }
    
    int counter = 0;
    for (int i = 0; i < KMAP_SIZE; ++i) {
        if (kmap_q2s[i] >= 0) continue;
        
        for (int k = 0; k < 8; ++k) {
            int l = (i >> (2 * k)) & 0x3;
            pos[k] = 2 * l + 1;
        }
        
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(kgrid_q2s + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) {
                d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            }
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), iq2_compare_func);
        
        kmap_q2s[i] = -(counter + 1);
        
        int d2_prev = dist2[0];
        uint16_t *start = &kneighbors_q2s[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2_prev) {
                if (nhave == nwant) break;
                d2_prev = dist2[2*j];
                ++nhave;
            }
            kneighbors_q2s[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;
    }
    
    free(dist2);
    iq2_s_initialized = 1;
}

void iq2_s_free_tables(void) {
    if (kgrid_q2s) { free(kgrid_q2s); kgrid_q2s = NULL; }
    if (kmap_q2s)  { free(kmap_q2s);  kmap_q2s = NULL; }
    if (kneighbors_q2s) { free(kneighbors_q2s); kneighbors_q2s = NULL; }
    iq2_s_initialized = 0;
}

/* ============================================================================
 * Array allocation and management
 * ============================================================================ */

static int64_t _get_iq2_s_array_size(const iq2_s_array_t *arr) {
    if (!arr) return 0;
    /* Header + d (2) + qs (64) + qh (8) + scales (8) per super block */
    return (int64_t)(sizeof(iq2_s_array_t) 
                   + arr->num_super_blocks * sizeof(uint16_t)  /* d */
                   + arr->num_super_blocks * 64                /* qs */
                   + arr->num_super_blocks * 8                 /* qh */
                   + arr->num_super_blocks * 8);               /* scales */
}

iq2_s_array_t *allocate_iq2_s_array(uint64_t num_elements) {
    if (!num_elements) return NULL;
    
    uint64_t num_super_blocks = (num_elements + IQ2_S_SUPER_BLOCK_SIZE - 1) / IQ2_S_SUPER_BLOCK_SIZE;
    
    size_t total = sizeof(iq2_s_array_t)
                 + num_super_blocks * sizeof(uint16_t)  /* d */
                 + num_super_blocks * 64                /* qs */
                 + num_super_blocks * 8                 /* qh */
                 + num_super_blocks * 8;                /* scales */
    
    iq2_s_array_t *arr = (iq2_s_array_t *)calloc(1, total);
    if (!arr) return NULL;
    
    arr->num_elements = num_elements;
    arr->num_super_blocks = num_super_blocks;
    arr->d = (uint16_t *)(arr + 1);
    arr->qs = (uint8_t *)(arr->d + num_super_blocks);
    arr->qh = arr->qs + num_super_blocks * 64;
    arr->scales = arr->qh + num_super_blocks * 8;
    
    return arr;
}

void free_iq2_s_array(iq2_s_array_t *arr) {
    if (arr) free(arr);
}

int64_t get_iq2_s_array_size(const iq2_s_array_t *arr) {
    return _get_iq2_s_array_size(arr);
}

iq2_s_array_t *load_iq2_s_array_from_buffer(const void *buffer, int64_t buffer_size) {
    if (!buffer || buffer_size < (int64_t)sizeof(iq2_s_array_t)) return NULL;
    
    iq2_s_array_t *arr = (iq2_s_array_t *)calloc(1, buffer_size);
    if (!arr) return NULL;
    
    memcpy(arr, buffer, buffer_size);
    
    const int64_t expected = _get_iq2_s_array_size(arr);
    if (expected == 0 || buffer_size < expected) {
        free(arr);
        return NULL;
    }
    
    arr->d = (uint16_t *)(arr + 1);
    arr->qs = (uint8_t *)(arr->d + arr->num_super_blocks);
    arr->qh = arr->qs + arr->num_super_blocks * 64;
    arr->scales = arr->qh + arr->num_super_blocks * 8;
    
    return arr;
}

/* ============================================================================
 * Dequantization
 * ============================================================================ */

int iq2_s_decompress(const iq2_s_array_t *arr, float *float_array) {
    if (!arr || !float_array) return 1;
    
    const uint64_t num_super_blocks = arr->num_super_blocks;
    const uint64_t num_elements = arr->num_elements;
    
#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t sb = 0; sb < num_super_blocks; ++sb) {
        const uint64_t block_start = sb * IQ2_S_SUPER_BLOCK_SIZE;
        uint64_t block_end = block_start + IQ2_S_SUPER_BLOCK_SIZE;
        if (block_end > num_elements) block_end = num_elements;
        
        const float d = fp16_ieee_to_fp32_value(arr->d[sb]);
        const uint8_t *qs = arr->qs + sb * 64;
        const uint8_t *qh = arr->qh + sb * 8;
        const uint8_t *signs = qs + 32;  /* Signs stored in second half of qs */
        const uint8_t *scales_block = arr->scales + sb * 8;
        
        /* Process 8 groups of 32 values */
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            float db[2];
            db[0] = d * (0.5f + (float)(scales_block[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (float)(scales_block[ib32] >> 4)) * 0.25f;
            
            /* Process 4 sub-groups of 8 values */
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                
                /* Grid index: 8 bits from qs, 2 high bits from qh */
                uint16_t grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + grid_idx);
                uint8_t sign_byte = signs[l];
                const uint64_t out_base = block_start + ib32 * 32 + l * 8;
                
                for (int j = 0; j < 8; ++j) {
                    const uint64_t out_idx = out_base + j;
                    if (out_idx < block_end) {
                        float val = dl * (float)grid[j];
                        float_array[out_idx] = (sign_byte & kmask_iq2xs[j]) ? -val : val;
                    }
                }
            }
            qs += 4;
            signs += 4;
        }
    }
    
    return 0;
}

/* ============================================================================
 * Quantization helpers
 * ============================================================================ */

static inline int nearest_int(float f) {
    return (int)(f + 0.5f - (f < 0));
}

static int iq2_find_best_neighbour(const uint16_t *neighbours, const uint64_t *grid,
                                   const float *xval, const float *weight, 
                                   float scale, int8_t *L) {
    int num_neighbors = neighbours[0];
    if (num_neighbors <= 0) return -1;
    
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = (float)pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    
    if (grid_index >= 0) {
        const int8_t *pg = (const int8_t *)(grid + grid_index);
        for (int i = 0; i < 8; ++i) {
            L[i] = (pg[i] - 1) / 2;
        }
    }
    
    return grid_index;
}

/* ============================================================================
 * Quantization
 * ============================================================================ */

int iq2_s_compress(const float *float_array, uint64_t num_elements, iq2_s_array_t **out) {
    if (!float_array || num_elements == 0 || !out || *out) return 1;
    
    if (!iq2_s_initialized) {
        iq2_s_init();
        if (!iq2_s_initialized) return 1;
    }
    
    iq2_s_array_t *arr = allocate_iq2_s_array(num_elements);
    if (!arr) return 1;
    
    const int kMaxQ = 3;
    const float GROUP_MAX_EPS = 1e-8f;
    
    const uint64_t num_super_blocks = arr->num_super_blocks;
    
#if defined(__linux__) && defined(_OPENMP)
#pragma omp parallel for
#endif
    for (uint64_t sb = 0; sb < num_super_blocks; ++sb) {
        float weight[16];
        float xval[16];
        float waux[16];
        int8_t L[16];
        int8_t Laux[16];
        uint8_t block_signs[2];
        
        /* Temporary storage for one super block */
        uint8_t qs_tmp[64];      /* 32 grid low + 32 signs */
        uint8_t qh_tmp[8];       /* grid high bits */
        uint8_t scales_tmp[8];
        float scales_f[16];
        
        memset(qs_tmp, 0, sizeof(qs_tmp));
        memset(qh_tmp, 0, sizeof(qh_tmp));
        memset(scales_tmp, 0, sizeof(scales_tmp));
        
        uint64_t block_start = sb * IQ2_S_SUPER_BLOCK_SIZE;
        uint64_t block_end = block_start + IQ2_S_SUPER_BLOCK_SIZE;
        if (block_end > num_elements) block_end = num_elements;
        
        float sumx2 = 0;
        for (uint64_t i = block_start; i < block_end; ++i) {
            sumx2 += float_array[i] * float_array[i];
        }
        float sigma2 = sumx2 / (float)IQ2_S_SUPER_BLOCK_SIZE;
        
        float max_scale = 0;
        int grid_indices[32];  /* Store grid indices for all 32 sub-groups */
        uint8_t sign_patterns[32];
        
        /* Process 16 sub-groups of 16 values */
        for (int ib = 0; ib < 16; ++ib) {
            uint64_t group_start = block_start + ib * 16;
            
            for (int i = 0; i < 16; ++i) {
                uint64_t idx = group_start + i;
                float v = (idx < num_elements) ? float_array[idx] : 0.0f;
                weight[i] = sqrtf(sigma2 + v * v);
                waux[i] = sqrtf(weight[i]);
            }
            
            /* Handle signs - NO parity constraint for IQ2_S (full 8-bit signs) */
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    uint64_t idx = group_start + 8 * k + i;
                    float v = (idx < num_elements) ? float_array[idx] : 0.0f;
                    
                    if (v >= 0) {
                        xval[8*k + i] = v;
                    } else {
                        xval[8*k + i] = -v;
                        s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            
            float max = xval[0];
            for (int i = 1; i < 16; ++i) {
                if (xval[i] > max) max = xval[i];
            }
            
            if (max < GROUP_MAX_EPS) {
                scales_f[ib] = 0;
                memset(L, 0, 16);
                grid_indices[2*ib + 0] = 0;
                grid_indices[2*ib + 1] = 0;
            } else {
                float best = 0;
                float scale = max / (2 * kMaxQ - 1);
                
                for (int is = -9; is <= 9; ++is) {
                    float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                    float this_scale = 1.0f / id;
                    
                    for (int k = 0; k < 2; ++k) {
                        for (int i = 0; i < 8; ++i) {
                            int l = nearest_int(0.5f * (id * xval[8*k+i] - 1));
                            if (l < 0) l = 0;
                            if (l > kMaxQ - 1) l = kMaxQ - 1;
                            Laux[8*k + i] = (int8_t)l;
                        }
                        
                        uint16_t u = 0;
                        for (int i = 0; i < 8; ++i) {
                            u |= (Laux[8*k+i] << (2*i));
                        }
                        
                        int grid_index = kmap_q2s[u];
                        if (grid_index < 0) {
                            const uint16_t *neighbours = kneighbors_q2s - kmap_q2s[u] - 1;
                            iq2_find_best_neighbour(neighbours, kgrid_q2s, 
                                                   xval + 8*k, waux + 8*k, 
                                                   this_scale, Laux + 8*k);
                        }
                    }
                    
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 16; ++i) {
                        float w = weight[i];
                        float q = 2 * Laux[i] + 1;
                        sumqx += w * xval[i] * q;
                        sumq2 += w * q * q;
                    }
                    
                    if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                        scale = sumqx / sumq2;
                        best = scale * sumqx;
                        memcpy(L, Laux, 16);
                    }
                }
                
                /* Final pass to get grid indices */
                if (scale > 0) {
                    float id = 1.0f / scale;
                    for (int k = 0; k < 2; ++k) {
                        uint16_t u = 0;
                        for (int i = 0; i < 8; ++i) {
                            int l = nearest_int(0.5f * (id * xval[8*k+i] - 1));
                            if (l < 0) l = 0;
                            if (l > kMaxQ - 1) l = kMaxQ - 1;
                            u |= (l << (2*i));
                        }
                        
                        int grid_index = kmap_q2s[u];
                        if (grid_index < 0) {
                            const uint16_t *neighbours = kneighbors_q2s - kmap_q2s[u] - 1;
                            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2s, 
                                                                xval + 8*k, waux + 8*k, 
                                                                scale, L + 8*k);
                        }
                        grid_indices[2*ib + k] = (grid_index >= 0) ? grid_index : 0;
                    }
                    
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 16; ++i) {
                        float w = weight[i];
                        float q = 2 * L[i] + 1;
                        sumqx += w * xval[i] * q;
                        sumq2 += w * q * q;
                    }
                    if (sumq2 > 0) scale = sumqx / sumq2;
                } else {
                    grid_indices[2*ib + 0] = 0;
                    grid_indices[2*ib + 1] = 0;
                }
                
                scales_f[ib] = (scale >= 0) ? scale : -scale;
                if (scales_f[ib] > max_scale) max_scale = scales_f[ib];
            }
            
            sign_patterns[2*ib + 0] = block_signs[0];
            sign_patterns[2*ib + 1] = block_signs[1];
        }
        
        /* Encode block scale and pack data */
        if (max_scale == 0) {
            arr->d[sb] = 0;
            memset(arr->qs + sb * 64, 0, 64);
            memset(arr->qh + sb * 8, 0, 8);
            memset(arr->scales + sb * 8, 0, 8);
        } else {
            float d = max_scale / 31.0f;
            arr->d[sb] = fp16_ieee_from_fp32_value(d);
            float id = 1.0f / d;
            
            /* Pack grid indices and signs */
            uint8_t *qs_out = arr->qs + sb * 64;
            uint8_t *qh_out = arr->qh + sb * 8;
            
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                /* Encode scales: 2 × 4-bit per byte */
                int l0 = nearest_int(0.5f * (id * scales_f[2*ib32 + 0] - 1));
                int l1 = nearest_int(0.5f * (id * scales_f[2*ib32 + 1] - 1));
                if (l0 < 0) l0 = 0; if (l0 > 15) l0 = 15;
                if (l1 < 0) l1 = 0; if (l1 > 15) l1 = 15;
                arr->scales[sb * 8 + ib32] = (uint8_t)(l0 | (l1 << 4));
                
                /* Pack grid indices: low 8 bits in qs, high 2 bits in qh */
                uint8_t qh_byte = 0;
                for (int l = 0; l < 4; ++l) {
                    int sub_idx = ib32 * 4 + l;  /* 0..31 */
                    int gi = grid_indices[sub_idx];
                    
                    qs_out[l] = (uint8_t)(gi & 0xFF);          /* low 8 bits */
                    qh_byte |= ((gi >> 8) & 0x3) << (2 * l);   /* high 2 bits */
                    
                    /* Signs in second half */
                    qs_out[32 + l] = sign_patterns[sub_idx];
                }
                qh_out[ib32] = qh_byte;
                qs_out += 4;
            }
        }
    }
    
    *out = arr;
    return 0;
}
