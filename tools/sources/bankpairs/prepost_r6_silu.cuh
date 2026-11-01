#pragma once
#include "prepost_helpers_best.cuh"
#include "prepost_r6_silu_lut.cuh"
namespace prepost_fourwarp {
__device__ __forceinline__ float prepost_r6_silu(float x) {
    unsigned h=__half_as_ushort(__float2half_rn(x));
    return E4M3(__ldg(prepost_r6_silu_lut+h));
}
}
