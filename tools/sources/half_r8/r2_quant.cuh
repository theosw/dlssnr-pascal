#pragma once
#include <cuda_fp16.h>

namespace swin1h_quant {
// Only for values whose f16 rounding is normal or zero; no overflow/NaN.
__device__ __forceinline__ float HalfNormal(float x) {
    unsigned b=__float_as_uint(x);
    b=(b+0xfffu+((b>>13)&1u))&0xffffe000u;
    return __uint_as_float(b);
}
__device__ __forceinline__ float SiluSpecialized(float h) {
    const float c=fminf(fmaxf(h,-4.0f),4.0f);
    const float p=HalfNormal(fmaf(-0.055908203125f,fabsf(c),0.447265625f));
    const float t=HalfNormal(fmaf(c,p,0.89453125f));
    return __half2float(__float2half_rn(h*t));
}
// Clamp endpoints are exact f16. RNE commutes with this clamp; the resulting
// exponent half is always normal (0x0400..0x48e0), so decoding is integer-only.
__device__ __forceinline__ float ExponentInteger(float v) {
    const float a=fminf(fmaxf(fmaf(v,0.044921875f,1.30078125f),1.03125f),1.5693359375f);
    const unsigned t=__float_as_uint(HalfNormal(a));
    return __uint_as_float(((t-0x3f800000u)<<5)+0x38000000u);
}
// Exact finite f32 -> f16 RNE -> f32, including subnormals and signed zero.
__device__ __forceinline__ float HalfInteger(float x) {
    const unsigned u=__float_as_uint(x), sign=u&0x80000000u, mag=u&0x7fffffffu;
    unsigned r=(mag+0xfffu+((mag>>13)&1u))&0xffffe000u;
    if(mag<0x38800000u) {
        const float a=__fadd_rn(__uint_as_float(mag),0.5f);
        r=__float_as_uint(__fsub_rn(a,0.5f));
    }
    if(mag>=0x477ff000u) r=0x7f800000u;
    if(mag>0x7f800000u) return __uint_as_float(0x7fffffffu);
    return __uint_as_float(sign|r);
}
// e4m3 satfinite RNE. Magic addition maps the subnormal grid to integer bits.
__device__ __forceinline__ unsigned EncodeInteger(float x) {
    const unsigned u=__float_as_uint(x),sign=(u>>24)&128u,mag=u&0x7fffffffu;
    unsigned b=min(mag,0x43e00000u);
    unsigned code=((b+0x7ffffu+((b>>20)&1u))>>20)-960u;
    if(b<0x3c800000u) code=__float_as_uint(__fadd_rn(__uint_as_float(b),16384.0f))-0x46800000u;
    if(mag>0x7f800000u) code=127u;
    return sign|code;
}
// Same result as the lane's E4M3(EncodeE4M3(x)), retained directly in f32.
__device__ __forceinline__ float Fp8Integer(float x) {
    const unsigned u=__float_as_uint(x),sign=u&0x80000000u,mag=u&0x7fffffffu;
    unsigned b=min(mag,0x43e00000u);
    unsigned r=(b+0x7ffffu+((b>>20)&1u))&0xfff00000u;
    if(b<0x3c800000u) r=__float_as_uint(__fsub_rn(__fadd_rn(__uint_as_float(b),16384.0f),16384.0f));
    // Existing decoder interprets FP8 0x7f as 480, not a NaN.
    if(mag>0x7f800000u) r=0x43f00000u;
    return __uint_as_float(sign|r);
}
}
