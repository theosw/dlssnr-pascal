#pragma once

namespace swinmh_quant_exact {
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
