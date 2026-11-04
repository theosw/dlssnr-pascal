#pragma once
// Research NR 310.9.16 half-resolution kernels, four warps per 8x8 window.
// Runtime weights, original numerical boundaries and lane view/flag protocol.
#include <cuda_fp16.h>
#include "r2_quant.cuh"

namespace swin1h_native_swin1h_r6_nf
{
    __device__ __forceinline__ float E4M3(unsigned bits)
    {
        return __uint_as_float(((bits & 0x80u) << 24) | ((bits & 0x7Fu) << 20)) * 1.329227995784916e36f;
    }

    __device__ __forceinline__ unsigned EncodeE4M3(float x) { return swin1h_quant::EncodeInteger(x); }

    __device__ __forceinline__ float Half(float v) { return __half2float(__float2half_rn(v)); }
    __device__ __forceinline__ float Fp8(float v) { return swin1h_quant::Fp8Integer(v); }
    __device__ __forceinline__ float LowHalf(unsigned w) { return __half2float(__ushort_as_half(static_cast<unsigned short>(w))); }
    __device__ __forceinline__ float HighHalf(unsigned w) { return __half2float(__ushort_as_half(static_cast<unsigned short>(w >> 16))); }

    // NVIDIA's MpCubicSiluActivation on an f16 value, every step rounded to f16.
    __device__ __forceinline__ float Silu3(float h)
    {
        const float c = fminf(fmaxf(h, -4.0f), 4.0f);
        const float p = Half(fmaf(-0.055908203125f, fabsf(c), 0.447265625f));
        const float t = Half(fmaf(c, p, 0.89453125f));
        return Half(h * t);
    }

    // Channel held at stored position s (0..15) of a 16-channel half (accumulator order).
    __device__ __forceinline__ int Channel16(int s) { return 2 * (s >> 2) + (s & 1) + 8 * ((s >> 1) & 1); }

    constexpr int kStride = 36;   // padded row of 32 floats: lane l's rows l and l + 32 hit distinct banks

    struct __align__(8) Swin1hParams
    {
        const unsigned char* input;    // +0: planar tinlayout, [half][y][x][16]
        unsigned char* output;         // +8: blocked 4x4, A-fragment order
        const unsigned char* blob;     // +16: weights
        int height, width;             // +24
        int x0, y0;                    // +32: window offset
        const void* unused40;          // +40
        const void* unused48;          // +48
        int* flags;                    // +56: per-window signal
        const void* unused64;          // +64
        int heightOverride, widthOverride;   // +72
        const void* unused80;          // +80
        const void* unused88;          // +88
    };
    static_assert(sizeof(Swin1hParams) == 96, "parameter block size");


}
