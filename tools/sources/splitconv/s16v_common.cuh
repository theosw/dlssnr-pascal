#pragma once
#include <cuda_fp16.h>
__device__ __forceinline__ float s16v_decode(unsigned b) { return __uint_as_float(((b&128u)<<24)|((b&127u)<<20))*1.329227995784916e36f; }
__device__ __forceinline__ float s16v_half(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ unsigned s16v_encode(float x) {
 unsigned s=(__float_as_uint(x)>>24)&128u;
 if(x!=x)return s|127u;
 float a=fminf(fabsf(x),448.f); unsigned b;
 if(a<.015625f)b=__float2int_rn(a*512.f);
 else {unsigned t=__float_as_uint(a); t+=0x7ffffu+((t>>20)&1);b=(t>>20)-960u;}
 return s|b;
}
__device__ __forceinline__ float s16v_fp8(float x) {return s16v_decode(s16v_encode(x));}
__device__ __forceinline__ int s16v_channel(int s) { return 2*(s>>2)+(s&1)+8*((s>>1)&1); }
__device__ __forceinline__ int s16v_stored(int c) { return 16*(c>>4)+4*((c&7)>>1)+2*((c>>3)&1)+(c&1); }
__device__ __forceinline__ int s16v_aoff(int tok,int k) { return (k/32)*512+4*(tok&7)*16+((k&15)/4)*16+((tok>>3)&1)*4+(k/16%2)*8+(k&3); }
__device__ __forceinline__ int s16v_woff(int k,int n,int N) {return (k/32)*N*32+(n/16)*512+((n&7)*4+(k&15)/4)*16+((n>>3)&1)*8+((k>>4)&1)*4+(k&3);}
__device__ __forceinline__ void s16v_wait(const int* f) {while(*reinterpret_cast<const volatile int*>(f)<0){} }
__device__ __forceinline__ float s16v_exp(float s) {
 float t=fminf(fmaxf(s16v_half(fmaf(s,.044921875f,1.30078125f)),1.03125f),1.5693359375f);
 return __half2float(__ushort_as_half((unsigned short)((__half_as_ushort(__float2half_rn(t))-0x3c00u)<<5)));
}
