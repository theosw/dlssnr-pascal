#pragma once
#define swinmh_r5_cache_state SWINMH_JOIN(swinmh_r5_cache_state_,SWINMH_VARIANT)
#define swinmh_r5_cache_key SWINMH_JOIN(swinmh_r5_cache_key_,SWINMH_VARIANT)
#define swinmh_r5_cache_checks SWINMH_JOIN(swinmh_r5_cache_checks_,SWINMH_VARIANT)
#define swinmh_r5_cache_builds SWINMH_JOIN(swinmh_r5_cache_builds_,SWINMH_VARIANT)
#define swinmh_r5_cache_raw SWINMH_JOIN(swinmh_r5_cache_raw_,SWINMH_VARIANT)
#define swinmh_r5_cache_data SWINMH_JOIN(swinmh_r5_cache_data_,SWINMH_VARIANT)
// A resident CTA wins the writer CAS. It does not wait on NR flags or another CTA.
// Readers pin the current snapshot; a different runtime pointer falls back immediately.
constexpr int swinmh_r5_cache_bytes=4*swinmh_C*swinmh_C+4096*swinmh_H;
constexpr int swinmh_r6_R=swinmh_C*swinmh_C;
constexpr int swinmh_r6_S=SWINMH_VARIANT==3?swinmh_r5_cache_bytes+3*swinmh_r6_R:swinmh_r5_cache_bytes+swinmh_r6_R+16;
constexpr int swinmh_r6_Q=SWINMH_VARIANT==3?swinmh_r6_S+4*swinmh_C:swinmh_r6_S+2*swinmh_C+16;
constexpr int swinmh_r6_P=swinmh_r6_Q+3*swinmh_r6_R+swinmh_H*8192+(swinmh_H>4?32:16);
constexpr int swinmh_r6_extra=3*swinmh_r6_R;
constexpr int swinmh_r6_raw_bytes=swinmh_r5_cache_bytes+swinmh_r6_extra;
#define swinmh_r6_cache_extra SWINMH_JOIN(swinmh_r6_cache_extra_,SWINMH_VARIANT)
__device__ __align__(128) unsigned swinmh_r6_cache_extra[swinmh_r6_extra/2];
__device__ __forceinline__ int swinmh_r6_offset(int byte) {
    constexpr int F=swinmh_r5_cache_bytes,R=swinmh_r6_R;
    if(byte<F)return byte;
    return swinmh_r6_Q+byte-F;
}
__device__ unsigned swinmh_r5_cache_state;
__device__ unsigned long long swinmh_r5_cache_key;
__device__ unsigned swinmh_r5_cache_checks,swinmh_r5_cache_builds;
__device__ __align__(128) unsigned swinmh_r5_cache_raw[swinmh_r6_raw_bytes/4];
__device__ __align__(128) float swinmh_r5_cache_data[swinmh_r5_cache_bytes];

__device__ __forceinline__ unsigned swinmh_r5_cg(const unsigned* p) {
    unsigned v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ float4 swinmh_r5_cache_load(const float* p) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");
    return v;
}
__device__ __forceinline__ bool swinmh_r5_cache_acquire(const unsigned char* w,float* temp) {
    const int tid=threadIdx.y*32+threadIdx.x;
    if(tid==0) {
        for(;;) {
            unsigned s=atomicAdd(&swinmh_r5_cache_state,0u);
            if(s==0) {if(atomicCAS(&swinmh_r5_cache_state,0u,0x80000000u)==0u){temp[0]=__uint_as_float(2u);break;}}
            else if((s&0x80000000u)==0 && atomicCAS(&swinmh_r5_cache_state,s,s+1)==s) {
                __threadfence();
                bool match=swinmh_r5_cache_key==reinterpret_cast<unsigned long long>(w);
                if(!match)atomicSub(&swinmh_r5_cache_state,1u);
                temp[0]=__uint_as_float(match?1u:0u);break;
            }
        }
    }
    __syncthreads();
    unsigned action=__float_as_uint(temp[0]);
    if(action==2) {
        unsigned diff=swinmh_r5_cache_key!=reinterpret_cast<unsigned long long>(w);
        for(int j=tid;j<swinmh_r6_raw_bytes/4;j+=64*swinmh_H)
            diff|=swinmh_r5_cg(reinterpret_cast<const unsigned*>(w+swinmh_r6_offset(j*4)))^swinmh_r5_cg(swinmh_r5_cache_raw+j);
        // Full raw comparison makes same-address blob replacement collision-free.
        int changed=__syncthreads_or(diff!=0);
        if(changed) {
            for(int word=tid;word<swinmh_r6_raw_bytes/4;word+=64*swinmh_H) {
                unsigned v=swinmh_r5_cg(reinterpret_cast<const unsigned*>(w+swinmh_r6_offset(word*4)));
                swinmh_r5_cache_raw[word]=v;
                int o=4*(word&255),j=o>>9,lane=(o>>4)&31,q=(o>>2)&3;
                int n=8*(2*j+(q>>1))+(lane>>2),k0=4*(lane&3)+16*(q&1);
                if(word>=swinmh_r5_cache_bytes/4) {
                    int nc=((n&7)>>1)*8+4*(n>>4)+2*((n>>3)&1)+(n&1);
                    #pragma unroll
                    for(int byte=0;byte<4;byte+=2) {
                        int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                        unsigned lo=__float_as_uint(swinmh_decode(v>>(8*byte)))>>16;
                        unsigned hi=__float_as_uint(swinmh_decode(v>>(8*(byte+1))))&0xffff0000u;
                        swinmh_r6_cache_extra[((word-swinmh_r5_cache_bytes/4)/256)*512+(k>>1)*32+(nc^swinmh_bank(k))]=lo|hi;
                    }
                } else {
                #if defined(SWINMH_CACHE_LEGACY)
                int nt=(n&7)>>1,nj=4*(n>>4)+2*((n>>3)&1)+(n&1),nc=nt*8+nj;
                if(word*4<4*swinmh_C*swinmh_C) {
                    #pragma unroll
                    for(int byte=0;byte<4;byte+=2) {
                        int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                        unsigned lo=__float_as_uint(swinmh_decode(v>>(8*byte)))>>16;
                        unsigned hi=__float_as_uint(swinmh_decode(v>>(8*(byte+1))))&0xffff0000u;
                        reinterpret_cast<unsigned*>(swinmh_r5_cache_data)[(word/256)*1024+(k>>1)*32+(nc^swinmh_bank(k))]=lo|hi;
                    }
                } else {
                    #pragma unroll
                    for(int byte=0;byte<4;byte++) {
                        int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                        swinmh_r5_cache_data[(word/256)*1024+k*32+(nc^swinmh_bank(k))]=swinmh_decode(v>>(8*byte));
                    }
                }
                #else
                #pragma unroll
                for(int byte=0;byte<4;byte++) {
                    int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                    swinmh_r5_cache_data[(word/256)*1024+k*32+(n^swinmh_bank(k))]=swinmh_decode(v>>(8*byte));
                }
                #endif
                }
            }
        }
        __threadfence();__syncthreads();
        if(tid==0) {
            swinmh_r5_cache_key=reinterpret_cast<unsigned long long>(w);
            atomicAdd(&swinmh_r5_cache_checks,1u);
            if(changed)atomicAdd(&swinmh_r5_cache_builds,1u);
            __threadfence();atomicExch(&swinmh_r5_cache_state,1u);
        }
    }
    __syncthreads();return action!=0;
}
__device__ __forceinline__ void swinmh_r5_cache_release(bool cached) {
    __syncthreads();
    if(cached&&threadIdx.x==0&&threadIdx.y==0)atomicSub(&swinmh_r5_cache_state,1u);
}
