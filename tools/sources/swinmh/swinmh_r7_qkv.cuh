#pragma once
#define swinmh_r7_state SWINMH_JOIN(swinmh_r7_state_,SWINMH_VARIANT)
#define swinmh_r7_key SWINMH_JOIN(swinmh_r7_key_,SWINMH_VARIANT)
#define swinmh_r7_raw SWINMH_JOIN(swinmh_r7_raw_,SWINMH_VARIANT)
#define swinmh_r7_data SWINMH_JOIN(swinmh_r7_data_,SWINMH_VARIANT)
#define swinmh_r7_checks SWINMH_JOIN(swinmh_r7_checks_,SWINMH_VARIANT)
#define swinmh_r7_builds SWINMH_JOIN(swinmh_r7_builds_,SWINMH_VARIANT)
constexpr int swinmh_r7_bytes=3*swinmh_C*swinmh_C;
__device__ unsigned swinmh_r7_state;
__device__ unsigned long long swinmh_r7_key;
__device__ unsigned swinmh_r7_checks,swinmh_r7_builds;
__device__ __align__(128) unsigned swinmh_r7_raw[swinmh_r7_bytes/4];
__device__ __align__(128) unsigned swinmh_r7_data[swinmh_r7_bytes/2];
__device__ __forceinline__ unsigned swinmh_r7_cg(const unsigned* p) {
    unsigned v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ uint2 swinmh_r7_cg2(const unsigned* p) {
    uint2 v;asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"l"(p):"memory");return v;
}
__device__ __forceinline__ bool swinmh_r7_acquire(const unsigned char* w,float* temp) {
    int tid=threadIdx.y*32+threadIdx.x;
    if(tid==0) {
        for(;;) {
            unsigned s=atomicAdd(&swinmh_r7_state,0u);
            if(s==0) {if(atomicCAS(&swinmh_r7_state,0u,0x80000000u)==0u){temp[0]=__uint_as_float(2u);break;}}
            else if((s&0x80000000u)==0 && atomicCAS(&swinmh_r7_state,s,s+1)==s) {
                __threadfence();
                bool match=swinmh_r7_key==reinterpret_cast<unsigned long long>(w);
                if(!match)atomicSub(&swinmh_r7_state,1u);
                temp[0]=__uint_as_float(match?1u:0u);break;
            }
        }
    }
    __syncthreads();
    unsigned action=__float_as_uint(temp[0]);
    if(action==2) {
        unsigned diff=swinmh_r7_key!=reinterpret_cast<unsigned long long>(w);
        for(int j=tid;j<swinmh_r7_bytes/4;j+=32*swinmh_H)
            diff|=swinmh_r7_cg(reinterpret_cast<const unsigned*>(w)+j)^swinmh_r7_cg(swinmh_r7_raw+j);
        int changed=__syncthreads_or(diff!=0);
        if(changed) {
            for(int word=tid;word<swinmh_r7_bytes/4;word+=32*swinmh_H) {
                unsigned v=swinmh_r7_cg(reinterpret_cast<const unsigned*>(w)+word);
                swinmh_r7_raw[word]=v;
                int o=4*(word&255),j=o>>9,lane=(o>>4)&31,q=(o>>2)&3;
                int n=8*(2*j+(q>>1))+(lane>>2),k0=4*(lane&3)+16*(q&1);
                int nc=((n&7)>>1)*8+4*(n>>4)+2*((n>>3)&1)+(n&1);
                #pragma unroll
                for(int byte=0;byte<4;byte+=2) {
                    int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                    unsigned a=__float_as_uint(swinmh_decode(v>>(8*byte)))>>16;
                    unsigned b=__float_as_uint(swinmh_decode(v>>(8*(byte+1))))>>16;
                    auto out=reinterpret_cast<unsigned short*>(swinmh_r7_data)+(word/256)*1024;
                    out[k*32+(nc^swinmh_bank(k))]=a;
                    out[(k+1)*32+(nc^swinmh_bank(k+1))]=b;
                }
            }
        }
        __threadfence();__syncthreads();
        if(tid==0) {
            swinmh_r7_key=reinterpret_cast<unsigned long long>(w);
            atomicAdd(&swinmh_r7_checks,1u);
            if(changed)atomicAdd(&swinmh_r7_builds,1u);
            __threadfence();atomicExch(&swinmh_r7_state,1u);
        }
    }
    __syncthreads();return action!=0;
}
__device__ __forceinline__ void swinmh_r7_release(bool cached) {
    __syncthreads();if(cached&&threadIdx.x==0&&threadIdx.y==0)atomicSub(&swinmh_r7_state,1u);
}
