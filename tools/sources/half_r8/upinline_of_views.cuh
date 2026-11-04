#pragma once
#include "upinline_of_core.cuh"
namespace swin1h_native_swin1h_r6_of {
struct __align__(8) Views {
    const unsigned char* input; unsigned char* output; const unsigned char* blob;
    int H,W,x0,y0; const int* ready; int producerX,producerY; int* signal;
    unsigned char* aux; int auxH,auxW; const unsigned char* skip; int viewH,viewW;
};
static_assert(sizeof(Views)==96,"ABI");
__device__ __forceinline__ void Wait(const Views& p,int wx,int wy) {
    if((threadIdx.x&127)==0&&p.ready) {
        const int nx=(p.W-min(p.producerX,0)+7)/8;
        const int ax=(max(wx,0)-p.producerX)/8,bx=(min(wx+7,p.W-1)-p.producerX)/8;
        const int ay=(max(wy,0)-p.producerY)/8,by=(min(wy+7,p.H-1)-p.producerY)/8;
        if(wx<p.W&&wx+8>0&&wy<p.H&&wy+8>0)
        for(int y=ay;y<=by;++y)for(int x=ax;x<=bx;++x) {
            const int* flag=p.ready+y*nx+x;int value;
            do { asm volatile("ld.global.cg.b32 %0,[%1];":"=r"(value):"l"(flag):"memory"); } while(value<0);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ void Signal(const Views& p) {
    __threadfence();__syncthreads();
    if((threadIdx.x&127)==0&&(2*blockIdx.x+(threadIdx.x>>7))<gridDim.x&&p.signal)*reinterpret_cast<volatile int*>(p.signal+blockIdx.y*((p.W-min(p.x0,0)+7)/8)+(2*blockIdx.x+(threadIdx.x>>7)))=0;
}
__device__ __forceinline__ Packed Blocked(const unsigned char* in,int H,int W,int x,int y) {
    Packed out;unsigned bytes[8]={};
    if(x>=0&&x<W&&y>=0&&y<H) {
        const int row=(x&3)+4*(y&1)+8*((y>>1)&1),g=row&7,wb=row>>3;
        const unsigned* block=reinterpret_cast<const unsigned*>(in+(size_t(y/4)*((W+3)/4)+x/4)*512);
#pragma unroll
        for(int half=0;half<2;++half) {
#pragma unroll
            for(int st=0;st<4;++st) {
                const unsigned w=__ldg(block+(4*g+st)*4+2*half+wb);
#pragma unroll
                for(int b=0;b<4;++b){const int c=16*half+Channel16(4*st+b);bytes[c>>2]|=((w>>(8*b))&255u)<<(8*(c&3));}
            }
        }
    }
#pragma unroll
    for(int j=0;j<8;++j)out.w[j]=bytes[j];
    return out;
}
__device__ __forceinline__ void StorePlanar(unsigned char* out,int H,int W,int X,int Y,const float (&v)[16]) {
    if((2*blockIdx.x+(threadIdx.x>>7))<gridDim.x&&X>=0&&X<W&&Y>=0&&Y<H) {
        const int half=(threadIdx.x&127)&1; unsigned words[4];
#pragma unroll
        for(int st=0;st<4;++st) {
            unsigned word=0;
#pragma unroll
            for(int b=0;b<4;++b)word|=EncodeE4M3(v[Channel16(4*st+b)])<<(8*b);
            words[st]=word;
        }
        *reinterpret_cast<uint4*>(out+((size_t(half)*H+Y)*W+X)*16)=make_uint4(words[0],words[1],words[2],words[3]);
    }
}
__device__ __forceinline__ int Token(int x,int y){return 16*((x>>2)+2*(y>>2))+(x&3)+4*(y&1)+8*((y>>1)&1);}
__device__ __forceinline__ void Downsample(const Views& p,Scratch& s,int wx,int wy,const float (&v)[16]) {
    __syncthreads();
    const int token=(threadIdx.x&127)>>1,ch=16*((threadIdx.x&127)&1);
#pragma unroll
    for(int d=0;d<16;++d)(s.kv+4096*(threadIdx.x>>7))[(ch+d)*64+token]=__half_as_ushort(__float2half_rn(v[d]));
    __syncthreads();
    const int tok=(threadIdx.x&127)>>3,grp=(threadIdx.x&127)&7,lx=2*(tok&3),ly=2*(tok>>2);
    const int X=wx/2+(tok&3),Y=wy/2+(tok>>2),H=p.auxH>0?p.auxH:(p.H+1)/2,W=p.auxW>0?p.auxW:(p.W+1)/2;
    float pooled[32];
#pragma unroll
    for(int k=0;k<32;++k) {
        const float a=__half2float(__ushort_as_half((s.kv+4096*(threadIdx.x>>7))[k*64+Token(lx,ly)]));
        const float b=__half2float(__ushort_as_half((s.kv+4096*(threadIdx.x>>7))[k*64+Token(lx+1,ly)]));
        const float c=__half2float(__ushort_as_half((s.kv+4096*(threadIdx.x>>7))[k*64+Token(lx,ly+1)]));
        const float d=__half2float(__ushort_as_half((s.kv+4096*(threadIdx.x>>7))[k*64+Token(lx+1,ly+1)]));
        pooled[k]=Fp8(Half(Half(Half(a+b)+Half(c+d))*0.25f));
    }
#pragma unroll 1
    for(int block=0;block<2;++block) {
        Weights(p.blob,20656+1024*block,s);
        float z[4]={};
#pragma unroll
        for(int k=0;k<32;++k) {
#pragma unroll
            for(int n=0;n<4;++n)z[n]=fmaf(pooled[k],s.weights[k][4*grp+n],z[n]);
        }
        if((2*blockIdx.x+(threadIdx.x>>7))<gridDim.x&&p.aux&&X>=0&&X<W&&Y>=0&&Y<H) {
#pragma unroll
            for(int n=0;n<4;++n) {
                const int c=32*block+4*grp+n,h=c/16,cn=c&15;
                const int stored=4*((cn&7)>>1)+(cn&1)+2*((cn>>3)&1);
                p.aux[((size_t(h)*H+Y)*W+X)*16+stored]=EncodeE4M3(Half(z[n]));
            }
        }
    }
}
__device__ __forceinline__ void Upsample(const Views& p,Scratch& s,int X,int Y,float (&v)[16]) {
    const int H=p.H/2,W=p.W/2;
    const Packed a=Planar(p.input,H,W,X/2,Y/2);
    const Packed b=Planar(p.input+size_t(H)*W*32,H,W,X/2,Y/2);
    Weights(p.blob,8192,s);Product(a,s,v);
#pragma unroll
    for(int n=0;n<16;++n)v[n]=Half(v[n]);
    float tail[16];Weights(p.blob,9216,s);Product(b,s,tail);
#pragma unroll
    for(int n=0;n<16;++n)v[n]=Half(v[n]+tail[n]);
}
template<bool BlockedInput>
__device__ __forceinline__ void swin1h_native_swin1h_r6(const unsigned char* input,int H,int W,int x,int y,Scratch& s) {
    const int tok=(threadIdx.x&127)>>1,half=(threadIdx.x&127)&1;
    const bool valid=x>=0&&x<W&&y>=0&&y<H;
#pragma unroll
    for(int st=0;st<4;st++) {
        unsigned word=0;
        if(valid) {
            if constexpr(BlockedInput) {
                const int row=(x&3)+4*(y&1)+8*((y>>1)&1),g=row&7,wb=row>>3;
                const unsigned* block=reinterpret_cast<const unsigned*>(input+(size_t(y/4)*((W+3)/4)+x/4)*512);
                word=__ldg(block+(4*g+st)*4+2*half+wb);
            } else word=__ldg(reinterpret_cast<const unsigned*>(input+((size_t(half)*H+y)*W+x)*16)+st);
        }
#pragma unroll
        for(int b=0;b<4;b++)reinterpret_cast<unsigned short*>((s.decoded+2048*(threadIdx.x>>7)))[tok*32+16*half+Channel16(4*st+b)]=__float_as_uint(E4M3(word>>(8*b)))>>16;
    }
}

}
