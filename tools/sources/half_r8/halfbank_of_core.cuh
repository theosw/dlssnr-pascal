#pragma once
#include "r4_silu_lut.cuh"
#include "native_swin1h_r6_of_helpers.cuh"

namespace swin1h_native_swin1h_r6_of {
__device__ __forceinline__ unsigned short swin1h_r4_state_read(const __half* p,int i) {
    unsigned a=__cvta_generic_to_shared(p+i);unsigned short v;
    asm("ld.shared.b16 %0,[%1];":"=h"(v):"r"(a));return v;
}
__device__ __forceinline__ float swin1h_r4_silu(float x) {
    unsigned h=__half_as_ushort(__float2half_rn(x));
    return E4M3(__ldg(swin1h_r4_silu_lut+h));
}
struct Packed { unsigned w[8]; };
struct __align__(16) Scratch { float weights[64][32]; union { unsigned short kv[8192]; float decoded[4096]; }; };
__device__ __forceinline__ float Read(const Packed& x,int k) { return E4M3(x.w[k>>2]>>(8*(k&3))); }
__device__ __forceinline__ unsigned Pair(float a,float b) {
    return unsigned(__half_as_ushort(__float2half_rn(a))) | (unsigned(__half_as_ushort(__float2half_rn(b)))<<16);
}
__device__ __forceinline__ void Pixel(int tok,int& x,int& y) {
    const int m=tok>>4,r=tok&15;
    x=4*(m&1)+(r&3); y=4*(m>>1)+((r>>2)&1)+2*(r>>3);
}
__device__ __forceinline__ void MergePack(const float (&x)[16],Packed& out) {
    const int half=(threadIdx.x&127)&1;
#pragma unroll
    for(int j=0;j<4;++j) {
        const unsigned word=EncodeE4M3(x[4*j]) | EncodeE4M3(x[4*j+1])<<8 |
            EncodeE4M3(x[4*j+2])<<16 | EncodeE4M3(x[4*j+3])<<24;
        const unsigned other=__shfl_xor_sync(0xFFFFFFFFu,word,1);
        out.w[j]=half?other:word; out.w[j+4]=half?word:other;
    }
}
__device__ __forceinline__ void Weights(const unsigned char* blob,int offset,Scratch& s) {
    __syncthreads();
    for(int word=threadIdx.x;word<256;word+=blockDim.x) {
        const unsigned v=__ldg(reinterpret_cast<const unsigned*>(blob+offset)+word);
        const int o=4*word,j=o>>9,l=(o>>4)&31,q=(o>>2)&3;
        const int n=8*(2*j+(q>>1))+(l>>2),k0=4*(l&3)+16*(q&1);
#pragma unroll
        for(int b=0;b<4;++b) { const int k=k0+b; s.weights[16*(k>>4)+Channel16(k&15)][n]=E4M3(v>>(8*b)); }
    }
    __syncthreads();
}
__device__ __forceinline__ void Product(const Packed& x,const Scratch& s,float (&h)[16]) {
    const int ch=16*((threadIdx.x&127)&1);
#pragma unroll
    for(int n=0;n<16;++n)h[n]=0.0f;
#pragma unroll
    for(int k=0;k<32;++k) {
        const float a=Read(x,k);
#pragma unroll
        for(int n=0;n<16;++n)h[n]=fmaf(a,s.weights[k][ch+n],h[n]);
    }
}
__device__ __forceinline__ void ProductFfn(const Packed& x,Scratch& s,float (&h)[16]) {
    const int tok=(threadIdx.x&127)>>1,ch=16*((threadIdx.x&127)&1);
    float* xs=(s.decoded+2048*(threadIdx.x>>7));
#pragma unroll
    for(int d=0;d<16;++d)xs[(ch+d)*64+tok]=Read(x,ch+d);
    __syncthreads();
#pragma unroll
    for(int n=0;n<16;++n)h[n]=0.0f;
#pragma unroll 2
    for(int k=0;k<32;++k) {
        const float a=xs[k*64+tok];
#pragma unroll
        for(int n=0;n<16;++n)h[n]=fmaf(a,s.weights[k][ch+n],h[n]);
    }
}
__device__ __forceinline__ void Norm(float (&x)[16]) {
    float ss=0.0f;
#pragma unroll
    for(int c=0;c<16;++c)ss+=Half(x[c]*x[c]);
    ss+=__shfl_xor_sync(0xFFFFFFFFu,ss,1);
    const float r=Half(rsqrtf(fmaxf(Half(ss),6.198883056640625e-5f)));
#pragma unroll
    for(int c=0;c<16;++c)x[c]=Half(x[c]*r);
}
__device__ __forceinline__ float Exponent(float v) {
    const float t=fminf(fmaxf(Half(fmaf(v,0.044921875f,1.30078125f)),1.03125f),1.5693359375f);
    return __half2float(__ushort_as_half((unsigned(__half_as_ushort(__float2half_rn(t)))-0x3C00u)<<5));
}
__device__ __forceinline__ float2 Exponents(const float (&q)[32],const Scratch& s,const unsigned* bias,int tok,int kp) {
    float a=0.0f,b=0.0f;
#pragma unroll
    for(int d=0;d<32;++d) {
        const unsigned k=*reinterpret_cast<const unsigned*>(&(s.kv+4096*(threadIdx.x>>7))[d*64+2*kp]);
        a=fmaf(q[d],__uint_as_float(k<<16),a); b=fmaf(q[d],__uint_as_float(k&0xFFFF0000u),b);
    }
    const int i=(tok&31)>>4,row=tok&15,g=row&7,wb=row>>3,j=kp>>2,t=kp&3;
    const unsigned bw=__ldg(bias+(4*i+(j>>1))*128+(4*g+t)*4+2*(j&1)+wb);
    return make_float2(Exponent(Half(a+LowHalf(bw))),Exponent(Half(b+HighHalf(bw))));
}

// Block128. Thread2*token+half owns16 natural channels; paired threads replicate Packed vectors.
// Scratch is12KB. Return16 f16-rounded output values; caller owns views, waits, signals and pooling.
template<int Count> __device__ __forceinline__ void swin1h_r3_loadweights(const unsigned char* blob,float* w) {
    for(int word=threadIdx.x;word<Count*256;word+=blockDim.x) {
        const unsigned v=__ldg(reinterpret_cast<const unsigned*>(blob)+word);
        const int o=4*(word&255),j=o>>9,l=(o>>4)&31,q=(o>>2)&3;
        const int n=8*(2*j+(q>>1))+(l>>2),k0=4*(l&3)+16*(q&1);
#pragma unroll
        for(int b=0;b<4;b++) {int k=k0+b;int nk=16*(k>>4)+Channel16(k&15);w[(word/256)*1024+nk*32+(n^(((nk&6)<<2)|((nk&16)>>2)))]=E4M3(v>>(8*b));}
    }
}
template<int T,int C,int UNROLL=1> __device__ __forceinline__ void swin1h_r3_r4_dot(const float* xs,const float* w,float (&h)[T][C]) {
    const int t0=((threadIdx.x&127)/(32/C))*T,c0=((threadIdx.x&127)%(32/C))*C;
#pragma unroll
    for(int t=0;t<T;t++) {
#pragma unroll
        for(int c=0;c<C;c++)h[t][c]=0;
    }
#pragma unroll UNROLL
    for(int k=0;k<32;k++) {
        float av[T],wv[C];
#pragma unroll
        for(int t=0;t<T/4;t++) {float4 a=((const float4*)(xs+k*64+(t0^((k>>2)*4))))[t];av[4*t]=a.x;av[4*t+1]=a.y;av[4*t+2]=a.z;av[4*t+3]=a.w;}
        if constexpr(T==2) {float2 a=*(const float2*)(xs+k*64+(t0^((k>>2)*4)));av[0]=a.x;av[1]=a.y;}
#pragma unroll
        for(int c=0;c<C/4;c++) {float4 a=((const float4*)(w+k*32+(c0^(((k&6)<<2)|((k&16)>>2)))))[c];wv[4*c]=a.x;wv[4*c+1]=a.y;wv[4*c+2]=a.z;wv[4*c+3]=a.w;}
#pragma unroll
        for(int t=0;t<T;t++) {
#pragma unroll
            for(int c=0;c<C;c++)h[t][c]=fmaf(av[t],wv[c],h[t][c]);
        }
    }
}

template<int T,int C,int U=1,int PERSIST=0> __device__ __forceinline__ void swin1h_r3_r4_tile(const __half* state,const unsigned char* blob,int scale,Scratch& s,float (&out)[16]) {
    float* xs=reinterpret_cast<float*>((s.kv+4096*(threadIdx.x>>7)));
    float* weights=&s.weights[0][0];
    const int t0=((threadIdx.x&127)/(32/C))*T,c0=((threadIdx.x&127)%(32/C))*C;
    if constexpr(PERSIST) {swin1h_r3_loadweights<8>(blob,weights);__syncthreads();}
    {
        unsigned orig[T/2][C],acc[T][C/2];float h[T][C];
#pragma unroll
        for(int t=0;t<T;t++) {
          float vals[C];
          #pragma unroll
          for(int c=0;c<C;c++) {
            float x=(scale==10256)?__half2float(__ushort_as_half(swin1h_r4_state_read(state,(t0+t)*32+c0+c))):__uint_as_float(unsigned(swin1h_r4_state_read(state,(t0+t)*32+c0+c))<<16);
            float o=(scale==10256)?Fp8(x):x;
            unsigned bits=__float_as_uint(o)>>16;
            if((t&1)==0)orig[t/2][c]=bits;else orig[t/2][c]|=bits<<16;
            vals[c]=x*__half2float(*((const __half*)(blob+scale)+c0+c));
          }
          #pragma unroll
          for(int c=0;c<C/2;c++)acc[t][c]=Pair(vals[2*c],vals[2*c+1]);
        }
        for(int b=0;b<4;b++) {
            __syncthreads();
#pragma unroll
            for(int c=0;c<C;c++)*(float4*)(xs+(c0+c)*64+(t0^(((c0+c)>>2)*4)))=make_float4(__uint_as_float(orig[0][c]<<16),__uint_as_float(orig[0][c]&0xffff0000u),__uint_as_float(orig[1][c]<<16),__uint_as_float(orig[1][c]&0xffff0000u));
            if constexpr(!PERSIST)swin1h_r3_loadweights<1>(blob+b*1024,weights);
            __syncthreads();swin1h_r3_r4_dot<T,C,U>(xs,weights+(PERSIST?b*1024:0),h);
            __syncthreads();
#pragma unroll
            for(int c=0;c<C;c++)*(float4*)(xs+(c0+c)*64+(t0^(((c0+c)>>2)*4)))=make_float4(swin1h_r4_silu(h[0][c]),swin1h_r4_silu(h[1][c]),swin1h_r4_silu(h[2][c]),swin1h_r4_silu(h[3][c]));
            if constexpr(!PERSIST)swin1h_r3_loadweights<1>(blob+(b+4)*1024,weights);
            __syncthreads();swin1h_r3_r4_dot<T,C,U>(xs,weights+(PERSIST?(b+4)*1024:0),h);
#pragma unroll
            for(int t=0;t<T;t++) {
#pragma unroll
                for(int c=0;c<C/2;c++)acc[t][c]=Pair(LowHalf(acc[t][c])+h[t][2*c],HighHalf(acc[t][c])+h[t][2*c+1]);
            }
        }
__syncthreads();
#pragma unroll
        for(int t=0;t<T;t++) {
#pragma unroll
            for(int c=0;c<C/2;c++)reinterpret_cast<unsigned*>((s.kv+4096*(threadIdx.x>>7)))[(t0+t)*16+c0/2+c]=acc[t][c];
        }
    }
    __syncthreads();
#pragma unroll
    for(int d=0;d<16;d++)out[d]=__half2float(reinterpret_cast<__half*>((s.kv+4096*(threadIdx.x>>7)))[((threadIdx.x&127)>>1)*32+16*((threadIdx.x&127)&1)+d]);
    __syncthreads();
}



__device__ __forceinline__ void swin1h_r3_ProductQkv(Scratch& s,float (&out)[16]) {
    const int t0=((threadIdx.x&127)>>3)*4,c0=((threadIdx.x&127)&7)*4;
    float h[4][4]={};
#pragma unroll 4
    for(int k=0;k<32;k++) {
        const uint2 a=*(const uint2*)((s.kv+4096*(threadIdx.x>>7))+2048+k*64+t0);
        const float4 w=*(const float4*)(&s.weights[k][c0]);
        const float av[4]={__uint_as_float(a.x<<16),__uint_as_float(a.x&0xffff0000u),__uint_as_float(a.y<<16),__uint_as_float(a.y&0xffff0000u)};
        const float wv[4]={w.x,w.y,w.z,w.w};
#pragma unroll
        for(int t=0;t<4;t++) {
#pragma unroll
            for(int c=0;c<4;c++)h[t][c]=fmaf(av[t],wv[c],h[t][c]);
        }
    }
    __syncthreads();
#pragma unroll
    for(int t=0;t<4;t++) {
#pragma unroll
        for(int c=0;c<2;c++)(reinterpret_cast<unsigned*>(s.weights)+1024*(threadIdx.x>>7))[(t0+t)*16+c0/2+c]=Pair(h[t][2*c],h[t][2*c+1]);
    }
    __syncthreads();
    const int tok=(threadIdx.x&127)>>1,ch=16*((threadIdx.x&127)&1);
#pragma unroll
    for(int d=0;d<8;d++) {unsigned w=(reinterpret_cast<unsigned*>(s.weights)+1024*(threadIdx.x>>7))[tok*16+ch/2+d];out[2*d]=LowHalf(w);out[2*d+1]=HighHalf(w);}
    __syncthreads();
}

template<bool OnePass=false,bool Prepared=false>
__device__ __forceinline__ void Core4(const Packed& input,const unsigned char* blob,Scratch& s,float (&out)[16],int extraOffset=0,const float* extra=nullptr) {
    const int tok=(threadIdx.x&127)>>1,half=(threadIdx.x&127)&1,ch=16*half;
    __half* state=reinterpret_cast<__half*>((s.decoded+2048*(threadIdx.x>>7)));
    __syncthreads();
#pragma unroll
    if constexpr(!Prepared) for(int d=0;d<16;d++)state[tok*32+ch+d]=extra?__float2half_rn(extra[d]):__ushort_as_half(__float_as_uint(Read(input,ch+d))>>16);
    __syncthreads();
    float h[16];swin1h_r3_r4_tile<4,4,32,0>(state,blob,extraOffset?10256:8208,s,h);
#pragma unroll
    for(int d=0;d<16;d++)(s.kv+4096*(threadIdx.x>>7))[2048+(ch+d)*64+tok]=__float_as_uint(Fp8(h[d]))>>16;
    __syncthreads();
    unsigned ys[8];
#pragma unroll
    for(int j=0;j<8;++j) {
        const unsigned scale=__ldg(reinterpret_cast<const unsigned*>(blob+20592+extraOffset)+8*half+j);
        ys[j]=Pair(h[2*j]*LowHalf(scale),h[2*j+1]*HighHalf(scale));
    }
    Weights(blob,9312+extraOffset,s);swin1h_r3_ProductQkv(s,h);
#pragma unroll
    for(int c=0;c<16;++c)h[c]=Half(h[c]);
    Norm(h);
#pragma unroll
    for(int d=0;d<16;++d)(s.kv+4096*(threadIdx.x>>7))[(ch+d)*64+tok]=__float_as_uint(Fp8(h[d]))>>16;
    Weights(blob,8288+extraOffset,s);swin1h_r3_ProductQkv(s,h);
#pragma unroll
    for(int c=0;c<16;++c)h[c]=Half(h[c]);
    Norm(h);
    const float tau=Half(__ldg(reinterpret_cast<const float*>(blob+19552+extraOffset)));
#pragma unroll
    for(int c=0;c<16;++c)h[c]=Half(h[c]*tau);
    Packed qp;MergePack(h,qp);
    Weights(blob,10336+extraOffset,s);swin1h_r3_ProductQkv(s,h);
    __syncthreads();
#pragma unroll
    for(int d=0;d<16;d++) {
        (reinterpret_cast<unsigned short*>(s.weights)+2048*(threadIdx.x>>7))[tok*32+ch+d]=__float_as_uint(Fp8(Half(h[d])))>>16;
        (s.kv+4096*(threadIdx.x>>7))[2048+(ch+d)*64+tok]=__float_as_uint(Read(qp,ch+d))>>16;
    }
    __syncthreads();

    const int qt0=((threadIdx.x&127)>>3)*4,kj0=((threadIdx.x&127)&7)*8;
    {
      const int qt=((threadIdx.x&127)>>3)*4,kt=((threadIdx.x&127)&7)*8;
      float score[4][8]={};
#pragma unroll 4
      for(int d=0;d<32;d++) {
        uint2 a=*(const uint2*)((s.kv+4096*(threadIdx.x>>7))+2048+d*64+qt);
        uint4 k=*(const uint4*)((s.kv+4096*(threadIdx.x>>7))+d*64+kt);
        float av[4]={__uint_as_float(a.x<<16),__uint_as_float(a.x&0xffff0000u),__uint_as_float(a.y<<16),__uint_as_float(a.y&0xffff0000u)};
        unsigned kw[4]={k.x,k.y,k.z,k.w};
#pragma unroll
        for(int c=0;c<4;c++) {
          float lo=__uint_as_float(kw[c]<<16),hi=__uint_as_float(kw[c]&0xffff0000u);
#pragma unroll
          for(int t=0;t<4;t++) {score[t][2*c]=fmaf(av[t],lo,score[t][2*c]);score[t][2*c+1]=fmaf(av[t],hi,score[t][2*c+1]);}
        }
      }
      __syncthreads(); // Q and K die together before exponent staging aliases them.
#pragma unroll
      for(int t=0;t<4;t++) {
        int token=qt+t;
        const unsigned* bias=reinterpret_cast<const unsigned*>(blob+11360+extraOffset+4096*(token>>5));
#pragma unroll
        for(int c=0;c<4;c++) {
          const int kp=kt/2+c,i=(token&31)>>4,row=token&15,g=row&7,wb=row>>3,j=kp>>2,tt=kp&3;
          unsigned bw=__ldg(bias+(4*i+(j>>1))*128+(4*g+tt)*4+2*(j&1)+wb);
          const float e0=Exponent(Half(score[t][2*c]+LowHalf(bw)));
          const float e1=Exponent(Half(score[t][2*c+1]+HighHalf(bw)));
          reinterpret_cast<unsigned*>(s.kv+4096*(threadIdx.x>>7))[(kt/2+c)*64+(token^((kt&32)>>1))]=(__float_as_uint(e0)>>16)|(__float_as_uint(e1)&0xffff0000u);
        }
      }
    }
    __syncthreads();
    float sum=0;
#pragma unroll 1
    for(int kp=0;kp<16;kp++) {
      const unsigned pair=reinterpret_cast<const unsigned*>(s.kv+4096*(threadIdx.x>>7))[(16*half+kp)*64+(tok^(16*half))];
      float a=__uint_as_float(pair<<16),b=__uint_as_float(pair&0xffff0000u);
      sum+=a+b;
    }
    sum+=__shfl_xor_sync(0xffffffffu,sum,1);
    const float rr=Half(1.0f/Half(sum));
#pragma unroll 1
    for(int kp=0;kp<16;kp++) {
      const unsigned pair=reinterpret_cast<const unsigned*>(s.kv+4096*(threadIdx.x>>7))[(16*half+kp)*64+(tok^(16*half))];
      float a=__uint_as_float(pair<<16),b=__uint_as_float(pair&0xffff0000u);
      const float pa=Fp8(Half(a*rr)),pb=Fp8(Half(b*rr));
      reinterpret_cast<unsigned*>(s.kv+4096*(threadIdx.x>>7))[(16*half+kp)*64+(tok^(16*half))]=(__float_as_uint(pa)>>16)|(__float_as_uint(pb)&0xffff0000u);
    }
    __syncthreads();
    const int t0=((threadIdx.x&127)>>3)*4,c0=((threadIdx.x&127)&7)*4;
    unsigned low[4][2];float pv[4][4];
#pragma unroll 1
    for(int part=0;part<2;part++) {
#pragma unroll
        for(int t=0;t<4;t++) {
#pragma unroll
            for(int c=0;c<4;c++)pv[t][c]=0;
        }
#pragma unroll 4
        for(int k=0;k<32;k++) {
            uint4 a=*(const uint4*)(reinterpret_cast<const unsigned*>(s.kv+4096*(threadIdx.x>>7))+(part*16+k/2)*64+(t0^(16*part)));
            uint2 w=*(const uint2*)((reinterpret_cast<const unsigned short*>(s.weights)+2048*(threadIdx.x>>7))+(part*32+k)*32+c0);
            float av[4]={__uint_as_float((k&1)?(a.x&0xffff0000u):(a.x<<16)),__uint_as_float((k&1)?(a.y&0xffff0000u):(a.y<<16)),__uint_as_float((k&1)?(a.z&0xffff0000u):(a.z<<16)),__uint_as_float((k&1)?(a.w&0xffff0000u):(a.w<<16))};
            float wv[4]={__uint_as_float(w.x<<16),__uint_as_float(w.x&0xffff0000u),__uint_as_float(w.y<<16),__uint_as_float(w.y&0xffff0000u)};
#pragma unroll
            for(int t=0;t<4;t++) {
#pragma unroll
                for(int c=0;c<4;c++)pv[t][c]=fmaf(av[t],wv[c],pv[t][c]);
            }
        }
        if(part==0) {
#pragma unroll
            for(int t=0;t<4;t++) {
#pragma unroll
                for(int c=0;c<2;c++)low[t][c]=Pair(pv[t][2*c],pv[t][2*c+1]);
            }
        }
    }
    __syncthreads();
    float* xs=reinterpret_cast<float*>((s.kv+4096*(threadIdx.x>>7)));
#pragma unroll
    for(int c=0;c<4;c++) {
        float v[4];
#pragma unroll
        for(int t=0;t<4;t++) {
            float lo=(c&1)?HighHalf(low[t][c/2]):LowHalf(low[t][c/2]);
            v[t]=Fp8(Half(lo+pv[t][c]));
        }
        *(float4*)(xs+(c0+c)*64+(t0^(((c0+c)>>2)*4)))=make_float4(v[0],v[1],v[2],v[3]);
    }
    swin1h_r3_loadweights<1>(blob+19568+extraOffset,&s.weights[0][0]);
    __syncthreads();swin1h_r3_r4_dot<4,4,32>(xs,&s.weights[0][0],pv);
    __syncthreads();
#pragma unroll
    for(int t=0;t<4;t++) {
#pragma unroll
        for(int c=0;c<4;c++)reinterpret_cast<float*>((s.kv+4096*(threadIdx.x>>7)))[(t0+t)*32+c0+c]=pv[t][c];
    }
    __syncthreads();
#pragma unroll
    for(int d=0;d<16;d++)out[d]=reinterpret_cast<float*>((s.kv+4096*(threadIdx.x>>7)))[tok*32+ch+d];

#pragma unroll
    for(int j=0;j<8;++j){out[2*j]=Half(out[2*j]+LowHalf(ys[j]));out[2*j+1]=Half(out[2*j+1]+HighHalf(ys[j]));}
}

__device__ __forceinline__ Packed Planar(const unsigned char* in,int H,int W,int x,int y) {
    Packed out;unsigned vals[8]={};
#pragma unroll
    for(int h=0;h<2;++h) {
        uint4 v=make_uint4(0,0,0,0);
        if(x>=0&&x<W&&y>=0&&y<H)v=__ldg(reinterpret_cast<const uint4*>(in+((size_t(h)*H+y)*W+x)*16));
        const unsigned words[4]={v.x,v.y,v.z,v.w};
#pragma unroll
        for(int s=0;s<16;++s){const int c=16*h+Channel16(s);vals[c>>2]|=((words[s>>2]>>(8*(s&3)))&255u)<<(8*(c&3));}
    }
#pragma unroll
    for(int j=0;j<8;++j)out.w[j]=vals[j];
    return out;
}
__device__ __forceinline__ void StoreBlocked(unsigned char* out,int H,int W,int wx,int wy,const float (&v)[16]) {
    const int tok=(threadIdx.x&127)>>1,half=(threadIdx.x&127)&1,m=tok>>4,row=tok&15;
    const int bx=(wx>>2)+(m&1),by=(wy>>2)+(m>>1),blocksX=(W+3)/4;
    if((2*blockIdx.x+(threadIdx.x>>7))<gridDim.x&&bx>=0&&bx<blocksX&&by>=0&&by<(H+3)/4) {
        unsigned* dst=reinterpret_cast<unsigned*>(out+(size_t(by)*blocksX+bx)*512);
#pragma unroll
        for(int st=0;st<4;++st) {
            unsigned word=0;
#pragma unroll
            for(int b=0;b<4;++b)word|=EncodeE4M3(v[Channel16(4*st+b)])<<(8*b);
            dst[(4*(row&7)+st)*4+2*half+(row>>3)]=word;
        }
    }
}
}
