// NR 310.9.16 ViT attention: 32 heads, f16 exponent-reduction tree and padded-token subtraction.
#include "s16v_common.cuh"
struct __align__(8) s16v_AttentionParams {const unsigned char *Q,*K,*V;unsigned char* O;const void* unused;const int* wait;int* signal;int H,W;};
static_assert(sizeof(s16v_AttentionParams)==64,"params");
__device__ __forceinline__ int s16v_qoffset(int tok,int dim,int head) {return (tok/16)*16384+head*512+s16v_aoff(tok&15,s16v_stored(dim));}
__device__ __forceinline__ int s16v_koffset(int tok,int dim,int head) {
 int a=s16v_aoff(tok&15,s16v_stored(dim));a=(a&~15)+(a&3)+((a&4)<<1)+((a&8)>>1);
 return (tok/16)*16384+head*512+a;
}
__device__ __forceinline__ int s16v_voffset(int tok,int dim,int head) {
 int pos=16*((tok&31)>>4)+s16v_stored(tok&15),l=(dim&7)*4+(pos&15)/4,r=((dim&15)/8)*2+pos/16;
 return (tok/32)*32768+head*1024+(dim/16)*512+l*16+r*4+(pos&3);
}
__device__ __forceinline__ float s16v_vitexp(float s) {
 float t=fminf(fmaxf(s16v_half(fmaf(s,s16v_half(.08953946828842163f),s16v_half(1.7093614339828491f))),1.439453125f),1.9775390625f);
 return __half2float(__ushort_as_half((unsigned short)((__half_as_ushort(__float2half_rn(t))-0x3c00u)<<4)));
}

// vitexp returns positive finite half values below2; addition rounds FP8 subnormals exactly.
__device__ __forceinline__ unsigned s16v_positive_exp8(float x) {
 unsigned b=__float_as_uint(x);
 if(b<0x3c800000u)return __float_as_uint(x+16384.f)-0x46800000u;
 return ((b+0x7ffffu+((b>>20)&1u))>>20)-960u;
}

extern "C" __global__ __launch_bounds__(128,3) void cc_vit_1d_attention_chained_fp8(s16v_AttentionParams p) {
 __shared__ __align__(16) float s16v_K[32][64];
 __shared__ __align__(16) float s16v_V[64][32];
 __shared__ unsigned char s16v_P[96][20];
 int l=threadIdx.x,w=threadIdx.y,th=w*32+l,g=l>>2,t=l&3,head=blockIdx.x,T=p.H*p.W;
 if((int)blockIdx.y>=(T+95)/96)return;
 if(th==0) {
  for(int tile=0;tile<(T+127)/128;++tile)s16v_wait(p.wait+tile*16+head/2);
  __threadfence();
 }
 __syncthreads();
 for(int work=(int)blockIdx.y;work<(T+95)/96;work+=(int)gridDim.y) {
  int qbase=work*96;
  float out[3][8]={},sum[3]={};
  unsigned q8[8];
#pragma unroll
  for(int word=0;word<8;++word) {
   unsigned bits=0;
#pragma unroll
   for(int z=0;z<4;++z) {
    int tok=qbase+w*24+l;
    bits|=(unsigned)(l<24&&tok<T?p.Q[s16v_qoffset(tok,word*4+z,head)]:0)<<(8*z);
   }
   q8[word]=bits;
  }
  for(int base=0;base<T;base+=64) {
   __syncthreads();
   uint4 kw={},vw={};
   int kb=th>>5,vb=th>>6;
   if(base+kb*16<T)kw=*reinterpret_cast<const uint4*>(p.K+((base/16)+kb)*16384+head*512+(th&31)*16);
   if(base+vb*32<T)vw=*reinterpret_cast<const uint4*>(p.V+((base/32)+vb)*32768+head*1024+(th&63)*16);
   unsigned ka[4]={kw.x,kw.y,kw.z,kw.w},va[4]={vw.x,vw.y,vw.z,vw.w};
#pragma unroll
   for(int word=0;word<4;++word) {
#pragma unroll
    for(int byte=0;byte<4;++byte) {
     int ko=(th&31)*16+word*4+byte;
     int kn=kb*16+((ko>>6)&7)+8*((ko>>3)&1);
     int kd=2*((ko>>4)&3)+16*((ko>>2)&1)+8*((ko>>1)&1)+(ko&1);
     s16v_K[kd][kn]=base+kn<T?s16v_decode(ka[word]>>(8*byte)):0;
     int vo=(th&63)*16+word*4+byte;
     int vn=vb*32+2*((vo>>4)&3)+8*((vo>>1)&1)+(vo&1)+16*((vo>>2)&1);
     int vd=16*((vo>>9)&1)+((vo>>6)&7)+8*((vo>>3)&1);
     s16v_V[vn][vd]=base+vn<T?s16v_decode(va[word]>>(8*byte)):0;
    }
   }
   __syncthreads();
   float sums[3][2]={};
#pragma unroll 1
   for(int part=0;part<4;++part) {
    float scores[3][4]={};
#pragma unroll
    for(int word=0;word<8;++word) {
     unsigned qw[3];
#pragma unroll
     for(int i=0;i<3;++i)qw[i]=__shfl_sync(0xffffffff,q8[word],g+8*i);
#pragma unroll
     for(int z=0;z<4;++z) {
      int d=word*4+z;
      float2 b0=*reinterpret_cast<float2*>(&s16v_K[d][part*16+t*2]);
      float2 b1=*reinterpret_cast<float2*>(&s16v_K[d][part*16+t*2+8]);
      float a[3],b[4]={b0.x,b0.y,b1.x,b1.y};
#pragma unroll
      for(int i=0;i<3;++i)a[i]=s16v_decode(qw[i]>>(8*z));
#pragma unroll
      for(int i=0;i<3;++i) {
#pragma unroll
       for(int j=0;j<4;++j)scores[i][j]=fmaf(a[i],b[j],scores[i][j]);
      }
     }
    }

#pragma unroll
    for(int i=0;i<3;++i) {
#pragma unroll
     for(int j=0;j<4;++j)scores[i][j]=s16v_vitexp(s16v_half(scores[i][j]));
#pragma unroll
     for(int j=0;j<2;++j)sums[i][j]=s16v_half(sums[i][j]+s16v_half(scores[i][j]+scores[i][j+2]));
     int row=w*24+g+8*i;
#pragma unroll
     for(int j=0;j<4;++j)s16v_P[row][t*2+(j&1)+8*(j>>1)]=s16v_positive_exp8(scores[i][j]);
    }
    __syncwarp();
#pragma unroll 2
    for(int n=0;n<16;++n) {
     float a[3];
#pragma unroll
     for(int i=0;i<3;++i)a[i]=s16v_decode(s16v_P[w*24+g+8*i][n]);
     float4 b0=*reinterpret_cast<float4*>(&s16v_V[part*16+n][t*8]);
     float4 b1=*reinterpret_cast<float4*>(&s16v_V[part*16+n][t*8+4]);
     float b[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
#pragma unroll
     for(int i=0;i<3;++i) {
#pragma unroll
      for(int j=0;j<8;++j)out[i][j]=fmaf(a[i],b[j],out[i][j]);
     }
    }
    if(part&1) {
#pragma unroll
     for(int i=0;i<3;++i) {
#pragma unroll
      for(int j=0;j<8;++j)out[i][j]=s16v_half(out[i][j]);
     }
    }
    __syncwarp();
   }
#pragma unroll
   for(int i=0;i<3;++i) {
    float lo=__shfl_sync(0xffffffff,sums[i][0],l&~3),hi=__shfl_sync(0xffffffff,sums[i][1],l&~3);
#pragma unroll
    for(int t2=1;t2<4;++t2) {
     lo=s16v_half(lo+__shfl_sync(0xffffffff,sums[i][0],(l&~3)|t2));
     hi=s16v_half(hi+__shfl_sync(0xffffffff,sums[i][1],(l&~3)|t2));
    }
    sum[i]=s16v_half(sum[i]+s16v_half(lo+hi));
   }
  }
#pragma unroll
  for(int i=0;i<3;++i) {
   int tok=qbase+w*24+g+8*i;
   if(tok<T) {
    int padded=((T+63)/64)*64-T;
    if(padded)sum[i]=s16v_half(sum[i]-s16v_half(s16v_vitexp(0)*padded));
    float inv=s16v_half(1.f/fmaxf(s16v_half(sum[i]),.00006198883f));
#pragma unroll
    for(int j=0;j<8;++j)p.O[s16v_qoffset(tok,t*8+j,head)]=s16v_encode(s16v_half(out[i][j]*inv));
   }
  }
  __threadfence();__syncthreads();
  if(th==0&&p.signal) {
   for(int tile=qbase/128;tile<=min((qbase+95)/128,(T-1)/128);++tile) {
    int first=tile*128/96,last=(min(T,(tile+1)*128)-1)/96;
    int* flag=p.signal+tile*32+head;
    if(atomicSub(flag,1)==-(last-first+1)) {
     __threadfence();*reinterpret_cast<volatile int*>(flag)=0;
    }
   }
  }
 }
}
